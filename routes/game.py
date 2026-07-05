"""
Auto-extracted from the monolithic server.py during the modularization split.
Every function/constant below is byte-identical to the original — only
imports and (for route files) the @app. -> @router. decorator were changed.
"""

import os, re, time, uuid, hmac, hashlib, asyncio, secrets, logging
import sqlite3
import math as _math
import subprocess as _subprocess
import threading as _threading
import time as _time
import httpx
import chess
import chess.engine
import anthropic
from datetime import datetime, timedelta, timezone
from typing import Optional
from pydantic import BaseModel
from fastapi import HTTPException, Header, Depends, Request
from fastapi import Request as FastAPIRequest
from fastapi.responses import JSONResponse, FileResponse
from fastapi import WebSocket, WebSocketDisconnect

from fastapi import APIRouter
router = APIRouter()

from app_core.config import DIFFICULTY_SETTINGS, ENGINE_PATH, SUPABASE_SERVICE_KEY, SUPABASE_URL
from app_core.engine_pool import engine_pool
from app_core.models import MoveRequest
from app_core.rating import elo_col_for_tc, update_elos
from app_core.state import _ENGINE_FAILURE_LIMIT, active_games, bot_semaphore, lobby_queue, pending_challenges
from app_core.ws_utils import broadcast, cleanup_game, deduct_clock, fairplay_log, new_game, send, validate_and_push
from routes.tournament import finalize_tournament_scoring, tournament_handle_forfeit

@router.post("/move")
async def get_move(req: MoveRequest):
    global _engine_failures
    import random

    try:
        # Validate FEN — no semaphore needed for this
        board = chess.Board(req.fen)
        if board.is_game_over():
            return {
                "move": None, "fen": req.fen,
                "is_game_over": True,
                "outcome": str(board.outcome()),
                "score_cp": 0, "eval_pawns": 0, "candidates": []
            }

        think_ms = DIFFICULTY_SETTINGS.get(req.difficulty, int(req.think_time * 1000))

        # Reconstruct game board from full move history (needed for repetition detection)
        game_board = chess.Board()
        if req.moves:
            for uci in req.moves:
                try:
                    game_board.push_uci(uci)
                except Exception:
                    game_board = chess.Board(req.fen)
                    break
        else:
            game_board = chess.Board(req.fen)

        # Initialise all result variables so both the random AND engine paths
        # always define them — avoids NameError on random move path (the 500 bug)
        score_cp   = 0
        candidates = []
        move       = None
        move_uci   = None

        # random_chance: probability of playing a random legal move instead of engine best.
        # Levels 1–3 use randomisation as the PRIMARY weakness mechanism.
        # These bypass the semaphore entirely — no engine spawn needed.
        random_chance = {
            1: 0.90,   # Beginner   — 90% random, 10% engine
            2: 0.65,   # Beginner+  — 65% random
            3: 0.25,   # Easy       — 25% random
            4: 0.0,    # Intermediate — always engine
            5: 0.0,    # Hard
            6: 0.0,    # Hard+
            7: 0.0,    # Expert
            8: 0.0,    # Master
        }
        # Initialise these before the if/else so both paths always define them
        score_cp   = 0
        candidates = []
        move       = None
        move_uci   = None

        if random.random() < random_chance.get(req.difficulty, 0):
            # Random move — instant, no engine, no semaphore
            move = random.choice(list(game_board.legal_moves))
            move_uci = move.uci()
            candidates = [{"move": move_uci, "eval_pawns": 0}]
        else:
            # Engine move — acquire bot semaphore to limit concurrent engine processes
            async with bot_semaphore:
                # Use raw subprocess so we can send the full position command.
                # python-chess engine.play() only sends `position fen <fen>` internally,
                # which means the engine's posHistory[] never gets populated — it can't
                # detect repetitions. We must send `position startpos moves <all>` ourselves.
                import subprocess
                if req.moves:
                    pos_cmd = f"position startpos moves {' '.join(req.moves)}"
                else:
                    pos_cmd = f"position fen {req.fen}"

                commands = f"uci\nucinewgame\n{pos_cmd}\ngo movetime {think_ms}\n"
                move_uci   = None
                # Give the engine generous grace: think time + 18s
                # (server engine can extend search via instability heuristic)
                hard_limit = think_ms / 1000 + 18

                # Use pool worker — no process startup cost
                move_uci = None
                try:
                    if engine_pool._queue is None:
                        raise RuntimeError("pool not ready")
                    async with engine_pool.acquire() as worker:
                        loop = asyncio.get_event_loop()
                        # Cap at hard_limit; worker.run already sends `stop` internally
                        stdout_data, _ = await asyncio.wait_for(
                            loop.run_in_executor(None, worker.run, pos_cmd, think_ms),
                            timeout=hard_limit
                        )
                    for line in stdout_data.splitlines():
                        if line.startswith("bestmove"):
                            parts = line.split()
                            if len(parts) >= 2 and parts[1] not in ("(none)", "0000"):
                                move_uci = parts[1]
                            break
                    if move_uci:
                        _engine_failures = 0
                    else:
                        _engine_failures += 1
                        print("[engine] no bestmove from pool worker", flush=True)
                except asyncio.TimeoutError:
                    _engine_failures += 1
                    print(f"[engine] pool worker timeout after {hard_limit}s — "
                          f"difficulty={req.difficulty} think_ms={think_ms}", flush=True)
                except Exception as eng_err:
                    _engine_failures += 1
                    print(f"[engine] pool worker error: {eng_err}", flush=True)
                    # Spawn-per-call fallback if pool not ready
                    if not move_uci:
                        print("[engine] falling back to spawn-per-call", flush=True)
                        import subprocess as _sp
                        try:
                            commands = f"uci\nucinewgame\n{pos_cmd}\ngo movetime {think_ms}\n"
                            p = _sp.Popen([ENGINE_PATH], stdin=_sp.PIPE,
                                          stdout=_sp.PIPE, stderr=_sp.PIPE, text=True)
                            out, _ = p.communicate(input=commands, timeout=hard_limit)
                            for line in out.splitlines():
                                if line.startswith("bestmove"):
                                    parts = line.split()
                                    if len(parts) >= 2 and parts[1] not in ("(none)", "0000"):
                                        move_uci = parts[1]
                                    break
                        except Exception as spawn_err:
                            print(f"[engine] spawn fallback also failed: {spawn_err}", flush=True)

                if _engine_failures >= _ENGINE_FAILURE_LIMIT:
                    print(f"[engine] ALERT: {_engine_failures} consecutive failures", flush=True)
                    _engine_failures = 0

                if not move_uci:
                    legal = list(game_board.legal_moves)
                    if not legal:
                        return {"move": None, "fen": game_board.fen(),
                                "is_game_over": True, "score_cp": 0, "eval_pawns": 0, "candidates": []}
                    move_uci = random.choice(legal).uci()
                    print("[engine] fallback random move used", flush=True)
                move = chess.Move.from_uci(move_uci)

        # Final guard — move must be legal
        if move is None or move not in game_board.legal_moves:
            legal = list(game_board.legal_moves)
            if not legal:
                return {"move": None, "fen": game_board.fen(),
                        "is_game_over": True, "score_cp": 0, "eval_pawns": 0, "candidates": []}
            print(f"[engine] illegal move {move} — using random", flush=True)
            move = random.choice(legal)

        game_board.push(move)
        return {
            "move":         move.uci(),
            "fen":          game_board.fen(),
            "is_game_over": game_board.is_game_over(),
            "outcome":      str(game_board.outcome()) if game_board.is_game_over() else None,
            "score_cp":     score_cp,
            "eval_pawns":   round(score_cp / 100, 2),
            "candidates":   candidates
        }
    except Exception as e:
        import traceback
        print(f"[/move] 500 error: {e}\n{traceback.format_exc()}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/games/live")
async def list_live_games():
    """Return currently active games for the watch page."""
    games = []
    for gid, g in active_games.items():
        if g.get("over"):
            continue
        wp = g.get("white_profile") or {}
        bp = g.get("black_profile") or {}
        games.append({
            "id":           gid,
            "white":        wp.get("username", "Player"),
            "black":        bp.get("username", "Player"),
            "white_elo":    wp.get("elo", 1500),
            "black_elo":    bp.get("elo", 1500),
            "time_control": g.get("time_control", "?"),
            "moves":        g.get("moves_made", 0),
            "fen":          g["board"].fen(),
            "clock":        g.get("clock", {"w": 0, "b": 0}),
            "spectators":   len(g.get("spectators", [])),
        })
    # Sort by most spectators first, then by most moves (most interesting games first)
    games.sort(key=lambda x: (-x["spectators"], -x["moves"]))
    return games

@router.websocket("/ws/watch/{game_id}")
async def watch_ws(ws: WebSocket, game_id: str):
    """
    Spectator WebSocket. Connects to a live game and receives all moves
    and game events in real time. Read-only — no input accepted.
    """
    await ws.accept()

    game = active_games.get(game_id)
    if not game or game.get("over"):
        await ws.send_json({"type": "error", "detail": "Game not found or already finished."})
        await ws.close()
        return

    # Register spectator
    if "spectators" not in game:
        game["spectators"] = []
    game["spectators"].append(ws)

    # Send current game state so spectator can render the board immediately
    wp = game.get("white_profile") or {}
    bp = game.get("black_profile") or {}
    await ws.send_json({
        "type":         "game_state",
        "fen":          game["board"].fen(),
        "clock":        game["clock"],
        "turn":         "white" if game["board"].turn else "black",
        "moves":        game.get("moves_made", 0),
        "time_control": game.get("time_control", "?"),
        "white":        wp.get("username", "Player"),
        "black":        bp.get("username", "Player"),
        "white_elo":    wp.get("elo", 1500),
        "black_elo":    bp.get("elo", 1500),
    })

    print(f"[watch] spectator joined game {game_id} "
          f"({len(game['spectators'])} watching)", flush=True)

    # Keep connection alive until spectator disconnects or game ends
    try:
        while True:
            try:
                # Accept pings, ignore everything else (read-only)
                await asyncio.wait_for(ws.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                # Send a keepalive ping
                try:
                    await ws.send_json({"type": "ping"})
                except Exception:
                    break
            except Exception:
                break
            # Check if game ended
            game = active_games.get(game_id)
            if not game or game.get("over"):
                try:
                    await ws.send_json({"type": "game_ended"})
                except Exception:
                    pass
                break
    finally:
        # Remove from spectators list
        game = active_games.get(game_id)
        if game and "spectators" in game:
            try:
                game["spectators"].remove(ws)
                print(f"[watch] spectator left game {game_id} "
                      f"({len(game['spectators'])} watching)", flush=True)
            except ValueError:
                pass

@router.websocket("/ws/lobby")
async def lobby(ws: WebSocket):
    await ws.accept()
    guest_id = "guest_" + uuid.uuid4().hex[:8]
    preferred_tc = "5+0"  # default, updated when client sends time_control message

    await send(ws, {"type": "waiting", "guest_id": guest_id})

    # Read initial messages to get time_control preference before matching
    # Give client 3s to send tc preference, then proceed with matching
    try:
        msg = await asyncio.wait_for(ws.receive_json(), timeout=3.0)
        if msg.get("type") == "time_control":
            preferred_tc = msg.get("tc", "5+0")
    except (asyncio.TimeoutError, Exception):
        pass

    # Match by same time control if possible, otherwise any opponent
    matched_opponent = None
    for i, opp in enumerate(lobby_queue):
        if opp.get("tc", "5+0") == preferred_tc:
            matched_opponent = lobby_queue.pop(i)
            break
    if matched_opponent is None and lobby_queue:
        matched_opponent = lobby_queue.pop(0)

    if matched_opponent:
        game_id  = uuid.uuid4().hex[:12]
        tc_used  = preferred_tc  # use the joining player's TC (or negotiate — keep simple)

        white_ws, white_id = matched_opponent["ws"], matched_opponent["guest_id"]
        black_ws, black_id = ws, guest_id

        game = new_game(game_id, white_ws, black_ws, white_id, black_id, tc_used)
        active_games[game_id] = game

        await send(white_ws, {
            "type":         "matched",
            "game_id":      game_id,
            "color":        "white",
            "opponent":     black_id,
            "time_control": tc_used,
        })
        await send(black_ws, {
            "type":         "matched",
            "game_id":      game_id,
            "color":        "black",
            "opponent":     white_id,
            "time_control": tc_used,
        })

        asyncio.create_task(clock_loop(game_id))
    else:
        lobby_queue.append({"ws": ws, "guest_id": guest_id, "tc": preferred_tc})

    # Keep lobby socket alive until matched or disconnected
    try:
        while True:
            data = await ws.receive_json()
            if isinstance(data, dict) and data.get("type") == "ping":
                pass  # keepalive, ignore
    except WebSocketDisconnect:
        lobby_queue[:] = [p for p in lobby_queue if p["guest_id"] != guest_id]

@router.get("/lobby/status")
def lobby_status():
    return {
        "waiting":      len(lobby_queue),
        "active_games": len(active_games),
    }

@router.post("/api/challenge")
async def create_challenge(req: Request):
    """
    Create a challenge invite link. Works for guests and signed-in users.
    Body: { time_control: "5+0" }
    Returns: { code: "abc123", time_control: "5+0", creator: "username" }
    """
    body     = await req.json()
    tc       = body.get("time_control", "5+0")
    username = "Guest"
    uid      = "guest_" + uuid.uuid4().hex[:8]

    # If user sends auth header, fetch their username
    auth = req.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(
                    f"{SUPABASE_URL}/auth/v1/user",
                    headers={"apikey": SUPABASE_SERVICE_KEY,
                             "Authorization": f"Bearer {token}"}
                )
                if r.status_code == 200:
                    user_data = r.json()
                    uid = user_data.get("id", uid)
                    # Fetch username from profiles
                    r2 = await client.get(
                        f"{SUPABASE_URL}/rest/v1/profiles",
                        params={"user_id": f"eq.{uid}", "select": "username"},
                        headers={"apikey": SUPABASE_SERVICE_KEY,
                                 "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
                    )
                    rows = r2.json()
                    if rows: username = rows[0].get("username") or username
        except Exception:
            pass

    # Generate short unique code
    import secrets, time as _time
    code = secrets.token_urlsafe(4)[:6].replace("-", "x").replace("_", "y")

    # Clean up expired challenges (>15 min old)
    now = _time.time()
    expired = [k for k, v in pending_challenges.items()
               if now - v.get("created_at", now) > 900]
    for k in expired:
        pending_challenges.pop(k, None)

    pending_challenges[code] = {
        "code":         code,
        "creator_id":   uid,
        "creator_name": username,
        "time_control": tc,
        "ws":           None,
        "created_at":   now,
    }
    print(f"[challenge] {username} created {code} ({tc})", flush=True)
    return {"code": code, "time_control": tc, "creator": username}

@router.get("/api/challenge/{code}")
async def get_challenge(code: str):
    """Return challenge info for the join page."""
    ch = pending_challenges.get(code)
    if not ch:
        raise HTTPException(404, "Challenge not found or expired")
    return {
        "code":         ch["code"],
        "creator":      ch["creator_name"],
        "time_control": ch["time_control"],
    }

@router.websocket("/ws/challenge/{code}")
async def challenge_ws(ws: WebSocket, code: str):
    """
    Both the creator and the joiner connect here.
    Creator connects first and waits.
    When the joiner connects, the game starts immediately.
    """
    await ws.accept()

    ch = pending_challenges.get(code)
    if not ch:
        await ws.send_json({"type": "error", "detail": "Challenge not found or expired."})
        await ws.close()
        return

    import time as _time

    # ── Creator is connecting ─────────────────────────────────────
    if ch["ws"] is None:
        ch["ws"] = ws
        await ws.send_json({
            "type":         "waiting",
            "code":         code,
            "time_control": ch["time_control"],
            "message":      "Waiting for your friend to join…",
        })
        print(f"[challenge] creator connected: {code}", flush=True)

        # Keep alive until matched or disconnected
        try:
            while True:
                try:
                    data = await asyncio.wait_for(ws.receive_json(), timeout=5)
                    if isinstance(data, dict) and data.get("type") == "ping":
                        pass
                except asyncio.TimeoutError:
                    pass
                # Check if we've been matched (ws replaced by None to signal completion)
                if ch.get("matched"):
                    break
        except Exception:
            pass
        finally:
            # If creator disconnects before anyone joins, clean up
            if not ch.get("matched"):
                pending_challenges.pop(code, None)
        return

    # ── Joiner is connecting ──────────────────────────────────────
    creator_ws = ch["ws"]
    tc         = ch["time_control"]

    # Get joiner identity
    joiner_id   = "guest_" + uuid.uuid4().hex[:8]
    creator_id  = ch["creator_id"]
    creator_name = ch["creator_name"]

    # Read ident from joiner (optional — they may send user_id + username)
    joiner_name = joiner_id
    try:
        ident = await asyncio.wait_for(ws.receive_json(), timeout=3)
        if ident.get("user_id"):   joiner_id   = ident["user_id"]
        if ident.get("username"):  joiner_name = ident["username"]
    except Exception:
        pass

    # Assign colours randomly
    import random as _random
    if _random.random() < 0.5:
        white_ws, white_id, white_name = creator_ws, creator_id, creator_name
        black_ws, black_id,  black_name  = ws,         joiner_id,  joiner_name
    else:
        white_ws, white_id, white_name = ws,         joiner_id,  joiner_name
        black_ws, black_id,  black_name  = creator_ws, creator_id, creator_name

    game_id = uuid.uuid4().hex[:12]
    game    = new_game(game_id, white_ws, black_ws, white_id, black_id, tc)
    active_games[game_id] = game

    ch["matched"] = True
    pending_challenges.pop(code, None)

    await send(white_ws, {
        "type":         "matched",
        "game_id":      game_id,
        "color":        "white",
        "opponent":     black_name,
        "time_control": tc,
    })
    await send(black_ws, {
        "type":         "matched",
        "game_id":      game_id,
        "color":        "black",
        "opponent":     white_name,
        "time_control": tc,
    })

    asyncio.create_task(clock_loop(game_id))
    asyncio.create_task(first_move_timeout_loop(game_id))
    print(f"[challenge] {code} matched: {white_name} vs {black_name}", flush=True)

    # Keep joiner WS alive until game WS takes over
    try:
        while True:
            try:
                await asyncio.wait_for(ws.receive_json(), timeout=30)
            except asyncio.TimeoutError:
                break
    except Exception:
        pass

@router.websocket("/ws/game/{game_id}")
async def game_ws(ws: WebSocket, game_id: str):
    await ws.accept()

    game = active_games.get(game_id)
    if not game:
        await send(ws, {"type": "error", "detail": "Game not found."})
        await ws.close()
        return

    # Identify which player this is by registering their new game socket
    player_id = None
    color = None
    if game["white_ws"] is None or game.get("white_game_ws") is None:
        # Check if this is the white player connecting
        # We'll resolve by slot: first to connect gets their slot
        pass

    # Wait for client to declare their color via first message
    # First message must be {"type": "claim", "color": "white"/"black"}
    try:
        claim_data = await ws.receive_json()
    except Exception:
        await ws.close()
        return

    claimed_color = claim_data.get("color", "")

    if not game.get("_lock"):
        game["_lock"] = asyncio.Lock()

    async with game["_lock"]:
        # For tournament games, verify this user is one of the two assigned players
        connecting_user_id = claim_data.get("user_id")
        if game.get("tournament_id") and connecting_user_id:
            if (claimed_color == "white" and connecting_user_id != game.get("white_id")) or \
               (claimed_color == "black" and connecting_user_id != game.get("black_id")):
                await send(ws, {"type": "error", "detail": "You are not a player in this tournament game."})
                await ws.close()
                return

        async def _slot_is_live(existing_ws):
            """Check whether a previously-claimed socket for this color is
            actually still alive, not just a stale handle left behind by a
            connection that's already gone (e.g. the player navigated away
            and back, or had a brief network drop, before the old socket's
            own disconnect handler had a chance to run). A harmless ping is
            the only reliable way to tell — without this, a legitimate
            reconnect could be rejected in favor of a dead old connection,
            leaving the player stuck with no working socket at all while
            the server keeps trying to talk to one that's already gone."""
            if existing_ws is None:
                return False
            try:
                await existing_ws.send_json({"type": "ping"})
                return True
            except Exception:
                return False

        if claimed_color == "white":
            if game.get("white_game_ws") is None or not await _slot_is_live(game.get("white_game_ws")):
                game["white_game_ws"] = ws
                game["white_ws"] = ws
                color = "w"
            else:
                await send(ws, {"type": "error", "detail": "Slot unavailable."})
                await ws.close()
                return
        elif claimed_color == "black":
            if game.get("black_game_ws") is None or not await _slot_is_live(game.get("black_game_ws")):
                game["black_game_ws"] = ws
                game["black_ws"] = ws
                color = "b"
            else:
                await send(ws, {"type": "error", "detail": "Slot unavailable."})
                await ws.close()
                return
        else:
            await send(ws, {"type": "error", "detail": "Slot unavailable."})
            await ws.close()
            return

    # Notify both players once both are connected
    if game.get("white_game_ws") and game.get("black_game_ws"):
        if not game.get("tournament_id"):
            # Casual lobby games: deadline starts once both strangers are present
            game["last_move_ts"]        = time.time()
            game["first_move_deadline"] = time.time() + game.get("first_move_timeout", 60)
            asyncio.create_task(first_move_timeout_loop(game["id"]))
        # Tournament games already have their deadline + timeout loop running
        # from the moment they were paired — connecting here does not reset it,
        # so a late-but-within-timeout player doesn't get extra time at their
        # opponent's expense.
        #
        # Tell the client the ACTUAL remaining seconds, not just the configured
        # total timeout — for tournament games the deadline may have already
        # been ticking for a while before this player finished navigating from
        # the lobby, so "first_move_timeout" alone would show a misleadingly
        # large countdown (or none at all, since the old client only showed it
        # for white).
        deadline = game.get("first_move_deadline")
        remaining_secs = max(0, int(deadline - time.time())) if deadline else game.get("first_move_timeout", 60)
        await broadcast(game, {"type": "both_connected",
                               "first_move_timeout": game.get("first_move_timeout", 60),
                               "first_move_remaining": remaining_secs})

        # Push already-received profile to the late-connecting player
        if color == "w" and game.get("black_profile"):
            await send(ws, {
                "type":     "opponent_profile",
                "username": game["black_profile"]["username"],
                "elo":      game["black_profile"]["elo"],
            })
        elif color == "b" and game.get("white_profile"):
            await send(ws, {
                "type":     "opponent_profile",
                "username": game["white_profile"]["username"],
                "elo":      game["white_profile"]["elo"],
            })

    try:
        while True:
            data = await ws.receive_json()

            msg_type = data.get("type")

            if game["over"]:
                if msg_type not in ("rematch_offer", "rematch_accept", "rematch_decline", "draw_claim", "ping"):
                    await send(ws, {"type": "error", "detail": "Game is over."})
                    continue


            # ── Keepalive ping — REPLY with pong ──────────────────────────
            # Previously just discarded ("continue", no reply). That made the
            # client's ping purely one-directional — it never got confirmation
            # the server (and the path to it) was actually still alive. On a
            # "zombie" connection (looks OPEN to the browser but the underlying
            # pipe is dead — happens after extended idle periods on mobile
            # data, backgrounded tabs, certain NAT timeouts), neither onclose
            # nor onerror reliably fires, so without a reply the client had no
            # way to ever detect it: moves stopped arriving, moves stopped
            # sending, with no error at all. Replying lets the client's
            # watchdog (see play_multiplayer.html) detect a MISSED pong and
            # proactively reconnect instead of hanging silently.
            if msg_type == "ping":
                await send(ws, {"type": "pong"})
                continue

            # ΓöÇΓöÇ Profile (sent on connect) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
            if msg_type == "profile":
                tc  = data.get("time_control") or game.get("time_control") or "5+0"
                col = elo_col_for_tc(tc)
                # Pick the correct per-category ELO for display
                elo_map = {
                    "elo_bullet": data.get("elo_bullet"),
                    "elo_blitz":  data.get("elo_blitz"),
                    "elo_rapid":  data.get("elo_rapid"),
                    "elo":        data.get("elo"),
                }
                display_elo = elo_map.get(col) or data.get("elo") or 1500

                profile = {
                    "username":   data.get("username", "guest"),
                    "elo":        display_elo,
                    "elo_bullet": data.get("elo_bullet"),
                    "elo_blitz":  data.get("elo_blitz"),
                    "elo_rapid":  data.get("elo_rapid"),
                    "user_id":    data.get("user_id"),
                }
                # Store time_control on game so update_elos can use correct K-factor
                if tc and not game.get("time_control"):
                    game["time_control"] = tc
                if color == "w":
                    game["white_profile"] = profile
                    await send(game.get("black_game_ws") or game["black_ws"], {
                        "type":     "opponent_profile",
                        "username": profile["username"],
                        "elo":      display_elo,
                    })
                else:
                    game["black_profile"] = profile
                    await send(game.get("white_game_ws") or game["white_ws"], {
                        "type":     "opponent_profile",
                        "username": profile["username"],
                        "elo":      display_elo,
                    })
                continue

            # ─── Move ────────────────────────────────────────────────────────
            if msg_type == "move":
                # Only the player whose turn it is can move
                expected = "w" if game["board"].turn == chess.WHITE else "b"
                if color != expected:
                    await send(ws, {"type": "error", "detail": "Not your turn."})
                    continue

                # Reject move if this player's clock is already at 0
                # (clock_loop may not have fired yet — this closes the race window)
                now_check = time.time()
                last_ts   = game.get("last_move_ts") or now_check
                elapsed   = now_check - last_ts
                if game["clock"][color] - elapsed <= 0:
                    # Flag this player immediately
                    if not game["over"]:
                        game["over"] = True
                        loser  = "white" if color == "w" else "black"
                        winner = "black" if loser == "white" else "white"
                        await broadcast(game, {
                            "type":   "gameover",
                            "result": winner,
                            "reason": "timeout",
                            "clock":  game["clock"],
                        })
                        if not game.get("_elo_updated"):
                            game["_elo_updated"] = True
                            await update_elos(game, winner)
                        fairplay_log(game, winner)
                        await finalize_tournament_scoring(game, winner)
                        active_games.pop(game_id, None)
                    continue

                uci = data.get("move", "")
                move = validate_and_push(game, uci)
                if move is None:
                    await send(ws, {"type": "error", "detail": "Illegal move."})
                    continue

                # Record move time for fair play analysis
                # IMPORTANT: capture elapsed BEFORE deduct_clock() runs, since
                # deduct_clock() reads last_move_ts itself to calculate the deduction.
                # Overwriting last_move_ts here first would zero out the elapsed time
                # used by the clock math, causing the clock to snap back to full TC.
                _now = time.time()
                if game.get("last_move_ts"):
                    _ms = int((_now - game["last_move_ts"]) * 1000)
                    game["move_times_w" if color == "w" else "move_times_b"].append(_ms)

                # Deduct clock — this reads and then updates last_move_ts internally
                remaining = deduct_clock(game)
                fen = game["board"].fen()

                # Check game over conditions
                if game["board"].is_game_over():
                    game["over"] = True
                    outcome = game["board"].outcome()
                    result = (
                        "white" if outcome.winner == chess.WHITE else
                        "black" if outcome.winner == chess.BLACK else
                        "draw"
                    )
                    reason = outcome.termination.name.lower()
                    await broadcast(game, {
                        "type":   "gameover",
                        "result": result,
                        "reason": reason,
                        "fen":    fen,
                        "clock":  game["clock"],
                    })
                    if not game.get("_elo_updated"):
                        game["_elo_updated"] = True
                        await update_elos(game, result)
                    fairplay_log(game, result)
                    await finalize_tournament_scoring(game, result)
                    active_games.pop(game_id, None)
                else:
                    await broadcast(game, {
                        "type":  "move",
                        "move":  uci,
                        "fen":   fen,
                        "clock": game["clock"],
                        "turn":  "white" if game["board"].turn == chess.WHITE else "black",
                    })

            # ΓöÇΓöÇ Resign ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
            elif msg_type == "resign":
                game["over"] = True
                winner = "black" if color == "w" else "white"
                await broadcast(game, {
                    "type":   "gameover",
                    "result": winner,
                    "reason": "resignation",
                    "clock":  game["clock"],
                })
                if not game.get("_elo_updated"):
                    game["_elo_updated"] = True
                    await update_elos(game, winner)
                fairplay_log(game, winner)
                await finalize_tournament_scoring(game, winner)
                asyncio.create_task(cleanup_game(game_id, delay=10))

            # ── Draw offer ─────────────────────────────────────────────────────
            elif msg_type == "draw_offer":
                # Send over the GAME websocket, not the lobby websocket
                opponent_ws = game.get("black_game_ws") if color == "w" else game.get("white_game_ws")
                if opponent_ws:
                    await send(opponent_ws, {"type": "draw_offer"})
                else:
                    # Fallback to lobby ws if game ws not yet connected
                    opponent_ws = game["black_ws"] if color == "w" else game["white_ws"]
                    if opponent_ws:
                        await send(opponent_ws, {"type": "draw_offer"})

            elif msg_type == "draw_accept":
                game["over"] = True
                await broadcast(game, {
                    "type":   "gameover",
                    "result": "draw",
                    "reason": "agreement",
                    "clock":  game["clock"],
                })
                if not game.get("_elo_updated"):
                    game["_elo_updated"] = True
                    await update_elos(game, "draw")
                fairplay_log(game, "draw")
                await finalize_tournament_scoring(game, "draw")
                asyncio.create_task(cleanup_game(game_id, delay=10))

            elif msg_type == "draw_claim":
                reason = data.get("reason", "threefold_repetition")
                board  = game["board"]
                valid  = (
                    (reason == "threefold_repetition" and board.is_repetition(3)) or
                    board.is_fifty_moves() or
                    board.is_insufficient_material()
                )
                if valid and not game["over"]:
                    game["over"] = True
                    await broadcast(game, {
                        "type":   "gameover",
                        "result": "draw",
                        "reason": reason,
                        "clock":  game["clock"],
                    })
                    if not game.get("_elo_updated"):
                        game["_elo_updated"] = True
                        await update_elos(game, "draw")
                    await finalize_tournament_scoring(game, "draw")
                    asyncio.create_task(cleanup_game(game_id, delay=10))

            elif msg_type == "takeback_offer":
                # Takebacks not allowed in tournament games
                if game.get("tournament_id"):
                    await send(ws, {"type": "error", "detail": "Takebacks are not allowed in tournament games."})
                elif game["board"].move_stack and not game.get("takeback_offered_by"):
                    game["takeback_offered_by"] = color
                    opponent_ws = game.get("black_game_ws") if color == "w" else game.get("white_game_ws")
                    if opponent_ws:
                        await send(opponent_ws, {"type": "takeback_offer"})

            elif msg_type == "takeback_accept":
                # Only accept if the opponent offered (not yourself)
                offered_by = game.get("takeback_offered_by")
                if offered_by and offered_by != color and not game["over"]:
                    game["takeback_offered_by"] = None
                    board = game["board"]
                    if board.move_stack:
                        board.pop()
                        # Decrement moves_made to stay accurate
                        game["moves_made"] = max(0, game.get("moves_made", 1) - 1)
                    # Reset the clock timestamp so clock_loop doesn't count
                    # time spent during/before the takeback as thinking time
                    game["last_move_ts"] = time.time()
                    new_fen = board.fen()
                    await broadcast(game, {
                        "type":  "takeback",
                        "fen":   new_fen,
                        "turn":  "white" if board.turn == chess.WHITE else "black",
                        "clock": game["clock"],
                    })

            elif msg_type == "takeback_decline":
                game["takeback_offered_by"] = None
                opponent_ws = game.get("black_game_ws") if color == "w" else game.get("white_game_ws")
                if opponent_ws:
                    await send(opponent_ws, {"type": "takeback_declined"})

            elif msg_type == "rematch_offer":
                # Only notify opponent — do NOT create game yet
                game["rematch_offered_by"] = color
                opponent_ws = game.get("black_game_ws") if color == "w" else game.get("white_game_ws")
                if opponent_ws:
                    await send(opponent_ws, {"type": "rematch_offer"})

            elif msg_type == "rematch_accept":
                # Only valid if opponent offered
                if game.get("rematch_offered_by") and game["rematch_offered_by"] != color:
                    old_white_profile = game["white_profile"]
                    old_black_profile = game["black_profile"]
                    old_tc            = game.get("time_control")
                    white_game_ws     = game.get("white_game_ws")
                    black_game_ws     = game.get("black_game_ws")
                    # IDs for the new game — colors swap
                    new_white_id = game.get("black_id", "?")
                    new_black_id = game.get("white_id", "?")

                    new_game_id = uuid.uuid4().hex[:12]
                    # Colors swapped: old black becomes new white
                    ng = new_game(new_game_id, None, None,
                                  new_white_id, new_black_id, old_tc)
                    ng["white_profile"] = old_black_profile
                    ng["black_profile"] = old_white_profile
                    # white_game_ws / black_game_ws left as None — players connect fresh
                    active_games[new_game_id] = ng

                    # Notify both players — they reconnect via /ws/game/{new_game_id}
                    if black_game_ws:
                        await send(black_game_ws, {
                            "type": "rematch_start", "game_id": new_game_id, "color": "white"
                        })
                    if white_game_ws:
                        await send(white_game_ws, {
                            "type": "rematch_start", "game_id": new_game_id, "color": "black"
                        })
                    asyncio.create_task(cleanup_game(game_id, delay=10))

            elif msg_type == "rematch_decline":
                opponent_ws = game["black_ws"] if color == "w" else game["white_ws"]
                await send(opponent_ws, {"type": "rematch_declined"})

    except WebSocketDisconnect:
        if not game["over"]:
            game["over"] = True
            winner = "black" if color == "w" else "white"
            opponent_ws = game["black_ws"] if color == "w" else game["white_ws"]
            await send(opponent_ws, {
                "type":   "gameover",
                "result": winner,
                "reason": "disconnect",
                "clock":  game["clock"],
            })
            if not game.get("_elo_updated"):
                game["_elo_updated"] = True
                await update_elos(game, winner)
            await finalize_tournament_scoring(game, winner)
            active_games.pop(game_id, None)
    except Exception as e:
        # Any OTHER error in this loop (not a clean disconnect) previously
        # propagated unhandled, crashing the connection with no cleanup, no
        # opponent notification, and no log detail. Confirmed as a real gap:
        # this is exactly the mechanism that would produce "opponent thinking
        # forever, move never arrived" — the connection responsible for
        # broadcasting the move died silently with nothing left to tell
        # either player what happened. Treat it the same as a disconnect for
        # game-state purposes, but log the actual cause so it's diagnosable
        # next time instead of showing up as a bare framework traceback.
        print(f"[game] {game_id} unexpected error for color={color}: "
              f"{type(e).__name__}: {e}", flush=True)
        try:
            if not game["over"]:
                game["over"] = True
                winner = "black" if color == "w" else "white"
                opponent_ws = game["black_ws"] if color == "w" else game["white_ws"]
                await send(opponent_ws, {
                    "type":   "gameover",
                    "result": winner,
                    "reason": "disconnect",
                    "clock":  game["clock"],
                })
                if not game.get("_elo_updated"):
                    game["_elo_updated"] = True
                    await update_elos(game, winner)
                await finalize_tournament_scoring(game, winner)
                active_games.pop(game_id, None)
        except Exception as cleanup_err:
            print(f"[game] {game_id} cleanup after error also failed: {cleanup_err}", flush=True)

async def first_move_timeout_loop(game_id: str):
    """
    Enforce first-move timeout for BOTH sides:
    - White must play within first_move_timeout seconds of game start
    - Black must respond within first_move_timeout seconds of white's first move
    After move 2 this loop exits and clock_loop handles normal time management.
    """
    await asyncio.sleep(1)
    black_deadline = None   # set after white's first move

    while True:
        await asyncio.sleep(1)
        game = active_games.get(game_id)
        if not game or game["over"]:
            return

        deadline = game.get("first_move_deadline")
        moves_made = game.get("moves_made", 0)

        # Phase 1: waiting for both_connected to set the deadline
        if deadline is None:
            continue

        # Phase 2: white hasn't moved yet
        if moves_made == 0:
            if time.time() >= deadline:
                game["over"] = True
                await broadcast(game, {
                    "type":   "gameover",
                    "result": "black",
                    "reason": "first_move_timeout",
                    "detail": "White did not move in time.",
                    "clock":  game["clock"],
                })
                if not game.get("_elo_updated"):
                    game["_elo_updated"] = True
                    await update_elos(game, "black")
                await finalize_tournament_scoring(game, "black")
                await tournament_handle_forfeit(game, loser_id=game.get("white_id"), winner_id=game.get("black_id"), result="black")
                await asyncio.sleep(1)
                active_games.pop(game_id, None)
                print(f"[game] {game_id} — white forfeited (no first move)", flush=True)
                return
            continue

        # Phase 3: white just made move 1 — start black's deadline
        if moves_made == 1 and black_deadline is None:
            black_deadline = time.time() + game["first_move_timeout"]
            # Tell black their response clock has started — without this,
            # black has no visible countdown at all for their first-response
            # deadline (this was previously silent, unlike white's deadline
            # which at least had a client-side, if inaccurate, countdown).
            black_ws = game.get("black_game_ws")
            if black_ws:
                await send(black_ws, {
                    "type": "first_move_started",
                    "first_move_remaining": game["first_move_timeout"],
                })
            continue

        # Phase 4: black hasn't responded
        if moves_made == 1 and black_deadline is not None:
            if time.time() >= black_deadline:
                game["over"] = True
                await broadcast(game, {
                    "type":   "gameover",
                    "result": "white",
                    "reason": "first_move_timeout",
                    "detail": "Black did not respond in time.",
                    "clock":  game["clock"],
                })
                if not game.get("_elo_updated"):
                    game["_elo_updated"] = True
                    await update_elos(game, "white")
                await finalize_tournament_scoring(game, "white")
                await tournament_handle_forfeit(game, loser_id=game.get("black_id"), winner_id=game.get("white_id"), result="white")
                await asyncio.sleep(1)
                active_games.pop(game_id, None)
                print(f"[game] {game_id} — black forfeited (no first response)", flush=True)
                return
            continue

        # Both sides have made their first moves — hand off to clock_loop
        return

async def clock_loop(game_id: str):
    """Background task — checks for flag fall every second.
    Ticks from the moment last_move_ts is set (game creation for tournament
    games, both-connected for casual lobby games) — NOT from the first move.
    This means the connection-grace + first-move-timeout window genuinely
    costs the player real clock time, same as Lichess, rather than being a
    free period that doesn't touch their time control at all.
    """
    while True:
        await asyncio.sleep(1)
        game = active_games.get(game_id)
        if not game or game["over"]:
            return

        # Wait until both game WebSockets are connected
        if not game.get("white_game_ws") or not game.get("black_game_ws"):
            continue

        last_ts = game.get("last_move_ts")
        if last_ts is None:
            continue

        now     = time.time()
        elapsed = now - last_ts
        # Before the first move, it's always white's clock running (board.turn
        # starts as WHITE) — same "whose turn is it" logic works whether or
        # not any moves have been made yet.
        turn    = "w" if game["board"].turn == chess.WHITE else "b"
        remaining = game["clock"][turn] - elapsed

        if remaining <= 0:
            game["over"] = True
            loser  = "white" if turn == "w" else "black"
            winner = "black" if loser == "white" else "white"
            await broadcast(game, {
                "type":   "gameover",
                "result": winner,
                "reason": "timeout",
                "clock":  game["clock"],
            })
            if not game.get("_elo_updated"):
                game["_elo_updated"] = True
                await update_elos(game, winner)
            fairplay_log(game, winner)
            await finalize_tournament_scoring(game, winner)
            await asyncio.sleep(1)
            active_games.pop(game_id, None)
            return

        # Send clock tick to both players
        await broadcast(game, {
            "type":  "clock",
            "white": round(game["clock"]["w"] - (elapsed if turn == "w" else 0), 1),
            "black": round(game["clock"]["b"] - (elapsed if turn == "b" else 0), 1),
        })
