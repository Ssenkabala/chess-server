"""
Auto-extracted from the monolithic server.py during the modularization split.
Every function/constant below is byte-identical to the original — only
imports and (for route files) the @app. -> @router. decorator were changed.
"""

from app_core.config import SUPABASE_SERVICE_KEY, SUPABASE_URL
from app_core.state import active_games


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

async def send(ws: WebSocket, msg: dict):
    """Safe send ΓÇö ignores errors if socket already closed."""
    try:
        await ws.send_json(msg)
    except Exception:
        pass

async def arena_send(ws: WebSocket, msg: dict):
    """Safe send for arena/tournament WebSockets. This is what every arena
    handler (join, pause/resume, pairing, game_ready) uses to talk to a
    player's tournament socket. It went missing during refactoring - the 17
    call sites remained but the definition was gone, so EVERY tournament
    WebSocket connection crashed with NameError: name 'arena_send' is not
    defined the instant it tried to send the first "connected" message. That
    killed the socket immediately, the client auto-reconnected, and it crashed
    again - the reconnect storm that made join/pause/resume/pairing all appear
    broken. Restored as a thin safe-send wrapper (same behaviour as send)."""
    try:
        await ws.send_json(msg)
    except Exception:
        pass

async def broadcast(game: dict, msg: dict):
    # Prefer the dedicated game WebSocket; fall back to lobby ws; skip if None
    w_ws = game.get("white_game_ws") or game.get("white_ws")
    b_ws = game.get("black_game_ws") or game.get("black_ws")
    if w_ws:
        await send(w_ws, msg)
    if b_ws:
        await send(b_ws, msg)
    # Also broadcast to spectators (read-only watchers)
    dead = []
    for spec_ws in game.get("spectators", []):
        try:
            await spec_ws.send_json(msg)
        except Exception:
            dead.append(spec_ws)
    for d in dead:
        try: game["spectators"].remove(d)
        except ValueError: pass

async def cleanup_game(game_id: str, delay: int = 10):
    await asyncio.sleep(delay)
    active_games.pop(game_id, None)

def parse_clock_seconds(time_control: str | None) -> tuple[int, int]:
    """
    Convert '5+0', '3+2', '10+5', '0+1' etc. to (base_seconds, increment_seconds).
    base_seconds: starting clock per player
    increment_seconds: added to clock after each move (Fischer increment)
    Special case: '0+1' → base=0s but minimum 0s start (pure increment game)
    """
    if not time_control:
        return 300, 0  # default 5+0
    try:
        parts = time_control.split('+')
        base_secs = int(float(parts[0]) * 60)
        inc_secs  = int(float(parts[1])) if len(parts) > 1 else 0
        return max(0, base_secs), max(0, inc_secs)
    except (ValueError, IndexError):
        return 300, 0

def new_game(game_id: str, white_ws: WebSocket, black_ws: WebSocket,
             white_id: str, black_id: str, time_control: str | None = None) -> dict:
    base_secs, inc_secs = parse_clock_seconds(time_control)
    # The clock starts the moment the game begins — same as a real chess
    # clock starting the instant both sides sit down, whether or not anyone
    # has touched a piece yet. If you're stuck in traffic, your clock is
    # still running.
    #
    # first_move_timeout is an early-forfeit rule on top of that: if a player
    # hasn't made their first move by the time 20% of their base time has
    # elapsed, they forfeit outright — this catches genuine no-shows quickly
    # instead of making their opponent wait out an entire clock (e.g. a full
    # 5 minutes in a 5+0 game) for someone who was never going to show up.
    # e.g. 1+0 → 12s, 2+1 → 24s, 3+0 → 36s, 5+0 → 60s, 10+0 → 120s
    first_move_timeout = max(10, int(base_secs * 0.20))
    return {
        "id":                   game_id,
        "board":                chess.Board(),
        "white_ws":             white_ws,
        "black_ws":             black_ws,
        "white_game_ws":        None,
        "black_game_ws":        None,
        "white_id":             white_id,
        "black_id":             black_id,
        "clock":                {"w": base_secs, "b": base_secs},
        "increment":            inc_secs,
        "last_move_ts":         None,
        "first_move_timeout":   first_move_timeout,
        "first_move_deadline":  None,
        "moves_made":           0,
        "over":                 False,
        "time_control":         time_control,
        "white_profile":        None,
        "black_profile":        None,
        "spectators":           [],   # list of WebSocket connections watching this game
        "takeback_offered_by":  None, # color ("w"/"b") that offered takeback, or None
        "move_times_w":         [],   # ms per white move for fair play analysis
        "move_times_b":         [],   # ms per black move for fair play analysis
    }

def deduct_clock(game: dict) -> float:
    """
    Deduct elapsed time from the side that just moved, then add increment (Fischer).
    The clock starts ticking from game creation (last_move_ts is set there now,
    not just on first move) — so the connection-grace + first-move-timeout
    window genuinely costs the player real clock time, the same as Lichess.
    Returns the remaining clock for the side that just moved (after increment).
    """
    now = time.time()
    game["moves_made"] = game.get("moves_made", 0) + 1
    inc = game.get("increment", 0)

    last_ts = game.get("last_move_ts") or now
    elapsed = now - last_ts
    # The side that just moved is OPPOSITE of board.turn (move already pushed)
    just_moved = "b" if game["board"].turn == chess.WHITE else "w"
    # Deduct time then add increment (Fischer: you always get the increment)
    game["clock"][just_moved] = max(0, game["clock"][just_moved] - elapsed + inc)
    game["last_move_ts"] = now
    return game["clock"][just_moved]

def validate_and_push(game: dict, uci_move: str) -> chess.Move | None:
    """Validate UCI move against current board state. Returns move or None."""
    try:
        move = chess.Move.from_uci(uci_move)
        if move in game["board"].legal_moves:
            game["board"].push(move)
            return move
    except Exception:
        pass
    return None

def analyse_move_times(move_times_ms: list) -> dict:
    """Score 0.0-1.0. >= 0.6 log for review. >= 0.85 check before prize payout."""
    if len(move_times_ms) < 10:
        return {"score": 0.0, "flags": []}
    import statistics
    flags = []; score = 0.0; times = move_times_ms
    mean = statistics.mean(times)
    if mean <= 0: return {"score": 0.0, "flags": []}
    stdev = statistics.stdev(times) if len(times) > 1 else 0
    cv = stdev / mean
    if cv < 0.25 and mean > 3000: flags.append("low_variance"); score += 0.35
    fast = sum(1 for t in times if t < 3000) / len(times)
    if fast < 0.05 and len(times) > 12: flags.append("no_fast_moves"); score += 0.25
    slow = sum(1 for t in times if t > 45000) / len(times)
    if slow < 0.02 and len(times) > 15: flags.append("no_slow_moves"); score += 0.15
    band = sum(1 for t in times if 5000 <= t <= 25000) / len(times)
    if band > 0.80: flags.append("consistent_band"); score += 0.25
    return {"score": round(min(score, 1.0), 2), "flags": flags}

def fairplay_log(game: dict, result: str):
    """Log suspicious move time patterns to Railway logs for manual review,
    and persist scores to Supabase so they're queryable before prize payouts.
    Called at every game end — only logs/persists when something is flagged."""
    gid = game.get("id", "?")
    scores = {}
    for color, pkey, tkey in [("white","white_profile","move_times_w"),("black","black_profile","move_times_b")]:
        times = game.get(tkey, [])
        if not times: continue
        a = analyse_move_times(times)
        scores[color] = a
        if a["score"] >= 0.6:
            p = game.get(pkey) or {}
            level = "SUSPICIOUS" if a["score"] >= 0.85 else "REVIEW"
            print(f"[fairplay] {level} game={gid} {color}={p.get('username','?')} "
                  f"uid={p.get('user_id','?')} result={result} "
                  f"score={a['score']} flags={a['flags']} "
                  f"moves={len(times)} mean={int(sum(times)/len(times))}ms", flush=True)

    # Persist scores to Supabase so they're queryable before prize payouts.
    # Only write if at least one player has a non-zero score — clean games
    # don't need a record. Uses a background task so it never blocks game flow.
    if any(v["score"] > 0 for v in scores.values()):
        asyncio.create_task(_fairplay_persist(gid, scores))

async def _fairplay_persist(game_id: str, scores: dict):
    """Write fairplay scores to the games table for pre-payout review.
    Silent failure — fairplay data is useful but not critical to game flow."""
    try:
        patch = {}
        if "white" in scores:
            patch["fairplay_score_w"] = scores["white"]["score"]
            patch["fairplay_flags_w"] = ",".join(scores["white"]["flags"]) or None
        if "black" in scores:
            patch["fairplay_score_b"] = scores["black"]["score"]
            patch["fairplay_flags_b"] = ",".join(scores["black"]["flags"]) or None
        if not patch:
            return
        async with httpx.AsyncClient() as client:
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/games",
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                    "Content-Type": "application/json",
                },
                params={"id": f"eq.{game_id}"},
                json=patch,
                timeout=5.0,
            )
    except Exception as e:
        print(f"[fairplay] persist error game={game_id}: {e}", flush=True)



# ═══════════════════════════════════════════════════════════════
#  ARENA ENGINE
# ═══════════════════════════════════════════════════════════════
    try:
        await ws.send_json(msg)
    except Exception:
        pass
