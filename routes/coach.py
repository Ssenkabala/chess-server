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

from fastapi import APIRouter
router = APIRouter()

from app_core.config import ANTHROPIC_API_KEY, COACH_LIMITS, SUPABASE_SERVICE_KEY, SUPABASE_URL
from app_core.db import verify_key
from app_core.engine_pool import analyse_position
from app_core.models import AnalyseRequest, CoachRequest, FreeCoachRequest
from app_core.state import engine_semaphore
from routes.profile import get_user_plan

@router.post("/coach")
def coach(req: CoachRequest, user=Depends(verify_key)):
    # 1. Get engine analysis
    try:
        analysis = analyse_position(req.fen, req.think_time)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Engine error: {str(e)}")

    score_pawns = round(analysis["score_cp"] / 100, 2)
    best_move = analysis["best_move"]
    pv = analysis["pv"]

    # 2. Build prompt
    board = chess.Board(req.fen)
    turn = "White" if board.turn == chess.WHITE else "Black"

    pv_san_list = analysis.get("pv_san", [])
    continuation = ' '.join(pv_san_list[:5]) if pv_san_list else ' '.join(pv[:5])
    mate_in = analysis.get("mate_in")
    eval_display = (f"Mate in {abs(mate_in)}" if mate_in else
                    f"{'+' if score_pawns >= 0 else ''}{score_pawns} pawns (White's perspective)")

    prompt = f"""You are Senkabala, an expert chess coach powered by a 2050 ELO engine.
Analyze this position and give coaching advice to a club-level player.

Position (FEN): {req.fen}
Side to move: {turn}
Engine evaluation: {eval_display}
Engine best move: {best_move}
Engine continuation (5 moves): {continuation}

These are the EXACT engine-calculated moves. Base your explanation on this line only.
Do not invent moves or variations not listed above.
"""

    if req.played_move and req.played_move != best_move:
        prompt += f"""
The player just played: {req.played_move}
This is not the engine's top choice. Briefly explain why {best_move} is better.
"""
    elif req.played_move and req.played_move == best_move:
        prompt += f"\nThe player found the best move: {req.played_move}. Confirm why this is strong.\n"

    if req.pgn:
        prompt += f"\nFull game PGN:\n{req.pgn}\nIdentify the key turning point and biggest mistake.\n"

    if req.lesson_type:
        prompt += f"\nFocus your explanation on {req.lesson_type} principles.\n"

    prompt += """
Respond in this exact format:
ASSESSMENT: (1 sentence on who stands better and why)
BEST MOVE: (explain the engine's best move in plain English)
CONTINUATION: (walk through the next 4-5 moves from the engine continuation, one short phrase per move)
PLAN: (2-3 sentences on the strategic plan going forward)
TIP: (one practical chess principle this position illustrates)
"""

    # 3. Call Claude
    try:
        ai_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = ai_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        explanation = message.content[0].text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Coach unavailable: {str(e)}")

    return {
        "best_move": best_move,
        "eval_pawns": score_pawns,
        "pv": pv,
        "coaching": explanation,
        "tier": user["tier"]
    }

@router.post("/coach-free")
async def coach_free(req: FreeCoachRequest):
    if not SUPABASE_SERVICE_KEY:
        raise HTTPException(503, "Service unavailable")

    today = datetime.utcnow().date().isoformat()

    # Fetch profile
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"user_id": f"eq.{req.user_id}", "select": "username,elo,coach_uses_today,coach_reset_date"},
            headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
        )
        rows = r.json()

    if not rows:
        raise HTTPException(404, "Profile not found")

    profile = rows[0]
    uses_today = profile.get("coach_uses_today") or 0
    reset_date = profile.get("coach_reset_date") or ""

    # Reset counter if new day
    if reset_date != today:
        uses_today = 0

    # Get plan-based limit
    plan = await get_user_plan(req.user_id)
    daily_limit = COACH_LIMITS.get(plan, COACH_LIMITS["free"])

    if uses_today >= daily_limit:
        upgrade_msg = "Upgrade to Club ($5/mo) for 200 analyses/day." if plan == "free" else "Daily limit reached."
        raise HTTPException(429, upgrade_msg)

    # Increment usage
    async with httpx.AsyncClient() as client:
        await client.patch(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"user_id": f"eq.{req.user_id}"},
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            json={"coach_uses_today": uses_today + 1, "coach_reset_date": today}
        )

    # Run analysis — skip if client already sent pre-computed WASM result
    if req.best_move and req.eval_pawns is not None:
        # Client ran WASM locally — use those results directly, no engine pool needed.
        # Use the full PV if the client sent one (it always should, now that
        # the analysis tab carries the real continuation forward instead of
        # just the opening move) — convert to SAN the same way the engine-pool
        # path does, validating each move's legality and stopping cleanly at
        # the first illegal one rather than trusting the input blindly.
        pv_uci = req.pv if req.pv else [req.best_move]
        pv_san = []
        try:
            pv_board = chess.Board(req.fen)
            for uci in pv_uci:
                move = chess.Move.from_uci(uci)
                if move in pv_board.legal_moves:
                    pv_san.append(pv_board.san(move))
                    pv_board.push(move)
                else:
                    break
        except Exception:
            pass
        analysis = {
            "best_move":  req.best_move,
            "score_cp":   int(req.eval_pawns * 100),
            "pv":         pv_uci,
            "pv_san":     pv_san,
            "mate_in":    None,
        }
    else:
        try:
            async with engine_semaphore:
                loop = asyncio.get_event_loop()
                analysis = await loop.run_in_executor(None, analyse_position, req.fen, req.think_time)
        except Exception as e:
            import traceback
            print(f"coach-free engine error: {traceback.format_exc()}", flush=True)
            raise HTTPException(500, f"Engine error: {e}")

    score_pawns = round(analysis["score_cp"] / 100, 2)

    if not analysis.get("best_move"):
        raise HTTPException(500, "Engine could not analyse this position. Please try again.")

    board = chess.Board(req.fen)
    turn = "White" if board.turn == chess.WHITE else "Black"

    # Build human-readable piece list so Claude doesn't misread the FEN
    piece_names = {
        chess.PAWN: "Pawn", chess.KNIGHT: "Knight", chess.BISHOP: "Bishop",
        chess.ROOK: "Rook", chess.QUEEN: "Queen", chess.KING: "King"
    }
    white_pieces, black_pieces = [], []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            sq_name = chess.square_name(sq)
            name = f"{piece_names[piece.piece_type]} on {sq_name}"
            if piece.color == chess.WHITE:
                white_pieces.append(name)
            else:
                black_pieces.append(name)

    # Describe the best move in human terms
    best_uci = analysis['best_move']
    best_move_desc = ""
    if best_uci and len(best_uci) >= 4:
        from_sq = chess.parse_square(best_uci[:2])
        to_sq = chess.parse_square(best_uci[2:4])
        moving_piece = board.piece_at(from_sq)
        captured_piece = board.piece_at(to_sq)
        from_name = chess.square_name(from_sq)
        to_name = chess.square_name(to_sq)
        piece_str = piece_names.get(moving_piece.piece_type, "Piece") if moving_piece else "Piece"
        color_str = "White" if (moving_piece and moving_piece.color == chess.WHITE) else "Black"
        if captured_piece:
            cap_str = piece_names.get(captured_piece.piece_type, "piece")
            best_move_desc = f"{color_str}'s {piece_str} on {from_name} captures the {cap_str} on {to_name}"
        else:
            best_move_desc = f"{color_str}'s {piece_str} moves from {from_name} to {to_name}"
        # Check if it gives check
        test_board = board.copy()
        test_board.push(chess.Move(from_sq, to_sq))
        if test_board.is_checkmate():
            best_move_desc += " — CHECKMATE"
        elif test_board.is_check():
            best_move_desc += " (giving check)"

    # Mate score display
    if abs(score_pawns) >= 99:
        eval_display = "Forced checkmate" if score_pawns < 0 else "Forced checkmate for White"
    else:
        eval_display = f"{'+' if score_pawns >= 0 else ''}{score_pawns} pawns (White's perspective)"

    prompt = f"""You are Senkabala, an expert chess coach powered by a strong chess engine.
Give coaching advice based on this EXACT position:

Side to move: {turn}
Engine evaluation: {eval_display}

White pieces: {', '.join(white_pieces) if white_pieces else 'none'}
Black pieces: {', '.join(black_pieces) if black_pieces else 'none'}

Engine best move: {best_move_desc}
Engine continuation: {' '.join(analysis.get('pv_san', analysis['pv'])[:5])}

These are the EXACT engine-calculated moves. Base your explanation on this specific line.
Do not invent moves or variations not in this list. Do not guess — use only the continuation above.
"""
    if req.played_move:
        played_desc = ""
        try:
            from_sq2 = chess.parse_square(req.played_move[:2])
            to_sq2 = chess.parse_square(req.played_move[2:4])
            played_piece = board.piece_at(from_sq2)
            p_str = piece_names.get(played_piece.piece_type, "Piece") if played_piece else "Piece"
            played_desc = f"{p_str} from {req.played_move[:2]} to {req.played_move[2:4]}"
        except Exception:
            played_desc = req.played_move

        if req.played_move != best_uci:
            prompt += f"\nThe player just played: {played_desc}\nBriefly explain why the engine move is better.\n"
        else:
            prompt += f"\nThe player found the best move: {played_desc}. Confirm why this is strong.\n"

    if req.pgn:
        prompt += f"\nGame moves so far: {req.pgn}\n"

    prompt += """
Respond in this exact format:
ASSESSMENT: (1 sentence on who stands better and why)
BEST MOVE: (explain the engine best move in plain English)
CONTINUATION: (walk through the next 4-5 moves from the engine continuation provided, explaining the idea behind each move in one short phrase)
PLAN: (2-3 sentences on the strategic plan going forward)
TIP: (one practical chess principle this position illustrates)
"""

    try:
        ai_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = ai_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        explanation = message.content[0].text
    except Exception as e:
        import traceback
        print(f"coach-free anthropic error: {traceback.format_exc()}", flush=True)
        raise HTTPException(500, f"Coach unavailable: {e}")

    return {
        "best_move": analysis["best_move"],
        "eval_pawns": score_pawns,
        "pv_san":     analysis.get("pv_san", []),
        "mate_in":    analysis.get("mate_in"),
        "pv": analysis["pv"],
        "coaching": explanation,
        "uses_today": uses_today + 1,
        "uses_remaining": max(0, daily_limit - (uses_today + 1)),
    }

@router.post("/analyse-position")
async def analyse_pos(req: AnalyseRequest):
    try:
        async with engine_semaphore:
            loop = asyncio.get_event_loop()
            analysis = await loop.run_in_executor(None, analyse_position, req.fen, 2.0)
        return {
            "best_move": analysis["best_move"],
            "eval_pawns": round(analysis["score_cp"] / 100, 2),
            "score_cp":   analysis["score_cp"],
            "pv":         analysis["pv"],
            "pv_san":     analysis.get("pv_san", []),
            "mate_in":    analysis.get("mate_in"),
        }
    except Exception as e:
        import traceback
        print(f"analyse-position error: {traceback.format_exc()}", flush=True)
        raise HTTPException(500, f"Engine error: {e}")

@router.get("/debug-coach")
async def debug_coach():
    """Test engine analysis in isolation."""
    import traceback
    try:
        loop = asyncio.get_event_loop()
        analysis = await loop.run_in_executor(
            None, analyse_position,
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", 0.5
        )
        return {"status": "ok", "analysis": analysis}
    except Exception as e:
        return {"status": "error", "error": str(e), "trace": traceback.format_exc()}

def debug_files():
    return {
        "cwd": os.getcwd(),
        "files": os.listdir(".")
    }
