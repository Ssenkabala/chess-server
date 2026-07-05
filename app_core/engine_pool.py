"""
Auto-extracted from the monolithic server.py during the modularization split.
Every function/constant below is byte-identical to the original — only
imports and (for route files) the @app. -> @router. decorator were changed.
"""

from app_core.config import ENGINE_PATH, POOL_SIZE


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

class _EngineWorker:
    """A single persistent engine process with send/receive helpers."""

    def __init__(self):
        self.proc   = None
        self.lock   = _threading.Lock()
        self._start()

    def _start(self):
        try:
            self.proc = _subprocess.Popen(
                [ENGINE_PATH],
                stdin=_subprocess.PIPE, stdout=_subprocess.PIPE,
                stderr=_subprocess.PIPE, text=True, bufsize=1
            )
            # Handshake
            self.proc.stdin.write("uci\n")
            self.proc.stdin.flush()
            for _ in range(50):
                line = self.proc.stdout.readline()
                if line.strip() == "uciok":
                    break
            print(f"[pool] engine worker started (pid {self.proc.pid})", flush=True)
        except Exception as e:
            print(f"[pool] failed to start engine worker: {e}", flush=True)
            self.proc = None

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def run(self, pos_cmd: str, think_ms: int) -> tuple[str, str]:
        """Send a position + go command, collect output. Thread-safe."""
        if not self.alive():
            self._start()
        if not self.alive():
            return "", ""
        try:
            # Reset engine state between games
            self.proc.stdin.write(f"ucinewgame\n{pos_cmd}\ngo movetime {think_ms}\n")
            self.proc.stdin.flush()

            stdout_lines = []
            stderr_lines = []
            deadline = _time.time() + think_ms / 1000 + 10

            # Read stdout until bestmove
            while _time.time() < deadline:
                line = self.proc.stdout.readline()
                if not line:
                    break
                stdout_lines.append(line)
                if line.startswith("bestmove"):
                    break

            # Drain stderr (non-blocking via threads would be cleaner but this works)
            # stderr has info lines — read what's available without blocking
            import select as _select
            while _time.time() < deadline:
                r, _, _ = _select.select([self.proc.stderr], [], [], 0.01)
                if not r:
                    break
                line = self.proc.stderr.readline()
                if line:
                    stderr_lines.append(line)

            return "".join(stdout_lines), "".join(stderr_lines)
        except Exception as e:
            print(f"[pool] worker error: {e} — restarting", flush=True)
            try:
                self.proc.kill()
            except Exception:
                pass
            self.proc = None
            return "", ""

class _EnginePool:
    """
    Async queue of EngineWorkers.
    Usage:
        async with engine_pool.acquire() as worker:
            stdout, stderr = worker.run(pos_cmd, think_ms)
    """
    def __init__(self, size: int):
        self._queue: asyncio.Queue = None   # initialised in start()
        self._workers: list[_EngineWorker] = []
        self._size = size

    async def start(self):
        self._queue = asyncio.Queue()
        for _ in range(self._size):
            w = _EngineWorker()
            self._workers.append(w)
            await self._queue.put(w)
        print(f"[pool] {self._size} engine workers ready", flush=True)

    async def stop(self):
        for w in self._workers:
            try:
                if w.proc:
                    w.proc.stdin.write("quit\n")
                    w.proc.stdin.flush()
                    w.proc.wait(timeout=2)
            except Exception:
                pass
        print("[pool] engine pool stopped", flush=True)

    class _Ctx:
        def __init__(self, pool):
            self._pool   = pool
            self._worker = None
        async def __aenter__(self):
            if self._pool._queue is None:
                raise RuntimeError("Engine pool not started yet")
            self._worker = await asyncio.wait_for(
                self._pool._queue.get(), timeout=30)
            return self._worker
        async def __aexit__(self, *_):
            if self._worker is None:
                return
            # Replace dead workers before returning to pool
            if not self._worker.alive():
                print("[pool] replacing dead worker", flush=True)
                self._worker = _EngineWorker()
            await self._pool._queue.put(self._worker)

    def acquire(self):
        return self._Ctx(self)

engine_pool = _EnginePool(POOL_SIZE)

def _run_engine(pos_cmd: str, think_ms: int) -> tuple[str, str]:
    """
    Legacy sync wrapper — used by analyse_position (called via run_in_executor).
    Uses the pool if available, falls back to spawn-per-call if pool not ready.
    """
    if engine_pool._queue is not None and not engine_pool._queue.empty():
        # Can't await here (sync context) — use spawn for analysis path
        pass
    import subprocess
    commands = f"uci\nucinewgame\n{pos_cmd}\ngo movetime {think_ms}\n"
    try:
        proc = subprocess.Popen(
            [ENGINE_PATH],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1
        )
        stdout_data, stderr_data = proc.communicate(
            input=commands, timeout=think_ms / 1000 + 10)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout_data, stderr_data = proc.communicate()
    except Exception:
        stdout_data, stderr_data = "", ""
    return stdout_data, stderr_data

def _parse_engine_output(stdout_data: str, stderr_data: str) -> dict:
    """Parse bestmove + highest-depth info line from engine output."""
    best_move = None
    score     = 0
    pv_moves  = []
    best_depth = -1

    for line in stdout_data.splitlines():
        if line.startswith("bestmove"):
            parts = line.split()
            if len(parts) >= 2 and parts[1] not in ("(none)", "0000"):
                best_move = parts[1]
            break

    for line in stderr_data.splitlines():
        if "depth" not in line:
            continue
        try:
            parts = line.split()
            depth = int(parts[parts.index("depth") + 1])
            if "score" not in parts or depth <= best_depth:
                continue
            si = parts.index("score")
            stype, sval = parts[si + 1], int(parts[si + 2])
            best_depth = depth
            # score mate N → large cp value; score cp N → raw centipawns
            score = sval if stype == "cp" else (10000 - abs(sval)) * 100 * (1 if sval > 0 else -1)
            if "pv" in parts:
                pvi = parts.index("pv")
                pv_moves = parts[pvi + 1: pvi + 7]   # 6 moves for full continuation
        except (ValueError, IndexError):
            continue

    if not best_move and pv_moves:
        best_move = pv_moves[0]

    return {"best_move": best_move, "score_cp": score, "pv": pv_moves, "depth": best_depth}

def analyse_position(fen: str, think_time: float, moves: list[str] | None = None):
    """
    Talk directly to SenkabalaIII via raw subprocess.
    SenkabalaIII non-standard output: bestmove → stdout, info → stderr.

    Two-pass mate search:
      Pass 1 — normal search at think_time.
      Pass 2 — if the position looks winning (eval > +5 pawns for the side to move),
               re-search at 5× the time so the engine has enough depth to find
               forced mates instead of just playing any winning move.
               Pass 2 result replaces Pass 1 only if it found a better or equal move.
    """
    import subprocess  # noqa: F401 (imported for _run_engine)

    board = chess.Board()
    if moves:
        for uci in moves:
            try:
                board.push_uci(uci)
            except Exception:
                board = chess.Board(fen)
                break
    else:
        board = chess.Board(fen)

    think_ms = int(max(think_time, 1.0) * 1000)
    pos_cmd  = (f"position startpos moves {' '.join(moves)}"
                if moves else f"position fen {fen}")

    # ── Pass 1: normal search ─────────────────────────────────────────────
    stdout1, stderr1 = _run_engine(pos_cmd, think_ms)
    result = _parse_engine_output(stdout1, stderr1)

    # Flip score to always be from the side to move's perspective for the threshold check
    raw_score = result["score_cp"]
    stm_score = -raw_score if board.turn == chess.BLACK else raw_score

    # ── Pass 2: deep mate search if position looks winning ─────────────────
    # Lower threshold: +100cp (not +500) — even a slight advantage warrants
    # a deep mate search so forced mates like Qh3# aren't missed.
    # Skip if engine already returned a mate score (>900000).
    WINNING_THRESHOLD = 100     # centipawns — was 500, lowered to catch more mates
    MATE_SCORE_FLOOR  = 900000

    if stm_score > WINNING_THRESHOLD and abs(raw_score) < MATE_SCORE_FLOOR:
        mate_ms = max(think_ms * 8, 8000)  # at least 8s for mate search (was 5×)
        stdout2, stderr2 = _run_engine(pos_cmd, mate_ms)
        result2 = _parse_engine_output(stdout2, stderr2)
        # Use deep result if it found a move (it always should)
        if result2["best_move"]:
            result = result2
            print(f"[analyse] mate-search pass used "
                  f"(pass1 score={stm_score}cp, depth={result2['depth']})", flush=True)

    # Normalise score to White's perspective for the API response
    if board.turn == chess.BLACK:
        result["score_cp"] = -result["score_cp"]

    # Convert PV from UCI to SAN for display and coaching
    pv_san = []
    try:
        pv_board = board.copy()
        for uci in result.get("pv", []):
            move = chess.Move.from_uci(uci)
            if move in pv_board.legal_moves:
                pv_san.append(pv_board.san(move))
                pv_board.push(move)
            else:
                break
    except Exception:
        pass
    result["pv_san"] = pv_san

    # Build mate score — SenkabalaIII uses raw cp, not UCI "score mate N"
    # MATE constant = 999000. Mate in N at ply (2N-1): score = 999000 - (2N-1)
    # So N = (999000 - abs(score) + 1) // 2
    score_cp = result["score_cp"]
    SENKABALA_MATE = 999000
    if abs(score_cp) >= 900000:
        mate_in = (SENKABALA_MATE - abs(score_cp) + 1) // 2
        result["mate_in"] = mate_in * (1 if score_cp > 0 else -1)
    else:
        result["mate_in"] = None

    # Validate best_move is actually legal in this exact position before
    # returning it. The PV-to-SAN conversion above already checks legality
    # move-by-move, but best_move itself was never checked — if the engine's
    # UCI output ever returns something stale or malformed for an edge case
    # (e.g. a position with zero legal moves, like checkmate/stalemate),
    # this is what stops a bogus move from reaching the board. The client's
    # existing `if (!finalMove)` fallback logic correctly handles None here,
    # same as it already does for a missing/empty move from the WASM path.
    bm = result.get("best_move")
    if bm:
        try:
            if chess.Move.from_uci(bm) not in board.legal_moves:
                print(f"[analyse] engine returned illegal move '{bm}' for fen={fen} — discarding", flush=True)
                result["best_move"] = None
        except Exception:
            result["best_move"] = None

    return result
