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

class MoveRequest(BaseModel):
    fen: str
    think_time: float = 1.0
    difficulty: int = 3  # 1=Beginner → 8=Master
    moves: list[str] = []  # full UCI move history for repetition detection

class CoachRequest(BaseModel):
    fen: str
    played_move: Optional[str] = None   # UCI format e.g. "e2e4"
    pgn: Optional[str] = None
    lesson_type: Optional[str] = None   # "opening", "middlegame", "endgame"
    think_time: float = 1.0

class RegisterRequest(BaseModel):
    email: str
    tier: str = "free"  # set to "club"/"pro" after Stripe confirms payment

class FreeCoachRequest(BaseModel):
    fen: str
    played_move: Optional[str] = None
    pgn: Optional[str] = None
    user_id: str
    think_time: float = 0.5
    # Optional: client can pre-compute these via WASM to skip server engine pool
    best_move: Optional[str] = None
    eval_pawns: Optional[float] = None
    pv: Optional[list[str]] = None  # full continuation (UCI moves), not just best_move

class AnalyseRequest(BaseModel):
    fen: str

class FeedbackRequest(BaseModel):
    rating:  int              # 1–5
    message: Optional[str] = None
    page:    Optional[str] = None

class TournamentStartRequest(BaseModel):
    tournament_id: str
    user_id: str

class TournamentResultRequest(BaseModel):
    game_id: str       # tournament_games.id
    result: str        # 'white' | 'black' | 'draw'
    user_id: str
    time_control: Optional[str] = None  # unused server-side (fetched from DB), kept for compat
