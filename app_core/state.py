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

_presence: dict[str, float] = {}

_PRESENCE_TTL = 35   # seconds before a session is considered gone

_tournament_locks: set = set()

bot_semaphore      = asyncio.Semaphore(6)   # bot games (random moves bypass this entirely)

analysis_semaphore = asyncio.Semaphore(2)   # analysis board + coach

engine_semaphore   = analysis_semaphore     # legacy alias

_engine_failures = 0          # consecutive engine failures

_ENGINE_FAILURE_LIMIT = 3     # after this many, log loudly and reset count

lobby_queue: list = []          # waiting players: [{"ws": ws, "guest_id": id}]

active_games: dict = {}         # game_id ΓåÆ game state dict

pending_challenges: dict = {}

tournament_connections: dict = {}

tournament_player_game: dict = {}

_arena_pair_locks: dict = {}

_submit_result_locks: dict = {}

_player_score_locks: dict = {}

_active_pairing_loops: set = set()
