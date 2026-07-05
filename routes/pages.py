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

@router.get("/")
def root():
    return FileResponse("landing.html")

@router.get("/play")
def play():
    return FileResponse("index.html")        # vs engine

@router.get("/multiplayer")
def multiplayer():
    return FileResponse("play_multiplayer.html")   # 1v1 live

@router.get("/landing")
def landing():
    return FileResponse("landing.html")

@router.get("/watch")
async def watch_page():
    """Serve the spectator/live games page."""
    return FileResponse("watch.html")

@router.get("/profile")
def profile():
    return FileResponse("profile.html")

@router.get("/history")
def history():
    return FileResponse("history.html")

@router.get("/tournaments")
def tournaments():
    return FileResponse("tournament.html")

@router.get("/leaderboard")
def leaderboard_page():
    return FileResponse("leaderboard.html")

@router.get("/alina-oct12")
def alina_page():
    return FileResponse("alina.html")

@router.get("/health")
def health_root():
    return {"status": "ok"}
