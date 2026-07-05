"""
Auto-extracted from the monolithic server.py during the modularization split.
Every function/constant below is byte-identical to the original — only
imports and (for route files) the @app. -> @router. decorator were changed.
"""

from app_core.config import TIER_LIMITS


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

def init_db():
    conn = sqlite3.connect("users.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            api_key     TEXT PRIMARY KEY,
            email       TEXT,
            tier        TEXT DEFAULT 'free',
            analyses_today  INTEGER DEFAULT 0,
            last_reset  TEXT,
            expires_at  TEXT
        )
    """)
    conn.commit()
    conn.close()

def verify_key(x_api_key: str = Header(...)):
    conn = sqlite3.connect("users.db")
    row = conn.execute(
        "SELECT * FROM users WHERE api_key = ?", (x_api_key,)
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid API key.")

    api_key, email, tier, analyses_today, last_reset, expires_at = row

    # Check subscription expiry
    if expires_at and datetime.fromisoformat(expires_at) < datetime.utcnow():
        raise HTTPException(status_code=402, detail="Subscription expired. Please renew at senkabalabot.com")

    # Reset daily counter if it's a new day
    today = datetime.utcnow().date().isoformat()
    if last_reset != today:
        conn = sqlite3.connect("users.db")
        conn.execute(
            "UPDATE users SET analyses_today = 0, last_reset = ? WHERE api_key = ?",
            (today, api_key)
        )
        conn.commit()
        conn.close()
        analyses_today = 0

    # Check daily limit
    limit = TIER_LIMITS.get(tier, 10)
    if analyses_today >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"Daily limit of {limit} analyses reached. Upgrade at senkabalabot.com"
        )

    # Increment counter
    conn = sqlite3.connect("users.db")
    conn.execute(
        "UPDATE users SET analyses_today = analyses_today + 1 WHERE api_key = ?",
        (api_key,)
    )
    conn.commit()
    conn.close()

    return {"email": email, "tier": tier}
