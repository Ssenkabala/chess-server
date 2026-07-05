"""
Auto-extracted from the monolithic server.py during the modularization split.
Every function/constant below is byte-identical to the original — only
imports and (for route files) the @app. -> @router. decorator were changed.
"""

from app_core.config import SUPABASE_SERVICE_KEY, SUPABASE_URL


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

async def verify_jwt(authorization: str) -> str:
    """Verify Supabase JWT and return the user_id. Raises 401 if invalid."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid authorization header")
    token = authorization.split(" ", 1)[1]
    # Verify token against Supabase auth API
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "apikey":        SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {token}"
            }
        )
    if r.status_code != 200:
        raise HTTPException(401, "Invalid or expired session token")
    return r.json().get("id", "")

async def supabase_get_profile(user_id: str) -> dict | None:
    """Fetch profile by user_id using service role key."""
    if not SUPABASE_SERVICE_KEY:
        return None
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"user_id": f"eq.{user_id}",
                    "select": "username,elo,elo_bullet,elo_blitz,elo_rapid,"
                              "banned,ban_reason,created_at,games_played"},
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            }
        )
        rows = r.json()
        return rows[0] if rows else None

async def check_banned(user_id: str):
    """Raise 403 immediately if the user is banned."""
    profile = await supabase_get_profile(user_id)
    if profile and profile.get("banned"):
        reason = profile.get("ban_reason") or "Violation of fair play rules."
        raise HTTPException(403, f"Account banned: {reason}")

async def check_prize_eligibility(user_id: str, client: httpx.AsyncClient):
    """
    For prize tournaments: enforce account age ≥ 7 days and ≥ 10 rated games.
    Raises 403 with a clear message if not eligible.
    Also checks banned status.
    """
    r = await client.get(
        f"{SUPABASE_URL}/rest/v1/profiles",
        params={"user_id": f"eq.{user_id}",
                "select": "banned,ban_reason,created_at,games_played"},
        headers={"apikey": SUPABASE_SERVICE_KEY,
                 "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
    )
    rows = r.json()
    if not rows:
        raise HTTPException(404, "Profile not found")
    p = rows[0]

    if p.get("banned"):
        reason = p.get("ban_reason") or "Violation of fair play rules."
        raise HTTPException(403, f"Account banned: {reason}")

    # Account age — created_at is an ISO string from Supabase
    if p.get("created_at"):
        try:
            created = datetime.fromisoformat(p["created_at"].replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - created).days
            if age_days < 7:
                raise HTTPException(403,
                    f"Account must be at least 7 days old to enter prize tournaments "
                    f"(yours is {age_days} day{'s' if age_days != 1 else ''} old).")
        except HTTPException:
            raise
        except Exception:
            pass  # can't parse date — don't block

    # Minimum rated games
    games_played = p.get("games_played") or 0
    if games_played < 10:
        raise HTTPException(403,
            f"You need at least 10 rated games to enter prize tournaments "
            f"(you have {games_played}). Play some ranked games first!")
