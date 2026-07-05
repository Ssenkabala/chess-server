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

from app_core.auth import verify_jwt
from app_core.config import ADMIN_USER_IDS, FREE_COACH_LIMIT, SUPABASE_SERVICE_KEY, SUPABASE_URL, _ADMIN_USER_IDS, _RESERVED_NAMES
from app_core.medals import grant_pioneer_medal
from app_core.models import RegisterRequest
from app_core.rating import elo_col_for_tc

@router.post("/register")
def register(req: RegisterRequest):
    api_key = secrets.token_urlsafe(32)
    expires = (datetime.utcnow() + timedelta(days=30)).isoformat()
    conn = sqlite3.connect("users.db")
    conn.execute(
        "INSERT INTO users (api_key, email, tier, expires_at, last_reset) VALUES (?, ?, ?, ?, ?)",
        (api_key, req.email, req.tier, expires, datetime.utcnow().date().isoformat())
    )
    conn.commit()
    conn.close()
    return {"api_key": api_key, "tier": req.tier, "expires_at": expires}

async def get_user_plan(user_id: str) -> str:
    """Return plan for user: free / club / pro"""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{SUPABASE_URL}/rest/v1/subscriptions",
                params={"user_id": f"eq.{user_id}", "select": "plan,status"},
                headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
            )
            rows = r.json()
        if rows and rows[0].get("status") == "active":
            return rows[0].get("plan", "free")
    except Exception:
        pass
    return "free"

@router.get("/api/profile/{user_id}")
async def get_profile_stats(user_id: str, x_user_id: str = Header(...)):
    if x_user_id != user_id:
        raise HTTPException(403, "Forbidden")
    if not SUPABASE_SERVICE_KEY:
        raise HTTPException(503, "Service unavailable")

    today = datetime.utcnow().date().isoformat()

    async with httpx.AsyncClient() as client:
        # Get profile
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"user_id": f"eq.{user_id}", "select": "*"},
            headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
        )
        profiles = r.json()
        if not profiles:
            raise HTTPException(404, "Profile not found")
        profile = profiles[0]

        # Get games
        r2 = await client.get(
            f"{SUPABASE_URL}/rest/v1/games",
            params={"user_id": f"eq.{user_id}", "select": "*", "order": "created_at.desc", "limit": "50"},
            headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
        )
        games = r2.json()

        # Award pioneer badge lazily on profile load (idempotent)
        asyncio.create_task(grant_pioneer_medal(user_id))

    # Compute stats
    wins = sum(1 for g in games if g.get("result") == g.get("player_color"))
    losses = sum(1 for g in games if g.get("result") not in (g.get("player_color"), "draw") and g.get("result"))
    draws = sum(1 for g in games if g.get("result") == "draw")
    total = len(games)

    # ELO history per time-control bucket from games (chronological, last 30 per TC)
    tc_histories: dict = {"bullet": [], "blitz": [], "rapid": [], "classical": []}
    for g in reversed(games):
        if not g.get("player_elo_after") or not g.get("created_at"):
            continue
        tc  = g.get("time_control")
        col = elo_col_for_tc(tc)   # elo_bullet | elo_blitz | elo_rapid | elo
        bucket = col.replace("elo_", "") if col != "elo" else "classical"
        if bucket in tc_histories:
            tc_histories[bucket].append({
                "date": g["created_at"][:10],
                "elo":  g["player_elo_after"],
            })

    # Trim to last 30 per TC
    for k in tc_histories:
        tc_histories[k] = tc_histories[k][-30:]

    # Legacy flat elo_history (blitz, for backwards compat)
    elo_history = tc_histories["blitz"] or tc_histories["rapid"] or tc_histories["bullet"] or tc_histories["classical"]

    # Coach usage
    uses_today = profile.get("coach_uses_today") or 0
    if profile.get("coach_reset_date") != today:
        uses_today = 0

    return {
        "username":   profile.get("username"),
        "elo":        profile.get("elo", 1500),
        "elo_bullet": profile.get("elo_bullet", 1500),
        "elo_blitz":  profile.get("elo_blitz",  1500),
        "elo_rapid":  profile.get("elo_rapid",  1500),
        "elo_classical": profile.get("elo", 1500),   # elo col = classical
        "created_at": profile.get("created_at"),
        "country":    profile.get("country"),
        "gender":     profile.get("gender"),
        "games_played": profile.get("games_played", 0),
        "wins":       wins,
        "losses":     losses,
        "draws":      draws,
        "total":      total,
        "win_rate":   round(wins / total * 100) if total else 0,
        "recent_games":  games[:10],
        "elo_history":   elo_history,        # legacy
        "tc_histories":  tc_histories,       # new: per-TC histories for multi-line chart
        "coach_uses_today":      uses_today,
        "coach_uses_remaining":  max(0, FREE_COACH_LIMIT - uses_today),
        "coach_limit":           FREE_COACH_LIMIT,
    }

@router.post("/api/set-username")
async def set_username(request: Request, authorization: str = Header(None)):
    """
    Create a new profile row with the chosen username.
    Validates format, reserved names, and uniqueness server-side
    so the check cannot be bypassed by calling Supabase directly.
    Admins (ADMIN_USER_IDS env var) may use reserved names.
    """
    user_id = await verify_jwt(authorization)  # raises 401 if invalid

    body = await request.json()
    username = (body.get("username") or "").strip()

    # Format check
    if not re.fullmatch(r'[a-zA-Z0-9_]{3,20}', username):
        raise HTTPException(400, "3–20 characters, letters/numbers/underscores only.")

    # Reserved name check (admins bypass)
    is_admin = user_id in _ADMIN_USER_IDS
    if not is_admin and _is_reserved(username):
        raise HTTPException(400, "That username is reserved. Please choose a different name.")

    # Normalise for uniqueness — store as-typed but check case-insensitively
    # so 'Isa', 'isa', 'ISA' can't coexist
    username_lower = username.lower()

    async with httpx.AsyncClient() as client:
        # Check uniqueness — case-insensitive so 'Isa' blocks 'isa' and 'ISA'
        check = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"username": f"ilike.{username_lower}", "select": "user_id"},
            headers={"apikey": SUPABASE_SERVICE_KEY,
                     "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
        )
        if check.json():
            raise HTTPException(400, "That username is taken. Try another.")

        # Check profile doesn't already exist for this user
        existing = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"user_id": f"eq.{user_id}", "select": "username"},
            headers={"apikey": SUPABASE_SERVICE_KEY,
                     "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
        )
        if existing.json() and existing.json()[0].get("username"):
            raise HTTPException(400, "Username already set.")

        # Insert profile
        r = await client.post(
            f"{SUPABASE_URL}/rest/v1/profiles",
            headers={
                "apikey":        SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type":  "application/json",
                "Prefer":        "return=minimal",
            },
            json={
                "user_id":    user_id,
                "username":   username,
                "elo":        1500,
                "elo_bullet": 1500,
                "elo_blitz":  1500,
                "elo_rapid":  1500,
            }
        )
        if r.status_code not in (200, 201):
            raise HTTPException(400, r.text or "Failed to create profile.")

    # Award pioneer badge if eligible (fire-and-forget)
    async with httpx.AsyncClient() as client:
        asyncio.create_task(grant_pioneer_medal(user_id))

    print(f"[profile] new user {user_id} → {username}", flush=True)
    return {"ok": True}

@router.post("/api/update-gender")
async def update_gender(request: Request, authorization: str = Header(None)):
    """Set the authenticated user's gender — locked after first save."""
    user_id = await verify_jwt(authorization)
    body    = await request.json()
    gender  = (body.get("gender") or "").strip()
    allowed = {"male", "female", "prefer_not_to_say"}

    if gender not in allowed:
        print(f"[gender] invalid value {gender!r} from {user_id}", flush=True)
        raise HTTPException(status_code=400, detail=f"Invalid gender value. Must be one of: male, female, prefer_not_to_say")

    async with httpx.AsyncClient() as client:
        # Check if profile exists and if gender already set
        check = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"user_id": f"eq.{user_id}", "select": "gender"},
            headers={"apikey": SUPABASE_SERVICE_KEY,
                     "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
        )
        rows = check.json()

        # If profile exists and gender already set — locked
        if rows and rows[0].get("gender"):
            raise HTTPException(status_code=400, detail="Gender is locked and cannot be changed.")

        if rows:
            # Profile exists, gender not set — patch it
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/profiles",
                params={"user_id": f"eq.{user_id}"},
                headers={"apikey": SUPABASE_SERVICE_KEY,
                         "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                         "Content-Type": "application/json",
                         "Prefer": "return=minimal"},
                json={"gender": gender}
            )
        else:
            # No profile row yet — shouldn't happen but handle gracefully
            raise HTTPException(status_code=400, detail="Profile not found. Please set a username first.")

    print(f"[gender] set {user_id} → {gender}", flush=True)
    return {"ok": True}

@router.post("/api/update-country")
async def update_country(request: Request, authorization: str = Header(None)):
    """Set user country once — cannot be changed after initial registration."""
    user_id = await verify_jwt(authorization)
    body = await request.json()
    country = body.get("country", None)

    if not country:
        raise HTTPException(400, "Country is required.")

    # Fetch current country — if already set, reject
    async with httpx.AsyncClient() as client:
        check = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"user_id": f"eq.{user_id}", "select": "country"},
            headers={"apikey": SUPABASE_SERVICE_KEY,
                     "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
        )
        rows = check.json()
        existing_country = rows[0].get("country") if rows else None

    if existing_country:
        raise HTTPException(403, "Country cannot be changed after registration.")

    async with httpx.AsyncClient() as client:
        r = await client.patch(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"user_id": f"eq.{user_id}"},
            headers={
                "apikey":        SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type":  "application/json",
                "Prefer":        "return=minimal"
            },
            json={"country": country}
        )
    if r.status_code not in (200, 204):
        raise HTTPException(500, f"Update failed: {r.text}")
    return {"ok": True, "country": country}

@router.post("/api/admin/ban")
async def ban_user(request: Request, authorization: str = Header(None)):
    """
    Ban or unban a player. Admin only.
    Body: { "target_user_id": "...", "reason": "...", "unban": false }
    """
    caller_id = await verify_jwt(authorization)
    if caller_id not in ADMIN_USER_IDS:
        raise HTTPException(403, "Admin access required")

    body = await request.json()
    target_id = body.get("target_user_id")
    reason    = body.get("reason", "Fair play violation")
    unban     = body.get("unban", False)

    if not target_id:
        raise HTTPException(400, "Missing target_user_id")
    if target_id in ADMIN_USER_IDS:
        raise HTTPException(400, "Cannot ban an admin account")

    async with httpx.AsyncClient() as client:
        patch = {"banned": not unban}
        if not unban:
            patch["ban_reason"] = reason
        else:
            patch["ban_reason"] = None

        r = await client.patch(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"user_id": f"eq.{target_id}"},
            headers={
                "apikey":        SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type":  "application/json",
                "Prefer":        "return=minimal",
            },
            json=patch
        )

    action = "unbanned" if unban else "banned"
    print(f"[admin] {action} {target_id} — {reason}", flush=True)
    return {"ok": True, "action": action, "target": target_id}

@router.get("/api/medals/{user_id}")
async def get_medals(user_id: str):
    """Return a player's medal collection."""
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"user_id": f"eq.{user_id}", "select": "medals,username"},
            headers={"apikey": SUPABASE_SERVICE_KEY,
                     "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
        )
    data = r.json()
    if not data:
        raise HTTPException(404, "Profile not found")
    return {
        "username": data[0].get("username"),
        "medals":   data[0].get("medals") or []
    }

@router.post("/api/admin/backfill-pioneer")
async def backfill_pioneer(request: Request):
    """
    One-time admin endpoint: award pioneer medal to all existing users if
    total user count <= 100. Protect with a secret key in the request body.
    Call once from curl:
      curl -X POST https://africhess.org/api/admin/backfill-pioneer \
           -H "Content-Type: application/json" \
           -d '{"secret": "YOUR_ADMIN_SECRET"}'
    """
    import json as _json
    body = await request.json()
    secret = os.getenv("ADMIN_SECRET", "")
    if not secret or body.get("secret") != secret:
        raise HTTPException(403, "Forbidden")

    async with httpx.AsyncClient() as client:
        # Count total users
        count_r = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"select": "count"},
            headers={"apikey": SUPABASE_SERVICE_KEY,
                     "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                     "Prefer": "count=exact", "Range": "0-0"}
        )
        total = int(count_r.headers.get("content-range", "0/0").split("/")[-1] or 9999)
        if total > 100:
            return {"ok": False, "reason": f"Too many users ({total}) — pioneer badge is for first 100 only"}

        # Fetch all profiles
        all_r = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"select": "user_id,medals"},
            headers={"apikey": SUPABASE_SERVICE_KEY,
                     "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
        )
        profiles = all_r.json()
        now = __import__("datetime").datetime.utcnow().isoformat() + "Z"
        awarded = 0
        skipped = 0

        for p in profiles:
            uid = p.get("user_id")
            if not uid:
                continue
            medals = p.get("medals") or []
            if any(m.get("id") == "pioneer" for m in medals):
                skipped += 1
                continue
            medals.append({
                "id": "pioneer", "label": "1st 100 Founder", "img": "pioneer",
                "reason": f"Among the first {total} players to join AfriChess",
                "tag": "founder", "awarded_at": now,
            })
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/profiles",
                params={"user_id": f"eq.{uid}"},
                headers={"apikey": SUPABASE_SERVICE_KEY,
                         "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                         "Content-Type": "application/json", "Prefer": "return=minimal"},
                json={"medals": medals}
            )
            awarded += 1
            print(f"[backfill-pioneer] awarded to {uid}", flush=True)

    return {"ok": True, "awarded": awarded, "skipped_already_had": skipped, "total_users": total}

def _is_reserved(username: str) -> bool:
    """
    Returns True if the username is reserved for non-admin users.
    Strips digits and underscores before checking so 'AfriChess1', 'africhess_'
    etc. are all caught.
    """
    cleaned = re.sub(r'[_0-9]', '', username.lower())
    return any(cleaned == r or cleaned.startswith(r) for r in _RESERVED_NAMES)
