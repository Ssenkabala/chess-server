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

from app_core.config import ANTHROPIC_API_KEY, POOL_SIZE, REASSURE_DAILY_LIMIT, REASSURE_PROMPTS, REGIONS, SUPABASE_SERVICE_KEY, SUPABASE_URL
from app_core.engine_pool import engine_pool
from app_core.models import FeedbackRequest
from app_core.state import _PRESENCE_TTL, _engine_failures, _presence, active_games, lobby_queue, tournament_connections, tournament_player_game

@router.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Store user feedback in Supabase feedback table."""
    if not (1 <= req.rating <= 5):
        raise HTTPException(400, "Rating must be 1–5")

    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{SUPABASE_URL}/rest/v1/feedback",
            headers={
                "apikey":        SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type":  "application/json",
                "Prefer":        "return=minimal"
            },
            json={
                "rating":     req.rating,
                "message":    req.message,
                "page":       req.page,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        )
    if r.status_code not in (200, 201, 204):
        raise HTTPException(500, f"DB error: {r.text}")
    return {"ok": True}

@router.post("/api/ping")
async def presence_ping(request: Request):
    """
    Called by every page every 30s to register presence.
    Uses a short random token from the client as the session key.
    Returns the current online count so the client can update its display.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    token = body.get("token", "")
    if token and len(token) <= 64:
        _presence[token] = _time.time()
    count = _presence_count()
    return {"online": count}

def _presence_count() -> int:
    """Count sessions seen within the last TTL seconds."""
    cutoff = _time.time() - _PRESENCE_TTL
    # Prune stale entries in-place
    stale = [k for k, v in _presence.items() if v < cutoff]
    for k in stale:
        del _presence[k]
    return len(_presence)

@router.get("/api/notification")
async def get_notification():
    """
    Returns the active platform notification, if any.
    Used by every page to show/hide the dismissible banner.

    Controlled entirely from Supabase — no deployment needed:
      INSERT INTO notifications (message, type, active)
      VALUES ('Tournament in 1 hour!', 'info', true);
    To take down: UPDATE notifications SET active = false WHERE active = true;

    Cached 60s so a burst of page loads doesn't hammer Supabase.
    type: 'info' (green) | 'warning' (yellow) | 'urgent' (red)
    """
    import time
    cache = get_notification._cache
    now = time.time()
    if now - cache.get("ts", 0) < 60 and "data" in cache:
        return cache["data"]
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{SUPABASE_URL}/rest/v1/notifications",
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                },
                params={
                    "select": "id,message,type",
                    "active": "eq.true",
                    "order": "created_at.desc",
                    "limit": "1",
                },
                timeout=5.0,
            )
            rows = r.json() if r.status_code == 200 else []
            result = rows[0] if rows else None
    except Exception:
        result = None  # fail silently — missing notification isn't critical
    data = {"notification": result}
    get_notification._cache = {"ts": now, "data": data}
    return data

get_notification._cache = {}

@router.get("/api/stats")
async def get_live_stats():
    """
    Public endpoint for landing page counters.
    Returns live in-memory counts + cached DB totals.
    Cached DB totals refresh every 60 seconds to avoid hammering Supabase.
    """
    import time
    now = time.time()

    # ── Live (in-memory, always fresh) ──────────────────────────
    # Active multiplayer games
    live_mp_games = len(active_games)

    # Active arena games (players mid-game inside tournaments)
    live_arena_games = sum(
        1 for games_map in tournament_player_game.values()
        for gid in games_map.values() if gid is not None
    ) // 2   # each game has 2 player entries

    live_games = live_mp_games + live_arena_games

    # Connected users: everyone pings /api/ping every 30s regardless of page.
    # _presence_count() is the single source of truth — no double-counting.
    live_connected = _presence_count()

    # ── Cached DB totals (refresh every 60s) ────────────────────
    cache = get_live_stats._cache
    if now - cache.get("ts", 0) > 60:
        try:
            async with httpx.AsyncClient() as client:
                # Total registered players (profiles with username)
                # HEAD request = no body, just headers — correct for count=exact
                pr = await client.head(
                    f"{SUPABASE_URL}/rest/v1/profiles",
                    params={"select": "count", "username": "not.is.null"},
                    headers={"apikey": SUPABASE_SERVICE_KEY,
                             "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                             "Prefer": "count=exact"}
                )
                players = int(pr.headers.get("content-range", "0/0").split("/")[-1] or 0)

                # Total games played (all tables)
                gr = await client.get(
                    f"{SUPABASE_URL}/rest/v1/games",
                    params={"select": "count"},
                    headers={"apikey": SUPABASE_SERVICE_KEY,
                             "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                             "Prefer": "count=exact", "Range": "0-0"}
                )
                bot_games = int(gr.headers.get("content-range", "0/0").split("/")[-1] or 0)

                tgr = await client.get(
                    f"{SUPABASE_URL}/rest/v1/tournament_games",
                    params={"select": "count"},
                    headers={"apikey": SUPABASE_SERVICE_KEY,
                             "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                             "Prefer": "count=exact", "Range": "0-0"}
                )
                tournament_games = int(tgr.headers.get("content-range", "0/0").split("/")[-1] or 0)

            cache["players"]       = players
            cache["total_games"]   = bot_games + tournament_games
            cache["ts"]            = now
        except Exception as e:
            print(f"[stats] DB fetch error: {e}", flush=True)
            # Keep stale cache on error

    return {
        "live_games":    live_games,
        "live_connected": live_connected,
        "total_players": cache.get("players", 0),
        "total_games":   cache.get("total_games", 0),
    }

get_live_stats._cache = {"ts": 0}   # module-level cache attached to function

@router.get("/api/leaderboard")
async def get_leaderboard(region: str = ""):
    """Return all profiles with ELO data for the continental leaderboard.
    Optionally filtered by African region (east_africa, west_africa, etc.)."""
    params: dict = {
        "select": "user_id,username,country,elo,elo_bullet,elo_blitz,elo_rapid,games_played,gender",
        "order":  "elo_blitz.desc.nullslast",
    }
    if region and region in REGIONS:
        country_list = ",".join(REGIONS[region])
        params["country"] = f"in.({country_list})"

    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{SUPABASE_URL}/rest/v1/profiles",
                params=params,
                headers={"apikey": SUPABASE_SERVICE_KEY,
                         "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
            )
        if r.status_code != 200:
            print(f"[leaderboard] Supabase error {r.status_code}: {r.text[:200]}", flush=True)
            raise HTTPException(status_code=502, detail="Could not fetch leaderboard from database")

        players = r.json()
        if not isinstance(players, list):
            print(f"[leaderboard] unexpected response type: {type(players)} — {str(players)[:200]}", flush=True)
            raise HTTPException(status_code=502, detail="Unexpected database response")

        # Filter out profiles without a username or country set
        players = [p for p in players if p.get("username") and p.get("country")]
        return players

    except HTTPException:
        raise
    except Exception as e:
        print(f"[leaderboard] exception: {e}", flush=True)
        raise HTTPException(status_code=500, detail="Leaderboard unavailable")

@router.get("/api/health")
async def health():
    """
    Health check endpoint. Railway can poll this to detect issues.
    Also exposes in-memory state counts so you can see if a restart
    wiped active games without checking logs.
    """
    pool_alive = sum(1 for w in engine_pool._workers if w.alive())
    return {
        "status":          "ok",
        "engine_pool":     {"size": POOL_SIZE, "alive": pool_alive},
        "active_games":    len(active_games),
        "lobby_queue":     len(lobby_queue),
        "tournaments":     len(tournament_connections),
        "presence":        _presence_count(),
        "engine_failures": _engine_failures,
    }

@router.post("/reassure")
async def reassure(request: FastAPIRequest):
    import random, json as _json
    from datetime import date

    # ── Simple daily rate limit via cookie ──────────────────────
    today = str(date.today())
    cookie_raw = request.cookies.get("alina_reassure", "{}")
    try:
        cookie = _json.loads(cookie_raw)
    except Exception:
        cookie = {}

    count = cookie.get("count", 0) if cookie.get("date") == today else 0

    if count >= REASSURE_DAILY_LIMIT:
        return JSONResponse(
            {"message": "Alina, you've had 20 sweet messages today — that's how loved you are. Come back tomorrow for more! 🌸"},
            status_code=200
        )

    # ── Call Anthropic ───────────────────────────────────────────
    prompt = random.choice(REASSURE_PROMPTS)
    try:
        ai_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = ai_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        text = message.content[0].text
    except Exception:
        text = "Alina, you are so incredibly loved. Don't ever forget that. 🌸"

    # ── Update cookie ────────────────────────────────────────────
    new_cookie = _json.dumps({"date": today, "count": count + 1})
    response = JSONResponse({"message": text})
    response.set_cookie("alina_reassure", new_cookie, max_age=86400, samesite="lax")
    return response
