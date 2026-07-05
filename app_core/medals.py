"""
Auto-extracted from the monolithic server.py during the modularization split.
Every function/constant below is byte-identical to the original — only
imports and (for route files) the @app. -> @router. decorator were changed.
"""

from app_core.config import MEDAL_TIERS, SUPABASE_SERVICE_KEY, SUPABASE_URL


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

async def award_medals(user_id: str, finish_pos: int, tournament_id: str,
                       tournament_name: str, scope: str, client) -> list[dict]:
    """
    Award podium medals + milestone medals after a tournament ends.

    Podium (every tournament):
        1st → Gold Lion
        2nd → Silver Lion
        3rd → Bronze Lion

    Milestone (lifetime, accumulate across all tournaments):
        3 regional wins   → Platinum Lion  (every multiple of 3)
        3 continental wins → Diamond Lion  (every multiple of 3)
        3 consecutive 1st-place wins → Platinum Lion  (hat-trick)
        5 consecutive 1st-place wins → Diamond Lion   (legend)

    Returns list of new medal dicts awarded this call.
    """
    if finish_pos > 3:
        return []   # only podium gets medals

    # Map finish position to medal
    pos_medal = {1: "gold", 2: "silver", 3: "bronze"}
    medal_id  = pos_medal[finish_pos]

    # Fetch current medals from profile
    profile_r = await client.get(
        f"{SUPABASE_URL}/rest/v1/profiles",
        params={"user_id": f"eq.{user_id}", "select": "medals"},
        headers={"apikey": SUPABASE_SERVICE_KEY,
                 "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
    )
    profile_data = profile_r.json()
    if not profile_data:
        return []
    current_medals: list = profile_data[0].get("medals") or []

    now = __import__("datetime").datetime.utcnow().isoformat() + "Z"
    new_medals: list[dict] = []

    def add(mid: str, reason: str, tag: str = ""):
        new_medals.append({
            "id":            mid,
            "label":         MEDAL_TIERS[mid]["label"],
            "img":           MEDAL_TIERS[mid]["img"],
            "reason":        reason,
            "tag":           tag,
            "tournament_id": tournament_id,
            "awarded_at":    now,
        })

    # ── 1. Podium medal (always) ──────────────────────────────
    pos_label = {1: "1st place", 2: "2nd place", 3: "3rd place"}
    add(medal_id, f"{pos_label[finish_pos]} — {tournament_name}")

    # ── 2. Milestone medals (1st-place wins only) ─────────────
    if finish_pos == 1:
        # Count past wins from medals already held
        past_wins_regional    = sum(1 for m in current_medals
                                    if m.get("tag","").startswith("win_regional"))
        past_wins_continental = sum(1 for m in current_medals
                                    if m.get("tag","").startswith("win_continental"))
        past_wins_any         = sum(1 for m in current_medals
                                    if m.get("tag","").startswith("win_"))

        # Tag this win in the medal so we can count it later
        win_tag = f"win_{scope}_{tournament_id}"
        add("gold", f"1st place — {tournament_name}", win_tag)   # duplicate? no — this IS the podium medal, just tagging it
        # Actually tag the podium medal we already added
        new_medals[-2]["tag"] = win_tag   # tag the podium medal (index -2, before this one)
        new_medals.pop()                  # remove the duplicate

        new_regional    = past_wins_regional    + (1 if scope == "regional"    else 0)
        new_continental = past_wins_continental + (1 if scope == "continental" else 0)
        new_any         = past_wins_any + 1

        # Regional milestone: every 3 regional wins → Platinum
        if scope == "regional" and new_regional % 3 == 0:
            add("platinum", f"{new_regional} regional tournament wins", f"milestone_regional_{new_regional}")

        # Continental milestone: every 3 continental wins → Diamond
        if scope == "continental" and new_continental % 3 == 0:
            add("diamond", f"{new_continental} continental tournament wins", f"milestone_continental_{new_continental}")

        # Consecutive wins — check last wins from medals list
        # Count how many of the most recent win_* medals are consecutive 1st-place
        win_medals = sorted(
            [m for m in current_medals if m.get("tag","").startswith("win_")],
            key=lambda m: m.get("awarded_at",""), reverse=True
        )
        consecutive = 1   # include current
        for m in win_medals:
            if m.get("id") == "gold":   # gold = 1st place
                consecutive += 1
            else:
                break

        if consecutive == 3:
            add("platinum", "Hat-Trick — 3 consecutive 1st-place wins", "hat_trick")
        elif consecutive == 5:
            add("diamond", "Legend — 5 consecutive 1st-place wins", "legend")

    if not new_medals:
        return []

    # Persist to Supabase
    updated_medals = current_medals + new_medals
    await client.patch(
        f"{SUPABASE_URL}/rest/v1/profiles",
        params={"user_id": f"eq.{user_id}"},
        headers={"apikey": SUPABASE_SERVICE_KEY,
                 "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                 "Content-Type": "application/json"},
        json={"medals": updated_medals}
    )
    print(f"[medals] {user_id} pos={finish_pos}: {[m['id'] for m in new_medals]}", flush=True)
    return new_medals

async def grant_pioneer_medal(user_id: str, client=None) -> bool:
    """
    Award '1st 100 Founder' badge to players who joined when total users <= 100.
    Always opens its own httpx client — safe to run as an asyncio.create_task()
    because it never borrows a caller's client that may already be closed.
    """
    import httpx as _httpx
    try:
        async with _httpx.AsyncClient(timeout=10) as c:
            count_r = await c.get(
                f"{SUPABASE_URL}/rest/v1/profiles",
                params={"select": "count"},
                headers={"apikey": SUPABASE_SERVICE_KEY,
                         "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                         "Prefer": "count=exact", "Range": "0-0"}
            )
            total_users = int(count_r.headers.get("content-range", "0/0").split("/")[-1] or 9999)
            if total_users > 100:
                return False

            profile_r = await c.get(
                f"{SUPABASE_URL}/rest/v1/profiles",
                params={"user_id": f"eq.{user_id}", "select": "medals"},
                headers={"apikey": SUPABASE_SERVICE_KEY,
                         "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
            )
            rows = profile_r.json()
            if not rows:
                return False
            current_medals = rows[0].get("medals") or []
            if any(m.get("id") == "pioneer" for m in current_medals):
                return False

            now = __import__("datetime").datetime.utcnow().isoformat() + "Z"
            await c.patch(
                f"{SUPABASE_URL}/rest/v1/profiles",
                params={"user_id": f"eq.{user_id}"},
                headers={"apikey": SUPABASE_SERVICE_KEY,
                         "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                         "Content-Type": "application/json", "Prefer": "return=minimal"},
                json={"medals": current_medals + [{"id": "pioneer", "label": "1st 100 Founder",
                      "img": "pioneer",
                      "reason": f"Among the first {total_users} players to join AfriChess",
                      "tag": "founder", "awarded_at": now}]}
            )
            print(f"[medals] pioneer badge awarded to {user_id} (user #{total_users})", flush=True)
            return True
    except Exception as e:
        print(f"[medals] pioneer grant failed for {user_id}: {e}", flush=True)
        return False

async def _grant_tournament_medals(tournament_id: str, client=None):
    """
    Fetch final standings and award podium medals to top 3.

    IMPORTANT: always creates its own httpx client internally, regardless of
    whether a `client` was passed in. This function is always invoked via
    asyncio.create_task() (fire-and-forget) by its callers — none of them
    await it directly. That means if it borrows a client from a caller's
    `async with httpx.AsyncClient() as client:` block, that block can exit
    and close the client before this task actually gets scheduled to run,
    since create_task() doesn't block the caller. This was confirmed live:
    every auto-ended tournament hit "Cannot send a request, as the client
    has been closed" and silently awarded zero medals. The `client` param
    is kept for backwards compatibility but is no longer used.
    """
    try:
        async with httpx.AsyncClient() as client:
            # Get tournament info (name, scope)
            t_r = await client.get(
                f"{SUPABASE_URL}/rest/v1/tournaments",
                params={"id": f"eq.{tournament_id}", "select": "name,scope"},
                headers={"apikey": SUPABASE_SERVICE_KEY,
                         "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
            )
            t_data = t_r.json()
            if not t_data:
                return
            t_name  = t_data[0].get("name", "Tournament")
            t_scope = t_data[0].get("scope", "open")

            # Get final standings ordered by score desc, then elo desc for tiebreak
            p_r = await client.get(
                f"{SUPABASE_URL}/rest/v1/tournament_players",
                params={"tournament_id": f"eq.{tournament_id}",
                        "select": "user_id,score,elo",
                        "order":  "score.desc,elo.desc"},
                headers={"apikey": SUPABASE_SERVICE_KEY,
                         "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
            )
            players = p_r.json()
            if not players:
                return

            # Award medals to top 3 (handles ties by elo tiebreak)
            for pos, player in enumerate(players[:3], start=1):
                uid = player.get("user_id")
                if uid:
                    awarded = await award_medals(uid, pos, tournament_id, t_name, t_scope, client)
                    if awarded:
                        # Also patch rank into tournament_players for record-keeping
                        await client.patch(
                            f"{SUPABASE_URL}/rest/v1/tournament_players",
                            params={"tournament_id": f"eq.{tournament_id}",
                                    "user_id":        f"eq.{uid}"},
                            headers={"apikey": SUPABASE_SERVICE_KEY,
                                     "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                                     "Content-Type": "application/json"},
                            json={"rank": pos}
                        )
    except Exception as e:
        print(f"[medals] _grant_tournament_medals error {tournament_id}: {e}", flush=True)
