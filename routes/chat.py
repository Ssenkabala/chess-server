"""
routes/chat.py — Continental Chat: a small, low-stakes community chat that
lives in a bubble on every page except the live game board.

Polling-based, not a WebSocket. This doesn't need sub-second latency, and
after two separate zombie-connection bugs elsewhere in this app (the
tournament socket's missing arena_send, and the game socket's missing
heartbeat), a third persistent connection for something this low-stakes
isn't worth the risk it reintroduces. The client polls this on a plain
timer; there's no server-side state to keep in sync here at all.

Rules enforced server-side (never trust the client-side copy of these):
  - Reading is public — no auth required.
  - Posting requires: a valid session, not banned, a country set on the
    profile (waived for admins), <=50 chars, no emoji, profanity filtered
    to asterisks, <=5 messages/day (unlimited for admins).
  - The table is pruned to the newest 50 messages after every send.
"""
import os, re, time, uuid, hmac, hashlib, asyncio, secrets, logging
import sqlite3
import math as _math
import subprocess as _subprocess
import threading as _threading
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

from app_core.config import ADMIN_USER_IDS, SUPABASE_URL, SUPABASE_SERVICE_KEY
from app_core.auth import verify_jwt, supabase_get_profile, check_banned
from app_core.profanity import filter_profanity


CHAT_MAX_MESSAGES = 50   # rolling window — oldest gets pruned past this
CHAT_MAX_CHARS    = 50
CHAT_DAILY_LIMIT  = 5    # per user; admins are unlimited


class ChatSendRequest(BaseModel):
    message: str


# Strips emoji/pictographs/flag characters. Covers the main emoji Unicode
# blocks rather than trying to enumerate every codepoint individually —
# deliberately broad, since "no emoji" is the actual requirement, not
# "no emoji except ones we forgot to list."
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"   # symbols & pictographs, emoticons, transport,
                              # supplemental symbols, chess symbols, extended-A
    "\U00002600-\U000027BF"   # misc symbols + dingbats
    "\U0001F1E6-\U0001F1FF"   # regional indicators (flag emoji)
    "\U00002B00-\U00002BFF"   # misc symbols and arrows
    "\U0000FE0F"              # variation selector (forces emoji presentation)
    "\U0000200D"              # zero-width joiner (glues compound emoji together)
    "]+",
    flags=re.UNICODE,
)


def _strip_emoji(text: str) -> str:
    return _EMOJI_PATTERN.sub("", text)


@router.get("/api/chat/messages")
async def get_chat_messages():
    """Public read — no auth required. Returns up to the last 50 messages,
    oldest first (natural reading order)."""
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/continental_chat",
            params={
                "select": "id,username,country,message,is_admin,created_at",
                "order": "created_at.desc",
                "limit": str(CHAT_MAX_MESSAGES),
            },
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            },
        )
        rows = r.json() if r.status_code == 200 else []
    rows.reverse()
    return {"messages": rows}


@router.post("/api/chat/send")
async def send_chat_message(req: ChatSendRequest, authorization: str = Header(None)):
    user_id = await verify_jwt(authorization)
    await check_banned(user_id)   # raises 403 if banned

    is_admin = user_id in ADMIN_USER_IDS
    profile = await supabase_get_profile(user_id)
    if not profile:
        raise HTTPException(404, "Profile not found")

    if not is_admin and not profile.get("country"):
        raise HTTPException(
            403,
            "Set your country in your profile before posting in chat — "
            "this is how everyone else's messages show a flag, and admins "
            "are the only exception.",
        )

    msg = (req.message or "").strip()
    if not msg:
        raise HTTPException(400, "Message can't be empty.")

    msg = _strip_emoji(msg).strip()
    if not msg:
        raise HTTPException(400, "Message can't be empty after removing emoji/stickers — text only.")

    if len(msg) > CHAT_MAX_CHARS:
        raise HTTPException(400, f"Message too long — {CHAT_MAX_CHARS} characters max.")

    msg = filter_profanity(msg)

    # Rate limit: live DB count, same pattern as check_prize_eligibility
    # elsewhere in this app — correct across restarts and multiple server
    # instances, unlike an in-memory counter.
    if not is_admin:
        today_start = (
            datetime.now(timezone.utc)
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .isoformat()
        )
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{SUPABASE_URL}/rest/v1/continental_chat",
                params={
                    "user_id": f"eq.{user_id}",
                    "created_at": f"gte.{today_start}",
                    "select": "id",
                },
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                },
            )
            sent_today = len(r.json()) if r.status_code == 200 else 0
        if sent_today >= CHAT_DAILY_LIMIT:
            raise HTTPException(
                429, f"Daily chat limit reached ({CHAT_DAILY_LIMIT} messages/day)."
            )

    async with httpx.AsyncClient() as client:
        await client.post(
            f"{SUPABASE_URL}/rest/v1/continental_chat",
            json={
                "user_id": user_id,
                "username": profile.get("username") or "Player",
                "country": profile.get("country"),
                "message": msg,
                "is_admin": is_admin,
            },
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json",
            },
        )

        # Prune to the newest CHAT_MAX_MESSAGES — rolling window. Only fetch
        # the ids PAST the cutoff (via offset) rather than the whole table,
        # so this stays cheap regardless of how long the table has existed.
        overflow_r = await client.get(
            f"{SUPABASE_URL}/rest/v1/continental_chat",
            params={
                "select": "id",
                "order": "created_at.desc",
                "offset": str(CHAT_MAX_MESSAGES),
            },
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            },
        )
        stale_ids = [row["id"] for row in overflow_r.json()] if overflow_r.status_code == 200 else []
        if stale_ids:
            await client.delete(
                f"{SUPABASE_URL}/rest/v1/continental_chat",
                params={"id": f"in.({','.join(stale_ids)})"},
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                },
            )

    return {"ok": True}
