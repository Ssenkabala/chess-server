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

from app_core.config import LS_CLUB_VARIANT, LS_PRO_VARIANT, LS_SIGNING_SECRET, SUPABASE_SERVICE_KEY, SUPABASE_URL

@router.post("/api/lemon-webhook")
async def lemon_webhook(request: Request):
    """Receives subscription events from Lemon Squeezy and updates Supabase."""
    body = await request.body()

    # Verify signature
    sig = request.headers.get("x-signature", "")
    if LS_SIGNING_SECRET:
        expected = hmac.new(
            LS_SIGNING_SECRET.encode(),
            body,
            hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected, sig):
            raise HTTPException(400, "Invalid signature")

    import json
    payload = json.loads(body)
    event     = payload.get("meta", {}).get("event_name", "")
    attrs     = payload.get("data", {}).get("attributes", {})
    variant_id = int(attrs.get("variant_id", 0))
    status     = attrs.get("status", "")
    ls_sub_id  = str(payload.get("data", {}).get("id", ""))
    ls_cust_id = str(attrs.get("customer_id", ""))
    renews_at  = attrs.get("renews_at")
    user_email = attrs.get("user_email", "")

    # Map variant → plan
    if variant_id == LS_CLUB_VARIANT:
        plan = "club"
    elif variant_id == LS_PRO_VARIANT:
        plan = "pro"
    else:
        return {"ok": True, "note": "unknown variant, ignored"}

    # Map LS status → our status
    sub_status = "active" if status in ("active", "on_trial") else "cancelled"

    # Find user_id from email via Supabase auth
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{SUPABASE_URL}/auth/v1/admin/users",
            params={"filter": f"email=={user_email}"},
            headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
        )
        data = r.json()
        users = data.get("users", [])

    if not users:
        # Store by email for now — will link when user next logs in
        return {"ok": True, "note": "user not found, skipped"}

    user_id = users[0]["id"]

    # Upsert into subscriptions table
    sub_row = {
        "user_id":         user_id,
        "plan":            plan if sub_status == "active" else "free",
        "ls_subscription_id": ls_sub_id,
        "ls_customer_id":  ls_cust_id,
        "status":          sub_status,
        "renews_at":       renews_at,
        "updated_at":      datetime.utcnow().isoformat()
    }

    async with httpx.AsyncClient() as client:
        await client.post(
            f"{SUPABASE_URL}/rest/v1/subscriptions",
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates,return=minimal"
            },
            json=sub_row
        )

    return {"ok": True, "plan": plan, "status": sub_status}
