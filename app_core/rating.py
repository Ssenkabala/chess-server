"""
Auto-extracted from the monolithic server.py during the modularization split.
Every function/constant below is byte-identical to the original — only
imports and (for route files) the @app. -> @router. decorator were changed.
"""

from app_core.config import SUPABASE_SERVICE_KEY, SUPABASE_URL, _GLICKO_EPS, _GLICKO_SCALE, _GLICKO_TAU
from app_core.ws_utils import send


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

async def _sb_patch(client, url: str, params: dict, headers: dict, json: dict,
                    retries: int = 3, backoff: float = 0.5) -> None:
    """
    Supabase PATCH with exponential backoff retry.
    Handles transient connection pool exhaustion gracefully.
    """
    last_err = None
    for attempt in range(retries):
        try:
            r = await client.patch(url, params=params, headers=headers, json=json)
            if r.status_code < 500:
                return r   # success or client error (don't retry 4xx)
            last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = str(e)
        wait = backoff * (2 ** attempt)
        print(f"[supabase] patch retry {attempt+1}/{retries} after {wait}s: {last_err}", flush=True)
        await asyncio.sleep(wait)
    print(f"[supabase] patch failed after {retries} retries: {last_err}", flush=True)

async def supabase_update_elo(user_id: str, new_elo: int, time_control: str | None = None,
                              rd: float | None = None, sigma: float | None = None):
    """Update the correct ELO column, Glicko-2 rd/sigma, and increment games_played."""
    if not SUPABASE_SERVICE_KEY:
        return
    col = elo_col_for_tc(time_control)
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"user_id": f"eq.{user_id}", "select": "games_played"},
            headers={"apikey": SUPABASE_SERVICE_KEY,
                     "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
        )
        rows = r.json()
        current_gp = rows[0].get("games_played", 0) if rows else 0
        patch: dict = {col: new_elo, "games_played": current_gp + 1}
        if rd    is not None: patch["rd"]    = rd
        if sigma is not None: patch["sigma"] = sigma
        await _sb_patch(
            client,
            url=f"{SUPABASE_URL}/rest/v1/profiles",
            params={"user_id": f"eq.{user_id}"},
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            json=patch
        )

def _g(phi: float) -> float:
    """Glicko-2 g function (phi is RD in mu-space, i.e. RD/173.7178)."""
    return 1.0 / _math.sqrt(1 + 3 * phi**2 / _math.pi**2)

def _E(mu: float, mu_j: float, phi_j: float) -> float:
    """Expected score in mu-space."""
    return 1.0 / (1 + _math.exp(-_g(phi_j) * (mu - mu_j)))

def calc_glicko2(
    my_elo: int, my_rd: float, my_sigma: float,
    opp_elo: int, opp_rd: float,
    my_color: str, winner_color: str,
) -> tuple[int, float, float]:
    """
    One-game Glicko-2 update (Glickman 2012, Appendix example verified).
    Returns (new_elo, new_rd, new_sigma).

    New players  (rd=350): large swings ±100-160, rd shrinks quickly
    Established  (rd=45):  small precise adjustments ±5-20
    """
    s = 0.5 if winner_color == "draw" else (1.0 if winner_color == my_color else 0.0)

    # Convert to mu-space
    mu    = (my_elo  - 1500) / _GLICKO_SCALE
    phi   = my_rd    / _GLICKO_SCALE
    mu_j  = (opp_elo - 1500) / _GLICKO_SCALE
    phi_j = opp_rd   / _GLICKO_SCALE
    sig   = my_sigma

    g_j   = _g(phi_j)
    E_val = _E(mu, mu_j, phi_j)
    v     = 1.0 / (g_j**2 * E_val * (1 - E_val))
    delta = v * g_j * (s - E_val)

    # Illinois algorithm — update sigma
    a = _math.log(sig**2)

    def f(x: float) -> float:
        ex  = _math.exp(x)
        num = ex * (delta**2 - phi**2 - v - ex)
        den = 2.0 * (phi**2 + v + ex)**2
        return num / den - (x - a) / (_GLICKO_TAU**2)

    A = a
    if delta**2 > phi**2 + v:
        B = _math.log(delta**2 - phi**2 - v)
    else:
        k = 1
        while f(a - k * _GLICKO_TAU) < 0:
            k += 1
        B = a - k * _GLICKO_TAU

    fA, fB = f(A), f(B)
    for _ in range(200):
        C  = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fB * fC < 0: A, fA = B, fB
        else:           fA /= 2
        B, fB = C, fC
        if abs(B - A) < _GLICKO_EPS:
            break

    new_sigma = _math.exp(A / 2)
    phi_star  = _math.sqrt(phi**2 + new_sigma**2)
    new_phi   = _math.sqrt(1.0 / (1.0 / phi_star**2 + 1.0 / v))
    new_mu    = mu + new_phi**2 * g_j * (s - E_val)

    # Convert back to Elo scale and clamp
    new_elo = max(100.0, 1500 + _GLICKO_SCALE * new_mu)
    new_rd  = max(45.0, min(350.0, _GLICKO_SCALE * new_phi))

    return round(new_elo), round(new_rd, 2), round(new_sigma, 6)

def calc_elo(my_elo: int, opp_elo: int, my_color: str, winner_color: str,
             time_control: str | None = None,
             my_rd: float = 200.0, opp_rd: float = 200.0,
             my_sigma: float = 0.06) -> int:
    """Convenience wrapper — returns new ELO only. Internally uses Glicko-2."""
    new_elo, _, _ = calc_glicko2(my_elo, my_rd, my_sigma, opp_elo, opp_rd, my_color, winner_color)
    return new_elo

def k_factor(time_control: str | None, elo: int) -> int:
    """Legacy reference only — Glicko-2 is used for all calculations."""
    equiv = 5.0
    if time_control:
        try:
            parts = time_control.split('+')
            equiv = float(parts[0]) + (float(parts[1]) * 40 / 60 if len(parts) > 1 else 0)
        except (ValueError, IndexError):
            pass
    if equiv < 3:   return 40
    if equiv < 10:  return 32
    if equiv <= 15: return 24
    return 16

def elo_col_for_tc(time_control: str | None) -> str:
    """
    Map a time control string to the correct ELO column in profiles.
      Bullet  (equiv < 3 min)   → elo_bullet
      Blitz   (3–9:59)          → elo_blitz
      Rapid   (10–15)           → elo_rapid
      Classical / unknown       → elo   (the original catch-all column)
    """
    if not time_control:
        return "elo_blitz"   # default: unspecified lobby games treated as blitz
    try:
        parts = time_control.split('+')
        base  = float(parts[0])
        inc   = float(parts[1]) if len(parts) > 1 else 0.0
        equiv = base + (40 * inc / 60)
    except (ValueError, IndexError):
        return "elo_blitz"

    if equiv < 3:
        return "elo_bullet"
    if equiv < 10:
        return "elo_blitz"
    if equiv <= 15:
        return "elo_rapid"
    return "elo"   # classical — uses the legacy column

async def update_elos(game: dict, result: str):
    """
    Calculate and persist ELO changes for both players using Glicko-2
    and the correct per-time-control column.
    Only runs when BOTH players are registered (non-guest) accounts.
    Guest IDs start with "guest_" — playing a guest never affects ELO.
    """
    wp = game.get("white_profile")
    bp = game.get("black_profile")
    if not wp or not bp:
        return

    def _is_registered(uid: str | None) -> bool:
        """True only for real Supabase UUIDs (36 chars with hyphens)."""
        if not uid:
            return False
        if uid.startswith("guest_"):
            return False
        # Supabase UUIDs: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        return len(uid) == 36 and uid.count("-") == 4

    if not _is_registered(wp.get("user_id")) or not _is_registered(bp.get("user_id")):
        print(f"[elo] skipping — guest player in game {game.get('id', '?')}", flush=True)
        return

    tc  = game.get("time_control")
    col = elo_col_for_tc(tc)

    async with httpx.AsyncClient() as client:
        wr = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"user_id": f"eq.{wp['user_id']}", "select": f"{col},rd,sigma"},
            headers={"apikey": SUPABASE_SERVICE_KEY,
                     "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
        )
        br = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"user_id": f"eq.{bp['user_id']}", "select": f"{col},rd,sigma"},
            headers={"apikey": SUPABASE_SERVICE_KEY,
                     "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
        )
    w_row = (wr.json() or [{}])[0]
    b_row = (br.json() or [{}])[0]

    w_elo_old = w_row.get(col) or 1500
    b_elo_old = b_row.get(col) or 1500
    w_rd      = float(w_row.get("rd")    or 350.0)
    b_rd      = float(b_row.get("rd")    or 350.0)
    w_sigma   = float(w_row.get("sigma") or 0.06)
    b_sigma   = float(b_row.get("sigma") or 0.06)

    w_elo_new, w_rd_new, w_sigma_new = calc_glicko2(
        w_elo_old, w_rd, w_sigma, b_elo_old, b_rd, "white", result)
    b_elo_new, b_rd_new, b_sigma_new = calc_glicko2(
        b_elo_old, b_rd, b_sigma, w_elo_old, w_rd, "black", result)

    await supabase_update_elo(wp["user_id"], w_elo_new, tc, rd=w_rd_new, sigma=w_sigma_new)
    await supabase_update_elo(bp["user_id"], b_elo_new, tc, rd=b_rd_new, sigma=b_sigma_new)

    w_ws = game.get("white_game_ws") or game.get("white_ws")
    b_ws = game.get("black_game_ws") or game.get("black_ws")
    cat  = col.replace("elo_", "").capitalize() if col != "elo" else "Classical"

    if w_ws:
        await send(w_ws, {"type": "elo_update", "old_elo": w_elo_old,
                          "new_elo": w_elo_new, "rd": w_rd_new,
                          "category": cat, "column": col})
    if b_ws:
        await send(b_ws, {"type": "elo_update", "old_elo": b_elo_old,
                          "new_elo": b_elo_new, "rd": b_rd_new,
                          "category": cat, "column": col})
