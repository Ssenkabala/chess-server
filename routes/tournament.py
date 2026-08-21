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
from fastapi import WebSocket, WebSocketDisconnect

from fastapi import APIRouter
router = APIRouter()

from app_core.auth import check_prize_eligibility, verify_jwt
from app_core.config import ADMIN_USER_IDS, COUNTRY_NAMES_SERVER, REGIONS, REGION_LABELS, SUPABASE_SERVICE_KEY, SUPABASE_URL
from app_core.config import (RECURRING_TOURNAMENT_NAME, RECURRING_TOURNAMENT_DESCRIPTION,
                              RECURRING_TOURNAMENT_TIME_CONTROL, RECURRING_TOURNAMENT_DURATION_MINUTES,
                              RECURRING_TOURNAMENT_PRIZE_POOL, RECURRING_TOURNAMENT_HOUR_EAT,
                              RECURRING_WARMUP_NAME, RECURRING_WARMUP_DESCRIPTION,
                              RECURRING_WARMUP_TIME_CONTROL, RECURRING_WARMUP_DURATION_MINUTES,
                              RECURRING_WARMUP_PRIZE_POOL, RECURRING_WARMUP_HOUR_EAT,
                              REGIONAL_TOURNAMENT_NAME_TEMPLATE, REGIONAL_TOURNAMENT_DESCRIPTION_TEMPLATE,
                              REGIONAL_TOURNAMENT_TIME_CONTROL, REGIONAL_TOURNAMENT_DURATION_MINUTES,
                              REGIONAL_TOURNAMENT_PRIZE_POOL, REGIONAL_TOURNAMENT_HOUR_LOCAL,
                              REGIONAL_TOURNAMENT_UTC_OFFSET)
from app_core.medals import _grant_tournament_medals
from app_core.models import TournamentResultRequest, TournamentStartRequest
from app_core.rating import elo_col_for_tc, update_elos
from app_core.state import _active_pairing_loops, _arena_pair_locks, _player_score_locks, _submit_result_locks, _tournament_locks, active_games, tournament_connections, tournament_player_game
from app_core.ws_utils import arena_send, broadcast, new_game

async def tournament_handle_forfeit(game: dict, loser_id: str, winner_id: str, result: str):
    """
    On a first-move-timeout forfeit in an arena tournament game:
      - The player who failed to move (loser) is silently paused —
        they will not be paired again until they click Resume.
      - The player who showed up (winner) is sent back into the pairing
        pool immediately, available for a new opponent.
      - The tournament_games row is marked with the result, so the
        pairings list stops showing a "Watch" link for a game that's
        actually finished (previously this was never written for
        forfeits, only for normal game-ends submitted via the client,
        leaving forfeited games looking permanently in-progress).
    No-op for casual (non-tournament) games.
    """
    tid = game.get("tournament_id")
    if not tid:
        return
    conns = tournament_connections.get(tid, {})
    pg    = tournament_player_game.setdefault(tid, {})

    # Mark the tournament_games row so it stops appearing as in-progress.
    # This mirrors what /api/tournament/result does for normal game-ends,
    # but is called directly server-side since there's no client to call
    # the endpoint when a player has forfeited by never connecting at all.
    db_game_id = game.get("tournament_db_id")
    if db_game_id and SUPABASE_SERVICE_KEY:
        try:
            async with httpx.AsyncClient() as client:
                await client.patch(
                    f"{SUPABASE_URL}/rest/v1/tournament_games",
                    params={"id": f"eq.{db_game_id}"},
                    headers={"apikey": SUPABASE_SERVICE_KEY,
                             "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                             "Content-Type": "application/json"},
                    json={"result": result, "played_at": datetime.utcnow().isoformat()}
                )
        except Exception as e:
            print(f"[arena] failed to write forfeit result for {db_game_id}: {e}", flush=True)

    if loser_id in conns:
        conns[loser_id]["available"] = False
        conns[loser_id]["paused"]    = True
        loser_ws = conns[loser_id].get("ws")
        if loser_ws:
            try:
                await arena_send(loser_ws, {"type": "paused_state_sync", "paused": True})
            except Exception:
                pass
    if loser_id in pg:
        pg[loser_id] = None

    if winner_id in conns:
        if conns[winner_id].get("connected"):
            conns[winner_id]["available"] = True
        conns[winner_id]["paused"]    = False
    if winner_id in pg:
        pg[winner_id] = None

    print(f"[DBG forfeit] {tid}: loser={loser_id} paused={conns.get(loser_id,{}).get('paused')} | "
          f"winner={winner_id} available={conns.get(winner_id,{}).get('available')} "
          f"connected={conns.get(winner_id,{}).get('connected')} pg={pg.get(winner_id)}", flush=True)
    print(f"[arena] forfeit in {tid}: {loser_id} paused, {winner_id} returned to pool", flush=True)

async def arena_auto_start_scheduler():
    """Poll every 30s, auto-start Arena tournaments whose starts_at has passed,
    and auto-end Arena tournaments whose duration has expired."""
    await asyncio.sleep(10)
    while True:
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(
                    f"{SUPABASE_URL}/rest/v1/tournaments",
                    params={"status": "eq.upcoming", "format": "eq.arena",
                            "select": "id,starts_at"},
                    headers={"apikey": SUPABASE_SERVICE_KEY,
                             "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
                )
                for t in r.json():
                    starts = datetime.fromisoformat(t["starts_at"].replace("Z", "+00:00"))
                    if datetime.now(timezone.utc) >= starts:
                        lock_key = f"start_{t['id']}"
                        if lock_key not in _tournament_locks:
                            asyncio.create_task(arena_auto_start(t["id"]))

                r2 = await client.get(
                    f"{SUPABASE_URL}/rest/v1/tournaments",
                    params={"status": "eq.active", "format": "eq.arena",
                            "select": "id,starts_at,duration_minutes"},
                    headers={"apikey": SUPABASE_SERVICE_KEY,
                             "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
                )
                now_utc = datetime.now(timezone.utc)
                for t in r2.json():
                    if not t.get("duration_minutes"):
                        continue
                    starts = datetime.fromisoformat(t["starts_at"].replace("Z", "+00:00"))
                    ends   = starts + timedelta(minutes=t["duration_minutes"])
                    if now_utc >= ends:
                        tid = str(t["id"])
                        lock_key = f"end_{tid}"
                        if lock_key in _tournament_locks:
                            continue
                        _tournament_locks.add(lock_key)
                        try:
                            # Zero-participant tournaments have nothing worth
                            # keeping — no games, no standings, nothing a
                            # "completed" listing would ever show. Delete the
                            # row outright instead of marking it completed,
                            # rather than letting empty tournaments pile up
                            # in tournament history forever.
                            players_check = await client.get(
                                f"{SUPABASE_URL}/rest/v1/tournament_players",
                                params={"tournament_id": f"eq.{tid}", "select": "id", "limit": 1},
                                headers={"apikey": SUPABASE_SERVICE_KEY,
                                         "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
                            )
                            if not players_check.json():
                                await client.delete(
                                    f"{SUPABASE_URL}/rest/v1/tournaments",
                                    params={"id": f"eq.{tid}"},
                                    headers={"apikey": SUPABASE_SERVICE_KEY,
                                             "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
                                )
                                print(f"[arena] deleted empty tournament {tid} (0 participants)", flush=True)
                                tournament_connections.pop(tid, None)
                                tournament_player_game.pop(tid, None)
                                continue

                            await client.patch(
                                f"{SUPABASE_URL}/rest/v1/tournaments",
                                params={"id": f"eq.{tid}"},
                                headers={"apikey": SUPABASE_SERVICE_KEY,
                                         "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                                         "Content-Type": "application/json"},
                                json={"status": "completed"}
                            )
                            print(f"[arena] auto-ended {tid}", flush=True)
                            asyncio.create_task(_grant_tournament_medals(tid, client))
                            conns = tournament_connections.get(tid, {})
                            ended_msg = {"type": "tournament_ended"}
                            for uid, info in list(conns.items()):
                                await arena_send(info["ws"], ended_msg)
                            tournament_connections.pop(tid, None)
                            tournament_player_game.pop(tid, None)
                        except Exception as e:
                            print(f"[arena] auto-end error {tid}: {e}", flush=True)
                        finally:
                            _tournament_locks.discard(lock_key)
        except Exception as e:
            print(f"[scheduler] {e}", flush=True)
        await asyncio.sleep(30)


def _last_friday_of_month(year: int, month: int) -> datetime:
    if month == 12:
        first_of_next = datetime(year + 1, 1, 1)
    else:
        first_of_next = datetime(year, month + 1, 1)
    last_day = first_of_next - timedelta(days=1)
    offset = (last_day.weekday() - 4) % 7
    return last_day - timedelta(days=offset)


def _next_recurring_occurrence_utc() -> datetime:
    eat = timezone(timedelta(hours=3))
    now_utc = datetime.now(timezone.utc)
    y, m = now_utc.year, now_utc.month
    for _ in range(3):
        lf = _last_friday_of_month(y, m)
        occurrence_eat = lf.replace(hour=RECURRING_TOURNAMENT_HOUR_EAT, minute=0, second=0, tzinfo=eat)
        occurrence_utc = occurrence_eat.astimezone(timezone.utc)
        if occurrence_utc > now_utc:
            return occurrence_utc
        m += 1
        if m > 12:
            m = 1
            y += 1
    raise RuntimeError("could not compute next recurring tournament occurrence")


async def recurring_tournament_scheduler():
    await asyncio.sleep(20)
    while True:
        try:
            occurrence_utc = _next_recurring_occurrence_utc()
            starts_at_iso = occurrence_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            async with httpx.AsyncClient() as client:
                existing = await client.get(
                    f"{SUPABASE_URL}/rest/v1/tournaments",
                    params={"name": f"eq.{RECURRING_TOURNAMENT_NAME}",
                            "starts_at": f"eq.{starts_at_iso}",
                            "select": "id"},
                    headers={"apikey": SUPABASE_SERVICE_KEY,
                             "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
                )
                if existing.status_code == 200 and not existing.json():
                    admin_id = next(iter(ADMIN_USER_IDS), None)
                    row = {
                        "name":             RECURRING_TOURNAMENT_NAME,
                        "description":      RECURRING_TOURNAMENT_DESCRIPTION,
                        "format":           "arena",
                        "time_control":     RECURRING_TOURNAMENT_TIME_CONTROL,
                        "rounds":           0,
                        "max_players":      9999,
                        "country":          None,
                        "region":           None,
                        "starts_at":        starts_at_iso,
                        "created_by":       admin_id,
                        "status":           "upcoming",
                        "duration_minutes": RECURRING_TOURNAMENT_DURATION_MINUTES,
                        "prize_pool":       RECURRING_TOURNAMENT_PRIZE_POOL,
                    }
                    r = await client.post(
                        f"{SUPABASE_URL}/rest/v1/tournaments",
                        headers={"apikey": SUPABASE_SERVICE_KEY,
                                 "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                                 "Content-Type": "application/json",
                                 "Prefer": "return=representation"},
                        json=row
                    )
                    if r.status_code in (200, 201):
                        print(f"[recurring] auto-created next Grand Prix: {starts_at_iso}", flush=True)
                    else:
                        print(f"[recurring] auto-create FAILED ({r.status_code}): {r.text}", flush=True)
        except Exception as e:
            print(f"[recurring] scheduler error: {e}", flush=True)
        await asyncio.sleep(86400)


def _is_last_friday_of_month(d: datetime) -> bool:
    lf = _last_friday_of_month(d.year, d.month)
    return d.year == lf.year and d.month == lf.month and d.day == lf.day


async def weekly_warmup_scheduler():
    await asyncio.sleep(35)
    while True:
        try:
            eat = timezone(timedelta(hours=3))
            now_utc = datetime.now(timezone.utc)
            now_eat = now_utc.astimezone(eat)
            days_until_friday = (4 - now_eat.weekday()) % 7
            candidate = (now_eat + timedelta(days=days_until_friday)).replace(
                hour=RECURRING_WARMUP_HOUR_EAT, minute=0, second=0, microsecond=0)
            if candidate <= now_eat:
                candidate += timedelta(days=7)

            async with httpx.AsyncClient() as client:
                for week in range(2):
                    friday_eat = candidate + timedelta(weeks=week)
                    if _is_last_friday_of_month(friday_eat):
                        continue
                    occurrence_utc = friday_eat.astimezone(timezone.utc)
                    starts_at_iso = occurrence_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

                    existing = await client.get(
                        f"{SUPABASE_URL}/rest/v1/tournaments",
                        params={"name": f"eq.{RECURRING_WARMUP_NAME}",
                                "starts_at": f"eq.{starts_at_iso}",
                                "select": "id"},
                        headers={"apikey": SUPABASE_SERVICE_KEY,
                                 "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
                    )
                    if existing.status_code == 200 and not existing.json():
                        admin_id = next(iter(ADMIN_USER_IDS), None)
                        row = {
                            "name":             RECURRING_WARMUP_NAME,
                            "description":      RECURRING_WARMUP_DESCRIPTION,
                            "format":           "arena",
                            "time_control":     RECURRING_WARMUP_TIME_CONTROL,
                            "rounds":           0,
                            "max_players":      9999,
                            "country":          None,
                            "region":           None,
                            "starts_at":        starts_at_iso,
                            "created_by":       admin_id,
                            "status":           "upcoming",
                            "duration_minutes": RECURRING_WARMUP_DURATION_MINUTES,
                            "prize_pool":       RECURRING_WARMUP_PRIZE_POOL,
                        }
                        r = await client.post(
                            f"{SUPABASE_URL}/rest/v1/tournaments",
                            headers={"apikey": SUPABASE_SERVICE_KEY,
                                     "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                                     "Content-Type": "application/json",
                                     "Prefer": "return=representation"},
                            json=row
                        )
                        if r.status_code in (200, 201):
                            print(f"[warmup] auto-created: {starts_at_iso}", flush=True)
                        else:
                            print(f"[warmup] auto-create FAILED ({r.status_code}): {r.text}", flush=True)
        except Exception as e:
            print(f"[warmup] scheduler error: {e}", flush=True)
        await asyncio.sleep(86400)


async def weekly_regional_scheduler():
    await asyncio.sleep(45)
    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            async with httpx.AsyncClient() as client:
                for region_key, (region_label, utc_offset) in REGIONAL_TOURNAMENT_UTC_OFFSET.items():
                    tz = timezone(timedelta(hours=utc_offset))
                    now_local = now_utc.astimezone(tz)
                    days_until_saturday = (5 - now_local.weekday()) % 7
                    candidate = (now_local + timedelta(days=days_until_saturday)).replace(
                        hour=REGIONAL_TOURNAMENT_HOUR_LOCAL, minute=0, second=0, microsecond=0)
                    if candidate <= now_local:
                        candidate += timedelta(days=7)
                    occurrence_utc = candidate.astimezone(timezone.utc)
                    starts_at_iso = occurrence_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
                    tournament_name = REGIONAL_TOURNAMENT_NAME_TEMPLATE.format(region=region_label)

                    existing = await client.get(
                        f"{SUPABASE_URL}/rest/v1/tournaments",
                        params={"region": f"eq.{region_key}",
                                "status": "in.(upcoming,active)",
                                "select": "id,status,starts_at"},
                        headers={"apikey": SUPABASE_SERVICE_KEY,
                                 "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
                    )
                    if existing.status_code == 200 and not existing.json():
                        admin_id = next(iter(ADMIN_USER_IDS), None)
                        row = {
                            "name":             tournament_name,
                            "description":      REGIONAL_TOURNAMENT_DESCRIPTION_TEMPLATE.format(region=region_label),
                            "format":           "arena",
                            "time_control":     REGIONAL_TOURNAMENT_TIME_CONTROL,
                            "rounds":           0,
                            "max_players":      9999,
                            "country":          None,
                            "region":           region_key,
                            "starts_at":        starts_at_iso,
                            "created_by":       admin_id,
                            "status":           "upcoming",
                            "duration_minutes": REGIONAL_TOURNAMENT_DURATION_MINUTES,
                            "prize_pool":       REGIONAL_TOURNAMENT_PRIZE_POOL,
                        }
                        r = await client.post(
                            f"{SUPABASE_URL}/rest/v1/tournaments",
                            headers={"apikey": SUPABASE_SERVICE_KEY,
                                     "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                                     "Content-Type": "application/json",
                                     "Prefer": "return=representation"},
                            json=row
                        )
                        if r.status_code in (200, 201):
                            print(f"[regional] auto-created {region_key}: {starts_at_iso}", flush=True)
                        else:
                            print(f"[regional] auto-create FAILED for {region_key} ({r.status_code}): {r.text}", flush=True)
        except Exception as e:
            print(f"[regional] scheduler error: {e}", flush=True)
        await asyncio.sleep(86400)


async def arena_auto_start(tournament_id: str):
    lock_key = f"start_{tournament_id}"
    if lock_key in _tournament_locks:
        return
    _tournament_locks.add(lock_key)
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{SUPABASE_URL}/rest/v1/tournaments",
                params={"id": f"eq.{tournament_id}", "select": "status,format"},
                headers={"apikey": SUPABASE_SERVICE_KEY,
                         "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
            )
            rows = r.json()
            if not rows or rows[0]["status"] != "upcoming":
                return
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/tournaments",
                params={"id": f"eq.{tournament_id}"},
                headers={"apikey": SUPABASE_SERVICE_KEY,
                         "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                         "Content-Type": "application/json"},
                json={"status": "active"}
            )
        print(f"[arena] auto-started {tournament_id}", flush=True)
        conns = tournament_connections.get(tournament_id, {})
        for uid, info in list(conns.items()):
            ws = info.get("ws")
            if ws:
                try:
                    await arena_send(ws, {"type": "tournament_started"})
                except Exception:
                    pass
        if rows[0].get("format") == "arena":
            asyncio.create_task(arena_pairing_loop(tournament_id))
    except Exception as e:
        print(f"[arena] auto-start error: {e}", flush=True)
    finally:
        _tournament_locks.discard(lock_key)

async def _arena_pair_impl(tournament_id: str):
    conns = tournament_connections.get(tournament_id, {})
    pg    = tournament_player_game.setdefault(tournament_id, {})
    available = [uid for uid, info in conns.items()
                 if info.get("available") and not info.get("paused")
                 and info.get("connected") and pg.get(uid) is None]
    _snapshot = {uid: {"avail": i.get("available"), "paused": i.get("paused"),
                       "conn": i.get("connected"), "pg": pg.get(uid)}
                 for uid, i in conns.items()}
    print(f"[DBG pair] {tournament_id}: tick — eligible={available} | all={_snapshot}", flush=True)
    if len(available) < 1:
        return

    try:
        async with httpx.AsyncClient() as _tc:
            _tr = await _tc.get(
                f"{SUPABASE_URL}/rest/v1/tournaments",
                params={"id": f"eq.{tournament_id}",
                        "select": "starts_at,duration_minutes,time_control,status"},
                headers={"apikey": SUPABASE_SERVICE_KEY,
                         "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
            )
            _rows = _tr.json()
        if _rows:
            _t = _rows[0]
            if _t.get("status") != "active":
                return
            _starts    = datetime.fromisoformat(_t["starts_at"].replace("Z", "+00:00"))
            _ends      = _starts + timedelta(minutes=_t.get("duration_minutes") or 60)
            _remaining = (_ends - datetime.now(timezone.utc)).total_seconds()
            _tc_str    = _t.get("time_control") or "5+0"
            try:
                _tc_secs = float(_tc_str.split("+")[0]) * 60
            except (ValueError, IndexError):
                _tc_secs = 300
            _cutoff = _tc_secs * 0.5
            if _remaining <= _cutoff:
                for uid in available:
                    await arena_send(conns[uid]["ws"], {
                        "type":    "pairings_closed",
                        "message": f"Pairings closed — {int(_remaining)}s left, tournament ending soon.",
                    })
                print(f"[arena] pairings closed for {tournament_id} "
                      f"({int(_remaining)}s left, cutoff {int(_cutoff)}s)", flush=True)
                return
    except Exception as _e:
        print(f"[arena] cutoff check error: {_e}", flush=True)

    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{SUPABASE_URL}/rest/v1/tournament_games",
                params={"tournament_id": f"eq.{tournament_id}",
                        "select": "white_id,black_id"},
                headers={"apikey": SUPABASE_SERVICE_KEY,
                         "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
            )
        pair_count: dict = {}
        color_balance: dict = {}
        for g in r.json():
            key = tuple(sorted([g["white_id"], g["black_id"]]))
            pair_count[key] = pair_count.get(key, 0) + 1
            color_balance[g["white_id"]] = color_balance.get(g["white_id"], 0) + 1
            color_balance[g["black_id"]] = color_balance.get(g["black_id"], 0) - 1
    except Exception:
        pair_count = {}
        color_balance = {}

    available.sort(key=lambda uid: (
        -conns[uid].get("rounds_waited", 0),
        -conns[uid].get("score", 0),
        -conns[uid].get("elo", 1500),
    ))

    def times_played(p1, p2):
        return pair_count.get(tuple(sorted([p1, p2])), 0)

    def assign_colors(p1, p2):
        b1 = color_balance.get(p1, 0)
        b2 = color_balance.get(p2, 0)
        if b1 > b2:
            return p2, p1
        elif b2 > b1:
            return p1, p2
        else:
            return p1, p2

    paired, used = [], set()
    for i, p1 in enumerate(available):
        if p1 in used:
            continue
        best_p2   = None
        best_score = None
        for p2 in available[i+1:]:
            if p2 in used:
                continue
            tp = times_played(p1, p2)
            score_diff = abs(conns[p1].get("score", 0) - conns[p2].get("score", 0))
            candidate = (tp, score_diff)
            if best_score is None or candidate < best_score:
                best_p2   = p2
                best_score = candidate
        if best_p2:
            white_id, black_id = assign_colors(p1, best_p2)
            paired.append((white_id, black_id))
            used.add(p1)
            used.add(best_p2)

    for uid in available:
        if uid in used:
            conns[uid]["rounds_waited"] = 0
        else:
            conns[uid]["rounds_waited"] = conns[uid].get("rounds_waited", 0) + 1

    if len(available) % 2 == 1:
        for uid in available:
            if uid not in used:
                await arena_send(conns[uid]["ws"], {
                    "type":    "waiting",
                    "message": "Waiting for an available opponent…"
                })
                print(f"[DBG pair] {tournament_id}: {uid} left WAITING (odd player out)", flush=True)
                break

    print(f"[DBG pair] {tournament_id}: DECISION paired={paired} "
          f"(from {len(available)} eligible)", flush=True)

    for white_id, black_id in paired:
        pg[white_id] = "pending"
        pg[black_id] = "pending"
        asyncio.create_task(arena_launch_game(tournament_id, white_id, black_id))

async def arena_pair(tournament_id: str):
    lock = _arena_pair_locks.setdefault(tournament_id, asyncio.Lock())
    async with lock:
        await _arena_pair_impl(tournament_id)

async def arena_pairing_loop(tournament_id: str):
    if tournament_id in _active_pairing_loops:
        print(f"[DBG loop] {tournament_id}: pairing loop already running — not starting a second", flush=True)
        return
    _active_pairing_loops.add(tournament_id)
    print(f"[DBG loop] {tournament_id}: PAIRING LOOP STARTED", flush=True)
    try:
        while True:
            await asyncio.sleep(5)
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.get(
                        f"{SUPABASE_URL}/rest/v1/tournaments",
                        params={"id": f"eq.{tournament_id}", "select": "status"},
                        headers={"apikey": SUPABASE_SERVICE_KEY,
                                 "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
                    )
                    rows = r.json()
                if not rows or rows[0].get("status") != "active":
                    print(f"[DBG loop] {tournament_id}: STOPPING (status={rows[0].get('status') if rows else 'not found'})", flush=True)
                    break
            except Exception as e:
                print(f"[DBG loop] {tournament_id}: status check error: {e}", flush=True)
                continue

            try:
                await arena_pair(tournament_id)
            except Exception as e:
                print(f"[arena] pairing loop tick error for {tournament_id}: {e}", flush=True)
    finally:
        _active_pairing_loops.discard(tournament_id)
        print(f"[arena] pairing loop ended for {tournament_id}", flush=True)

async def arena_launch_game(tournament_id: str, white_id: str, black_id: str):
    conns = tournament_connections.get(tournament_id, {})
    pg    = tournament_player_game.setdefault(tournament_id, {})
    if white_id not in conns or black_id not in conns:
        return
    existing_white = pg.get(white_id)
    existing_black = pg.get(black_id)
    if existing_white not in (None, "pending") or existing_black not in (None, "pending"):
        print(f"[arena] REFUSING duplicate launch for {tournament_id}: "
              f"white={white_id} (pg={existing_white}), black={black_id} (pg={existing_black})", flush=True)
        if existing_white not in (None, "pending") and white_id in conns:
            pg[black_id] = None
            if black_id in conns and not conns[black_id].get("paused") and conns[black_id].get("connected"):
                conns[black_id]["available"] = True
        elif existing_black not in (None, "pending") and black_id in conns:
            pg[white_id] = None
            if white_id in conns and not conns[white_id].get("paused") and conns[white_id].get("connected"):
                conns[white_id]["available"] = True
        return
    white_info = conns[white_id]
    black_info = conns[black_id]
    pg[white_id] = "pending"; pg[black_id] = "pending"
    white_info["available"] = False; black_info["available"] = False
    game_id = uuid.uuid4().hex[:12]

    time_control = "5+0"
    try:
        async with httpx.AsyncClient() as _tc_client:
            _tc_r = await _tc_client.get(
                f"{SUPABASE_URL}/rest/v1/tournaments",
                params={"id": f"eq.{tournament_id}", "select": "time_control"},
                headers={"apikey": SUPABASE_SERVICE_KEY,
                         "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
            )
            _tc_rows = _tc_r.json()
            if _tc_rows:
                time_control = _tc_rows[0].get("time_control", "5+0")
    except Exception:
        pass

    try:
        async with httpx.AsyncClient() as client:
            ins = await client.post(
                f"{SUPABASE_URL}/rest/v1/tournament_games",
                headers={"apikey": SUPABASE_SERVICE_KEY,
                         "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                         "Content-Type": "application/json",
                         "Prefer": "return=representation"},
                json={"tournament_id": tournament_id, "round": 1,
                      "white_id": white_id, "black_id": black_id,
                      "white_username": white_info["username"],
                      "black_username": black_info["username"],
                      "game_id": game_id}
            )
            db_rows = ins.json()
            db_game_id = db_rows[0]["id"] if db_rows else None
            if db_game_id is None:
                print(f"[DBG launch] {tournament_id}: WARNING tournament_games insert returned NO ROW "
                      f"(status={ins.status_code}, body={str(db_rows)[:200]}) — scoring will be impossible for this game", flush=True)
    except Exception as e:
        print(f"[DBG launch] {tournament_id}: tournament_games INSERT FAILED: {type(e).__name__}: {e}", flush=True)
        pg[white_id] = None; pg[black_id] = None
        white_info["available"] = True; black_info["available"] = True
        return

    pg[white_id] = game_id; pg[black_id] = game_id
    game = new_game(game_id, None, None, white_id, black_id, time_control)
    game["tournament_id"]    = tournament_id
    game["tournament_db_id"] = db_game_id
    game["white_profile"] = {"username": white_info["username"],
                             "elo": white_info.get("elo", 1500), "user_id": white_id}
    game["black_profile"] = {"username": black_info["username"],
                             "elo": black_info.get("elo", 1500), "user_id": black_id}
    game["last_move_ts"]        = time.time()
    game["first_move_deadline"] = time.time() + game.get("first_move_timeout", 60)
    active_games[game_id] = game
    from routes.game import clock_loop, first_move_timeout_loop
    asyncio.create_task(clock_loop(game_id))
    asyncio.create_task(first_move_timeout_loop(game_id))
    print(f"[arena] {game_id}: {white_info['username']} vs {black_info['username']}", flush=True)

    elo_col = elo_col_for_tc(time_control)
    w_display_elo = white_info.get(elo_col) or white_info.get("elo", 1500)
    b_display_elo = black_info.get(elo_col) or black_info.get("elo", 1500)

    conns = tournament_connections.get(tournament_id, {})
    ranked = sorted(conns.values(), key=lambda x: (-x.get("score", 0), -x.get("elo", 1500)))
    uid_to_rank = {list(conns.keys())[i]: i+1
                   for i, uid in enumerate(c.get("user_id", "") for c in ranked)
                   if True}
    sorted_uids = sorted(conns.keys(),
                         key=lambda u: (-conns[u].get("score",0), -conns[u].get("elo",1500)))
    uid_rank = {uid: i+1 for i, uid in enumerate(sorted_uids)}
    w_rank = uid_rank.get(white_id, 0)
    b_rank = uid_rank.get(black_id, 0)

    await arena_send(white_info["ws"], {
        "type": "game_ready", "game_id": game_id,
        "color": "white", "opponent": black_info["username"],
        "opponent_elo": b_display_elo,
        "my_rank": w_rank, "opponent_rank": b_rank,
        "tournament_db_id": db_game_id,
        "time_control": time_control,
    })
    await arena_send(black_info["ws"], {
        "type": "game_ready", "game_id": game_id,
        "color": "black", "opponent": white_info["username"],
        "opponent_elo": w_display_elo,
        "my_rank": b_rank, "opponent_rank": w_rank,
        "tournament_db_id": db_game_id,
        "time_control": time_control,
    })
    print(f"[DBG launch] {tournament_id}: game_ready SENT game_id={game_id} db_id={db_game_id} "
          f"white={white_id}(ws={'live' if white_info.get('ws') else 'MISSING'}) "
          f"black={black_id}(ws={'live' if black_info.get('ws') else 'MISSING'})", flush=True)

async def arena_pair_delayed(tournament_id: str, delay: float = 2.5):
    await asyncio.sleep(delay)
    await arena_pair(tournament_id)

async def _arena_end_exhausted(tournament_id: str):
    """End an Arena tournament because all players have played each other."""
    lock_key = f"end_{tournament_id}"
    if lock_key in _tournament_locks:
        return
    _tournament_locks.add(lock_key)
    try:
        async with httpx.AsyncClient() as client:
            # Same zero-participant guard as the time-based auto-end path —
            # defensive here, since reaching "everyone has played everyone"
            # genuinely requires participants to exist in practice, but
            # costs nothing to check and keeps both completion paths
            # consistent.
            players_check = await client.get(
                f"{SUPABASE_URL}/rest/v1/tournament_players",
                params={"tournament_id": f"eq.{tournament_id}", "select": "id", "limit": 1},
                headers={"apikey": SUPABASE_SERVICE_KEY,
                         "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
            )
            if not players_check.json():
                await client.delete(
                    f"{SUPABASE_URL}/rest/v1/tournaments",
                    params={"id": f"eq.{tournament_id}"},
                    headers={"apikey": SUPABASE_SERVICE_KEY,
                             "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
                )
                print(f"[arena] deleted empty tournament {tournament_id} (0 participants)", flush=True)
                tournament_connections.pop(tournament_id, None)
                tournament_player_game.pop(tournament_id, None)
                return

            await client.patch(
                f"{SUPABASE_URL}/rest/v1/tournaments",
                params={"id": f"eq.{tournament_id}"},
                headers={"apikey": SUPABASE_SERVICE_KEY,
                         "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                         "Content-Type": "application/json"},
                json={"status": "completed"}
            )
        print(f"[arena] exhausted — ended {tournament_id}", flush=True)
        conns = tournament_connections.get(tournament_id, {})
        asyncio.create_task(_grant_tournament_medals(tournament_id))
        msg = {"type": "tournament_ended",
               "reason": "All players have faced each other. Final standings are shown below."}
        for uid, info in list(conns.items()):
            await arena_send(info["ws"], msg)
        tournament_connections.pop(tournament_id, None)
        tournament_player_game.pop(tournament_id, None)
    except Exception as e:
        print(f"[arena] exhausted-end error: {e}", flush=True)
    finally:
        _tournament_locks.discard(lock_key)

@router.websocket("/ws/tournament/{tournament_id}")
async def tournament_ws(ws: WebSocket, tournament_id: str):
    await ws.accept()
    user_id = None
    try:
        ident = await asyncio.wait_for(ws.receive_json(), timeout=20)
        user_id  = ident.get("user_id")
        username = ident.get("username", "?")
        elo      = int(ident.get("elo", 1500))
        score    = float(ident.get("score", 0))
    except Exception as _e:
        print(f"[DBG join] {tournament_id}: identity receive FAILED: {type(_e).__name__}: {_e}", flush=True)
        await ws.close(); return
    if not user_id:
        print(f"[DBG join] {tournament_id}: no user_id in identity payload {ident!r}", flush=True)
        await ws.close(); return
    print(f"[DBG join] {tournament_id}: identity ok user={user_id} name={username} elo={elo} score={score}", flush=True)

    try:
        async with httpx.AsyncClient() as client:
            reg_check = await client.get(
                f"{SUPABASE_URL}/rest/v1/tournament_players",
                params={"tournament_id": f"eq.{tournament_id}",
                        "user_id": f"eq.{user_id}", "select": "id"},
                headers={"apikey": SUPABASE_SERVICE_KEY,
                         "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
            )
        if not reg_check.json():
            print(f"[DBG join] {tournament_id}: user={user_id} NOT REGISTERED — rejecting WS (must click Join first)", flush=True)
            await arena_send(ws, {
                "type": "error",
                "detail": "You haven't joined this tournament yet. Go to the tournament page and click Join."
            })
            await ws.close()
            return
        print(f"[DBG join] {tournament_id}: user={user_id} registration OK", flush=True)
    except Exception as e:
        print(f"[DBG join] {tournament_id}: registration check EXCEPTION for {user_id}: {type(e).__name__}: {e}", flush=True)
        await ws.close()
        return

    tournament_connections.setdefault(tournament_id, {})
    tournament_player_game.setdefault(tournament_id, {})

    _existing = tournament_connections[tournament_id].get(user_id, {})
    tournament_connections[tournament_id][user_id] = {
        "ws":               ws,
        "username":         username,
        "elo":              elo,
        "elo_bullet":       int(ident.get("elo_bullet", elo)),
        "elo_blitz":        int(ident.get("elo_blitz",  elo)),
        "elo_rapid":        int(ident.get("elo_rapid",  elo)),
        "score":            _existing.get("score", score),
        "available":        not _existing.get("paused", False),
        "paused":           _existing.get("paused", False),
        "consecutive_wins": _existing.get("consecutive_wins", 0),
        "rounds_waited":    _existing.get("rounds_waited", 0),
        "connected":        True,
    }
    if tournament_player_game[tournament_id].get(user_id):
        tournament_connections[tournament_id][user_id]["available"] = False

    _st = tournament_connections[tournament_id][user_id]
    print(f"[DBG join] {tournament_id}: state set user={user_id} "
          f"available={_st['available']} paused={_st['paused']} "
          f"pg={tournament_player_game[tournament_id].get(user_id)} "
          f"carried_from_existing={{score:{_existing.get('score')}, paused:{_existing.get('paused')}}} "
          f"total_conns={len(tournament_connections[tournament_id])}", flush=True)

    await arena_send(ws, {"type": "connected", "user_id": user_id})
    if _existing.get("paused"):
        await arena_send(ws, {"type": "paused_state_sync", "paused": True})

    pending_game_id = tournament_player_game[tournament_id].get(user_id)
    if pending_game_id and pending_game_id != "pending":
        g = active_games.get(pending_game_id)
        if g and not g.get("over"):
            is_white = g.get("white_id") == user_id
            opp_id = g.get("black_id") if is_white else g.get("white_id")
            opp_info = tournament_connections.get(tournament_id, {}).get(opp_id, {})
            conns_now = tournament_connections[tournament_id]
            sorted_uids = sorted(conns_now.keys(),
                                  key=lambda u: (-conns_now[u].get("score", 0), -conns_now[u].get("elo", 1500)))
            uid_rank = {u: i + 1 for i, u in enumerate(sorted_uids)}
            await arena_send(ws, {
                "type": "game_ready", "game_id": pending_game_id,
                "color": "white" if is_white else "black",
                "opponent": opp_info.get("username", "Opponent"),
                "opponent_elo": opp_info.get("elo", 1500),
                "my_rank": uid_rank.get(user_id, 0),
                "opponent_rank": uid_rank.get(opp_id, 0),
                "tournament_db_id": g.get("tournament_db_id"),
                "time_control": g.get("time_control", "5+0"),
            })
            print(f"[arena] resent game_ready to {username} for {pending_game_id} on reconnect", flush=True)

    print(f"[arena] {username} connected to {tournament_id}", flush=True)

    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{SUPABASE_URL}/rest/v1/tournaments",
                params={"id": f"eq.{tournament_id}", "select": "status,format"},
                headers={"apikey": SUPABASE_SERVICE_KEY,
                         "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
            )
            rows = r.json()
        if rows and rows[0]["status"] == "active" and rows[0].get("format") == "arena":
            pass
    except Exception:
        pass

    try:
        while True:
            data = await ws.receive_json()
            if data.get("type") == "ping":
                await arena_send(ws, {"type": "pong"})

            elif data.get("type") == "available":
                conns = tournament_connections.get(tournament_id, {})
                pg    = tournament_player_game.get(tournament_id, {})
                _was_paused = conns.get(user_id, {}).get("paused")
                if user_id in conns:
                    if not conns[user_id].get("paused"):
                        conns[user_id]["available"] = True
                if user_id in pg and not conns.get(user_id, {}).get("paused"):
                    pg[user_id] = None
                print(f"[DBG avail] {tournament_id}: user={user_id} sent 'available' "
                      f"was_paused={_was_paused} -> available={conns.get(user_id,{}).get('available')} "
                      f"pg={pg.get(user_id)}", flush=True)

            elif data.get("type") == "pause":
                conns = tournament_connections.get(tournament_id, {})
                if user_id in conns:
                    conns[user_id]["available"] = False
                    conns[user_id]["paused"]    = True
                await arena_send(ws, {"type": "paused",
                    "message": "You are paused. You won't be paired until you resume."})
                print(f"[DBG pause] {tournament_id}: user={user_id} PAUSED "
                      f"available={conns.get(user_id,{}).get('available')} "
                      f"paused={conns.get(user_id,{}).get('paused')}", flush=True)

            elif data.get("type") == "resume":
                conns = tournament_connections.get(tournament_id, {})
                pg    = tournament_player_game.get(tournament_id, {})
                if user_id in conns:
                    conns[user_id]["available"] = True
                    conns[user_id]["paused"]    = False
                if user_id in pg:
                    pg[user_id] = None
                await arena_send(ws, {"type": "resumed",
                    "message": "You're back! Looking for an opponent…"})
                print(f"[DBG resume] {tournament_id}: user={user_id} RESUMED "
                      f"available={conns.get(user_id,{}).get('available')} "
                      f"paused={conns.get(user_id,{}).get('paused')} pg={pg.get(user_id)} "
                      f"connected={conns.get(user_id,{}).get('connected')}", flush=True)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[arena] tournament_ws error for {user_id} on {tournament_id}: "
              f"{type(e).__name__}: {e}", flush=True)
    finally:
        conns = tournament_connections.get(tournament_id, {})
        if user_id and user_id in conns:
            conns[user_id]["available"] = False
            conns[user_id]["connected"] = False
        print(f"[arena] {username} left {tournament_id}", flush=True)

def swiss_pair(players: list, existing_games: list) -> list:
    played = set()
    for g in existing_games:
        played.add((g['white_id'], g['black_id']))
        played.add((g['black_id'], g['white_id']))

    ranked = sorted(players, key=lambda p: (-p.get('score', 0), -p.get('elo', 1500)))
    paired = []
    used   = set()

    for i, p1 in enumerate(ranked):
        if p1['user_id'] in used:
            continue
        for j in range(i + 1, len(ranked)):
            p2 = ranked[j]
            if p2['user_id'] in used:
                continue
            if (p1['user_id'], p2['user_id']) not in played:
                paired.append((p1, p2))
                used.add(p1['user_id'])
                used.add(p2['user_id'])
                break

    for p in ranked:
        if p['user_id'] not in used:
            paired.append((p, None))

    return paired

@router.post("/api/tournament/start")
async def start_tournament(req: TournamentStartRequest, authorization: str = Header(None)):
    """Start a tournament and generate round 1 pairings."""
    if not SUPABASE_SERVICE_KEY:
        raise HTTPException(503, "Service unavailable")
    user_id = await verify_jwt(authorization)
    lock_key = f"start_{req.tournament_id}"
    if lock_key in _tournament_locks:
        raise HTTPException(429, "Tournament start already in progress")
    _tournament_locks.add(lock_key)
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{SUPABASE_URL}/rest/v1/tournaments",
                params={"id": f"eq.{req.tournament_id}", "select": "*"},
                headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
            )
            ts = r.json()
            if not ts:
                raise HTTPException(404, "Tournament not found")
            t = ts[0]
            if t["created_by"] != user_id:
                raise HTTPException(403, "Only the creator can start the tournament")
            if t["status"] != "upcoming":
                raise HTTPException(400, "Tournament already started or completed")

            r2 = await client.get(
                f"{SUPABASE_URL}/rest/v1/tournament_players",
                params={"tournament_id": f"eq.{req.tournament_id}", "select": "*"},
                headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
            )
            players = r2.json()
            if len(players) < 2:
                raise HTTPException(400, "Need at least 2 players to start")

            await client.patch(
                f"{SUPABASE_URL}/rest/v1/tournaments",
                params={"id": f"eq.{req.tournament_id}"},
                headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                         "Content-Type": "application/json"},
                json={"status": "active"}
            )

            conns = tournament_connections.get(req.tournament_id, {})
            for uid, info in list(conns.items()):
                ws = info.get("ws")
                if ws:
                    try:
                        await arena_send(ws, {"type": "tournament_started"})
                    except Exception:
                        pass

            if t.get("format") == "arena":
                asyncio.create_task(arena_pairing_loop(req.tournament_id))
                return {"ok": True, "format": "arena", "players": len(players)}

            pairs = swiss_pair(players, [])
            games_to_insert = []
            for white, black in pairs:
                if black is None:
                    await client.patch(
                        f"{SUPABASE_URL}/rest/v1/tournament_players",
                        params={"tournament_id": f"eq.{req.tournament_id}", "user_id": f"eq.{white['user_id']}"},
                        headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                                 "Content-Type": "application/json"},
                        json={"score": white.get("score", 0) + 2}
                    )
                    continue
                games_to_insert.append({
                    "tournament_id":  req.tournament_id,
                    "round":          1,
                    "white_id":       white["user_id"],
                    "black_id":       black["user_id"],
                    "white_username": white["username"],
                    "black_username": black["username"],
                })

            if games_to_insert:
                await client.post(
                    f"{SUPABASE_URL}/rest/v1/tournament_games",
                    headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                             "Content-Type": "application/json", "Prefer": "return=minimal"},
                    json=games_to_insert
                )

        return {"ok": True, "round": 1, "pairings": len(games_to_insert)}
    finally:
        _tournament_locks.discard(f"start_{req.tournament_id}")

@router.post("/api/tournament/next-round")
async def next_round(req: TournamentStartRequest, authorization: str = Header(None)):
    """Generate pairings for the next round after all current round games are done."""
    if not SUPABASE_SERVICE_KEY:
        raise HTTPException(503, "Service unavailable")
    user_id = await verify_jwt(authorization)
    lock_key = f"next_{req.tournament_id}"
    if lock_key in _tournament_locks:
        raise HTTPException(429, "Round generation already in progress")
    _tournament_locks.add(lock_key)
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{SUPABASE_URL}/rest/v1/tournaments",
                params={"id": f"eq.{req.tournament_id}", "select": "*"},
                headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
            )
            t = r.json()[0]
            if t["created_by"] != user_id:
                raise HTTPException(403, "Only the creator can advance rounds")

            r2 = await client.get(
                f"{SUPABASE_URL}/rest/v1/tournament_games",
                params={"tournament_id": f"eq.{req.tournament_id}", "select": "*", "order": "round.desc"},
                headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
            )
            all_games = r2.json()
            if not all_games:
                raise HTTPException(400, "No games found")

            current_round = all_games[0]["round"]
            if current_round >= t["rounds"]:
                await client.patch(
                    f"{SUPABASE_URL}/rest/v1/tournaments",
                    params={"id": f"eq.{req.tournament_id}"},
                    headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                             "Content-Type": "application/json"},
                    json={"status": "completed"}
                )
                asyncio.create_task(_grant_tournament_medals(req.tournament_id, client))
                return {"ok": True, "completed": True}

            current_games = [g for g in all_games if g["round"] == current_round]
            pending = [g for g in current_games if not g.get("result")]
            if pending:
                raise HTTPException(400, f"{len(pending)} game(s) still pending in round {current_round}")

            r3 = await client.get(
                f"{SUPABASE_URL}/rest/v1/tournament_players",
                params={"tournament_id": f"eq.{req.tournament_id}", "select": "*"},
                headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
            )
            players = r3.json()

            next_r = current_round + 1
            pairs = swiss_pair(players, all_games)
            games_to_insert = []
            for white, black in pairs:
                if black is None:
                    await client.patch(
                        f"{SUPABASE_URL}/rest/v1/tournament_players",
                        params={"tournament_id": f"eq.{req.tournament_id}", "user_id": f"eq.{white['user_id']}"},
                        headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                                 "Content-Type": "application/json"},
                        json={"score": white.get("score", 0) + 2}
                    )
                    continue
                games_to_insert.append({
                    "tournament_id":  req.tournament_id,
                    "round":          next_r,
                    "white_id":       white["user_id"],
                    "black_id":       black["user_id"],
                    "white_username": white["username"],
                    "black_username": black["username"],
                })

            if games_to_insert:
                await client.post(
                    f"{SUPABASE_URL}/rest/v1/tournament_games",
                    headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                             "Content-Type": "application/json", "Prefer": "return=minimal"},
                    json=games_to_insert
                )

        return {"ok": True, "round": next_r, "pairings": len(games_to_insert)}
    finally:
        _tournament_locks.discard(f"next_{req.tournament_id}")

@router.post("/api/tournament/result")
async def submit_result(req: TournamentResultRequest, authorization: str = Header(None)):
    """
    Submit a tournament game result (client-called endpoint).
    """
    if not SUPABASE_SERVICE_KEY:
        raise HTTPException(503, "Service unavailable")
    user_id = await verify_jwt(authorization)
    if req.result not in ('white', 'black', 'draw'):
        raise HTTPException(400, "Invalid result")
    return await _apply_tournament_result(req.game_id, req.result, submitting_user_id=user_id)

async def _apply_tournament_result(game_db_id: str, result: str, submitting_user_id: str = None):
    """
    Core tournament-result scoring.
    """
    if result not in ('white', 'black', 'draw'):
        raise HTTPException(400, "Invalid result")

    lock = _submit_result_locks.setdefault(game_db_id, asyncio.Lock())
    async with lock:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{SUPABASE_URL}/rest/v1/tournament_games",
                params={"id": f"eq.{game_db_id}", "select": "*"},
                headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
            )
            games = r.json()
            if not games:
                raise HTTPException(404, "Game not found")
            g = games[0]
            if g.get('result'):
                raise HTTPException(400, "Result already submitted")

            if submitting_user_id is not None and submitting_user_id not in (g['white_id'], g['black_id']):
                raise HTTPException(403, "Not a player in this game")

            req_game_id = game_db_id
            req_result  = result

            tid = g['tournament_id']

            tc_r = await client.get(
                f"{SUPABASE_URL}/rest/v1/tournaments",
                params={"id": f"eq.{tid}", "select": "format,status,time_control"},
                headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
            )
            tc_rows = tc_r.json()
            time_control = tc_rows[0].get("time_control") if tc_rows else None
            elo_col = elo_col_for_tc(time_control)

            tournament_status = tc_rows[0].get("status") if tc_rows else None

            await client.patch(
                f"{SUPABASE_URL}/rest/v1/tournament_games",
                params={"id": f"eq.{req_game_id}"},
                headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                         "Content-Type": "application/json"},
                json={"result": req_result, "played_at": datetime.utcnow().isoformat()}
            )

            if tournament_status != "active":
                print(f"[arena] game {req_game_id} finished after tournament {tid} ended — "
                      f"result recorded, standings NOT updated", flush=True)
                return {"ok": True, "result": req_result, "counted_toward_standings": False}

            elo_r = await asyncio.gather(
                client.get(
                    f"{SUPABASE_URL}/rest/v1/profiles",
                    params={"user_id": f"eq.{g['white_id']}", "select": elo_col},
                    headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
                ),
                client.get(
                    f"{SUPABASE_URL}/rest/v1/profiles",
                    params={"user_id": f"eq.{g['black_id']}", "select": elo_col},
                    headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
                ),
            )
            w_elo = (elo_r[0].json()[0].get(elo_col) or 1500) if elo_r[0].json() else 1500
            b_elo = (elo_r[1].json()[0].get(elo_col) or 1500) if elo_r[1].json() else 1500

            conns_now = tournament_connections.get(tid, {})

            async def add_score(uid, pts, elo_snapshot):
                lock = _player_score_locks.setdefault((tid, uid), asyncio.Lock())
                async with lock:
                    if uid not in conns_now:
                        r2 = await client.get(
                            f"{SUPABASE_URL}/rest/v1/tournament_players",
                            params={"tournament_id": f"eq.{tid}", "user_id": f"eq.{uid}", "select": "score"},
                            headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
                        )
                        current = r2.json()[0].get('score', 0) if r2.json() else 0
                    else:
                        current = conns_now[uid].get("score", 0)
                    new_score = current + pts
                    if uid in conns_now:
                        conns_now[uid]["score"] = new_score
                    patch_body = {"score": new_score, "elo": elo_snapshot}
                    if elo_col in ("elo_bullet", "elo_blitz", "elo_rapid"):
                        patch_body[elo_col] = elo_snapshot
                    await client.patch(
                        f"{SUPABASE_URL}/rest/v1/tournament_players",
                        params={"tournament_id": f"eq.{tid}", "user_id": f"eq.{uid}"},
                        headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                                 "Content-Type": "application/json"},
                        json=patch_body
                    )

            def streak_bonus(uid, won):
                if uid not in conns_now:
                    return 0
                if won:
                    conns_now[uid]["consecutive_wins"] = conns_now[uid].get("consecutive_wins", 0) + 1
                    streak = conns_now[uid]["consecutive_wins"]
                    return 1 if streak >= 3 else 0
                else:
                    conns_now[uid]["consecutive_wins"] = 0
                    return 0

            if req_result == 'white':
                w_bonus = streak_bonus(g['white_id'], won=True)
                b_bonus = streak_bonus(g['black_id'], won=False)
                await add_score(g['white_id'], 2 + w_bonus, w_elo)
                await add_score(g['black_id'], 0,            b_elo)
            elif req_result == 'black':
                w_bonus = streak_bonus(g['white_id'], won=False)
                b_bonus = streak_bonus(g['black_id'], won=True)
                await add_score(g['white_id'], 0,            w_elo)
                await add_score(g['black_id'], 2 + b_bonus,  b_elo)
            else:
                streak_bonus(g['white_id'], won=False)
                streak_bonus(g['black_id'], won=False)
                await add_score(g['white_id'], 1, w_elo)
                await add_score(g['black_id'], 1, b_elo)

        try:
            if tc_rows and tc_rows[0].get("format") == "arena" and tc_rows[0].get("status") == "active":
                pg    = tournament_player_game.get(tid, {})
                conns = tournament_connections.get(tid, {})
                for uid in (g['white_id'], g['black_id']):
                    if uid in pg:
                        pg[uid] = None
                    if uid in conns:
                        if not conns[uid].get("paused") and conns[uid].get("connected"):
                            conns[uid]["available"] = True
                        won = (req_result == "white" and uid == g['white_id']) or \
                              (req_result == "black" and uid == g['black_id'])
                        streak = conns_now.get(uid, {}).get("consecutive_wins", 0)
                        bonus = 1 if (won and streak >= 3) else 0
                        tc_str = tc_rows[0].get("time_control") if tc_rows else None
                        elo_col = elo_col_for_tc(tc_str)
                        old_elo = w_elo if uid == g['white_id'] else b_elo
                        await arena_send(conns[uid]["ws"], {
                            "type":       "game_over",
                            "result":     req_result,
                            "my_score":   conns[uid]["score"],
                            "streak":     streak if won else 0,
                            "streak_bonus": bonus,
                            "old_elo":    old_elo,
                            "elo_col":    elo_col,
                        })
        except Exception as e:
            print(f"[arena] re-pair error: {e}", flush=True)

        return {"ok": True, "result": req_result}

async def finalize_tournament_scoring(game: dict, result: str):
    """
    Apply tournament standings scoring for a game that just ended, SERVER-SIDE.
    """
    tdb = game.get("tournament_db_id")
    if not tdb or not game.get("tournament_id"):
        return
    try:
        await _apply_tournament_result(tdb, result)
    except HTTPException as he:
        if he.status_code != 400:
            print(f"[arena] server-side scoring error for {tdb}: {he.detail}", flush=True)
    except Exception as e:
        print(f"[arena] server-side scoring exception for {tdb}: {e}", flush=True)

@router.post("/api/create-tournament")
async def create_tournament(request: Request, authorization: str = Header(None)):
    """Create a tournament — server-side to avoid client RLS issues."""
    user_id = await verify_jwt(authorization)
    body = await request.json()

    fmt = body.get("format", "arena")
    required = ["name", "time_control", "starts_at"]
    if fmt != "arena":
        required.append("rounds")
    for field in required:
        if field not in body:
            raise HTTPException(400, f"Missing field: {field}")

    region = body.get("region") or None
    if region and region not in REGIONS:
        raise HTTPException(400, f"Unknown region: {region}")

    row = {
        "name":             body["name"],
        "description":      body.get("description") or None,
        "format":           fmt,
        "time_control":     body["time_control"],
        "rounds":           int(body.get("rounds") or 0),
        "max_players":      int(body.get("max_players") or 9999),
        "country":          body.get("country") or None,
        "region":           region,
        "starts_at":        body["starts_at"],
        "created_by":       user_id,
        "status":           "upcoming",
        "duration_minutes": int(body["duration_minutes"]) if body.get("duration_minutes") else 60,
        "prize_pool":       float(body["prize_pool"]) if body.get("prize_pool") else 0,
    }

    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{SUPABASE_URL}/rest/v1/tournaments",
            headers={
                "apikey":        SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type":  "application/json",
                "Prefer":        "return=representation"
            },
            json=row
        )
    if r.status_code not in (200, 201):
        raise HTTPException(500, f"Create failed: {r.text}")
    created = r.json()
    return {"ok": True, "tournament": created[0] if created else {}}

@router.post("/api/join-tournament")
async def join_tournament(request: Request, authorization: str = Header(None)):
    """Join a tournament — server-side to avoid client RLS issues."""
    user_id = await verify_jwt(authorization)
    body = await request.json()
    tournament_id = body.get("tournament_id")
    if not tournament_id:
        raise HTTPException(400, "Missing tournament_id")

    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/tournaments",
            params={"id": f"eq.{tournament_id}",
                    "select": "country,region,max_players,status,prize_pool,format"},
            headers={"apikey": SUPABASE_SERVICE_KEY,
                     "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
        )
        ts = r.json()
        if not ts:
            raise HTTPException(404, "Tournament not found")
        t = ts[0]

        is_arena = t.get("format") == "arena"
        if t["status"] == "completed":
            raise HTTPException(400, "Tournament has ended")
        if t["status"] == "active" and not is_arena:
            raise HTTPException(400, "This tournament has already started and is not open for late registration")
        if t["status"] not in ("upcoming", "active"):
            raise HTTPException(400, "Tournament is not open for registration")

        has_prizes = bool(t.get("prize_pool") and float(t.get("prize_pool") or 0) > 0)
        if has_prizes:
            await check_prize_eligibility(user_id, client)
        else:
            ban_r = await client.get(
                f"{SUPABASE_URL}/rest/v1/profiles",
                params={"user_id": f"eq.{user_id}", "select": "banned,ban_reason"},
                headers={"apikey": SUPABASE_SERVICE_KEY,
                         "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
            )
            ban_rows = ban_r.json()
            if ban_rows and ban_rows[0].get("banned"):
                reason = ban_rows[0].get("ban_reason") or "Violation of fair play rules."
                raise HTTPException(403, f"Account banned: {reason}")

        profile_r = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"user_id": f"eq.{user_id}",
                    "select": "country,username,elo,elo_bullet,elo_blitz,elo_rapid"},
            headers={"apikey": SUPABASE_SERVICE_KEY,
                     "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
        )
        profile = profile_r.json()
        prof = profile[0] if profile else {}
        player_country = prof.get("country")

        if t.get("country") and player_country != t["country"]:
            raise HTTPException(403,
                f"This tournament is restricted to players from {COUNTRY_NAMES_SERVER.get(t['country'], t['country'])}.")

        if t.get("region"):
            if not player_in_region(player_country, t["region"]):
                label = REGION_LABELS.get(t["region"], t["region"])
                raise HTTPException(403,
                    f"This tournament is restricted to players from {label}. "
                    f"Make sure your country is set correctly in your profile.")

        if not player_country:
            raise HTTPException(403,
                "Please set your country in your profile before joining a tournament. "
                "Go to Profile → select your country → Save.")

        existing = await client.get(
            f"{SUPABASE_URL}/rest/v1/tournament_players",
            params={"tournament_id": f"eq.{tournament_id}",
                    "user_id": f"eq.{user_id}", "select": "id"},
            headers={"apikey": SUPABASE_SERVICE_KEY,
                     "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
        )
        if existing.json():
            raise HTTPException(400, "Already registered")

        if t.get("max_players"):
            count_r = await client.get(
                f"{SUPABASE_URL}/rest/v1/tournament_players",
                params={"tournament_id": f"eq.{tournament_id}", "select": "id"},
                headers={"apikey": SUPABASE_SERVICE_KEY,
                         "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
            )
            if len(count_r.json()) >= t["max_players"]:
                raise HTTPException(400, "Tournament is full")

        r2 = await client.post(
            f"{SUPABASE_URL}/rest/v1/tournament_players",
            headers={
                "apikey":        SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type":  "application/json",
                "Prefer":        "return=minimal"
            },
            json={
                "tournament_id": tournament_id,
                "user_id":       user_id,
                "username":      prof.get("username"),
                "country":       prof.get("country"),
                "elo":           prof.get("elo", 1500),
                "elo_bullet":    prof.get("elo_bullet", 1500),
                "elo_blitz":     prof.get("elo_blitz",  1500),
                "elo_rapid":     prof.get("elo_rapid",  1500),
            }
        )
    if r2.status_code not in (200, 201):
        raise HTTPException(500, f"Join failed: {r2.text}")
    return {"ok": True}

@router.delete("/api/leave-tournament")
async def leave_tournament(request: Request, authorization: str = Header(None)):
    """Leave a tournament."""
    user_id = await verify_jwt(authorization)
    body = await request.json()
    tournament_id = body.get("tournament_id")

    async with httpx.AsyncClient() as client:
        await client.delete(
            f"{SUPABASE_URL}/rest/v1/tournament_players",
            params={"tournament_id": f"eq.{tournament_id}", "user_id": f"eq.{user_id}"},
            headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
        )
    return {"ok": True}

@router.get("/api/tournaments")
async def get_tournaments(status: str = "upcoming"):
    order = "starts_at.asc" if status == "upcoming" else "starts_at.desc"
    async with httpx.AsyncClient() as client:
        t_r, p_r = await asyncio.gather(
            client.get(
                f"{SUPABASE_URL}/rest/v1/tournaments",
                params={"status": f"eq.{status}", "select": "*", "order": order},
                headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
            ),
            client.get(
                f"{SUPABASE_URL}/rest/v1/tournament_players",
                params={"select": "tournament_id"},
                headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
            ),
        )
    tournaments = t_r.json()
    counts: dict[str, int] = {}
    for row in p_r.json():
        tid = str(row.get("tournament_id",""))
        counts[tid] = counts.get(tid, 0) + 1
    for t in tournaments:
        t["player_count"] = counts.get(str(t.get("id","")), 0)
    return tournaments

@router.get("/api/tournaments/{tournament_id}")
async def get_tournament(tournament_id: str):
    """Get single tournament with players and games."""
    async with httpx.AsyncClient() as client:
        t_r, p_r, g_r = await asyncio.gather(
            client.get(f"{SUPABASE_URL}/rest/v1/tournaments",
                params={"id": f"eq.{tournament_id}", "select": "*"},
                headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}),
            client.get(f"{SUPABASE_URL}/rest/v1/tournament_players",
                params={"tournament_id": f"eq.{tournament_id}", "select": "*", "order": "score.desc"},
                headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}),
            client.get(f"{SUPABASE_URL}/rest/v1/tournament_games",
                params={"tournament_id": f"eq.{tournament_id}", "select": "*", "order": "played_at.asc"},
                headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}),
        )
    ts = t_r.json()
    if not ts:
        raise HTTPException(404, "Tournament not found")

    players = p_r.json()
    conns = tournament_connections.get(tournament_id, {})
    for p in players:
        uid = p.get("user_id")
        if uid and uid in conns:
            p["paused"]           = conns[uid].get("paused", False)
            p["consecutive_wins"] = conns[uid].get("consecutive_wins", 0)
        else:
            p["paused"]           = False
            p["consecutive_wins"] = 0

    return {"tournament": ts[0], "players": players, "games": g_r.json()}

@router.post("/api/admin/force-end-game")
async def force_end_game(request: Request, authorization: str = Header(None)):
    """
    Force-end a stuck game. Admin only.
    """
    caller_id = await verify_jwt(authorization)
    if caller_id not in ADMIN_USER_IDS:
        raise HTTPException(403, "Admin access required")

    body    = await request.json()
    game_id = body.get("game_id")
    winner  = body.get("winner", "draw")

    if not game_id:
        games = [
            {"id": gid, "tc": g.get("time_control"), "moves": g.get("moves_made", 0),
             "over": g.get("over"), "white": g.get("white_profile", {}) and g["white_profile"].get("username"),
             "black": g.get("black_profile", {}) and g["black_profile"].get("username")}
            for gid, g in active_games.items()
        ]
        return {"active_games": games}

    game = active_games.get(game_id)
    if not game:
        raise HTTPException(404, f"Game {game_id} not found in active_games")

    game["over"] = True
    result_msg = winner if winner != "draw" else "draw"
    await broadcast(game, {
        "type":   "gameover",
        "result": result_msg,
        "reason": "admin_ended",
        "detail": "This game was ended by an administrator.",
        "clock":  game["clock"],
    })
    if winner != "draw" and not game.get("_elo_updated"):
        game["_elo_updated"] = True
        await update_elos(game, winner)
    await finalize_tournament_scoring(game, result_msg)
    await asyncio.sleep(1)
    active_games.pop(game_id, None)
    print(f"[admin] {caller_id} force-ended game {game_id} → {winner}", flush=True)
    return {"ok": True, "game_id": game_id, "winner": winner}

def player_in_region(player_country: str | None, region: str) -> bool:
    if not player_country:
        return False
    return player_country in REGIONS.get(region, [])
