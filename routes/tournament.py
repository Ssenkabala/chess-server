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
                              RECURRING_WARMUP_PRIZE_POOL, RECURRING_WARMUP_HOUR_EAT)
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
        # Silent state sync — not a visible message, just corrects the
        # client's local _arenaPaused flag so it stops sending "available"
        # automatically after this game. The player sees nothing; they'll
        # simply notice they're not getting paired and can click Resume
        # themselves when they're back, per the no-message-on-forfeit design.
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
                # Auto-START upcoming Arena tournaments
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

                # Auto-END active Arena tournaments whose time is up
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
                            await client.patch(
                                f"{SUPABASE_URL}/rest/v1/tournaments",
                                params={"id": f"eq.{tid}"},
                                headers={"apikey": SUPABASE_SERVICE_KEY,
                                         "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                                         "Content-Type": "application/json"},
                                json={"status": "completed"}
                            )
                            print(f"[arena] auto-ended {tid}", flush=True)
                            # Award podium medals
                            asyncio.create_task(_grant_tournament_medals(tid, client))
                            # Broadcast tournament_ended to all connected players
                            conns = tournament_connections.get(tid, {})
                            ended_msg = {"type": "tournament_ended"}
                            for uid, info in list(conns.items()):
                                await arena_send(info["ws"], ended_msg)
                            # Clean up in-memory state
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
    """The calendar date of the last Friday in the given month, as a naive
    datetime at midnight (date-only — caller sets the actual time)."""
    if month == 12:
        first_of_next = datetime(year + 1, 1, 1)
    else:
        first_of_next = datetime(year, month + 1, 1)
    last_day = first_of_next - timedelta(days=1)
    # weekday(): Monday=0 ... Friday=4 ... Sunday=6
    offset = (last_day.weekday() - 4) % 7
    return last_day - timedelta(days=offset)


def _next_recurring_occurrence_utc() -> datetime:
    """The next upcoming 'last Friday of the month, 7PM EAT' occurrence, as
    a UTC datetime. Checks the current month first; if that date has
    already passed this month, rolls forward to next month automatically."""
    eat = timezone(timedelta(hours=3))  # East Africa Time — no DST
    now_utc = datetime.now(timezone.utc)
    y, m = now_utc.year, now_utc.month
    for _ in range(3):  # a handful of tries is always enough to find a future date
        lf = _last_friday_of_month(y, m)
        occurrence_eat = lf.replace(hour=RECURRING_TOURNAMENT_HOUR_EAT, minute=0, second=0, tzinfo=eat)
        occurrence_utc = occurrence_eat.astimezone(timezone.utc)
        if occurrence_utc > now_utc:
            return occurrence_utc
        m += 1
        if m > 12:
            m = 1
            y += 1
    raise RuntimeError("could not compute next recurring tournament occurrence")  # should be unreachable


async def recurring_tournament_scheduler():
    """Auto-creates the next AfriChess Grand Prix — last Friday of every
    month, 7PM EAT — so this recurring tournament never has to be created
    by hand. Checks once a day; as soon as the currently-scheduled
    occurrence's date has passed, the next month's gets created
    automatically. This naturally keeps roughly a month of advance notice
    live at all times, matching the ~2-week+ lead time that turned out to
    matter for signups.

    Idempotent by design: checks for an existing tournament at the exact
    same name + starts_at before inserting, so it's always safe to run —
    whether this is genuinely the first time creating an occurrence, or a
    restart re-checking a date it already handled. A tournament for this
    same date created or edited by hand (e.g. rescheduling, matching the
    earlier July 10 -> July 17 move) is left alone; this only ever fills
    in a date that doesn't already have one.
    """
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
        await asyncio.sleep(86400)  # once a day is plenty for a monthly schedule


def _is_last_friday_of_month(d: datetime) -> bool:
    """True if d (assumed to already be a Friday) is the LAST Friday of its
    month — i.e. the Grand Prix's slot, which the warmup scheduler must
    never double-book."""
    lf = _last_friday_of_month(d.year, d.month)
    return d.year == lf.year and d.month == lf.month and d.day == lf.day


async def weekly_warmup_scheduler():
    """Auto-creates the AfriChess Continental Warmup — every Friday EXCEPT
    the last one, which is reserved for the Grand Prix
    (recurring_tournament_scheduler above). Looks about 6 weeks ahead and
    fills in any missing warmup occurrence, so there's always a healthy
    runway of upcoming warmups visible, not just the very next one.

    Idempotent by design, same as the Grand Prix scheduler: checks for an
    existing tournament at the exact same name + starts_at before
    inserting, so it's always safe to run repeatedly without creating
    duplicates, and it never touches a date that already has one —
    whether auto-created earlier or scheduled by hand.
    """
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
                candidate += timedelta(days=7)  # today's slot (if today is Friday) already passed

            async with httpx.AsyncClient() as client:
                for week in range(6):
                    friday_eat = candidate + timedelta(weeks=week)
                    if _is_last_friday_of_month(friday_eat):
                        continue  # the Grand Prix's slot, not a warmup
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
    # "connected" matters here, not just available/paused/pg. Right after a
    # forfeit, the winner is marked available immediately (server-side,
    # synchronous) — but their BROWSER is still sitting on the just-ended
    # game's page for a few seconds (the "returning to tournament in Ns…"
    # countdown) before it navigates back and re-opens a tournament socket.
    # Without this check, a pairing tick landing in that window would pick
    # the winner using their OLD, already-closed tournament WebSocket —
    # arena_send() on it fails silently (by design, for a closed socket),
    # so game_ready never reaches them. There IS a resend-on-reconnect
    # fallback for this (see tournament_ws, "resent game_ready ... on
    # reconnect"), so it self-heals once they land back on the page — but
    # not pairing a demonstrably-disconnected socket in the first place
    # closes the race instead of just recovering from it a few seconds
    # late, which is what showed up as "network challenges" in whichever
    # game got silently created while they were still in transit.
    available = [uid for uid, info in conns.items()
                 if info.get("available") and not info.get("paused")
                 and info.get("connected") and pg.get(uid) is None]
    # Full snapshot of every connected player and why they are / aren't eligible.
    _snapshot = {uid: {"avail": i.get("available"), "paused": i.get("paused"),
                       "conn": i.get("connected"), "pg": pg.get(uid)}
                 for uid, i in conns.items()}
    print(f"[DBG pair] {tournament_id}: tick — eligible={available} | all={_snapshot}", flush=True)
    if len(available) < 1:
        return

    # ── Pairing cutoff: stop pairing when remaining time ≤ 0.5 × time_control ──
    # This mirrors Lichess arena behaviour — no new games can start if they
    # couldn't finish before the tournament ends.
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
                return  # tournament not active, don't pair
            _starts    = datetime.fromisoformat(_t["starts_at"].replace("Z", "+00:00"))
            _ends      = _starts + timedelta(minutes=_t.get("duration_minutes") or 60)
            _remaining = (_ends - datetime.now(timezone.utc)).total_seconds()
            # Parse time_control "base+inc" → total seconds (use base only for cutoff)
            _tc_str    = _t.get("time_control") or "5+0"
            try:
                _tc_secs = float(_tc_str.split("+")[0]) * 60
            except (ValueError, IndexError):
                _tc_secs = 300
            _cutoff = _tc_secs * 0.5   # half the time control
            if _remaining <= _cutoff:
                # Broadcast pairings_closed to all available players
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
        # On error, allow pairing to proceed rather than blocking games

    # Fetch all games played so far to build a played-count map
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{SUPABASE_URL}/rest/v1/tournament_games",
                params={"tournament_id": f"eq.{tournament_id}",
                        "select": "white_id,black_id"},
                headers={"apikey": SUPABASE_SERVICE_KEY,
                         "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
            )
        # Count how many times each pair has played (for rematch avoidance preference)
        # and each player's color balance (white_games - black_games) for color fairness.
        # Without this, pairing has no concept of color at all — whichever player
        # happens to sort earlier (by score/ELO) always becomes white against a
        # given opponent, every single time they're paired. With only a couple of
        # players cycling against each other repeatedly (a small arena), this
        # produced one player playing black 5 games in a row against the same
        # opponent — a real, meaningful disadvantage, not a cosmetic issue.
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

    # Sort available players: whoever has waited longest goes first,
    # overriding score/ELO ordering — this is what actually fixes
    # starvation. The OLD sort was purely (score desc, elo desc) with no
    # concept of wait time at all, so a player who happened to sort low
    # (by chance among freshly-created accounts with identical starting
    # ELO, or simply from losing early games) could be left as the odd
    # one out round after round indefinitely, since nothing in the
    # algorithm ever corrected for it. Confirmed real via a 33-player load
    # test: median time-to-first-pairing was 5.8s, but the max was 202.9s —
    # one player waited the better part of the whole test while everyone
    # else paired normally. Now rounds_waited is the PRIMARY sort key, so
    # the longer someone sits, the more their turn gets prioritized,
    # regardless of where they'd otherwise rank.
    available.sort(key=lambda uid: (
        -conns[uid].get("rounds_waited", 0),
        -conns[uid].get("score", 0),
        -conns[uid].get("elo", 1500),
    ))

    def times_played(p1, p2):
        return pair_count.get(tuple(sorted([p1, p2])), 0)

    def assign_colors(p1, p2):
        """Whoever has played MORE white games (higher balance) gets black this
        time; whoever has played more black (lower balance) gets white. A tie
        falls back to p1=white, p2=black (arbitrary but stable) — this only
        matters on a player's very first pairing of the tournament, where
        fairness can't be evaluated yet anyway."""
        b1 = color_balance.get(p1, 0)
        b2 = color_balance.get(p2, 0)
        if b1 > b2:
            return p2, p1   # p2 has played black more / white less -> p2 gets white
        elif b2 > b1:
            return p1, p2
        else:
            return p1, p2

    # Pair greedily: prefer opponents not yet played, then fewest rematches, then score proximity
    paired, used = [], set()
    for i, p1 in enumerate(available):
        if p1 in used:
            continue
        best_p2   = None
        best_score = None  # (times_played, score_diff) — lower is better
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

    # Update rounds_waited for everyone considered this tick — reset to 0
    # for anyone who got paired (their wait is over), increment for anyone
    # left unpaired. Without this update, the new wait-time sort key above
    # would never actually accumulate and the starvation fix would do
    # nothing — the counter only has teeth if it's actually maintained
    # every single tick, not just read.
    for uid in available:
        if uid in used:
            conns[uid]["rounds_waited"] = 0
        else:
            conns[uid]["rounds_waited"] = conns[uid].get("rounds_waited", 0) + 1

    # Odd player — just leave them waiting, no bye points
    # Arena players should wait until a free opponent becomes available
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
        # Mark both players as committed to this pairing IMMEDIATELY, inside
        # the lock — not later inside arena_launch_game (which only runs as
        # a separately-scheduled task and could be delayed). Without this,
        # a second concurrent call to arena_pair() could acquire the lock
        # after this one releases it but BEFORE the first arena_launch_game
        # task actually executes, find both players still showing pg=None,
        # and pair the exact same two people again — producing two separate
        # games for one pair simultaneously. This was confirmed in testing:
        # the same two players got launched into two back-to-back games,
        # both timing out together.
        pg[white_id] = "pending"
        pg[black_id] = "pending"
        asyncio.create_task(arena_launch_game(tournament_id, white_id, black_id))

async def arena_pair(tournament_id: str):
    """
    Serializes concurrent pairing attempts so a forfeit handler and one or
    two /api/tournament/result calls (the winner's client and sometimes the
    loser's, both reacting to the same game-over moment) can't race each
    other and each see a half-updated conns dict. Without this, multiple
    near-simultaneous calls could each independently decide a player has
    "no available opponent" a few milliseconds before the other call
    actually marks that opponent available — producing spurious
    "waiting (odd player)" results and wasted pairing passes even though a
    real opponent existed the whole time.

    As of the single-poller rewrite, this function has exactly one caller:
    arena_pairing_loop below. Nothing else should call this directly —
    forfeits, result submissions, and reconnects only ever mutate state
    (available/paused/pg) and let the loop's next tick handle pairing.
    """
    lock = _arena_pair_locks.setdefault(tournament_id, asyncio.Lock())
    async with lock:
        await _arena_pair_impl(tournament_id)

async def arena_pairing_loop(tournament_id: str):
    """
    The single background poller for one active arena tournament. Runs every
    5 seconds for as long as the tournament is active, and is the ONLY
    caller of arena_pair(). Every other event in the system (a forfeit, a
    player resuming, a game ending) only ever updates state — available,
    paused, pg — and waits for this loop's next tick to actually act on it.

    This exists specifically to eliminate an entire class of race condition
    that came up repeatedly: several different triggers (forfeit handler,
    both players' independent result submissions, reconnects) could each
    call pairing logic concurrently, racing each other against a
    half-updated in-memory dict. With exactly one caller on a fixed
    schedule, there's structurally nothing left to race against — every
    pairing decision sees fully-settled state from whatever ran on the
    previous tick.
    """
    if tournament_id in _active_pairing_loops:
        print(f"[DBG loop] {tournament_id}: pairing loop already running — not starting a second", flush=True)
        return  # already running for this tournament — don't start a second one
    _active_pairing_loops.add(tournament_id)
    print(f"[DBG loop] {tournament_id}: PAIRING LOOP STARTED", flush=True)
    try:
        while True:
            await asyncio.sleep(5)  # per design spec: check the lobby every 5 seconds
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
                continue  # transient DB error — try again next tick rather than stopping

            try:
                await arena_pair(tournament_id)
            except Exception as e:
                print(f"[arena] pairing loop tick error for {tournament_id}: {e}", flush=True)
                # Don't break on a single bad tick — log and keep polling.
    finally:
        _active_pairing_loops.discard(tournament_id)
        print(f"[arena] pairing loop ended for {tournament_id}", flush=True)

async def arena_launch_game(tournament_id: str, white_id: str, black_id: str):
    conns = tournament_connections.get(tournament_id, {})
    pg    = tournament_player_game.setdefault(tournament_id, {})
    if white_id not in conns or black_id not in conns:
        return
    # Defensive guard: if either player is already committed to a REAL game
    # (a genuine game_id, not the "pending" marker this exact pairing should
    # have set), refuse to proceed. This protects against a player ending up
    # in two simultaneous games regardless of how the duplicate pairing
    # decision occurred upstream — the actual irreversible damage only
    # happens here, at the point a second tournament_games row gets created,
    # so this is the one place that needs to be unconditionally safe.
    existing_white = pg.get(white_id)
    existing_black = pg.get(black_id)
    if existing_white not in (None, "pending") or existing_black not in (None, "pending"):
        print(f"[arena] REFUSING duplicate launch for {tournament_id}: "
              f"white={white_id} (pg={existing_white}), black={black_id} (pg={existing_black})", flush=True)
        # Release whichever player ISN'T the one with a genuine conflicting
        # game — they were marked pending/unavailable by the pairing
        # decision that led to this call, and without this they'd be stuck
        # forever with no path back into pairing, even though they did
        # nothing wrong and have no real game of their own right now.
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

    # Fetch tournament time_control for this game
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
    # Clock starts the instant pairing happens — do NOT wait for both
    # players' WebSockets to connect. This is exactly like a real chess
    # clock: it's running the moment the game starts, whether or not either
    # player has shown up yet. A genuine no-show forfeits when 20% of their
    # base time has elapsed, same as anyone who connects but never moves.
    game["last_move_ts"]        = time.time()
    game["first_move_deadline"] = time.time() + game.get("first_move_timeout", 60)
    active_games[game_id] = game
    # Deferred import — breaks a circular dependency: game.py needs
    # finalize_tournament_scoring/tournament_handle_forfeit from THIS file,
    # so this file can't also import FROM game.py at the top level without
    # creating an import cycle. This is the standard, safe fix: import
    # locally, right where it's used, instead of at module load time —
    # by the time this function actually RUNS (a request/task, not import
    # time), both modules are already fully loaded, so this always resolves.
    from routes.game import clock_loop, first_move_timeout_loop
    asyncio.create_task(clock_loop(game_id))
    asyncio.create_task(first_move_timeout_loop(game_id))
    print(f"[arena] {game_id}: {white_info['username']} vs {black_info['username']}", flush=True)

    # Use the TC-specific ELO for display in game_ready
    elo_col = elo_col_for_tc(time_control)
    w_display_elo = white_info.get(elo_col) or white_info.get("elo", 1500)
    b_display_elo = black_info.get(elo_col) or black_info.get("elo", 1500)

    # Compute live standings rank for display in-game
    conns = tournament_connections.get(tournament_id, {})
    ranked = sorted(conns.values(), key=lambda x: (-x.get("score", 0), -x.get("elo", 1500)))
    uid_to_rank = {list(conns.keys())[i]: i+1
                   for i, uid in enumerate(c.get("user_id", "") for c in ranked)
                   if True}
    # Simpler: rank by index in sorted list
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

    # ── Registration gate ────────────────────────────────────────────────────
    # Opening this WebSocket must NOT be sufficient to enter pairing. A player
    # must have actually registered via /api/join-tournament (which enforces
    # country/region eligibility, ban checks, capacity, and prize eligibility).
    # Without this check, anyone who knows a tournament_id can connect directly
    # and get paired into real games with real ELO/prize consequences, bypassing
    # every eligibility check above.
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

    # Preserve persistent state across reconnects. The tournament WebSocket
    # gets torn down and recreated on every full page navigation — and the
    # client navigates away to /multiplayer for each game, then back to
    # /tournaments after it ends. Previously this dict was rebuilt from
    # scratch on every single reconnect, silently wiping out any "paused"
    # state that a forfeit had just set moments earlier. That meant a
    # player who forfeited (no-show) got paused, immediately reconnected
    # via the post-game redirect, and was treated as fresh/available again
    # — letting them get paired and forfeit repeatedly with zero way to
    # actually stay excluded from pairing until they explicitly resumed.
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
        # Tell the reconnecting client it's still paused, so its UI (Resume
        # button state) matches reality instead of assuming a fresh connect
        # means available again.
        await arena_send(ws, {"type": "paused_state_sync", "paused": True})

    # If this player already has a real (non-pending) game waiting, resend
    # game_ready now. The original send happens once, at the moment the game
    # is created, directly to whatever WebSocket each player has open right
    # then — if that's not their tournament-page socket (e.g. they're still
    # sitting on the previous game's endgame popup, which has no tournament
    # connection open at all), the message is silently lost with no resend.
    # This was the actual cause of a player getting a new opponent with no
    # visible transition: the game existed, they just never found out.
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

    # If tournament already active, try pairing immediately
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
            pass  # state is set; arena_pairing_loop's next tick (≤5s) picks it up
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
                # No direct arena_pair() call — the poller's next tick handles it.

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
                # No direct pairing call — the poller picks this up on its next tick.
    except WebSocketDisconnect:
        pass
    except Exception as e:
        # Anything other than a clean disconnect was previously invisible —
        # it propagated past this function as an unhandled exception at the
        # ASGI level, with the finally block below still running cleanup
        # correctly but no detail ever logged about what actually failed.
        # Logging it here doesn't change behavior, but means a future
        # occurrence is actually diagnosable instead of showing up as a
        # generic Uvicorn traceback with no application-level context.
        print(f"[arena] tournament_ws error for {user_id} on {tournament_id}: "
              f"{type(e).__name__}: {e}", flush=True)
    finally:
        conns = tournament_connections.get(tournament_id, {})
        if user_id and user_id in conns:
            conns[user_id]["available"] = False
            conns[user_id]["connected"] = False
            # Do NOT delete the entry — a reconnect (refresh, navigating to
            # Watch and back, the redirect after a game ends) needs to find
            # this entry still here so it can read paused/consecutive_wins
            # and carry them forward. Deleting it here meant every single
            # disconnect silently erased a player's pause state moments
            # before the reconnect logic ran, even though that logic was
            # specifically designed to preserve it. This was the actual
            # cause of paused players coming back as available after any
            # refresh, and of forfeited players getting re-paired
            # immediately on their post-game redirect.
        print(f"[arena] {username} left {tournament_id}", flush=True)

def swiss_pair(players: list, existing_games: list) -> list:
    """
    Simple Swiss pairing:
    - Sort by score desc, then ELO desc
    - Pair adjacent players, avoiding repeat pairings
    - Unpaired player gets a bye (1 point)
    Returns list of (white, black) tuples where each is a player dict.
    """
    # Build set of already-played pairs
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
                # Alternate colours: higher score gets white
                paired.append((p1, p2))
                used.add(p1['user_id'])
                used.add(p2['user_id'])
                break

    # Handle bye (odd player out)
    for p in ranked:
        if p['user_id'] not in used:
            paired.append((p, None))  # None = bye

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

            # Mark active first
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/tournaments",
                params={"id": f"eq.{req.tournament_id}"},
                headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                         "Content-Type": "application/json"},
                json={"status": "active"}
            )

            # Tell every player already connected to this tournament's lobby
            # that it just started — without this, a player who was sitting
            # in the waiting room before start had no way to know anything
            # changed until they manually refreshed the page. arena_pair()
            # below only notifies players it successfully pairs in this pass;
            # anyone left unpaired (odd count, still mid-connect, etc.) would
            # otherwise see no update at all.
            conns = tournament_connections.get(req.tournament_id, {})
            for uid, info in list(conns.items()):
                ws = info.get("ws")
                if ws:
                    try:
                        await arena_send(ws, {"type": "tournament_started"})
                    except Exception:
                        pass

            # Arena: pairing via the single background poller (started here,
            # runs every 5s for the lifetime of the active tournament)
            if t.get("format") == "arena":
                asyncio.create_task(arena_pairing_loop(req.tournament_id))
                return {"ok": True, "format": "arena", "players": len(players)}

            # Swiss: generate round 1
            pairs = swiss_pair(players, [])
            games_to_insert = []
            for white, black in pairs:
                if black is None:
                    await client.patch(
                        f"{SUPABASE_URL}/rest/v1/tournament_players",
                        params={"tournament_id": f"eq.{req.tournament_id}", "user_id": f"eq.{white['user_id']}"},
                        headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                                 "Content-Type": "application/json"},
                        json={"score": white.get("score", 0) + 2}  # bye = full win value
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
                        json={"score": white.get("score", 0) + 2}  # bye = full win value
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
    Thin authentication wrapper around _apply_tournament_result(), which holds
    all the actual scoring logic so the SERVER can also apply results directly
    at game-end (see the game-over path in the game WebSocket) without relying
    on the client to call back — the client callback is fragile (needs the
    tournament_db_id in its URL, a live session, and the page not to navigate
    away first), and if it doesn't fire, standings never update. Both callers
    are safe to run for the same game: the per-game lock plus the
    "result already submitted" guard inside make it idempotent.
    """
    if not SUPABASE_SERVICE_KEY:
        raise HTTPException(503, "Service unavailable")
    user_id = await verify_jwt(authorization)
    if req.result not in ('white', 'black', 'draw'):
        raise HTTPException(400, "Invalid result")
    return await _apply_tournament_result(req.game_id, req.result, submitting_user_id=user_id)

async def _apply_tournament_result(game_db_id: str, result: str, submitting_user_id: str = None):
    """
    Core tournament-result scoring. Called by the client endpoint (with an
    authenticated submitting_user_id) AND directly by the server at game-end
    (submitting_user_id=None, already authorized). Idempotent per game.

    - Marks the game result in tournament_games
    - Updates tournament_players.score (win=2, draw=1, loss=0, +1 streak bonus on 3rd+ consecutive win)
    - Syncs tournament_players.elo snapshot from profiles (ELO already updated by update_elos via WS)
    - Does NOT recalculate ELO — that is handled by update_elos() when the game ends over WebSocket
    """
    if result not in ('white', 'black', 'draw'):
        raise HTTPException(400, "Invalid result")

    # Both players' clients AND the server can independently trigger scoring
    # for the same game. Without this lock, two near-simultaneous calls for the
    # same game_id could each read tournament_games.result as still null
    # (neither has committed yet) and both proceed to score the game — doubling
    # every point awarded. Confirmed live: every win in a tournament was
    # recorded at exactly double its correct value.
    lock = _submit_result_locks.setdefault(game_db_id, asyncio.Lock())
    async with lock:
        async with httpx.AsyncClient() as client:
            # Get the game
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

            # Only white or black player can submit — but ONLY enforced for
            # client calls. Server-side calls (submitting_user_id=None) are
            # already trusted: the server owns the game and knows the result.
            if submitting_user_id is not None and submitting_user_id not in (g['white_id'], g['black_id']):
                raise HTTPException(403, "Not a player in this game")

            req_game_id = game_db_id  # local alias so the body below reads unchanged
            req_result  = result

            tid = g['tournament_id']

            # Fetch tournament time_control to determine ELO column
            tc_r = await client.get(
                f"{SUPABASE_URL}/rest/v1/tournaments",
                params={"id": f"eq.{tid}", "select": "format,status,time_control"},
                headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
            )
            tc_rows = tc_r.json()
            time_control = tc_rows[0].get("time_control") if tc_rows else None
            elo_col = elo_col_for_tc(time_control)

            tournament_status = tc_rows[0].get("status") if tc_rows else None

            # Mark game result — this happens regardless of tournament status.
            # It's what makes the tournament page stop showing this game as
            # "ongoing" (the Watch button / live-games list both key off
            # whether tournament_games.result is set), so it must always run
            # even for a game that outlived its tournament.
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/tournament_games",
                params={"id": f"eq.{req_game_id}"},
                headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                         "Content-Type": "application/json"},
                json={"result": req_result, "played_at": datetime.utcnow().isoformat()}
            )

            if tournament_status != "active":
                # The tournament ended while this game was still being played
                # (its clock ran out, but the game itself plays to a real
                # conclusion rather than being force-resolved). The game's own
                # result is recorded above so it correctly stops showing as
                # ongoing, but it no longer counts toward standings — the
                # tournament is already over. ELO is unaffected: that's
                # updated separately by update_elos() over the game's own
                # WebSocket the moment it actually ends, independent of this
                # endpoint and independent of tournament status.
                print(f"[arena] game {req_game_id} finished after tournament {tid} ended — "
                      f"result recorded, standings NOT updated", flush=True)
                return {"ok": True, "result": req_result, "counted_toward_standings": False}

            # Fetch updated ELOs from the correct column (already written by update_elos over WS)
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

            # Streak bonus: +1 extra point for every win once a streak reaches 3+
            # consecutive wins (a loss resets it to 0). Tracked in-memory in
            # tournament_connections so it survives across games without a DB
            # round trip.
            conns_now = tournament_connections.get(tid, {})

            # Update tournament standings: score + ELO snapshot.
            # IMPORTANT: uses the in-memory conns_now[uid]["score"] as the
            # authoritative running total, NOT a fresh read from the database.
            # The old version did read-current-then-write-current+pts as two
            # separate HTTP round trips — if two games for the same player ended
            # close together (entirely possible in a fast arena, and especially
            # likely while the duplicate-pairing bug existed earlier this
            # session), the second call's read could land before the first
            # call's write had committed, silently dropping points. Mutating the
            # in-memory dict has no such gap — it's a synchronous Python
            # operation with no await between read and write, so it can't be
            # interleaved by another coroutine on the same event loop.
            async def add_score(uid, pts, elo_snapshot):
                lock = _player_score_locks.setdefault((tid, uid), asyncio.Lock())
                async with lock:
                    if uid not in conns_now:
                        # Player isn't connected right now (shouldn't normally happen
                        # mid-game, but fall back to a DB read so we don't silently
                        # lose points if it does)
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
                    # Also update the per-TC column (elo_bullet/elo_blitz/elo_rapid)
                    # that matches this tournament's actual time control, in
                    # addition to the generic `elo` field. tournament_players'
                    # per-TC columns were only ever written once, at registration
                    # — never touched again as the player's real rating changed
                    # mid-tournament. The lobby standings table reads p[eloCol]
                    # (e.g. p.elo_blitz) specifically, so it kept showing each
                    # player's join-time snapshot indefinitely, while the generic
                    # `elo` field (and the profile page, which reads `profiles`
                    # directly) updated correctly the whole time.
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
                """Update streak counter and return bonus points.
                Bonus applies to EVERY win once the streak reaches 3+ consecutive
                wins, continuing until the player loses (which resets it to 0).
                The 1st and 2nd wins in a streak are plain, no bonus yet."""
                if uid not in conns_now:
                    return 0
                if won:
                    conns_now[uid]["consecutive_wins"] = conns_now[uid].get("consecutive_wins", 0) + 1
                    streak = conns_now[uid]["consecutive_wins"]
                    return 1 if streak >= 3 else 0   # bonus on every win once streak hits 3+
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
                streak_bonus(g['white_id'], won=False)   # draw resets streak
                streak_bonus(g['black_id'], won=False)
                await add_score(g['white_id'], 1, w_elo)
                await add_score(g['black_id'], 1, b_elo)

        # Arena: release players and re-pair
        try:
            if tc_rows and tc_rows[0].get("format") == "arena" and tc_rows[0].get("status") == "active":
                pg    = tournament_player_game.get(tid, {})
                conns = tournament_connections.get(tid, {})
                for uid in (g['white_id'], g['black_id']):
                    if uid in pg:
                        pg[uid] = None
                    if uid in conns:
                        # IMPORTANT: do not blindly set available=True here. If this
                        # player was just paused by a forfeit (tournament_handle_forfeit
                        # sets paused=True synchronously before this code can run),
                        # the OTHER player's client independently calling this same
                        # endpoint after receiving the gameover broadcast must not
                        # undo that pause. This was the actual cause of forfeited
                        # players keep getting re-paired indefinitely — the winner's
                        # own result-submission was unconditionally re-enabling the
                        # loser's pairing eligibility moments after it was correctly
                        # disabled.
                        if not conns[uid].get("paused") and conns[uid].get("connected"):
                            conns[uid]["available"] = True
                        won = (req_result == "white" and uid == g['white_id']) or \
                              (req_result == "black" and uid == g['black_id'])
                        # NOTE: do NOT recompute pts/streak and add to conns[uid]["score"]
                        # here — add_score() above already updated this exact value
                        # (conns_now and conns are the same dict, same tournament_id).
                        # Doing it again here was a genuine double-count bug: every
                        # game silently added its points TWICE to the in-memory score,
                        # which is what the lobby UI displays. This is very likely why
                        # an 8-game win streak showed fewer points than the formula
                        # actually produces — some of the doubled increments were
                        # then lost to the OTHER race (the old read-then-write DB
                        # pattern in add_score, now also fixed), so the visible total
                        # ended up an inconsistent mix of double-counted and dropped
                        # points rather than cleanly wrong in one direction.
                        streak = conns_now.get(uid, {}).get("consecutive_wins", 0)
                        bonus = 1 if (won and streak >= 3) else 0
                        # Include ELO change so tournament.html can display it
                        # (the elo_update WS msg goes to the game socket which closes on redirect)
                        tc_str = tc_rows[0].get("time_control") if tc_rows else None
                        elo_col = elo_col_for_tc(tc_str)
                        # Use the real ELO fetched fresh from Supabase earlier in this
                        # function (w_elo/b_elo), not the in-memory conns[uid] value —
                        # that was only ever a one-time snapshot taken when this player
                        # first connected to the tournament WebSocket, and never gets
                        # updated as their actual rating changes from playing games.
                        # Showing it in the lobby produced exactly the "ELO looks wrong"
                        # symptom, since it could be many games stale.
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
                # No direct pairing call — players were released above
                # (available=True, pg cleared); the single poller's next
                # tick (≤5s) handles the actual re-pairing.
        except Exception as e:
            print(f"[arena] re-pair error: {e}", flush=True)

        return {"ok": True, "result": req_result}

async def finalize_tournament_scoring(game: dict, result: str):
    """
    Apply tournament standings scoring for a game that just ended, SERVER-SIDE.
    No-op for casual (non-tournament) games. Idempotent: safe to call alongside
    the client's /api/tournament/result POST — the per-game lock and the
    "already submitted" guard inside _apply_tournament_result make whichever
    call arrives second a harmless 400. Call this at EVERY game-end path
    (checkmate/stalemate, resignation, timeout/flag, draw agreement) right
    after update_elos(), so standings never depend on the client calling back.
    """
    tdb = game.get("tournament_db_id")
    if not tdb or not game.get("tournament_id"):
        return
    try:
        await _apply_tournament_result(tdb, result)
    except HTTPException as he:
        if he.status_code != 400:   # 400 = already submitted (client beat us); expected
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

        # Arena tournaments allow late joining while active — same as Lichess.
        # You enter with 0 points and play whatever rounds remain.
        # Swiss/round-robin formats stay closed once started, since their
        # pairing system assumes a fixed player list from round 1.
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

        # Fetch profile — needed for all restriction checks
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

        # Country restriction (most specific)
        if t.get("country") and player_country != t["country"]:
            raise HTTPException(403,
                f"This tournament is restricted to players from {COUNTRY_NAMES_SERVER.get(t['country'], t['country'])}.")

        # Region restriction
        if t.get("region"):
            if not player_in_region(player_country, t["region"]):
                label = REGION_LABELS.get(t["region"], t["region"])
                raise HTTPException(403,
                    f"This tournament is restricted to players from {label}. "
                    f"Make sure your country is set correctly in your profile.")

        # Country must be set for ALL tournaments — needed for regional eligibility
        # and leaderboards even on open tournaments
        if not player_country:
            raise HTTPException(403,
                "Please set your country in your profile before joining a tournament. "
                "Go to Profile → select your country → Save.")

        # Check not already joined
        existing = await client.get(
            f"{SUPABASE_URL}/rest/v1/tournament_players",
            params={"tournament_id": f"eq.{tournament_id}",
                    "user_id": f"eq.{user_id}", "select": "id"},
            headers={"apikey": SUPABASE_SERVICE_KEY,
                     "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
        )
        if existing.json():
            raise HTTPException(400, "Already registered")

        # Check not full
        if t.get("max_players"):
            count_r = await client.get(
                f"{SUPABASE_URL}/rest/v1/tournament_players",
                params={"tournament_id": f"eq.{tournament_id}", "select": "id"},
                headers={"apikey": SUPABASE_SERVICE_KEY,
                         "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
            )
            if len(count_r.json()) >= t["max_players"]:
                raise HTTPException(400, "Tournament is full")

        # Insert player
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
    # Count players per tournament
    counts: dict[str, int] = {}
    for row in p_r.json():
        tid = str(row.get("tournament_id",""))
        counts[tid] = counts.get(tid, 0) + 1
    for t in tournaments:
        t["player_count"] = counts.get(str(t.get("id","")), 0)
    return tournaments

@router.get("/api/tournaments/{tournament_id}")
async def get_tournament(tournament_id: str):
    """Get single tournament with players and games.
    Merges live in-memory paused/streak state from tournament_connections."""
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
    # Merge live paused state and streak from in-memory connections
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
    Body: { "game_id": "...", "winner": "white"|"black"|"draw" }
    """
    caller_id = await verify_jwt(authorization)
    if caller_id not in ADMIN_USER_IDS:
        raise HTTPException(403, "Admin access required")

    body    = await request.json()
    game_id = body.get("game_id")
    winner  = body.get("winner", "draw")  # "white", "black", or "draw"

    if not game_id:
        # List all active games for the admin
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
