# server.py
import chess
import chess.engine
import anthropic
import os
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import sqlite3
import secrets
from datetime import datetime, timedelta, timezone
import asyncio
import uuid
import time
import hmac
import hashlib
from fastapi import WebSocket, WebSocketDisconnect
import asyncio, uuid, time

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    asyncio.create_task(arena_auto_start_scheduler())
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ENGINE_PATH = "./engines/engine.exe" if os.name == "nt" else "./engines/engine"
ANTHROPIC_API_KEY    = os.getenv("ANTHROPIC_API_KEY", "your-key-here")
SUPABASE_URL         = os.getenv("SUPABASE_URL", "https://nbskgzsvygdmlvwbetxn.supabase.co")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")  # set on Railway

# Tournament race condition guard
_tournament_locks: set = set()

# Lemon Squeezy
LS_SIGNING_SECRET  = os.getenv("LEMONSQUEEZY_SIGNING_SECRET", "")
LS_CLUB_VARIANT    = int(os.getenv("LS_CLUB_VARIANT_ID", "1667817"))
LS_PRO_VARIANT     = int(os.getenv("LS_PRO_VARIANT_ID",  "1667860"))

# Coach limits per plan
COACH_LIMITS = {"free": 10, "club": 200, "pro": 999999}


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

# ΓöÇΓöÇΓöÇ Supabase admin client (service role ΓÇö bypasses RLS) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
import httpx

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

async def supabase_update_elo(user_id: str, new_elo: int, time_control: str | None = None):
    """Update the correct ELO column and increment games_played."""
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
        await client.patch(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"user_id": f"eq.{user_id}"},
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            json={col: new_elo, "games_played": current_gp + 1}
        )

# ΓöÇΓöÇΓöÇ Database setup ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

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

init_db()

TIER_LIMITS = {
    "free": 10,
    "club": 200,
    "pro": 999999
}

# ΓöÇΓöÇΓöÇ Models ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

# Times in ms sent as wtime/btime with movestogo=1
# Engine adds 200ms buffer to movetime, so we use wtime directly
DIFFICULTY_SETTINGS = {
    1: 100,    # engine gets 100ms ΓÇö genuinely weak
    2: 300,
    3: 800,
    4: 2000,
    5: 5000,
}

class MoveRequest(BaseModel):
    fen: str
    think_time: float = 1.0
    difficulty: int = 3  # 1=Beginner → 5=Expert
    moves: list[str] = []  # full UCI move history for repetition detection

class CoachRequest(BaseModel):
    fen: str
    played_move: Optional[str] = None   # UCI format e.g. "e2e4"
    pgn: Optional[str] = None
    lesson_type: Optional[str] = None   # "opening", "middlegame", "endgame"
    think_time: float = 1.0

class RegisterRequest(BaseModel):
    email: str
    tier: str = "free"  # set to "club"/"pro" after Stripe confirms payment

# ΓöÇΓöÇΓöÇ Auth helper ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

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

# ΓöÇΓöÇΓöÇ Engine helper ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def analyse_position(fen: str, think_time: float, moves: list[str] | None = None):
    """
    Talk directly to SenkabalaIII via raw subprocess.
    SenkabalaIII non-standard output: bestmove → stdout, info → stderr.
    Sends full move history so engine posHistory[] tracks repetitions.
    """
    import subprocess

    board = chess.Board()
    if moves:
        for uci in moves:
            try:
                board.push_uci(uci)
            except Exception:
                board = chess.Board(fen)
                break
    else:
        board = chess.Board(fen)

    think_ms = int(max(think_time, 1.0) * 1000)
    pos_cmd = f"position startpos moves {' '.join(moves)}" if moves else f"position fen {fen}"
    commands = f"uci\nucinewgame\n{pos_cmd}\ngo movetime {think_ms}\n"

    try:
        proc = subprocess.Popen(
            [ENGINE_PATH],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1
        )
        stdout_data, stderr_data = proc.communicate(input=commands, timeout=think_ms/1000 + 8)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout_data, stderr_data = proc.communicate()

    best_move = None
    score = 0
    pv_moves = []
    best_depth = -1

    # bestmove on stdout
    for line in stdout_data.splitlines():
        if line.startswith("bestmove"):
            parts = line.split()
            if len(parts) >= 2 and parts[1] not in ("(none)", "0000"):
                best_move = parts[1]
            break

    # info lines on stderr
    for line in stderr_data.splitlines():
        if "depth" not in line:
            continue
        try:
            parts = line.split()
            depth = int(parts[parts.index("depth") + 1])
            if "score" not in parts or depth <= best_depth:
                continue
            si = parts.index("score")
            stype, sval = parts[si+1], int(parts[si+2])
            best_depth = depth
            score = sval if stype == "cp" else (10000 - abs(sval)) * 100 * (1 if sval > 0 else -1)
            if "pv" in parts:
                pvi = parts.index("pv")
                pv_moves = parts[pvi+1:pvi+6]
        except (ValueError, IndexError):
            continue

    if board.turn == chess.BLACK:
        score = -score
    if not best_move and pv_moves:
        best_move = pv_moves[0]

    return {"best_move": best_move, "score_cp": score, "pv": pv_moves}


# ΓöÇΓöÇΓöÇ Original /move endpoint (unchanged) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
engine_semaphore = asyncio.Semaphore(3)

@app.post("/move")
async def get_move(req: MoveRequest):
    async with engine_semaphore:
        try:
            # Validate FEN
            board = chess.Board(req.fen)
            if board.is_game_over():
                return {
                    "move": None, "fen": req.fen,
                    "is_game_over": True,
                    "outcome": str(board.outcome()),
                    "score_cp": 0, "eval_pawns": 0, "candidates": []
                }

            import random
            think_ms = DIFFICULTY_SETTINGS.get(req.difficulty, int(req.think_time * 1000))

            # Reconstruct game board from full move history (needed for repetition detection)
            game_board = chess.Board()
            if req.moves:
                for uci in req.moves:
                    try:
                        game_board.push_uci(uci)
                    except Exception:
                        game_board = chess.Board(req.fen)
                        break
            else:
                game_board = chess.Board(req.fen)

            # Low difficulties: random legal move
            random_chance = {1: 0.75, 2: 0.40, 3: 0.15, 4: 0.0, 5: 0.0}
            if random.random() < random_chance.get(req.difficulty, 0):
                move = random.choice(list(game_board.legal_moves))
                move_uci = move.uci()
            else:
                # Use raw subprocess so we can send the full position command.
                # python-chess engine.play() only sends `position fen <fen>` internally,
                # which means the engine's posHistory[] never gets populated — it can't
                # detect repetitions. We must send `position startpos moves <all>` ourselves.
                import subprocess
                if req.moves:
                    pos_cmd = f"position startpos moves {' '.join(req.moves)}"
                else:
                    pos_cmd = f"position fen {req.fen}"

                commands = f"uci\nucinewgame\n{pos_cmd}\ngo movetime {think_ms}\n"
                move_uci = None
                try:
                    proc = subprocess.Popen(
                        [ENGINE_PATH],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1
                    )
                    stdout_data, _ = proc.communicate(
                        input=commands,
                        timeout=think_ms / 1000 + 8
                    )
                    for line in stdout_data.splitlines():
                        if line.startswith("bestmove"):
                            parts = line.split()
                            if len(parts) >= 2 and parts[1] not in ("(none)", "0000"):
                                move_uci = parts[1]
                            break
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()

                if not move_uci:
                    # Fallback: any legal move
                    move_uci = random.choice(list(game_board.legal_moves)).uci()

                # Convert UCI string to chess.Move for pushing
                move = chess.Move.from_uci(move_uci)

            # Candidate moves — analyse current position
            fen_board = chess.Board(req.fen)
            candidates = []
            try:
                with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine2:
                    infos = engine2.analyse(fen_board, chess.engine.Limit(time=0.5), multipv=5)
                    info_list = infos if isinstance(infos, list) else [infos]
                    for info in info_list:
                        if info.get("pv"):
                            cp = info["score"].white().score(mate_score=10000)
                            candidates.append({
                                "move": info["pv"][0].uci(),
                                "eval_pawns": round(cp / 100, 2) if cp is not None else 0
                            })
            except Exception:
                pass

            score_cp = int(candidates[0]["eval_pawns"] * 100) if candidates else 0

            game_board.push(move)
            return {
                "move":         move.uci(),
                "fen":          game_board.fen(),
                "is_game_over": game_board.is_game_over(),
                "outcome":      str(game_board.outcome()) if game_board.is_game_over() else None,
                "score_cp":     score_cp,
                "eval_pawns":   round(score_cp / 100, 2),
                "candidates":   candidates
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
# ΓöÇΓöÇΓöÇ /coach endpoint ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

@app.post("/coach")
def coach(req: CoachRequest, user=Depends(verify_key)):
    # 1. Get engine analysis
    try:
        analysis = analyse_position(req.fen, req.think_time)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Engine error: {str(e)}")

    score_pawns = round(analysis["score_cp"] / 100, 2)
    best_move = analysis["best_move"]
    pv = analysis["pv"]

    # 2. Build prompt
    board = chess.Board(req.fen)
    turn = "White" if board.turn == chess.WHITE else "Black"

    prompt = f"""You are Senkabala, an expert chess coach powered by a 2050 ELO engine.
Analyze this position and give coaching advice to a club-level player.

Position (FEN): {req.fen}
Side to move: {turn}
Engine evaluation: {'+' if score_pawns >= 0 else ''}{score_pawns} pawns (from White's perspective)
Engine best move: {best_move}
Suggested continuation: {' '.join(pv)}
"""

    if req.played_move and req.played_move != best_move:
        prompt += f"""
The player just played: {req.played_move}
This is not the engine's top choice. Briefly explain why {best_move} is better.
"""
    elif req.played_move and req.played_move == best_move:
        prompt += f"\nThe player found the best move: {req.played_move}. Confirm why this is strong.\n"

    if req.pgn:
        prompt += f"\nFull game PGN:\n{req.pgn}\nIdentify the key turning point and biggest mistake.\n"

    if req.lesson_type:
        prompt += f"\nFocus your explanation on {req.lesson_type} principles.\n"

    prompt += """
Respond in this exact format:
ASSESSMENT: (1 sentence on who stands better and why)
BEST MOVE: (explain the engine's best move in plain English)
PLAN: (2-3 sentences on the strategic plan going forward)
TIP: (one practical chess principle this position illustrates)
"""

    # 3. Call Claude
    try:
        ai_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = ai_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        explanation = message.content[0].text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Coach unavailable: {str(e)}")

    return {
        "best_move": best_move,
        "eval_pawns": score_pawns,
        "pv": pv,
        "coaching": explanation,
        "tier": user["tier"]
    }

# ΓöÇΓöÇΓöÇ /register endpoint (call this from your Stripe webhook) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

@app.post("/register")
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


# ΓöÇΓöÇ In-memory game state ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

lobby_queue: list = []          # waiting players: [{"ws": ws, "guest_id": id}]
active_games: dict = {}         # game_id ΓåÆ game state dict

CLOCK_SECONDS = 300             # 5 minutes each side

# ── Arena tournament state ──────────────────────────────────────
# tournament_connections[tid][user_id] = {ws, username, elo, score, available}
tournament_connections: dict = {}
# tournament_player_game[tid][user_id] = game_id | None
tournament_player_game: dict = {}


# ΓöÇΓöÇ Helpers ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

# ΓöÇΓöÇΓöÇ ELO calculation ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def calc_elo(my_elo: int, opp_elo: int, my_color: str, winner_color: str,
             time_control: str | None = None) -> int:
    """
    Standard ELO with K-factor adjusted by time control and rating level.
    winner_color: 'white' | 'black' | 'draw'
    """
    if winner_color == 'draw':
        score = 0.5
    elif winner_color == my_color:
        score = 1.0
    else:
        score = 0.0

    expected = 1 / (1 + 10 ** ((opp_elo - my_elo) / 400))
    k = k_factor(time_control, my_elo)
    new_elo = round(my_elo + k * (score - expected))
    return max(100, new_elo)


def k_factor(time_control: str | None, elo: int) -> int:
    """
    K-factor based on AfriChess time control categories:
      < 0:30          → Hyperbullet → K=40
      0:30 – 2:59     → Bullet      → K=40
      3:00 – 9:59     → Blitz       → K=32
      10:00 – 15:00   → Rapid       → K=24
      > 15:00         → Classical   → K=16
    Uses FIDE-style equivalent time: base_mins + 40*increment_secs/60
    Players under 1400 get +8 to converge faster.
    """
    equiv = 5.0  # default to blitz if unknown
    if time_control:
        try:
            parts = time_control.split('+')
            base  = float(parts[0])
            inc   = float(parts[1]) if len(parts) > 1 else 0.0
            equiv = base + (40 * inc / 60)
        except (ValueError, IndexError):
            pass

    if equiv < 3:
        k = 40    # Hyperbullet / Bullet (< 3 min equivalent)
    elif equiv < 10:
        k = 32    # Blitz (3–9:59)
    elif equiv <= 15:
        k = 24    # Rapid (10–15)
    else:
        k = 16    # Classical (> 15)

    if elo < 1400:
        k = min(k + 8, 40)

    return k


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


async def cleanup_game(game_id: str, delay: int = 10):
    await asyncio.sleep(delay)
    active_games.pop(game_id, None)


def parse_clock_seconds(time_control: str | None) -> tuple[int, int]:
    """
    Convert '5+0', '3+2', '10+5', '0+1' etc. to (base_seconds, increment_seconds).
    base_seconds: starting clock per player
    increment_seconds: added to clock after each move (Fischer increment)
    Special case: '0+1' → base=0s but minimum 0s start (pure increment game)
    """
    if not time_control:
        return 300, 0  # default 5+0
    try:
        parts = time_control.split('+')
        base_secs = int(float(parts[0]) * 60)
        inc_secs  = int(float(parts[1])) if len(parts) > 1 else 0
        return max(0, base_secs), max(0, inc_secs)
    except (ValueError, IndexError):
        return 300, 0


def new_game(game_id: str, white_ws: WebSocket, black_ws: WebSocket,
             white_id: str, black_id: str, time_control: str | None = None) -> dict:
    base_secs, inc_secs = parse_clock_seconds(time_control)
    # First-move timeout = 25% of base time, minimum 10s
    # e.g. 1+0 → 15s, 3+0 → 45s, 5+0 → 75s, 10+0 → 150s
    first_move_timeout = max(10, int(base_secs * 0.25))
    return {
        "id":                   game_id,
        "board":                chess.Board(),
        "white_ws":             white_ws,
        "black_ws":             black_ws,
        "white_game_ws":        None,
        "black_game_ws":        None,
        "white_id":             white_id,
        "black_id":             black_id,
        "clock":                {"w": base_secs, "b": base_secs},
        "increment":            inc_secs,
        "last_move_ts":         None,
        "first_move_timeout":   first_move_timeout,
        "first_move_deadline":  None,
        "moves_made":           0,
        "over":                 False,
        "time_control":         time_control,
        "white_profile":        None,
        "black_profile":        None,
    }


async def first_move_timeout_loop(game_id: str):
    """
    Enforce first-move timeout for BOTH sides:
    - White must play within first_move_timeout seconds of game start
    - Black must respond within first_move_timeout seconds of white's first move
    After move 2 this loop exits and clock_loop handles normal time management.
    """
    await asyncio.sleep(1)
    black_deadline = None   # set after white's first move

    while True:
        await asyncio.sleep(1)
        game = active_games.get(game_id)
        if not game or game["over"]:
            return

        deadline = game.get("first_move_deadline")
        moves_made = game.get("moves_made", 0)

        # Phase 1: waiting for both_connected to set the deadline
        if deadline is None:
            continue

        # Phase 2: white hasn't moved yet
        if moves_made == 0:
            if time.time() >= deadline:
                game["over"] = True
                await broadcast(game, {
                    "type":   "gameover",
                    "result": "black",
                    "reason": "first_move_timeout",
                    "detail": "White did not move in time.",
                    "clock":  game["clock"],
                })
                if not game.get("_elo_updated"):
                    game["_elo_updated"] = True
                    await update_elos(game, "black")
                await asyncio.sleep(1)
                active_games.pop(game_id, None)
                print(f"[game] {game_id} — white forfeited (no first move)", flush=True)
                return
            continue

        # Phase 3: white just made move 1 — start black's deadline
        if moves_made == 1 and black_deadline is None:
            black_deadline = time.time() + game["first_move_timeout"]
            continue

        # Phase 4: black hasn't responded
        if moves_made == 1 and black_deadline is not None:
            if time.time() >= black_deadline:
                game["over"] = True
                await broadcast(game, {
                    "type":   "gameover",
                    "result": "white",
                    "reason": "first_move_timeout",
                    "detail": "Black did not respond in time.",
                    "clock":  game["clock"],
                })
                if not game.get("_elo_updated"):
                    game["_elo_updated"] = True
                    await update_elos(game, "white")
                await asyncio.sleep(1)
                active_games.pop(game_id, None)
                print(f"[game] {game_id} — black forfeited (no first response)", flush=True)
                return
            continue

        # Both sides have made their first moves — hand off to clock_loop
        return


async def send(ws: WebSocket, msg: dict):
    """Safe send ΓÇö ignores errors if socket already closed."""
    try:
        await ws.send_json(msg)
    except Exception:
        pass


async def broadcast(game: dict, msg: dict):
    # Prefer the dedicated game WebSocket; fall back to lobby ws; skip if None
    w_ws = game.get("white_game_ws") or game.get("white_ws")
    b_ws = game.get("black_game_ws") or game.get("black_ws")
    if w_ws:
        await send(w_ws, msg)
    if b_ws:
        await send(b_ws, msg)


def deduct_clock(game: dict) -> float:
    """
    Deduct elapsed time from the side that just moved, then add increment (Fischer).
    On the very first move, just starts the clock without deducting.
    Returns the remaining clock for the side that just moved (after increment).
    """
    now = time.time()
    game["moves_made"] = game.get("moves_made", 0) + 1
    inc = game.get("increment", 0)

    if game["moves_made"] == 1:
        # First move — start the clock, don't deduct. Add increment to white's time.
        game["last_move_ts"] = now
        game["clock"]["w"] = game["clock"]["w"] + inc
        return game["clock"]["w"]

    last_ts = game.get("last_move_ts") or now
    elapsed = now - last_ts
    # The side that just moved is OPPOSITE of board.turn (move already pushed)
    just_moved = "b" if game["board"].turn == chess.WHITE else "w"
    # Deduct time then add increment (Fischer: you always get the increment)
    game["clock"][just_moved] = max(0, game["clock"][just_moved] - elapsed + inc)
    game["last_move_ts"] = now
    return game["clock"][just_moved]


async def clock_loop(game_id: str):
    """Background task — checks for flag fall every second.
    Does not tick until both players have connected AND the first move has been made.
    """
    while True:
        await asyncio.sleep(1)
        game = active_games.get(game_id)
        if not game or game["over"]:
            return

        # Wait until both game WebSockets are connected
        if not game.get("white_game_ws") or not game.get("black_game_ws"):
            continue

        # Wait until first move has been made (first_move_timeout_loop handles pre-game)
        if game.get("moves_made", 0) == 0:
            continue

        last_ts = game.get("last_move_ts")
        if last_ts is None:
            continue

        now     = time.time()
        elapsed = now - last_ts
        turn    = "w" if game["board"].turn == chess.WHITE else "b"
        remaining = game["clock"][turn] - elapsed

        if remaining <= 0:
            game["over"] = True
            loser  = "white" if turn == "w" else "black"
            winner = "black" if loser == "white" else "white"
            await broadcast(game, {
                "type":   "gameover",
                "result": winner,
                "reason": "timeout",
                "clock":  game["clock"],
            })
            if not game.get("_elo_updated"):
                game["_elo_updated"] = True
                await update_elos(game, winner)
            # Delay removal so gameover message is delivered before WS closes
            await asyncio.sleep(1)
            active_games.pop(game_id, None)
            return

        # Send clock tick to both players
        await broadcast(game, {
            "type":  "clock",
            "white": round(game["clock"]["w"] - (elapsed if turn == "w" else 0), 1),
            "black": round(game["clock"]["b"] - (elapsed if turn == "b" else 0), 1),
        })


def validate_and_push(game: dict, uci_move: str) -> chess.Move | None:
    """Validate UCI move against current board state. Returns move or None."""
    try:
        move = chess.Move.from_uci(uci_move)
        if move in game["board"].legal_moves:
            game["board"].push(move)
            return move
    except Exception:
        pass
    return None


# ΓöÇΓöÇΓöÇ ELO update helper ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

async def update_elos(game: dict, result: str):
    """
    Calculate and persist ELO changes for both players using the correct
    per-time-control column (elo_bullet, elo_blitz, elo_rapid, or elo).
    Sends elo_update message to each player.
    """
    wp = game.get("white_profile")
    bp = game.get("black_profile")
    if not wp or not bp:
        return
    if not wp.get("user_id") or not bp.get("user_id"):
        return

    tc  = game.get("time_control")
    col = elo_col_for_tc(tc)

    # Fetch current ELO from the correct column for both players
    async with httpx.AsyncClient() as client:
        wr = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"user_id": f"eq.{wp['user_id']}", "select": col},
            headers={"apikey": SUPABASE_SERVICE_KEY,
                     "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
        )
        br = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"user_id": f"eq.{bp['user_id']}", "select": col},
            headers={"apikey": SUPABASE_SERVICE_KEY,
                     "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
        )
    w_rows = wr.json(); b_rows = br.json()
    w_elo_old = (w_rows[0].get(col) or 1500) if w_rows else 1500
    b_elo_old = (b_rows[0].get(col) or 1500) if b_rows else 1500

    w_elo_new = calc_elo(w_elo_old, b_elo_old, "white", result, tc)
    b_elo_new = calc_elo(b_elo_old, w_elo_old, "black", result, tc)

    await supabase_update_elo(wp["user_id"], w_elo_new, tc)
    await supabase_update_elo(bp["user_id"], b_elo_new, tc)

    # Notify players — prefer game WS
    w_ws = game.get("white_game_ws") or game.get("white_ws")
    b_ws = game.get("black_game_ws") or game.get("black_ws")
    cat  = col.replace("elo_", "").capitalize() if col != "elo" else "Classical"

    if w_ws:
        await send(w_ws, {"type": "elo_update", "old_elo": w_elo_old,
                          "new_elo": w_elo_new, "category": cat, "column": col})
    if b_ws:
        await send(b_ws, {"type": "elo_update", "old_elo": b_elo_old,
                          "new_elo": b_elo_new, "category": cat, "column": col})


# ΓöÇΓöÇ WebSocket: Lobby (matchmaking) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

# ═══════════════════════════════════════════════════════════════
#  ARENA ENGINE
# ═══════════════════════════════════════════════════════════════

async def arena_send(ws, msg: dict):
    try:
        await ws.send_json(msg)
    except Exception:
        pass


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
        await arena_pair(tournament_id)
    except Exception as e:
        print(f"[arena] auto-start error: {e}", flush=True)
    finally:
        _tournament_locks.discard(lock_key)


async def arena_pair(tournament_id: str):
    conns = tournament_connections.get(tournament_id, {})
    pg    = tournament_player_game.setdefault(tournament_id, {})
    available = [uid for uid, info in conns.items()
                 if info.get("available") and not info.get("paused")
                 and pg.get(uid) is None]
    if len(available) < 1:
        return

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
        pair_count: dict = {}
        for g in r.json():
            key = tuple(sorted([g["white_id"], g["black_id"]]))
            pair_count[key] = pair_count.get(key, 0) + 1
    except Exception:
        pair_count = {}

    # Sort available players by score desc, then ELO desc
    available.sort(key=lambda uid: (-conns[uid].get("score", 0), -conns[uid].get("elo", 1500)))

    def times_played(p1, p2):
        return pair_count.get(tuple(sorted([p1, p2])), 0)

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
            paired.append((p1, best_p2))
            used.add(p1)
            used.add(best_p2)

    # Odd player — just leave them waiting, no bye points
    # Arena players should wait until a free opponent becomes available
    if len(available) % 2 == 1:
        for uid in available:
            if uid not in used:
                await arena_send(conns[uid]["ws"], {
                    "type":    "waiting",
                    "message": "Waiting for an available opponent…"
                })
                print(f"[arena] {uid} waiting (odd player) in {tournament_id}", flush=True)
                break

    for white_id, black_id in paired:
        asyncio.create_task(arena_launch_game(tournament_id, white_id, black_id))


async def arena_launch_game(tournament_id: str, white_id: str, black_id: str):
    conns = tournament_connections.get(tournament_id, {})
    pg    = tournament_player_game.setdefault(tournament_id, {})
    if white_id not in conns or black_id not in conns:
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
    except Exception as e:
        print(f"[arena] launch error: {e}", flush=True)
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
    active_games[game_id] = game
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


@app.websocket("/ws/tournament/{tournament_id}")
async def tournament_ws(ws: WebSocket, tournament_id: str):
    await ws.accept()
    user_id = None
    try:
        ident = await asyncio.wait_for(ws.receive_json(), timeout=20)
        user_id  = ident.get("user_id")
        username = ident.get("username", "?")
        elo      = int(ident.get("elo", 1500))
        score    = float(ident.get("score", 0))
    except Exception:
        await ws.close(); return
    if not user_id:
        await ws.close(); return

    tournament_connections.setdefault(tournament_id, {})
    tournament_player_game.setdefault(tournament_id, {})
    tournament_connections[tournament_id][user_id] = {
        "ws":         ws,
        "username":   username,
        "elo":        elo,
        "elo_bullet": int(ident.get("elo_bullet", elo)),
        "elo_blitz":  int(ident.get("elo_blitz",  elo)),
        "elo_rapid":  int(ident.get("elo_rapid",  elo)),
        "score":      score,
        "available":  True,
        "paused":     False,
    }
    if tournament_player_game[tournament_id].get(user_id):
        tournament_connections[tournament_id][user_id]["available"] = False

    await arena_send(ws, {"type": "connected", "user_id": user_id})
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
            asyncio.create_task(arena_pair(tournament_id))
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
                if user_id in conns:
                    conns[user_id]["available"] = True
                    conns[user_id]["paused"]    = False
                    conns[user_id]["score"]     = float(data.get("score", score))
                if user_id in pg:
                    pg[user_id] = None
                asyncio.create_task(arena_pair(tournament_id))

            elif data.get("type") == "pause":
                conns = tournament_connections.get(tournament_id, {})
                if user_id in conns:
                    conns[user_id]["available"] = False
                    conns[user_id]["paused"]    = True
                await arena_send(ws, {"type": "paused",
                    "message": "You are paused. You won't be paired until you resume."})
                print(f"[arena] {username} paused in {tournament_id}", flush=True)

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
                print(f"[arena] {username} resumed in {tournament_id}", flush=True)
                asyncio.create_task(arena_pair_delayed(tournament_id, delay=1.0))
    except WebSocketDisconnect:
        pass
    finally:
        conns = tournament_connections.get(tournament_id, {})
        if user_id and user_id in conns:
            conns[user_id]["available"] = False
            # Keep paused flag — if they reconnect they're still paused until they resume
            del conns[user_id]
        print(f"[arena] {username} left {tournament_id}", flush=True)


@app.websocket("/ws/lobby")
async def lobby(ws: WebSocket):
    await ws.accept()
    guest_id = "guest_" + uuid.uuid4().hex[:8]
    preferred_tc = "5+0"  # default, updated when client sends time_control message

    await send(ws, {"type": "waiting", "guest_id": guest_id})

    # Read initial messages to get time_control preference before matching
    # Give client 3s to send tc preference, then proceed with matching
    try:
        msg = await asyncio.wait_for(ws.receive_json(), timeout=3.0)
        if msg.get("type") == "time_control":
            preferred_tc = msg.get("tc", "5+0")
    except (asyncio.TimeoutError, Exception):
        pass

    # Match by same time control if possible, otherwise any opponent
    matched_opponent = None
    for i, opp in enumerate(lobby_queue):
        if opp.get("tc", "5+0") == preferred_tc:
            matched_opponent = lobby_queue.pop(i)
            break
    if matched_opponent is None and lobby_queue:
        matched_opponent = lobby_queue.pop(0)

    if matched_opponent:
        game_id  = uuid.uuid4().hex[:12]
        tc_used  = preferred_tc  # use the joining player's TC (or negotiate — keep simple)

        white_ws, white_id = matched_opponent["ws"], matched_opponent["guest_id"]
        black_ws, black_id = ws, guest_id

        game = new_game(game_id, white_ws, black_ws, white_id, black_id, tc_used)
        active_games[game_id] = game

        await send(white_ws, {
            "type":         "matched",
            "game_id":      game_id,
            "color":        "white",
            "opponent":     black_id,
            "time_control": tc_used,
        })
        await send(black_ws, {
            "type":         "matched",
            "game_id":      game_id,
            "color":        "black",
            "opponent":     white_id,
            "time_control": tc_used,
        })

        asyncio.create_task(clock_loop(game_id))
    else:
        lobby_queue.append({"ws": ws, "guest_id": guest_id, "tc": preferred_tc})

    # Keep lobby socket alive until matched or disconnected
    try:
        while True:
            data = await ws.receive_json()
            if isinstance(data, dict) and data.get("type") == "ping":
                pass  # keepalive, ignore
    except WebSocketDisconnect:
        lobby_queue[:] = [p for p in lobby_queue if p["guest_id"] != guest_id]


# ΓöÇΓöÇ WebSocket: Active game ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

@app.websocket("/ws/game/{game_id}")
async def game_ws(ws: WebSocket, game_id: str):
    await ws.accept()

    game = active_games.get(game_id)
    if not game:
        await send(ws, {"type": "error", "detail": "Game not found."})
        await ws.close()
        return

    # Identify which player this is by registering their new game socket
    player_id = None
    color = None
    if game["white_ws"] is None or game.get("white_game_ws") is None:
        # Check if this is the white player connecting
        # We'll resolve by slot: first to connect gets their slot
        pass

    # Wait for client to declare their color via first message
    # First message must be {"type": "claim", "color": "white"/"black"}
    try:
        claim_data = await ws.receive_json()
    except Exception:
        await ws.close()
        return

    claimed_color = claim_data.get("color", "")

    if not game.get("_lock"):
        game["_lock"] = asyncio.Lock()

    async with game["_lock"]:
        # For tournament games, verify this user is one of the two assigned players
        connecting_user_id = claim_data.get("user_id")
        if game.get("tournament_id") and connecting_user_id:
            if (claimed_color == "white" and connecting_user_id != game.get("white_id")) or \
               (claimed_color == "black" and connecting_user_id != game.get("black_id")):
                await send(ws, {"type": "error", "detail": "You are not a player in this tournament game."})
                await ws.close()
                return

        if claimed_color == "white" and game.get("white_game_ws") is None:
            game["white_game_ws"] = ws
            game["white_ws"] = ws
            color = "w"
        elif claimed_color == "black" and game.get("black_game_ws") is None:
            game["black_game_ws"] = ws
            game["black_ws"] = ws
            color = "b"
        else:
            await send(ws, {"type": "error", "detail": "Slot unavailable."})
            await ws.close()
            return

    # Notify both players once both are connected
    if game.get("white_game_ws") and game.get("black_game_ws"):
        # Set first-move deadline now that both players are present
        game["last_move_ts"]        = time.time()
        game["first_move_deadline"] = time.time() + game.get("first_move_timeout", 60)
        asyncio.create_task(first_move_timeout_loop(game["id"]))
        await broadcast(game, {"type": "both_connected",
                               "first_move_timeout": game.get("first_move_timeout", 60)})

        # Push already-received profile to the late-connecting player
        if color == "w" and game.get("black_profile"):
            await send(ws, {
                "type":     "opponent_profile",
                "username": game["black_profile"]["username"],
                "elo":      game["black_profile"]["elo"],
            })
        elif color == "b" and game.get("white_profile"):
            await send(ws, {
                "type":     "opponent_profile",
                "username": game["white_profile"]["username"],
                "elo":      game["white_profile"]["elo"],
            })

    try:
        while True:
            data = await ws.receive_json()

            msg_type = data.get("type")

            if game["over"]:
                if msg_type not in ("rematch_offer", "rematch_accept", "rematch_decline", "draw_claim", "ping"):
                    await send(ws, {"type": "error", "detail": "Game is over."})
                    continue


            # ΓöÇΓöÇ Keepalive ping (ignore) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
            if msg_type == "ping":
                continue

            # ΓöÇΓöÇ Profile (sent on connect) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
            if msg_type == "profile":
                tc  = data.get("time_control") or game.get("time_control") or "5+0"
                col = elo_col_for_tc(tc)
                # Pick the correct per-category ELO for display
                elo_map = {
                    "elo_bullet": data.get("elo_bullet"),
                    "elo_blitz":  data.get("elo_blitz"),
                    "elo_rapid":  data.get("elo_rapid"),
                    "elo":        data.get("elo"),
                }
                display_elo = elo_map.get(col) or data.get("elo") or 1500

                profile = {
                    "username":   data.get("username", "guest"),
                    "elo":        display_elo,
                    "elo_bullet": data.get("elo_bullet"),
                    "elo_blitz":  data.get("elo_blitz"),
                    "elo_rapid":  data.get("elo_rapid"),
                    "user_id":    data.get("user_id"),
                }
                # Store time_control on game so update_elos can use correct K-factor
                if tc and not game.get("time_control"):
                    game["time_control"] = tc
                if color == "w":
                    game["white_profile"] = profile
                    await send(game.get("black_game_ws") or game["black_ws"], {
                        "type":     "opponent_profile",
                        "username": profile["username"],
                        "elo":      display_elo,
                    })
                else:
                    game["black_profile"] = profile
                    await send(game.get("white_game_ws") or game["white_ws"], {
                        "type":     "opponent_profile",
                        "username": profile["username"],
                        "elo":      display_elo,
                    })
                continue

            # ─── Move ────────────────────────────────────────────────────────
            if msg_type == "move":
                # Only the player whose turn it is can move
                expected = "w" if game["board"].turn == chess.WHITE else "b"
                if color != expected:
                    await send(ws, {"type": "error", "detail": "Not your turn."})
                    continue

                # Reject move if this player's clock is already at 0
                # (clock_loop may not have fired yet — this closes the race window)
                now_check = time.time()
                last_ts   = game.get("last_move_ts") or now_check
                elapsed   = now_check - last_ts
                if game["clock"][color] - elapsed <= 0:
                    # Flag this player immediately
                    if not game["over"]:
                        game["over"] = True
                        loser  = "white" if color == "w" else "black"
                        winner = "black" if loser == "white" else "white"
                        await broadcast(game, {
                            "type":   "gameover",
                            "result": winner,
                            "reason": "timeout",
                            "clock":  game["clock"],
                        })
                        if not game.get("_elo_updated"):
                            game["_elo_updated"] = True
                            await update_elos(game, winner)
                        active_games.pop(game_id, None)
                    continue

                uci = data.get("move", "")
                move = validate_and_push(game, uci)
                if move is None:
                    await send(ws, {"type": "error", "detail": "Illegal move."})
                    continue

                # Deduct clock
                remaining = deduct_clock(game)
                fen = game["board"].fen()

                # Check game over conditions
                if game["board"].is_game_over():
                    game["over"] = True
                    outcome = game["board"].outcome()
                    result = (
                        "white" if outcome.winner == chess.WHITE else
                        "black" if outcome.winner == chess.BLACK else
                        "draw"
                    )
                    reason = outcome.termination.name.lower()
                    await broadcast(game, {
                        "type":   "gameover",
                        "result": result,
                        "reason": reason,
                        "fen":    fen,
                        "clock":  game["clock"],
                    })
                    if not game.get("_elo_updated"):
                        game["_elo_updated"] = True
                        await update_elos(game, result)
                    active_games.pop(game_id, None)
                else:
                    await broadcast(game, {
                        "type":  "move",
                        "move":  uci,
                        "fen":   fen,
                        "clock": game["clock"],
                        "turn":  "white" if game["board"].turn == chess.WHITE else "black",
                    })

            # ΓöÇΓöÇ Resign ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
            elif msg_type == "resign":
                game["over"] = True
                winner = "black" if color == "w" else "white"
                await broadcast(game, {
                    "type":   "gameover",
                    "result": winner,
                    "reason": "resignation",
                    "clock":  game["clock"],
                })
                if not game.get("_elo_updated"):
                    game["_elo_updated"] = True
                    await update_elos(game, winner)
                asyncio.create_task(cleanup_game(game_id, delay=10))

            # ── Draw offer ─────────────────────────────────────────────────────
            elif msg_type == "draw_offer":
                # Send over the GAME websocket, not the lobby websocket
                opponent_ws = game.get("black_game_ws") if color == "w" else game.get("white_game_ws")
                if opponent_ws:
                    await send(opponent_ws, {"type": "draw_offer"})
                else:
                    # Fallback to lobby ws if game ws not yet connected
                    opponent_ws = game["black_ws"] if color == "w" else game["white_ws"]
                    if opponent_ws:
                        await send(opponent_ws, {"type": "draw_offer"})

            elif msg_type == "draw_accept":
                game["over"] = True
                await broadcast(game, {
                    "type":   "gameover",
                    "result": "draw",
                    "reason": "agreement",
                    "clock":  game["clock"],
                })
                if not game.get("_elo_updated"):
                    game["_elo_updated"] = True
                    await update_elos(game, "draw")
                asyncio.create_task(cleanup_game(game_id, delay=10))

            elif msg_type == "draw_claim":
                reason = data.get("reason", "threefold_repetition")
                board  = game["board"]
                valid  = (
                    (reason == "threefold_repetition" and board.is_repetition(3)) or
                    board.is_fifty_moves() or
                    board.is_insufficient_material()
                )
                if valid and not game["over"]:
                    game["over"] = True
                    await broadcast(game, {
                        "type":   "gameover",
                        "result": "draw",
                        "reason": reason,
                        "clock":  game["clock"],
                    })
                    if not game.get("_elo_updated"):
                        game["_elo_updated"] = True
                        await update_elos(game, "draw")
                    asyncio.create_task(cleanup_game(game_id, delay=10))

            elif msg_type == "rematch_offer":
                game["rematch_offered_by"] = color
                opponent_ws = game["black_ws"] if color == "w" else game["white_ws"]
                await send(opponent_ws, {"type": "rematch_offer"})

            elif msg_type == "rematch_accept":
                old_white_profile = game["white_profile"]
                old_black_profile = game["black_profile"]
                old_white_ws      = game["white_ws"]
                old_black_ws      = game["black_ws"]

                new_game_id = uuid.uuid4().hex[:12]
                # Colors swapped: old black becomes new white
                ng = new_game(new_game_id, old_black_ws, old_white_ws,
                              old_black_profile.get("username", "?") if old_black_profile else "?",
                              old_white_profile.get("username", "?") if old_white_profile else "?")
                ng["white_profile"]   = old_black_profile
                ng["black_profile"]   = old_white_profile
                # Clear stale lobby sockets — players will connect fresh via /ws/game/{new_game_id}
                ng["white_ws"] = None
                ng["black_ws"] = None
                active_games[new_game_id] = ng

                # Notify players on the OLD game sockets before they disconnect
                await send(old_black_ws, {
                    "type": "rematch_start", "game_id": new_game_id, "color": "white"
                })
                await send(old_white_ws, {
                    "type": "rematch_start", "game_id": new_game_id, "color": "black"
                })
                asyncio.create_task(clock_loop(new_game_id))
                asyncio.create_task(cleanup_game(game_id, delay=10))

            elif msg_type == "rematch_decline":
                opponent_ws = game["black_ws"] if color == "w" else game["white_ws"]
                await send(opponent_ws, {"type": "rematch_declined"})

    except WebSocketDisconnect:
        if not game["over"]:
            game["over"] = True
            winner = "black" if color == "w" else "white"
            opponent_ws = game["black_ws"] if color == "w" else game["white_ws"]
            await send(opponent_ws, {
                "type":   "gameover",
                "result": winner,
                "reason": "disconnect",
                "clock":  game["clock"],
            })
            if not game.get("_elo_updated"):
                game["_elo_updated"] = True
                await update_elos(game, winner)
            active_games.pop(game_id, None)


# ΓöÇΓöÇ Lobby status (optional debug endpoint) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

@app.get("/lobby/status")
def lobby_status():
    return {
        "waiting":      len(lobby_queue),
        "active_games": len(active_games),
    }

# ΓöÇΓöÇΓöÇ Health / static ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ


@app.get("/profile")
def profile():
    return FileResponse("profile.html")

# ─── Free coach endpoint (no API key needed, uses daily quota) ────────────────

class FreeCoachRequest(BaseModel):
    fen: str
    played_move: Optional[str] = None
    pgn: Optional[str] = None
    user_id: str
    think_time: float = 0.5

FREE_COACH_LIMIT = 10  # free tier daily limit (kept for legacy refs)

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

@app.post("/coach-free")
async def coach_free(req: FreeCoachRequest):
    if not SUPABASE_SERVICE_KEY:
        raise HTTPException(503, "Service unavailable")

    today = datetime.utcnow().date().isoformat()

    # Fetch profile
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"user_id": f"eq.{req.user_id}", "select": "username,elo,coach_uses_today,coach_reset_date"},
            headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
        )
        rows = r.json()

    if not rows:
        raise HTTPException(404, "Profile not found")

    profile = rows[0]
    uses_today = profile.get("coach_uses_today") or 0
    reset_date = profile.get("coach_reset_date") or ""

    # Reset counter if new day
    if reset_date != today:
        uses_today = 0

    # Get plan-based limit
    plan = await get_user_plan(req.user_id)
    daily_limit = COACH_LIMITS.get(plan, COACH_LIMITS["free"])

    if uses_today >= daily_limit:
        upgrade_msg = "Upgrade to Club ($5/mo) for 200 analyses/day." if plan == "free" else "Daily limit reached."
        raise HTTPException(429, upgrade_msg)

    # Increment usage
    async with httpx.AsyncClient() as client:
        await client.patch(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"user_id": f"eq.{req.user_id}"},
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            json={"coach_uses_today": uses_today + 1, "coach_reset_date": today}
        )

    # Run analysis — use semaphore and run in thread to avoid blocking event loop
    try:
        async with engine_semaphore:
            loop = asyncio.get_event_loop()
            analysis = await loop.run_in_executor(None, analyse_position, req.fen, req.think_time)
    except Exception as e:
        import traceback
        print(f"coach-free engine error: {traceback.format_exc()}", flush=True)
        raise HTTPException(500, f"Engine error: {e}")

    score_pawns = round(analysis["score_cp"] / 100, 2)

    if not analysis.get("best_move"):
        raise HTTPException(500, "Engine could not analyse this position. Please try again.")

    board = chess.Board(req.fen)
    turn = "White" if board.turn == chess.WHITE else "Black"

    # Build human-readable piece list so Claude doesn't misread the FEN
    piece_names = {
        chess.PAWN: "Pawn", chess.KNIGHT: "Knight", chess.BISHOP: "Bishop",
        chess.ROOK: "Rook", chess.QUEEN: "Queen", chess.KING: "King"
    }
    white_pieces, black_pieces = [], []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            sq_name = chess.square_name(sq)
            name = f"{piece_names[piece.piece_type]} on {sq_name}"
            if piece.color == chess.WHITE:
                white_pieces.append(name)
            else:
                black_pieces.append(name)

    # Describe the best move in human terms
    best_uci = analysis['best_move']
    best_move_desc = ""
    if best_uci and len(best_uci) >= 4:
        from_sq = chess.parse_square(best_uci[:2])
        to_sq = chess.parse_square(best_uci[2:4])
        moving_piece = board.piece_at(from_sq)
        captured_piece = board.piece_at(to_sq)
        from_name = chess.square_name(from_sq)
        to_name = chess.square_name(to_sq)
        piece_str = piece_names.get(moving_piece.piece_type, "Piece") if moving_piece else "Piece"
        color_str = "White" if (moving_piece and moving_piece.color == chess.WHITE) else "Black"
        if captured_piece:
            cap_str = piece_names.get(captured_piece.piece_type, "piece")
            best_move_desc = f"{color_str}'s {piece_str} on {from_name} captures the {cap_str} on {to_name}"
        else:
            best_move_desc = f"{color_str}'s {piece_str} moves from {from_name} to {to_name}"
        # Check if it gives check
        test_board = board.copy()
        test_board.push(chess.Move(from_sq, to_sq))
        if test_board.is_checkmate():
            best_move_desc += " — CHECKMATE"
        elif test_board.is_check():
            best_move_desc += " (giving check)"

    # Mate score display
    if abs(score_pawns) >= 99:
        eval_display = "Forced checkmate" if score_pawns < 0 else "Forced checkmate for White"
    else:
        eval_display = f"{'+' if score_pawns >= 0 else ''}{score_pawns} pawns (White's perspective)"

    prompt = f"""You are Senkabala, an expert chess coach powered by a strong chess engine.
Give coaching advice based on this EXACT position:

Side to move: {turn}
Engine evaluation: {eval_display}

White pieces: {', '.join(white_pieces) if white_pieces else 'none'}
Black pieces: {', '.join(black_pieces) if black_pieces else 'none'}

Engine best move: {best_move_desc}
Suggested continuation (UCI): {' '.join(analysis['pv'][:3])}

Base your entire response on the piece positions listed above. Do not invent pieces or squares not listed.
"""
    if req.played_move:
        played_desc = ""
        try:
            from_sq2 = chess.parse_square(req.played_move[:2])
            to_sq2 = chess.parse_square(req.played_move[2:4])
            played_piece = board.piece_at(from_sq2)
            p_str = piece_names.get(played_piece.piece_type, "Piece") if played_piece else "Piece"
            played_desc = f"{p_str} from {req.played_move[:2]} to {req.played_move[2:4]}"
        except Exception:
            played_desc = req.played_move

        if req.played_move != best_uci:
            prompt += f"\nThe player just played: {played_desc}\nBriefly explain why the engine move is better.\n"
        else:
            prompt += f"\nThe player found the best move: {played_desc}. Confirm why this is strong.\n"

    if req.pgn:
        prompt += f"\nGame moves so far: {req.pgn}\n"

    prompt += """
Respond in this exact format:
ASSESSMENT: (1 sentence on who stands better and why, based on the piece positions above)
BEST MOVE: (explain the engine best move in plain English using the piece description provided)
PLAN: (2-3 sentences on the strategic plan going forward)
TIP: (one practical chess principle this position illustrates)
"""

    try:
        ai_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = ai_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        explanation = message.content[0].text
    except Exception as e:
        import traceback
        print(f"coach-free anthropic error: {traceback.format_exc()}", flush=True)
        raise HTTPException(500, f"Coach unavailable: {e}")

    return {
        "best_move": analysis["best_move"],
        "eval_pawns": score_pawns,
        "pv": analysis["pv"],
        "coaching": explanation,
        "uses_today": uses_today + 1,
        "uses_remaining": max(0, daily_limit - (uses_today + 1)),
    }

# ─── Profile stats endpoint ───────────────────────────────────────────────────

@app.get("/api/profile/{user_id}")
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

    # Compute stats
    wins = sum(1 for g in games if g.get("result") == g.get("player_color"))
    losses = sum(1 for g in games if g.get("result") not in (g.get("player_color"), "draw") and g.get("result"))
    draws = sum(1 for g in games if g.get("result") == "draw")
    total = len(games)

    # ELO history from games
    elo_history = []
    for g in reversed(games):
        if g.get("player_elo_after"):
            elo_history.append({"date": g["created_at"][:10], "elo": g["player_elo_after"]})

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
        "created_at": profile.get("created_at"),
        "country":    profile.get("country"),
        "games_played": profile.get("games_played", 0),
        "wins":       wins,
        "losses":     losses,
        "draws":      draws,
        "total":      total,
        "win_rate":   round(wins / total * 100) if total else 0,
        "recent_games":  games[:10],
        "elo_history":   elo_history[-20:],
        "coach_uses_today":      uses_today,
        "coach_uses_remaining":  max(0, FREE_COACH_LIMIT - uses_today),
        "coach_limit":           FREE_COACH_LIMIT,
    }


class AnalyseRequest(BaseModel):
    fen: str

@app.post("/analyse-position")
async def analyse_pos(req: AnalyseRequest):
    try:
        async with engine_semaphore:
            loop = asyncio.get_event_loop()
            analysis = await loop.run_in_executor(None, analyse_position, req.fen, 2.0)
        return {
            "best_move": analysis["best_move"],
            "eval_pawns": round(analysis["score_cp"] / 100, 2),
            "score_cp":   analysis["score_cp"],
            "pv":         analysis["pv"],
        }
    except Exception as e:
        import traceback
        print(f"analyse-position error: {traceback.format_exc()}", flush=True)
        raise HTTPException(500, f"Engine error: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}

class FeedbackRequest(BaseModel):
    rating:  int              # 1–5
    message: Optional[str] = None
    page:    Optional[str] = None

@app.post("/api/feedback")
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

app.mount("/img", StaticFiles(directory="img"), name="img")
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def root():
    return FileResponse("landing.html")

@app.get("/play")
def play():
    return FileResponse("index.html")        # vs engine

@app.get("/multiplayer")
def multiplayer():
    return FileResponse("play_multiplayer.html")   # 1v1 live

@app.get("/landing")
def landing():
    return FileResponse("landing.html")

@app.get("/logo.png")
def logo():
    return FileResponse("logo.png")


# ── Lemon Squeezy Webhook ─────────────────────────────────────────
from fastapi import Request

@app.post("/api/lemon-webhook")
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



# ═══════════════════════════════════════════════════════════════
#  TOURNAMENT ENGINE
# ═══════════════════════════════════════════════════════════════

class TournamentStartRequest(BaseModel):
    tournament_id: str
    user_id: str

class TournamentResultRequest(BaseModel):
    game_id: str       # tournament_games.id
    result: str        # 'white' | 'black' | 'draw'
    user_id: str
    time_control: Optional[str] = None  # unused server-side (fetched from DB), kept for compat


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


@app.post("/api/tournament/start")
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

            # Arena: pairing via WebSocket engine
            if t.get("format") == "arena":
                asyncio.create_task(arena_pair(req.tournament_id))
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
                        json={"score": white.get("score", 0) + 1}
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

@app.post("/api/tournament/next-round")
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
                        json={"score": white.get("score", 0) + 1}
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

@app.post("/api/tournament/result")
async def submit_result(req: TournamentResultRequest, authorization: str = Header(None)):
    """
    Submit a tournament game result.
    - Marks the game result in tournament_games
    - Updates tournament_players.score (win=1, draw=0.5, loss=0)
    - Syncs tournament_players.elo snapshot from profiles (ELO already updated by update_elos via WS)
    - Does NOT recalculate ELO — that is handled by update_elos() when the game ends over WebSocket
    """
    if not SUPABASE_SERVICE_KEY:
        raise HTTPException(503, "Service unavailable")
    user_id = await verify_jwt(authorization)
    if req.result not in ('white', 'black', 'draw'):
        raise HTTPException(400, "Invalid result")

    async with httpx.AsyncClient() as client:
        # Get the game
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/tournament_games",
            params={"id": f"eq.{req.game_id}", "select": "*"},
            headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
        )
        games = r.json()
        if not games:
            raise HTTPException(404, "Game not found")
        g = games[0]
        if g.get('result'):
            raise HTTPException(400, "Result already submitted")

        # Only white or black player can submit
        if user_id not in (g['white_id'], g['black_id']):
            raise HTTPException(403, "Not a player in this game")

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

        # Mark game result
        await client.patch(
            f"{SUPABASE_URL}/rest/v1/tournament_games",
            params={"id": f"eq.{req.game_id}"},
            headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                     "Content-Type": "application/json"},
            json={"result": req.result, "played_at": datetime.utcnow().isoformat()}
        )

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

        # Update tournament standings: score + ELO snapshot
        async def add_score(uid, pts, elo_snapshot):
            r2 = await client.get(
                f"{SUPABASE_URL}/rest/v1/tournament_players",
                params={"tournament_id": f"eq.{tid}", "user_id": f"eq.{uid}", "select": "score"},
                headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
            )
            current = r2.json()[0].get('score', 0) if r2.json() else 0
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/tournament_players",
                params={"tournament_id": f"eq.{tid}", "user_id": f"eq.{uid}"},
                headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                         "Content-Type": "application/json"},
                json={"score": current + pts, "elo": elo_snapshot}
            )

        if req.result == 'white':
            await add_score(g['white_id'], 1,   w_elo)
            await add_score(g['black_id'], 0,   b_elo)
        elif req.result == 'black':
            await add_score(g['white_id'], 0,   w_elo)
            await add_score(g['black_id'], 1,   b_elo)
        else:
            await add_score(g['white_id'], 0.5, w_elo)
            await add_score(g['black_id'], 0.5, b_elo)

    # Arena: release players and re-pair
    try:
        if tc_rows and tc_rows[0].get("format") == "arena" and tc_rows[0].get("status") == "active":
            pg    = tournament_player_game.get(tid, {})
            conns = tournament_connections.get(tid, {})
            for uid in (g['white_id'], g['black_id']):
                if uid in pg:
                    pg[uid] = None
                if uid in conns:
                    conns[uid]["available"] = True
                    conns[uid]["score"] = (
                        conns[uid].get("score", 0) + 1   if (
                            (req.result == "white" and uid == g['white_id']) or
                            (req.result == "black" and uid == g['black_id'])
                        ) else
                        conns[uid].get("score", 0) + 0.5 if req.result == "draw"
                        else conns[uid].get("score", 0)
                    )
                    await arena_send(conns[uid]["ws"], {"type": "game_over", "result": req.result})
            asyncio.create_task(arena_pair_delayed(tid))
    except Exception as e:
        print(f"[arena] re-pair error: {e}", flush=True)

    return {"ok": True, "result": req.result}


@app.post("/api/update-country")
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


@app.post("/api/create-tournament")
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


@app.post("/api/join-tournament")
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
                    "select": "country,region,max_players,status,prize_pool"},
            headers={"apikey": SUPABASE_SERVICE_KEY,
                     "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
        )
        ts = r.json()
        if not ts:
            raise HTTPException(404, "Tournament not found")
        t = ts[0]

        if t["status"] != "upcoming":
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


ADMIN_USER_IDS = set(os.getenv("ADMIN_USER_IDS", "9c51d331-8eba-4da5-b644-64cd4fc168d1").split(","))

# ── African regions ────────────────────────────────────────────────────────────
REGIONS: dict[str, list[str]] = {
    "east_africa":    ["UG","KE","TZ","RW","BI","ET","SS","SO","ER","DJ","SD"],
    "west_africa":    ["NG","GH","SN","CI","CM","BJ","TG","GN","GW","SL","LR","GM","MR","CV","NE","BF","ML"],
    "north_africa":   ["EG","LY","TN","DZ","MA"],
    "south_africa":   ["ZA","ZW","ZM","MW","MZ","NA","BW","LS","SZ","AO","MG","MU","SC","KM","ST","CV"],
    "central_africa": ["CD","CG","CF","GA","GQ","TD"],
}
REGION_LABELS: dict[str, str] = {
    "east_africa":    "East Africa",
    "west_africa":    "West Africa",
    "north_africa":   "North Africa",
    "south_africa":   "Southern Africa",
    "central_africa": "Central Africa",
}

COUNTRY_NAMES_SERVER: dict[str, str] = {
    "UG":"Uganda","KE":"Kenya","TZ":"Tanzania","RW":"Rwanda","BI":"Burundi",
    "ET":"Ethiopia","SS":"South Sudan","SO":"Somalia","ER":"Eritrea","DJ":"Djibouti",
    "SD":"Sudan","NG":"Nigeria","GH":"Ghana","SN":"Senegal","CI":"Ivory Coast",
    "CM":"Cameroon","EG":"Egypt","LY":"Libya","TN":"Tunisia","DZ":"Algeria",
    "MA":"Morocco","ZA":"South Africa","ZW":"Zimbabwe","ZM":"Zambia","MW":"Malawi",
    "MZ":"Mozambique","NA":"Namibia","BW":"Botswana","LS":"Lesotho","SZ":"Eswatini",
    "AO":"Angola","CD":"DR Congo","CG":"Congo","CF":"Central African Republic",
    "GA":"Gabon","GQ":"Equatorial Guinea","TD":"Chad","MG":"Madagascar",
}

def player_in_region(player_country: str | None, region: str) -> bool:
    if not player_country:
        return False
    return player_country in REGIONS.get(region, [])

@app.post("/api/admin/force-end-game")
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
    await asyncio.sleep(1)
    active_games.pop(game_id, None)
    print(f"[admin] {caller_id} force-ended game {game_id} → {winner}", flush=True)
    return {"ok": True, "game_id": game_id, "winner": winner}


@app.post("/api/admin/ban")
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


@app.delete("/api/leave-tournament")
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


@app.get("/api/tournaments")
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

@app.get("/api/tournaments/{tournament_id}")
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
    # Merge live paused state from in-memory connections
    conns = tournament_connections.get(tournament_id, {})
    for p in players:
        uid = p.get("user_id")
        if uid and uid in conns:
            p["paused"] = conns[uid].get("paused", False)
        else:
            p["paused"] = False

    return {"tournament": ts[0], "players": players, "games": g_r.json()}

@app.get("/favicon.ico")
def favicon():
    return FileResponse("favicon.ico")

@app.get("/favicon_16x16.png")
def favicon16():
    return FileResponse("favicon_16x16.png")

@app.get("/favicon_32x32.png")
def favicon32():
    return FileResponse("favicon_32x32.png")

@app.get("/apple-touch-icon.png")
def apple_touch():
    return FileResponse("apple-touch-icon.png")

@app.get("/android-chrome-192x192.png")
def android192():
    return FileResponse("android-chrome-192x192.png")

@app.get("/android-chrome-512x512.png")
def android512():
    return FileResponse("android-chrome-512x512.png")

@app.get("/site.webmanifest")
def webmanifest():
    return FileResponse("site.webmanifest", media_type="application/manifest+json")

@app.get("/sitemap.xml")
def sitemap():
    return FileResponse("sitemap.xml", media_type="application/xml")

@app.get("/history")
def history():
    return FileResponse("history.html")

@app.get("/tournaments")
def tournaments():
    return FileResponse("tournament.html")

@app.get("/chessboard-1.0.0.min.css")
def cb_css():
    return FileResponse("chessboard-1.0.0.min.css")

@app.get("/chessboard-1.0.0.min.js")
def cb_js():
    return FileResponse("chessboard-1.0.0.min.js")

@app.get("/jquery.min.js")
def jquery():
    return FileResponse("jquery.min.js")

@app.get("/chess.min.js")
def chess_js():
    return FileResponse("chess.min.js")

import os

@app.get("/debug-coach")
async def debug_coach():
    """Test engine analysis in isolation."""
    import traceback
    try:
        loop = asyncio.get_event_loop()
        analysis = await loop.run_in_executor(
            None, analyse_position,
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", 0.5
        )
        return {"status": "ok", "analysis": analysis}
    except Exception as e:
        return {"status": "error", "error": str(e), "trace": traceback.format_exc()}


def debug_files():
    return {
        "cwd": os.getcwd(),
        "files": os.listdir(".")
    }




if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

