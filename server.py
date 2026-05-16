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
from datetime import datetime, timedelta
import asyncio
import uuid
import time
from fastapi import WebSocket, WebSocketDisconnect
import asyncio, uuid, time

app = FastAPI()

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

# ΓöÇΓöÇΓöÇ Supabase admin client (service role ΓÇö bypasses RLS) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
import httpx

async def supabase_get_profile(user_id: str) -> dict | None:
    """Fetch profile by user_id using service role key."""
    if not SUPABASE_SERVICE_KEY:
        return None
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"user_id": f"eq.{user_id}", "select": "username,elo"},
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            }
        )
        rows = r.json()
        return rows[0] if rows else None

async def supabase_update_elo(user_id: str, new_elo: int):
    """Update ELO for a user using service role key."""
    if not SUPABASE_SERVICE_KEY:
        return
    async with httpx.AsyncClient() as client:
        await client.patch(
            f"{SUPABASE_URL}/rest/v1/profiles",
            params={"user_id": f"eq.{user_id}"},
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            json={"elo": new_elo}
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
    difficulty: int = 3  # 1=Beginner ΓåÆ 5=Expert

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

def analyse_position(fen: str, think_time: float):
    """Returns best_move and score by talking directly to engine process."""
    import subprocess, threading

    board = chess.Board(fen)
    think_ms = int(max(think_time, 2.0) * 1000)

    best_move = None
    score = 0
    pv_moves = []
    stderr_lines = []

    proc = subprocess.Popen(
        [ENGINE_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    def read_stderr():
        for line in proc.stderr:
            stderr_lines.append(line.strip())

    t = threading.Thread(target=read_stderr, daemon=True)
    t.start()

    commands = f"uci\nucinewgame\nposition fen {board.fen()}\ngo movetime {think_ms}\n"
    try:
        stdout_data, _ = proc.communicate(input=commands, timeout=think_ms/1000 + 5)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout_data, _ = proc.communicate()
    t.join(timeout=2)

    # Parse stdout for bestmove
    for line in stdout_data.splitlines():
        line = line.strip()
        if line.startswith("bestmove"):
            parts = line.split()
            if len(parts) >= 2 and parts[1] not in ("(none)", "0000"):
                best_move = parts[1]

    # Parse stderr for score and pv — engine writes info lines there
    best_depth = -1
    for line in stderr_lines:
        if "score cp" in line and "depth" in line:
            try:
                parts = line.split()
                depth = int(parts[parts.index("depth") + 1]) if "depth" in parts else 0
                cp_idx = parts.index("cp")
                cp = int(parts[cp_idx + 1])
                if depth > best_depth:
                    best_depth = depth
                    score = cp
                    # Parse pv from this line if present
                    if "pv" in parts:
                        pv_idx = parts.index("pv")
                        pv_moves = parts[pv_idx + 1: pv_idx + 6]
            except (ValueError, IndexError):
                continue

    # Score is from side-to-move perspective — convert to white's perspective
    if board.turn == chess.BLACK:
        score = -score

    # Fallback best_move from pv
    if not best_move and pv_moves:
        best_move = pv_moves[0]

    print(f"DEBUG analyse: best_move={best_move}, score={score}, depth={best_depth}", flush=True)
    return {"best_move": best_move, "score_cp": score, "pv": pv_moves}

# ΓöÇΓöÇΓöÇ Original /move endpoint (unchanged) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
engine_semaphore = asyncio.Semaphore(3)

@app.post("/move")
async def get_move(req: MoveRequest):
    async with engine_semaphore:
        try:
            board = chess.Board(req.fen)
            # Don't run engine on finished positions
            if board.is_game_over():
                return {
                    "move": None,
                    "fen": req.fen,
                    "is_game_over": True,
                    "outcome": str(board.outcome()),
                    "score_cp": 0,
                    "eval_pawns": 0,
                    "candidates": []
                }

            # Instance 1: get the best move
            import random
            think_ms = DIFFICULTY_SETTINGS.get(req.difficulty, int(req.think_time * 1000))

            # At low difficulty, randomly pick a legal move instead of engine's best
            random_chance = {1: 0.75, 2: 0.40, 3: 0.15, 4: 0.0, 5: 0.0}
            if random.random() < random_chance.get(req.difficulty, 0):
                move = random.choice(list(board.legal_moves))
            else:
                with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
                    result = engine.play(board, chess.engine.Limit(
                        white_clock=think_ms / 1000,
                        black_clock=think_ms / 1000,
                        remaining_moves=1
                    ))
                    move = result.move

            # Instance 2: get candidates separately
            candidates = []
            try:
                with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine2:
                    infos = engine2.analyse(
                        board,
                        chess.engine.Limit(time=0.5),
                        multipv=5
                    )
                    info_list = infos if isinstance(infos, list) else [infos]
                    for info in info_list:
                        if info.get("pv"):
                            cp = info["score"].white().score(mate_score=10000)
                            candidates.append({
                                "move": info["pv"][0].uci(),
                                "eval_pawns": round(cp / 100, 2) if cp is not None else 0
                            })
            except Exception:
                pass  # candidates are optional, don't break the move if this fails

            score_cp = int(candidates[0]["eval_pawns"] * 100) if candidates else 0

            board.push(move)
            return {
                "move": move.uci(),
                "fen": board.fen(),
                "is_game_over": board.is_game_over(),
                "outcome": str(board.outcome()) if board.is_game_over() else None,
                "score_cp": score_cp,
                "eval_pawns": round(score_cp / 100, 2),
                "candidates": candidates
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


# ΓöÇΓöÇ Helpers ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

# ΓöÇΓöÇΓöÇ ELO calculation ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def calc_elo(my_elo: int, opp_elo: int, my_color: str, winner_color: str) -> int:
    diff  = my_elo - opp_elo   # positive = I am higher rated
    tiers = int(diff / 100)    # truncate toward zero
    base  = 7

    if winner_color == 'draw':
        return my_elo

    if winner_color == my_color:
        # I won — gain less if I was favored (higher), more if underdog (lower)
        if tiers >= 0:
            # I am higher rated — expected win — gain less
            effect = base * (0.5 ** tiers)
        else:
            # I am lower rated — upset win — gain more
            effect = base * (1.5 ** abs(tiers))
        effect = max(1, round(effect))
    else:
        # I lost — lose less if I was underdog (lower), more if favored (higher)
        if tiers <= 0:
            # I am lower rated — expected loss — lose less
            effect = base * (0.5 ** abs(tiers))
        else:
            # I am higher rated — upset loss — lose more
            effect = base * (1.5 ** tiers)
        effect = -max(1, round(effect))

    return max(100, my_elo + int(effect))

async def cleanup_game(game_id: str, delay: int = 10):
    await asyncio.sleep(delay)
    active_games.pop(game_id, None)


def new_game(game_id: str, white_ws: WebSocket, black_ws: WebSocket,
             white_id: str, black_id: str) -> dict:
    return {
        "id":           game_id,
        "board":        chess.Board(),
        "white_ws":     white_ws,
        "black_ws":     black_ws,
        "white_game_ws": None,
        "black_game_ws": None,
        "white_id":     white_id,
        "black_id":     black_id,
        "clock":        {"w": CLOCK_SECONDS, "b": CLOCK_SECONDS},
        "last_move_ts": time.time(),
        "over":         False,
        # Profile data filled in when players send 'profile' message
        "white_profile": None,   # {username, elo, user_id}
        "black_profile": None,
    }


async def send(ws: WebSocket, msg: dict):
    """Safe send ΓÇö ignores errors if socket already closed."""
    try:
        await ws.send_json(msg)
    except Exception:
        pass


async def broadcast(game: dict, msg: dict):
    await send(game["white_ws"], msg)
    await send(game["black_ws"], msg)


def deduct_clock(game: dict) -> float:
    """Deduct elapsed time from the side that just moved, return remaining."""
    now = time.time()
    elapsed = now - game["last_move_ts"]
    # The side that just moved is the OPPOSITE of board.turn (move already pushed)
    just_moved = "b" if game["board"].turn == chess.WHITE else "w"
    game["clock"][just_moved] = max(0, game["clock"][just_moved] - elapsed)
    game["last_move_ts"] = now
    return game["clock"][just_moved]


async def clock_loop(game_id: str):
    """Background task ΓÇö checks for flag fall every second."""
    while True:
        await asyncio.sleep(1)
        game = active_games.get(game_id)
        if not game or game["over"]:
            return

        now = time.time()
        elapsed = now - game["last_move_ts"]
        turn = "w" if game["board"].turn == chess.WHITE else "b"
        remaining = game["clock"][turn] - elapsed

        if remaining <= 0:
            game["over"] = True
            loser = "white" if turn == "w" else "black"
            winner = "black" if loser == "white" else "white"
            await broadcast(game, {
                "type":   "gameover",
                "result": winner,
                "reason": "timeout",
                "clock":  game["clock"],
            })
            await update_elos(game, winner)
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
    Calculate and persist ELO changes for both players.
    Only runs if both players have profiles with user_ids.
    Sends elo_update message to each player.
    """
    wp = game.get("white_profile")
    bp = game.get("black_profile")

    # Both must be logged-in users with ELO
    if not wp or not bp:
        return
    if not wp.get("user_id") or not bp.get("user_id"):
        return
    if wp.get("elo") is None or bp.get("elo") is None:
        return

    w_elo_old = wp["elo"]
    b_elo_old = bp["elo"]

    w_elo_new = calc_elo(w_elo_old, b_elo_old, "white", result)
    b_elo_new = calc_elo(b_elo_old, w_elo_old, "black", result)

    # Persist to Supabase
    await supabase_update_elo(wp["user_id"], w_elo_new)
    await supabase_update_elo(bp["user_id"], b_elo_new)

    # Notify players
    await send(game["white_ws"], {
        "type":    "elo_update",
        "old_elo": w_elo_old,
        "new_elo": w_elo_new,
    })
    await send(game["black_ws"], {
        "type":    "elo_update",
        "old_elo": b_elo_old,
        "new_elo": b_elo_new,
    })


# ΓöÇΓöÇ WebSocket: Lobby (matchmaking) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

@app.websocket("/ws/lobby")
async def lobby(ws: WebSocket):
    await ws.accept()
    guest_id = "guest_" + uuid.uuid4().hex[:8]

    await send(ws, {"type": "waiting", "guest_id": guest_id})

    # Check if someone is already waiting
    if lobby_queue:
        opponent = lobby_queue.pop(0)
        game_id  = uuid.uuid4().hex[:12]

        # Randomly assign colors (first in queue gets white)
        white_ws, white_id = opponent["ws"], opponent["guest_id"]
        black_ws, black_id = ws, guest_id

        game = new_game(game_id, white_ws, black_ws, white_id, black_id)
        active_games[game_id] = game

        await send(white_ws, {
            "type":    "matched",
            "game_id": game_id,
            "color":   "white",
            "opponent": black_id,
        })
        await send(black_ws, {
            "type":    "matched",
            "game_id": game_id,
            "color":   "black",
            "opponent": white_id,
        })

        # Start clock loop
        asyncio.create_task(clock_loop(game_id))
    else:
        lobby_queue.append({"ws": ws, "guest_id": guest_id})

    # Keep lobby socket alive until matched or disconnected
    try:
        while True:
            await ws.receive_text()   # just keep connection open
    except WebSocketDisconnect:
        # Remove from queue if still waiting
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
        await broadcast(game, {"type": "both_connected"})

    try:
        while True:
            data = await ws.receive_json()

            msg_type = data.get("type")

            if game["over"]:
                if msg_type not in ("rematch_offer", "rematch_accept", "rematch_decline", "ping"):
                    await send(ws, {"type": "error", "detail": "Game is over."})
                    continue


            # ΓöÇΓöÇ Keepalive ping (ignore) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
            if msg_type == "ping":
                continue

            # ΓöÇΓöÇ Profile (sent on connect) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
            if msg_type == "profile":
                profile = {
                    "username": data.get("username", "guest"),
                    "elo":      data.get("elo"),
                    "user_id":  data.get("user_id"),
                }
                if color == "w":
                    game["white_profile"] = profile
                    await send(game["black_ws"], {
                        "type":     "opponent_profile",
                        "username": profile["username"],
                        "elo":      profile["elo"],
                    })
                else:
                    game["black_profile"] = profile
                    await send(game["white_ws"], {
                        "type":     "opponent_profile",
                        "username": profile["username"],
                        "elo":      profile["elo"],
                    })
                continue

            # ΓöÇΓöÇ Move ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
            if msg_type == "move":
                # Only the player whose turn it is can move
                expected = "w" if game["board"].turn == chess.WHITE else "b"
                if color != expected:
                    await send(ws, {"type": "error", "detail": "Not your turn."})
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
                await update_elos(game, winner)
                # Keep game alive briefly for rematch negotiation
                asyncio.create_task(cleanup_game(game_id, delay=10))

            # ΓöÇΓöÇ Draw offer (future) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
            elif msg_type == "draw_offer":
                opponent_ws = game["black_ws"] if color == "w" else game["white_ws"]
                await send(opponent_ws, {"type": "draw_offer"})

            elif msg_type == "draw_accept":
                game["over"] = True
                await broadcast(game, {
                    "type":   "gameover",
                    "result": "draw",
                    "reason": "agreement",
                    "clock":  game["clock"],
                })
                await update_elos(game, "draw")
                # Keep game alive briefly for rematch negotiation
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
                active_games[new_game_id] = ng

                # Notify players — colors are swapped
                # old black → new white; old white → new black
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
            # Send directly to opponent ΓÇö broadcast may fail if our socket is dead
            await send(opponent_ws, {
                "type":   "gameover",
                "result": winner,
                "reason": "disconnect",
                "clock":  game["clock"],
            })
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

FREE_COACH_LIMIT = 5  # uses per day

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

    if uses_today >= FREE_COACH_LIMIT:
        raise HTTPException(429, f"Daily limit of {FREE_COACH_LIMIT} coach uses reached. Upgrade for unlimited access.")

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
        "uses_remaining": FREE_COACH_LIMIT - (uses_today + 1),
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
        "username": profile.get("username"),
        "elo": profile.get("elo", 1500),
        "created_at": profile.get("created_at"),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "total": total,
        "win_rate": round(wins / total * 100) if total else 0,
        "recent_games": games[:10],
        "elo_history": elo_history[-20:],
        "coach_uses_today": uses_today,
        "coach_uses_remaining": FREE_COACH_LIMIT - uses_today,
        "coach_limit": FREE_COACH_LIMIT,
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

@app.get("/favicon.ico")
def favicon():
    return FileResponse("logo.png")

@app.get("/history")
def history():
    return FileResponse("history.html")

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

