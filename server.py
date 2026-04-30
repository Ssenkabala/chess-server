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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ENGINE_PATH = "./engines/engine.exe" if os.name == "nt" else "./engines/engine"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "your-key-here")

# ─── Database setup ───────────────────────────────────────────────────────────

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

# ─── Models ───────────────────────────────────────────────────────────────────

class MoveRequest(BaseModel):
    fen: str
    think_time: float = 1.0

class CoachRequest(BaseModel):
    fen: str
    played_move: Optional[str] = None   # UCI format e.g. "e2e4"
    pgn: Optional[str] = None
    lesson_type: Optional[str] = None   # "opening", "middlegame", "endgame"
    think_time: float = 1.0

class RegisterRequest(BaseModel):
    email: str
    tier: str = "free"  # set to "club"/"pro" after Stripe confirms payment

# ─── Auth helper ──────────────────────────────────────────────────────────────

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

# ─── Engine helper ────────────────────────────────────────────────────────────

def analyse_position(fen: str, think_time: float):
    """Returns best_move (UCI), score in centipawns, and top PV moves."""
    board = chess.Board(fen)
    with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
        info = engine.analyse(
            board,
            chess.engine.Limit(time=think_time)
            # removed multipv=3 since your engine doesn't support it
        )
        best_move = info["pv"][0].uci() if info.get("pv") else None
        score = info["score"].white().score(mate_score=10000)  # centipawns
        pv_moves = [m.uci() for m in info.get("pv", [])[:5]]

    return {"best_move": best_move, "score_cp": score, "pv": pv_moves}

# ─── Original /move endpoint (unchanged) ──────────────────────────────────────

@app.post("/move")
def get_move(req: MoveRequest):
    board = chess.Board(req.fen)
    with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
        info = engine.analyse(board, chess.engine.Limit(time=req.think_time))
        best_move = info["pv"][0] if info.get("pv") else None
        if best_move is None:
            raise HTTPException(status_code=500, detail="Engine returned no move")
        score_cp = None
        if "score" in info:
            score_cp = info["score"].white().score(mate_score=10000)
        board.push(best_move)
    return {
        "move": best_move.uci(),
        "fen": board.fen(),
        "is_game_over": board.is_game_over(),
        "outcome": str(board.outcome()) if board.is_game_over() else None,
        "score_cp": score_cp,
        "eval_pawns": round(score_cp / 100, 2) if score_cp is not None else 0
    }

# ─── /coach endpoint ──────────────────────────────────────────────────────────

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

# ─── /register endpoint (call this from your Stripe webhook) ─────────────────

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

# ─── Health / static ──────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}

app.mount("/img", StaticFiles(directory="img"), name="img")
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def root():
    return FileResponse("index.html")

@app.get("/landing")
def landing():
    return FileResponse("landing.html")

@app.get("/logo.png")
def logo():
    return FileResponse("logo.png")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)