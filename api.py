# server.py
import chess
import chess.engine
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ENGINE_PATH = "./engines/engine.exe"

class MoveRequest(BaseModel):
    fen: str
    think_time: float = 1.0  # seconds

@app.post("/move")
def get_move(req: MoveRequest):
    board = chess.Board(req.fen)
    
    with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
        result = engine.play(board, chess.engine.Limit(time=req.think_time))
        move = result.move.uci()
    
    board.push(result.move)
    
    return {
        "move": move,
        "fen": board.fen(),
        "is_game_over": board.is_game_over(),
        "outcome": str(board.outcome()) if board.is_game_over() else None
    }

@app.get("/health")
def health():
    return {"status": "ok"}