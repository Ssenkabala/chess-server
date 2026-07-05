"""
server.py — slim entrypoint. Creates the FastAPI app, wires middleware and the
startup lifecycle, then mounts every domain's router. Each router is a
"standalone server path" — a fix to tournament logic literally cannot touch
game, profile, or any other domain's file.
"""
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app_core import db as _db  # runs init_db() at import time, as before
from app_core.state import _presence  # noqa: F401 (touched to confirm shared state loads)

class EndpointFilter(logging.Filter):
    ALWAYS_SUPPRESS = {"/api/ping", "/api/stats", "/api/leaderboard", "/api/notification"}
    def filter(self, record):
        msg = record.getMessage()
        return not any(p in msg for p in self.ALWAYS_SUPPRESS)

logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

async def _warm_stats_cache():
    pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    from routes.tournament import arena_auto_start_scheduler
    import asyncio
    asyncio.create_task(arena_auto_start_scheduler())
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class TimeoutMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        return await call_next(request)

app.add_middleware(TimeoutMiddleware)

from routes import static_assets, pages, game, tournament, profile, coach, misc, payments
for r in (static_assets, pages, game, tournament, profile, coach, misc, payments):
    app.include_router(r.router)

app.mount("/img", StaticFiles(directory="img"), name="img")
app.mount("/static", StaticFiles(directory="."), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
