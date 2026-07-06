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

@router.get("/senkabala.wasm")
async def serve_wasm():
    from fastapi.responses import FileResponse
    return FileResponse("senkabala.wasm", media_type="application/wasm")

@router.get("/senkabala.js")
async def serve_wasm_js():
    from fastapi.responses import FileResponse
    return FileResponse("senkabala.js", media_type="application/javascript")

@router.get("/senkabala_wasm.js")
async def serve_wasm_wrapper():
    from fastapi.responses import FileResponse
    return FileResponse("senkabala_wasm.js", media_type="application/javascript")

@router.get("/engine_worker.js")
async def serve_engine_worker():
    from fastapi.responses import FileResponse
    return FileResponse("engine_worker.js", media_type="application/javascript")

@router.get("/book.bin")
def serve_book():
    import os
    if not os.path.exists("book.bin"):
        raise HTTPException(status_code=404, detail="Opening book not found")
    return FileResponse("book.bin", media_type="application/octet-stream")

@router.get("/openings.js")
def serve_openings():
    return FileResponse("openings.js", media_type="application/javascript")

@router.get("/openings_detector.js")
def serve_openings_detector():
    return FileResponse("openings_detector.js", media_type="application/javascript")

@router.get("/analysis_accuracy.js")
def serve_analysis_accuracy():
    return FileResponse("analysis_accuracy.js", media_type="application/javascript")

@router.get("/aac_strings.js")
def serve_aac_strings():
    return FileResponse("aac_strings.js", media_type="application/javascript")

@router.get("/aac.js")
def serve_aac():
    return FileResponse("aac.js", media_type="application/javascript")

@router.get("/aac_grid.js")
def serve_aac_grid():
    return FileResponse("aac_grid.js", media_type="application/javascript")

@router.get("/logo.png")
def logo():
    return FileResponse("logo.png")

@router.get("/favicon.ico")
def favicon():
    return FileResponse("favicon.ico")

@router.get("/favicon_16x16.png")
def favicon16():
    return FileResponse("favicon_16x16.png")

@router.get("/favicon_32x32.png")
def favicon32():
    return FileResponse("favicon_32x32.png")

@router.get("/apple-touch-icon.png")
def apple_touch():
    return FileResponse("apple-touch-icon.png")

@router.get("/android-chrome-192x192.png")
def android192():
    return FileResponse("android-chrome-192x192.png")

@router.get("/android-chrome-512x512.png")
def android512():
    return FileResponse("android-chrome-512x512.png")

@router.get("/site.webmanifest")
def webmanifest():
    return FileResponse("site.webmanifest", media_type="application/manifest+json")

@router.get("/sitemap.xml")
def sitemap():
    return FileResponse("sitemap.xml", media_type="application/xml")

@router.get("/chessboard-1.0.0.min.css")
def cb_css():
    return FileResponse("chessboard-1.0.0.min.css")

@router.get("/chessboard-1.0.0.min.js")
def cb_js():
    return FileResponse("chessboard-1.0.0.min.js")

@router.get("/jquery.min.js")
def jquery():
    return FileResponse("jquery.min.js")

@router.get("/chess.min.js")
def chess_js():
    return FileResponse("chess.min.js")
