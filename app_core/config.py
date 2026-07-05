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

ENGINE_PATH = "./engines/engine.exe" if os.name == "nt" else "./engines/engine"

ANTHROPIC_API_KEY    = os.getenv("ANTHROPIC_API_KEY", "your-key-here")

SUPABASE_URL         = os.getenv("SUPABASE_URL", "https://nbskgzsvygdmlvwbetxn.supabase.co")

SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")  # set on Railway

MEDAL_TIERS = {
    # Awarded for 1st place wins — tier escalates with total wins
    "bronze":   {"label": "Bronze Lion",   "img": "bronze"},
    "silver":   {"label": "Silver Lion",   "img": "silver"},
    "gold":     {"label": "Gold Lion",     "img": "gold"},
    "platinum": {"label": "Platinum Lion", "img": "platinum"},
    "diamond":  {"label": "Diamond Lion",  "img": "diamond"},
}

LS_SIGNING_SECRET  = os.getenv("LEMONSQUEEZY_SIGNING_SECRET", "")

LS_CLUB_VARIANT    = int(os.getenv("LS_CLUB_VARIANT_ID", "1667817"))

LS_PRO_VARIANT     = int(os.getenv("LS_PRO_VARIANT_ID",  "1667860"))

COACH_LIMITS = {"free": 10, "club": 200, "pro": 999999}

TIER_LIMITS = {
    "free": 10,
    "club": 200,
    "pro": 999999
}

DIFFICULTY_SETTINGS = {
    # Levels 1–3: think time is irrelevant — weakness comes from random_chance below
    # Levels 4–8: pure engine, increasing think time gives more depth = stronger play
    1: 200,    # Beginner     — mostly random moves
    2: 200,    # Beginner+    — mostly random moves
    3: 500,    # Easy         — occasional best move
    4: 1000,   # Intermediate — full engine, shallow search
    5: 2000,   # Hard         — full engine, 2s
    6: 4000,   # Hard+        — full engine, 4s
    7: 8000,   # Expert       — full engine, 8s
    8: 15000,  # Master       — full engine, 15s (~2050 ELO)
}

POOL_SIZE = int(os.getenv("ENGINE_POOL_SIZE", "4"))  # tune via Railway env var

CLOCK_SECONDS = 300             # 5 minutes each side

_GLICKO_SCALE = 173.7178

_GLICKO_TAU   = 0.5      # volatility change constraint (Lichess default)

_GLICKO_EPS   = 1e-6     # convergence tolerance

FREE_COACH_LIMIT = 10  # free tier daily limit (kept for legacy refs)

_RESERVED_NAMES = {
    'africhess', 'admin', 'administrator', 'moderator', 'mod',
    'staff', 'support', 'official', 'senkabala', 'system',
    'root', 'superuser', 'owner', 'operator', 'bot',
}

_ADMIN_USER_IDS: set[str] = set(filter(None, os.getenv("ADMIN_USER_IDS", "").split(",")))

ADMIN_USER_IDS = set(os.getenv("ADMIN_USER_IDS", "9c51d331-8eba-4da5-b644-64cd4fc168d1").split(","))

REGIONS: dict[str, list[str]] = {
    "east_africa":    ["UG","KE","TZ","RW","BI","ET","SS","SO","ER","DJ","SD"],
    "west_africa":    ["NG","GH","SN","CI","CM","BJ","TG","GN","GW","SL","LR","GM","MR","CV","NE","BF","ML"],
    "north_africa":   ["EG","LY","TN","DZ","MA"],
    "south_africa":   ["ZA","ZW","ZM","MW","MZ","NA","BW","LS","SZ","AO","MG","MU","SC","KM","ST"],
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

REASSURE_PROMPTS = [
    "You are a warm, loving companion writing to a beautiful Pakistani girl named Alina (sometimes called Lina or Luna). She needs reassurance right now. Write her a sweet, genuine, heartfelt message (3-5 sentences) that uses her name naturally and makes her feel truly seen, loved, and enough. Be specific, warm, and avoid clichés. Vary the message each time.",
    "You are the most supportive presence in Alina's life — a beautiful Pakistani girl who sometimes goes by Lina or Luna. She needs to hear something kind today. Write her a tender, uplifting message (3-5 sentences) about how wonderful she is. Use her name warmly and make her feel like the entire universe is rooting for her.",
    "Alina — a stunning Pakistani girl also lovingly called Lina or Luna — needs a big emotional hug right now. Write her a sweet, comforting message (3-5 sentences) full of warmth and sincerity. Use her name at least once. Make her feel safe, cherished, and deeply loved. Be playful but genuine.",
    "Write a short, sweet reassurance note (3-5 sentences) for Alina, a beautiful Pakistani girl whose nicknames are Lina and Luna. She's doubting herself and needs to hear how amazing she truly is. Be specific, heartfelt, and make her smile. Use her name naturally.",
    "Alina (also called Lina or Luna) is a gorgeous Pakistani girl who needs some love right now. Write her a warm, poetic little message (3-5 sentences) that celebrates who she is — her beauty, her heart, her strength. Use her name and make it feel personal and real, not generic.",
]

REASSURE_DAILY_LIMIT = 20
