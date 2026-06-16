"""
AfriChess Load Test Suite
Tests the actual endpoints that matter during a tournament:
  1. HTTP benchmark  — landing page, static assets
  2. /move endpoint  — engine moves under concurrent load
  3. WebSocket       — lobby + game WS connections
  4. Simulated game  — full request flow per player

Usage:
  pip install aiohttp
  python africhess_loadtest.py [--target https://africhess.org] [--mode http|move|ws|full]
"""

import asyncio, time, random, argparse, json, sys
import aiohttp

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL  = "https://africhess.org"
WS_BASE   = "wss://africhess.org"

# Realistic game positions for /move testing
TEST_POSITIONS = [
    {"fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "moves": "e2e4"},
    {"fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", "moves": "e2e4 e7e5"},
    {"fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "moves": "e2e4 e7e5 g1f3 b8c6"},
    {"fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "moves": "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6"},
    {"fen": "5K1k/6pp/4Q2n/8/7N/8/8/8 w - - 0 1", "moves": ""},  # endgame
]

# ── HTTP Benchmark ─────────────────────────────────────────────────────────────
async def http_request(session, url, semaphore, req_id):
    async with semaphore:
        t = time.perf_counter()
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                await r.read()
                return {"id": req_id, "status": r.status, "ms": (time.perf_counter()-t)*1000, "ok": r.status < 400}
        except Exception as e:
            return {"id": req_id, "status": type(e).__name__, "ms": (time.perf_counter()-t)*1000, "ok": False}

async def bench_http(target, total=200, concurrency=50):
    print(f"\n{'='*55}")
    print(f"  HTTP BENCHMARK  →  {target}")
    print(f"  {total} requests | {concurrency} concurrent")
    print(f"{'='*55}")

    sem = asyncio.Semaphore(concurrency)
    conn = aiohttp.TCPConnector(limit=None, ttl_dns_cache=300)
    t0 = time.perf_counter()

    async with aiohttp.ClientSession(connector=conn) as sess:
        results = await asyncio.gather(*[
            http_request(sess, target, sem, i) for i in range(total)
        ])

    dur = time.perf_counter() - t0
    _report(results, dur, total)

# ── /move Benchmark ────────────────────────────────────────────────────────────
async def move_request(session, base, semaphore, req_id):
    pos = random.choice(TEST_POSITIONS)
    # Must match MoveRequest model: fen, think_time, difficulty, moves
    payload = {
        "fen":        pos["fen"],
        "moves":      pos["moves"].split() if pos["moves"] else [],
        "difficulty": random.choice([3, 4, 5]),  # Easy-Hard for load testing
        "think_time": 1.0,
    }
    async with semaphore:
        t = time.perf_counter()
        try:
            async with session.post(
                f"{base}/move",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as r:
                data = await r.json()
                ms = (time.perf_counter()-t)*1000
                ok = r.status == 200 and data.get("move") is not None
                return {"id": req_id, "status": r.status, "ms": ms, "ok": ok,
                        "move": data.get("move","?"), "detail": data.get("detail","")}
        except Exception as e:
            return {"id": req_id, "status": type(e).__name__,
                    "ms": (time.perf_counter()-t)*1000, "ok": False, "move": "", "detail": str(e)}

async def bench_move(target, total=100, concurrency=20):
    print(f"\n{'='*55}")
    print(f"  /move ENDPOINT BENCHMARK  →  {target}/move")
    print(f"  {total} requests | {concurrency} concurrent")
    print(f"  (simulates {concurrency} simultaneous games)")
    print(f"{'='*55}")

    sem = asyncio.Semaphore(concurrency)
    conn = aiohttp.TCPConnector(limit=None)
    t0 = time.perf_counter()

    async with aiohttp.ClientSession(connector=conn) as sess:
        results = await asyncio.gather(*[
            move_request(sess, target, sem, i) for i in range(total)
        ])

    dur = time.perf_counter() - t0
    _report(results, dur, total)
    moves = [r.get("move","?") for r in results if r["ok"]]
    if moves:
        print(f"  Sample moves: {moves[:5]}")
    else:
        # Show why moves failed
        bad = [r for r in results if not r["ok"]]
        if bad:
            print(f"  First failure: status={bad[0]['status']} detail={bad[0].get('detail','')[:80]}")

# ── WebSocket Benchmark ────────────────────────────────────────────────────────
async def ws_connection(session, ws_url, semaphore, conn_id):
    async with semaphore:
        t = time.perf_counter()
        try:
            async with session.ws_connect(
                ws_url,
                timeout=aiohttp.ClientTimeout(total=10),
                heartbeat=5.0
            ) as ws:
                # Send lobby ping — simulates player connecting to matchmaking
                await ws.send_str(json.dumps({"type": "ping"}))
                msg = await asyncio.wait_for(ws.receive(), timeout=5.0)
                ms = (time.perf_counter()-t)*1000
                return {"id": conn_id, "status": msg.type.name, "ms": ms, "ok": True}
        except asyncio.TimeoutError:
            return {"id": conn_id, "status": "TIMEOUT", "ms": (time.perf_counter()-t)*1000, "ok": False}
        except Exception as e:
            return {"id": conn_id, "status": type(e).__name__, "ms": (time.perf_counter()-t)*1000, "ok": False}

async def bench_ws(target, total=100, concurrency=30):
    ws_url = target.replace("https://", "wss://").replace("http://", "ws://") + "/ws/lobby"
    print(f"\n{'='*55}")
    print(f"  WEBSOCKET BENCHMARK  →  {ws_url}")
    print(f"  {total} connections | {concurrency} concurrent")
    print(f"{'='*55}")

    sem = asyncio.Semaphore(concurrency)
    conn = aiohttp.TCPConnector(limit=None)
    t0 = time.perf_counter()

    async with aiohttp.ClientSession(connector=conn) as sess:
        results = await asyncio.gather(*[
            ws_connection(sess, ws_url, sem, i) for i in range(total)
        ])

    dur = time.perf_counter() - t0
    _report(results, dur, total)

# ── Full Tournament Simulation ─────────────────────────────────────────────────
async def simulate_player(session, base, player_id, sem_http, sem_ws):
    """
    Simulates one player's full session:
    1. Load landing page
    2. Connect to lobby WS
    3. Fire 3-5 /move requests (simulating a game)
    """
    results = []
    t_total = time.perf_counter()

    # Step 1: page load
    r = await http_request(session, base, sem_http, player_id)
    results.append(("page_load", r["ok"], r["ms"]))

    # Step 2: lobby WS ping
    ws_url = base.replace("https://","wss://").replace("http://","ws://") + "/ws/lobby"
    r2 = await ws_connection(session, ws_url, sem_ws, player_id)
    results.append(("lobby_ws", r2["ok"], r2["ms"]))

    # Step 3: 3-5 moves
    for _ in range(random.randint(3, 5)):
        r3 = await move_request(session, base, sem_http, player_id)
        results.append(("move", r3["ok"], r3["ms"]))
        await asyncio.sleep(random.uniform(0.5, 2.0))  # think time

    total_ms = (time.perf_counter()-t_total)*1000
    ok = all(r[1] for r in results)
    return {"player": player_id, "ok": ok, "total_ms": total_ms, "steps": results}

async def bench_tournament(target, players=30):
    print(f"\n{'='*55}")
    print(f"  TOURNAMENT SIMULATION  →  {target}")
    print(f"  {players} simultaneous players, full session each")
    print(f"{'='*55}")

    sem_http = asyncio.Semaphore(players)
    sem_ws   = asyncio.Semaphore(players)
    conn = aiohttp.TCPConnector(limit=None)
    t0 = time.perf_counter()

    async with aiohttp.ClientSession(connector=conn) as sess:
        results = await asyncio.gather(*[
            simulate_player(sess, target, i, sem_http, sem_ws)
            for i in range(players)
        ])

    dur = time.perf_counter() - t0
    ok_players   = sum(1 for r in results if r["ok"])
    avg_sess_ms  = sum(r["total_ms"] for r in results) / len(results)
    step_results = [s for r in results for s in r["steps"]]
    by_type = {}
    for name, ok, ms in step_results:
        if name not in by_type: by_type[name] = []
        by_type[name].append((ok, ms))

    print(f"\n  Players completed:    {ok_players} / {players}")
    print(f"  Total wall time:      {dur:.2f}s")
    print(f"  Avg session time:     {avg_sess_ms:.0f}ms")
    print(f"\n  Step breakdown:")
    for name, vals in by_type.items():
        ok_cnt = sum(1 for o,_ in vals if o)
        avg_ms = sum(m for _,m in vals) / len(vals)
        print(f"    {name:15} ok={ok_cnt}/{len(vals)}  avg={avg_ms:.0f}ms")

# ── Report helper ──────────────────────────────────────────────────────────────
def _report(results, dur, total):
    ok  = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    lats = sorted([r["ms"] for r in ok])
    avg = sum(lats)/len(lats) if lats else 0
    p95 = lats[int(len(lats)*0.95)] if lats else 0
    p99 = lats[int(len(lats)*0.99)] if lats else 0

    status_dist = {}
    for r in results:
        status_dist[r["status"]] = status_dist.get(r["status"], 0) + 1

    print(f"\n  Wall time:      {dur:.2f}s")
    print(f"  Throughput:     {total/dur:.1f} req/s")
    print(f"  Success:        {len(ok)}/{total}  ({100*len(ok)/total:.1f}%)")
    print(f"  Failed:         {len(bad)}/{total}")
    if lats:
        print(f"\n  Latency (ms):")
        print(f"    avg = {avg:.1f}   min = {lats[0]:.1f}   max = {lats[-1]:.1f}")
        print(f"    p95 = {p95:.1f}   p99 = {p99:.1f}")
    print(f"\n  Status codes:   {status_dist}")

    # Warning thresholds
    if len(ok)/total < 0.95:
        print(f"\n  ⚠️  WARNING: {100*(1-len(ok)/total):.1f}% failure rate — investigate before tournament")
    elif avg > 2000:
        print(f"\n  ⚠️  WARNING: avg latency {avg:.0f}ms is high — engine pool may be undersized")
    else:
        print(f"\n  ✅ Looks healthy")

# ── Entry point ────────────────────────────────────────────────────────────────
async def run(args):
    target = args.target.rstrip("/")
    mode   = args.mode

    if mode in ("http", "all"):
        await bench_http(target, total=args.requests, concurrency=args.concurrency)
    if mode in ("move", "all"):
        await bench_move(target, total=min(args.requests, 100), concurrency=min(args.concurrency, 20))
    if mode in ("ws", "all"):
        await bench_ws(target, total=min(args.requests, 100), concurrency=min(args.concurrency, 30))
    if mode in ("tournament", "all"):
        await bench_tournament(target, players=args.players)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AfriChess Load Test Suite")
    parser.add_argument("--target",      default=BASE_URL,         help="Base URL to test")
    parser.add_argument("--mode",        default="all",            choices=["http","move","ws","tournament","all"])
    parser.add_argument("--requests",    default=200, type=int,    help="Total HTTP requests (http/move mode)")
    parser.add_argument("--concurrency", default=30,  type=int,    help="Max concurrent connections")
    parser.add_argument("--players",     default=20,  type=int,    help="Simulated players (tournament mode)")
    args = parser.parse_args()

    print(f"\n🌍 AfriChess Load Test — {args.target}")
    print(f"   Mode: {args.mode} | {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    asyncio.run(run(args))
