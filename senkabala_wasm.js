/**
 * senkabala_wasm.js
 *
 * Compiles senkabala.wasm once on the main thread using streaming compilation
 * (fastest possible — browser can compile while downloading), then transfers
 * the compiled WebAssembly.Module to the worker. The worker only needs to
 * instantiate it (milliseconds) rather than recompile (minutes).
 */

class SenkabalaEngine {
  constructor() {
    this._worker  = null;
    this._ready   = false;
    this._pending = new Map();
    this._idSeq   = 0;
    this._readyWaiters = [];

    this._start();
  }

  async _start() {
    try {
      // Step 1: compile WASM on the main thread (streaming = fastest)
      const wasmModule = await WebAssembly.compileStreaming(
        fetch('/senkabala.wasm')
      );

      // Step 2: spin up the worker
      this._worker = new Worker('/engine_worker.js');
      this._worker.onmessage = (e) => this._onMessage(e.data);
      this._worker.onerror   = (err) => {
        console.error('[SenkabalaWASM] Worker error:', err.message);
        for (const { reject } of this._pending.values()) {
          reject(new Error(err.message));
        }
        this._pending.clear();
      };

      // Step 3: send the compiled module to the worker (transferable)
      this._worker.postMessage({ type: 'module', wasmModule });

    } catch(e) {
      console.warn('[SenkabalaWASM] Failed to start:', e);
      this._worker = null;
    }
  }

  _onMessage(msg) {
    if (msg.type === 'ready') {
      this._ready = true;
      console.log('[SenkabalaWASM] Engine ready');
      this._readyWaiters.forEach(fn => fn());
      this._readyWaiters = [];
      return;
    }

    if (msg.type === 'info') {
      // Uncomment to show search depth in console:
      // console.log(`[engine] depth ${msg.depth} time ${msg.time}ms pv ${msg.pv}`);
      return;
    }

    if (msg.type === 'result' || msg.type === 'error') {
      const p = this._pending.get(msg.id);
      if (!p) return;
      this._pending.delete(msg.id);
      if (msg.type === 'result') p.resolve(msg.move);
      else p.reject(new Error(msg.message));
    }
  }

  ready() {
    if (this._ready) return Promise.resolve();
    if (!this._worker) return Promise.reject(new Error('Engine unavailable'));
    return new Promise((resolve) => {
      this._readyWaiters.push(resolve);
    });
  }

  bestMove(fen, moves = [], movetime = 1000) {
    if (!this._worker) return Promise.reject(new Error('Worker not available'));

    if (!this._ready) {
      return this.ready().then(() => this.bestMove(fen, moves, movetime));
    }

    const id = ++this._idSeq;
    // Grace = 3s for WASM (runs locally, no network round-trip).
    // If worker doesn't respond, reject so caller can fall back to server.
    const grace = movetime > 2000 ? 10000 : 3000;
    const timeoutMs = movetime + grace;

    return new Promise((resolve, reject) => {
      // Build timer-aware handlers BEFORE registering in _pending
      // so there is no window where the worker can resolve with the wrong handler.
      let timer;
      const wrappedResolve = (v) => { clearTimeout(timer); resolve(v); };
      const wrappedReject  = (e) => { clearTimeout(timer); reject(e);  };

      this._pending.set(id, { resolve: wrappedResolve, reject: wrappedReject });
      this._worker.postMessage({ type: 'search', id, fen, moves, movetime_ms: movetime });

      timer = setTimeout(() => {
        if (this._pending.has(id)) {
          this._pending.delete(id);
          console.warn(`[WASM] move timeout after ${timeoutMs}ms — falling back to server`);
          reject(new Error('WASM timeout'));
        }
      }, timeoutMs);
    });
  }

  newGame() {
    // TT persists across games in the worker — intentional (helps with opening book)
  }
}

// Must exactly match server.py DIFFICULTY_SETTINGS
const WASM_DIFFICULTY_MS = {
  1:   200,
  2:   200,
  3:   500,
  4:  1000,
  5:  2000,
  6:  4000,
  7:  8000,
  8: 15000,
};

// Must exactly match server.py random_chance
const WASM_RANDOM_CHANCE = {
  1: 0.90,
  2: 0.65,
  3: 0.25,
  4: 0.00,
  5: 0.00,
  6: 0.00,
  7: 0.00,
  8: 0.00,
};

async function wasmGetMove({ fen, moves, difficulty, engine, chess }) {
  const randomChance = WASM_RANDOM_CHANCE[difficulty] ?? 0;

  if (Math.random() < randomChance) {
    const legal = chess.moves({ verbose: true });
    if (legal.length === 0) return null;
    const pick = legal[Math.floor(Math.random() * legal.length)];
    return pick.from + pick.to + (pick.promotion || '');
  }

  const ms = WASM_DIFFICULTY_MS[difficulty] ?? 1000;
  return engine.bestMove(fen, moves, ms);
}

if (typeof module !== 'undefined') {
  module.exports = { SenkabalaEngine, wasmGetMove, WASM_DIFFICULTY_MS, WASM_RANDOM_CHANCE };
}
