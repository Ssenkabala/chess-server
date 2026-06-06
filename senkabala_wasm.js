/**
 * senkabala_wasm.js — Main-thread interface to SenkabalaIII WASM engine
 *
 * Uses a WebWorker so the engine runs in a background thread:
 *   - UI never freezes during long searches
 *   - Browser watchdog never fires
 *   - Multiple searches can be queued
 *
 * Usage:
 *   const engine = new SenkabalaEngine();
 *   await engine.ready();
 *   const move = await engine.bestMove(fen, moves, movetime_ms);
 */

class SenkabalaEngine {
  constructor() {
    this._worker   = null;
    this._ready    = false;
    this._pending  = new Map();   // id → { resolve, reject }
    this._idSeq    = 0;
    this._readyWaiters = [];

    this._init();
  }

  _init() {
    try {
      this._worker = new Worker('/engine_worker.js');
      this._worker.onmessage = (e) => this._onMessage(e.data);
      this._worker.onerror   = (e) => {
        console.error('[SenkabalaWASM] Worker error:', e.message);
        // Reject all pending
        for (const [id, { reject }] of this._pending) {
          reject(new Error(e.message));
        }
        this._pending.clear();
      };
    } catch(e) {
      console.warn('[SenkabalaWASM] WebWorker failed to start:', e);
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
      // Optional: could display depth/score in UI
      // console.log(`[engine] depth ${msg.depth} score ${msg.score} time ${msg.time} pv ${msg.pv}`);
      return;
    }

    if (msg.type === 'result' || msg.type === 'error') {
      const p = this._pending.get(msg.id);
      if (!p) return;
      this._pending.delete(msg.id);
      if (msg.type === 'result') p.resolve(msg.move);
      else                        p.reject(new Error(msg.message));
      return;
    }
  }

  ready() {
    if (this._ready) return Promise.resolve();
    if (!this._worker) return Promise.reject(new Error('Worker not available'));
    return new Promise((resolve) => {
      this._readyWaiters.push(resolve);
    });
  }

  bestMove(fen, moves = [], movetime = 1000) {
    if (!this._worker) return Promise.reject(new Error('Worker not available'));
    if (!this._ready)  return Promise.reject(new Error('Engine not ready'));

    const id = ++this._idSeq;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this._worker.postMessage({
        type:        'search',
        id,
        fen,
        moves,
        movetime_ms: movetime,
      });
    });
  }

  newGame() {
    // TT is cleared on next init — reinitialise the worker for clean state
    if (this._worker) {
      this._worker.postMessage({ type: 'init' });
    }
  }
}

// Difficulty → movetime mapping (mirrors server DIFFICULTY_SETTINGS)
// Must exactly match server.py DIFFICULTY_SETTINGS
const WASM_DIFFICULTY_MS = {
  1:   200,   // Beginner     — mostly random (movetime irrelevant)
  2:   200,   // Beginner+
  3:   500,   // Easy
  4:  1000,   // Intermediate
  5:  2000,   // Hard
  6:  4000,   // Hard+
  7:  8000,   // Expert
  8: 15000,   // Master
};

// Random move chance per difficulty (mirrors server random_chance)
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

/**
 * Get engine move respecting difficulty.
 * Replaces the /move API call when WASM engine is available.
 */
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
