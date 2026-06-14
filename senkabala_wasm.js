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

      // Step 4: fetch the Polyglot opening book and send to worker
      // Book is only ~1-4MB and loads in the background — if unavailable,
      // the worker simply falls through to the engine search (graceful degradation)
      fetch('/book.bin')
        .then(function(r) {
          if (!r.ok) throw new Error('book not found');
          return r.arrayBuffer();
        })
        .then(function(buf) {
          // Transfer the ArrayBuffer — zero-copy, instantly available in worker
          this._worker.postMessage({ type: 'book', bookData: buf }, [buf]);
          console.log('[SenkabalaWASM] Opening book loaded');
        }.bind(this))
        .catch(function() {
          console.log('[SenkabalaWASM] No opening book — engine search only');
        });

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
      // Forward live depth/eval/pv data to whoever is listening (e.g. analysePosition)
      if (typeof this._analysisInfoHandler === 'function') {
        this._analysisInfoHandler(msg);
      }
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
    return this._search('search', fen, moves, movetime, {});
  }

  // Multi-line analysis — calls engine_analyse() in the worker with multipv lines.
  // Info messages stream back per depth via _analysisInfoHandler.
  analyse(fen, moves = [], movetime = 5000, multipv = 3) {
    return this._search('analyse', fen, moves, movetime, { multipv });
  }

  _search(msgType, fen, moves, movetime, extra) {
    if (!this._worker) return Promise.reject(new Error('Worker not available'));

    if (!this._ready) {
      return this.ready().then(() => this._search(msgType, fen, moves, movetime, extra));
    }

    const id = ++this._idSeq;
    const grace = movetime > 2000 ? 10000 : 3000;
    const timeoutMs = movetime + grace;

    return new Promise((resolve, reject) => {
      let timer;
      const wrappedResolve = (v) => { clearTimeout(timer); resolve(v); };
      const wrappedReject  = (e) => { clearTimeout(timer); reject(e);  };

      this._pending.set(id, { resolve: wrappedResolve, reject: wrappedReject });
      this._worker.postMessage({ type: msgType, id, fen, moves, movetime_ms: movetime, ...extra });

      timer = setTimeout(() => {
        if (this._pending.has(id)) {
          this._pending.delete(id);
          console.warn(`[WASM] ${msgType} timeout after ${timeoutMs}ms — falling back to server`);
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
