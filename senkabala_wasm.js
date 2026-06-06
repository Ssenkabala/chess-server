/**
 * senkabala_wasm.js
 * Client-side wrapper for SenkabalaIII compiled to WebAssembly.
 *
 * Usage:
 *   <script src="senkabala.js"></script>   <!-- Emscripten-generated loader -->
 *   <script src="senkabala_wasm.js"></script>
 *
 *   const engine = new SenkabalaEngine();
 *   await engine.ready();
 *   const move = await engine.bestMove('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1', [], 1000);
 *   console.log(move); // e.g. "e7e5"
 */

class SenkabalaEngine {
  constructor() {
    this._module  = null;
    this._init    = null;
    this._bestMove = null;
    this._ready   = false;
    this._queue   = [];   // pending { resolve, fen, moves, ms } while loading
    this._busy    = false; // prevent concurrent searches (WASM is single-threaded)

    this._load();
  }

  async _load() {
    try {
      // SenkabalaModule is the Emscripten MODULARIZE export from senkabala.js
      this._module = await SenkabalaModule();

      // Bind exported C functions
      this._init     = this._module.cwrap('engine_init',      null,   []);
      this._bestMove = this._module.cwrap('engine_best_move', 'string', ['string', 'string', 'number']);

      // Initialise engine (sets up attack tables, TT, etc.)
      this._init();
      this._ready = true;

      console.log('[SenkabalaWASM] Engine ready');

      // Drain any queued requests
      while (this._queue.length > 0) {
        const { resolve, reject, fen, moves, ms } = this._queue.shift();
        this._run(fen, moves, ms).then(resolve).catch(reject);
      }
    } catch (err) {
      console.error('[SenkabalaWASM] Failed to load:', err);
      // Reject all queued promises
      this._queue.forEach(({ reject }) => reject(err));
      this._queue = [];
    }
  }

  /**
   * Returns a promise that resolves when the engine is ready.
   */
  ready() {
    if (this._ready) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const check = () => {
        if (this._ready) resolve();
        else setTimeout(check, 50);
      };
      check();
    });
  }

  /**
   * Get best move for a position.
   *
   * @param {string} fen        - FEN string of the position
   * @param {string[]} moves    - Array of UCI moves played so far e.g. ['e2e4', 'e7e5']
   * @param {number} movetime   - Think time in milliseconds
   * @returns {Promise<string>} - UCI move string e.g. 'e2e4'
   */
  bestMove(fen, moves = [], movetime = 1000) {
    if (!this._ready) {
      return new Promise((resolve, reject) => {
        this._queue.push({ resolve, reject, fen, moves, ms: movetime });
      });
    }
    return this._run(fen, moves, movetime);
  }

  async _run(fen, moves, ms) {
    // WASM is single-threaded — queue if already searching
    if (this._busy) {
      await new Promise(resolve => setTimeout(resolve, 10));
      return this._run(fen, moves, ms);
    }
    this._busy = true;
    try {
      const movesStr = (moves || []).join(' ');
      // Run in a microtask so UI doesn't completely freeze during search
      const result = await new Promise((resolve) => {
        setTimeout(() => {
          const move = this._bestMove(fen, movesStr, ms);
          resolve(move);
        }, 0);
      });
      return result && result !== '0000' ? result : null;
    } finally {
      this._busy = false;
    }
  }

  /**
   * Clear the transposition table (call between games).
   */
  newGame() {
    if (this._ready && this._init) {
      this._init(); // re-init clears TT
    }
  }
}

// Difficulty → movetime mapping (mirrors server DIFFICULTY_SETTINGS)
const WASM_DIFFICULTY_MS = {
  1:   200,   // Beginner     — mostly random (handled in JS, not engine)
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
 * High-level function: get engine move respecting difficulty.
 * Replaces the /move API call when WASM engine is available.
 *
 * @param {object} params
 * @param {string}   params.fen        - Current FEN
 * @param {string[]} params.moves      - Move history (UCI)
 * @param {number}   params.difficulty - 1–8
 * @param {SenkabalaEngine} params.engine
 * @param {chess.js instance} params.chess - For generating legal moves (random fallback)
 * @returns {Promise<string>} UCI move
 */
async function wasmGetMove({ fen, moves, difficulty, engine, chess }) {
  const randomChance = WASM_RANDOM_CHANCE[difficulty] ?? 0;

  // Random move path (levels 1–3)
  if (Math.random() < randomChance) {
    const legal = chess.moves({ verbose: true });
    if (legal.length === 0) return null;
    const pick = legal[Math.floor(Math.random() * legal.length)];
    return pick.from + pick.to + (pick.promotion || '');
  }

  // Engine path
  const ms = WASM_DIFFICULTY_MS[difficulty] ?? 1000;
  return engine.bestMove(fen, moves, ms);
}

// Export for module environments
if (typeof module !== 'undefined') {
  module.exports = { SenkabalaEngine, wasmGetMove, WASM_DIFFICULTY_MS, WASM_RANDOM_CHANCE };
}
