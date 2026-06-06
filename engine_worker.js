/**
 * engine_worker.js — WebWorker wrapper for SenkabalaIII WASM
 *
 * Runs the engine in a background thread so:
 *   1. The UI never freezes during long searches
 *   2. The browser watchdog timer never fires
 *   3. The main thread stays fully responsive
 *
 * Messages IN  (main → worker):
 *   { type: 'init' }
 *   { type: 'search', id, fen, moves, movetime_ms }
 *   { type: 'stop' }
 *
 * Messages OUT (worker → main):
 *   { type: 'ready' }
 *   { type: 'result', id, move }
 *   { type: 'error',  id, message }
 *   { type: 'info',   depth, score, time, pv }  ← optional, for display
 */

// Suppress engine's info lines from flooding DevTools
// Emscripten routes cerr to console.error — we intercept it
const _origError = console.error.bind(console);
console.error = function(...args) {
    const msg = args[0];
    if (typeof msg === 'string' && msg.startsWith('info ')) {
        // Parse and forward as structured info message
        const parts = msg.split(' ');
        const depthIdx = parts.indexOf('depth');
        const scoreIdx = parts.indexOf('cp');
        const timeIdx  = parts.indexOf('time');
        const pvIdx    = parts.indexOf('pv');
        if (depthIdx >= 0) {
            self.postMessage({
                type:  'info',
                depth: parseInt(parts[depthIdx + 1]) || 0,
                score: scoreIdx >= 0 ? parseInt(parts[scoreIdx + 1]) : 0,
                time:  timeIdx  >= 0 ? parseInt(parts[timeIdx  + 1]) : 0,
                pv:    pvIdx    >= 0 ? parts[pvIdx + 1] : '',
            });
        }
        return;  // don't log to console
    }
    _origError(...args);
};

// Load the Emscripten module
importScripts('/senkabala.js');

let _module  = null;
let _init    = null;
let _bestMove = null;
let _ready   = false;

async function loadEngine() {
    try {
        _module   = await SenkabalaModule();
        _init     = _module.cwrap('engine_init',      null,     []);
        _bestMove = _module.cwrap('engine_best_move', 'string', ['string', 'string', 'number']);
        _init();
        _ready = true;
        self.postMessage({ type: 'ready' });
    } catch(e) {
        self.postMessage({ type: 'error', id: null, message: String(e) });
    }
}

loadEngine();

self.onmessage = function(e) {
    const msg = e.data;

    if (msg.type === 'init') {
        if (_ready) self.postMessage({ type: 'ready' });
        return;
    }

    if (msg.type === 'search') {
        if (!_ready) {
            self.postMessage({ type: 'error', id: msg.id, message: 'Engine not ready' });
            return;
        }
        try {
            const movesStr = (msg.moves || []).join(' ');
            const move = _bestMove(msg.fen, movesStr, msg.movetime_ms);
            self.postMessage({
                type: 'result',
                id:   msg.id,
                move: (move && move !== '0000') ? move : null,
            });
        } catch(e) {
            self.postMessage({ type: 'error', id: msg.id, message: String(e) });
        }
        return;
    }
};
