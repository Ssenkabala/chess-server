/**
 * engine_worker.js — WebWorker for SenkabalaIII WASM
 *
 * Accepts a pre-compiled WebAssembly.Module from the main thread
 * to avoid recompiling the WASM binary in the worker context.
 *
 * Messages IN:
 *   { type: 'module', wasmModule }   ← send compiled module first
 *   { type: 'search', id, fen, moves, movetime_ms }
 *
 * Messages OUT:
 *   { type: 'ready' }
 *   { type: 'result', id, move }
 *   { type: 'error',  id, message }
 *   { type: 'info',   depth, score, time, pv }
 */

importScripts('/senkabala.js');

let _module   = null;
let _init     = null;
let _bestMove = null;
let _ready    = false;

async function initWithModule(wasmModule) {
    try {
        // Instantiate from the pre-compiled module — fast, no recompilation
        _module = await SenkabalaModule({
            instantiateWasm: function(imports, successCallback) {
                WebAssembly.instantiate(wasmModule, imports).then(function(instance) {
                    successCallback(instance, wasmModule);
                });
                return {};
            },
            // Silence engine info lines (depth/score/pv) from flooding DevTools.
            // Emscripten routes cout/cerr through these hooks.
            print: function(text) {
                // info lines from iterative deepening — parse and forward
                if (text && text.startsWith('info ')) {
                    var parts = text.split(' ');
                    var di = parts.indexOf('depth'),
                        si = parts.indexOf('cp'),
                        ti = parts.indexOf('time'),
                        pi = parts.indexOf('pv');
                    if (di >= 0) {
                        self.postMessage({
                            type:  'info',
                            depth: parseInt(parts[di+1]) || 0,
                            score: si >= 0 ? parseInt(parts[si+1]) : 0,
                            time:  ti >= 0 ? parseInt(parts[ti+1]) : 0,
                            pv:    pi >= 0 ? parts[pi+1] : '',
                        });
                    }
                    return;  // suppress from console
                }
                // bestmove line — suppress (we get the move from cwrap return value)
                if (text && text.startsWith('bestmove')) return;
                // anything else: log normally
                console.log('[engine]', text);
            },
            printErr: function(text) {
                // same as print — engine writes info lines to both streams
                if (text && (text.startsWith('info ') || text.startsWith('bestmove'))) return;
                console.warn('[engine]', text);
            }
        });
        _init     = _module.cwrap('engine_init',      null,     []);
        _bestMove = _module.cwrap('engine_best_move', 'string', ['string', 'string', 'number']);
        _init();
        _ready = true;
        self.postMessage({ type: 'ready' });
    } catch(e) {
        self.postMessage({ type: 'error', id: null, message: 'Engine init failed: ' + String(e) });
    }
}

self.onmessage = function(e) {
    const msg = e.data;

    if (msg.type === 'module') {
        initWithModule(msg.wasmModule);
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
