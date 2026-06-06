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
