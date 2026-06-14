/**
 * engine_worker.js — WebWorker for SenkabalaIII WASM
 *
 * Messages IN:
 *   { type: 'module', wasmModule }
 *   { type: 'search', id, fen, moves, movetime_ms }
 *
 * Messages OUT:
 *   { type: 'ready' }
 *   { type: 'result', id, move }
 *   { type: 'error',  id, message }
 *   { type: 'info',   depth, score, time, pv, multipv }
 */

importScripts('/senkabala.js');

let _module   = null;
let _init     = null;
let _bestMove = null;
let _analyse  = null;
let _ready    = false;

// Parse and forward UCI info lines to the main thread
function handleInfoLine(text) {
    if (!text || !text.startsWith('info ')) return false;
    var parts = text.split(' ');
    var di  = parts.indexOf('depth'),
        si  = parts.indexOf('cp'),
        ti  = parts.indexOf('time'),
        pi  = parts.indexOf('pv'),
        mi  = parts.indexOf('multipv'),
        mti = parts.indexOf('mate');

    if (di < 0) return true; // suppress non-depth info lines

    var score = si >= 0 ? parseInt(parts[si + 1]) : 0;
    var isMate = mti >= 0;
    if (isMate) score = parseInt(parts[mti + 1]) * 100000; // sentinel for mate

    // Collect full PV (all moves after 'pv' keyword)
    var pvMoves = [];
    if (pi >= 0) {
        for (var i = pi + 1; i < parts.length; i++) {
            if (parts[i].length >= 4) pvMoves.push(parts[i]);
            else break;
        }
    }

    self.postMessage({
        type:    'info',
        depth:   di >= 0 ? parseInt(parts[di + 1]) : 0,
        score:   score,
        isMate:  isMate,
        time:    ti >= 0 ? parseInt(parts[ti + 1]) : 0,
        pv:      pvMoves[0] || '',
        pvLine:  pvMoves,
        multipv: mi >= 0 ? parseInt(parts[mi + 1]) : 1,
    });
    return true;
}

async function initWithModule(wasmModule) {
    try {
        _module = await SenkabalaModule({
            instantiateWasm: function(imports, successCallback) {
                WebAssembly.instantiate(wasmModule, imports).then(function(instance) {
                    successCallback(instance, wasmModule);
                });
                return {};
            },
            print: function(text) {
                if (handleInfoLine(text)) return;
                if (text && text.startsWith('bestmove')) return;
                console.log('[engine]', text);
            },
            printErr: function(text) {
                // Engine writes info lines to cerr — forward them too
                if (handleInfoLine(text)) return;
                if (text && text.startsWith('bestmove')) return;
                console.warn('[engine]', text);
            }
        });
        _init     = _module.cwrap('engine_init',      null,     []);
        _bestMove = _module.cwrap('engine_best_move', 'string', ['string', 'string', 'number']);
        _analyse  = _module.cwrap('engine_analyse',   'string', ['string', 'string', 'number', 'number']);
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

    if (msg.type === 'analyse') {
        // Multi-line analysis — uses engine_analyse() with configurable MultiPV.
        // Info lines are emitted per-depth via the print/printErr hooks above.
        if (!_ready) {
            self.postMessage({ type: 'error', id: msg.id, message: 'Engine not ready' });
            return;
        }
        try {
            const movesStr = (msg.moves || []).join(' ');
            const multipv  = msg.multipv || 3;
            const move = _analyse(msg.fen, movesStr, msg.movetime_ms, multipv);
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
