/**
 * engine_worker.js — WebWorker for SenkabalaIII WASM
 *
 * Messages IN:
 *   { type: 'module', wasmModule }
 *   { type: 'book',   bookData }     ← ArrayBuffer of .bin Polyglot file
 *   { type: 'search', id, fen, moves, movetime_ms }
 *   { type: 'analyse', id, fen, moves, movetime_ms, multipv }
 *
 * Messages OUT:
 *   { type: 'ready' }
 *   { type: 'result', id, move, fromBook }
 *   { type: 'error',  id, message }
 *   { type: 'info',   depth, score, time, pv, multipv }
 */

// Load the Emscripten glue at the SAME version as everything else. The glue
// (senkabala.js) and the binary (senkabala.wasm) are generated together by a
// single emcc run and are a matched pair — instantiating a new binary against
// a stale glue traps or hangs (engine never signals ready; the page sits on
// "Loading engine..." forever). This was previously hardcoded to "?v=3", so a
// rebuilt engine would silently mismatch. self.location.search is the query
// this worker was created with (senkabala_wasm.js now appends it — see there),
// so bumping the single _v in index.html keeps glue + binary + worker in lockstep.
var _WORKER_VER = self.location.search || '?v=4';
importScripts('/senkabala.js' + _WORKER_VER);

let _module   = null;
let _init     = null;
let _bestMove = null;
let _analyse  = null;
let _ready    = false;

// ── Polyglot opening book ─────────────────────────────────────────────────────
// Uses 32-bit integer pairs [hi, lo] instead of BigInt for broad browser/worker
// compatibility. XOR is done on each half independently.
let _bookData = null;

const PG = (function() {
    // 64-bit Polyglot random numbers stored as [hi32, lo32] pairs (big-endian)
    // Piece index: 0=BP 1=BR 2=BB 3=BN 4=BQ 5=BK 6=WP 7=WR 8=WB 9=WN 10=WQ 11=WK
    const RP = [
        [0x9D39247E,0x33776D41],[0x2AF73980,0x05AAA5C7],[0x44DB0150,0x24623547],[0x9C15F73E,0x62A76AE2],
        [0x75834465,0x489C0C89],[0x3290AC3A,0x203001BF],[0x0FBBAD1F,0x61042279],[0xE83A908F,0xF2FB60CA],
        [0x0D7E765D,0x58755C10],[0x1A083822,0xCEAFE02D],[0x9605D5F0,0xE25EC3B0],[0xD021FF5C,0xD13A2ED5],
        [0x40BDF15D,0x4A672D37],[0x01135514,0x6FD56395],[0x5DB48320,0x46F3D9E5],[0x239F8B2D,0x7FF719CC],
        [0x05D1A1AE,0x85B49AA1],[0x679F848F,0x6E8FC971],[0x7449BBFF,0x801FED0B],[0x7D11CDB1,0xC3B7ADF0],
        [0x82C7709E,0x781EB7CC],[0xF3218F1C,0x9510786C],[0x331478F3,0xAF51BBE6],[0x4BB38DE5,0xE7219443],
        [0xAA649C6E,0xBCFD50FC],[0x8DBD98A3,0x52AFD40B],[0x87D2074B,0x81D79217],[0x19F3C751,0xD3E92AE1],
        [0xB4AB30F0,0x62B19ABF],[0x7B0500AC,0x42047AC4],[0xC9452CA8,0x1A09D85D],[0x24AA6C51,0x4DA27500],
        [0x4C9F3442,0x7501B447],[0x14A280EB,0x7E09CF2F],[0xC0855F91,0x43544455],[0x5B422063,0xCAFE30AF],
        [0x0D73AE18,0xA5F7B8AC],[0x80F8E21B,0x3A8FF7A1],[0xECBE8B92,0xFE57ADBE],[0x1B0BF505,0x2D65B1C5],
        [0x80CF7B60,0xDA70E29C],[0x9C1C5EB2,0x9B58DF1C],[0x2BA5AB2B,0x5D9B4B34],[0x6E31FBC5,0x2BEDB4A9],
        [0x7BFC4E0F,0x38C0E47C],[0x2F807F52,0xB87A4B06],[0x69D5B4FA,0x0FAD5E65],[0x0E0773D2,0xCF1EB9D8],
        [0x95D4B1D8,0x532DB14C],[0x78D9E64E,0x5E7A40F2],[0x06C2E77E,0x8BACF86D],[0x3A4B9FCE,0xED5F2AC0],
        [0x2ABA28A0,0xFC5E8F0A],[0xA6B7B9FA,0xE64C8621],[0x8AEB77F2,0xFB49CF46],[0x7FDBD6A2,0x5DA38C1E],
        [0x8BB9374C,0x30ED6F92],[0xCBD6CFDE,0x59E1CA70],[0x4EB6BE14,0xFD6FEFF4],[0xA484048F,0xD3C82E52],
        [0x82FFD24C,0xB97B08BA],[0xC49B5ADB,0x6D42EB4C],[0x7A476C2B,0x7F26A7BC],[0x7D0E0CC5,0xDB96F3D3],
        [0xB4ED4FD6,0xFD73CACB],[0x5CBEEF88,0xEBCE40BD],[0x3B0085C6,0x8CF68AE9],[0x73ADC8DE,0x96CAEDBE],
        [0x68E21B87,0xF57B8F42],[0xF5BC36C5,0x0A2B9D25],[0x6D5AA02B,0x7BCC37A9],[0xC5C43F93,0xAB6A5DC3],
        [0xB07A17A4,0xE2B15E99],[0xD48EAB8A,0xF2E9A3A9],[0x700DB6D3,0xDB4C5B71],[0x0895E7A5,0x7AC1D24A],
        [0xF06EFA0A,0xE534B6B9],[0xD2F9F0F7,0xB2FE2CEE],[0xCEED6D6A,0xECCDBB30],[0x7F5B7BDF,0xE64A99CC],
        [0x47EC3524,0xDD48A58B],[0x60E97D10,0xD02ACE8B],[0x3D025C79,0xE9929F74],[0x0F0A8A44,0xA8EA4B6F],
        [0x28ACC0C1,0xB87B76F4],[0xDC5B3E7D,0x1AEB1C51],[0xBF0EF5C9,0x5C34FA52],[0x4A1B40D8,0x5D41B7E6],
        [0x2D13DE0E,0xD8D5B2E6],[0x7DC0D1E9,0xE7C9E2A9],[0x25440FCD,0xE4DE0A14],[0xC4ACF1B7,0xD81CB2A0],
        [0x7ABD8B12,0xE6E98E97],[0x4A3D553A,0x3DA1E2AA],[0x7F0FB8C3,0xE8B9DBC5],[0x0A0E8A65,0xA58AE4A0],
        [0xBF2BCA6D,0x3ABE3C7A],[0x14BEB73D,0xF6B15D51],[0x40E8E42D,0xE52B1B43],[0xE7E2FC1B,0x2B4A3A00],
        [0x48AC99E3,0xF2D5F3DF],[0xA8A08A8A,0x22D0A2B8],[0x6ADCBB80,0xEFFB50E2],[0x25CDE52E,0xEB4D5C25],
        [0xA9EAE45D,0x8EF1BB8A],[0x9C74AE26,0xB82F55E1],[0x025819C7,0xB3C8FE86],[0x0EA8A86A,0x37F27A03],
        [0x06AD7B61,0x7E80BF5A],[0x8723AF2B,0x6B32FD8F],[0x12B1EA9F,0xAA7C6855],[0x7C6A5D33,0xD45B7EE4],
        [0x831A40A5,0x6FFB97BA],[0x3CC2985B,0x8A0C2895],[0x4BCC36B0,0xC61BADC0],[0x2D3FA74A,0x7F17B22F],
        [0xC56A47E5,0x861DDDFF],[0x7CC6E3A8,0xDE9E5AA4],[0x60F4D5A7,0xC7A0ABBA],[0x4AEE4B0E,0x8CF82FC4],
        [0x1B9B026B,0x53B97498],[0x7CD39DD6,0xAFA1D955],[0x1CE820A2,0x3A9C97FA],[0xEBDE9E0A,0xAEAF8A28],
        [0xDE46B7BB,0x8F1A0D04],[0x4F0FF64E,0x2C2D9479],[0x040E94CB,0x7B22C5AA],[0xCA8A2F08,0xE32E8C2E],
        [0xD47E42BD,0x4A05F012],[0xBB6A7E07,0xBE9E1B27],[0xB44E6B9E,0xC7CA23D4],[0xBF4EF46C,0xA4C82ECD],
        [0x18B37F0D,0x81AD8BDC],[0x8E9A96F3,0xC4E2EFEE],[0x92FD8F1C,0xAA2DC7F2],[0xCB0DDB1A,0xC0A1960F],
        [0xE4C08A9E,0xBD5C3ED5],[0x7E2FAC40,0xE3B9CC06],[0x2E9B4F77,0xA32F2DCA],[0x92BC83D9,0xA01A0D26],
        [0xD9B0A406,0xB40F73DE],[0x5C2BAD6A,0xB1A11B36],[0x07C5E44F,0xBBDDDE2B],[0x03773A09,0xA5F0CF2A],
        [0x1F43E396,0xFA4EBFCC],[0x84F08FFA,0x47A3C8BE],[0x1D9D7B0C,0x9AF7F2BE],[0x42EE5F28,0xDC76AB91],
        [0x4BF5A08F,0xEA5AB97A],[0xCA48BEB5,0xE8E0A2C3],[0x6FBCCB71,0xF9A9E8D1],[0x0B2E0A39,0xAFF77CA0],
        [0x8DAD9697,0x5C1EB40E],[0x5BF5867F,0xC2B0E27C],[0x7B74879D,0xE7AC5F73],[0x2E97A2C7,0x0FC54EA6],
        [0x4C59FDCF,0xD9A4ADAB],[0x3A79E03C,0x62ABA6B0],[0x4C61A2B0,0x0F2B95E9],[0x2BD3D0F7,0xDEE7F9CF],
        [0xB78FA00A,0x39DBFA0D],[0x62640F8A,0xDFEAFF07],[0x68D8B9F8,0x4A671F1B],[0x5ECE1A4A,0x10C90813],
        [0x22B0E7FB,0xDD85AB3E],[0x0C28FDC9,0x2AFD3C14],[0x5BC7B8D1,0xB2D90437],[0x67CCA9E1,0xBC64CB8E],
        [0xD60E5B42,0x8B9BDC84],[0x7E9E99B1,0x1B7F0FFE],[0x7F18862B,0xFBABF6FB],[0x0EE7B7CD,0x3FBA3B89],
        [0xDCE6EC3D,0x9095E0F9],[0x52CB9FAF,0x4D2B05DE],[0x48B6E35A,0xD0D97FBA],[0x8E703D8A,0x0E1C3B58],
        [0x53BB0DDF,0x5CA44A8C],[0x2DF6E19B,0xDB1F67B7],[0x06ABBFC9,0x1E95EF91],[0xC8EAB3D1,0xDE3C62F2],
        [0xA1E94ED8,0xC98B9C62],[0x33D9E29D,0x6ABDD4F4],[0xA9A39D32,0xD8E18C6C],[0x0B8E7A2A,0xE97D0B19],
        [0x99D6920E,0x5AC45AC5],[0xE23B6C5C,0x04E8FA00],[0x576B72C1,0x0BD83EC9],[0x24CA37CB,0x90E2B1B8],
        [0xCF1B6E35,0xAE93D5F6],[0x3CE6BEF8,0x8E1B26B9],[0x4F86B9AA,0x7D1C6F9F],[0xC4EA06B9,0xE7C62F00]
    ];
    const RC  = [[0x31D71DCE,0x64281BF4],[0xF165B587,0xDF898190],[0xA57E6339,0xDD2CF3A0],[0x1EF6E6DB,0xB1961EC9]];
    const REP = [[0x70CC73D9,0x0BC26E24],[0xE21A6B35,0xDF0C3AD7],[0x003A93D8,0xB2806962],[0x1C99DED3,0x3CB890A1],
                 [0xCF3145DE,0x0ADD4289],[0xD0E4427A,0x5514FB72],[0x77C621CC,0x9FB3A483],[0x67A34DAC,0x4356550B]];
    const RT  = [0xF8D626AA,0xAF278509];

    const PIECE_MAP = { p:0,r:1,b:2,n:3,q:4,k:5, P:6,R:7,B:8,N:9,Q:10,K:11 };

    // XOR two [hi,lo] pairs — each half independently
    function xp(a, b) { return [((a[0]^b[0])>>>0), ((a[1]^b[1])>>>0)]; }

    function hashPosition(fen) {
        var parts = fen.split(' ');
        var board = parts[0], turn = parts[1], castle = parts[2]||'-', ep = parts[3]||'-';
        var h = [0,0];
        var rank = 7, file = 0;
        for (var i = 0; i < board.length; i++) {
            var ch = board[i];
            if (ch === '/') { rank--; file = 0; }
            else if (ch >= '1' && ch <= '8') { file += +ch; }
            else {
                var idx = PIECE_MAP[ch];
                if (idx !== undefined) h = xp(h, RP[idx * 64 + (rank * 8 + file)]);
                file++;
            }
        }
        if (turn === 'w') h = xp(h, RT);
        if (castle.indexOf('K') >= 0) h = xp(h, RC[0]);
        if (castle.indexOf('Q') >= 0) h = xp(h, RC[1]);
        if (castle.indexOf('k') >= 0) h = xp(h, RC[2]);
        if (castle.indexOf('q') >= 0) h = xp(h, RC[3]);
        if (ep !== '-') h = xp(h, REP[ep.charCodeAt(0) - 97]);
        return h;
    }

    function decodeMove(mv) {
        var FILES  = 'abcdefgh';
        var PROMOS = ['','n','b','r','q'];
        var toFile = (mv >> 0) & 7, toRank = (mv >> 3) & 7;
        var frFile = (mv >> 6) & 7, frRank = (mv >> 9) & 7;
        var promo  = (mv >> 12) & 7;
        return FILES[frFile]+(frRank+1)+FILES[toFile]+(toRank+1)+(PROMOS[promo]||'');
    }

    function probe(fen) {
        if (!_bookData) return null;
        try {
            var h   = hashPosition(fen);
            var hi  = h[0], lo = h[1];
            var view = _bookData;
            var n   = (view.byteLength / 16) | 0;
            var lft = 0, rgt = n - 1;
            while (lft <= rgt) {
                var mid = (lft + rgt) >> 1;
                var off = mid * 16;
                var khi = view.getUint32(off,     false);
                var klo = view.getUint32(off + 4, false);
                if (khi === hi && klo === lo) {
                    // Scan back to first matching entry
                    var s = mid;
                    while (s > 0) {
                        var o2 = (s-1)*16;
                        if (view.getUint32(o2,false)!==hi||view.getUint32(o2+4,false)!==lo) break;
                        s--;
                    }
                    var best = null, bestW = -1;
                    for (var j = s; j < n; j++) {
                        var oj  = j * 16;
                        if (view.getUint32(oj,false)!==hi||view.getUint32(oj+4,false)!==lo) break;
                        var mv  = view.getUint16(oj + 8, false);
                        var wt  = view.getUint16(oj + 10, false);
                        if (wt > bestW) { bestW = wt; best = decodeMove(mv); }
                    }
                    return best;
                } else if (khi < hi || (khi === hi && klo < lo)) { lft = mid + 1; }
                else { rgt = mid - 1; }
            }
            return null;
        } catch(e) { return null; }
    }

    return { probe };
})();

// ── Parse and forward UCI info lines ─────────────────────────────────────────
// parseInt on a missing/malformed token returns NaN, not null or undefined.
// That matters because `NaN != null` is `true` in JavaScript (NaN is the
// one value that doesn't loosely-equal null the way undefined does) — so
// every downstream `if (mate != null)`-style check treats a NaN mate/cp as
// "yes, a real value is here" and formats it, producing the literal string
// "NaN" in the UI instead of falling back cleanly. Confirmed live. Fixing
// it here, at the one place every numeric field gets parsed, means no
// downstream consumer (formatEval, evalCpApprox, accuracy math, anything
// future) has to remember to guard against it individually.
function safeInt(str, fallback) {
    var n = parseInt(str);
    return isNaN(n) ? (fallback === undefined ? null : fallback) : n;
}

function handleInfoLine(text) {
    // One-time final PV, printed exactly once after a search completes
    // (engine_wasm.cpp's engine_analyse, after the per-depth loop is done).
    // Deliberately a SEPARATE line type from 'info' — sharing the same
    // depth/score fields as the streaming info lines risked overwriting the
    // real, correct final depth/score (already sent moments earlier by the
    // last per-depth info line) with stale or fabricated values.
    //
    // Format: "pvfinal <multipvIdx> cp <n>|mate <n> <uci moves...>"
    if (text && text.startsWith('pvfinal ')) {
        var rest      = text.slice('pvfinal '.length).trim().split(' ').filter(Boolean);
        var mpvIdx    = safeInt(rest[0], 1);
        var scoreType = rest[1];               // 'cp' or 'mate'
        var scoreVal  = safeInt(rest[2]);       // null if the token is missing/malformed — never NaN
        var pvLine    = rest.slice(3);
        var isMate    = scoreType === 'mate';
        self.postMessage({
            type:    'pv_final',
            multipv: mpvIdx,
            pvLine:  pvLine,
            cp:      isMate ? null : scoreVal,
            mate:    isMate ? scoreVal : null
        });
        return true;
    }
    if (!text || !text.startsWith('info ')) return false;
    var parts = text.split(' ');
    var di  = parts.indexOf('depth'),
        cpi = parts.indexOf('cp'),
        mti = parts.indexOf('mate'),
        ti  = parts.indexOf('time'),
        pi  = parts.indexOf('pv'),
        mi  = parts.indexOf('multipv');

    if (di < 0) return true;

    // cp and mate are mutually exclusive, mirroring standard UCI and
    // Lichess's own Score type (separate optional fields, never a single
    // magic-number-encoded value a consumer has to decode).
    var isMate = mti >= 0;
    var cp     = (!isMate && cpi >= 0) ? safeInt(parts[cpi + 1]) : null;
    var mate   = isMate ? safeInt(parts[mti + 1]) : null;

    var pvMoves = [];
    if (pi >= 0) {
        for (var i = pi + 1; i < parts.length; i++) {
            if (parts[i].length >= 4) pvMoves.push(parts[i]);
            else break;
        }
    }

    self.postMessage({
        type:    'info',
        depth:   di >= 0 ? safeInt(parts[di + 1], 0) : 0,
        cp:      cp,
        mate:    mate,
        time:    ti >= 0 ? safeInt(parts[ti + 1], 0) : 0,
        pv:      pvMoves[0] || '',
        pvLine:  pvMoves,
        multipv: mi >= 0 ? safeInt(parts[mi + 1], 1) : 1,
    });
    return true;
}

// ── WASM init ─────────────────────────────────────────────────────────────────
async function initWithModule(wasmModule) {
    try {
        _module = await SenkabalaModule({
            instantiateWasm: function(imports, successCallback) {
                WebAssembly.instantiate(wasmModule, imports).then(function(instance) {
                    successCallback(instance, wasmModule);
                });
                return {};
            },
            print:    function(text) { if (handleInfoLine(text)) return; if (text && text.startsWith('bestmove')) return; console.log('[engine]', text); },
            printErr: function(text) { if (handleInfoLine(text)) return; if (text && text.startsWith('bestmove')) return; console.warn('[engine]', text); }
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

// ── Message handler ───────────────────────────────────────────────────────────
self.onmessage = function(e) {
    const msg = e.data;

    if (msg.type === 'module') {
        initWithModule(msg.wasmModule);
        return;
    }

    // Receive the book binary — stored as DataView for binary search
    if (msg.type === 'book') {
        _bookData = new DataView(msg.bookData);
        console.log('[book] Polyglot book loaded:', (_bookData.byteLength / 16) + ' entries');
        return;
    }

    if (msg.type === 'search') {
        if (!_ready) { self.postMessage({ type: 'error', id: msg.id, message: 'Engine not ready' }); return; }
        try {
            // ── Book probe first ────────────────────────────────────────────
            // Skip book for difficulty 1-3 (random moves already handled client-side)
            // and only use book for the first 15 moves
            const moveCount = (msg.moves || []).length;
            if (moveCount < 15 && (msg.movetime_ms || 0) >= 500) {
                const bookMove = PG.probe(msg.fen);
                if (bookMove) {
                    self.postMessage({ type: 'result', id: msg.id, move: bookMove, fromBook: true });
                    return;
                }
            }
            // ── Engine search ───────────────────────────────────────────────
            const movesStr = (msg.moves || []).join(' ');
            const move = _bestMove(msg.fen, movesStr, msg.movetime_ms);
            self.postMessage({ type: 'result', id: msg.id, move: (move && move !== '0000') ? move : null, fromBook: false });
        } catch(e) {
            self.postMessage({ type: 'error', id: msg.id, message: String(e) });
        }
        return;
    }

    if (msg.type === 'analyse') {
        if (!_ready) { self.postMessage({ type: 'error', id: msg.id, message: 'Engine not ready' }); return; }
        try {
            // Analysis never uses the book — shows engine evaluation, not book moves
            const movesStr = (msg.moves || []).join(' ');
            const multipv  = msg.multipv || 3;
            const move = _analyse(msg.fen, movesStr, msg.movetime_ms, multipv);
            self.postMessage({ type: 'result', id: msg.id, move: (move && move !== '0000') ? move : null, fromBook: false });
        } catch(e) {
            self.postMessage({ type: 'error', id: msg.id, message: String(e) });
        }
        return;
    }
};
