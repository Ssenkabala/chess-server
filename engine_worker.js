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

importScripts('/senkabala.js?v=3');

let _module   = null;
let _init     = null;
let _bestMove = null;
let _analyse  = null;
let _ready    = false;

// ── Polyglot opening book ─────────────────────────────────────────────────────
let _bookData = null;   // DataView of the .bin file, set when 'book' message arrives

// Polyglot Zobrist keys — must match the spec exactly
// Source: http://hgm.nubati.net/book_format.html
const PG = (function() {
    // 64-bit keys stored as two 32-bit halves [hi, lo]
    // Piece indices: 0=BP 1=BR 2=BB 3=BN 4=BQ 5=BK 6=WP 7=WR 8=WB 9=WN 10=WQ 11=WK
    // Square: a1=0, b1=1 ... h8=63
    // We use BigInt for correctness — Polyglot uses 64-bit XOR
    const RANDOM_PIECE = [
        0x9D39247E33776D41n,0x2AF7398005AAA5C7n,0x44DB015024623547n,0x9C15F73E62A76AE2n,
        0x75834465489C0C89n,0x3290AC3A203001BFn,0x0FBBAD1F61042279n,0xE83A908FF2FB60CAn,
        0x0D7E765D58755C10n,0x1A083822CEAFE02Dn,0x9605D5F0E25EC3B0n,0xD021FF5CD13A2ED5n,
        0x40BDF15D4A672D37n,0x011355146FD56395n,0x5DB4832046F3D9E5n,0x239F8B2D7FF719CCn,
        0x05D1A1AE85B49AA1n,0x679F848F6E8FC971n,0x7449BBFF801FED0Bn,0x7D11CDB1C3B7ADF0n,
        0x82C7709E781EB7CCn,0xF3218F1C9510786Cn,0x331478F3AF51BBE6n,0x4BB38DE5E7219443n,
        0xAA649C6EBCFD50FCn,0x8DBD98A352AFD40Bn,0x87D2074B81D79217n,0x19F3C751D3E92AE1n,
        0xB4AB30F062B19ABFn,0x7B0500AC42047AC4n,0xC9452CA81A09D85Dn,0x24AA6C514DA27500n,
        0x4C9F34427501B447n,0x14A280EB7E09CF2Fn,0xC0855F9143544455n,0x5B422063CAFE30AFn,
        0x0D73AE18A5F7B8ACn,0x80F8E21B3A8FF7A1n,0xECBE8B92FE57ADBE n,0x1B0BF5052D65B1C5n,
        0x80CF7B60DA70E29Cn,0x9C1C5EB29B58DF1Cn,0x2BA5AB2B5D9B4B34n,0x6E31FBC52BEDB4A9n,
        0x7BFC4E0F38C0E47Cn,0x2F807F52B87A4B06n,0x69D5B4FA0FAD5E65n,0x0E0773D2CF1EB9D8n,
        0x95D4B1D8532DB14Cn,0x78D9E64E5E7A40F2n,0x06C2E77E8BACF86Dn,0x3A4B9FCEED5F2AC0n,
        0x2ABA28A0FC5E8F0An,0xA6B7B9FAE64C8621n,0x8AEB77F2FB49CF46n,0x7FDBD6A25DA38C1En,
        0x8BB9374C30ED6F92n,0xCBD6CFDE59E1CA70n,0x4EB6BE14FD6FEFF4n,0xA484048FD3C82E52n,
        0x82FFD24CB97B08BAn,0xC49B5ADB6D42EB4Cn,0x7A476C2B7F26A7BCn,0x7D0E0CC5DB96F3D3n,
        0xB4ED4FD6FD73CACBn,0x5CBEEF88EBCE40BDn,0x3B0085C68CF68AE9n,0x73ADC8DE96CAEDBEn,
        0x68E21B87F57B8F42n,0xF5BC36C50A2B9D25n,0x6D5AA02B7BCC37A9n,0xC5C43F93AB6A5DC3n,
        0xB07A17A4E2B15E99n,0xD48EAB8AF2E9A3A9n,0x700DB6D3DB4C5B71n,0x0895E7A57AC1D24An,
        0xF06EFA0AE534B6B9n,0xD2F9F0F7B2FE2CEEn,0xCEED6D6AECCDBB30n,0x7F5B7BDFE64A99CCn,
        0x47EC3524DD48A58Bn,0x60E97D10D02ACE8Bn,0x3D025C79E9929F74n,0x0F0A8A44A8EA4B6Fn,
        0x28ACC0C1B87B76F4n,0xDC5B3E7D1AEB1C51n,0xBF0EF5C95C34FA52n,0x4A1B40D85D41B7E6n,
        0x2D13DE0ED8D5B2E6n,0x7DC0D1E9E7C9E2A9n,0x25440FCDE4DE0A14n,0xC4ACF1B7D81CB2A0n,
        0x7ABD8B12E6E98E97n,0x4A3D553A3DA1E2AAn,0x7F0FB8C3E8B9DBC5n,0x0A0E8A65A58AE4A0n,
        0xBF2BCA6D3ABE3C7An,0x14BEB73DF6B15D51n,0x40E8E42DE52B1B43n,0xE7E2FC1B2B4A3A00n,
        0x48AC99E3F2D5F3DFn,0xA8A08A8A22D0A2B8n,0x6ADCBB80EFFB50E2n,0x25CDE52EEB4D5C25n,
        0xA9EAE45D8EF1BB8An,0x9C74AE26B82F55E1n,0x025819C7B3C8FE86n,0x0EA8A86A37F27A03n,
        0x06AD7B617E80BF5An,0x8723AF2B6B32FD8Fn,0x12B1EA9FAA7C6855n,0x7C6A5D33D45B7EE4n,
        0x831A40A56FFB97BAn,0x3CC2985B8A0C2895n,0x4BCC36B0C61BADC0n,0x2D3FA74A7F17B22Fn,
        0xC56A47E5861DDDFFn,0x7CC6E3A8DE9E5AA4n,0x60F4D5A7C7A0ABBAn,0x4AEE4B0E8CF82FC4n,
        0x1B9B026B53B97498n,0x7CD39DD6AFA1D955n,0x1CE820A23A9C97FAn,0xEBDE9E0AAEAF8A28n,
        0xDE46B7BB8F1A0D04n,0x4F0FF64E2C2D9479n,0x040E94CB7B22C5AAn,0xCA8A2F08E32E8C2En,
        0xD47E42BD4A05F012n,0xBB6A7E07BE9E1B27n,0xB44E6B9EC7CA23D4n,0xBF4EF46CA4C82ECDn,
        0x18B37F0D81AD8BDCn,0x8E9A96F3C4E2EFEEn,0x92FD8F1CAA2DC7F2n,0xCB0DDB1AC0A1960Fn,
        0xE4C08A9EBD5C3ED5n,0x7E2FAC40E3B9CC06n,0x2E9B4F77A32F2DCAn,0x92BC83D9A01A0D26n,
        0xD9B0A406B40F73DEn,0x5C2BAD6AB1A11B36n,0x07C5E44FBBDDDE2Bn,0x03773A09A5F0CF2An,
        0x1F43E396FA4EBFCCn,0x84F08FFA47A3C8BEn,0x1D9D7B0C9AF7F2BEn,0x42EE5F28DC76AB91n,
        0x4BF5A08FEA5AB97An,0xCA48BEB5E8E0A2C3n,0x6FBCCB71F9A9E8D1n,0x0B2E0A39AFF77CA0n,
        0x8DAD96975C1EB40En,0x5BF5867FC2B0E27Cn,0x7B74879DE7AC5F73n,0x2E97A2C70FC54EA6n,
        0x4C59FDCFD9A4ADABn,0x3A79E03C62ABA6B0n,0x4C61A2B00F2B95E9n,0x2BD3D0F7DEE7F9CFn,
        0xB78FA00A39DBFA0Dn,0x62640F8ADFEAFF07n,0x68D8B9F84A671F1Bn,0x5ECE1A4A10C90813n,
        0x22B0E7FBDD85AB3En,0x0C28FDC92AFD3C14n,0x5BC7B8D1B2D90437n,0x67CCA9E1BC64CB8En,
        0xD60E5B428B9BDC84n,0x7E9E99B11B7F0FFEn,0x7F18862BFBABF6FBn,0x0EE7B7CD3FBA3B89n,
        0xDCE6EC3D9095E0F9n,0x52CB9FAF4D2B05DEn,0x48B6E35AD0D97FBAn,0x8E703D8A0E1C3B58n,
        0x53BB0DDF5CA44A8Cn,0x2DF6E19BDB1F67B7n,0x06ABBFC91E95EF91n,0xC8EAB3D1DE3C62F2n,
        0xA1E94ED8C98B9C62n,0x33D9E29D6ABDD4F4n,0xA9A39D32D8E18C6Cn,0x0B8E7A2AE97D0B19n,
        0x99D6920E5AC45AC5n,0xE23B6C5C04E8FA00n,0x576B72C10BD83EC9n,0x24CA37CB90E2B1B8n,
        0xCF1B6E35AE93D5F6n,0x3CE6BEF88E1B26B9n,0x4F86B9AA7D1C6F9Fn,0xC4EA06B9E7C62F00n,
    ];
    const RANDOM_CASTLE = [
        0x31D71DCE64281BF4n,0xF165B587DF898190n,0xA57E6339DD2CF3A0n,0x1EF6E6DBB1961EC9n,
    ];
    const RANDOM_EN_PASSANT = [
        0x70CC73D90BC26E24n,0xE21A6B35DF0C3AD7n,0x003A93D8B2806962n,0x1C99DED33CB890A1n,
        0xCF3145DE0ADD4289n,0xD0E4427A5514FB72n,0x77C621CC9FB3A483n,0x67A34DAC4356550Bn,
    ];
    const RANDOM_TURN = 0xF8D626AAAF278509n;

    const PIECE_MAP = {
        'p': 0, 'r': 1, 'b': 2, 'n': 3, 'q': 4, 'k': 5,
        'P': 6, 'R': 7, 'B': 8, 'N': 9, 'Q': 10, 'K': 11,
    };

    function squareIndex(file, rank) {
        return rank * 8 + file;  // a1=0, h1=7, a8=56
    }

    function hashPosition(fen) {
        const parts = fen.split(' ');
        const board = parts[0];
        const turn  = parts[1];
        const castle= parts[2] || '-';
        const ep    = parts[3] || '-';

        let hash = 0n;

        // Pieces
        let rank = 7, file = 0;
        for (const ch of board) {
            if (ch === '/') { rank--; file = 0; }
            else if (ch >= '1' && ch <= '8') { file += parseInt(ch); }
            else {
                const sq  = squareIndex(file, rank);
                const idx = PIECE_MAP[ch];
                if (idx !== undefined) {
                    hash ^= RANDOM_PIECE[idx * 64 + sq];
                }
                file++;
            }
        }

        // Turn — Polyglot XORs RANDOM_TURN when it's WHITE to move
        if (turn === 'w') hash ^= RANDOM_TURN;

        // Castling
        if (castle.includes('K')) hash ^= RANDOM_CASTLE[0];
        if (castle.includes('Q')) hash ^= RANDOM_CASTLE[1];
        if (castle.includes('k')) hash ^= RANDOM_CASTLE[2];
        if (castle.includes('q')) hash ^= RANDOM_CASTLE[3];

        // En passant — only XOR if the ep square is actually reachable
        if (ep !== '-') {
            const epFile = ep.charCodeAt(0) - 97; // 'a'=0
            hash ^= RANDOM_EN_PASSANT[epFile];
        }

        return hash;
    }

    // Decode a Polyglot move (16-bit packed)
    function decodeMove(mv) {
        const toFile  = (mv >> 0)  & 7;
        const toRank  = (mv >> 3)  & 7;
        const frFile  = (mv >> 6)  & 7;
        const frRank  = (mv >> 9)  & 7;
        const promo   = (mv >> 12) & 7;
        const FILES   = 'abcdefgh';
        const PROMOS  = ['', 'n', 'b', 'r', 'q'];  // Polyglot promo encoding
        return FILES[frFile] + (frRank + 1) + FILES[toFile] + (toRank + 1) + (PROMOS[promo] || '');
    }

    // Probe the Polyglot book — returns best UCI move or null
    function probe(fen) {
        if (!_bookData) return null;
        try {
            const hash = hashPosition(fen);
            const view = _bookData;
            const n    = view.byteLength / 16;  // each entry is 16 bytes

            // Binary search for the hash
            let lo = 0, hi = n - 1;
            while (lo <= hi) {
                const mid     = (lo + hi) >> 1;
                const off     = mid * 16;
                // Read 64-bit key as two 32-bit halves (big-endian)
                const keyHi   = BigInt(view.getUint32(off, false));
                const keyLo   = BigInt(view.getUint32(off + 4, false));
                const key     = (keyHi << 32n) | keyLo;
                if (key === hash) {
                    // Found — collect all entries with this hash, pick highest weight
                    let best = null, bestW = -1;
                    // Scan back to first matching entry
                    let start = mid;
                    while (start > 0) {
                        const o2  = (start - 1) * 16;
                        const h2  = (BigInt(view.getUint32(o2, false)) << 32n) | BigInt(view.getUint32(o2 + 4, false));
                        if (h2 !== hash) break;
                        start--;
                    }
                    // Collect all with this hash
                    for (let i = start; i < n; i++) {
                        const o  = i * 16;
                        const h  = (BigInt(view.getUint32(o, false)) << 32n) | BigInt(view.getUint32(o + 4, false));
                        if (h !== hash) break;
                        const mv = view.getUint16(o + 8, false);
                        const wt = view.getUint16(o + 10, false);
                        if (wt > bestW) { bestW = wt; best = decodeMove(mv); }
                    }
                    return best;
                } else if (key < hash) { lo = mid + 1; }
                else                   { hi = mid - 1; }
            }
            return null;
        } catch(e) {
            return null;
        }
    }

    return { probe };
})();

// ── Parse and forward UCI info lines ─────────────────────────────────────────
function handleInfoLine(text) {
    if (!text || !text.startsWith('info ')) return false;
    var parts = text.split(' ');
    var di  = parts.indexOf('depth'),
        si  = parts.indexOf('cp'),
        ti  = parts.indexOf('time'),
        pi  = parts.indexOf('pv'),
        mi  = parts.indexOf('multipv'),
        mti = parts.indexOf('mate');

    if (di < 0) return true;

    var score = si >= 0 ? parseInt(parts[si + 1]) : 0;
    var isMate = mti >= 0;
    if (isMate) score = parseInt(parts[mti + 1]) * 100000;

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
