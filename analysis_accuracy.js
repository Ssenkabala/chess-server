// ═══════════════════════════════════════════════════════════════════════
// Move accuracy and classification — based directly on Lichess's
// open-source AccuracyPercent.scala and AccuracyCP.scala modules.
// Translated to JavaScript for use in the AfriChess analysis board.
//
// Sources:
//   lila.analyse.AccuracyPercent.fromWinPercents
//   lila.analyse.AccuracyPercent.fromEvalsAndPov
//   lila.analyse.AccuracyCP.diffsList
//   lila.analyse.JsonView.moves  (for output data structure)
// ═══════════════════════════════════════════════════════════════════════

// ── Win percentage from centipawns ───────────────────────────────────
// Standard sigmoid used by both Lichess and Stockfish.
// Maps centipawn eval → probability of winning (0–100).
function cpToWinPercent(cp) {
  // Clamp to avoid overflow on mate scores
  var clampedCp = Math.max(-1000, Math.min(1000, cp));
  return 50 + 50 * (2 / (1 + Math.exp(-0.00368208 * clampedCp)) - 1);
}

// ── Accuracy of a single move ─────────────────────────────────────────
// Direct translation of Lichess's AccuracyPercent.fromWinPercents.
// beforeWp and afterWp are from the PERSPECTIVE OF THE SIDE MAKING THE MOVE
// (positive = they're winning). afterWp is the position they left.
//
// The exponential curve was fitted by Lichess on real game data:
//   103.1668 * exp(-0.04354 * winDiff) + -3.1669  + 1 (uncertainty bonus)
function accuracyFromWinPercents(beforeWp, afterWp) {
  if (afterWp >= beforeWp) return 100;
  var winDiff = beforeWp - afterWp;
  var raw = 103.1668100711649 * Math.exp(-0.04354415386753951 * winDiff) + -3.166924740191411;
  raw += 1; // uncertainty bonus for imperfect analysis depth
  return Math.max(0, Math.min(100, raw));
}

// Convenience: accuracy directly from centipawn evals, from the MOVER'S
// perspective (positive cp = mover is winning).
function accuracyFromCp(cpBefore, cpAfter) {
  return accuracyFromWinPercents(cpToWinPercent(cpBefore), cpToWinPercent(cpAfter));
}

// ── Move classification ───────────────────────────────────────────────
// Thresholds from Lichess's AccuracyCP and Advice modules.
// cpDrop = how many centipawns the move cost from the mover's perspective.
// (bestEval - playedEval, both from mover's POV, so positive = they lost ground)
var CLASSIFY = {
  MISSED_MATE: 'missed-mate',   // best was forced mate, played move wasn't
  BLUNDER:     'blunder',       // >= 300cp drop
  MISTAKE:     'mistake',       // >= 120cp drop
  INACCURACY:  'inaccuracy',    // >= 50cp drop
  GOOD:        null             // fine — no flag
};

function classifyMove(cpBefore, cpAfter, bestIsMate, playedBestMove) {
  // Missed forced mate: best was a mate, played move wasn't the best
  if (bestIsMate && !playedBestMove) {
    return {
      tag:    CLASSIFY.MISSED_MATE,
      label:  'Missed forced mate',  // caller adds "in N" if known
      symbol: '??',
      severity: 4
    };
  }
  var drop = cpBefore - cpAfter;  // positive = the move lost ground
  if (drop >= 300) return { tag: CLASSIFY.BLUNDER,    label: 'Blunder',     symbol: '??', severity: 3 };
  if (drop >= 120) return { tag: CLASSIFY.MISTAKE,    label: 'Mistake',     symbol: '?',  severity: 2 };
  if (drop >= 50)  return { tag: CLASSIFY.INACCURACY, label: 'Inaccuracy',  symbol: '!?', severity: 1 };
  return null; // no annotation
}

// ── Per-move data structure ───────────────────────────────────────────
// Matches Lichess's JsonView.moves output shape, adapted to what
// AfriChess's engine actually returns (cp score, best move UCI, PV line).
//
// Returns an object for each move:
// {
//   ply:         number,          // 1-indexed ply number
//   san:         string,          // the move played, e.g. "Nf3"
//   played:      string,          // UCI of move played, e.g. "g1f3"
//   best:        string|null,     // UCI of best move, e.g. "e2e4"
//   bestSan:     string|null,     // SAN of best move if different
//   variation:   string[],        // UCI continuation from best move
//   variationSan: string[],       // SAN continuation from best move
//   cpBefore:    number|null,     // eval of position before move (mover's POV)
//   cpAfter:     number|null,     // eval of position after move (mover's POV)
//   accuracy:    number|null,     // 0-100 accuracy of this specific move
//   isMate:      boolean,         // best available was a forced mate
//   mateIn:      number|null,     // mate in N if isMate
//   judgment:    object|null      // { tag, label, symbol, severity } or null
// }
function buildMoveData(opts) {
  var ply      = opts.ply;
  var san      = opts.san;
  var played   = opts.played;           // UCI
  var best     = opts.best;             // UCI
  var bestSan  = opts.bestSan || null;
  var variation      = opts.variation || [];      // UCI list
  var variationSan   = opts.variationSan || [];   // SAN list
  var cpBefore = opts.cpBefore != null ? opts.cpBefore : null;
  var cpAfter  = opts.cpAfter  != null ? opts.cpAfter  : null;
  var isMate   = !!opts.isMate;
  var mateIn   = opts.mateIn || null;

  var playedBestMove = best && played && best.slice(0, 4) === played.slice(0, 4);

  var accuracy = (cpBefore != null && cpAfter != null && !isMate)
    ? Math.round(accuracyFromCp(cpBefore, cpAfter))
    : null;

  var judgment = (cpBefore != null && cpAfter != null)
    ? classifyMove(cpBefore, cpAfter, isMate, playedBestMove)
    : null;

  if (judgment && judgment.tag === CLASSIFY.MISSED_MATE && mateIn) {
    judgment.label = 'Missed forced mate in ' + mateIn;
  }

  return {
    ply,
    san,
    played,
    best:         best || null,
    bestSan:      playedBestMove ? null : (bestSan || null),  // only show if different
    variation,
    variationSan,
    cpBefore,
    cpAfter,
    accuracy,
    isMate,
    mateIn,
    judgment
  };
}

// ── Game-level accuracy summary ───────────────────────────────────────
// Average accuracy across all of a player's moves in a game.
// Lichess uses a weighted harmonic mean (see AccuracyPercent.gameAccuracy)
// but a simple mean is a reasonable first approximation that's much easier
// to implement and still produces meaningful numbers.
function gameAccuracy(moveDataList) {
  var accuracies = moveDataList
    .filter(function(m) { return m.accuracy != null; })
    .map(function(m) { return m.accuracy; });
  if (!accuracies.length) return null;
  var sum = accuracies.reduce(function(a, b) { return a + b; }, 0);
  return Math.round(sum / accuracies.length);
}

// ── Export for use in analysis board ─────────────────────────────────
// When this is inlined into index.html, these become global functions.
// If later extracted to a module, export them here.
//
// Functions available:
//   cpToWinPercent(cp)                              → number (0–100)
//   accuracyFromWinPercents(beforeWp, afterWp)      → number (0–100)
//   accuracyFromCp(cpBefore, cpAfter)               → number (0–100)
//   classifyMove(cpBefore, cpAfter, isMate, played) → judgment obj or null
//   buildMoveData(opts)                             → move data object
//   gameAccuracy(moveDataList)                      → number or null
