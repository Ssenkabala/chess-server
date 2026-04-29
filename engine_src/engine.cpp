/*
 * SenkabalaIII v20 — classical evaluation engine
 *
 * Fixes applied in v20 (on top of v19):
 *   - FIX: Halfmove clock progress penalty. In won positions, score
 *     decreases quadratically as halfmove clock rises (no captures/
 *     pawn moves). Ramps from 0cp at hm=8 to ~80cp at hm=88. Forces
 *     engine to make progress rather than shuffling indefinitely.
 *     Root cause of VCilvVhq (110-move Rb1/Rd1 shuffle with c6 passer).
 *   - EVAL: Rook-behind-passer bonus (+35cp eg). Rewards the engine
 *     for placing rook on same file behind its passed pawn — the
 *     standard KRP technique. Also strengthened king/file cutoff
 *     bonuses (15->25cp) and 7th rank bonus (20->25cp).
 *
 * Fixes applied in v19 (on top of v18):
 *   - FIX: TT scores no longer trusted for positions already seen in
 *     the current search line (inRepLine guard). Previously, cached
 *     scores from non-repetition contexts caused the engine to walk
 *     into threefold repetition — king-walking with f7 pawn on board
 *     (kqyqa0O7: drew a trivially won K+P+Q vs K endgame, 46 moves).
 *     Fix: if repetitionCount(b) > 0, skip TT early-return; still use
 *     TT move for ordering. Small perf cost, correctness gain.
 *
 * Fixes applied in v18 (on top of v17):
 *   - FIX: Repetition contempt now scales with static eval. When engine
 *     is clearly winning (se > 150cp), draw penalty grows proportionally:
 *     e.g. +500cp eval -> contempt ~-90 instead of -15. Prevents giving
 *     perpetuals in won endings (e.g. Q+R vs bare king).
 *   - TUNE: Bullet time management overhauled. defaultMoves 40->30,
 *     pctCap 33->20 (3%->5%). Instability extension capped at +1500ms
 *     in bullet (was +5000ms) to prevent flagging.
 *
 * Fixes applied in v17 (on top of v16):
 *   - FIX: parseMove now handles opponent rook/bishop underpromotions.
 *     When opponent plays e.g. "c2c1r" (promote to rook), the engine was
 *     returning NULL_MOVE because it never generates PROMO_R moves itself.
 *     This caused the position handler to break out of move parsing early,
 *     leaving the internal board in a stale state. The engine then searched
 *     from the wrong position and output illegal moves, causing resignation.
 *     Fix: if the string ends in 'r' or 'b' and no match is found, map it
 *     to the queen promotion on the same squares. The pawn leaves the board
 *     correctly either way.
 *
 *   - FIX: Ponder hang causing time forfeits (confirmed NXZcjvPU: bot had
 *     2:07 on clock in 3+3, never responded to move 16).
 *     Root cause: the ponderhit handler just cleared pondering=false and
 *     called finishSearch() which did a plain join() — trusting the search
 *     to exit on its own time budget. But the ponder search was running with
 *     the time parameters from the "go ponder" command. After ponderhit, the
 *     board has advanced, but the search was still on the ponder position's
 *     budget. If the search was mid-way through a deep iteration when
 *     pondering was cleared, it could run for the full remaining budget (up
 *     to searchTimeMs from the ponder go) before exiting — potentially
 *     minutes in a rapid game.
 *     Fix: ponderhit now:
 *       1. Sets stopNow=true to force-stop the ponder search immediately
 *       2. Joins the (now quickly-exiting) ponder thread
 *       3. Resets stopNow=false
 *       4. Re-launches a fresh search on the actual board with the saved
 *          time parameters from the last "go ponder" command
 *     This guarantees a fast, clean response to ponderhit.
 *   - FIX: finishSearch() now always sets stopNow=true before joining,
 *     guaranteeing the search exits within one negamax check interval
 *     (microseconds to a few ms) rather than waiting for the time budget.
 *   - FIX: Time parameters from "go" are now saved (lastWt/Bt/etc.) so
 *     ponderhit can re-use them for the re-launched search.
 *
 * Fixes applied in v15:
 *   - TUNE: Repetition contempt strengthened (-50/-40/-30/-20/-15 by ply).
 *
 * Fixes applied in v14:
 *   - FIX: Repetition contempt restored (ply-based, no evaluatePos call).
 *
 * Fixes applied in v13:
 *   - FIX: Threaded UCI. stopNow/pondering atomic. go movetime handled.
 *   - FIX: Time management recalibrated (40 moves, 3%/4% caps).
 *
 * Fixes applied in v12:
 *   - FIX: posHistory 1024→2048. evaluatePos removed from repetition.
 *   - FIX: Passed pawn extension fires at most once.
 *
 * Fixes applied in v11:
 *   - FIX: PROMO_Q ordering, PROMO_B/R removed, ttProbe by value.
 *
 * Added in v3: Zobrist, TT, killers, history, MVV-LVA, aspiration,
 *   null move, LMR, reverse futility, PVS.
 */

#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <climits>
#include <fstream>
#include <thread>
#include <atomic>
#include <intrin.h>

using namespace std;

#ifdef USE_SYZYGY
extern "C" {
#include "tbprobe.h"
}
#else
// Stub constants and functions — no-ops when Syzygy not compiled in
#define TB_LOSS         0u
#define TB_BLESSED_LOSS 1u
#define TB_DRAW         2u
#define TB_CURSED_WIN   3u
#define TB_WIN          4u
#define TB_RESULT_FAILED (~0u)
#define TB_GET_WDL(res)      ((res) & 0x0F)
#define TB_GET_FROM(res)     (((res) >> 6)  & 0x3F)
#define TB_GET_TO(res)       (((res) >> 12) & 0x3F)
#define TB_GET_PROMOTES(res) (((res) >> 18) & 0x07)
static int TB_LARGEST = 0;
#define TB_MAX_MOVES 256
static inline bool     tb_init(const char*){ return false; }
static inline unsigned tb_probe_wdl(
    uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,
    uint64_t,uint64_t,uint64_t,unsigned,unsigned,unsigned,bool)
    { return TB_RESULT_FAILED; }
static inline unsigned tb_probe_root(
    uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,
    uint64_t,uint64_t,uint64_t,unsigned,unsigned,unsigned,bool,unsigned*)
    { return TB_RESULT_FAILED; }
#endif

static bool   syzygyEnabled = false;
static string syzygyPath    = "";

// ============================================================
// NNUE — include inference engine
// ============================================================
#ifdef USE_NNUE
#include "nnue.h"
static bool   nnueEnabled = false;
static string nnuePath    = "nn.nnue";
#else
static bool   nnueEnabled = false;
#endif

typedef unsigned long long U64;
typedef int Square;


#ifdef _WIN32
  #include <intrin.h>
  inline int lsb(U64 bb) {
      unsigned long idx;
      if (bb & 0xFFFFFFFFULL) { _BitScanForward(&idx, (unsigned)bb); return (int)idx; }
      _BitScanForward(&idx, (unsigned)(bb >> 32)); return (int)idx + 32;
  }
  inline int msb(U64 bb) {
      unsigned long idx;
      if (bb >> 32) { _BitScanReverse(&idx, (unsigned)(bb>>32)); return (int)idx+32; }
      _BitScanReverse(&idx, (unsigned)bb); return (int)idx;
  }
  inline int popcnt(U64 bb) {
      return (int)(__popcnt((unsigned)(bb&0xFFFFFFFF)) + __popcnt((unsigned)(bb>>32)));
  }
#else
  inline int lsb(U64 bb)   { return __builtin_ctzll(bb); }
  inline int msb(U64 bb)   { return 63 - __builtin_clzll(bb); }
  inline int popcnt(U64 bb){ return __builtin_popcountll(bb); }
#endif
inline int popLSB(U64& bb) { int s=lsb(bb); bb&=bb-1; return s; }

// ============================================================
// SQUARES
// ============================================================
enum {
    A1,B1,C1,D1,E1,F1,G1,H1,
    A2,B2,C2,D2,E2,F2,G2,H2,
    A3,B3,C3,D3,E3,F3,G3,H3,
    A4,B4,C4,D4,E4,F4,G4,H4,
    A5,B5,C5,D5,E5,F5,G5,H5,
    A6,B6,C6,D6,E6,F6,G6,H6,
    A7,B7,C7,D7,E7,F7,G7,H7,
    A8,B8,C8,D8,E8,F8,G8,H8, NO_SQ=64
};
#define SQ(f,r)     ((r)*8+(f))
#define FILE_OF(sq) ((sq)&7)
#define RANK_OF(sq) ((sq)>>3)
#define BIT(sq)     (1ULL<<(sq))

enum PieceType { PAWN=0,KNIGHT,BISHOP,ROOK,QUEEN,KING };
enum Color      { WHITE=0, BLACK=1 };

// ============================================================
// MOVES
// ============================================================
typedef unsigned int Move;
string moveStr(Move m); // forward declaration
#define MK_MOVE(f,t,fl) ((unsigned)(f)|((unsigned)(t)<<6)|((unsigned)(fl)<<12))
#define MV_FROM(m)      ((m)&0x3F)
#define MV_TO(m)        (((m)>>6)&0x3F)
#define MV_FLAGS(m)     (((m)>>12)&0xF)
#define NULL_MOVE       0u

enum Flags {
    QUIET=0,DOUBLE_PUSH=1,CASTLE_K=2,CASTLE_Q=3,
    CAPTURE=4,EP_CAP=5,
    PROMO_N=8,PROMO_B=9,PROMO_R=10,PROMO_Q=11,
    PROMO_CN=12,PROMO_CB=13,PROMO_CR=14,PROMO_CQ=15
};
inline bool isCapture(int f){return f==CAPTURE||f==EP_CAP||f>=PROMO_CN;}
inline bool isPromo(int f)  {return f>=PROMO_N;}
inline PieceType promoTo(int f){
    int b=(f>=PROMO_CN)?f-PROMO_CN:f-PROMO_N;
    return (PieceType)(KNIGHT+b);
}

// ============================================================
// ZOBRIST HASHING
// ============================================================
U64 zPiece[2][6][64], zTurn, zCastle[16], zEP[8];

void initZobrist() {
    U64 s = 0xdeadbeefcafeULL;
    auto rng = [&]() -> U64 {
        s ^= s>>12; s ^= s<<25; s ^= s>>27;
        return s * 0x2545F4914F6CDD1DULL;
    };
    for(int c=0;c<2;c++) for(int p=0;p<6;p++) for(int sq=0;sq<64;sq++) zPiece[c][p][sq]=rng();
    zTurn=rng();
    for(int i=0;i<16;i++) zCastle[i]=rng();
    for(int i=0;i<8;i++) zEP[i]=rng();
}


// ============================================================
// BOARD
// ============================================================
struct Board {
    U64   pieces[2][6];
    U64   occ[3];          // occ[WHITE], occ[BLACK], occ[2]=both
    int   mailbox[64];     // piece type, -1=empty
    int   mailboxC[64];    // piece color, -1=empty
    Color turn;
    int   castling;
    U64   hash;     // incrementally updated Zobrist hash
    int   histCount;  // how many positions are in the global history stack        // 1=WK 2=WQ 4=BK 8=BQ
    Square ep;
    int   halfmove, fullmove;

    void clear() {
        memset(pieces,0,sizeof(pieces));
        memset(occ,0,sizeof(occ));
        memset(mailbox,-1,sizeof(mailbox));
        memset(mailboxC,-1,sizeof(mailboxC));
        turn=WHITE; castling=0; ep=NO_SQ; halfmove=0; fullmove=1; hash=0; histCount=0;
    }
    void place(Color c, PieceType pt, Square sq) {
        pieces[c][pt]|=BIT(sq); occ[c]|=BIT(sq); occ[2]|=BIT(sq);
        mailbox[sq]=pt; mailboxC[sq]=c;
    }
    void rem(Color c, PieceType pt, Square sq) {
        pieces[c][pt]&=~BIT(sq); occ[c]&=~BIT(sq); occ[2]&=~BIT(sq);
        mailbox[sq]=-1; mailboxC[sq]=-1;
    }
};


// Global position history — avoids copying 2KB in Board for singular extensions
U64 posHistory[2048];  // position history for repetition detection — 2048 prevents overflow in long games

U64 computeHash(const Board& b) {
    U64 h=0;
    for(int c=0;c<2;c++) for(int p=0;p<6;p++) {
        U64 bb=b.pieces[c][p];
        while(bb) h^=zPiece[c][p][popLSB(bb)];
    }
    if(b.turn==BLACK) h^=zTurn;
    h^=zCastle[b.castling];
    if(b.ep!=NO_SQ) h^=zEP[FILE_OF(b.ep)];
    return h;
}
// ============================================================
// ATTACK TABLES
// ============================================================
U64 knightAtt[64], kingAtt[64], pawnAtt[2][64];

void initAttacks() {
    for(int sq=0;sq<64;sq++){
        int r=RANK_OF(sq),f=FILE_OF(sq);
        // knight
        U64 kn=0;
        int dx[]={-2,-2,-1,-1,1,1,2,2},dy[]={-1,1,-2,2,-2,2,-1,1};
        for(int i=0;i<8;i++){int nr=r+dy[i],nf=f+dx[i];if(nr>=0&&nr<8&&nf>=0&&nf<8)kn|=BIT(SQ(nf,nr));}
        knightAtt[sq]=kn;
        // king
        U64 kg=0;
        for(int dr=-1;dr<=1;dr++)for(int df=-1;df<=1;df++){
            if(!dr&&!df)continue;
            int nr=r+dr,nf=f+df;if(nr>=0&&nr<8&&nf>=0&&nf<8)kg|=BIT(SQ(nf,nr));
        }
        kingAtt[sq]=kg;
        // pawns
        U64 wp=0,bp=0;
        if(r<7){if(f>0)wp|=BIT(SQ(f-1,r+1));if(f<7)wp|=BIT(SQ(f+1,r+1));}
        if(r>0){if(f>0)bp|=BIT(SQ(f-1,r-1));if(f<7)bp|=BIT(SQ(f+1,r-1));}
        pawnAtt[WHITE][sq]=wp; pawnAtt[BLACK][sq]=bp;
    }
}

// Loop-based sliding attacks — 100% correct, no magic needed
U64 rookAtt(Square sq, U64 occ) {
    U64 att=0; int r=RANK_OF(sq),f=FILE_OF(sq);
    for(int i=r+1;i<8;i++){int s=SQ(f,i);att|=BIT(s);if(occ&BIT(s))break;}
    for(int i=r-1;i>=0;i--){int s=SQ(f,i);att|=BIT(s);if(occ&BIT(s))break;}
    for(int i=f+1;i<8;i++){int s=SQ(i,r);att|=BIT(s);if(occ&BIT(s))break;}
    for(int i=f-1;i>=0;i--){int s=SQ(i,r);att|=BIT(s);if(occ&BIT(s))break;}
    return att;
}
U64 bishAtt(Square sq, U64 occ) {
    U64 att=0; int r=RANK_OF(sq),f=FILE_OF(sq);
    for(int i=1;r+i<8&&f+i<8;i++){int s=SQ(f+i,r+i);att|=BIT(s);if(occ&BIT(s))break;}
    for(int i=1;r+i<8&&f-i>=0;i++){int s=SQ(f-i,r+i);att|=BIT(s);if(occ&BIT(s))break;}
    for(int i=1;r-i>=0&&f+i<8;i++){int s=SQ(f+i,r-i);att|=BIT(s);if(occ&BIT(s))break;}
    for(int i=1;r-i>=0&&f-i>=0;i++){int s=SQ(f-i,r-i);att|=BIT(s);if(occ&BIT(s))break;}
    return att;
}
U64 queenAtt(Square sq,U64 occ){return rookAtt(sq,occ)|bishAtt(sq,occ);}

// Precomputed file masks (avoids inner-loop recomputation in eval)
U64 fileMask[8];
void initMasks() {
    for(int f=0;f<8;f++){
        fileMask[f]=0;
        for(int r=0;r<8;r++) fileMask[f]|=BIT(SQ(f,r));
    }
}

// ============================================================
// PAWN HASH TABLE
// Caches pawn structure scores — recomputed only when pawn
// positions change (rarely). Gives ~40% eval speedup.
// ============================================================
struct PawnEntry {
    U64 key;
    int mg, eg;
};
const int PAWN_TT_SIZE = 1<<14; // 16384 entries ~256KB
PawnEntry pawnTT[PAWN_TT_SIZE];

// Compute a Zobrist key for just the pawns
U64 pawnKey(const Board& b) {
    U64 k = 0;
    U64 wp=b.pieces[WHITE][PAWN], bp=b.pieces[BLACK][PAWN];
    U64 tmp=wp; while(tmp) k^=zPiece[WHITE][PAWN][popLSB(tmp)];
    tmp=bp;     while(tmp) k^=zPiece[BLACK][PAWN][popLSB(tmp)];
    return k;
}

// Evaluate pawn structure only (no king, no pieces)
// Returns mg and eg scores from White's perspective
void evalPawnStructure(const Board& b, int& mg, int& eg) {
    mg=0; eg=0;
    for (int c=0; c<2; c++) {
        int mul = (c==WHITE) ? 1 : -1;
        U64 ourPawns = b.pieces[c][PAWN];
        U64 tmp = ourPawns;
        while (tmp) {
            Square sq = popLSB(tmp);
            int f = FILE_OF(sq), r = RANK_OF(sq);
            U64 file = fileMask[f];

            // Doubled pawn penalty
            if (popcnt(ourPawns & file) > 1) { mg+=mul*(-10); eg+=mul*(-20); }

            // Isolated pawn penalty
            U64 adjFiles = 0;
            if (f>0) adjFiles |= fileMask[f-1];
            if (f<7) adjFiles |= fileMask[f+1];
            if (!(ourPawns & adjFiles)) { mg+=mul*(-15); eg+=mul*(-20); }

            // Passed pawn bonus
            U64 passedMask = 0;
            if (c==WHITE) { for(int rr=r+1;rr<8;rr++){ if(f>0)passedMask|=BIT(SQ(f-1,rr)); passedMask|=BIT(SQ(f,rr)); if(f<7)passedMask|=BIT(SQ(f+1,rr)); } }
            else          { for(int rr=r-1;rr>=0;rr--){ if(f>0)passedMask|=BIT(SQ(f-1,rr)); passedMask|=BIT(SQ(f,rr)); if(f<7)passedMask|=BIT(SQ(f+1,rr)); } }
            if (!(b.pieces[1-c][PAWN] & passedMask)) {
                int advance = (c==WHITE) ? r : (7-r);
                // Base passed pawn bonus — scales sharply with advancement
                eg += mul * (10 + advance*advance*8);
                mg += mul * (5  + advance*advance*3);  // also reward in MG
                // Extra urgency bonus for very advanced passed pawns (rank 5-7)
                if (advance >= 4) { eg += mul * (advance-3) * 40; mg += mul * (advance-3) * 20; }
                if (advance >= 5) { eg += mul * 60;  mg += mul * 40; } // rank 6: nearly unstoppable
                if (advance >= 6) { eg += mul * 120; mg += mul * 80; } // rank 7: promote next move
                // Bonus if passed pawn is protected by another pawn
                if (pawnAtt[1-c][sq] & ourPawns) { eg += mul * 20; mg += mul * 10; }
            }
        }

        // Connected passed pawns bonus — multiple passers on adjacent files are very dangerous
        // Find all passed pawns for this side
        U64 passedBB = 0;
        U64 allPawns = b.pieces[c][PAWN];
        U64 tmp3 = allPawns;
        while (tmp3) {
            Square psq = popLSB(tmp3);
            int pf = FILE_OF(psq), pr = RANK_OF(psq);
            U64 pmask = 0;
            if (c==WHITE) { for(int rr=pr+1;rr<8;rr++){ if(pf>0)pmask|=BIT(SQ(pf-1,rr)); pmask|=BIT(SQ(pf,rr)); if(pf<7)pmask|=BIT(SQ(pf+1,rr)); } }
            else          { for(int rr=pr-1;rr>=0;rr--){ if(pf>0)pmask|=BIT(SQ(pf-1,rr)); pmask|=BIT(SQ(pf,rr)); if(pf<7)pmask|=BIT(SQ(pf+1,rr)); } }
            if (!(b.pieces[1-c][PAWN] & pmask)) passedBB |= BIT(psq);
        }
        int nPassed = popcnt(passedBB);
        if (nPassed >= 2) {
            // Check how many are connected (adjacent files)
            int connected = 0;
            U64 pb2 = passedBB;
            while (pb2) {
                Square psq = popLSB(pb2);
                int pf = FILE_OF(psq);
                // Check adjacent files for another passer
                if ((pf > 0 && (passedBB & fileMask[pf-1])) ||
                    (pf < 7 && (passedBB & fileMask[pf+1])))
                    connected++;
            }
            if (connected >= 2) {
                int bonus = connected * 30;  // 2 connected = +60, 4 connected = +120
                mg += mul * bonus;
                eg += mul * bonus * 2;  // even bigger in endgame
            }
        }
    }
}

// Cached pawn eval lookup
void evalPawnsCached(const Board& b, int& mg, int& eg) {
    U64 k = pawnKey(b);
    PawnEntry& pe = pawnTT[k & (PAWN_TT_SIZE-1)];
    if (pe.key == k) { mg=pe.mg; eg=pe.eg; return; }
    evalPawnStructure(b, mg, eg);
    pe = {k, mg, eg};
}

// Returns 0 = new position, 1 = seen once before, 2+ = draw
int repetitionCount(const Board& b) {
    int count = 0;
    for (int i = b.histCount-2; i >= 0 && i >= b.histCount-b.halfmove-1; i -= 2) {
        if (posHistory[i] == b.hash) count++;
    }
    return count;
}

bool isAttacked(const Board& b, Square sq, Color by) {
    U64 occ=b.occ[2];
    if(pawnAtt[1-by][sq] & b.pieces[by][PAWN])  return true;
    if(knightAtt[sq]     & b.pieces[by][KNIGHT]) return true;
    if(kingAtt[sq]       & b.pieces[by][KING])   return true;
    if(bishAtt(sq,occ)   & (b.pieces[by][BISHOP]|b.pieces[by][QUEEN])) return true;
    if(rookAtt(sq,occ)   & (b.pieces[by][ROOK]  |b.pieces[by][QUEEN])) return true;
    return false;
}
bool inCheck(const Board& b, Color c) {
    U64 k=b.pieces[c][KING];
    return k && isAttacked(b, lsb(k), (Color)(1-c));
}

// ============================================================
// MAKE / UNMAKE
// ============================================================
struct UndoInfo {
    int    movedPiece;      // original piece type (PAWN for promotions)
    int    capturedPiece;   // -1 if none
    int    capturedColor;
    Square capturedSq;
    Square ep;
    int    castling;
    int    halfmove;
    U64    hash;            // full board hash before this move
};

// Forward declaration
void unmakeMove(Board& b, Move m, const UndoInfo& u);

bool makeMove(Board& b, Move m, UndoInfo& u) {
    Square from=MV_FROM(m), to=MV_TO(m);
    int    fl=MV_FLAGS(m);
    Color  us=b.turn, them=(Color)(1-us);

    // Save undo info
    u.movedPiece    = b.mailbox[from];    // original piece (PAWN even for promo)
    u.capturedPiece = -1;
    u.capturedColor = -1;
    u.capturedSq    = NO_SQ;
    u.ep            = b.ep;
    u.castling      = b.castling;
    u.halfmove      = b.halfmove;
    u.hash          = b.hash;

    // Safety: must have a piece to move
    if (u.movedPiece < 0) return false;

    // Hash: remove EP square contribution before we change it
    if (b.ep != NO_SQ) b.hash ^= zEP[FILE_OF(b.ep)];
    // Hash: remove castling before we might change it
    b.hash ^= zCastle[b.castling];

    // Remove captured piece (regular capture or promo-capture)
    if (fl==CAPTURE || fl>=PROMO_CN) {
        int cap=b.mailbox[to];
        if (cap >= 0) {
            u.capturedPiece = cap;
            u.capturedColor = them;
            u.capturedSq    = to;
            b.rem(them, (PieceType)cap, to);
            b.hash ^= zPiece[them][cap][to];
        }
    }
    // En passant capture
    if (fl==EP_CAP) {
        Square capSq = (us==WHITE) ? to-8 : to+8;
        u.capturedPiece = PAWN;
        u.capturedColor = them;
        u.capturedSq    = capSq;
        b.rem(them, PAWN, capSq);
        b.hash ^= zPiece[them][PAWN][capSq];
    }

    // Move the piece (handle promotion)
    b.hash ^= zPiece[us][u.movedPiece][from];
    b.rem(us, (PieceType)u.movedPiece, from);
    PieceType landing = isPromo(fl) ? promoTo(fl) : (PieceType)u.movedPiece;
    b.place(us, landing, to);
    b.hash ^= zPiece[us][landing][to];

    // Castling rook moves
    if (fl==CASTLE_K) {
        Square rf=(us==WHITE)?H1:H8, rt=(us==WHITE)?F1:F8;
        b.hash ^= zPiece[us][ROOK][rf] ^ zPiece[us][ROOK][rt];
        b.rem(us,ROOK,rf); b.place(us,ROOK,rt);
    }
    if (fl==CASTLE_Q) {
        Square rf=(us==WHITE)?A1:A8, rt=(us==WHITE)?D1:D8;
        b.hash ^= zPiece[us][ROOK][rf] ^ zPiece[us][ROOK][rt];
        b.rem(us,ROOK,rf); b.place(us,ROOK,rt);
    }

    // Update ep square
    b.ep = NO_SQ;
    if (fl==DOUBLE_PUSH) {
        b.ep = (us==WHITE) ? from+8 : from-8;
        b.hash ^= zEP[FILE_OF(b.ep)];
    }

    // Update castling rights
    b.castling = u.castling;
    if (u.movedPiece==KING) b.castling &= (us==WHITE) ? ~3 : ~12;
    if (from==A1||to==A1) b.castling &= ~2;
    if (from==H1||to==H1) b.castling &= ~1;
    if (from==A8||to==A8) b.castling &= ~8;
    if (from==H8||to==H8) b.castling &= ~4;
    b.hash ^= zCastle[b.castling];

    // Halfmove clock
    b.halfmove = (u.movedPiece==PAWN || isCapture(fl)) ? 0 : u.halfmove+1;

    // Switch turn
    if (us==BLACK) b.fullmove++;
    b.turn = them;
    b.hash ^= zTurn;
    // Push to global repetition history
    if (b.histCount < 2048) { posHistory[b.histCount] = b.hash; b.histCount++; }

    // Legality check — if our king is in check, unmake and return false
    if (inCheck(b, us)) {
        unmakeMove(b, m, u);
        return false;
    }
    return true;
}

void unmakeMove(Board& b, Move m, const UndoInfo& u) {
    Square from=MV_FROM(m), to=MV_TO(m);
    int fl=MV_FLAGS(m);

    // Restore turn first so 'us' is correct
    b.turn = (Color)(1-b.turn);
    Color us=b.turn;

    // Remove piece from destination
    // Use mailbox[to] to find what's there (the promoted piece if promotion)
    int atTo = b.mailbox[to];
    if (atTo >= 0) b.rem(us, (PieceType)atTo, to);

    // Restore original piece at from
    b.place(us, (PieceType)u.movedPiece, from);

    // Restore captured piece
    if (u.capturedPiece >= 0)
        b.place((Color)u.capturedColor, (PieceType)u.capturedPiece, u.capturedSq);

    // Restore castling rook
    if (fl==CASTLE_K) {
        Square rt=(us==WHITE)?F1:F8, rf=(us==WHITE)?H1:H8;
        b.rem(us,ROOK,rt); b.place(us,ROOK,rf);
    }
    if (fl==CASTLE_Q) {
        Square rt=(us==WHITE)?D1:D8, rf=(us==WHITE)?A1:A8;
        b.rem(us,ROOK,rt); b.place(us,ROOK,rf);
    }

    // Restore board state
    b.ep       = u.ep;
    b.castling = u.castling;
    b.halfmove = u.halfmove;
    b.hash     = u.hash;          // restore full hash — simplest and safest
    if (b.histCount > 0) b.histCount--;
    if (us==BLACK) b.fullmove--;
}

// ============================================================
// MOVE GENERATION
// ============================================================
struct MoveList { Move m[320]; int n=0; void add(Move mv){if(n<320)m[n++]=mv;} };

void genMoves(const Board& b, MoveList& ml) {
    Color us=b.turn, them=(Color)(1-us);
    U64 my=b.occ[us], their=b.occ[them], all=b.occ[2];

    // Pawns
    {
        U64 pawns=b.pieces[us][PAWN];
        while(pawns){
            Square from=popLSB(pawns);
            int r=RANK_OF(from), f=FILE_OF(from);
            if(us==WHITE){
                // Push
                Square to=from+8;
                if(!(all&BIT(to))){
                    if(r==6){ml.add(MK_MOVE(from,to,PROMO_Q));ml.add(MK_MOVE(from,to,PROMO_N));}  // R/B promos never correct
                    else{ml.add(MK_MOVE(from,to,QUIET));if(r==1&&!(all&BIT(to+8)))ml.add(MK_MOVE(from,to+8,DOUBLE_PUSH));}
                }
                // Captures
                U64 att=pawnAtt[WHITE][from]&their;
                while(att){Square t=popLSB(att);if(RANK_OF(t)==7){ml.add(MK_MOVE(from,t,PROMO_CQ));ml.add(MK_MOVE(from,t,PROMO_CN));}else ml.add(MK_MOVE(from,t,CAPTURE));}
                // EP
                if(b.ep!=NO_SQ&&(pawnAtt[WHITE][from]&BIT(b.ep)))ml.add(MK_MOVE(from,b.ep,EP_CAP));
            } else {
                Square to=from-8;
                if(!(all&BIT(to))){
                    if(r==1){ml.add(MK_MOVE(from,to,PROMO_Q));ml.add(MK_MOVE(from,to,PROMO_N));}  // R/B promos never correct
                    else{ml.add(MK_MOVE(from,to,QUIET));if(r==6&&!(all&BIT(to-8)))ml.add(MK_MOVE(from,to-8,DOUBLE_PUSH));}
                }
                U64 att=pawnAtt[BLACK][from]&their;
                while(att){Square t=popLSB(att);if(RANK_OF(t)==0){ml.add(MK_MOVE(from,t,PROMO_CQ));ml.add(MK_MOVE(from,t,PROMO_CN));}else ml.add(MK_MOVE(from,t,CAPTURE));}  // R/B capture-promos removed
                if(b.ep!=NO_SQ&&(pawnAtt[BLACK][from]&BIT(b.ep)))ml.add(MK_MOVE(from,b.ep,EP_CAP));
            }
        }
    }
    // Knights
    {U64 kn=b.pieces[us][KNIGHT];while(kn){Square f=popLSB(kn);U64 att=knightAtt[f]&~my;while(att){Square t=popLSB(att);ml.add(MK_MOVE(f,t,(their&BIT(t))?CAPTURE:QUIET));}}}
    // Bishops
    {U64 bi=b.pieces[us][BISHOP];while(bi){Square f=popLSB(bi);U64 att=bishAtt(f,all)&~my;while(att){Square t=popLSB(att);ml.add(MK_MOVE(f,t,(their&BIT(t))?CAPTURE:QUIET));}}}
    // Rooks
    {U64 ro=b.pieces[us][ROOK];while(ro){Square f=popLSB(ro);U64 att=rookAtt(f,all)&~my;while(att){Square t=popLSB(att);ml.add(MK_MOVE(f,t,(their&BIT(t))?CAPTURE:QUIET));}}}
    // Queens
    {U64 qu=b.pieces[us][QUEEN];while(qu){Square f=popLSB(qu);U64 att=queenAtt(f,all)&~my;while(att){Square t=popLSB(att);ml.add(MK_MOVE(f,t,(their&BIT(t))?CAPTURE:QUIET));}}}
    // King
    {
        U64 kg=b.pieces[us][KING];
        if(kg){
            Square from=lsb(kg);
            U64 att=kingAtt[from]&~my;
            while(att){Square t=popLSB(att);ml.add(MK_MOVE(from,t,(their&BIT(t))?CAPTURE:QUIET));}
            // Castling
            if(us==WHITE){
                if((b.castling&1)&&!(all&0x60ULL)&&!isAttacked(b,E1,BLACK)&&!isAttacked(b,F1,BLACK)&&!isAttacked(b,G1,BLACK))ml.add(MK_MOVE(E1,G1,CASTLE_K));
                if((b.castling&2)&&!(all&0xEULL) &&!isAttacked(b,E1,BLACK)&&!isAttacked(b,D1,BLACK)&&!isAttacked(b,C1,BLACK))ml.add(MK_MOVE(E1,C1,CASTLE_Q));
            } else {
                if((b.castling&4)&&!(all&0x6000000000000000ULL)&&!isAttacked(b,E8,WHITE)&&!isAttacked(b,F8,WHITE)&&!isAttacked(b,G8,WHITE))ml.add(MK_MOVE(E8,G8,CASTLE_K));
                if((b.castling&8)&&!(all&0x0E00000000000000ULL)&&!isAttacked(b,E8,WHITE)&&!isAttacked(b,D8,WHITE)&&!isAttacked(b,C8,WHITE))ml.add(MK_MOVE(E8,C8,CASTLE_Q));
            }
        }
    }
}

// ============================================================
// EVALUATION — tapered eval, pawn structure, mobility, king safety
// ============================================================

// Piece values [mg, eg]
const int MAT[6]   = {100, 320, 330, 500, 900, 0};  // used by move ordering
const int MAT_MG[6]= {100, 325, 335, 500, 975, 0};
const int MAT_EG[6]= {120, 290, 305, 560, 990, 0};

// Game phase weights (max phase = 24)
const int PHASE_W[6] = {0, 1, 1, 2, 4, 0};
const int MAX_PHASE  = 24;

// PST tables [mg][eg] — white relative (a1=0)
const int PST_MG[6][64] = {
// PAWN mg
{ 0, 0, 0, 0, 0, 0, 0, 0,
 -5, 0, 0,-10,-10, 0, 0,-5,
 -5,-5,-5,  0,  0,-5,-5,-5,
  0, 0, 0, 20, 20, 0, 0, 0,
  5, 5,10, 25, 25,10, 5, 5,
 10,10,20, 30, 30,20,10,10,
 50,50,50, 50, 50,50,50,50,
  0, 0, 0,  0,  0, 0, 0, 0},
// KNIGHT mg
{-50,-40,-30,-30,-30,-30,-40,-50,
 -40,-20,  0,  5,  5,  0,-20,-40,
 -30,  5, 10, 15, 15, 10,  5,-30,
 -30,  0, 15, 20, 20, 15,  0,-30,
 -30,  5, 15, 20, 20, 15,  5,-30,
 -30,  0, 10, 15, 15, 10,  0,-30,
 -40,-20,  0,  0,  0,  0,-20,-40,
 -50,-40,-30,-30,-30,-30,-40,-50},
// BISHOP mg
{-20,-10,-10,-10,-10,-10,-10,-20,
 -10,  0,  0,  0,  0,  0,  0,-10,
 -10,  0,  5, 10, 10,  5,  0,-10,
 -10,  5,  5, 10, 10,  5,  5,-10,
 -10,  0, 10, 10, 10, 10,  0,-10,
 -10, 10, 10, 10, 10, 10, 10,-10,
 -10,  5,  0,  0,  0,  0,  5,-10,
 -20,-10,-10,-10,-10,-10,-10,-20},
// ROOK mg
{  0,  0,  0,  5,  5,  0,  0,  0,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
   5, 10, 10, 10, 10, 10, 10,  5,
   0,  0,  0,  0,  0,  0,  0,  0},
// QUEEN mg
{-20,-10,-10, -5, -5,-10,-10,-20,
 -10,  0,  0,  0,  0,  0,  0,-10,
 -10,  0,  5,  5,  5,  5,  0,-10,
  -5,  0,  5,  5,  5,  5,  0, -5,
   0,  0,  5,  5,  5,  5,  0, -5,
 -10,  5,  5,  5,  5,  5,  0,-10,
 -10,  0,  5,  0,  0,  0,  0,-10,
 -20,-10,-10, -5, -5,-10,-10,-20},
// KING mg — want castled, tucked away
{ 30, 40, 40, 10, 10, 20, 40, 30,
  30, 30,  0,  0,  0,  0, 30, 30,
 -10,-20,-20,-20,-20,-20,-20,-10,
 -20,-30,-30,-40,-40,-30,-30,-20,
 -30,-40,-40,-50,-50,-40,-40,-30,
 -30,-40,-40,-50,-50,-40,-40,-30,
 -30,-40,-40,-50,-50,-40,-40,-30,
 -30,-40,-40,-50,-50,-40,-40,-30}
};

const int PST_EG[6][64] = {
// PAWN eg — advanced pawns rewarded heavily
{ 0, 0, 0, 0, 0, 0, 0, 0,
  5,10,10,10,10,10,10, 5,
 -5, 0, 0, 0, 0, 0, 0,-5,
 -5, 0, 0, 0, 0, 0, 0,-5,
  0, 0, 0, 5, 5, 0, 0, 0,
  5, 5,10,20,20,10, 5, 5,
 20,25,30,40,40,30,25,20,
  0, 0, 0, 0, 0, 0, 0, 0},
// KNIGHT eg — centralise, avoid edges
{-50,-40,-30,-30,-30,-30,-40,-50,
 -40,-20,  0,  0,  0,  0,-20,-40,
 -30,  0, 10, 15, 15, 10,  0,-30,
 -30,  5, 15, 20, 20, 15,  5,-30,
 -30,  0, 15, 20, 20, 15,  0,-30,
 -30,  5, 10, 15, 15, 10,  5,-30,
 -40,-20,  0,  5,  5,  0,-20,-40,
 -50,-40,-30,-30,-30,-30,-40,-50},
// BISHOP eg
{-20,-10,-10,-10,-10,-10,-10,-20,
 -10,  0,  0,  0,  0,  0,  0,-10,
 -10,  0,  5,  5,  5,  5,  0,-10,
 -10,  0,  5, 10, 10,  5,  0,-10,
 -10,  0,  5, 10, 10,  5,  0,-10,
 -10,  0,  5,  5,  5,  5,  0,-10,
 -10,  0,  0,  0,  0,  0,  0,-10,
 -20,-10,-10,-10,-10,-10,-10,-20},
// ROOK eg
{  0,  0,  0,  0,  0,  0,  0,  0,
   5, 10, 10, 10, 10, 10, 10,  5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
   0,  0,  0,  5,  5,  0,  0,  0},
// QUEEN eg
{-20,-10,-10, -5, -5,-10,-10,-20,
 -10,  0,  0,  0,  0,  0,  0,-10,
 -10,  0,  5,  5,  5,  5,  0,-10,
  -5,  0,  5,  5,  5,  5,  0, -5,
   0,  0,  5,  5,  5,  5,  0, -5,
 -10,  5,  5,  5,  5,  5,  0,-10,
 -10,  0,  5,  0,  0,  0,  0,-10,
 -20,-10,-10, -5, -5,-10,-10,-20},
// KING eg — centralise in endgame
{-50,-40,-30,-20,-20,-30,-40,-50,
 -30,-20,-10,  0,  0,-10,-20,-30,
 -30,-10, 20, 30, 30, 20,-10,-30,
 -30,-10, 30, 40, 40, 30,-10,-30,
 -30,-10, 30, 40, 40, 30,-10,-30,
 -30,-10, 20, 30, 30, 20,-10,-30,
 -30,-30,  0,  0,  0,  0,-30,-30,
 -50,-30,-30,-30,-30,-30,-30,-50}
};

int evaluate(const Board& b) {
    int mg=0, eg=0, phase=0;

    // Tempo bonus — small reward for side to move
    mg += (b.turn==WHITE) ? 10 : -10;

    // Pawn structure via cache
    { int pmg=0, peg=0; evalPawnsCached(b,pmg,peg); mg+=pmg; eg+=peg; }

    for (int c=0; c<2; c++) {
        int mul = (c==WHITE) ? 1 : -1;
        for (int pt=0; pt<6; pt++) {
            U64 bb = b.pieces[c][pt];
            phase += popcnt(bb) * PHASE_W[pt];
            while (bb) {
                Square sq = popLSB(bb);
                int pstSq = (c==WHITE) ? sq : (sq^56);
                mg += mul * (MAT_MG[pt] + PST_MG[pt][pstSq]);
                eg += mul * (MAT_EG[pt] + PST_EG[pt][pstSq]);
            }
        }

        // Bishop pair bonus
        if (popcnt(b.pieces[c][BISHOP]) >= 2) { mg += mul*30; eg += mul*50; }

        // Rook on open/semi-open file + rook on 7th rank
        U64 rooks = b.pieces[c][ROOK], tmp=rooks;
        int rank7 = (c==WHITE) ? 6 : 1;
        int rank8 = (c==WHITE) ? 7 : 0;
        U64 enemyKing = b.pieces[1-c][KING];
        while (tmp) {
            Square sq = popLSB(tmp);
            int f = FILE_OF(sq);
            U64 file = fileMask[f];
            bool noOurs   = !(b.pieces[c][PAWN]   & file);
            bool noTheirs = !(b.pieces[1-c][PAWN] & file);
            if (noOurs && noTheirs) { mg += mul*25; eg += mul*30; }  // open file — more valuable in EG
            else if (noOurs)        { mg += mul*12; eg += mul*18; } // semi-open
            // Rook on 7th — enemy king or pawns on 8th/7th makes this powerful
            if (RANK_OF(sq)==rank7 && (b.pieces[1-c][PAWN] || (enemyKing && RANK_OF(lsb(enemyKing))==rank8)))
                { mg += mul*20; eg += mul*30; }
        }

        // Mobility
        U64 all = b.occ[2], myPieces = b.occ[c];
        U64 kn = b.pieces[c][KNIGHT];
        while(kn){ Square sq=popLSB(kn); mg+=mul*(popcnt(knightAtt[sq]&~myPieces)-4)*3; }
        U64 bi = b.pieces[c][BISHOP];
        while(bi){ Square sq=popLSB(bi); mg+=mul*(popcnt(bishAtt(sq,all)&~myPieces)-7)*2; }
        U64 ro = b.pieces[c][ROOK];
        while(ro){ Square sq=popLSB(ro); int rmob=popcnt(rookAtt(sq,all)&~myPieces)-7; mg+=mul*rmob*2; eg+=mul*rmob*3; } // rook mobility matters more in EG

        // Knight outposts — knights on squares that can't be attacked by enemy pawns
        // and are supported by our own pawn
        U64 knightBB = b.pieces[c][KNIGHT];
        U64 enemyPawns = b.pieces[1-c][PAWN];
        U64 ourPawnsBB = b.pieces[c][PAWN];
        U64 tmp2 = knightBB;
        while (tmp2) {
            Square sq = popLSB(tmp2);
            // Can this square be attacked by an enemy pawn?
            bool safeFromPawn = !(pawnAtt[c][sq] & enemyPawns);
            // Is it supported by one of our pawns?
            bool supported = (pawnAtt[1-c][sq] & ourPawnsBB) != 0;
            if (safeFromPawn && supported) {
                // Extra bonus for outposts on rank 4-6 for white (3-5 for black)
                int r = RANK_OF(sq);
                int advance = (c==WHITE) ? r : (7-r);
                if (advance >= 3) { mg += mul*(15 + (advance-3)*5); eg += mul*10; }
            }
        }

        // Connected rooks — bonus when our two rooks are on the same rank/file
        // with no pieces between them
        if (popcnt(b.pieces[c][ROOK]) >= 2) {
            U64 rb = b.pieces[c][ROOK];
            Square r1 = popLSB(rb), r2 = lsb(rb);
            U64 all2 = b.occ[2];
            // Check if rooks see each other (same rank or file, no blockers)
            if (RANK_OF(r1)==RANK_OF(r2) || FILE_OF(r1)==FILE_OF(r2)) {
                U64 between = rookAtt(r1, all2) & BIT(r2);
                if (between) { mg += mul*15; eg += mul*10; }
            }
        }

        // Queen early development penalty — discourage queen sorties before pieces developed
        // Count minor pieces (knights + bishops) still on home squares
        U64 queen = b.pieces[c][QUEEN];
        if (queen && phase > 16) {  // only in opening/early middlegame
            Square qsq = lsb(queen);
            int qHomeRank = (c == WHITE) ? 0 : 7;
            if (RANK_OF(qsq) != qHomeRank) {
                // Count undeveloped minor pieces (on home rank)
                int homeRank = (c == WHITE) ? 0 : 7;
                int undeveloped = 0;
                U64 minors = b.pieces[c][KNIGHT] | b.pieces[c][BISHOP];
                U64 tmp3 = minors;
                while(tmp3){ Square s=popLSB(tmp3); if(RANK_OF(s)==homeRank) undeveloped++; }
                // Penalty scales with how many pieces are still undeveloped
                if (undeveloped >= 3) mg += mul * (-30);
                else if (undeveloped >= 2) mg += mul * (-15);
            }
        }

        // King safety — applies in both mg and eg, scales with enemy attacking pieces
        U64 kg = b.pieces[c][KING];
        U64 ourPawns = b.pieces[c][PAWN];
        if (kg) {
            Square ksq = lsb(kg);
            int kf = FILE_OF(ksq), kr = RANK_OF(ksq);

            // Pawn shield — only meaningful in mg
            U64 shield = kingAtt[ksq] & ourPawns;
            mg += mul * (int)(popcnt(shield) * 8);

            // Count enemy attacking pieces and their weight
            int attackWeight = 0;
            bool enemyQueen = b.pieces[1-c][QUEEN] != 0;
            U64 zone = kingAtt[ksq] | BIT(ksq);  // king + adjacent squares

            // Knights attacking zone
            U64 zt = zone;
            while(zt){ Square zsq=popLSB(zt);
                if(knightAtt[zsq] & b.pieces[1-c][KNIGHT]) attackWeight += 2;
            }
            // Sliding pieces attacking zone
            zt = zone;
            while(zt){ Square zsq=popLSB(zt);
                if(bishAtt(zsq,all) & (b.pieces[1-c][BISHOP]|b.pieces[1-c][QUEEN])) attackWeight += 3;
                if(rookAtt(zsq,all) & (b.pieces[1-c][ROOK]  |b.pieces[1-c][QUEEN])) attackWeight += 3;
            }

            // Base penalty
            int penalty = attackWeight * 10;

            // Extra penalty if enemy queen is close to our king (within 3 squares)
            if (enemyQueen) {
                U64 eq = b.pieces[1-c][QUEEN];
                while(eq){
                    Square qsq = popLSB(eq);
                    int dist = abs(FILE_OF(qsq)-kf) + abs(RANK_OF(qsq)-kr);
                    if (dist <= 2) penalty += (3-dist) * 25;  // very close = big penalty
                    else if (dist <= 4) penalty += (5-dist) * 10;
                }
            }

            // Penalty for open files near king (no pawn cover)
            for (int df=-1; df<=1; df++) {
                int f2=kf+df; if(f2<0||f2>7) continue;
                if (!(ourPawns & fileMask[f2])) penalty += 8;
            }

            // Apply penalty — scale down heavily in endgame
            // In endgame (low phase), king should be ACTIVE, not hiding
            // Only apply eg penalty if enemy has queen (genuine mating threats)
            mg += mul * (-penalty);
            if (enemyQueen) eg += mul * (-penalty / 2);

            // King activity bonus in endgame — reward centralised king
            // Scales with how endgame-like the position is (inverse of phase)
            int egWeight = MAX_PHASE - min(phase, MAX_PHASE);
            if (egWeight > 8) {  // meaningfully in endgame
                // Bonus for king being centralised (closer to d4/d5/e4/e5)
                int centreDist = abs(kf - 3) + abs(kr - 3);
                centreDist = min(centreDist, abs(kf - 4) + abs(kr - 4));
                // Base centralisation bonus
                int kingBonus = (6 - centreDist) * (egWeight / 4);
                // Extra bonus in queenless endgames — king is a fighting piece
                bool noQueens = (b.pieces[WHITE][QUEEN] == 0 && b.pieces[BLACK][QUEEN] == 0);
                if (noQueens) kingBonus += (6 - centreDist) * (egWeight / 3);
                eg += mul * kingBonus;
            }
        }
    }

    // Mop-up eval — drive enemy king to corner, close our king, scale with advantage
    int totalMat = 0;
    for(int pt=KNIGHT;pt<=QUEEN;pt++)
        totalMat += (popcnt(b.pieces[WHITE][pt])+popcnt(b.pieces[BLACK][pt]))*MAT[pt];
    if (totalMat < 3000) {
        for (int c=0; c<2; c++) {
            int mul=(c==WHITE)?1:-1, myM=0, thM=0;
            for(int pt=PAWN;pt<=QUEEN;pt++){ myM+=popcnt(b.pieces[c][pt])*MAT[pt]; thM+=popcnt(b.pieces[1-c][pt])*MAT[pt]; }
            int advantage = myM - thM;
            if (advantage > 100) {
                U64 ek=b.pieces[1-c][KING]; U64 mk=b.pieces[c][KING];
                if (ek && mk) {
                    Square eks=lsb(ek), mks=lsb(mk);
                    int ef = FILE_OF(eks), er = RANK_OF(eks);
                    int mf = FILE_OF(mks), mr = RANK_OF(mks);

                    // Drive enemy king to corner (Manhattan distance to nearest corner)
                    int cornerScore = max(3-ef, ef-4) + max(3-er, er-4);
                    eg += mul * cornerScore * 25;  // was 15, now 25

                    // Close our king to enemy king
                    int kingDist = abs(mf-ef) + abs(mr-er);
                    eg += mul * (14 - kingDist) * 12;  // was 8, now 12

                    // Extra bonus scaled by material advantage
                    // (bigger lead = push harder)
                    int scale = min(advantage, 1500) / 100;
                    eg += mul * cornerScore * scale * 3;
                    eg += mul * (14 - kingDist) * scale * 2;
                }
            }
        }
    }

    // ============================================================
    // ENDGAME PATTERN RECOGNITION
    // ============================================================
    // Helper: distance between two squares (Chebyshev)
    auto chebDist = [](Square a, Square b) {
        return max(abs(FILE_OF(a)-FILE_OF(b)), abs(RANK_OF(a)-RANK_OF(b)));
    };
    // Helper: Manhattan distance
    auto manhDist = [](Square a, Square b) {
        return abs(FILE_OF(a)-FILE_OF(b)) + abs(RANK_OF(a)-RANK_OF(b));
    };
    // Helper: corner distance for enemy king (lower = more cornered)
    auto cornerDist = [](Square s) {
        int f=FILE_OF(s), r=RANK_OF(s);
        int d0=f+r, d1=f+(7-r), d2=(7-f)+r, d3=(7-f)+(7-r); return (min)(d0,(min)(d1,(min)(d2,d3)));
    };

    for (int c=0; c<2; c++) {
        int them = 1-c;
        int mul = (c==WHITE) ? 1 : -1;

        U64 myKing   = b.pieces[c][KING];
        U64 theirKing = b.pieces[them][KING];
        if (!myKing || !theirKing) continue;

        Square mks = lsb(myKing);
        Square eks = lsb(theirKing);

        int myQ  = popcnt(b.pieces[c][QUEEN]);
        int myR  = popcnt(b.pieces[c][ROOK]);
        int myB  = popcnt(b.pieces[c][BISHOP]);
        int myN  = popcnt(b.pieces[c][KNIGHT]);
        int myP  = popcnt(b.pieces[c][PAWN]);
        int theirQ = popcnt(b.pieces[them][QUEEN]);
        int theirR = popcnt(b.pieces[them][ROOK]);
        int theirB = popcnt(b.pieces[them][BISHOP]);
        int theirN = popcnt(b.pieces[them][KNIGHT]);
        int theirP = popcnt(b.pieces[them][PAWN]);
        int theirPieces = theirQ+theirR+theirB+theirN+theirP;

        // Only apply mop-up when we have a decisive material advantage
        // and opponent has very little
        bool theirBareKing  = (theirQ+theirR+theirB+theirN+theirP == 0);
        bool theirKingOnly  = theirBareKing;
        bool theirKingPawns = (theirQ+theirR+theirB+theirN == 0) && theirP <= 2;

        int cd = cornerDist(eks);   // 0=corner, 6=center
        int kd = manhDist(mks, eks); // our king to their king

        // ---- KQvK, KRRvK, KQRvK etc: pure mop-up ----
        if (theirKingOnly && (myQ > 0 || myR >= 2)) {
            // Push enemy king to corner, bring our king close
            eg += mul * (6 - cd) * 90;   // corner pressure (max +540)
            eg += mul * (14 - kd) * 35;  // king proximity (max +490)
        }

        // ---- KRvK: rook mop-up (needs king cooperation) ----
        if (theirKingOnly && myQ==0 && myR==1 && myB==0 && myN==0) {
            eg += mul * (6 - cd) * 60;
            eg += mul * (14 - kd) * 25;
        }

        // ---- KBBvK: two bishops ----
        if (theirKingOnly && myQ==0 && myR==0 && myB==2 && myN==0) {
            eg += mul * (6 - cd) * 80;
            eg += mul * (14 - kd) * 30;
        }

        // ---- KBNvK: bishop + knight (hardest basic mate) ----
        // Enemy king must go to corner of bishop's color
        if (theirKingOnly && myQ==0 && myR==0 && myB==1 && myN==1) {
            // Find bishop color
            Square bsq = lsb(b.pieces[c][BISHOP]);
            bool bishLight = (FILE_OF(bsq) + RANK_OF(bsq)) % 2 == 0;
            // Distance to correct-color corner
            int ef = FILE_OF(eks), er = RANK_OF(eks);
            int darkCornerDist  = min(ef+er, (7-ef)+(7-er));
            int lightCornerDist = min((7-ef)+er, ef+(7-er));
            int correctCorner = bishLight ? lightCornerDist : darkCornerDist;
            eg += mul * (6 - correctCorner) * 70;
            eg += mul * (14 - kd) * 25;
        }

        // ---- KPvK: king and pawn endgame ----
        // Bonus for pawn advancement and king support
        if (theirKingOnly && myQ==0 && myR==0 && myB==0 && myN==0 && myP==1) {
            U64 pawn = b.pieces[c][PAWN];
            if (pawn) {
                Square psq = lsb(pawn);
                int pr = RANK_OF(psq);
                // Bonus for advanced pawn
                eg += mul * (c==WHITE ? pr : 7-pr) * 15;
                // King in front of pawn bonus
                int pf = FILE_OF(psq);
                int kingAheadRank = (c==WHITE) ? pr+1 : pr-1;
                if (RANK_OF(mks) == kingAheadRank && abs(FILE_OF(mks)-pf) <= 1)
                    eg += mul * 40; // king leading the pawn
            }
        }

        // ---- Rook endgame: cut off enemy king ----
        if (myR >= 1 && theirQ==0 && theirR==0 && theirPieces <= 2) {
            U64 rooks = b.pieces[c][ROOK];  // local copy — popLSB modifies it
            while (rooks) {
                Square rsq = popLSB(rooks);
                int rr = RANK_OF(rsq), rf = FILE_OF(rsq);
                // Bonus for rook on rank/file cutting off enemy king
                if (rr == RANK_OF(eks)) eg += mul * 25;   // was 15
                if (rf == FILE_OF(eks)) eg += mul * 25;   // was 15
                // Bonus for rook on 7th (already handled above but reinforce in EG)
                if ((c==WHITE && rr==6) || (c==BLACK && rr==1))
                    eg += mul * 25;   // was 20
                // Rook behind passed pawn — the key KRP technique bonus
                // Rook on same file as a friendly passed pawn, behind it
                U64 myPassers = b.pieces[c][PAWN];
                while (myPassers) {
                    Square psq = popLSB(myPassers);
                    if (FILE_OF(psq) == rf) {
                        // Check it's actually a passer
                        int pf = FILE_OF(psq), pr = RANK_OF(psq);
                        U64 pmask = 0;
                        if (c==WHITE) { for(int rr2=pr+1;rr2<8;rr2++){ if(pf>0)pmask|=BIT(SQ(pf-1,rr2)); pmask|=BIT(SQ(pf,rr2)); if(pf<7)pmask|=BIT(SQ(pf+1,rr2)); } }
                        else          { for(int rr2=pr-1;rr2>=0;rr2--){ if(pf>0)pmask|=BIT(SQ(pf-1,rr2)); pmask|=BIT(SQ(pf,rr2)); if(pf<7)pmask|=BIT(SQ(pf+1,rr2)); } }
                        if (!(b.pieces[1-c][PAWN] & pmask)) {
                            // Rook is behind the passer (White: rook rank < pawn rank; Black: rook rank > pawn rank)
                            bool rookBehind = (c==WHITE) ? (rr < pr) : (rr > pr);
                            if (rookBehind) eg += mul * 35;  // rook behind passer — ideal setup
                        }
                    }
                }
            }
        }

        // ---- Winning Q+pieces vs bare king: scale bonus with how winning ----
        if (theirKingOnly && (myQ+myR+myB+myN >= 2)) {
            eg += mul * (6 - cd) * 50;
            eg += mul * (14 - kd) * 20;
        }
    }

    // Taper between mg and eg
    phase = min(phase, MAX_PHASE);
    int score = (mg*phase + eg*(MAX_PHASE-phase)) / MAX_PHASE;

    // Halfmove clock progress penalty — penalise the winning side as the
    // clock ticks up. Forces the engine to make progress rather than
    // shuffling indefinitely in won endgames. Ramps from 0 at hm=0 to
    // about -60cp at hm=80 (near the 50-move draw). The penalty is from
    // the perspective of the WINNING side (whoever has the higher score),
    // so it makes shuffling score progressively worse, pushing the engine
    // toward decisive moves (pawn pushes, captures, king advances).
    if (b.halfmove > 8) {
        int hm_penalty = (b.halfmove - 8) * (b.halfmove - 8) / 80;  // quadratic ramp
        hm_penalty = min(hm_penalty, 80);
        // Apply to the winning side — penalise score toward 0
        if (score > 30)  score = max(30, score - hm_penalty);
        else if (score < -30) score = min(-30, score + hm_penalty);
    }

    return (b.turn==WHITE) ? score : -score;
}

// ============================================================
// TRANSPOSITION TABLE
// ============================================================
enum TTFlag { TT_EXACT, TT_LOWER, TT_UPPER };
struct TTEntry { U64 hash; int score, depth; Move bestMove; TTFlag flag; };
const int TT_SIZE = 1<<23;  // 8M entries ~160MB
TTEntry* tt = nullptr;

void ttClear() { if(tt) memset(tt, 0, TT_SIZE*sizeof(TTEntry)); }

struct TTResult { int score, depth, flag; Move bestMove; };

bool ttProbe(U64 hash, TTResult& out) {
    if (!tt) return false;
    TTEntry* e = &tt[hash & (TT_SIZE-1)];
    if (e->hash != hash) return false;
    out.score    = e->score;
    out.depth    = e->depth;
    out.flag     = (int)e->flag;
    out.bestMove = e->bestMove;
    return true;
}

void ttStore(U64 hash, int score, int depth, Move best, TTFlag flag) {
    if (!tt) return;
    TTEntry* e = &tt[hash & (TT_SIZE-1)];
    if (e->hash == hash || depth >= e->depth) {
        if (best == NULL_MOVE && e->hash == hash) best = e->bestMove;
        e->hash     = hash;
        e->score    = score;
        e->depth    = depth;
        e->bestMove = best;
        e->flag     = flag;
    }
}


// ============================================================
// SYZYGY PROBE HELPER
// ============================================================
// Convert our board to Syzygy bitboard format and probe
// Returns: TB_WIN, TB_DRAW, TB_LOSS, or TB_RESULT_FAILED
unsigned tbProbeWDL(const Board& b) {
    if (!syzygyEnabled) return TB_RESULT_FAILED;
    int pieceCount = popcnt(b.occ[2]);
    if (pieceCount > TB_LARGEST) return TB_RESULT_FAILED;
    if (b.castling) return TB_RESULT_FAILED; // can't probe with castling rights
    unsigned ep = (b.ep != NO_SQ) ? (unsigned)b.ep : 0;
    return tb_probe_wdl(
        b.occ[WHITE],                    // white pieces
        b.occ[BLACK],                    // black pieces
        b.pieces[WHITE][KING]   | b.pieces[BLACK][KING],
        b.pieces[WHITE][QUEEN]  | b.pieces[BLACK][QUEEN],
        b.pieces[WHITE][ROOK]   | b.pieces[BLACK][ROOK],
        b.pieces[WHITE][BISHOP] | b.pieces[BLACK][BISHOP],
        b.pieces[WHITE][KNIGHT] | b.pieces[BLACK][KNIGHT],
        b.pieces[WHITE][PAWN]   | b.pieces[BLACK][PAWN],
        ep, 0, 0,                        // ep, rule50, castling (always 0)
        b.turn == WHITE                  // true = white to move
    );
}

// Probe root for best TB move (uses DTZ for shortest win)
unsigned tbProbeRoot(const Board& b, unsigned* results) {
    if (!syzygyEnabled) return TB_RESULT_FAILED;
    int pieceCount = popcnt(b.occ[2]);
    if (pieceCount > TB_LARGEST) return TB_RESULT_FAILED;
    if (b.castling) return TB_RESULT_FAILED;
    unsigned ep = (b.ep != NO_SQ) ? (unsigned)b.ep : 0;
    return tb_probe_root(
        b.occ[WHITE], b.occ[BLACK],
        b.pieces[WHITE][KING]   | b.pieces[BLACK][KING],
        b.pieces[WHITE][QUEEN]  | b.pieces[BLACK][QUEEN],
        b.pieces[WHITE][ROOK]   | b.pieces[BLACK][ROOK],
        b.pieces[WHITE][BISHOP] | b.pieces[BLACK][BISHOP],
        b.pieces[WHITE][KNIGHT] | b.pieces[BLACK][KNIGHT],
        b.pieces[WHITE][PAWN]   | b.pieces[BLACK][PAWN],
        ep, 0, 0,
        b.turn == WHITE,
        results
    );
}

// Convert TB WDL result to centipawn score (from side-to-move perspective)
int tbScore(unsigned wdl, int ply) {
    switch(wdl) {
        case TB_WIN:          return 20000 - ply;  // win, prefer faster
        case TB_CURSED_WIN:   return 1;             // technically winning but 50-move draw risk
        case TB_DRAW:         return 0;
        case TB_BLESSED_LOSS: return -1;            // technically losing but 50-move draw risk
        case TB_LOSS:         return -20000 + ply;  // loss, prefer slower
        default:              return 0;
    }
}


// ============================================================
// NNUE-AWARE EVALUATION WRAPPER
// ============================================================
// Calls NNUE if loaded, otherwise falls back to classical eval
int evaluatePos(const Board& b) {
#ifdef USE_NNUE
    if (nnueEnabled) {
        // Build 6-piece bitboard array: [color][type] with king included
        // Engine piece order: PAWN=0,KNIGHT=1,BISHOP=2,ROOK=3,QUEEN=4,KING=5
        uint64_t pieces[2][6];
        for (int c = 0; c < 2; c++) {
            for (int pt = 0; pt < 5; pt++)
                pieces[c][pt] = b.pieces[c][pt];
            pieces[c][5] = b.pieces[c][KING];  // KING=5 in our 768-feature indexing
        }
        float feats[NN_INPUT];
        extractFeatures(feats, pieces, b.turn);
        int score = nnueForward(feats);
        // score is already from side-to-move perspective (extractFeatures flips for black)
        return score;
    }
#endif
    return evaluate(b);
}

// ============================================================
// SEARCH GLOBALS
// ============================================================
const int INF=1000000, MATE=999000;
const int MAX_PLY=64;

// LMR table
int LMR[MAX_PLY][64];
// LMP — max quiet moves to try at low depth before pruning
const int LMP_LIMIT[4] = {0, 5, 10, 20}; // indexed by depth 0-3
void initLMR() {
    for(int d=0;d<MAX_PLY;d++) for(int m=0;m<64;m++)
        LMR[d][m] = (d<2||m<2) ? 0 : (int)(0.75+log((double)d)*log((double)m)/2.25);
}

// Search state
Move killers[2][MAX_PLY];
int  history[2][64][64];
Move counterMove[64][64];

auto searchStart = chrono::steady_clock::now();
int  searchTimeMs = 1000;
atomic<bool> stopNow{false};
atomic<bool> pondering{false};  // true while pondering (infinite search until "stop")
Move ponderMove = NULL_MOVE;    // the expected opponent reply we're pondering on

// Search runs in a background thread so the UCI loop stays responsive
// to "stop" and "ponderhit" commands during pondering.
Move  searchResult = NULL_MOVE;
atomic<bool> searchDone{false};
thread searchThread;

bool timeUp() {
    if (pondering) return false; // never time out while pondering
    return chrono::duration_cast<chrono::milliseconds>(
        chrono::steady_clock::now()-searchStart).count() >= searchTimeMs;
}
int quiesce(Board& b, int alpha, int beta) {
    if(stopNow||timeUp()){stopNow=true;return 0;}
    int stand=evaluatePos(b);
    // Delta pruning — if even capturing the best possible piece can't raise alpha, skip
    const int DELTA=900; // queen value
    if(stand < alpha - DELTA) return alpha;
    if(stand>=beta) return beta;
    if(stand>alpha) alpha=stand;

    MoveList ml; genMoves(b,ml);
    for(int i=0;i<ml.n;i++){
        Move mv=ml.m[i];
        int fl=MV_FLAGS(mv);
        if(!isCapture(fl) && fl!=PROMO_Q) continue;  // search captures + queen promos
        UndoInfo u; if(!makeMove(b,mv,u)) continue;
        // Stalemate check — if opponent has no pieces left except king,
        // verify they have at least one legal king move to avoid gifting stalemate
        bool oppInCheck = ::inCheck(b, b.turn);
        if (!oppInCheck) {
            U64 oppNonKing = b.occ[(int)b.turn] & ~b.pieces[(int)b.turn][KING];
            if (!oppNonKing) {
                // Only king left — check if king has any legal move
                U64 kg = b.pieces[(int)b.turn][KING];
                if (kg) {
                    Square ksq = lsb(kg);
                    U64 moves = kingAtt[ksq] & ~b.occ[(int)b.turn];
                    bool hasKingMove = false;
                    while (moves && !hasKingMove) {
                        Square tsq = popLSB(moves);
                        // Use correct flag — CAPTURE if target occupied, else QUIET
                        int mvfl = (b.mailbox[tsq] >= 0) ? CAPTURE : QUIET;
                        Move km = MK_MOVE(ksq, tsq, mvfl);
                        UndoInfo tu; if(makeMove(b, km, tu)){
                            hasKingMove = true; unmakeMove(b, km, tu);
                        }
                    }
                    if (!hasKingMove) { unmakeMove(b,mv,u); continue; } // stalemate — skip
                }
            }
        }
        int sc=-quiesce(b,-beta,-alpha);
        unmakeMove(b,mv,u);
        if(sc>=beta) return beta;
        if(sc>alpha) alpha=sc;
    }
    return alpha;
}

// Score a move for ordering

int scoreMove(const Board& b, Move m, Move ttMove, int ply) {
    if (m == ttMove) return 2000000;
    int fl = MV_FLAGS(m);
    // MVV-LVA capture ordering
    if (isCapture(fl)) {
        int attacker = b.mailbox[MV_FROM(m)];
        int victim   = b.mailbox[MV_TO(m)];
        if (fl == EP_CAP) victim = PAWN;
        int victimVal  = (victim  >= 0) ? MAT[victim]  : 0;
        int attackVal  = (attacker >= 0) ? MAT[attacker] : 0;
        // Winning captures (victim > attacker) get extra boost to ensure
        // they're always searched before quiet moves regardless of history
        int base = 1000000 + victimVal*10 - attackVal;
        if (victimVal >= attackVal) base += 500000;  // winning/equal capture bonus
        return base;
    }
    // Promotions — queen always highest, underpromotions very low
    if (isPromo(fl)) {
        PieceType pt = promoTo(fl);
        if (pt == QUEEN)  return 1900000;  // just below TT move
        if (pt == KNIGHT) return 500;      // rare underpromo (smothered mate etc.)
        return 100;                        // rook/bishop underpromo: almost never want
    }
    // Killers
    if (ply < MAX_PLY) {
        if (m == killers[0][ply]) return 900000;
        if (m == killers[1][ply]) return 800000;
    }
    // Countermove heuristic (needs prevMove — passed via global for simplicity)
    // History
    int from=MV_FROM(m), to=MV_TO(m);
    return history[(int)b.turn][from][to];
}


// SEE — Static Exchange Evaluation
// Returns the material gain/loss from a capture sequence on 'to'
// Positive = we win material, negative = we lose material
void sortMoves(const Board& b, MoveList& ml, Move ttMove, int ply) {
    int scores[320];
    for (int i=0;i<ml.n;i++) scores[i]=scoreMove(b,ml.m[i],ttMove,ply);
    // Selection sort (fine for 256 moves)
    for (int i=0;i<ml.n;i++) {
        int best=i;
        for (int j=i+1;j<ml.n;j++) if(scores[j]>scores[best]) best=j;
        if (best!=i) { swap(ml.m[i],ml.m[best]); swap(scores[i],scores[best]); }
    }
}

int negamax(Board& b, int depth, int alpha, int beta, int ply, bool inNull=false) {
    if(stopNow||timeUp()){stopNow=true;return 0;}
    if(b.halfmove>=100) return 0;
    if(ply>0){
        int rc = repetitionCount(b);
        // Repetition contempt — scaled by ply, always negative.
        // The engine should never willingly repeat a position; it should
        // only accept repetition if the alternative is genuinely worse.
        //
        // Scale: strong penalty at shallow ply (engine choosing to repeat),
        // milder at deep ply (opponent may be forcing it, or it's a true draw).
        //   ply 1: -50  (engine is directly choosing the repeating move)
        //   ply 2: -40
        //   ply 3: -30
        //   ply 4: -20
        //   ply 5+: -15  (deep in tree, could be opponent-forced)
        //   rc>=2:   0   (third occurrence = draw by rule)
        //
        // These values are well below any real winning continuation (+100 to
        // +500 in a won position) but above a genuinely losing line (-200+),
        // so the engine correctly avoids self-imposed draws while still
        // accepting repetition when it's actually losing.
        if(rc >= 2) return 0;
        if(rc == 1) {
            int contempt = (ply==1) ? -50 : (ply==2) ? -40 : (ply==3) ? -30 : (ply==4) ? -20 : -15;
            // Scale with static eval: penalise draw more when winning, less when losing.
            int se = evaluate(b);
            if (se > 150)  contempt -= (se - 150) / 8;
            if (se < -150) contempt += (-se - 150) / 16;
            return contempt;
        }
    }
    if(depth==0) return quiesce(b,alpha,beta);

    // Syzygy tablebase probe — perfect play for <= 5 pieces
    if (syzygyEnabled && popcnt(b.occ[2]) <= TB_LARGEST && !b.castling) {
        unsigned wdl = tbProbeWDL(b);
        if (wdl != TB_RESULT_FAILED) {
            int sc = tbScore(wdl, ply);
            TTFlag flag = (wdl==TB_WIN||wdl==TB_CURSED_WIN) ? TT_LOWER :
                          (wdl==TB_LOSS||wdl==TB_BLESSED_LOSS) ? TT_UPPER : TT_EXACT;
            ttStore(b.hash, sc, depth, NULL_MOVE, flag);
            return sc;
        }
    }

    bool pvNode = (beta - alpha) > 1;
    bool inCheck = ::inCheck(b, b.turn);
    int alphaOrig = alpha;
    Move ttMove = NULL_MOVE;

    // TT probe — move ordering and score bounds only, NEVER at root (ply==0)
    // IMPORTANT: don't use TT_EXACT scores for positions that have been seen
    // earlier in this line (repetitionCount > 0). The cached score was stored
    // in a context where the position was not a candidate repetition, so it
    // doesn't account for the draw risk. Using it causes the engine to keep
    // "finding" +300cp for king moves that actually lead to threefold.
    // We still use the TT move for ordering — just don't early-return the score.
    bool inRepLine = (repetitionCount(b) > 0);
    if (ply > 0) {
        TTResult tte; bool tteHit = ttProbe(b.hash, tte);
        if (tteHit) {
            ttMove = tte.bestMove;
            if (tte.depth >= depth && !inRepLine) {
                if (tte.flag == TT_EXACT) return tte.score;
                if (tte.flag == TT_LOWER) alpha = max(alpha, tte.score);
                if (tte.flag == TT_UPPER) beta  = min(beta,  tte.score);
                if (alpha >= beta) return tte.score;
            }
        }
    }

    // IID — no TT move at deep node: do shallow search to get a good move to order first
    if (ttMove == NULL_MOVE && depth >= 5 && pvNode) {
        negamax(b, depth-4, alpha, beta, ply);
        TTResult iidTTE; if (ttProbe(b.hash, iidTTE)) ttMove = iidTTE.bestMove;
    }

    // Singular extension — if TT move is much better than all alternatives,
    // extend its search by 1 ply
    bool singularExtension = false;
    if (!pvNode && ply > 0 && depth >= 6 && ttMove != NULL_MOVE) {
        TTResult tte2; bool tte2Hit = ttProbe(b.hash, tte2);
        if (tte2Hit && tte2.depth >= depth-3 && tte2.flag == TT_LOWER) {
            int singBeta = tte2.score - depth*2;
            // Search all moves EXCEPT the TT move at reduced depth
            // If none beat singBeta, the TT move is singular
            int singDepth = (depth-1)/2;
            Board bCopy = b;
            MoveList sml; genMoves(bCopy, sml);
            bool anyBeat = false;
            for (int i=0; i<sml.n && !anyBeat; i++) {
                if (sml.m[i] == ttMove) continue;
                UndoInfo su;
                if (!makeMove(bCopy, sml.m[i], su)) continue;
                int sc = -negamax(bCopy, singDepth, -singBeta-1, -singBeta, ply+1);
                unmakeMove(bCopy, sml.m[i], su);
                if (sc >= singBeta) anyBeat = true;
            }
            if (!anyBeat) singularExtension = true;
        }
    }

    // Reverse futility pruning (static eval cutoff)
    if (!pvNode && !inCheck && depth <= 3 && ply > 0) {
        int se = evaluatePos(b);
        if (se - 80*depth >= beta) return se;
    }

    // Null move pruning — skip our turn and see if opponent still can't beat beta
    if (!pvNode && !inCheck && !inNull && depth >= 3 && ply > 0) {
        // Don't null move in pawn/king-only positions (zugzwang risk)
        U64 nonPawns = b.pieces[b.turn][KNIGHT]|b.pieces[b.turn][BISHOP]
                      |b.pieces[b.turn][ROOK]  |b.pieces[b.turn][QUEEN];
        if (nonPawns) {
            // Make null move: just flip the turn
            UndoInfo nu;
            nu.movedPiece=-1; nu.capturedPiece=-1; nu.capturedColor=-1;
            nu.capturedSq=NO_SQ; nu.ep=b.ep; nu.castling=b.castling;
            nu.halfmove=b.halfmove; nu.hash=b.hash;

            if (b.ep != NO_SQ) b.hash ^= zEP[FILE_OF(b.ep)];
            b.ep = NO_SQ;
            b.turn = (Color)(1-b.turn);
            b.hash ^= zTurn;

            int R = (depth >= 6) ? 3 : 2;
            int nullSc = -negamax(b, depth-R-1, -beta, -beta+1, ply+1, true);

            // Restore
            b.turn = (Color)(1-b.turn);
            b.ep = nu.ep;
            b.hash = nu.hash;

            if (!stopNow && nullSc >= beta) return beta;
        }
    }

    MoveList ml; genMoves(b,ml);
    sortMoves(b, ml, ttMove, ply);

    int bestScore=-INF;
    Move bestMove=NULL_MOVE;
    int legalMoves=0;

    for(int i=0;i<ml.n;i++){
        int fl = MV_FLAGS(ml.m[i]);
        bool capture = isCapture(fl);

        // Late move pruning — skip quiet moves beyond LMP_LIMIT at low depths
        if (!inCheck && !pvNode && depth <= 3 && !capture && legalMoves >= LMP_LIMIT[depth])
            continue;

        // Futility pruning — skip quiet moves near the horizon that can't raise alpha
        if (!inCheck && depth <= 2 && !capture && legalMoves > 0) {
            int se = evaluatePos(b);
            if (se + 150*depth <= alpha) continue;
        }

        UndoInfo u;
        if(!makeMove(b,ml.m[i],u)) continue;
        legalMoves++;
        bool givesCheck = ::inCheck(b, b.turn);

        // Check extension + singular extension
        int extend = givesCheck ? 1 : (singularExtension && ml.m[i]==ttMove ? 1 : 0);

        // LMR — reduce late quiet moves
        int reduction = 0;
        if (depth >= 3 && legalMoves > 1 && !inCheck && !capture && !givesCheck) {
            int mi = min(legalMoves-1, 63);
            int di = min(depth-1, MAX_PLY-1);
            reduction = LMR[di][mi];
            if (pvNode) reduction--;
            reduction = max(0, min(reduction, depth-2));
        }

        int sc;
        int newDepth = depth - 1 - reduction + extend;
        if (legalMoves == 1) {
            sc = -negamax(b, depth-1+extend, -beta, -alpha, ply+1);
        } else {
            // Null window search
            sc = -negamax(b, newDepth, -alpha-1, -alpha, ply+1);
            // Re-search if LMR failed high
            if (reduction > 0 && sc > alpha && !stopNow)
                sc = -negamax(b, depth-1+extend, -alpha-1, -alpha, ply+1);
            // Re-search full window if PV move
            if (sc > alpha && sc < beta && !stopNow)
                sc = -negamax(b, depth-1+extend, -beta, -alpha, ply+1);
        }

        unmakeMove(b,ml.m[i],u);
        if(stopNow) return 0;
        if(sc>bestScore){ bestScore=sc; bestMove=ml.m[i]; }
        if(sc>alpha){
            alpha=sc;
            if(alpha>=beta){
                if(!capture && ply<MAX_PLY){
                    killers[1][ply]=killers[0][ply];
                    killers[0][ply]=ml.m[i];
                    // Countermove: record what refutes the previous move
                    if(ply>0){
                        Move prev=killers[0][ply-1];
                        if(prev) counterMove[MV_FROM(prev)][MV_TO(prev)]=ml.m[i];
                    }
                }
                if(!capture)
                    history[(int)b.turn][MV_FROM(ml.m[i])][MV_TO(ml.m[i])]+=depth*depth;
                break;
            }
        }
    }

    if(legalMoves==0) return inCheck ? -MATE+ply : 0;

    if (!stopNow) {
        TTFlag flag = (bestScore<=alphaOrig)?TT_UPPER:(bestScore>=beta)?TT_LOWER:TT_EXACT;
        ttStore(b.hash, bestScore, depth, bestMove, flag);
    }
    return bestScore;
}

// Root search — bestMove ALWAYS tracked here directly, never extracted from TT
Move search(Board& b, int wtime, int btime, int movestogo, int winc, int binc) {
    searchStart=chrono::steady_clock::now();
    stopNow=false;

    int totalTime = (b.turn==WHITE) ? wtime : btime;
    int timeMs  = max(0, totalTime - 200);  // subtract 200ms safety buffer
    int inc     = (b.turn==WHITE) ? winc  : binc;

    // ----------------------------------------------------------------
    // TIME MANAGEMENT (v18 revision)
    //
    // Bullet (<=60s): assume 30 moves remaining — positions resolve
    // faster, opponent moves quickly. Cap raised to 5% so the engine
    // uses available think time per move without going short.
    //
    // Blitz (60-300s): assume 40 moves, 4% cap.
    // Rapid+ (>300s):  assume 40 moves, 3% cap.
    // ----------------------------------------------------------------
    bool isBullet = (totalTime <= 60000);
    int defaultMoves = isBullet ? 30 : 40;
    int moves = (movestogo > 0) ? movestogo : defaultMoves;
    searchTimeMs = max(10, timeMs / moves + (inc * 3) / 5);

    // Hard cap — never spend more than N% of remaining clock on one move.
    // pctCap is the divisor: cap = timeMs/pctCap.
    //   Bullet  (<=60s):  20 => ~5%  => ~3s max on full 60s clock
    //   Blitz   (<=300s): 25 => ~4%  => ~12s max on full 5-min clock
    //   Rapid+  (>300s):  33 => ~3%  => ~18s max on full 10-min clock
    int pctCap;
    if      (isBullet)            pctCap = 20;
    else if (totalTime <= 300000) pctCap = 25;
    else                          pctCap = 33;
    searchTimeMs = min(searchTimeMs, timeMs / pctCap);
    searchTimeMs = max(searchTimeMs, 10);

    // Advanced passed pawn: one-shot +15% extension, only for our pawns
    // on rank 6+ (truly about to promote). Does not compound.
    {
        int bestAdv = 0;
        U64 pawns = b.pieces[(int)b.turn][PAWN];
        while (pawns) {
            Square psq = popLSB(pawns);
            int pr = RANK_OF(psq), pf = FILE_OF(psq);
            int adv = (b.turn==WHITE) ? pr : 7-pr;
            if (adv >= 5) {
                U64 pm = 0;
                if (b.turn==WHITE) { for(int rr=pr+1;rr<8;rr++){ if(pf>0)pm|=BIT(SQ(pf-1,rr)); pm|=BIT(SQ(pf,rr)); if(pf<7)pm|=BIT(SQ(pf+1,rr)); } }
                else               { for(int rr=pr-1;rr>=0;rr--){ if(pf>0)pm|=BIT(SQ(pf-1,rr)); pm|=BIT(SQ(pf,rr)); if(pf<7)pm|=BIT(SQ(pf+1,rr)); } }
                if (!(b.pieces[1-(int)b.turn][PAWN] & pm))
                    if (adv > bestAdv) bestAdv = adv;
            }
        }
        if (bestAdv >= 5)
            searchTimeMs = min((int)(searchTimeMs * 1.15), timeMs / pctCap);
    }
    int baseTime = searchTimeMs;  // save for instability extension

    // Reset per-search state
    memset(killers,     0, sizeof(killers));
    memset(history,     0, sizeof(history));
    memset(counterMove, 0, sizeof(counterMove));

    Move bestMove=NULL_MOVE;
    int  bestScore=0;

    for(int depth=1; depth<=64; depth++){
        // Generate all root moves once per depth
        MoveList ml; genMoves(b,ml);
        // Order root moves: put previous bestMove first
        sortMoves(b, ml, bestMove, 0);

        int  depthBestScore=-INF;
        Move depthBest=NULL_MOVE;
        int  alpha=-INF, beta=INF;

        // Aspiration windows from depth 4+
        if(depth>=4){
            alpha=max(-INF, bestScore-50);
            beta =min( INF, bestScore+50);
        }

        bool research=false;
        do {
            research=false;
            depthBestScore=-INF;
            depthBest=NULL_MOVE;

            for(int i=0;i<ml.n;i++){
                UndoInfo u;
                if(!makeMove(b,ml.m[i],u)) continue;
                int sc;
                if(depthBest==NULL_MOVE)
                    sc=-negamax(b,depth-1,-beta,-alpha,1);  // full window for first move
                else {
                    sc=-negamax(b,depth-1,-alpha-1,-alpha,1); // null window
                    if(sc>alpha && sc<beta && !stopNow)
                        sc=-negamax(b,depth-1,-beta,-alpha,1); // re-search
                }
                unmakeMove(b,ml.m[i],u);
                if(stopNow) goto done;
                if(sc>depthBestScore){ depthBestScore=sc; depthBest=ml.m[i]; }
                if(sc>alpha) alpha=sc;
                if(alpha>=beta) break;
            }

            // Aspiration window widening
            if(depth>=4){
                if(depthBestScore<=alpha-50){ alpha=max(-INF,alpha-100); research=true; }
                else if(depthBestScore>=beta+50){ beta=min(INF,beta+100);  research=true; }
            }
        } while(research && !stopNow);

        done:
        if(!stopNow && depthBest!=NULL_MOVE){
            // Instability extension: if score drops significantly from previous depth,
            // spend more time searching, but cap tightly relative to base time only.
            // Old cap was timeMs/6 which allowed 100s on a 10-min game.
            if(depth > 4 && bestMove != NULL_MOVE) {
                int scoreDrop = bestScore - depthBestScore;
                // Bullet: cap instability extension at +1500ms to avoid flagging.
                // Blitz/rapid: allow up to +5000ms / +3000ms as before.
                int extCap1 = isBullet ? baseTime + 1500 : baseTime + 5000;
                int extCap2 = isBullet ? baseTime + 800  : baseTime + 3000;
                if(scoreDrop > 30)       searchTimeMs = min(baseTime*3, extCap1);
                else if(scoreDrop > 15)  searchTimeMs = min(baseTime*2, extCap2);
                else                     searchTimeMs = baseTime;
            }
            bestMove  = depthBest;
            bestScore = depthBestScore;
            // Record ponder move: make bestMove, take first reply
            {
                UndoInfo pu; Board pb=b;
                if(makeMove(pb,bestMove,pu)){
                    MoveList pml; genMoves(pb,pml);
                    ponderMove=NULL_MOVE;
                    for(int pi=0;pi<pml.n;pi++){
                        UndoInfo pu2; if(makeMove(pb,pml.m[pi],pu2)){ponderMove=pml.m[pi];unmakeMove(pb,pml.m[pi],pu2);break;}
                    }
                    unmakeMove(pb,bestMove,pu);
                }
            }
            int elapsed=(int)chrono::duration_cast<chrono::milliseconds>(
                chrono::steady_clock::now()-searchStart).count();
            cerr<<"info depth "<<depth<<" score cp "<<bestScore
                <<" time "<<elapsed<<" pv "<<moveStr(bestMove)<<"\n";
        }

        int elapsed=(int)chrono::duration_cast<chrono::milliseconds>(
            chrono::steady_clock::now()-searchStart).count();
        if(elapsed>=searchTimeMs) break;
    }

    // Safety net
    if(bestMove==NULL_MOVE){
        MoveList ml; genMoves(b,ml);
        for(int i=0;i<ml.n;i++){
            UndoInfo u; if(makeMove(b,ml.m[i],u)){unmakeMove(b,ml.m[i],u);bestMove=ml.m[i];break;}
        }
    }
    return bestMove;
}

// ============================================================
// HELPERS: FEN, move string
// ============================================================
Board parseFEN(const string& fen) {
    Board b; b.clear();
    istringstream ss(fen);
    string board,turn,castle,ep,hm,fm;
    ss>>board>>turn>>castle>>ep>>hm>>fm;

    int sq=A8;
    for(char c:board){
        if(c=='/'){sq-=16;continue;}
        if(isdigit(c)){sq+=c-'0';continue;}
        Color col=isupper(c)?WHITE:BLACK;
        PieceType pt;
        switch(tolower(c)){
            case 'p':pt=PAWN;break; case 'n':pt=KNIGHT;break;
            case 'b':pt=BISHOP;break;case 'r':pt=ROOK;break;
            case 'q':pt=QUEEN;break; case 'k':pt=KING;break;
            default:sq++;continue;
        }
        b.place(col,pt,sq++);
    }
    b.turn=(turn=="w")?WHITE:BLACK;
    b.castling=0;
    if(castle!="-"){
        if(castle.find('K')!=string::npos)b.castling|=1;
        if(castle.find('Q')!=string::npos)b.castling|=2;
        if(castle.find('k')!=string::npos)b.castling|=4;
        if(castle.find('q')!=string::npos)b.castling|=8;
    }
    b.ep=NO_SQ;
    if(ep!="-"){b.ep=SQ(ep[0]-'a',ep[1]-'1');}
    b.halfmove=hm.empty()?0:stoi(hm);
    b.fullmove=fm.empty()?1:stoi(fm);
    b.hash=computeHash(b);
    b.histCount=0;
    posHistory[0]=b.hash;
    b.histCount=1;
    return b;
}

string moveStr(Move m){
    if(!m) return "0000";
    char s[6];
    s[0]='a'+FILE_OF(MV_FROM(m)); s[1]='1'+RANK_OF(MV_FROM(m));
    s[2]='a'+FILE_OF(MV_TO(m));   s[3]='1'+RANK_OF(MV_TO(m));
    int fl=MV_FLAGS(m);
    if(fl==PROMO_Q||fl==PROMO_CQ){s[4]='q';s[5]=0;}
    else if(fl==PROMO_R||fl==PROMO_CR){s[4]='r';s[5]=0;}
    else if(fl==PROMO_B||fl==PROMO_CB){s[4]='b';s[5]=0;}
    else if(fl==PROMO_N||fl==PROMO_CN){s[4]='n';s[5]=0;}
    else{s[4]=0;}
    return string(s);
}

Move parseMove(const Board& b, const string& s){
    MoveList ml; genMoves(const_cast<Board&>(b),ml);
    for(int i=0;i<ml.n;i++) if(moveStr(ml.m[i])==s) return ml.m[i];

    // Handle opponent rook/bishop underpromotions (we never generate these but
    // must parse them when the opponent plays one, e.g. "c2c1r" or "a7a8b").
    // Map to the queen promotion on the same squares — the pawn leaves the board
    // either way, so position tracking remains correct enough to continue playing.
    if(s.size()==5 && (s[4]=='r'||s[4]=='b')){
        string qprom = s.substr(0,4)+"q";
        for(int i=0;i<ml.n;i++) if(moveStr(ml.m[i])==qprom) return ml.m[i];
    }

    return NULL_MOVE;
}

// ============================================================
// UCI LOOP
// ============================================================
#ifdef _WIN32
// Set stack size to 8MB (matches Linux default) to prevent stack overflow
#pragma comment(linker, "/STACK:8388608")
#endif

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    tt = new TTEntry[TT_SIZE]();

#ifdef USE_NNUE
    // Auto-load nn.nnue if present in working directory
    nnueEnabled = nnueLoad("nn.nnue");
    if (!nnueEnabled) cerr<<"NNUE: nn.nnue not found, using classical eval\n";
#endif
    initAttacks();
    initMasks();
    initZobrist();
    initLMR();
    ttClear();

    int lastWt=60000, lastBt=60000, lastMtg=0, lastWi=0, lastBi=0; // saved from last go
    Board board = parseFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    string line;

    // Helper: stop any running search, join the thread, and output bestmove.
    // Always sets stopNow=true before joining to guarantee the search exits
    // quickly regardless of remaining time budget (critical for ponderhit).
    auto finishSearch = [&](){
        if (searchThread.joinable()) {
            stopNow = true;       // force search to exit at next timeUp() check
            searchThread.join();  // guaranteed to return quickly now
            stopNow = false;
            if (ponderMove != NULL_MOVE)
                cout << "bestmove " << moveStr(searchResult) << " ponder " << moveStr(ponderMove) << "\n";
            else
                cout << "bestmove " << moveStr(searchResult) << "\n";
            cout.flush();
        }
    };

    while(getline(cin,line)){
        istringstream ss(line);
        string cmd; ss>>cmd;

        if(cmd=="uci"){
            cout<<"id name SenkabalaIII v20\n"
                <<"id author Senkabala\n"
                <<"option name SyzygyPath type string default\n"
                <<"option name Hash type spin default 128 min 1 max 1024\n"
                <<"uciok\n";
        }
        else if(cmd=="isready"){
            // Must not be searching when we reply readyok
            stopNow=true;
            if(searchThread.joinable()) searchThread.join();
            stopNow=false;
            cout<<"readyok\n";
        }
        else if(cmd=="setoption"){
            string name, value, tok;
            while(ss>>tok){
                if(tok=="name") ss>>name;
                else if(tok=="value") { getline(ss,value);
                    if(!value.empty()&&value[0]==' ') value=value.substr(1);
                }
            }
            if(name=="NNUEPath" && !value.empty()){
#ifdef USE_NNUE
                nnuePath = value;
                nnueEnabled = nnueLoad(nnuePath.c_str());
                if(!nnueEnabled) cerr<<"NNUE: failed to load "<<nnuePath<<"\n";
#else
                cerr<<"NNUE: not compiled in (recompile with /DUSE_NNUE)\n";
#endif
            }
            else if(name=="SyzygyPath" && !value.empty()){
                syzygyPath=value;
                syzygyEnabled = tb_init(syzygyPath.c_str());
                if(syzygyEnabled)
                    cerr<<"Syzygy TB loaded from "<<syzygyPath<<" (max "<<TB_LARGEST<<" pieces)\n";
                else
                    cerr<<"Syzygy TB failed to load from "<<syzygyPath<<"\n";
            }
            else if(name=="Hash"){
                // Could resize TT here — for now ignore
            }
        }
        else if(cmd=="ucinewgame"){
            // Stop any running search first
            stopNow=true;
            if(searchThread.joinable()) searchThread.join();
            stopNow=false;
            board=parseFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
            ttClear();
            memset(killers,0,sizeof(killers));
            memset(history,0,sizeof(history));
            memset(counterMove,0,sizeof(counterMove));
            memset(posHistory,0,sizeof(posHistory));
        }
        else if(cmd=="position"){
            // Safe to update position — search is never running here in normal UCI flow
            // (position always comes before go, after stop/bestmove)
            string type; ss>>type;
            if(type=="startpos")
                board=parseFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
            else if(type=="fen"){
                string fen,tok;
                int fields=0;
                while(fields<6 && ss>>tok){
                    if(tok=="moves") break;
                    fen+=(fields?" ":"")+tok;
                    fields++;
                }
                board=parseFEN(fen);
            }
            string tok; bool moves=false;
            while(ss>>tok){
                if(tok=="moves"){moves=true;continue;}
                if(!moves) continue;
                Move m=parseMove(board,tok);
                if(m==NULL_MOVE){ cerr<<"info string bad move: "<<tok<<"\n"; break; }
                UndoInfo u;
                if(!makeMove(board,m,u)){ cerr<<"info string illegal move: "<<tok<<"\n"; break; }
            }
        }
        else if(cmd=="go"){
            // Stop any previous search (shouldn't be running, but be safe)
            stopNow=true;
            if(searchThread.joinable()) searchThread.join();
            stopNow=false;

            int wt=60000,bt=60000,mtg=0,wi=0,bi=0;
            bool doPonder=false;
            string tok;
            while(ss>>tok){
                if(tok=="wtime")      ss>>wt;
                else if(tok=="btime") ss>>bt;
                else if(tok=="movestogo") ss>>mtg;
                else if(tok=="winc")  ss>>wi;
                else if(tok=="binc")  ss>>bi;
                else if(tok=="ponder")   doPonder=true;
                else if(tok=="infinite") doPonder=true;
                else if(tok=="movetime"){ int mt; ss>>mt; wt=bt=mt+200; mtg=1; }
            }
            // Save time params so ponderhit can re-launch with the same budget
            lastWt=wt; lastBt=bt; lastMtg=mtg; lastWi=wi; lastBi=bi;

            pondering = doPonder;
            searchResult = NULL_MOVE;
            searchDone = false;

            // Capture go parameters and board by value for the thread
            Board boardCopy = board;
            int wt_=wt, bt_=bt, mtg_=mtg, wi_=wi, bi_=bi;

            // Launch search in background thread
            searchThread = thread([boardCopy,wt_,bt_,mtg_,wi_,bi_]() mutable {
                searchResult = search(boardCopy, wt_, bt_, mtg_, wi_, bi_);
                searchDone = true;
            });

            if (!doPonder) {
                // Normal search — block until done, then output bestmove
                searchThread.join();
                pondering = false;
                if(ponderMove!=NULL_MOVE)
                    cout<<"bestmove "<<moveStr(searchResult)<<" ponder "<<moveStr(ponderMove)<<"\n";
                else
                    cout<<"bestmove "<<moveStr(searchResult)<<"\n";
                cout.flush();
            }
            // If pondering: thread runs in background, UCI loop stays live
            // for "ponderhit" or "stop"
        }
        else if(cmd=="ponderhit"){
            // Opponent played our predicted move.
            // Stop the ponder search immediately (it was searching the ponder
            // position with infinite time — we don't want to use that result
            // directly since the position may differ from what we expected).
            // Re-launch a fresh search on the actual board position with the
            // saved time parameters from the last "go ponder" command.
            stopNow=true;
            pondering=false;
            if(searchThread.joinable()) searchThread.join();
            stopNow=false;

            searchResult = NULL_MOVE;
            searchDone = false;
            Board boardCopy = board;
            int wt_=lastWt, bt_=lastBt, mtg_=lastMtg, wi_=lastWi, bi_=lastBi;

            searchThread = thread([boardCopy,wt_,bt_,mtg_,wi_,bi_]() mutable {
                searchResult = search(boardCopy, wt_, bt_, mtg_, wi_, bi_);
                searchDone = true;
            });
            // Block until done, then output bestmove (same as normal go)
            searchThread.join();
            if(ponderMove!=NULL_MOVE)
                cout<<"bestmove "<<moveStr(searchResult)<<" ponder "<<moveStr(ponderMove)<<"\n";
            else
                cout<<"bestmove "<<moveStr(searchResult)<<"\n";
            cout.flush();
        }
        else if(cmd=="stop"){
            pondering=false;
            finishSearch();  // sets stopNow=true, joins, resets, outputs bestmove
        }
        else if(cmd=="quit"){
            stopNow=true;
            pondering=false;
            if(searchThread.joinable()) searchThread.join();
            break;
        }

        cout.flush();
    }
    return 0;
}

