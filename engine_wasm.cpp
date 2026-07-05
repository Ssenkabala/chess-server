/*
 * SenkabalaIII v20 — classical evaluation engine
 * (sliding-attack PEXT tables, NNUE-ready, WASM deployment layer)
 */

// Build stamp — printed to the console on engine_init(). Bump this whenever
// you recompile so you can confirm from the browser console EXACTLY which
// binary is live.
#define ENGINE_BUILD_ID "2026-07-05-sliding-mpvfix-1"

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
#include <cmath>
#include <vector>
#include <cmath>
#ifdef _MSC_VER
#include <intrin.h>
#include <cstdint>
static inline int __builtin_ctzll(unsigned long long x) {
    unsigned long i; _BitScanForward64(&i, x); return (int)i;
}
static inline int __builtin_clzll(unsigned long long x) {
    unsigned long i; _BitScanReverse64(&i, x); return 63 - (int)i;
}
static inline int __builtin_popcountll(unsigned long long x) {
    return (int)__popcnt64(x);
}
#endif
/* intrinsics replaced with GCC builtins */

using namespace std;

// ============================================================
// SYZYGY TABLEBASE INTERFACE
// Uses Fathom — a standalone Syzygy probe library
// 
// SETUP INSTRUCTIONS:
// 1. Download Fathom from: https://github.com/jdart1/Fathom
//    (just tbprobe.h and tbprobe.c are needed)
// 2. Place tbprobe.h + tbprobe.c in same folder as this file
// 3. Download Syzygy .rtbw files from: https://syzygy-tables.info
//    (3-4-5 piece = ~1GB, place in a folder e.g. C:/syzygy)
// 4. Compile: cl /O2 /EHsc /DUSE_SYZYGY tbprobe.c engine_v10.cpp /Fe:engine.exe
// 5. Set SyzygyPath in config.yml or pass via UCI setoption
//
// Without Fathom, compile normally and TB probing is silently disabled:
//    cl /O2 /EHsc engine_v10.cpp /Fe:engine.exe
// ============================================================
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


inline int lsb(U64 bb) {
    return __builtin_ctzll(bb);
}
inline int msb(U64 bb) {
    return 63 - __builtin_clzll(bb);
}
inline int popLSB(U64& bb) { int s=lsb(bb); bb&=bb-1; return s; }
inline int popcnt(U64 bb) {
    return __builtin_popcountll(bb);
}

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
// ---- loop-based sliding attacks: kept as reference / table builders ----
U64 rookAttSlow(Square sq, U64 occ) {
    U64 att=0; int r=RANK_OF(sq),f=FILE_OF(sq);
    for(int i=r+1;i<8;i++){int s=SQ(f,i);att|=BIT(s);if(occ&BIT(s))break;}
    for(int i=r-1;i>=0;i--){int s=SQ(f,i);att|=BIT(s);if(occ&BIT(s))break;}
    for(int i=f+1;i<8;i++){int s=SQ(i,r);att|=BIT(s);if(occ&BIT(s))break;}
    for(int i=f-1;i>=0;i--){int s=SQ(i,r);att|=BIT(s);if(occ&BIT(s))break;}
    return att;
}
U64 bishAttSlow(Square sq, U64 occ) {
    U64 att=0; int r=RANK_OF(sq),f=FILE_OF(sq);
    for(int i=1;r+i<8&&f+i<8;i++){int s=SQ(f+i,r+i);att|=BIT(s);if(occ&BIT(s))break;}
    for(int i=1;r+i<8&&f-i>=0;i++){int s=SQ(f-i,r+i);att|=BIT(s);if(occ&BIT(s))break;}
    for(int i=1;r-i>=0&&f+i<8;i++){int s=SQ(f+i,r-i);att|=BIT(s);if(occ&BIT(s))break;}
    for(int i=1;r-i>=0&&f-i>=0;i++){int s=SQ(f-i,r-i);att|=BIT(s);if(occ&BIT(s))break;}
    return att;
}

// ============================================================
// PEXT SLIDING ATTACKS (BMI2) — same results as the loop versions,
// computed with a single table lookup. Falls back to a portable
// software pext if BMI2 is unavailable (still correct, just slower).
// ============================================================
#if defined(__BMI2__) || defined(_MSC_VER)
#include <immintrin.h>
static inline U64 pext_u64(U64 v, U64 m){ return _pext_u64(v, m); }
#else
static inline U64 pext_u64(U64 v, U64 m){
    U64 r=0; int i=0;
    while(m){ U64 b=m&(~m+1); if(v&b) r|=(1ULL<<i); i++; m&=m-1; }
    return r;
}
#endif

U64 RookMask[64], BishopMask[64];
U64 RookTable[64][4096];    // 2^12 max
U64 BishopTable[64][512];   // 2^9  max

static U64 rookMaskCalc(int sq){
    U64 m=0; int r=RANK_OF(sq),f=FILE_OF(sq);
    for(int i=r+1;i<=6;i++) m|=BIT(SQ(f,i));
    for(int i=r-1;i>=1;i--) m|=BIT(SQ(f,i));
    for(int i=f+1;i<=6;i++) m|=BIT(SQ(i,r));
    for(int i=f-1;i>=1;i--) m|=BIT(SQ(i,r));
    return m;
}
static U64 bishopMaskCalc(int sq){
    U64 m=0; int r=RANK_OF(sq),f=FILE_OF(sq);
    for(int i=1;r+i<=6&&f+i<=6;i++) m|=BIT(SQ(f+i,r+i));
    for(int i=1;r+i<=6&&f-i>=1;i++) m|=BIT(SQ(f-i,r+i));
    for(int i=1;r-i>=1&&f+i<=6;i++) m|=BIT(SQ(f+i,r-i));
    for(int i=1;r-i>=1&&f-i>=1;i++) m|=BIT(SQ(f-i,r-i));
    return m;
}
void initSliders(){
    for(int sq=0;sq<64;sq++){
        RookMask[sq]   = rookMaskCalc(sq);
        BishopMask[sq] = bishopMaskCalc(sq);
        U64 m = RookMask[sq], sub=0;
        do { RookTable[sq][pext_u64(sub,m)] = rookAttSlow(sq,sub); sub=(sub-m)&m; } while(sub);
        m = BishopMask[sq]; sub=0;
        do { BishopTable[sq][pext_u64(sub,m)] = bishAttSlow(sq,sub); sub=(sub-m)&m; } while(sub);
    }
}
static inline U64 rookAtt(Square sq, U64 occ){ return RookTable[sq][pext_u64(occ, RookMask[sq])]; }
static inline U64 bishAtt(Square sq, U64 occ){ return BishopTable[sq][pext_u64(occ, BishopMask[sq])]; }
U64 queenAtt(Square sq,U64 occ){return rookAtt(sq,occ)|bishAtt(sq,occ);}

// Verify fast == slow across all squares and many random occupancies.
static bool verifySliders(){
    U64 st=0x123456789abcdefULL;
    auto rng=[&](){ st^=st>>12; st^=st<<25; st^=st>>27; return st*0x2545F4914F6CDD1DULL; };
    for(int sq=0;sq<64;sq++){
        for(int t=0;t<20000;t++){
            U64 occ = rng() & rng();  // sparse-ish random occupancy
            if(rookAtt(sq,occ)!=rookAttSlow(sq,occ)) return false;
            if(bishAtt(sq,occ)!=bishAttSlow(sq,occ)) return false;
        }
    }
    return true;
}

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
#ifdef TUNE
  #define WCONST
#else
  #define WCONST const
#endif

// ===== WAVE 2 TUNABLE SCALARS (defaults = old hardcoded values) =====
WCONST int MOB_N_MG = -3,  MOB_N_EG = 10;    // knight mobility multiplier
WCONST int MOB_B_MG = 4,  MOB_B_EG = 9;    // bishop mobility
WCONST int MOB_R_MG = 6,  MOB_R_EG = 4;    // rook mobility
WCONST int MOB_Q_MG = 0,  MOB_Q_EG = 22;    // queen mobility
WCONST int BISHOP_PAIR_MG = 11, BISHOP_PAIR_EG = 68;
WCONST int ROOK_OPEN_MG = 53, ROOK_OPEN_EG = 33;   // rook on fully open file
WCONST int ROOK_SEMI_MG = 37, ROOK_SEMI_EG = 51;   // rook on semi-open file
WCONST int DOUBLED_MG = -3, DOUBLED_EG = -1;
WCONST int ISOLATED_MG = -9, ISOLATED_EG = -36;
WCONST int TEMPO = 43;
// ====================================================================

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
            if (popcnt(ourPawns & file) > 1) { mg+=mul*DOUBLED_MG; eg+=mul*DOUBLED_EG; }

            // Isolated pawn penalty
            U64 adjFiles = 0;
            if (f>0) adjFiles |= fileMask[f-1];
            if (f<7) adjFiles |= fileMask[f+1];
            if (!(ourPawns & adjFiles)) { mg+=mul*ISOLATED_MG; eg+=mul*ISOLATED_EG; }

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
                if (advance >= 6) { eg += mul * 200; mg += mul * 150; } // rank 7: promotes next move — treat as near-queen
                // Promotion square threat — if nothing blocks, this IS a queen
                // Give it queen-like value so the engine prioritises stopping/promoting
                if (advance >= 6) {
                    // Check if promotion square is reachable (no blocker)
                    Square promoSq = (c==WHITE) ? SQ(f,7) : SQ(f,0);
                    Square nextSq  = (c==WHITE) ? SQ(f,r+1) : SQ(f,r-1);
                    if (!(b.occ[2] & BIT(nextSq))) {
                        eg += mul * 300; mg += mul * 250; // effectively a queen threat
                    }
                }
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
#ifdef TUNE
    evalPawnStructure(b, mg, eg); return;   // no caching while tuning
#endif
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
WCONST int MAT_MG[6]= {41, 315, 348, 429, 1027, 0};
WCONST int MAT_EG[6]= {74, 351, 393, 631, 1078, 0};

// Game phase weights (max phase = 24)
const int PHASE_W[6] = {0, 1, 1, 2, 4, 0};
const int MAX_PHASE  = 24;

// PST tables [mg][eg] — white relative (a1=0)
WCONST int PST_MG[6][64] = {
// PAWN mg
{0,0,0,0,0,0,0,0,-6,0,-5,-96,-3,18,36,-29,8,-10,-6,-8,5,10,27,4,-3,-1,-5,33,55,24,27,-8,19,18,51,58,66,57,16,1,-28,-56,-2,-58,-58,41,-16,-24,-38,-38,-38,-38,-38,-38,-38,-38,0,0,0,0,0,0,0,0},
// KNIGHT mg
{-94,-79,3,-25,35,-12,-34,8,-118,-108,-56,-3,-8,-4,36,45,-107,-26,-7,24,27,15,-1,-44,-32,-61,56,-26,45,55,-36,16,54,-24,85,42,-4,45,21,58,9,40,8,53,0,77,-43,9,-52,61,88,60,26,-26,-14,-81,-61,-113,-16,-23,-3,-110,-4,-126},
// BISHOP mg
{-2,-37,23,12,-2,-21,34,63,45,30,-1,23,3,7,64,20,22,17,28,15,27,19,24,-14,32,-15,-21,36,38,26,-24,19,-2,-11,6,-23,7,-3,14,-49,22,53,-76,-16,-75,-78,-21,10,-50,36,-10,-35,-76,20,-72,38,-72,-52,-31,-93,-46,-98,16,-5},
// ROOK mg
{-14,-25,4,17,10,29,-42,-13,-9,-78,-42,-55,-58,10,-60,-91,-29,-32,-18,-18,10,28,-18,-28,-18,-27,36,-13,-15,-9,-26,2,50,-53,33,2,36,43,-73,38,77,65,28,74,62,70,48,83,41,-6,73,69,60,87,76,82,-28,-2,-40,-47,-30,14,47,86},
// QUEEN mg
{-5,8,-14,35,-9,4,-22,-9,-11,27,41,24,36,41,-43,-84,0,-20,49,15,35,33,9,-27,-10,-35,20,21,24,2,-22,7,53,15,-5,20,18,67,-52,25,9,-38,9,52,-45,23,56,-3,49,-11,-34,35,22,34,2,19,36,21,-40,3,37,65,37,59},
// KING mg
{27,63,38,-46,46,-32,76,53,48,-3,-60,3,-32,50,60,21,-68,-44,-33,-8,-10,-19,-32,-68,-84,13,-50,-108,-52,-45,-41,-96,-103,-41,-100,-65,-81,-49,16,-24,-58,43,-7,-71,-69,43,45,6,-69,27,-7,-64,10,-19,40,-3,-89,-31,-17,2,-12,-28,-28,21}
};

WCONST int PST_EG[6][64] = {
// PAWN eg
{0,0,0,0,0,0,0,0,21,16,9,-53,19,38,27,16,3,15,12,-29,-39,17,2,-15,39,26,19,-51,-15,15,23,4,39,20,8,-34,-29,9,39,6,-29,-41,-73,-68,-68,-29,-14,-19,-68,-63,-58,-48,-48,-58,-63,-68,0,0,0,0,0,0,0,0},
// KNIGHT eg
{-89,-102,-76,26,-22,-4,29,-47,-51,13,43,-19,-2,27,19,-89,7,-24,11,47,26,11,62,-12,-6,30,-21,68,47,36,45,-74,-45,52,59,28,63,85,7,46,27,43,36,38,46,54,47,8,-59,-37,6,89,-37,45,-56,-6,-117,-128,36,-36,-59,-36,-23,-83},
// BISHOP eg
{-101,30,-1,15,-10,-19,12,-27,51,12,48,40,35,77,9,27,56,20,86,40,74,65,36,5,14,44,80,69,15,70,6,43,6,77,93,49,98,-9,60,61,17,46,11,88,-9,55,86,30,17,58,35,18,68,65,79,27,-9,74,64,16,29,-41,-34,16},
// ROOK eg
{4,3,-2,3,-7,-5,-14,-10,-19,-45,-18,-7,-2,-32,-24,-4,34,-2,0,-5,-41,13,-21,-33,23,41,-15,7,-21,12,16,-11,68,60,79,10,35,23,34,64,80,79,84,46,55,34,59,67,24,26,10,10,14,-3,47,27,18,51,44,18,30,14,36,87},
// QUEEN eg
{-30,-75,-83,-41,-46,-58,-73,-37,10,16,-45,-3,-6,-16,-46,-63,15,11,18,39,2,49,10,-14,19,-6,-7,41,56,62,-31,35,49,53,49,76,87,2,76,23,57,58,72,75,85,84,78,50,51,71,30,66,70,82,77,58,62,73,58,80,76,58,32,64},
// KING eg
{-37,-35,-10,2,-53,-40,-25,-83,-41,4,18,-20,-18,-13,9,-24,-44,-7,-26,-17,-14,-2,13,-1,1,-46,-14,-18,-13,9,7,-31,-41,29,22,-1,9,73,78,40,58,78,104,64,65,108,78,58,-117,58,75,71,71,49,58,56,-138,53,-60,49,53,-33,57,-51}
};

int evaluate(const Board& b) {
    int mg=0, eg=0, phase=0;

    // Tempo bonus — small reward for side to move
    mg += (b.turn==WHITE) ? TEMPO : -TEMPO;

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
        if (popcnt(b.pieces[c][BISHOP]) >= 2) { mg += mul*BISHOP_PAIR_MG; eg += mul*BISHOP_PAIR_EG; }

        // Rook on open/semi-open file + rook on 7th rank + endgame patterns
        U64 rooks = b.pieces[c][ROOK], tmp=rooks;
        int rank7 = (c==WHITE) ? 6 : 1;
        int rank8 = (c==WHITE) ? 7 : 0;
        U64 enemyKing = b.pieces[1-c][KING];
        Square ekSq = enemyKing ? lsb(enemyKing) : NO_SQ;
        while (tmp) {
            Square sq = popLSB(tmp);
            int f = FILE_OF(sq), r = RANK_OF(sq);
            U64 file = fileMask[f];
            bool noOurs   = !(b.pieces[c][PAWN]   & file);
            bool noTheirs = !(b.pieces[1-c][PAWN] & file);
            if (noOurs && noTheirs) { mg += mul*ROOK_OPEN_MG; eg += mul*ROOK_OPEN_EG; }
            else if (noOurs)        { mg += mul*ROOK_SEMI_MG; eg += mul*ROOK_SEMI_EG; }
            // Rook on 7th
            if (r==rank7 && (b.pieces[1-c][PAWN] || (enemyKing && RANK_OF(lsb(enemyKing))==rank8)))
                { mg += mul*20; eg += mul*30; }

            // Tarrasch rule: rook behind own passed pawn
            // Use precomputed passed pawn mask (passed_bb computed earlier if available)
            // Simple check: any own pawn on same file ahead of rook with no enemy pawn ahead
            {
                U64 ownPawnsOnFile = b.pieces[c][PAWN] & file;
                while(ownPawnsOnFile) {
                    Square psq = popLSB(ownPawnsOnFile);
                    int pr = RANK_OF(psq);
                    bool rookBehind = (c==WHITE) ? (r < pr) : (r > pr);
                    if(rookBehind) {
                        // Quick passed pawn check: no enemy pawns on file or adjacent files ahead
                        U64 aheadMask = (c==WHITE)
                            ? (fileMask[f] & ~((BIT(SQ(f,pr))-1)|BIT(SQ(f,pr))))
                            : (fileMask[f] & ((BIT(SQ(f,pr)))-1));
                        U64 adjFiles = fileMask[f];
                        if(f>0) adjFiles |= fileMask[f-1];
                        if(f<7) adjFiles |= fileMask[f+1];
                        U64 enemyAhead = b.pieces[1-c][PAWN] & adjFiles & aheadMask;
                        if(!enemyAhead) { eg += mul*25; mg += mul*15; }
                    }
                }
            }

            // Rook cutting off enemy king (enemy king can't cross this file/rank)
            // Bonus when rook is on same rank as enemy king or cuts off escape
            if(ekSq != NO_SQ) {
                int ekf = FILE_OF(ekSq), ekr = RANK_OF(ekSq);
                // Rook cuts off king on file (king can't advance/retreat)
                if(f == ekf) eg += mul*15;
                // Rook on same rank as enemy king — lateral cut
                if(r == ekr) eg += mul*10;
                // Rook restricts king to edge (king on rank 0/7/file 0/7)
                bool kingOnEdge = (ekr==0||ekr==7||ekf==0||ekf==7);
                if(kingOnEdge && (f==ekf || r==ekr)) eg += mul*20;
            }
        }

        // Penalty for rook passivity — rook on same rank/file as own king with no open lines
        // This catches the shuffling pattern (Rb1-Rb2-Rb1 oscillation)
        U64 ourKing = b.pieces[c][KING];
        if(ourKing && b.pieces[c][ROOK]) {
            Square kSq = lsb(ourKing);
            U64 rr2 = b.pieces[c][ROOK];
            while(rr2) {
                Square rsq = popLSB(rr2);
                // Rook on same file as own king with no passed pawn — passive
                if(FILE_OF(rsq)==FILE_OF(kSq) && !b.pieces[c][PAWN]) {
                    eg += mul*(-8); // small passivity penalty
                }
            }
        }

        // Mobility
        U64 all = b.occ[2], myPieces = b.occ[c];
        U64 kn=b.pieces[c][KNIGHT];
        while(kn){Square sq=popLSB(kn);int km=popcnt(knightAtt[sq]&~myPieces)-4;mg+=mul*km*MOB_N_MG;eg+=mul*km*MOB_N_EG;}
        U64 bi=b.pieces[c][BISHOP];
        while(bi){Square sq=popLSB(bi);int bm=popcnt(bishAtt(sq,all)&~myPieces)-7;mg+=mul*bm*MOB_B_MG;eg+=mul*bm*MOB_B_EG;}
        U64 ro=b.pieces[c][ROOK];
        while(ro){Square sq=popLSB(ro);int rm=popcnt(rookAtt(sq,all)&~myPieces)-7;mg+=mul*rm*MOB_R_MG;eg+=mul*rm*MOB_R_EG;}
        U64 qu2=b.pieces[c][QUEEN];
        while(qu2){Square sq=popLSB(qu2);int qm=popcnt(queenAtt(sq,all)&~myPieces)-14;mg+=mul*qm*MOB_Q_MG;eg+=mul*qm*MOB_Q_EG;}

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

            // Extended king danger (skip in pure endgames where king is active)
            if(phase > MAX_PHASE/4) {
            // This catches threats like Rb1 with king on b7 that zone-check misses
            {
                U64 kfile = fileMask[kf];
                U64 krank = 0; for(int ff=0;ff<8;ff++) krank|=BIT(SQ(ff,kr));
                U64 enemyRQ = b.pieces[1-c][ROOK] | b.pieces[1-c][QUEEN];
                U64 enemyBQ = b.pieces[1-c][BISHOP] | b.pieces[1-c][QUEEN];
                // Rook/queen on same file — X-ray pressure
                if (enemyRQ & kfile) attackWeight += 4;
                // Rook/queen on same rank
                if (enemyRQ & krank) attackWeight += 4;
                // Bishop/queen on same diagonal
                U64 diagAtt = bishAtt(ksq, all);
                if (diagAtt & enemyBQ) attackWeight += 3;
                // Extra weight if king is trapped on back rank with enemy rook on file
                bool kingOnEdge = (kr == 0 || kr == 7 || kf == 0 || kf == 7);
                if (kingOnEdge && (enemyRQ & kfile)) attackWeight += 3;
                if (kingOnEdge && (enemyRQ & krank)) attackWeight += 3;
            }
            } // end extended king danger

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
            int egWeight = MAX_PHASE - min(phase, MAX_PHASE);
            bool noQueens = (b.pieces[WHITE][QUEEN] == 0 && b.pieces[BLACK][QUEEN] == 0);

            if (egWeight > 4) {  // activate king earlier (was > 8)
                int centreDist = abs(kf - 3) + abs(kr - 3);
                centreDist = min(centreDist, abs(kf - 4) + abs(kr - 4));

                // Sharper scaling — king activity matters much more in endgame
                int kingBonus = (6 - centreDist) * (egWeight / 3);  // was /4
                if (noQueens) kingBonus += (6 - centreDist) * (egWeight / 2);  // was /3
                eg += mul * kingBonus;

                // Extra bonus for king advancing toward enemy pawns in pawnless endgame
                if (!b.pieces[1-c][PAWN] && b.pieces[c][PAWN]) {
                    // King closer to enemy king = better in K+P vs K
                    if(enemyKing && ekSq != NO_SQ) {
                        int kDist = abs(kf - FILE_OF(ekSq)) + abs(kr - RANK_OF(ekSq));
                        eg += mul * (10 - kDist) * (egWeight / 4);
                    }
                }

                // Penalty for king staying on back rank in endgame (passive king)
                int backRank = (c==WHITE) ? 0 : 7;
                if(kr == backRank && egWeight > 8) eg += mul * (-20);
            }
        }
    }

    // Bishop color complex — only run in endgame (saves time in middlegame)
    if(phase < MAX_PHASE/2) {
        const U64 lightSqs=0x55AA55AA55AA55AAULL;
        for(int c=0;c<2;c++){
            int mul=(c==WHITE)?1:-1;
            U64 bish=b.pieces[c][BISHOP];
            U64 pawns=b.pieces[c][PAWN];
            if(!bish || !pawns) continue;
            Square bsq=lsb(bish);
            bool bishLight=((FILE_OF(bsq)+RANK_OF(bsq))%2==0);
            U64 bishColorMask = bishLight ? lightSqs : ~lightSqs;
            U64 offColor = pawns & ~bishColorMask;
            int nOff = popcnt(offColor);
            if(nOff >= 2) eg += mul*(-15*nOff);
            int nEnemyOn = popcnt(b.pieces[1-c][PAWN] & bishColorMask);
            if(nEnemyOn >= 2) eg += mul*(-10*nEnemyOn);
        }
    }

    // Mating net — only compute when material is reduced (saves time)
    if(popcnt(b.occ[2]) <= 12) {
    for (int c=0; c<2; c++) {
        int mul=(c==WHITE)?1:-1;
        U64 ek=b.pieces[1-c][KING];
        if (!ek) continue;
        Square eks=lsb(ek);
        bool hasQueen = b.pieces[c][QUEEN] != 0;
        bool hasRook  = b.pieces[c][ROOK]  != 0;
        if (hasQueen || hasRook) {
            // Count king escape squares
            U64 escapes = kingAtt[eks] & ~b.occ[1-c];
            int escapeSqs = popcnt(escapes);
            if (escapeSqs <= 2) {
                // King nearly trapped — bonus scales with how trapped it is
                int netBonus = (3 - escapeSqs) * 30;
                if (hasQueen && hasRook) netBonus *= 2; // Q+R = very dangerous
                mg += mul * netBonus;
                eg += mul * netBonus;
            }
        }
    }
    } // end mating net

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

    // Lucena/Philidor — only when very few pieces (pure rook endgame)
    if(popcnt(b.occ[2]) <= 8) {
    for(int c=0;c<2;c++){
        int mul=(c==WHITE)?1:-1;
        // Lucena: we have R+P, they have R only
        bool weHaveRP = (popcnt(b.pieces[c][ROOK])==1 && popcnt(b.pieces[c][PAWN])>=1
                         && !b.pieces[c][QUEEN] && !b.pieces[c][BISHOP] && !b.pieces[c][KNIGHT]);
        bool theyHaveR = (popcnt(b.pieces[1-c][ROOK])==1 && !b.pieces[1-c][QUEEN]
                          && !b.pieces[1-c][PAWN] && !b.pieces[1-c][BISHOP] && !b.pieces[1-c][KNIGHT]);
        if(weHaveRP && theyHaveR){
            U64 ourK=b.pieces[c][KING];
            U64 pp=b.pieces[c][PAWN];
            if(ourK && pp){
                Square ksq=lsb(ourK), psq=lsb(pp);
                int kr2=RANK_OF(ksq), kf2=FILE_OF(ksq);
                int pr=RANK_OF(psq), pf=FILE_OF(psq);
                // Bonus if our king is in front of our pawn (Lucena position)
                bool kingInFront = (c==WHITE) ? (kr2 > pr && kf2==pf)
                                              : (kr2 < pr && kf2==pf);
                if(kingInFront) eg += mul*50;
                // Bonus for advanced pawn
                int advance=(c==WHITE)?pr:(7-pr);
                eg += mul*advance*12;
                // Bonus if king escorts pawn (adjacent)
                int dist=abs(kr2-pr)+abs(kf2-pf);
                if(dist<=1) eg += mul*30;
                else eg += mul*(6-min(dist,6))*8;
            }
        }
    }
    } // end Lucena

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

// Search score constants — defined here (rather than lower in SEARCH
// GLOBALS) because the TT store/probe below need MATE_THRESHOLD to adjust
// mate scores by ply. Real evaluations never approach MATE_THRESHOLD, so
// any score at or beyond it is a forced-mate score.
const int INF=1000000, MATE=999000;
const int MAX_PLY=64;
const int MATE_THRESHOLD = 900000;

inline bool isMateScore(int score) {
    return std::abs(score) >= MATE_THRESHOLD;
}
// Converts a raw mate score (MATE - ply, or -MATE + ply) into a signed
// "mate in N" move count — positive if the side to move at the score's
// origin delivers mate, negative if they get mated. Standard UCI semantics.
inline int mateMovesFromScore(int score) {
    int absScore = std::abs(score);
    int plyToMate = MATE - absScore + 1;
    int n = (plyToMate + 1) / 2;
    return score > 0 ? n : -n;
}

// ── Mate-score TT adjustment ──────────────────────────────────────────
// Mate scores are stored as (MATE - distance_from_root). But the TT is
// keyed by position, and the SAME position can be probed at a different
// ply than it was stored at (transpositions, and iterative deepening
// re-searching the same line). If a root-relative mate score is written
// and later read back at a different ply, it's wrong — and it drifts
// toward MATE-1 ("mate in 1"), which then dominates every deeper search
// via the TT and poisons the result. This exact bug was found and fixed
// in the WASM build earlier — this file predates that fix (no ply
// parameter on ttStore/ttProbe at all), and it's the same TT here.
//
// Fix: store mate scores relative to the CURRENT node (add ply on the way
// in), translate back to root-relative on the way out (subtract ply).
// Non-mate scores pass through untouched.
inline int scoreToTT(int score, int ply) {
    if (score >=  MATE_THRESHOLD) return score + ply;
    if (score <= -MATE_THRESHOLD) return score - ply;
    return score;
}
inline int scoreFromTT(int score, int ply) {
    if (score >=  MATE_THRESHOLD) return score - ply;
    if (score <= -MATE_THRESHOLD) return score + ply;
    return score;
}

enum TTFlag { TT_EXACT, TT_LOWER, TT_UPPER };
struct TTEntry { U64 hash; int score, depth; Move bestMove; TTFlag flag; };
#ifdef __EMSCRIPTEN__
const int TT_SIZE     = 1<<20;  // WASM: 1M entries ~20MB — main table: pvIdx==0 and single-PV (bot play) only
const int TT_SIZE_MPV = 1<<15;  // WASM: 32K entries ~640KB — scratch table: excluded-move (pvIdx>=1) searches only
#else
const int TT_SIZE     = 1<<23;  // Native: 8M entries ~160MB — main table: pvIdx==0 and single-PV (bot play) only
const int TT_SIZE_MPV = 1<<18;  // Native: 256K entries ~5MB — scratch table: excluded-move (pvIdx>=1) searches only
#endif
TTEntry* tt  = nullptr;
TTEntry* tt2 = nullptr;

// ttStore/ttProbe read through activeTT/activeTTMask rather than a
// hardcoded table. This is what lets the multipv loop isolate excluded-move
// (pvIdx>=1) searches into their OWN table without touching negamax's
// signature at all.
//
// Why this exists: pvIdx=0's deep exploration of the true best line fills
// the (shared) TT with many entries, including mate scores correctly
// ply-adjusted for THAT search's own recursion. When pvIdx>=1 then explores
// a totally different, unrelated root move, its recursion can transpose
// into (or hash-collide with) a position that only has a real mate
// available via a forcing sequence pvIdx=0 found — not one the excluded
// move actually forces. The cached score gets read back and ply-adjusted
// "correctly" for that isolated read, but it doesn't describe what the
// excluded move actually achieves. Confirmed live: fully clearing the TT
// before every pvIdx>=1 search eliminated the corruption outright (a
// non-checking rook move stopped reporting "mate in 1"), which pinned the
// mechanism down precisely — but a full clear of the 160MB main table on
// every secondary line, every depth, is far too expensive to ship. A
// separate, much smaller table for exactly those searches gets the same
// isolation without that cost, and never touches the main table pvIdx==0
// and single-PV (bot play) depend on — those two are completely unaffected
// by any of this.
TTEntry* activeTT     = nullptr;
int      activeTTMask = 0;

void ttClear() {
    if(tt)  memset(tt,  0, TT_SIZE*sizeof(TTEntry));
    if(tt2) memset(tt2, 0, TT_SIZE_MPV*sizeof(TTEntry));
}

struct TTResult { int score, depth, flag; Move bestMove; };

bool ttProbe(U64 hash, TTResult& out, int ply) {
    if (!activeTT) return false;
    TTEntry* e = &activeTT[hash & activeTTMask];
    if (e->hash != hash) return false;
    out.score    = scoreFromTT(e->score, ply);
    out.depth    = e->depth;
    out.flag     = (int)e->flag;
    out.bestMove = e->bestMove;
    return true;
}

void ttStore(U64 hash, int score, int depth, Move best, TTFlag flag, int ply) {
    if (!activeTT) return;
    TTEntry* e = &activeTT[hash & activeTTMask];
    if (e->hash == hash || depth >= e->depth) {
        if (best == NULL_MOVE && e->hash == hash) best = e->bestMove;
        e->hash     = hash;
        e->score    = scoreToTT(score, ply);
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
        // Defensive guard: a NaN or Inf produced somewhere inside
        // nnueForward's floating-point math (malformed/corrupted weights,
        // an overflow in the forward pass, etc.) converting to `int` is
        // undefined behavior — it will NOT reliably be caught by a normal
        // range check, since the resulting int can be anything, including
        // a value that happens to land in or near MATE_THRESHOLD and gets
        // misread as a forced mate downstream. Since nnueForward's return
        // is already `int` here (whatever conversion happens is inside
        // nnue.h, which this file doesn't have visibility into), the most
        // robust check available at this boundary is a plausibility range:
        // no real chess evaluation is anywhere close to a mate score.
        // Falling back to the classical eval (always sane, pure integer
        // arithmetic, no floating point at all) is far safer than trusting
        // a value this implausible. This does NOT fix whatever is actually
        // producing the bad value inside nnue.h — it only stops it from
        // corrupting search/comparisons elsewhere. The real fix needs
        // nnue.h itself.
        if (score <= -MATE_THRESHOLD || score >= MATE_THRESHOLD) {
            return evaluate(b);
        }
        // score is already from side-to-move perspective (extractFeatures flips for black)
        return score;
    }
#endif
    return evaluate(b);
}

// ============================================================
// SEARCH GLOBALS
// ============================================================
// INF, MATE, MAX_PLY, MATE_THRESHOLD are defined earlier (just above the
// TT code, which needs MATE_THRESHOLD for mate-score adjustment).

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
bool nativeMovetimeRequest = false;  // true only for the duration of a "go movetime N"
                                      // search — see the movetime-bypass branch in search()
atomic<bool> pondering{false};  // true while pondering (infinite search until "stop")
Move ponderMove = NULL_MOVE;    // the expected opponent reply we're pondering on

// ── WASM / MultiPV support ──────────────────────────────────────────
// wasmTimerPreset: set true only by the Emscripten entry points
// (engine_best_move/engine_analyse) right before calling search(), so the
// clock-percentage time formula is bypassed in favour of an exact
// millisecond budget from JS. Always false in a native build (no other
// code ever sets it), so this has zero effect on native UCI play.
bool wasmTimerPreset = false;

// Exposes the top root moves (in true multipv order) so a caller — the
// WASM analysis path — can extract PVs using its OWN clean board copy,
// never search()'s internal one (which can carry corrupted continuation
// state across its own recursive calls; a freshly re-parsed board sidesteps
// that entirely). Populated once per depth, only on a depth that completed
// without being cut off by time.
const int MAX_EXPOSED_ROOT_MOVES = 5;
Move exposedRootMoves[MAX_EXPOSED_ROOT_MOVES];
int  exposedRootScores[MAX_EXPOSED_ROOT_MOVES];
int  exposedRootMoveCount = 0;

int  multiPVCount = 1;  // UCI MultiPV option — how many best moves to report

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
// ============================================================
// SEE — Static Exchange Evaluation
// ============================================================
static const int SEE_VAL[6] = {100, 300, 320, 500, 900, 20000};
int see(const Board& b, Square to, Square from, int fromPt) {
    int gain[32]; int d=0;
    U64 occ=b.occ[2];
    gain[d]=(b.mailbox[to]>=0)?SEE_VAL[b.mailbox[to]]:0;
    occ&=~BIT(from);
    auto attackers=[&](U64 occ2)->U64{
        U64 att=0;
        att|=knightAtt[to]&(b.pieces[WHITE][KNIGHT]|b.pieces[BLACK][KNIGHT]);
        att|=kingAtt[to]&(b.pieces[WHITE][KING]|b.pieces[BLACK][KING]);
        att|=pawnAtt[WHITE][to]&b.pieces[BLACK][PAWN];
        att|=pawnAtt[BLACK][to]&b.pieces[WHITE][PAWN];
        att|=rookAtt(to,occ2)&(b.pieces[WHITE][ROOK]|b.pieces[BLACK][ROOK]|b.pieces[WHITE][QUEEN]|b.pieces[BLACK][QUEEN]);
        att|=bishAtt(to,occ2)&(b.pieces[WHITE][BISHOP]|b.pieces[BLACK][BISHOP]|b.pieces[WHITE][QUEEN]|b.pieces[BLACK][QUEEN]);
        return att&occ2;
    };
    U64 att=attackers(occ); Color side=(Color)(1-b.turn); int pt=fromPt;
    while(true){
        d++; gain[d]=SEE_VAL[pt]-gain[d-1];
        U64 sa=att&b.occ[(int)side]; if(!sa) break;
        int lp=-1; Square ls=NO_SQ;
        for(int p=PAWN;p<=KING;p++){U64 x=sa&b.pieces[(int)side][p];if(x){lp=p;ls=lsb(x);break;}}
        if(lp<0) break;
        occ&=~BIT(ls); att=attackers(occ); side=(Color)(1-(int)side); pt=lp;
    }
    while(--d) gain[d-1]=-max(-gain[d-1],gain[d]);
    return gain[0];
}
inline bool seeGE(const Board& b, Move m, int threshold=0){
    int fl=MV_FLAGS(m); if(!isCapture(fl)) return true;
    Square from=MV_FROM(m),to=MV_TO(m);
    int fp=b.mailbox[from]; if(fp<0) return true;
    return see(b,to,from,fp)>=threshold;
}
void genCaptures(const Board& b, MoveList& ml){
    ml.n=0; Color us=b.turn; U64 their=b.occ[1-us],all=b.occ[2];
    U64 pawns=b.pieces[us][PAWN];
    if(us==WHITE){
        U64 cl=(pawns<<7)&their&~fileMask[7],cr=(pawns<<9)&their&~fileMask[0];
        U64 pp=(pawns<<8)&~all&0xFF00000000000000ULL;
        while(pp){Square t=popLSB(pp);ml.add(MK_MOVE(t-8,t,PROMO_Q));ml.add(MK_MOVE(t-8,t,PROMO_N));}
        while(cl){Square t=popLSB(cl);int f=RANK_OF(t)==7?PROMO_CQ:CAPTURE;ml.add(MK_MOVE(t-7,t,f));if(f==PROMO_CQ)ml.add(MK_MOVE(t-7,t,PROMO_CN));}
        while(cr){Square t=popLSB(cr);int f=RANK_OF(t)==7?PROMO_CQ:CAPTURE;ml.add(MK_MOVE(t-9,t,f));if(f==PROMO_CQ)ml.add(MK_MOVE(t-9,t,PROMO_CN));}
        if(b.ep!=NO_SQ){U64 ep=pawnAtt[BLACK][b.ep]&pawns;while(ep){Square fr=popLSB(ep);ml.add(MK_MOVE(fr,b.ep,EP_CAP));}}
    } else {
        U64 cl=(pawns>>9)&their&~fileMask[7],cr=(pawns>>7)&their&~fileMask[0];
        U64 pp=(pawns>>8)&~all&0xFFULL;
        while(pp){Square t=popLSB(pp);ml.add(MK_MOVE(t+8,t,PROMO_Q));ml.add(MK_MOVE(t+8,t,PROMO_N));}
        while(cl){Square t=popLSB(cl);int f=RANK_OF(t)==0?PROMO_CQ:CAPTURE;ml.add(MK_MOVE(t+9,t,f));if(f==PROMO_CQ)ml.add(MK_MOVE(t+9,t,PROMO_CN));}
        while(cr){Square t=popLSB(cr);int f=RANK_OF(t)==0?PROMO_CQ:CAPTURE;ml.add(MK_MOVE(t+7,t,f));if(f==PROMO_CQ)ml.add(MK_MOVE(t+7,t,PROMO_CN));}
        if(b.ep!=NO_SQ){U64 ep=pawnAtt[WHITE][b.ep]&pawns;while(ep){Square fr=popLSB(ep);ml.add(MK_MOVE(fr,b.ep,EP_CAP));}}
    }
    U64 kn=b.pieces[us][KNIGHT]; while(kn){Square f=popLSB(kn);U64 a=knightAtt[f]&their;while(a){Square t=popLSB(a);ml.add(MK_MOVE(f,t,CAPTURE));}}
    U64 bi=b.pieces[us][BISHOP]; while(bi){Square f=popLSB(bi);U64 a=bishAtt(f,all)&their;while(a){Square t=popLSB(a);ml.add(MK_MOVE(f,t,CAPTURE));}}
    U64 ro=b.pieces[us][ROOK];   while(ro){Square f=popLSB(ro);U64 a=rookAtt(f,all)&their;while(a){Square t=popLSB(a);ml.add(MK_MOVE(f,t,CAPTURE));}}
    U64 qu=b.pieces[us][QUEEN];  while(qu){Square f=popLSB(qu);U64 a=queenAtt(f,all)&their;while(a){Square t=popLSB(a);ml.add(MK_MOVE(f,t,CAPTURE));}}
    U64 kg=b.pieces[us][KING];   if(kg){Square f=lsb(kg);U64 a=kingAtt[f]&their;while(a){Square t=popLSB(a);ml.add(MK_MOVE(f,t,CAPTURE));}}
}

int quiesce(Board& b, int alpha, int beta) {
    if(stopNow||timeUp()){stopNow=true;return 0;}
    int stand=evaluatePos(b);
    // Delta pruning — if even capturing the best possible piece can't raise alpha, skip
    const int DELTA=900; // queen value
    if(stand < alpha - DELTA) return alpha;
    if(stand>=beta) return beta;
    if(stand>alpha) alpha=stand;

    MoveList ml; genCaptures(b,ml);
    int qsc[320];
    for(int i=0;i<ml.n;i++){int v=(b.mailbox[MV_TO(ml.m[i])]>=0)?MAT[b.mailbox[MV_TO(ml.m[i])]]:0;int a=(b.mailbox[MV_FROM(ml.m[i])]>=0)?MAT[b.mailbox[MV_FROM(ml.m[i])]]:0;qsc[i]=v*10-a;}
    for(int i=0;i<ml.n;i++){int best=i;for(int j=i+1;j<ml.n;j++)if(qsc[j]>qsc[best])best=j;if(best!=i){swap(ml.m[i],ml.m[best]);swap(qsc[i],qsc[best]);}}
    for(int i=0;i<ml.n;i++){
        Move mv=ml.m[i]; int fl=MV_FLAGS(mv);
        if(isCapture(fl)&&!seeGE(b,mv,-50)) continue;
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
    if (isCapture(fl)) {
        int attacker=b.mailbox[MV_FROM(m)],victim=b.mailbox[MV_TO(m)];
        if(fl==EP_CAP) victim=PAWN;
        int vv=(victim>=0)?MAT[victim]:0, av=(attacker>=0)?MAT[attacker]:0;
        return seeGE(b,m,0) ? 1500000+vv*10-av : 100000+vv*10-av;
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
    if(ply>0 && ply-1<MAX_PLY){Move prev=killers[0][ply-1];if(prev&&counterMove[MV_FROM(prev)][MV_TO(prev)]==m)return 750000;}
    int from=MV_FROM(m),to=MV_TO(m);
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
    // killers[2][MAX_PLY] is sized for the OUTER iterative-deepening depth
    // limit (64), but ply itself is NOT bounded by that — check and
    // singular extensions add +1 to the effective remaining depth on every
    // recursive call, so a long enough forcing sequence (many consecutive
    // checks) pushes ply past MAX_PLY-1 with no relation to how deep the
    // current iterative-deepening iteration nominally is. Every killers[]
    // access below used the raw ply directly, with no bounds check —
    // genuine undefined behavior once ply reached 64+. Confirmed live: a
    // MultiPV secondary line (searching a bad move with the real best line
    // excluded, in a position with overwhelming material offering many
    // spite-checks) reached ply 60+ and produced a physically-impossible
    // "mate in 1" for a move that doesn't even give check — memory
    // corruption near the scoring/TT logic producing a value that
    // happened to read as a mate score. Single-PV/bot-play never hits
    // this: the real best line resolves quickly and never approaches
    // anywhere near this depth. killerPly is used ONLY for the killers[]
    // indexing below; every other use of ply (mate-distance math,
    // repetition-contempt scaling, the ply+1 passed to recursive calls)
    // keeps the true, unclamped value — clamping those would silently cap
    // reported mate distances and break repetition scoring.
    int killerPly = min(ply, MAX_PLY-1);
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
            int contempt = (ply==1) ? -80 : (ply==2) ? -60 : (ply==3) ? -45 : (ply==4) ? -30 : -20;
            // Scale with static eval: penalise draw much harder when winning.
            int se = evaluate(b);
            if (se > 100)  contempt -= (se - 100) / 3;   // much stronger penalty when winning
            if (se < -150) contempt += (-se - 150) / 16;  // slight relief when losing
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
            ttStore(b.hash, sc, depth, NULL_MOVE, flag, ply);
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
        TTResult tte; bool tteHit = ttProbe(b.hash, tte, ply);
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
        TTResult iidTTE; if (ttProbe(b.hash, iidTTE, ply)) ttMove = iidTTE.bestMove;
    }

    // Singular extension — if TT move is much better than all alternatives,
    // extend its search by 1 ply
    bool singularExtension = false;
    if (!pvNode && ply > 0 && depth >= 6 && ttMove != NULL_MOVE) {
        TTResult tte2; bool tte2Hit = ttProbe(b.hash, tte2, ply);
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

            int R = (depth >= 7) ? 3 : 2;
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

        if (!inCheck && depth <= 2 && !capture && legalMoves > 0) {
            int se = evaluatePos(b);
            if (se + 150*depth <= alpha) continue;
        }
        if (!inCheck && depth <= 4 && capture && legalMoves > 0) {
            if (!seeGE(b, ml.m[i], -depth*80)) continue;
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
        ttStore(b.hash, bestScore, depth, bestMove, flag, ply);
    }
    return bestScore;
}

// Walks the transposition table forward from a position, following each
// position's stored best move, to reconstruct a real multi-move principal
// variation. Without this, the engine's info output only ever reported the
// single root move — even for a forced mate, where the actual winning line
// is several moves deep and was always sitting right there in the TT, just
// never collected into a sequence. Capped at a modest depth and defended
// against TT collisions/cycles: stops cleanly if a position repeats (a
// corrupted or shallow TT entry pointing back at an earlier position) or if
// a stored move turns out illegal in the position it's being applied to (a
// hash collision, rare but possible with a limited-size table and no
// verification beyond the hash match itself).
//
// IMPORTANT: this always reads through whatever activeTT currently points
// to. Every caller of this function is responsible for explicitly setting
// activeTT = tt (the main table) immediately before calling it — this
// function does NOT do that itself, because it has no way to know whether
// it's being asked about the main line or an excluded-move scratch line.
// Getting this wrong would silently read from the wrong table.
string extractPV(Board startBoard, Move firstMove, int maxLen = 16) {
    string result = moveStr(firstMove);
    Board cur = startBoard;
    UndoInfo u0;
    if (!makeMove(cur, firstMove, u0)) return result;  // shouldn't happen — firstMove was already validated by the caller

    vector<U64> seenHashes;
    seenHashes.push_back(cur.hash);

    for (int i = 1; i < maxLen; i++) {
        // Stop if the current side to move has no legal moves — the line
        // has reached checkmate or stalemate, which is the true end of the
        // PV. Without this, a completed mating line could keep appending
        // whatever the TT happened to hold for the mated position,
        // producing extra "moves" after a mate that has already ended the
        // game.
        {
            MoveList term; genMoves(cur, term);
            bool anyLegal = false;
            for (int mi = 0; mi < term.n; mi++) {
                UndoInfo tu;
                if (makeMove(cur, term.m[mi], tu)) { unmakeMove(cur, term.m[mi], tu); anyLegal = true; break; }
            }
            if (!anyLegal) break;  // checkmate or stalemate — PV ends here
        }

        TTResult tte;
        if (!ttProbe(cur.hash, tte, 0)) break;  // ply irrelevant here — only tte.bestMove is used, not the score
        if (tte.bestMove == NULL_MOVE) break;

        // Verify the stored move is actually legal here before trusting it —
        // a hash collision could otherwise produce a bogus or illegal move
        // in the printed PV. Cheap insurance: confirm it appears in the
        // genuinely-generated legal move list for this exact position.
        MoveList ml; genMoves(cur, ml);
        bool isLegal = false;
        for (int mi = 0; mi < ml.n; mi++) {
            if (ml.m[mi] == tte.bestMove) { isLegal = true; break; }
        }
        if (!isLegal) break;

        UndoInfo u;
        Board next = cur;
        if (!makeMove(next, tte.bestMove, u)) break;  // legality check above should prevent this, but stay defensive

        // Guard against the TT pointing back into a position already in
        // this exact line (a cycle) — would otherwise repeat forever up to
        // maxLen with no real new information.
        bool seenBefore = false;
        for (U64 h : seenHashes) { if (h == next.hash) { seenBefore = true; break; } }
        if (seenBefore) break;

        result += " " + moveStr(tte.bestMove);
        cur = next;
        seenHashes.push_back(cur.hash);
    }
    return result;
}

// Root search — bestMove ALWAYS tracked here directly, never extracted from TT
Move search(Board& b, int wtime, int btime, int movestogo, int winc, int binc) {
    searchStart=chrono::steady_clock::now();
    stopNow=false;

    int totalTime = (b.turn==WHITE) ? wtime : btime;
    int timeMs  = max(0, totalTime - 200);  // subtract 200ms safety buffer
    int inc     = (b.turn==WHITE) ? winc  : binc;
    int baseTime;
    bool isBullet;

    // TWO independent bypasses of the clock-based formula below, for two
    // different calling contexts:
    //   - nativeMovetimeRequest: a native UCI "go movetime N" — see the
    //     detailed comment where this flag is declared.
    //   - wasmTimerPreset (Emscripten builds only): engine_best_move /
    //     engine_analyse call in from JS with an exact millisecond budget,
    //     same idea, different caller.
    // Neither exists in the other build target, so without both, a native
    // build compiled from what's otherwise the WASM source would silently
    // reintroduce the exact movetime-collapse bug this fixes — confirmed
    // while testing this exact scenario.
    bool bypassClockFormula = nativeMovetimeRequest;
#ifdef __EMSCRIPTEN__
    bypassClockFormula = bypassClockFormula || wasmTimerPreset;
#endif

    if (bypassClockFormula) {
        searchTimeMs = max(10, totalTime - 200);
        baseTime     = searchTimeMs;
        isBullet     = (totalTime <= 60000);
    } else {
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
        isBullet = (totalTime <= 60000);
        bool isBlitz  = (totalTime > 60000 && totalTime <= 300000);
        bool isRapid  = (totalTime > 300000 && totalTime <= 600000);
        int safetyMs  = isBullet ? 400 : 200;
        timeMs        = max(0, totalTime - safetyMs);
        int defaultMoves = isBullet ? 50 : 40;
        int moves = (movestogo > 0) ? movestogo : defaultMoves;
        searchTimeMs = max(10, timeMs / moves + (inc * 4) / 5);
        int pctCap;
        if      (isBullet) pctCap = 50;
        else if (isBlitz)  pctCap = 25;
        else if (isRapid)  pctCap = 33;
        else               pctCap = 40;
        searchTimeMs = min(searchTimeMs, timeMs / pctCap);
        searchTimeMs = max(searchTimeMs, 10);

        // Advanced passed pawn: one-shot +15% extension, only for our pawns
        // on rank 6+ (truly about to promote). Does not compound. Only
        // meaningful with a real clock — skipped when bypassed above.
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
        baseTime = searchTimeMs;  // save for instability extension
    }


    // Reset per-search state
    memset(killers,     0, sizeof(killers));
    memset(counterMove, 0, sizeof(counterMove));
    for(int c=0;c<2;c++) for(int f=0;f<64;f++) for(int t=0;t<64;t++) history[c][f][t]/=4;
    // Guarantee a legal move exists BEFORE searching, so no interrupt path
    // can ever return NULL_MOVE (0000). Fixes illegal-move losses in
    // positions with very few legal moves.
    Move bestMove=NULL_MOVE;
    int  bestScore=0;
    {
        MoveList ml; genMoves(b,ml);
        for(int i=0;i<ml.n;i++){
            UndoInfo u; if(makeMove(b,ml.m[i],u)){ unmakeMove(b,ml.m[i],u); bestMove=ml.m[i]; break; }
        }
    }

    // ----------------------------------------------------------------
    // ROOT MOVE LIST — persists across depths, carries per-move scores
    // ----------------------------------------------------------------
    struct RootMove { Move m; int score; };
    vector<RootMove> rootMoves;
    {
        MoveList ml; genMoves(b, ml);
        sortMoves(b, ml, NULL_MOVE, 0);
        for(int i = 0; i < ml.n; i++){
            UndoInfo u;
            if(makeMove(b, ml.m[i], u)){ unmakeMove(b, ml.m[i], u); rootMoves.push_back({ml.m[i], 0}); }
        }
    }

    for(int depth=1; depth<=64; depth++){

        // Per-line results for THIS depth, captured in multipv order (line 0
        // = best, line 1 = 2nd best, ...) exactly as the pvIdx loop finds
        // them. These are the source of truth for what gets exposed to the
        // analysis path — NOT rootMoves, whose order is re-sorted at the top
        // of each depth and so does not reliably match the multipv ranking
        // the search actually reported. Only overwritten by a FULLY
        // completed depth, so an interrupted final depth can't corrupt a
        // good previous one.
        Move thisDepthMoves[MAX_EXPOSED_ROOT_MOVES];
        int  thisDepthScores[MAX_EXPOSED_ROOT_MOVES];
        int  thisDepthCount = 0;

        // Sort root moves by score from previous depth (best first)
        if(depth > 1)
            stable_sort(rootMoves.begin(), rootMoves.end(),
                [](const RootMove& a, const RootMove& b){ return a.score > b.score; });

        // MultiPV loop — each iteration finds the next-best root move
        vector<Move> excluded;
        int pvLimit = min(multiPVCount, (int)rootMoves.size());

        for(int pvIdx = 0; pvIdx < pvLimit; pvIdx++){

            // Main table for the true best line and all single-PV (bot play)
            // search — completely unaffected by anything below. Scratch
            // table, freshly cleared, for every excluded-move line — see the
            // detailed comment on activeTT above for why this isolation is
            // necessary. Clearing tt2 here (not just once) matters: it's
            // cheap (256K entries vs the main table's 8M) and prevents one
            // secondary line's exploration from contaminating the NEXT
            // secondary line the same way the main table was contaminating
            // all of them before this fix.
            if(pvIdx == 0 || multiPVCount == 1){
                activeTT     = tt;
                activeTTMask = TT_SIZE - 1;
            } else {
                memset(tt2, 0, TT_SIZE_MPV*sizeof(TTEntry));
                activeTT     = tt2;
                activeTTMask = TT_SIZE_MPV - 1;
            }

            int prevScore = rootMoves[pvIdx].score;
            int alpha = -INF, beta = INF;
            // Single-PV (game play): tight two-sided aspiration window.
            // MultiPV (analysis): a one-sided SEEDED window — a lower
            // anchor (prevScore - margin) so the PVS scouts below can still
            // cut off worse moves cheaply, but beta stays at INF so a line
            // is never clamped from above and always gets its true score.
            // If the whole line turns out to be below the anchor, the
            // fail-low re-search below widens the anchor and re-runs.
            int mpvAnchor = -INF;
            if(depth >= 4){
                if(multiPVCount == 1){
                    alpha = max(-INF, prevScore - 50);
                    beta  = min( INF, prevScore + 50);
                } else if(prevScore > -MATE_THRESHOLD && prevScore < MATE_THRESHOLD){
                    // Only seed a finite anchor for normal scores — a mate
                    // score last depth is tens of thousands of cp away and
                    // any finite anchor would just fail and force a re-search.
                    alpha     = max(-INF, prevScore - 300);
                    mpvAnchor = alpha;
                }
            }

            Move depthBest      = NULL_MOVE;
            int  depthBestScore = -INF;
            bool research        = false;
            int  researchAttempts = 0;

            do {
                research       = false;
                depthBest      = NULL_MOVE;
                depthBestScore = -INF;

                for(auto& rm : rootMoves){
                    bool skip = false;
                    for(Move ex : excluded) if(rm.m == ex){ skip=true; break; }
                    if(skip) continue;

                    UndoInfo u;
                    if(!makeMove(b, rm.m, u)) continue;

                    int sc;
                    if(repetitionCount(b) >= 2){
                        sc = -200 + depth;
                        unmakeMove(b, rm.m, u);
                        if(sc > depthBestScore){ depthBestScore = sc; depthBest = rm.m; }
                        continue;
                    }
                    if(depthBest == NULL_MOVE)
                        sc = -negamax(b, depth-1, -beta, -alpha, 1);
                    else {
                        sc = -negamax(b, depth-1, -alpha-1, -alpha, 1);
                        if(sc > alpha && sc < beta && !stopNow)
                            sc = -negamax(b, depth-1, -beta, -alpha, 1);
                    }
                    unmakeMove(b, rm.m, u);
                    if(stopNow) goto done;
                    if(sc > depthBestScore){ depthBestScore = sc; depthBest = rm.m; }
                    if(sc > alpha) alpha = sc;
                    if(alpha >= beta) break;
                }

                if(depth >= 4 && multiPVCount == 1){
                    if(depthBestScore <= alpha - 50){
                        researchAttempts++;
                        if(researchAttempts >= 4){
                            // Repeated narrow re-search attempts all failed —
                            // the true score is far outside the aspiration
                            // window (typically a forced mate found partway
                            // through this depth). Fall back to a fully open
                            // window so the real score/move are found directly.
                            alpha = -INF;
                        } else {
                            alpha = max(-INF, alpha-100);
                        }
                        research = true;
                    }
                    else if(depthBestScore >= beta + 50){
                        researchAttempts++;
                        if(researchAttempts >= 4){
                            beta = INF;
                        } else {
                            beta = min(INF, beta+100);
                        }
                        research = true;
                    }
                }
                else if(multiPVCount > 1 && mpvAnchor > -INF){
                    // MultiPV fail-low: the whole line came back at or below
                    // the seeded lower anchor, so the anchor was too high and
                    // every score is a clamped bound. Drop it fully open and
                    // re-run so the true ranking/score are found. (beta was
                    // INF throughout, so there's never a fail-high to handle.)
                    if(depthBestScore <= mpvAnchor){
                        alpha     = -INF;
                        mpvAnchor = -INF;   // don't retrigger
                        research  = true;
                    }
                }
            } while(research && !stopNow);

            done:
            if(!stopNow && depthBest != NULL_MOVE){
                for(auto& rm : rootMoves) if(rm.m == depthBest){ rm.score = depthBestScore; break; }
                excluded.push_back(depthBest);

                // Record this line's result in multipv order (pvIdx 0,1,2...)
                if(pvIdx < MAX_EXPOSED_ROOT_MOVES){
                    thisDepthMoves[pvIdx]  = depthBest;
                    thisDepthScores[pvIdx] = depthBestScore;
                    if(pvIdx + 1 > thisDepthCount) thisDepthCount = pvIdx + 1;
                }

                if(pvIdx == 0){
#ifdef __EMSCRIPTEN__
                    if (wasmTimerPreset) {
                        searchTimeMs = baseTime;  // no instability extension in WASM
                    } else {
#endif
                    // Instability extension: if score drops significantly from
                    // previous depth, spend more time searching, capped tightly
                    // relative to base time only.
                    if(depth > 4 && bestMove != NULL_MOVE) {
                        int scoreDrop = bestScore - depthBestScore;
                        int extCap1 = isBullet ? baseTime + 1500 : baseTime + 5000;
                        int extCap2 = isBullet ? baseTime + 800  : baseTime + 3000;
                        if(scoreDrop > 30)       searchTimeMs = min(baseTime*3, extCap1);
                        else if(scoreDrop > 15)  searchTimeMs = min(baseTime*2, extCap2);
                        else                     searchTimeMs = baseTime;
                    }
#ifdef __EMSCRIPTEN__
                    }
#endif
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
                }

                int elapsed=(int)chrono::duration_cast<chrono::milliseconds>(
                    chrono::steady_clock::now()-searchStart).count();
                cerr<<"info depth "<<depth<<" multipv "<<(pvIdx+1);
                if (isMateScore(depthBestScore)) {
                    cerr << " score mate " << mateMovesFromScore(depthBestScore);
                } else {
                    cerr << " score cp " << depthBestScore;
                }
                cerr << " time "<<elapsed<<" pv "<<moveStr(depthBest)<<"\n";
            }
        } // end pvIdx loop

        // Commit this depth's per-line results to the exposed arrays, but
        // ONLY if the depth completed all its lines without being cut off by
        // time — a fully completed later depth supersedes an earlier one.
        if(!stopNow && thisDepthCount > 0){
            exposedRootMoveCount = min(thisDepthCount, MAX_EXPOSED_ROOT_MOVES);
            for(int i = 0; i < exposedRootMoveCount; i++){
                exposedRootMoves[i]  = thisDepthMoves[i];
                exposedRootScores[i] = thisDepthScores[i];
            }
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


// ======================= WAVE 1+2 TEXEL TUNER (compile with /DTUNE) ==========
#ifdef TUNE
#include <cstdlib>

static double sigmoidT(double x, double K){ return 1.0/(1.0+pow(10.0,-K*x/400.0)); }

struct TuneSample { Board b; double result; double scp; double target; };
static std::vector<TuneSample> gData;
static std::vector<int*>       gParams;

static double whiteEval(Board& b){ int e=evaluate(b); return (b.turn==WHITE)?(double)e:(double)(-e); }

static void buildParams(){
    gParams.clear();
    for(int p=0;p<5;p++) gParams.push_back(&MAT_MG[p]);
    for(int p=0;p<5;p++) gParams.push_back(&MAT_EG[p]);
    for(int p=0;p<6;p++) for(int sq=0;sq<64;sq++){
        if(p==PAWN && (sq<8||sq>=56)) continue;
        gParams.push_back(&PST_MG[p][sq]);
    }
    for(int p=0;p<6;p++) for(int sq=0;sq<64;sq++){
        if(p==PAWN && (sq<8||sq>=56)) continue;
        gParams.push_back(&PST_EG[p][sq]);
    }
    int* w2[] = {
        &MOB_N_MG,&MOB_N_EG,&MOB_B_MG,&MOB_B_EG,&MOB_R_MG,&MOB_R_EG,&MOB_Q_MG,&MOB_Q_EG,
        &BISHOP_PAIR_MG,&BISHOP_PAIR_EG,&ROOK_OPEN_MG,&ROOK_OPEN_EG,
        &ROOK_SEMI_MG,&ROOK_SEMI_EG,&DOUBLED_MG,&DOUBLED_EG,
        &ISOLATED_MG,&ISOLATED_EG,&TEMPO
    };
    for(int* p : w2) gParams.push_back(p);
}

static void loadData(const std::string& path){
    std::ifstream f(path);
    if(!f){ std::cerr<<"cannot open "<<path<<"\n"; exit(1); }
    std::string line; std::getline(f,line);
    int kept=0,skipped=0;
    while(std::getline(f,line)){
        if(line.empty()) continue;
        size_t c1=line.find(',');
        if(c1==std::string::npos) continue;
        std::string fen=line.substr(0,c1), rest=line.substr(c1+1);
        size_t a=rest.find(','), b2=rest.find(',',a+1);
        if(a==std::string::npos||b2==std::string::npos) continue;
        std::string whitecp=rest.substr(a+1,b2-(a+1)), result=rest.substr(b2+1);
        if(result.empty()){ skipped++; continue; }
        double scp=atof(whitecp.c_str());
        if(fabs(scp)>=3000.0){ skipped++; continue; }
        TuneSample ts; ts.b=parseFEN(fen); ts.result=atof(result.c_str()); ts.scp=scp; ts.target=0.0;
        gData.push_back(ts); kept++;
    }
    std::cerr<<"loaded "<<kept<<" positions ("<<skipped<<" skipped)\n";
}

static double solveK(){
    double best=1.0,bestE=1e18;
    for(double K=0.20;K<=2.0;K+=0.02){
        double sum=0; for(auto&s:gData){ double e=sigmoidT(whiteEval(s.b),K)-s.result; sum+=e*e; }
        double err=sum/gData.size(); if(err<bestE){bestE=err;best=K;}
    }
    return best;
}

static void fillTargets(double K,double blend){
    for(auto&s:gData) s.target=(1.0-blend)*s.result+blend*sigmoidT(s.scp,K);
}
static double avgError(double K){
    double sum=0; for(auto&s:gData){ double e=sigmoidT(whiteEval(s.b),K)-s.target; sum+=e*e; }
    return sum/gData.size();
}

static void dumpWeights(const std::string& path){
    std::ofstream o(path);
    auto arr=[&](const char* nm,int* a,int n){
        o<<"WCONST int "<<nm<<"["<<n<<"]= {";
        for(int i=0;i<n;i++){ o<<a[i]; if(i<n-1)o<<", "; } o<<"};\n";
    };
    arr("MAT_MG",MAT_MG,6); arr("MAT_EG",MAT_EG,6);
    auto pst=[&](const char* nm,int a[][64]){
        o<<"WCONST int "<<nm<<"[6][64] = {\n";
        for(int p=0;p<6;p++){ o<<"{"; for(int sq=0;sq<64;sq++){o<<a[p][sq]; if(sq<63)o<<",";} o<<"}"; if(p<5)o<<","; o<<"\n"; }
        o<<"};\n";
    };
    pst("PST_MG",PST_MG); pst("PST_EG",PST_EG);
    o<<"// --- Wave 2 scalars ---\n";
    o<<"WCONST int MOB_N_MG = "<<MOB_N_MG<<",  MOB_N_EG = "<<MOB_N_EG<<";\n";
    o<<"WCONST int MOB_B_MG = "<<MOB_B_MG<<",  MOB_B_EG = "<<MOB_B_EG<<";\n";
    o<<"WCONST int MOB_R_MG = "<<MOB_R_MG<<",  MOB_R_EG = "<<MOB_R_EG<<";\n";
    o<<"WCONST int MOB_Q_MG = "<<MOB_Q_MG<<",  MOB_Q_EG = "<<MOB_Q_EG<<";\n";
    o<<"WCONST int BISHOP_PAIR_MG = "<<BISHOP_PAIR_MG<<", BISHOP_PAIR_EG = "<<BISHOP_PAIR_EG<<";\n";
    o<<"WCONST int ROOK_OPEN_MG = "<<ROOK_OPEN_MG<<", ROOK_OPEN_EG = "<<ROOK_OPEN_EG<<";\n";
    o<<"WCONST int ROOK_SEMI_MG = "<<ROOK_SEMI_MG<<", ROOK_SEMI_EG = "<<ROOK_SEMI_EG<<";\n";
    o<<"WCONST int DOUBLED_MG  = "<<DOUBLED_MG<<", DOUBLED_EG  = "<<DOUBLED_EG<<";\n";
    o<<"WCONST int ISOLATED_MG = "<<ISOLATED_MG<<", ISOLATED_EG = "<<ISOLATED_EG<<";\n";
    o<<"WCONST int TEMPO = "<<TEMPO<<";\n";
}

static void runTune(const std::string& csv,double blend){
    loadData(csv);
    if(gData.empty()){ std::cerr<<"no data\n"; return; }
    buildParams();
    double K=solveK(); fillTargets(K,blend);
    std::cerr<<"K="<<K<<"  params="<<gParams.size()<<"  blend="<<blend<<"\n";
    double best=avgError(K);
    std::cerr<<"start error="<<best<<"\n";
    bool improved=true; int pass=0;
    while(improved){
        improved=false; pass++;
        for(size_t i=0;i<gParams.size();i++){
            int* w=gParams[i]; int orig=*w;
            *w=orig+1; double e=avgError(K);
            if(e<best){ best=e; improved=true; }
            else { *w=orig-1; e=avgError(K); if(e<best){ best=e; improved=true; } else *w=orig; }
        }
        std::cerr<<"pass "<<pass<<"  error="<<best<<"\n";
        dumpWeights("tuned_weights.txt");
    }
    std::cerr<<"done after "<<pass<<" passes. tuned weights in tuned_weights.txt\n";
}
#endif
// ===================== END TUNER BLOCK ======================================

// Output-layer safety: never emit 0000 or an illegal move. If the search
// result is null/illegal, substitute the first legal move. Structurally
// guarantees a legal bestmove on every code path.
Move ensureLegal(Board& b, Move m){
    MoveList ml; genMoves(b,ml);
    for(int i=0;i<ml.n;i++) if(ml.m[i]==m){
        UndoInfo u; if(makeMove(b,ml.m[i],u)){ unmakeMove(b,ml.m[i],u); return m; }
    }
    for(int i=0;i<ml.n;i++){
        UndoInfo u; if(makeMove(b,ml.m[i],u)){ unmakeMove(b,ml.m[i],u); return ml.m[i]; }
    }
    return m; // no legal moves (mate/stalemate): caller shouldn't be moving
}

int main(int argc, char** argv){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    tt  = new TTEntry[TT_SIZE]();
    tt2 = new TTEntry[TT_SIZE_MPV]();
    activeTT     = tt;          // default: main table (single-PV / pvIdx==0)
    activeTTMask = TT_SIZE - 1;

#ifdef USE_NNUE
    // Auto-load nn.nnue if present in working directory
    nnueEnabled = nnueLoad("nn.nnue");
    if (!nnueEnabled) cerr<<"NNUE: nn.nnue not found, using classical eval\n";
#endif
    initAttacks();
    initSliders();
    initMasks();
    initZobrist();
    initLMR();
    ttClear();
#ifdef VERIFY_SLIDERS
    { bool ok=verifySliders(); std::cerr<<"slider verify: "<<(ok?"PASS":"FAIL")<<"\n"; return ok?0:1; }
#endif

#ifdef TUNE
    if (argc >= 3 && std::string(argv[1]) == "tune") {
        double blend = (argc >= 4) ? atof(argv[3]) : 0.5;
        runTune(argv[2], blend);
        return 0;
    }
#endif

    int lastWt=60000, lastBt=60000, lastMtg=0, lastWi=0, lastBi=0; // saved from last go
    bool lastNativeMovetimeRequest = false;
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
                cout << "bestmove " << moveStr(ensureLegal(board, searchResult)) << " ponder " << moveStr(ponderMove) << "\n";
            else
                cout << "bestmove " << moveStr(ensureLegal(board, searchResult)) << "\n";
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
                <<"option name MultiPV type spin default 1 min 1 max 5\n"
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
            else if(name=="MultiPV" && !value.empty()){
                int mpv = atoi(value.c_str());
                multiPVCount = max(1, min(5, mpv));
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
            nativeMovetimeRequest = false;  // reset every go — must not leak from a previous request
            string tok;
            while(ss>>tok){
                if(tok=="wtime")      ss>>wt;
                else if(tok=="btime") ss>>bt;
                else if(tok=="movestogo") ss>>mtg;
                else if(tok=="winc")  ss>>wi;
                else if(tok=="binc")  ss>>bi;
                else if(tok=="ponder")   doPonder=true;
                else if(tok=="infinite") doPonder=true;
                else if(tok=="movetime"){ int mt; ss>>mt; wt=bt=mt+200; mtg=1; nativeMovetimeRequest=true; }
            }
            // Save time params (and whether this was a movetime request) so
            // ponderhit can re-launch with the same budget and behaviour.
            lastWt=wt; lastBt=bt; lastMtg=mtg; lastWi=wi; lastBi=bi;
            lastNativeMovetimeRequest = nativeMovetimeRequest;

            pondering = doPonder;
            searchResult = NULL_MOVE;
            searchDone = false;

            // Capture go parameters and board by value for the thread
            Board boardCopy = board;
            int wt_=wt, bt_=bt, mtg_=mtg, wi_=wi, bi_=bi;

            if (searchThread.joinable()) { stopNow=true; searchThread.join(); stopNow=false; }
            searchThread = thread([boardCopy,wt_,bt_,mtg_,wi_,bi_]() mutable {
                // try/catch here is a native-only safety net (a real
                // std::thread swallowing an exception rather than crashing
                // the whole process) — incompatible with -fno-exceptions,
                // and moot for WASM anyway: this whole go/ponderhit code
                // path lives in main(), which a WASM/MODULARIZE build never
                // actually calls (JS drives engine_best_move/engine_analyse
                // directly), so the code just needs to compile, never run.
#ifndef __EMSCRIPTEN__
                try { searchResult = search(boardCopy, wt_, bt_, mtg_, wi_, bi_); }
                catch (...) {}
#else
                searchResult = search(boardCopy, wt_, bt_, mtg_, wi_, bi_);
#endif
                searchDone = true;
            });

            if (!doPonder) {
                // Normal search — block until done, then output bestmove
                searchThread.join();
                pondering = false;
                if(ponderMove!=NULL_MOVE)
                    cout<<"bestmove "<<moveStr(ensureLegal(board, searchResult))<<" ponder "<<moveStr(ponderMove)<<"\n";
                else
                    cout<<"bestmove "<<moveStr(ensureLegal(board, searchResult))<<"\n";
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
            nativeMovetimeRequest = lastNativeMovetimeRequest;  // match the original go's behaviour

            searchThread = thread([boardCopy,wt_,bt_,mtg_,wi_,bi_]() mutable {
#ifndef __EMSCRIPTEN__
                try { searchResult = search(boardCopy, wt_, bt_, mtg_, wi_, bi_); }
                catch (...) {}
#else
                searchResult = search(boardCopy, wt_, bt_, mtg_, wi_, bi_);
#endif
                searchDone = true;
            });
            // Block until done, then output bestmove (same as normal go)
            searchThread.join();
            if(ponderMove!=NULL_MOVE)
                cout<<"bestmove "<<moveStr(ensureLegal(board, searchResult))<<" ponder "<<moveStr(ponderMove)<<"\n";
            else
                cout<<"bestmove "<<moveStr(ensureLegal(board, searchResult))<<"\n";
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


// ============================================================
// WASM DEPLOYMENT LAYER
// ============================================================
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>

static bool wasmInitDone = false;

extern "C" {

EMSCRIPTEN_KEEPALIVE
void engine_init() {
    if (wasmInitDone) return;
    cerr << "[engine] build " << ENGINE_BUILD_ID << "\n";  // confirms which binary is actually running
    tt  = new TTEntry[TT_SIZE]();
    tt2 = new TTEntry[TT_SIZE_MPV]();
    activeTT     = tt;          // default: main table
    activeTTMask = TT_SIZE - 1;
    initAttacks();
    initMasks();
    initSliders();   // builds the PEXT rook/bishop attack tables
    initZobrist();
    initLMR();
    ttClear();
    wasmInitDone = true;
}

EMSCRIPTEN_KEEPALIVE
const char* engine_best_move(const char* fen_str,
                              const char* moves_str,
                              int movetime_ms) {
    if (!wasmInitDone) engine_init();

    static char result[8];
    result[0] = '\0';

    // Reset per-game state — prevents ghost repetitions and biased history
    // from leaking across separate games or calls
    memset(posHistory, 0, sizeof(posHistory));
    memset(history,    0, sizeof(history));
    activeTT     = tt;          // always the main table for bot play — never tt2
    activeTTMask = TT_SIZE - 1;

    Board board = parseFEN(std::string(fen_str));
    if (moves_str && moves_str[0] != '\0') {
        std::istringstream ss(moves_str);
        std::string tok;
        while (ss >> tok) {
            Move m = parseMove(board, tok);
            if (m == NULL_MOVE) break;
            UndoInfo u;
            if (!makeMove(board, m, u)) break;
        }
    }

    stopNow         = false;
    nativeMovetimeRequest = false;  // this is the WASM path — wasmTimerPreset is the equivalent flag
    ponderMove      = NULL_MOVE;
    searchTimeMs    = std::max(10, movetime_ms);
    wasmTimerPreset = true;
    Move best = search(board, 9999999, 9999999, 1, 0, 0);
    wasmTimerPreset = false;

    // ensureLegal: a final safety net confirming the returned move is
    // actually legal in the real position before it's ever sent back to
    // the site — independent of (and in addition to) everything else in
    // this file, including the native UCI path's own use of it.
    best = ensureLegal(board, best);

    if (best != NULL_MOVE) {
        std::string ms = moveStr(best);
        strncpy(result, ms.c_str(), 7);
        result[7] = '\0';
    }
    return result;
}

// ── Analysis export ── MultiPV search for the analysis tab ──────────────────
// Uses engine_analyse() so game moves (engine_best_move) are never slowed down.
// multipv: 1-5 lines to search simultaneously.
EMSCRIPTEN_KEEPALIVE
const char* engine_analyse(const char* fen_str,
                            const char* moves_str,
                            int movetime_ms,
                            int multipv) {
    if (!wasmInitDone) engine_init();

    static char result[8];
    result[0] = '\0';

    memset(posHistory, 0, sizeof(posHistory));
    memset(history,    0, sizeof(history));
    // Clear the transposition table too — it is never cleared anywhere else
    // in this function, and persists across every call for the life of the
    // worker. Without this, a second Load press on the EXACT SAME position
    // would silently benefit from cached entries the first press never had,
    // making the result non-deterministic for identical input.
    ttClear();
    activeTT     = tt;          // start every analysis call on the main table
    activeTTMask = TT_SIZE - 1;

    Board board = parseFEN(std::string(fen_str));
    if (moves_str && moves_str[0] != '\0') {
        std::istringstream ss(moves_str);
        std::string tok;
        while (ss >> tok) {
            Move m = parseMove(board, tok);
            if (m == NULL_MOVE) break;
            UndoInfo u;
            if (!makeMove(board, m, u)) break;
        }
    }

    // Keep a genuinely untouched copy of the position BEFORE search() runs.
    Board cleanBoardForPV = board;

    // Set MultiPV for this search only — restore after so game searches are unaffected
    int savedMultiPV = multiPVCount;
    multiPVCount     = std::max(1, std::min(multipv, 5));

    stopNow         = false;
    nativeMovetimeRequest = false;
    ponderMove      = NULL_MOVE;
    searchTimeMs    = std::max(10, movetime_ms);
    wasmTimerPreset = true;
    Move best = search(board, 9999999, 9999999, 1, 0, 0);
    wasmTimerPreset = false;

    best = ensureLegal(board, best);

    multiPVCount = savedMultiPV;  // restore — never affects game searches

    // Extract each line's PV into a string IMMEDIATELY after that line's own
    // search, while the shared global TT still reflects that exact line —
    // then print them all afterwards. Line 1 (i==0) is captured FIRST,
    // before any mini-search runs, because it's the one line whose real,
    // full-depth continuation the main search just left in the TT.
    const int SECONDARY_PV_DEPTH    = 12;
    const int SECONDARY_PV_TIME_MS  = 600;
    bool savedStopNow      = stopNow;
    auto savedSearchStart  = searchStart;
    int  savedSearchTimeMs = searchTimeMs;

    std::string pvStrings[MAX_EXPOSED_ROOT_MOVES];
    for (int i = 0; i < exposedRootMoveCount; i++) {
        if (i >= 1) {
            Board pvScratch = cleanBoardForPV;   // fully separate, disposable
            UndoInfo u;
            if (makeMove(pvScratch, exposedRootMoves[i], u)) {
                stopNow      = false;
                searchStart  = chrono::steady_clock::now();
                searchTimeMs = SECONDARY_PV_TIME_MS;
                // Explicitly use the MAIN table here, not whatever search()'s
                // own multipv loop last left activeTT pointing to (it could
                // be tt2, if the last completed pvIdx before returning was an
                // excluded-move line). This mini-search's whole purpose is to
                // leave a deeper continuation behind for extractPV to walk
                // via the shared main-table context, same as line 1's — using
                // the wrong table here would either pollute the isolation the
                // multipv fix depends on, or silently fail to find anything.
                activeTT     = tt;
                activeTTMask = TT_SIZE - 1;
                negamax(pvScratch, SECONDARY_PV_DEPTH, -INF, INF, 1);  // return value intentionally discarded — only its TT side effect is used
            }
        }
        // Always read extractPV through the main table explicitly — same
        // reasoning as above.
        activeTT     = tt;
        activeTTMask = TT_SIZE - 1;
        pvStrings[i] = extractPV(cleanBoardForPV, exposedRootMoves[i]);
    }

    stopNow      = savedStopNow;
    searchStart  = savedSearchStart;
    searchTimeMs = savedSearchTimeMs;

    // Scores come straight from the main search's already-vetted per-move
    // scores, PVs from the strings captured above.
    for (int i = 0; i < exposedRootMoveCount; i++) {
        int lineScore = exposedRootScores[i];
        cerr << "pvfinal " << (i+1) << " ";
        if (isMateScore(lineScore)) {
            cerr << "mate " << mateMovesFromScore(lineScore);
        } else {
            cerr << "cp " << lineScore;
        }
        cerr << " " << pvStrings[i] << "\n";
    }

    if (best != NULL_MOVE) {
        std::string ms = moveStr(best);
        strncpy(result, ms.c_str(), 7);
        result[7] = '\0';
    }
    return result;
}

} // extern "C"
#endif // __EMSCRIPTEN__
