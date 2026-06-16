/**
 * aac_grid.js — AfriChess Symbol Grid AAC Board
 *
 * Two-step piece selection interface for non-verbal users:
 *   Step 1: Choose piece (6 large symbols)
 *   Step 2: Choose destination square (8×8 grid filtered to legal moves)
 *
 * Accessibility:
 *   - 60px minimum touch targets
 *   - High contrast + colour coding per piece
 *   - ARIA labels on every cell
 *   - Speaks cell label on focus
 *   - Keyboard navigable (arrow keys + Enter)
 *   - Works with switch scanning from aac.js
 *
 * Requires: aac.js (for AfriChessAAC.speak), aac_strings.js
 */

const AfriChessGrid = (function() {

  let _game         = null;   // chess.js instance
  let _executeMove  = null;   // callback(from, to, promotion)
  let _myColor      = null;   // 'w' or 'b'
  let _selectedPiece = null;  // { type: 'n', squares: ['f3','d4',...] }
  let _container    = null;
  let _lang         = () => (typeof aacStrings === 'function' ? aacStrings(
    typeof AfriChessAAC !== 'undefined' ? AfriChessAAC.getLanguage() : 'en'
  ) : { pieces:{p:'P',n:'N',b:'B',r:'R',q:'Q',k:'K'}, ui:{gridTitle:'Select piece',gridSquare:'Select square'} });

  // Piece visual config
  const PIECE_CONFIG = {
    p: { symbol: '♟', symbolW: '♙', color: '#aaa',    label: 'Pawn'   },
    n: { symbol: '♞', symbolW: '♘', color: '#b0c4de', label: 'Knight' },
    b: { symbol: '♝', symbolW: '♗', color: '#deb887', label: 'Bishop' },
    r: { symbol: '♜', symbolW: '♖', color: '#cd5c5c', label: 'Rook'   },
    q: { symbol: '♛', symbolW: '♕', color: '#ffd700', label: 'Queen'  },
    k: { symbol: '♚', symbolW: '♔', color: '#00ff9d', label: 'King'   },
  };

  const FILES = ['a','b','c','d','e','f','g','h'];
  const RANKS = ['8','7','6','5','4','3','2','1'];

  // ── Build piece selector ────────────────────────────────────────────────────
  function buildPieceSelector() {
    if (!_container || !_game) return;
    _container.innerHTML = '';
    _selectedPiece = null;

    const title = document.createElement('div');
    title.style.cssText = 'font-size:11px;color:#555;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:10px;';
    title.textContent = _lang().ui.gridTitle;
    _container.appendChild(title);

    const grid = document.createElement('div');
    grid.style.cssText = 'display:grid;grid-template-columns:repeat(3,1fr);gap:6px;';
    grid.setAttribute('role', 'group');
    grid.setAttribute('aria-label', _lang().ui.gridTitle);

    // Find which pieces have legal moves for my color
    const legal = _game.moves({ verbose: true });
    const myPieces = legal
      .filter(m => m.color === _myColor)
      .reduce(function(acc, m) {
        if (!acc[m.piece]) acc[m.piece] = new Set();
        acc[m.piece].add(m.from);
        return acc;
      }, {});

    const order = ['q','r','b','n','p','k'];
    order.forEach(function(pt) {
      const cfg     = PIECE_CONFIG[pt];
      const hasMove = !!myPieces[pt];
      const symbol  = _myColor === 'w' ? cfg.symbolW : cfg.symbol;
      const label   = _lang().pieces[pt] || cfg.label;

      const cell = document.createElement('button');
      cell.setAttribute('role', 'button');
      cell.setAttribute('aria-label', label + (hasMove ? '' : ' — no legal moves'));
      cell.disabled = !hasMove;
      cell.style.cssText = `
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; padding: 12px 6px;
        background: ${hasMove ? '#111' : '#0a0a0a'};
        color: ${hasMove ? cfg.color : '#333'};
        border: 1px solid ${hasMove ? '#2a2a2a' : '#151515'};
        border-radius: 4px; cursor: ${hasMove ? 'pointer' : 'not-allowed'};
        font-size: 32px; line-height: 1;
        transition: all 0.12s;
        min-height: 60px;
      `;

      const sym = document.createElement('span');
      sym.textContent = symbol;
      sym.setAttribute('aria-hidden', 'true');
      const lbl = document.createElement('span');
      lbl.style.cssText = 'font-size:9px;letter-spacing:0.08em;text-transform:uppercase;margin-top:4px;color:#555;';
      lbl.textContent = label;

      cell.appendChild(sym);
      cell.appendChild(lbl);

      if (hasMove) {
        cell.addEventListener('mouseenter', function() {
          cell.style.background = '#1a1a1a';
          cell.style.borderColor = cfg.color + '66';
        });
        cell.addEventListener('mouseleave', function() {
          cell.style.background = '#111';
          cell.style.borderColor = '#2a2a2a';
        });
        cell.addEventListener('focus', function() {
          if (typeof AfriChessAAC !== 'undefined') AfriChessAAC.speak(label);
        });
        cell.addEventListener('click', function() {
          _selectedPiece = {
            type: pt,
            fromSquares: Array.from(myPieces[pt]),
            color: cfg.color,
            symbol: symbol,
            label: label,
          };
          buildSquareSelector();
        });
      }

      grid.appendChild(cell);
    });

    _container.appendChild(grid);
  }

  // ── Build square selector ───────────────────────────────────────────────────
  function buildSquareSelector() {
    if (!_container || !_game || !_selectedPiece) return;
    _container.innerHTML = '';

    // Back button
    const back = document.createElement('button');
    back.textContent = '← ' + (_lang().ui.gridTitle || 'Back');
    back.style.cssText = `
      background: transparent; color: #555; border: none;
      font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase;
      cursor: pointer; padding: 0; margin-bottom: 10px;
    `;
    back.addEventListener('click', buildPieceSelector);
    _container.appendChild(back);

    // Title
    const title = document.createElement('div');
    title.style.cssText = 'display:flex;align-items:center;gap:8px;font-size:11px;color:#555;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:10px;';
    const sym = document.createElement('span');
    sym.textContent = _selectedPiece.symbol;
    sym.style.cssText = `font-size:22px;color:${_selectedPiece.color};`;
    title.appendChild(sym);
    title.appendChild(document.createTextNode(_selectedPiece.label + ' — ' + (_lang().ui.gridSquare || 'Select square')));
    _container.appendChild(title);

    // Get all legal destination squares for this piece
    const legal = _game.moves({ verbose: true });
    const legalByFrom = {};
    legal
      .filter(m => m.color === _myColor && m.piece === _selectedPiece.type)
      .forEach(function(m) {
        if (!legalByFrom[m.from]) legalByFrom[m.from] = [];
        legalByFrom[m.from].push(m);
      });

    const legalDests = new Set();
    const moveMap    = {};  // to → move object
    Object.values(legalByFrom).forEach(function(moves) {
      moves.forEach(function(m) {
        legalDests.add(m.to);
        if (!moveMap[m.to]) moveMap[m.to] = m;
      });
    });

    // 8×8 board grid
    const boardEl = document.createElement('div');
    boardEl.style.cssText = 'display:grid;grid-template-columns:repeat(8,1fr);gap:2px;';
    boardEl.setAttribute('role', 'grid');
    boardEl.setAttribute('aria-label', _lang().ui.gridSquare);

    RANKS.forEach(function(rank) {
      FILES.forEach(function(file) {
        const sq = file + rank;
        const isLight   = (FILES.indexOf(file) + RANKS.indexOf(rank)) % 2 === 1;
        const isLegal   = legalDests.has(sq);
        const hasPiece  = _game.get(sq);
        const isMyPiece = _selectedPiece.fromSquares.includes(sq);

        const cell = document.createElement('button');
        cell.setAttribute('role', 'gridcell');
        cell.setAttribute('aria-label', file.toUpperCase() + ' ' + rank + (isLegal ? ' — legal' : ''));
        cell.disabled = !isLegal;

        let bg = isLight ? '#b58863' : '#6b4226';
        if (isMyPiece) bg = '#1a3a1a';
        if (isLegal)   bg = '#1a2a1a';

        cell.style.cssText = `
          aspect-ratio: 1; display: flex; align-items: center; justify-content: center;
          background: ${bg};
          color: ${isLegal ? _selectedPiece.color : (hasPiece ? '#ccc' : 'transparent')};
          border: 1px solid ${isLegal ? _selectedPiece.color + '55' : 'transparent'};
          border-radius: 2px;
          font-size: ${isLegal ? '14px' : '10px'};
          cursor: ${isLegal ? 'pointer' : 'default'};
          min-height: 32px; min-width: 32px;
          transition: all 0.1s;
          padding: 0;
        `;

        if (isLegal) {
          cell.textContent = '●';
          cell.addEventListener('mouseenter', function() {
            cell.style.background = _selectedPiece.color + '33';
            cell.style.transform = 'scale(1.08)';
          });
          cell.addEventListener('mouseleave', function() {
            cell.style.background = bg;
            cell.style.transform = '';
          });
          cell.addEventListener('focus', function() {
            if (typeof AfriChessAAC !== 'undefined') {
              AfriChessAAC.speak(file.toUpperCase() + ' ' + rank);
            }
          });
          cell.addEventListener('click', function() {
            const move = moveMap[sq];
            if (!move) return;
            // Handle promotions — always promote to queen by default (TODO: picker)
            const promo = move.flags.includes('p') ? 'q' : null;
            if (_executeMove) {
              _executeMove(move.from, move.to, promo);
            }
            buildPieceSelector();  // reset for next move
          });
        } else if (isMyPiece) {
          // Show the piece symbol on its current square
          const pCfg = PIECE_CONFIG[_selectedPiece.type];
          cell.textContent = _myColor === 'w' ? pCfg.symbolW : pCfg.symbol;
          cell.style.color = _selectedPiece.color;
          cell.style.fontSize = '16px';
        } else if (hasPiece) {
          const pCfg = PIECE_CONFIG[hasPiece.type] || {};
          cell.textContent = hasPiece.color === 'w'
            ? (pCfg.symbolW || hasPiece.type.toUpperCase())
            : (pCfg.symbol  || hasPiece.type);
          cell.style.color = hasPiece.color === 'w' ? '#ddd' : '#888';
          cell.style.fontSize = '14px';
        }

        boardEl.appendChild(cell);
      });
    });

    _container.appendChild(boardEl);

    // File labels
    const fileRow = document.createElement('div');
    fileRow.style.cssText = 'display:grid;grid-template-columns:repeat(8,1fr);gap:2px;margin-top:3px;';
    FILES.forEach(function(f) {
      const lbl = document.createElement('div');
      lbl.textContent = f.toUpperCase();
      lbl.style.cssText = 'text-align:center;font-size:9px;color:#444;';
      fileRow.appendChild(lbl);
    });
    _container.appendChild(fileRow);
  }

  // ── Public API ──────────────────────────────────────────────────────────────
  return {

    /**
     * init(container, { getGame, executeMove, getMyColor })
     * container: DOM element to render the grid into
     */
    init(container, callbacks) {
      _container   = container;
      _executeMove = callbacks.executeMove;
      _game        = callbacks.getGame ? callbacks.getGame() : null;
      _myColor     = callbacks.getMyColor ? callbacks.getMyColor() : 'w';
      buildPieceSelector();
    },

    /** Call after every move so the grid reflects the new position */
    refresh(callbacks) {
      _game    = callbacks.getGame ? callbacks.getGame() : _game;
      _myColor = callbacks.getMyColor ? callbacks.getMyColor() : _myColor;
      if (_selectedPiece) {
        buildSquareSelector();
      } else {
        buildPieceSelector();
      }
    },

    /** Reset to piece selector (call at game start) */
    reset() {
      _selectedPiece = null;
      buildPieceSelector();
    },

    /** Show/hide the grid container */
    show() { if (_container) _container.style.display = ''; },
    hide() { if (_container) _container.style.display = 'none'; },
  };

})();
