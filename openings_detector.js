/**
 * openings_detector.js — AfriChess Opening Detection
 *
 * Requires: openings.js (OPENING_DB constant)
 * Works on: vs bot, multiplayer, tournament pages
 *
 * Usage:
 *   OpeningDetector.detect(uciMoveArray) → { eco, name } or null
 *   OpeningDetector.render(uciMoveArray, containerEl)
 */

const OpeningDetector = (function() {

  // Find the deepest matching opening for the current move sequence
  function detect(uciMoves) {
    if (!uciMoves || !uciMoves.length || typeof OPENING_DB === 'undefined') return null;

    let best = null;
    // Walk from full sequence back to 1 move — find longest match
    for (let len = uciMoves.length; len >= 1; len--) {
      const key = uciMoves.slice(0, len).join(' ');
      if (OPENING_DB[key]) {
        best = { ...OPENING_DB[key], moves: len };
        break;
      }
    }
    return best;
  }

  // Render opening name into a container element
  // Creates the element if it doesn't exist yet
  function render(uciMoves, container) {
    if (!container) return;

    let el = container.querySelector('.opening-name-display');
    if (!el) {
      el = document.createElement('div');
      el.className = 'opening-name-display';
      el.style.cssText = `
        font-size: 11px;
        color: #555;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 4px 0 2px;
        min-height: 18px;
        transition: color 0.3s;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      `;
      container.appendChild(el);
    }

    const opening = detect(uciMoves);
    if (opening) {
      el.textContent = opening.eco + '  ' + opening.name;
      el.style.color = '#00ff9d88';
      el.title = opening.eco + ' — ' + opening.name;
    } else if (uciMoves && uciMoves.length > 0) {
      el.textContent = 'Out of book';
      el.style.color = '#333';
      el.title = '';
    } else {
      el.textContent = '';
    }
  }

  return { detect, render };
})();
