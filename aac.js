/**
 * aac.js — AfriChess Voice Accessibility Engine
 *
 * Phase 1: Voice OUTPUT (announcements) + language picker
 * Phase 2: Voice INPUT (spoken moves) — microphone button
 * Phase 3: Symbol grid (aac_grid.js)
 *
 * Usage:
 *   AfriChessAAC.init({ getGame, executeMove, getMyColor, getClocks });
 *   AfriChessAAC.onMove(move, board);   // call after every move
 *   AfriChessAAC.onGameEnd(result);     // call on game over
 *
 * Zero server dependencies. Zero external libraries.
 */

const AfriChessAAC = (function() {

  // ── State ──────────────────────────────────────────────────────────────────
  let _lang       = localStorage.getItem('aac_lang') || AAC_DEFAULT_LANG;
  let _enabled    = localStorage.getItem('aac_enabled') === 'true';  // default OFF
  let _micEnabled = false;
  let _recogniser = null;
  let _speaking   = false;
  let _callbacks  = {};
  let _scanMode   = false;
  let _scanInterval = null;
  let _scanIndex  = 0;

  // ── Helpers ─────────────────────────────────────────────────────────────────
  function S() { return aacStrings(_lang); }

  function sq(sqName) {
    // Humanise square name: "e4" → "E 4"
    if (!sqName || sqName.length < 2) return sqName || '';
    return sqName[0].toUpperCase() + ' ' + sqName[1];
  }

  function pieceName(pieceSymbol) {
    // pieceSymbol: 'p','n','b','r','q','k' (lowercase)
    return S().pieces[pieceSymbol.toLowerCase()] || pieceSymbol;
  }

  // ── Speech Synthesis ────────────────────────────────────────────────────────
  function speak(text, priority) {
    if (!_enabled || !text) return;
    if (!window.speechSynthesis) return;

    if (priority === 'interrupt') {
      window.speechSynthesis.cancel();
    }

    const utt = new SpeechSynthesisUtterance(text);
    utt.lang  = S().speechCode;
    utt.rate  = 0.95;
    utt.pitch = 1.0;

    // Try to find a voice for this language
    const voices = window.speechSynthesis.getVoices();
    const primary = voices.find(v => v.lang.startsWith(S().speechCode.split('-')[0]));
    const alt     = voices.find(v => v.lang.startsWith((S().altCode || 'en').split('-')[0]));
    if (primary) utt.voice = primary;
    else if (alt) utt.voice = alt;

    utt.onstart = () => { _speaking = true; };
    utt.onend   = () => { _speaking = false; };

    window.speechSynthesis.speak(utt);
  }

  // ── Move Announcer ──────────────────────────────────────────────────────────
  function announceMove(moveObj, board) {
    if (!moveObj) return;

    const flags = moveObj.flags || '';
    const piece = pieceName(moveObj.piece || 'p');
    const from  = moveObj.from;
    const to    = moveObj.to;
    const s     = S().moves;
    let text    = '';

    if (flags.includes('k')) {
      text = s.castle_k();
    } else if (flags.includes('q')) {
      text = s.castle_q();
    } else if (flags.includes('p') && moveObj.promotion) {
      const promo = pieceName(moveObj.promotion);
      text = s.promotes(piece, sq(to), promo);
    } else if (flags.includes('c') || flags.includes('e')) {
      // capture or en-passant
      const captured = moveObj.captured ? pieceName(moveObj.captured) : '';
      text = s.captures(piece, sq(to));
    } else {
      text = s.moveTo(piece, sq(to));
    }

    // Append check/checkmate
    if (board) {
      if (board.in_checkmate ? board.in_checkmate() : false) {
        const winner = board.turn() === 'b'
          ? S().game.white
          : S().game.black;
        text += '. ' + s.checkmate(winner);
      } else if (board.in_stalemate ? board.in_stalemate() : false) {
        text += '. ' + s.stalemate();
      } else if (board.in_check ? board.in_check() : false) {
        text += '. ' + s.check();
      }
    }

    speak(text, 'interrupt');
  }

  function announceGameState(myColor, opponentName, clocks) {
    const s = S().game;
    let text = s.yourTurn();
    if (clocks && myColor) {
      const key = myColor === 'w' ? 'w' : 'b';
      const secs = Math.floor(clocks[key] || 0);
      const mins = Math.floor(secs / 60);
      const rem  = secs % 60;
      const timeStr = mins + ':' + String(rem).padStart(2,'0');
      text += '. ' + s.timeLeft(timeStr);
    }
    speak(text);
  }

  function announceGameEnd(result) {
    const s = S().game;
    let text;
    if (result === 'draw')    text = s.draw();
    else if (result === 'win') text = s.youWin();
    else if (result === 'loss') text = s.youLose();
    else text = result || '';
    speak(text, 'interrupt');
  }

  // ── Voice Input Parser ──────────────────────────────────────────────────────
  // Parses a natural-language utterance into a UCI move attempt.
  // Returns { from, to, promotion } or null.
  function parseSpokenMove(utterance, board) {
    if (!utterance || !board) return null;

    const raw   = utterance.toLowerCase().trim();
    const vocab = S().voicePieces;
    const files = S().voiceFiles;

    // Castle check
    const castleQ = S().voiceCastleQ || [];
    const castle  = S().voiceCastle  || [];
    if (castleQ.some(c => raw.includes(c))) {
      return tryMove(board, null, null, 'q-castle');
    }
    if (castle.some(c => raw.includes(c)) || raw === 'castle') {
      return tryMove(board, null, null, 'k-castle');
    }

    // Normalise spoken file letters (alpha=a, bravo=b...)
    let text = raw;
    for (const [word, letter] of Object.entries(files)) {
      text = text.replace(new RegExp('\\b' + word + '\\b', 'g'), letter);
    }

    // Remove noise words
    const noise = ['to', 'the', 'on', 'at', 'square', 'please',
                   'move', 'play', 'goes', 'takes', 'captures',
                   'check', 'checkmate', 'en passant'];
    for (const w of noise) {
      text = text.replace(new RegExp('\\b' + w + '\\b', 'g'), ' ');
    }
    text = text.replace(/\s+/g, ' ').trim();

    // Extract piece
    let pieceCode = null;
    for (const [word, code] of Object.entries(vocab)) {
      if (text.includes(word)) {
        pieceCode = code;
        text = text.replace(word, '').trim();
        break;
      }
    }

    // Extract squares — match patterns like "e4", "e 4", "e-4"
    const sqPattern = /([a-h])\s*[-]?\s*([1-8])/g;
    const squares = [];
    let m;
    while ((m = sqPattern.exec(text)) !== null) {
      squares.push(m[1] + m[2]);
    }

    if (squares.length === 0) return null;

    const toSq   = squares[squares.length - 1];
    const fromSq = squares.length > 1 ? squares[0] : null;

    // Promotion
    let promotion = null;
    const promoWords = { queen:'q', rook:'r', bishop:'b', knight:'n',
                         ...Object.fromEntries(Object.entries(vocab).map(([w,c]) => [w, c])) };
    for (const [word, code] of Object.entries(promoWords)) {
      if (text.includes(word) && code !== pieceCode) {
        promotion = code;
        break;
      }
    }

    return tryMove(board, fromSq, toSq, null, pieceCode, promotion);
  }

  function tryMove(board, from, to, special, pieceCode, promotion) {
    if (!board) return null;

    const legalMoves = board.moves({ verbose: true });

    if (special === 'k-castle') {
      return legalMoves.find(m => m.flags.includes('k')) || null;
    }
    if (special === 'q-castle') {
      return legalMoves.find(m => m.flags.includes('q')) || null;
    }

    let candidates = legalMoves.filter(m => m.to === to);

    if (from)      candidates = candidates.filter(m => m.from === from);
    if (pieceCode) candidates = candidates.filter(m => m.piece === pieceCode);
    if (promotion) candidates = candidates.filter(m => m.promotion === promotion);

    if (candidates.length === 1) return candidates[0];
    if (candidates.length > 1 && promotion) {
      const p = candidates.find(m => m.promotion === 'q') || candidates[0];
      return p;
    }
    return null;
  }

  // ── Speech Recognition ──────────────────────────────────────────────────────
  function startListening() {
    if (!window.SpeechRecognition && !window.webkitSpeechRecognition) {
      speak(S().ui.notHeard);
      return;
    }
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    _recogniser = new SR();
    _recogniser.lang = S().speechCode;
    _recogniser.continuous = false;
    _recogniser.interimResults = false;

    updateMicButton(true);
    speak(S().ui.micOn, 'interrupt');

    _recogniser.onresult = function(e) {
      const transcript = e.results[0][0].transcript;
      handleSpokenMove(transcript);
    };

    _recogniser.onerror = function() {
      speak(S().ui.notHeard);
      updateMicButton(false);
    };

    _recogniser.onend = function() {
      updateMicButton(false);
    };

    _recogniser.start();
  }

  function stopListening() {
    if (_recogniser) {
      _recogniser.stop();
      _recogniser = null;
    }
    updateMicButton(false);
  }

  function handleSpokenMove(transcript) {
    if (!_callbacks.getGame || !_callbacks.executeMove) return;

    const board = _callbacks.getGame();
    if (!board) return;

    const move = parseSpokenMove(transcript, board);
    if (!move) {
      speak(S().ui.notHeard, 'interrupt');
      return;
    }

    // Confirm before executing
    const uiS = S().ui;
    const moveDesc = move.san || (move.from + move.to);
    speak(uiS.confirm(moveDesc), 'interrupt');

    // Wait for yes/no
    const confirmSR = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    confirmSR.lang = S().speechCode;
    confirmSR.onresult = function(e) {
      const ans = e.results[0][0].transcript.toLowerCase().trim();
      const yes = S().voiceYes || ['yes'];
      const no  = S().voiceNo  || ['no'];
      if (yes.some(w => ans.includes(w))) {
        _callbacks.executeMove(move.from, move.to, move.promotion || null);
      } else if (no.some(w => ans.includes(w))) {
        speak(S().ui.micOn);
        startListening();
      } else {
        speak(S().ui.notHeard);
      }
    };
    confirmSR.onerror = function() { speak(S().ui.notHeard); };
    confirmSR.start();
  }

  // ── UI Panel ────────────────────────────────────────────────────────────────
  function buildPanel() {
    // Remove any existing panel
    const existing = document.getElementById('aac-panel');
    if (existing) existing.remove();

    const s = S().ui;

    const panel = document.createElement('div');
    panel.id = 'aac-panel';
    panel.setAttribute('role', 'region');
    panel.setAttribute('aria-label', 'Voice Chess Controls');
    panel.style.cssText = `
      position: fixed;
      bottom: 80px;
      right: 16px;
      width: 240px;
      background: #0d0d0d;
      border: 1px solid #1e1e1e;
      border-radius: 6px;
      padding: 16px;
      z-index: 9999;
      font-family: monospace;
      font-size: 13px;
      color: #aaa;
      box-shadow: 0 4px 24px rgba(0,0,0,0.6);
      display: none;
    `;

    // Language picker
    const langRow = document.createElement('div');
    langRow.style.cssText = 'margin-bottom: 12px;';
    const langLabel = document.createElement('label');
    langLabel.textContent = s.language + ': ';
    langLabel.style.color = '#666';
    langLabel.htmlFor = 'aac-lang-select';
    const langSel = document.createElement('select');
    langSel.id = 'aac-lang-select';
    langSel.style.cssText = `
      background: #1a1a1a; color: #ccc; border: 1px solid #333;
      border-radius: 3px; padding: 4px 8px; font-size: 12px;
      cursor: pointer; width: 100%; margin-top: 4px;
    `;
    AAC_LANGUAGES.forEach(function(l) {
      const opt = document.createElement('option');
      opt.value = l.code;
      opt.textContent = l.name;
      opt.selected = l.code === _lang;
      langSel.appendChild(opt);
    });
    langSel.addEventListener('change', function() {
      _lang = this.value;
      localStorage.setItem('aac_lang', _lang);
      buildPanel();   // rebuild with new language
      showPanel();
      speak(S().game.gameStart());
    });
    langRow.appendChild(langLabel);
    langRow.appendChild(langSel);
    panel.appendChild(langRow);

    // Voice announcements toggle
    const announceRow = document.createElement('div');
    announceRow.style.cssText = 'display: flex; align-items: center; gap: 10px; margin-bottom: 10px;';
    const announceToggle = document.createElement('input');
    announceToggle.type = 'checkbox';
    announceToggle.id = 'aac-announce-toggle';
    announceToggle.checked = _enabled;
    announceToggle.style.cssText = 'width: 18px; height: 18px; cursor: pointer;';
    const announceLabel = document.createElement('label');
    announceLabel.htmlFor = 'aac-announce-toggle';
    announceLabel.textContent = _enabled ? s.voiceOn : s.voiceOff;
    announceLabel.style.color = _enabled ? '#00ff9d' : '#555';
    announceToggle.addEventListener('change', function() {
      _enabled = this.checked;
      localStorage.setItem('aac_enabled', String(_enabled));
      announceLabel.textContent = _enabled ? s.voiceOn : s.voiceOff;
      announceLabel.style.color = _enabled ? '#00ff9d' : '#555';
      if (_enabled) speak(s.voiceOn);
    });
    announceRow.appendChild(announceToggle);
    announceRow.appendChild(announceLabel);
    panel.appendChild(announceRow);

    // Microphone button
    const micBtn = document.createElement('button');
    micBtn.id = 'aac-mic-btn';
    micBtn.textContent = '🎤 ' + s.micOff;
    micBtn.setAttribute('aria-label', s.micOff);
    micBtn.style.cssText = `
      width: 100%; padding: 10px; margin-bottom: 10px;
      background: #111; color: #888; border: 1px solid #333;
      border-radius: 4px; cursor: pointer; font-size: 13px;
      letter-spacing: 0.06em; text-transform: uppercase;
      transition: all 0.15s;
    `;
    micBtn.addEventListener('click', function() {
      if (_micEnabled) {
        stopListening();
        _micEnabled = false;
      } else {
        _micEnabled = true;
        startListening();
      }
    });
    panel.appendChild(micBtn);

    // Speak current position button
    const posBtn = document.createElement('button');
    posBtn.id = 'aac-pos-btn';
    posBtn.textContent = '♟ Read Position';
    posBtn.setAttribute('aria-label', 'Read current position aloud');
    posBtn.style.cssText = micBtn.style.cssText + 'margin-bottom: 8px;';
    posBtn.addEventListener('click', function() {
      if (_callbacks.announcePosition) _callbacks.announcePosition();
      else if (_callbacks.getGame && _callbacks.getMyColor && _callbacks.getClocks) {
        announceGameState(
          _callbacks.getMyColor(),
          null,
          _callbacks.getClocks()
        );
      }
    });
    panel.appendChild(posBtn);

    // Switch scanning toggle
    const scanRow = document.createElement('div');
    scanRow.style.cssText = 'display: flex; align-items: center; gap: 10px; margin-top: 4px;';
    const scanToggle = document.createElement('input');
    scanToggle.type = 'checkbox';
    scanToggle.id = 'aac-scan-toggle';
    scanToggle.checked = _scanMode;
    scanToggle.style.cssText = 'width: 16px; height: 16px; cursor: pointer;';
    const scanLabel = document.createElement('label');
    scanLabel.htmlFor = 'aac-scan-toggle';
    scanLabel.textContent = s.scanning;
    scanLabel.style.cssText = 'color: #555; font-size: 11px;';
    scanToggle.addEventListener('change', function() {
      _scanMode = this.checked;
      if (_scanMode) startScanning(panel);
      else stopScanning();
    });
    scanRow.appendChild(scanToggle);
    scanRow.appendChild(scanLabel);
    panel.appendChild(scanRow);

    document.body.appendChild(panel);
    return panel;
  }

  function updateMicButton(listening) {
    const btn = document.getElementById('aac-mic-btn');
    if (!btn) return;
    const s = S().ui;
    if (listening) {
      btn.textContent = '🔴 ' + s.listening;
      btn.style.color = '#00ff9d';
      btn.style.borderColor = '#00ff9d44';
      _micEnabled = true;
    } else {
      btn.textContent = '🎤 ' + s.micOff;
      btn.style.color = '#888';
      btn.style.borderColor = '#333';
      _micEnabled = false;
    }
  }

  // ── Toggle Button (floating ♿ button) ─────────────────────────────────────
  function buildToggleButton() {
    const existing = document.getElementById('aac-toggle-btn');
    if (existing) existing.remove();

    const btn = document.createElement('button');
    btn.id = 'aac-toggle-btn';
    btn.textContent = '♿';
    btn.title = S().ui.toggle;
    btn.setAttribute('aria-label', S().ui.toggle);
    btn.style.cssText = `
      position: fixed;
      bottom: 20px;
      right: 16px;
      width: 48px;
      height: 48px;
      border-radius: 50%;
      background: #1a1a1a;
      color: #00ff9d;
      border: 1px solid #00ff9d44;
      font-size: 22px;
      cursor: pointer;
      z-index: 9999;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 2px 12px rgba(0,0,0,0.5);
      transition: background 0.15s;
    `;
    btn.addEventListener('mouseenter', () => btn.style.background = '#222');
    btn.addEventListener('mouseleave', () => btn.style.background = '#1a1a1a');
    btn.addEventListener('click', function() {
      togglePanel();
      // Update button style to reflect panel state
      btn.style.background    = _panelVisible ? '#00ff9d22' : '#1a1a1a';
      btn.style.borderColor   = _panelVisible ? '#00ff9d88' : '#00ff9d44';
    });
    document.body.appendChild(btn);
  }

  let _panelVisible = false;

  function showPanel() {
    const panel = document.getElementById('aac-panel') || buildPanel();
    panel.style.display = 'block';
    _panelVisible = true;
  }

  function hidePanel() {
    const panel = document.getElementById('aac-panel');
    if (panel) panel.style.display = 'none';
    _panelVisible = false;
  }

  function togglePanel() {
    if (_panelVisible) hidePanel();
    else { buildPanel(); showPanel(); }
  }

  // ── Switch Scanning ─────────────────────────────────────────────────────────
  function startScanning(panel) {
    stopScanning();
    const focusable = panel.querySelectorAll('button, input, select');
    _scanIndex = 0;
    _scanInterval = setInterval(function() {
      focusable.forEach(function(el) { el.style.outline = ''; });
      if (focusable[_scanIndex]) {
        focusable[_scanIndex].style.outline = '3px solid #00ff9d';
        const label = focusable[_scanIndex].getAttribute('aria-label')
                   || focusable[_scanIndex].textContent;
        if (label) speak(label.trim().split('\n')[0]);
      }
      _scanIndex = (_scanIndex + 1) % focusable.length;
    }, 2500);

    // Space bar activates current item
    document.addEventListener('keydown', _scanKeyHandler);
  }

  function stopScanning() {
    if (_scanInterval) { clearInterval(_scanInterval); _scanInterval = null; }
    document.removeEventListener('keydown', _scanKeyHandler);
    document.querySelectorAll('#aac-panel button, #aac-panel input, #aac-panel select')
      .forEach(function(el) { el.style.outline = ''; });
  }

  function _scanKeyHandler(e) {
    if (e.code === 'Space' || e.code === 'Enter') {
      const panel = document.getElementById('aac-panel');
      if (!panel) return;
      const focusable = panel.querySelectorAll('button, input, select');
      const target = focusable[(_scanIndex - 1 + focusable.length) % focusable.length];
      if (target) target.click();
    }
  }

  // ── Public API ──────────────────────────────────────────────────────────────
  return {

    /**
     * init({ getGame, executeMove, getMyColor, getClocks, announcePosition })
     * Call once per page after the game is ready.
     * getGame()        → chess.js Chess instance
     * executeMove(f,t,p) → executes the move on the board
     * getMyColor()     → 'w' or 'b'
     * getClocks()      → { w: seconds, b: seconds }
     */
    init(callbacks) {
      _callbacks = callbacks || {};
      buildToggleButton();

      // Preload voices (Chrome needs this triggered by user interaction)
      if (window.speechSynthesis) {
        window.speechSynthesis.onvoiceschanged = function() {
          window.speechSynthesis.getVoices();
        };
        window.speechSynthesis.getVoices();
      }
    },

    /**
     * Call after every move with the chess.js move object and current board.
     * move: { piece, from, to, flags, captured, promotion, san }
     */
    onMove(moveObj, board) {
      announceMove(moveObj, board);

      // After opponent's move, announce it's your turn
      setTimeout(function() {
        const myColor = _callbacks.getMyColor ? _callbacks.getMyColor() : null;
        const clocks  = _callbacks.getClocks  ? _callbacks.getClocks()  : null;
        if (board && myColor) {
          const boardTurn = board.turn ? board.turn() : null;
          if (boardTurn === myColor) {
            announceGameState(myColor, null, clocks);
          }
        }
      }, 1200);
    },

    /** Call when the game ends */
    onGameEnd(result) {
      announceGameEnd(result);
    },

    /** Called when a new game starts */
    onGameStart(myColor) {
      const s = S().game;
      const colorName = myColor === 'w' ? s.white : s.black;
      speak(s.gameStart() + '. ' + s.youPlay(colorName), 'interrupt');
    },

    /** Programmatically speak any text */
    speak(text, priority) {
      speak(text, priority);
    },

    /** Change language */
    setLanguage(code) {
      _lang = code;
      localStorage.setItem('aac_lang', code);
      if (_panelVisible) { buildPanel(); showPanel(); }
    },

    /** Toggle voice on/off */
    setEnabled(val) {
      _enabled = val;
      localStorage.setItem('aac_enabled', val);
    },

    isEnabled() { return _enabled; },
    getLanguage() { return _lang; },
  };

})();
