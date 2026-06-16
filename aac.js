/**
 * aac.js — AfriChess Voice Accessibility Engine
 * Default OFF. User opts in via ♿ button.
 * Fixed issues: mobile positioning, auto-scan, auto-speak on open,
 *               mic starting without active game, self-interaction loop.
 */

const AfriChessAAC = (function() {

  // ── State ──────────────────────────────────────────────────────────────────
  let _lang         = localStorage.getItem('aac_lang') || AAC_DEFAULT_LANG;
  let _enabled      = localStorage.getItem('aac_enabled') === 'true'; // default OFF
  let _micEnabled   = false;
  let _recogniser   = null;
  let _callbacks    = {};
  let _scanMode     = false;   // never auto-start scanning
  let _scanInterval = null;
  let _scanIndex    = 0;
  let _panelVisible = false;
  let _gameActive   = false;   // true only when a game is actually in progress

  function S() { return aacStrings(_lang); }

  // ── Speech Synthesis ────────────────────────────────────────────────────────
  function speak(text, priority) {
    if (!_enabled || !text) return;
    if (!window.speechSynthesis) return;
    if (priority === 'interrupt') window.speechSynthesis.cancel();

    // Pause the mic while speaking — browser conflicts if both run together
    // _micEnabled stays true so we know to restart after
    if (_recogniser) {
      try { _recogniser.abort(); } catch(e) {}
      _recogniser = null;
    }
    const utt = new SpeechSynthesisUtterance(text);
    utt.lang  = S().speechCode;
    utt.rate  = 0.95;
    utt.pitch = 1.0;
    const voices  = window.speechSynthesis.getVoices();
    const primary = voices.find(v => v.lang.startsWith(S().speechCode.split('-')[0]));
    const alt     = S().altCode ? voices.find(v => v.lang.startsWith(S().altCode.split('-')[0])) : null;
    if (primary) utt.voice = primary;
    else if (alt) utt.voice = alt;

    // When synthesis ends, restart mic if user had it on
    utt.onend = function() {
      if (_micEnabled && _gameActive) {
        // Small delay so the mic doesn't pick up the tail of the TTS audio
        setTimeout(function() {
          if (_micEnabled && _gameActive && _recogniser) {
            try { _recogniser.start(); } catch(e) {}
          } else if (_micEnabled && _gameActive) {
            startListening();
          }
        }, 400);
      }
    };

    window.speechSynthesis.speak(utt);
  }

  // ── Move Announcer ──────────────────────────────────────────────────────────
  function announceMove(moveObj, board) {
    if (!moveObj || !_enabled) return;
    const flags  = moveObj.flags || '';
    const piece  = (S().pieces[moveObj.piece] || moveObj.piece || '').toLowerCase();
    const to     = moveObj.to ? (moveObj.to[0].toUpperCase() + ' ' + moveObj.to[1]) : '';
    const s      = S().moves;
    let text     = '';

    if (flags.includes('k'))      text = s.castle_k();
    else if (flags.includes('q')) text = s.castle_q();
    else if (flags.includes('p') && moveObj.promotion)
      text = s.promotes(piece, to, S().pieces[moveObj.promotion] || moveObj.promotion);
    else if (flags.includes('c') || flags.includes('e'))
      text = s.captures(piece, to);
    else
      text = s.moveTo(piece, to);

    if (board) {
      try {
        if (board.in_checkmate && board.in_checkmate()) {
          const winner = board.turn() === 'b' ? S().game.white : S().game.black;
          text += '. ' + s.checkmate(winner);
        } else if (board.in_stalemate && board.in_stalemate()) {
          text += '. ' + s.stalemate();
        } else if (board.in_check && board.in_check()) {
          text += '. ' + s.check();
        }
      } catch(e) {}
    }
    speak(text, 'interrupt');
  }

  function announceGameState() {
    if (!_gameActive) return;
    const myColor = _callbacks.getMyColor ? _callbacks.getMyColor() : null;
    const clocks  = _callbacks.getClocks  ? _callbacks.getClocks()  : null;
    let text = S().game.yourTurn();
    if (clocks && myColor) {
      const key  = myColor === 'w' ? 'w' : 'b';
      const secs = Math.floor(clocks[key] || 0);
      const mins = Math.floor(secs / 60);
      const rem  = secs % 60;
      text += '. ' + S().game.timeLeft(mins + ':' + String(rem).padStart(2, '0'));
    }
    speak(text);
  }

  // ── Voice Input ─────────────────────────────────────────────────────────────
  function parseSpokenMove(utterance, board) {
    if (!utterance || !board) return null;
    const raw   = utterance.toLowerCase().trim();
    const vocab = S().voicePieces || {};
    const files = S().voiceFiles || {};

    const castleQ = S().voiceCastleQ || [];
    const castle  = S().voiceCastle  || [];
    if (castleQ.some(c => raw.includes(c))) return tryMove(board, null, null, 'q-castle');
    if (castle.some(c => raw.includes(c)) || raw === 'castle') return tryMove(board, null, null, 'k-castle');

    let text = raw;
    for (const [word, letter] of Object.entries(files)) {
      text = text.replace(new RegExp('\\b' + word + '\\b', 'g'), letter);
    }
    const noise = ['to','the','on','at','square','please','move','play','goes',
                   'takes','captures','check','checkmate','en passant'];
    for (const w of noise) text = text.replace(new RegExp('\\b' + w + '\\b', 'g'), ' ');
    text = text.replace(/\s+/g, ' ').trim();

    let pieceCode = null;
    for (const [word, code] of Object.entries(vocab)) {
      if (text.includes(word)) { pieceCode = code; text = text.replace(word, '').trim(); break; }
    }

    const sqPattern = /([a-h])\s*[-]?\s*([1-8])/g;
    const squares = [];
    let m;
    while ((m = sqPattern.exec(text)) !== null) squares.push(m[1] + m[2]);
    if (squares.length === 0) return null;

    const toSq   = squares[squares.length - 1];
    const fromSq = squares.length > 1 ? squares[0] : null;
    return tryMove(board, fromSq, toSq, null, pieceCode, null);
  }

  function tryMove(board, from, to, special, pieceCode, promotion) {
    if (!board) return null;
    const legalMoves = board.moves({ verbose: true });
    if (special === 'k-castle') return legalMoves.find(m => m.flags.includes('k')) || null;
    if (special === 'q-castle') return legalMoves.find(m => m.flags.includes('q')) || null;
    let candidates = legalMoves.filter(m => m.to === to);
    if (from)      candidates = candidates.filter(m => m.from === from);
    if (pieceCode) candidates = candidates.filter(m => m.piece === pieceCode);
    if (candidates.length === 1) return candidates[0];
    if (candidates.length > 1) return candidates.find(m => !m.promotion || m.promotion === 'q') || candidates[0];
    return null;
  }

  function startListening() {
    // Guard: only allow mic if a game is active
    if (!_gameActive) {
      speak('No active game. Start a game first.');
      return;
    }
    if (!window.SpeechRecognition && !window.webkitSpeechRecognition) {
      speak(S().ui.notHeard); return;
    }
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    _recogniser = new SR();
    _recogniser.lang = S().speechCode;
    _recogniser.continuous = false;      // one utterance at a time
    _recogniser.interimResults = false;
    _recogniser.maxAlternatives = 1;

    updateMicButton(true);

    _recogniser.onresult = function(e) {
      const transcript = e.results[0][0].transcript;
      updateMicButton(false);
      _micEnabled = false;
      handleSpokenMove(transcript);
    };
    _recogniser.onerror = function(e) {
      // 'no-speech' = silence timeout — restart automatically if still enabled
      if (e.error === 'no-speech' && _micEnabled) {
        try { _recogniser.start(); return; } catch(err) {}
      }
      speak(S().ui.notHeard);
      updateMicButton(false);
      _micEnabled = false;
    };
    _recogniser.onend = function() {
      // Auto-restart if user hasn't manually stopped
      if (_micEnabled) {
        try { _recogniser.start(); return; } catch(err) {}
      }
      updateMicButton(false);
    };
    try {
      _recogniser.start();
    } catch(e) {
      updateMicButton(false);
      _micEnabled = false;
    }
  }

  function stopListening() {
    _micEnabled = false;  // set BEFORE stop() so onend doesn't restart
    if (_recogniser) {
      try { _recogniser.abort(); } catch(e) {}
      _recogniser = null;
    }
    updateMicButton(false);
  }

  function handleSpokenMove(transcript) {
    if (!_callbacks.getGame || !_callbacks.executeMove || !_gameActive) return;
    const board = _callbacks.getGame();
    if (!board) return;
    const move = parseSpokenMove(transcript, board);
    if (!move) { speak(S().ui.notHeard, 'interrupt'); return; }

    const moveDesc = move.san || (move.from + move.to);
    speak(S().ui.confirm(moveDesc), 'interrupt');

    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { _callbacks.executeMove(move.from, move.to, move.promotion || null); return; }

    const confirmSR = new SR();
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
    try { confirmSR.start(); } catch(e) {}
  }

  // ── Panel ───────────────────────────────────────────────────────────────────
  function buildPanel() {
    const existing = document.getElementById('aac-panel');
    if (existing) existing.remove();
    const s = S().ui;

    const panel = document.createElement('div');
    panel.id = 'aac-panel';
    panel.setAttribute('role', 'region');
    panel.setAttribute('aria-label', 'Voice Chess Controls');

    // Position: always bottom-left so it never conflicts with hamburger (top-right)
    // Panel opens upward from the button
    const isMobile = window.innerWidth <= 600;
    panel.style.cssText = `
      position: fixed;
      bottom: ${isMobile ? '60px' : '76px'};
      left: ${isMobile ? '8px' : '16px'};
      width: ${isMobile ? '200px' : '240px'};
      background: #0d0d0d;
      border: 1px solid #1e1e1e;
      border-radius: 6px;
      padding: 12px;
      z-index: 9999;
      font-family: monospace;
      font-size: ${isMobile ? '11px' : '13px'};
      color: #aaa;
      box-shadow: 0 4px 24px rgba(0,0,0,0.8);
      display: none;
      max-height: 70vh;
      overflow-y: auto;
    `;

    // ── Close button ──
    const closeBtn = document.createElement('button');
    closeBtn.textContent = '✕';
    closeBtn.style.cssText = 'float:right;background:none;border:none;color:#555;font-size:14px;cursor:pointer;padding:0;margin:-2px -2px 6px 0;';
    closeBtn.onclick = function() { hidePanel(); };
    panel.appendChild(closeBtn);

    // ── Language picker ──
    const langRow = document.createElement('div');
    langRow.style.cssText = 'margin-bottom: 10px; clear: both;';
    const langSel = document.createElement('select');
    langSel.id = 'aac-lang-select';
    langSel.style.cssText = 'background:#1a1a1a;color:#ccc;border:1px solid #333;border-radius:3px;padding:3px 6px;font-size:11px;cursor:pointer;width:100%;';
    AAC_LANGUAGES.forEach(function(l) {
      const opt = document.createElement('option');
      opt.value = l.code; opt.textContent = l.name; opt.selected = l.code === _lang;
      langSel.appendChild(opt);
    });
    langSel.addEventListener('change', function() {
      _lang = this.value;
      localStorage.setItem('aac_lang', _lang);
      buildPanel(); showPanel();
      // Don't speak on language change — avoids self-trigger loop
    });
    langRow.appendChild(langSel);
    panel.appendChild(langRow);

    // ── Voice toggle ──
    const announceRow = document.createElement('div');
    announceRow.style.cssText = 'display:flex;align-items:center;gap:8px;margin-bottom:8px;';
    const announceToggle = document.createElement('input');
    announceToggle.type = 'checkbox'; announceToggle.id = 'aac-announce-toggle';
    announceToggle.checked = _enabled;
    announceToggle.style.cssText = 'width:16px;height:16px;cursor:pointer;flex-shrink:0;';
    const announceLabel = document.createElement('label');
    announceLabel.htmlFor = 'aac-announce-toggle';
    announceLabel.textContent = _enabled ? s.voiceOn : s.voiceOff;
    announceLabel.style.cssText = 'color:' + (_enabled ? '#00ff9d' : '#555') + ';font-size:11px;cursor:pointer;';
    announceToggle.addEventListener('change', function() {
      _enabled = this.checked;
      localStorage.setItem('aac_enabled', String(_enabled));
      announceLabel.textContent = _enabled ? s.voiceOn : s.voiceOff;
      announceLabel.style.color = _enabled ? '#00ff9d' : '#555';
      // Only speak if turning ON and game is active
      if (_enabled && _gameActive) speak(s.voiceOn);
    });
    announceRow.appendChild(announceToggle);
    announceRow.appendChild(announceLabel);
    panel.appendChild(announceRow);

    // ── Mic button ──
    const micBtn = document.createElement('button');
    micBtn.id = 'aac-mic-btn';
    micBtn.textContent = '🎤 ' + s.micOff;
    micBtn.setAttribute('aria-label', s.micOff);
    const btnCss = 'width:100%;padding:8px;margin-bottom:6px;background:#111;color:#888;border:1px solid #333;border-radius:3px;cursor:pointer;font-size:11px;letter-spacing:0.04em;text-transform:uppercase;';
    micBtn.style.cssText = btnCss;
    micBtn.addEventListener('click', function(e) {
      e.stopPropagation();  // prevent panel click-through
      if (_micEnabled) { stopListening(); _micEnabled = false; }
      else { _micEnabled = true; startListening(); }
    });
    panel.appendChild(micBtn);

    // ── Read position ──
    const posBtn = document.createElement('button');
    posBtn.id = 'aac-pos-btn';
    posBtn.textContent = '♟ Read Position';
    posBtn.style.cssText = btnCss;
    posBtn.addEventListener('click', function(e) {
      e.stopPropagation();
      announceGameState();
    });
    panel.appendChild(posBtn);

    // ── Switch scanning — hidden on mobile (too complex, touch is better) ──
    if (!isMobile) {
      const scanRow = document.createElement('div');
      scanRow.style.cssText = 'display:flex;align-items:center;gap:8px;margin-top:4px;';
      const scanToggle = document.createElement('input');
      scanToggle.type = 'checkbox'; scanToggle.id = 'aac-scan-toggle';
      scanToggle.checked = false; // always starts off
      scanToggle.style.cssText = 'width:14px;height:14px;cursor:pointer;';
      const scanLabel = document.createElement('label');
      scanLabel.htmlFor = 'aac-scan-toggle';
      scanLabel.textContent = s.scanning;
      scanLabel.style.cssText = 'color:#444;font-size:10px;';
      scanToggle.addEventListener('change', function() {
        _scanMode = this.checked;
        if (_scanMode) startScanning(panel);
        else stopScanning();
      });
      scanRow.appendChild(scanToggle); scanRow.appendChild(scanLabel);
      panel.appendChild(scanRow);
    }

    document.body.appendChild(panel);
    return panel;
  }

  function updateMicButton(listening) {
    const btn = document.getElementById('aac-mic-btn');
    if (!btn) return;
    const s = S().ui;
    if (listening) {
      btn.textContent = '🔴 ' + s.listening;
      btn.style.color = '#00ff9d'; btn.style.borderColor = '#00ff9d44';
    } else {
      btn.textContent = '🎤 ' + s.micOff;
      btn.style.color = '#888'; btn.style.borderColor = '#333';
      _micEnabled = false;
    }
  }

  // ── Toggle button ─────────────────────────────────────────────────────────
  function buildToggleButton() {
    const existing = document.getElementById('aac-toggle-btn');
    if (existing) existing.remove();

    const btn = document.createElement('button');
    btn.id = 'aac-toggle-btn';
    btn.textContent = '♿';
    btn.title = S().ui.toggle;
    btn.setAttribute('aria-label', S().ui.toggle);

    const isMobile = window.innerWidth <= 600;
    btn.style.cssText = `
      position: fixed;
      bottom: ${isMobile ? '12px' : '20px'};
      left: ${isMobile ? '8px' : '16px'};
      width: ${isMobile ? '36px' : '44px'};
      height: ${isMobile ? '36px' : '44px'};
      font-size: ${isMobile ? '16px' : '20px'};
      border-radius: 50%;
      background: #111;
      color: #555;
      border: 1px solid #2a2a2a;
      cursor: pointer;
      z-index: 10000;
      display: flex; align-items: center; justify-content: center;
      box-shadow: 0 2px 8px rgba(0,0,0,0.4);
      transition: all 0.15s;
      opacity: 0.7;
    `;
    btn.addEventListener('click', function(e) {
      e.stopPropagation();
      togglePanel();
      btn.style.opacity     = _panelVisible ? '1' : '0.7';
      btn.style.color       = _panelVisible ? '#00ff9d' : '#555';
      btn.style.borderColor = _panelVisible ? '#00ff9d44' : '#2a2a2a';
    });
    document.body.appendChild(btn);
  }

  function showPanel() {
    const panel = document.getElementById('aac-panel') || buildPanel();
    panel.style.display = 'block';
    _panelVisible = true;
    // Do NOT speak anything on open — avoid self-trigger on mobile
  }

  function hidePanel() {
    const panel = document.getElementById('aac-panel');
    if (panel) panel.style.display = 'none';
    stopScanning();
    stopListening();
    _panelVisible = false;
  }

  function togglePanel() {
    if (_panelVisible) hidePanel();
    else { buildPanel(); showPanel(); }
  }

  // ── Switch scanning ────────────────────────────────────────────────────────
  function startScanning(panel) {
    stopScanning();
    const focusable = panel.querySelectorAll('button:not([disabled]), input, select');
    _scanIndex = 0;
    _scanInterval = setInterval(function() {
      focusable.forEach(function(el) { el.style.outline = ''; });
      const el = focusable[_scanIndex];
      if (el) {
        el.style.outline = '2px solid #00ff9d';
        // Only speak label if voice is on
        if (_enabled) {
          const label = el.getAttribute('aria-label') || el.textContent.trim().split('\n')[0];
          if (label) speak(label.slice(0, 40));
        }
      }
      _scanIndex = (_scanIndex + 1) % focusable.length;
    }, 2500);
    document.addEventListener('keydown', _scanKeyHandler);
  }

  function stopScanning() {
    if (_scanInterval) { clearInterval(_scanInterval); _scanInterval = null; }
    document.removeEventListener('keydown', _scanKeyHandler);
    document.querySelectorAll('#aac-panel button, #aac-panel input, #aac-panel select')
      .forEach(function(el) { el.style.outline = ''; });
    _scanMode = false;
    const tog = document.getElementById('aac-scan-toggle');
    if (tog) tog.checked = false;
  }

  function _scanKeyHandler(e) {
    if (e.code === 'Space' || e.code === 'Enter') {
      const panel = document.getElementById('aac-panel');
      if (!panel) return;
      const focusable = panel.querySelectorAll('button:not([disabled]), input, select');
      const idx = (_scanIndex - 1 + focusable.length) % focusable.length;
      if (focusable[idx]) focusable[idx].click();
    }
  }

  // ── Public API ─────────────────────────────────────────────────────────────
  return {

    init(callbacks) {
      _callbacks = callbacks || {};
      _gameActive = false;  // no game yet
      buildToggleButton();
      if (window.speechSynthesis) {
        window.speechSynthesis.onvoiceschanged = function() {
          window.speechSynthesis.getVoices();
        };
        window.speechSynthesis.getVoices();
      }
    },

    onGameStart(myColor) {
      _gameActive = true;
      stopListening();   // clean state at game start
      if (!_enabled) return;
      const s = S().game;
      const colorName = myColor === 'w' ? s.white : s.black;
      speak(s.gameStart() + '. ' + s.youPlay(colorName), 'interrupt');
    },

    onMove(moveObj, board) {
      if (!_gameActive) return;
      announceMove(moveObj, board);
      // Announce whose turn after a short delay
      setTimeout(function() {
        if (!_gameActive) return;
        const myColor = _callbacks.getMyColor ? _callbacks.getMyColor() : null;
        if (board && myColor && board.turn && board.turn() === myColor) {
          announceGameState();
        }
      }, 1200);
    },

    onGameEnd(result) {
      _gameActive = false;
      stopListening();
      if (!_enabled) return;
      const s = S().game;
      const text = result === 'win' ? s.youWin()
                 : result === 'loss' ? s.youLose()
                 : s.draw();
      speak(text, 'interrupt');
    },

    speak(text, priority) { speak(text, priority); },
    setLanguage(code) {
      _lang = code;
      localStorage.setItem('aac_lang', code);
      if (_panelVisible) { buildPanel(); showPanel(); }
    },
    setEnabled(val) {
      _enabled = val;
      localStorage.setItem('aac_enabled', String(val));
    },
    isEnabled()   { return _enabled; },
    getLanguage() { return _lang; },
  };

})();
