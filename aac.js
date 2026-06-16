/**
 * aac.js — AfriChess Voice Accessibility Engine
 * Default OFF. User opts in via ♿ button.
 * Fixed issues: mobile positioning, auto-scan, auto-speak on open,
 *               mic starting without active game, self-interaction loop.
 */

const AfriChessAAC = (function() {

  // ── State ──────────────────────────────────────────────────────────────────
  let _lang              = localStorage.getItem('aac_lang') || AAC_DEFAULT_LANG;
  let _enabled           = localStorage.getItem('aac_enabled') === 'true'; // default OFF
  let _micEnabled        = false;
  let _recogniser        = null;
  let _callbacks         = {};
  let _scanMode          = false;
  let _scanInterval      = null;
  let _scanIndex         = 0;
  let _panelVisible      = false;
  let _gameActive        = false;
  let _isSpeaking        = false;  // true while TTS is playing — guards mic restart
  let _voiceState        = 'IDLE'; // IDLE | LISTENING | CONFIRMING
  let _pendingMove       = null;   // move awaiting yes/no confirmation

  function S() { return aacStrings(_lang); }

  // ── Speech Synthesis ────────────────────────────────────────────────────────
  // safeSpeak: the ONLY way audio is produced.
  // Stops mic, speaks, then restarts mic if it was on.
  // _isSpeaking flag prevents onend from restarting the mic while still talking.
  function speak(text, priority) {
    if (!_enabled || !text) return;
    if (!window.speechSynthesis) return;
    if (priority === 'interrupt') window.speechSynthesis.cancel();

    // Stop mic before speaking — sets _isSpeaking so onend knows not to auto-restart
    _isSpeaking = true;
    if (_recogniser) {
      try { _recogniser.abort(); } catch(e) {}
      _recogniser = null;
      updateMicButton(false);
    }

    const utt   = new SpeechSynthesisUtterance(text);
    utt.lang    = S().speechCode;
    utt.rate    = 0.95;
    utt.pitch   = 1.0;

    const voices  = window.speechSynthesis.getVoices();
    const primary = voices.find(v => v.lang.startsWith(S().speechCode.split('-')[0]));
    const alt     = S().altCode ? voices.find(v => v.lang.startsWith(S().altCode.split('-')[0])) : null;
    if (primary) utt.voice = primary;
    else if (alt) utt.voice = alt;

    utt.onend = function() {
      _isSpeaking = false;
      // Restart mic only if user had it on AND we're in the right state
      // State machine decides whether to listen for move or for confirmation
      if (_micEnabled && _gameActive) {
        setTimeout(function() {
          if (!_isSpeaking && _micEnabled && _gameActive) {
            if (_voiceState === 'CONFIRMING') {
              startConfirmationListener();
            } else if (_voiceState === 'LISTENING') {
              startListening();
            }
          }
        }, 800);  // 800ms — enough for TTS audio to fully clear the mic
      }
    };

    // Catch synthesis failures (mobile sometimes fails silently)
    utt.onerror = function() { _isSpeaking = false; };

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
    if (!_gameActive || _isSpeaking) return;
    if (!window.SpeechRecognition && !window.webkitSpeechRecognition) return;
    if (_recogniser) return;  // already active

    _voiceState = 'LISTENING';
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    _recogniser = new SR();
    _recogniser.lang = S().speechCode;
    _recogniser.continuous = false;
    _recogniser.interimResults = false;
    _recogniser.maxAlternatives = 1;

    updateMicButton(true);

    _recogniser.onresult = function(e) {
      const transcript = e.results[0][0].transcript;
      _recogniser = null;
      updateMicButton(false);
      processVoiceInput(transcript);
    };

    _recogniser.onerror = function(e) {
      _recogniser = null;
      if (_isSpeaking) return;  // speaking caused this — onend will restart
      if (e.error === 'no-speech') {
        // Timed out waiting — restart silently
        if (_micEnabled && _gameActive) setTimeout(startListening, 200);
        return;
      }
      updateMicButton(false);
    };

    _recogniser.onend = function() {
      _recogniser = null;
      if (_isSpeaking) return;  // TTS caused abort — speak.utt.onend will restart
      if (_micEnabled && _gameActive && _voiceState === 'LISTENING') {
        setTimeout(startListening, 200);
      } else {
        updateMicButton(false);
      }
    };

    try {
      _recogniser.start();
    } catch(e) {
      _recogniser = null;
      _micEnabled = false;
      updateMicButton(false);
    }
  }

  function startConfirmationListener() {
    if (!_gameActive || _isSpeaking || !_pendingMove) return;
    if (!window.SpeechRecognition && !window.webkitSpeechRecognition) return;

    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    const confirmSR = new SR();
    confirmSR.lang = S().speechCode;
    confirmSR.continuous = false;
    confirmSR.interimResults = false;

    confirmSR.onresult = function(e) {
      const ans  = e.results[0][0].transcript.toLowerCase().trim();
      const yes  = S().voiceYes || ['yes'];
      const no   = S().voiceNo  || ['no'];
      const move = _pendingMove;

      if (yes.some(w => ans.includes(w))) {
        // Confirmed — execute the move
        _pendingMove = null;
        _voiceState  = 'LISTENING';
        _callbacks.executeMove(move.from, move.to, move.promotion || null);
        // Mic restarts via speak(move announcement).utt.onend after engine responds

      } else if (no.some(w => ans.includes(w))) {
        // Cancelled — go back to listening for a new move
        _pendingMove = null;
        _voiceState  = 'LISTENING';
        if (_micEnabled && _gameActive) setTimeout(startListening, 300);

      } else {
        // Heard something but it wasn't yes/no (e.g. tail of TTS, ambient noise)
        // Stay in CONFIRMING, try again — do NOT reset _pendingMove
        if (_micEnabled && _gameActive && _pendingMove) {
          setTimeout(startConfirmationListener, 300);
        }
      }
    };

    confirmSR.onerror = function() {
      if (_isSpeaking) return;
      if (_micEnabled && _gameActive) setTimeout(startConfirmationListener, 300);
    };

    confirmSR.onend = function() {
      if (_isSpeaking) return;
      // If still in confirmation state, restart confirmation listener
      if (_micEnabled && _gameActive && _voiceState === 'CONFIRMING') {
        setTimeout(startConfirmationListener, 200);
      }
    };

    try { confirmSR.start(); } catch(e) {}
  }

  function stopListening() {
    _micEnabled  = false;
    _voiceState  = 'IDLE';
    _pendingMove = null;
    if (_recogniser) {
      try { _recogniser.abort(); } catch(e) {}
      _recogniser = null;
    }
    updateMicButton(false);
  }

  function processVoiceInput(transcript) {
    // Only called from startListening.onresult — state is LISTENING here
    if (!_callbacks.getGame || !_callbacks.executeMove || !_gameActive) return;
    if (_voiceState !== 'LISTENING') return;  // ignore if in confirmation flow
    const board = _callbacks.getGame();
    if (!board) return;

    const move = parseSpokenMove(transcript, board);
    if (!move) {
      // Nothing understood — restart listening silently
      if (_micEnabled && _gameActive) setTimeout(startListening, 200);
      return;
    }

    // Valid move — ask for confirmation
    _pendingMove = move;
    _voiceState  = 'CONFIRMING';
    const moveDesc = move.san || (move.from + move.to);
    speak(S().ui.confirm(moveDesc), 'interrupt');
    // speak() sets _isSpeaking=true, stops mic
    // utt.onend fires after TTS → startConfirmationListener()
  }

  function handleSpokenMove(transcript) {
    processVoiceInput(transcript);
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
      e.stopPropagation();
      if (_micEnabled) {
        stopListening();  // stopListening sets _micEnabled=false and _voiceState=IDLE
      } else {
        _micEnabled = true;
        _voiceState = 'LISTENING';
        startListening();
      }
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
