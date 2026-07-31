/*
 * Continental Chat widget — extracted from 8 pages where it was previously
 * inlined (identically on 6 of them; analysis.html and leaderboard.html had
 * each drifted slightly — analysis.html was missing the ccUserLink() helper
 * entirely, so usernames there rendered as plain text instead of clickable
 * links; leaderboard.html reused the page's own userLink() instead of having
 * its own. This consolidated version restores full ccUserLink() functionality
 * to all pages, which is a real (small, positive) behavior change on those
 * two, not just a relocation.
 *
 * Self-contained: only depends on `sb` (Supabase client) and `currentUser`,
 * both of which every including page already defines. Deliberately not
 * included on play_multiplayer.html (no chat clutter during a live game) or
 * pgn_backfill.html (internal admin tool).
 *
 * Injects its own HTML and CSS into the page synchronously before running,
 * since ccFetch() runs immediately at the bottom of this script (so the
 * unread badge works even before the panel is ever opened) and needs
 * #cc-messages/#cc-badge to already exist in the DOM at that point.
 */

(function() {
  const CC_HTML = `<div id="cc-bubble" onclick="ccToggle()">
  <span id="cc-bubble-icon">💬</span>
  <span id="cc-badge" class="cc-badge"></span>
</div>

<div id="cc-panel">
  <div class="cc-header">
    <span>🌍 Continental Chat</span>
    <span class="cc-close" onclick="ccToggle()">✕</span>
  </div>
  <div id="cc-rules-bar" class="cc-rules-bar" onclick="ccShowRulesModal()">
    📌 Be respectful · Keep it chess-related · <span class="cc-rules-link">Full rules</span>
  </div>
  <div id="cc-messages" class="cc-messages"></div>
  <div id="cc-gate" class="cc-gate" style="display:none;"></div>
  <div id="cc-composer" class="cc-composer" style="display:none;">
    <input id="cc-input" name="chat-message" autocomplete="off" maxlength="50" placeholder="Say something…"
           oninput="ccUpdateCount()" onkeydown="if(event.key==='Enter')ccSend()">
    <button id="cc-send-btn" onclick="ccSend()">Send</button>
  </div>
  <div id="cc-meta" class="cc-meta"></div>
</div>

<div id="cc-rules-modal" class="cc-rules-modal-overlay" style="display:none;" onclick="if(event.target===this)ccDismissRulesModal()">
  <div class="cc-rules-modal-box">
    <div class="cc-rules-modal-title">🌍 Continental Chat — quick guide</div>
    <ul class="cc-rules-list">
      <li>Be respectful — no harassment, hate speech, or personal attacks.</li>
      <li>Keep it chess-related — no spam or unrelated promotion.</li>
      <li>English helps everyone here follow along.</li>
      <li>50 characters, 5 messages a day — keep it short.</li>
      <li>Breaking these rules can get your account banned.</li>
    </ul>
    <button class="cc-rules-modal-btn" onclick="ccDismissRulesModal()">Got it</button>
  </div>
</div>`;

  const CC_CSS = `#cc-bubble {
    position: fixed; right: 18px; bottom: 18px; width: 52px; height: 52px;
    border-radius: 50%; background: #262624; border: 1px solid #33312d;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; cursor: pointer; z-index: 400;
    box-shadow: 0 2px 10px rgba(0,0,0,0.5);
    transition: transform 0.15s ease;
  }
  #cc-bubble:hover { transform: scale(1.06); }
  .cc-badge {
    position: absolute; top: -4px; right: -4px; min-width: 18px; height: 18px;
    border-radius: 9px; background: var(--green, #4a9d7e); color: #000;
    font-size: 11px; font-weight: bold; display: none;
    align-items: center; justify-content: center; padding: 0 4px;
  }
  #cc-panel {
    position: fixed; right: 18px; bottom: 82px; width: min(320px, calc(100vw - 36px));
    height: min(460px, calc(100vh - 160px)); background: #262624;
    border: 1px solid #33312d; border-radius: 12px; z-index: 401;
    display: none; flex-direction: column; overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,0.6);
  }
  #cc-panel.open { display: flex; }
  .cc-header {
    padding: 12px 14px; border-bottom: 1px solid #33312d;
    display: flex; align-items: center; justify-content: space-between;
    font-size: 13px; letter-spacing: 0.04em; color: #eee; flex-shrink: 0;
  }
  .cc-close { cursor: pointer; color: var(--muted, #888); font-size: 14px; }
  .cc-rules-bar {
    padding: 6px 14px; font-size: 10.5px; color: var(--muted, #888);
    background: #111; border-bottom: 1px solid #33312d; cursor: pointer;
    flex-shrink: 0; line-height: 1.4;
  }
  .cc-rules-bar:hover { color: #ccc; }
  .cc-rules-link { color: var(--green, #4a9d7e); text-decoration: underline; }
  .cc-messages {
    flex: 1; overflow-y: auto; padding: 10px 12px; display: flex;
    flex-direction: column; gap: 8px;
  }
  .cc-msg { font-size: 12.5px; line-height: 1.4; }
  .cc-msg-head { display: flex; align-items: baseline; gap: 5px; margin-bottom: 1px; }
  .cc-msg-name { font-weight: bold; color: #ddd; }
  .cc-msg-admin {
    font-size: 9px; font-weight: bold; color: #000; background: var(--green, #4a9d7e);
    padding: 1px 5px; border-radius: 4px; letter-spacing: 0.05em;
  }
  .cc-msg-time { font-size: 10px; color: var(--muted, #666); margin-left: auto; }
  .cc-msg-text { color: #ccc; word-break: break-word; }
  .cc-gate {
    padding: 12px 14px; border-top: 1px solid #33312d; font-size: 12px;
    color: var(--muted, #888); text-align: center; flex-shrink: 0;
  }
  .cc-gate a { color: var(--green, #4a9d7e); cursor: pointer; text-decoration: underline; }
  .cc-composer {
    padding: 10px 12px; border-top: 1px solid #33312d; display: flex; gap: 8px; flex-shrink: 0;
  }
  .cc-composer input {
    flex: 1; background: #33312d; border: 1px solid #33312d; border-radius: 6px;
    color: #eee; padding: 8px 10px; font-size: 12.5px; outline: none;
  }
  .cc-composer button {
    background: var(--green, #4a9d7e); color: #000; border: none; border-radius: 6px;
    padding: 8px 14px; font-size: 12px; font-weight: bold; cursor: pointer;
  }
  .cc-meta {
    padding: 3px 14px 8px; font-size: 10px; color: var(--muted, #555);
    text-align: right; flex-shrink: 0;
  }
  #cc-rules-modal.open, .cc-rules-modal-overlay[style*="flex"] {}
  .cc-rules-modal-overlay {
    position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 500;
    align-items: center; justify-content: center; padding: 24px;
  }
  .cc-rules-modal-box {
    background: #262624; border: 1px solid #33312d; border-radius: 12px;
    padding: 20px; max-width: 340px; width: 100%;
    box-shadow: 0 8px 32px rgba(0,0,0,0.7);
  }
  .cc-rules-modal-title {
    font-size: 14px; color: #eee; font-weight: bold; margin-bottom: 12px;
  }
  .cc-rules-list {
    margin: 0 0 16px; padding-left: 18px; color: #ccc; font-size: 12.5px; line-height: 1.6;
  }
  .cc-rules-list li { margin-bottom: 6px; }
  .cc-rules-modal-btn {
    width: 100%; background: var(--green, #4a9d7e); color: #000; border: none;
    border-radius: 6px; padding: 10px; font-size: 13px; font-weight: bold; cursor: pointer;
  }
  @media (max-width: 480px) {
    #cc-panel { right: 10px; bottom: 76px; }
    #cc-bubble { right: 12px; bottom: 12px; }
  }`;

  const styleEl = document.createElement('style');
  styleEl.textContent = CC_CSS;
  document.head.appendChild(styleEl);

  document.body.insertAdjacentHTML('beforeend', CC_HTML);

  const CC_MAX_CHARS = 50;
  let ccOpen = false;
  let ccPollTimer = null;
  let ccLastSeenAt = localStorage.getItem('cc.lastSeenAt') || '';
  let ccLatestMessages = [];

  const CC_FLAG = code => code ? ({
    DZ:'🇩🇿',AO:'🇦🇴',BJ:'🇧🇯',BW:'🇧🇼',BF:'🇧🇫',BI:'🇧🇮',CM:'🇨🇲',CV:'🇨🇻',CF:'🇨🇫',
    TD:'🇹🇩',KM:'🇰🇲',CG:'🇨🇬',CD:'🇨🇩',DJ:'🇩🇯',EG:'🇪🇬',GQ:'🇬🇶',ER:'🇪🇷',SZ:'🇸🇿',
    ET:'🇪🇹',GA:'🇬🇦',GM:'🇬🇲',GH:'🇬🇭',GN:'🇬🇳',GW:'🇬🇼',CI:'🇨🇮',KE:'🇰🇪',LS:'🇱🇸',
    LR:'🇱🇷',LY:'🇱🇾',MG:'🇲🇬',MW:'🇲🇼',ML:'🇲🇱',MR:'🇲🇷',MU:'🇲🇺',MA:'🇲🇦',MZ:'🇲🇿',
    NA:'🇳🇦',NE:'🇳🇪',NG:'🇳🇬',RW:'🇷🇼',ST:'🇸🇹',SN:'🇸🇳',SC:'🇸🇨',SL:'🇸🇱',SO:'🇸🇴',
    ZA:'🇿🇦',SS:'🇸🇸',SD:'🇸🇩',TZ:'🇹🇿',TG:'🇹🇬',TN:'🇹🇳',UG:'🇺🇬',ZM:'🇿🇲',ZW:'🇿🇼'
  }[code] || '🌍') : '🌍';

  function ccTimeAgo(iso) {
    const diff = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
    if (diff < 60) return 'now';
    if (diff < 3600) return Math.floor(diff / 60) + 'm';
    if (diff < 86400) return Math.floor(diff / 3600) + 'h';
    return Math.floor(diff / 86400) + 'd';
  }

  function ccEscape(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  function ccUserLink(username) {
    if (!username) return ccEscape('Anonymous');
    return '<a href="/history?u=' + encodeURIComponent(username) + '" style="color:inherit;text-decoration:none;" onclick="event.stopPropagation()">' + ccEscape(username) + '</a>';
  }

  window.ccShowRulesModal = function() {
    document.getElementById('cc-rules-modal').style.display = 'flex';
  };
  window.ccDismissRulesModal = function() {
    document.getElementById('cc-rules-modal').style.display = 'none';
    localStorage.setItem('cc.rulesSeen', '1');
  };

  window.ccToggle = function() {
    ccOpen = !ccOpen;
    document.getElementById('cc-panel').classList.toggle('open', ccOpen);
    if (ccOpen) {
      // First-ever open on this browser — show the fuller guide once,
      // automatically. The short pinned bar stays visible on every
      // subsequent open as an ongoing, low-friction reminder; tapping it
      // re-opens this same full guide any time.
      if (!localStorage.getItem('cc.rulesSeen')) {
        ccShowRulesModal();
      }
      ccFetch();
      if (ccPollTimer) clearInterval(ccPollTimer);
      ccPollTimer = setInterval(ccFetch, 4000);
      if (ccLatestMessages.length) {
        ccLastSeenAt = ccLatestMessages[ccLatestMessages.length - 1].created_at;
        localStorage.setItem('cc.lastSeenAt', ccLastSeenAt);
        ccUpdateBadge();
      }
    } else {
      if (ccPollTimer) clearInterval(ccPollTimer);
      ccPollTimer = setInterval(ccFetch, 25000);
    }
    const loggedIn = typeof currentUser !== 'undefined' && currentUser;
    document.getElementById('cc-composer').style.display = loggedIn ? 'flex' : 'none';
    document.getElementById('cc-gate').style.display = loggedIn ? 'none' : 'block';
    if (!loggedIn) {
      document.getElementById('cc-gate').innerHTML = typeof openAuth === 'function'
        ? '<a onclick="openAuth()">Sign in</a> to join the conversation'
        : 'Sign in to join the conversation';
    }
    ccUpdateCount();
  };

  function ccUpdateBadge() {
    const unread = ccLatestMessages.filter(m => !ccLastSeenAt || m.created_at > ccLastSeenAt).length;
    const badge = document.getElementById('cc-badge');
    if (unread > 0 && !ccOpen) {
      badge.textContent = unread > 9 ? '9+' : unread;
      badge.style.display = 'flex';
    } else {
      badge.style.display = 'none';
    }
  }

  async function ccFetch() {
    try {
      const r = await fetch('/api/chat/messages');
      const data = await r.json();
      ccLatestMessages = data.messages || [];
      ccRender();
      ccUpdateBadge();
    } catch (e) { /* transient — next poll will retry */ }
  }

  function ccRender() {
    const box = document.getElementById('cc-messages');
    const wasAtBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 20;
    box.innerHTML = ccLatestMessages.map(m => `
      <div class="cc-msg">
        <div class="cc-msg-head">
          <span class="cc-msg-name">${ccUserLink(m.username)}</span>
          ${m.is_admin ? '<span class="cc-msg-admin">ADMIN</span>' : `<span>${CC_FLAG(m.country)}</span>`}
          <span class="cc-msg-time">${ccTimeAgo(m.created_at)}</span>
        </div>
        <div class="cc-msg-text">${ccEscape(m.message)}</div>
      </div>
    `).join('') || '<div style="color:var(--muted,#666);font-size:12px;text-align:center;padding:20px 0;">No messages yet — say hello 👋</div>';
    if (wasAtBottom || ccOpen) box.scrollTop = box.scrollHeight;
  }

  window.ccUpdateCount = function() {
    const input = document.getElementById('cc-input');
    const meta = document.getElementById('cc-meta');
    const n = input.value.length;
    meta.textContent = `${n}/${CC_MAX_CHARS}`;
    meta.style.color = n > CC_MAX_CHARS ? '#ff5555' : 'var(--muted,#555)';
  };

  window.ccSend = async function() {
    const input = document.getElementById('cc-input');
    const msg = input.value.trim();
    if (!msg) return;
    if (typeof sb === 'undefined' || !sb) { ccShowGateMessage('Chat needs a session — please reload the page.'); return; }
    const { data: { session } } = await sb.auth.getSession();
    if (!session) { if (typeof openAuth === 'function') openAuth(); return; }

    const btn = document.getElementById('cc-send-btn');
    btn.disabled = true;
    try {
      const r = await fetch('/api/chat/send', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${session.access_token}` },
        body: JSON.stringify({ message: msg }),
      });
      const data = await r.json().catch(() => ({}));
      if (r.ok) {
        input.value = '';
        ccUpdateCount();
        ccFetch();
      } else {
        ccShowGateMessage(data.detail || 'Message could not be sent.');
      }
    } catch (e) {
      ccShowGateMessage('Network error — try again.');
    } finally {
      btn.disabled = false;
    }
  };

  let ccGateTimer = null;
  function ccShowGateMessage(text) {
    const gate = document.getElementById('cc-gate');
    gate.textContent = text;
    gate.style.display = 'block';
    if (ccGateTimer) clearTimeout(ccGateTimer);
    ccGateTimer = setTimeout(() => { gate.style.display = 'none'; }, 4000);
  }

  // Light background poll from page load, so the badge can show unread
  // activity even before the user ever opens the panel.
  ccFetch();
  ccPollTimer = setInterval(ccFetch, 25000);
})();
