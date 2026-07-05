// Bianque front-end. Vanilla JS, fetch-based SSE streaming.
//
// Two layers:
//  - Native chrome: settings, composer, clarify box, context meter, reset.
//    These are app shell, not agent output, so they stay hand-written.
//  - A2UI renderer: the transcript region renders an A2UI v0.9 envelope stream
//    (createSurface / updateComponents / updateDataModel / deleteSurface) emitted
//    by the backend A2UIAdapter. Agent output (reasoning / tools / sub-agent /
//    answer) is described as a component tree there.
//
// The SSE stream also carries a few native passthroughs: `context` (meter),
// `clarification` (ask_user box), and terminal signals (`answer`/`error`/
// `cancelled`) which close the stream — their visual block is already an A2UI
// component, so they need no native rendering.

const $ = (id) => document.getElementById(id);
const LS_KEY = "bianque.settings.v1";
const THEME_KEY = "bianque.theme";
const THEMES = { night: "昼", paper: "夜" }; // 按钮显示目标主题: 夜诊中显示“昼”

const state = {
  cfg: null,          // {apikey}
  server: null,       // {repo_root, provider, base_url, model, context_window}
  sessionId: null,
  streaming: false,
  abortCtrl: null,
  pendingClarify: null,
  userPinned: true,   // auto-scroll only while the user is parked at the bottom
  theme: "night",
};

// ---------- A2UI renderer ----------

const surfaces = new Map(); // surfaceId -> {mount, components, model, bound}

function createSurface(payload) {
  // One surface per turn; its mount sits in #transcript after the user bubble,
  // so agent output and user bubbles interleave chronologically.
  const mount = document.createElement("div");
  mount.className = "a2ui-surface";
  mount.dataset.surfaceId = payload.surfaceId;
  $("transcript").appendChild(mount);
  surfaces.set(payload.surfaceId, {
    mount, components: new Map(), model: {}, bound: new Map(),
  });
}

function updateComponents(payload) {
  const surf = surfaces.get(payload.surfaceId);
  if (!surf) return;
  for (const c of payload.components) surf.components.set(c.id, c);
  if (surf.components.has("root")) rebuild(surf);
}

function updateDataModel(payload) {
  const surf = surfaces.get(payload.surfaceId);
  if (!surf) return;
  const path = payload.path || "/";
  setPointer(surf.model, path, payload.value);
  // In-place update of every binding affected by this path (exact, parent, or
  // child match) — avoids rebuilding the tree on every streamed token.
  for (const [bpath, applies] of surf.bound) {
    if (bpath === path || bpath.startsWith(path + "/") || path.startsWith(bpath + "/")) {
      for (const fn of applies) fn();
    }
  }
}

function deleteSurface(payload) {
  const surf = surfaces.get(payload.surfaceId);
  if (surf) { surf.mount.remove(); surfaces.delete(payload.surfaceId); }
}

function rebuild(surf) {
  surf.bound = new Map();
  surf.mount.innerHTML = "";
  const rootEl = renderComponent(surf.components.get("root"), surf);
  if (rootEl) surf.mount.appendChild(rootEl);
}

function renderComponent(def, surf) {
  if (!def) return null;
  switch (def.component) {
    case "Column": return renderContainer(def, surf, "a2ui-column");
    case "Row": return renderContainer(def, surf, "a2ui-row");
    case "Card": return renderCard(def, surf);
    case "Text": return renderText(def, surf);
    case "Tabs": return renderTabs(def, surf);
    case "Divider": { const d = document.createElement("div"); d.className = "a2ui-divider"; return d; }
    case "Chips": return renderChips(def, surf);
    default:
      return renderUnknown(def);
  }
}

function renderContainer(def, surf, cls) {
  const el = document.createElement("div");
  el.className = cls + toneClass(def.tone);
  for (const cid of (def.children || [])) {
    const child = renderComponent(surf.components.get(cid), surf);
    if (child) el.appendChild(child);
  }
  return el;
}

function renderCard(def, surf) {
  const card = document.createElement("div");
  card.className = "a2ui-card" + toneClass(def.tone);
  const head = document.createElement("div");
  head.className = "a2ui-card-head";
  if (def.icon) {
    const ic = document.createElement("span");
    ic.className = "a2ui-icon";
    ic.textContent = def.icon;
    head.appendChild(ic);
  }
  const titleEl = document.createElement("span");
  titleEl.className = "a2ui-card-title";
  const subEl = document.createElement("span");
  subEl.className = "a2ui-card-sub";
  const badge = document.createElement("span");
  badge.className = "a2ui-badge";
  head.append(titleEl, subEl, badge);

  const applyTitle = () => { titleEl.textContent = resolveDynamic(def.title, surf.model); };
  const applySub = () => { subEl.textContent = resolveDynamic(def.subtitle, surf.model); };
  const applyStatus = () => updateBadge(badge, resolveDynamic(def.status, surf.model));
  applyTitle(); applySub(); applyStatus();
  bindIf(surf, def.title, applyTitle);
  bindIf(surf, def.subtitle, applySub);
  bindIf(surf, def.status, applyStatus);

  const body = document.createElement("div");
  body.className = "a2ui-card-body";
  if (def.child) {
    const childEl = renderComponent(surf.components.get(def.child), surf);
    if (childEl) body.appendChild(childEl);
  }
  if (def.collapsible) {
    // Collapsed by default: the header (tool / args / status) stays visible as a
    // single compact line; the (potentially large) body is hidden until clicked.
    // Affordances: a rotating chevron + an 展开/收起 hint + hover highlight.
    const chev = document.createElement("span");
    chev.className = "a2ui-chevron";
    chev.textContent = "▾";
    head.insertBefore(chev, head.firstChild);
    const hint = document.createElement("span");
    hint.className = "a2ui-collapse-hint";
    head.appendChild(hint);
    head.classList.add("clickable");
    head.title = "点击展开 / 收起";
    // The surface rebuilds from scratch on every updateComponents (e.g. each new
    // explore segment / substep while streaming), which would otherwise reset
    // this card to its default state. Remember the user's choice on the
    // persistent `surf`, keyed by the (stable) card component id — same trick
    // the Tabs component uses for its active tab.
    if (!surf.cardCollapsed) surf.cardCollapsed = new Map();
    const initial = surf.cardCollapsed.has(def.id)
      ? surf.cardCollapsed.get(def.id)
      : def.collapsed !== false; // collapsible ⇒ collapsed by default
    const setCollapsed = (c) => {
      body.classList.toggle("collapsed", c);
      head.classList.toggle("is-collapsed", c);
      hint.textContent = c ? "展开" : "收起";
      surf.cardCollapsed.set(def.id, c);
    };
    setCollapsed(initial);
    head.addEventListener("click", () => setCollapsed(!body.classList.contains("collapsed")));
  }
  card.append(head, body);
  return card;
}

function updateBadge(badge, status) {
  badge.textContent = badgeLabel(status);
  badge.className = "a2ui-badge" + (status ? " status-" + status : " hidden");
}
function badgeLabel(s) {
  return ({ running: "运行中", exploring: "探索中", done: "完成", failed: "失败" })[s] || s || "";
}

function renderText(def, surf) {
  const variant = def.variant || "body";
  const el = document.createElement("div");
  el.className = "a2ui-text variant-" + variant + toneClass(def.tone);
  const apply = () => applyText(el, resolveDynamic(def.text, surf.model), variant);
  apply();
  bindIf(surf, def.text, apply);
  return el;
}
function applyText(el, text, variant) {
  if (variant === "body") el.innerHTML = renderMarkdown(text || "");
  else el.textContent = text || ""; // reasoning / result / caption: plain (CSS handles mono/pre-wrap)
}

function renderTabs(def, surf) {
  const wrap = document.createElement("div");
  wrap.className = "a2ui-tabs";
  const bar = document.createElement("div");
  bar.className = "a2ui-tabbar";
  const panels = document.createElement("div");
  panels.className = "a2ui-tabpanels";
  // The surface is rebuilt from scratch on every updateComponents (e.g. each new
  // explore sub-step), which would otherwise reset the tabs to the default. Keep
  // the user's chosen tab on the persistent `surf`, keyed by the (stable) Tabs
  // component id, and restore it on re-render.
  if (!surf.activeTabs) surf.activeTabs = new Map();
  const tabs = def.tabs || [];
  let active = def.id && surf.activeTabs.has(def.id) ? surf.activeTabs.get(def.id) : 0;
  if (active < 0 || active >= tabs.length) active = 0;
  tabs.forEach((tab, i) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "a2ui-tab" + (i === active ? " active" : "");
    btn.textContent = tab.title || "";
    const panel = document.createElement("div");
    panel.className = "a2ui-tabpanel" + (i === active ? "" : " hidden");
    if (tab.child) {
      const c = renderComponent(surf.components.get(tab.child), surf);
      if (c) panel.appendChild(c);
    }
    btn.addEventListener("click", () => {
      bar.querySelectorAll(".a2ui-tab").forEach((b) => b.classList.remove("active"));
      panels.querySelectorAll(".a2ui-tabpanel").forEach((p) => p.classList.add("hidden"));
      btn.classList.add("active");
      panel.classList.remove("hidden");
      if (def.id) surf.activeTabs.set(def.id, i);
    });
    bar.appendChild(btn);
    panels.appendChild(panel);
  });
  wrap.append(bar, panels);
  return wrap;
}

function renderChips(def, surf) {
  const wrap = document.createElement("div");
  wrap.className = "a2ui-chips";
  const renderItems = (list) => {
    wrap.innerHTML = "";
    if (def.label && list.length) {
      const lab = document.createElement("span");
      lab.className = "a2ui-chips-label";
      lab.textContent = def.label;
      wrap.appendChild(lab);
    }
    for (const it of list) {
      const c = document.createElement("span");
      c.className = "a2ui-chip";
      c.textContent = it;
      wrap.appendChild(c);
    }
  };
  const items = Array.isArray(def.items) ? def.items.slice() : [];
  renderItems(items);
  // items may be a bound list ({path}); support it for completeness.
  if (def.items && typeof def.items === "object" && def.items.path) {
    const apply = () => renderItems(asStringList(resolvePointer(surf.model, def.items.path)));
    bindIf(surf, def.items, apply);
  }
  return wrap;
}
function asStringList(v) { return Array.isArray(v) ? v.map(String) : []; }

function renderUnknown(def) {
  const el = document.createElement("div");
  el.className = "a2ui-unknown";
  el.textContent = "⚠ 未知组件: " + (def.component || "?");
  return el;
}

// ---------- data binding helpers ----------

function resolveDynamic(spec, model) {
  if (spec == null) return "";
  if (typeof spec === "string") return spec;
  if (typeof spec === "object") {
    if ("path" in spec) { const v = resolvePointer(model, spec.path); return v == null ? "" : String(v); }
    if ("call" in spec) return ""; // client-side functions unsupported; adapter pre-formats
  }
  return String(spec);
}
function isBound(spec) { return spec != null && typeof spec === "object" && "path" in spec; }
function bindIf(surf, spec, apply) { if (isBound(spec)) addBound(surf, spec.path, apply); }
function addBound(surf, path, apply) {
  if (!surf.bound.has(path)) surf.bound.set(path, []);
  surf.bound.get(path).push(apply);
}
function resolvePointer(model, path) {
  if (!path) return undefined;
  let cur = model;
  for (const seg of path.split("/").filter((p) => p !== "")) {
    if (cur == null) return undefined;
    cur = cur[seg];
  }
  return cur;
}
function setPointer(model, path, value) {
  if (!path || path === "/") {
    for (const k of Object.keys(model)) delete model[k];
    Object.assign(model, value || {});
    return;
  }
  const segs = path.split("/").filter((p) => p !== "");
  let cur = model;
  for (let i = 0; i < segs.length - 1; i++) {
    const s = segs[i];
    if (cur[s] == null || typeof cur[s] !== "object") cur[s] = {};
    cur = cur[s];
  }
  const last = segs[segs.length - 1];
  if (value === undefined) delete cur[last];
  else cur[last] = value;
}
function toneClass(tone) { return tone ? " tone-" + tone : ""; }

// ---------- safe markdown (escape-first, zero-depend subset) ----------

function renderMarkdown(text) {
  let s = escapeHtml(text);
  // fenced code blocks (extract first so their content is never transformed)
  const blocks = [];
  s = s.replace(/```(\w*)\n?([\s\S]*?)```/g, (_m, lang, code) => {
    const i = blocks.length;
    blocks.push('<pre class="md-code"><code class="lang-' + (lang || "text") + '">' + code.replace(/\n$/, "") + "</code></pre>");
    return " B" + i + " ";
  });
  // inline code (extract before emphasis transforms)
  const inl = [];
  s = s.replace(/`([^`\n]+)`/g, (_m, c) => {
    const i = inl.length;
    inl.push('<code class="md-inline">' + c + "</code>");
    return "I" + i + "";
  });
  // headings
  s = s.replace(/^######\s?(.*)$/gm, "<h6>$1</h6>")
       .replace(/^#####\s?(.*)$/gm, "<h5>$1</h5>")
       .replace(/^####\s?(.*)$/gm, "<h4>$1</h4>")
       .replace(/^###\s?(.*)$/gm, "<h3>$1</h3>")
       .replace(/^##\s?(.*)$/gm, "<h2>$1</h2>")
       .replace(/^#\s?(.*)$/gm, "<h1>$1</h1>");
  // blockquote (escaped '&gt;'), hr, bold, italic, links
  s = s.replace(/^&gt;\s?(.*)$/gm, "<blockquote>$1</blockquote>")
       .replace(/^---+\s*$/gm, "<hr>")
       .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
       .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
       .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  // lists
  s = renderLists(s);
  // restore inline code, then turn remaining newlines into <br>
  s = s.replace(/I(\d+)/g, (_m, i) => inl[i]);
  s = s.replace(/\n/g, "<br>");
  // restore fenced blocks last (their internal newlines are untouched)
  s = s.replace(/ B(\d+) /g, (_m, i) => blocks[i]);
  return s;
}

function renderLists(s) {
  const lines = s.split("\n");
  const out = [];
  let kind = null, items = [];
  const flush = () => {
    if (kind) { out.push("<" + kind + ">" + items.map((i) => "<li>" + i + "</li>").join("") + "</" + kind + ">"); kind = null; items = []; }
  };
  for (const line of lines) {
    const ul = line.match(/^\s{0,3}[-*]\s+(.*)$/);
    const ol = line.match(/^\s{0,3}\d+\.\s+(.*)$/);
    if (ul) { if (kind !== "ul") { flush(); kind = "ul"; } items.push(ul[1]); }
    else if (ol) { if (kind !== "ol") { flush(); kind = "ol"; } items.push(ol[1]); }
    else { flush(); out.push(line); }
  }
  flush();
  return out.join("\n");
}

// ---------- native chrome: settings ----------

function loadSettings() { try { return JSON.parse(localStorage.getItem(LS_KEY)); } catch { return null; } }
function saveSettings(cfg) { localStorage.setItem(LS_KEY, JSON.stringify(cfg)); state.cfg = cfg; }
function loadTheme() {
  try { return localStorage.getItem(THEME_KEY) || "night"; } catch { return "night"; }
}
function applyTheme(theme) {
  if (!THEMES[theme]) theme = "night";
  state.theme = theme;
  document.documentElement.classList.toggle("theme-night", theme === "night");
  document.documentElement.classList.toggle("theme-paper", theme === "paper");
  const btn = $("themeBtn");
  if (btn) {
    btn.textContent = THEMES[theme];
    btn.title = theme === "night" ? "切换到宣纸亮色" : "切换到夜诊暖暗";
    btn.setAttribute("aria-label", btn.title);
  }
  try { localStorage.setItem(THEME_KEY, theme); } catch { /* ignore */ }
}
function openSettings() { $("cfgApikey").value = (state.cfg && state.cfg.apikey) || ""; $("settings").classList.remove("hidden"); }
function closeSettings() { $("settings").classList.add("hidden"); }

$("themeBtn").addEventListener("click", () => applyTheme(state.theme === "night" ? "paper" : "night"));
$("settingsBtn").addEventListener("click", openSettings);
$("closeSettingsBtn").addEventListener("click", closeSettings);
$("saveSettingsBtn").addEventListener("click", () => {
  const apikey = $("cfgApikey").value.trim();
  if (!apikey) { alert("请填写 API Key"); return; }
  saveSettings({ apikey });
  closeSettings();
});
$("clearSettingsBtn").addEventListener("click", () => {
  if (confirm("清除本地保存的 API Key?")) { localStorage.removeItem(LS_KEY); state.cfg = null; openSettings(); }
});

// ---------- native chrome: context meter ----------

async function loadServerConfig() {
  try {
    const r = await fetch("/api/config");
    if (r.ok) state.server = await r.json();
  } catch { /* server unreachable */ }
  initContextMeter();
}
function initContextMeter() {
  const w = (state.server && state.server.context_window) || 0;
  $("ctxMeter").classList.remove("high");
  $("ctxText").textContent = w ? `0 / ${fmt(w)}` : "—";
}
function updateContext(used, window_) {
  const w = window_ || (state.server && state.server.context_window) || 0;
  const ratio = w ? used / w : 0;
  $("ctxMeter").classList.toggle("high", ratio >= 0.8);
  $("ctxText").textContent = w ? `${fmt(used)} / ${fmt(w)} · ${Math.round(ratio * 100)}%` : fmt(used);
}
function fmt(n) { return n >= 1000 ? (n / 1000).toFixed(n % 1000 ? 1 : 0) + "k" : String(n); }

// ---------- native chrome: status, scroll, clarify ----------

function setStatus(text, cls) { const s = $("status"); s.textContent = text; s.className = "status " + (cls || "idle"); }

function nearBottom() { const t = $("transcript"); return t.scrollHeight - t.scrollTop - t.clientHeight < 80; }
function scrollToBottom() { const t = $("transcript"); if (state.userPinned) t.scrollTop = t.scrollHeight; }
$("transcript").addEventListener("scroll", () => { state.userPinned = nearBottom(); });

function renderClarification(ev) {
  const box = $("clarifyBox");
  $("clarifyQ").textContent = "问  " + ev.question;
  $("clarifyInput").value = "";
  box.classList.remove("hidden");
  $("clarifyInput").focus();
  state.pendingClarify = { call_id: ev.call_id };
}

// ---------- event dispatch ----------

function handleEvent(ev) {
  if (ev.createSurface) createSurface(ev.createSurface);
  else if (ev.updateComponents) updateComponents(ev.updateComponents);
  else if (ev.updateDataModel) updateDataModel(ev.updateDataModel);
  else if (ev.deleteSurface) deleteSurface(ev.deleteSurface);
  else if (ev.type === "context") updateContext(ev.used, ev.window);
  else if (ev.type === "clarification") renderClarification(ev);
  // answer/error/cancelled: terminal signals only (A2UI already drew the block)
  scrollToBottom();
}

// ---------- streaming ----------

async function ensureSession() {
  if (state.sessionId) return state.sessionId;
  let r;
  try {
    r = await fetch("/api/sessions", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ apikey: state.cfg.apikey }),
    });
  } catch (e) {
    throw new Error("网络错误，无法连接服务端");
  }
  if (!r.ok) {
    let code = null, message = null;
    try {
      const j = await r.json();
      const detail = j && j.detail;
      if (detail && typeof detail === "object") { code = detail.code; message = detail.message; }
      else if (typeof detail === "string") { message = detail; }
    } catch { /* 非 JSON 错误体，忽略 */ }
    if (code === "session_limit" || r.status === 503) {
      // 医馆满员：弹主题告示，不当作普通错误条。
      showNotice(message || "现在医馆内人员过多，请稍后再来。");
      const err = new Error("session_limit"); err.code = "session_limit"; throw err;
    }
    throw new Error(`创建会话失败 (${r.status})${message ? ": " + message : ""}`);
  }
  const data = await r.json();
  state.sessionId = data.session_id;
  return state.sessionId;
}

function showNotice(message) {
  $("noticeBody").textContent = message || "";
  $("notice").classList.remove("hidden");
}
function hideNotice() { $("notice").classList.add("hidden"); }

async function sendMessage() {
  if (!state.cfg || !state.cfg.apikey) { openSettings(); return; }
  const question = $("question").value.trim();
  if (!question || state.streaming) return;

  // 先确保会话可用。若医馆满员 ensureSession 会弹告示并抛带 code 的错；
  // 此时提问文本保留(未清空)，便于用户点"再候片刻"重试。
  let sid;
  try {
    sid = await ensureSession();
  } catch (e) {
    if (e && e.code === "session_limit") return;             // 告示已显示
    const b = document.createElement("div"); b.className = "block error";
    b.textContent = "误  " + e.message;
    $("transcript").appendChild(b);
    scrollToBottom();
    return;
  }

  const uq = document.createElement("div");
  uq.className = "block user"; uq.textContent = question;
  $("transcript").appendChild(uq);
  $("question").value = "";
  state.userPinned = true;
  toggleComposer(true);
  setStatus("思考中…", "busy");

  try {
    const r = await fetch(`/api/sessions/${sid}/message`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!r.ok) throw new Error(`发起分析失败 (${r.status}): ${await r.text()}`);
    const { stream_url } = await r.json();
    await streamEvents(stream_url);
  } catch (e) {
    const b = document.createElement("div"); b.className = "block error";
    b.textContent = "误  " + e.message;
    $("transcript").appendChild(b);
  } finally {
    toggleComposer(false);
    setStatus("空闲", "idle");
  }
}

async function streamEvents(streamUrl) {
  state.streaming = true;
  state.abortCtrl = new AbortController();
  const resp = await fetch(streamUrl, { signal: state.abortCtrl.signal });
  if (!resp.ok) throw new Error(`流式连接失败 (${resp.status})`);
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  const TERMINAL = ["answer", "cancelled", "error"];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, idx); buf = buf.slice(idx + 2);
      const line = chunk.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      const data = line.slice(5).trim();
      if (!data) continue;
      let ev;
      try { ev = JSON.parse(data); } catch { continue; }
      handleEvent(ev);
      if (ev.type && TERMINAL.includes(ev.type)) { state.streaming = false; return; }
    }
  }
  state.streaming = false;
}

function toggleComposer(busy) {
  $("sendBtn").disabled = busy;
  $("stopBtn").classList.toggle("hidden", !busy);
}

$("composer").addEventListener("submit", (e) => { e.preventDefault(); sendMessage(); });
$("stopBtn").addEventListener("click", async () => {
  if (state.sessionId) await fetch(`/api/sessions/${state.sessionId}/cancel`, { method: "POST" });
  if (state.abortCtrl) state.abortCtrl.abort();
});
$("clarifySendBtn").addEventListener("click", async () => {
  if (!state.pendingClarify) return;
  const text = $("clarifyInput").value.trim();
  if (!text) return;
  const ub = document.createElement("div"); ub.className = "block user"; ub.textContent = text;
  $("transcript").appendChild(ub);
  await fetch(`/api/sessions/${state.sessionId}/answer`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ call_id: state.pendingClarify.call_id, text }),
  });
  state.pendingClarify = null;
  $("clarifyBox").classList.add("hidden");
});
$("resetBtn").addEventListener("click", () => {
  state.sessionId = null;
  surfaces.clear();
  $("transcript").innerHTML = "";
  $("clarifyBox").classList.add("hidden");
  initContextMeter();
});
$("noticeClose").addEventListener("click", hideNotice);
$("noticeRetry").addEventListener("click", () => { hideNotice(); sendMessage(); });

// ---------- utils ----------

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---------- init ----------

applyTheme(loadTheme());
state.cfg = loadSettings();
loadServerConfig();
if (!state.cfg || !state.cfg.apikey) openSettings();
