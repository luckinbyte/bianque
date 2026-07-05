// Bianque front-end. Vanilla JS, fetch-based SSE streaming.
// The deployment owns repo_root/provider/model; the browser only stores the
// user's own API key (in localStorage) and sends it in the session body.

const $ = (id) => document.getElementById(id);
const LS_KEY = "bianque.settings.v1";

const state = {
  cfg: null,          // {apikey}
  server: null,       // {repo_root, provider, base_url, model, context_window} from /api/config
  sessionId: null,
  streaming: false,
  abortCtrl: null,
  pendingClarify: null,
  reasoningEl: null,
};

// ---------- settings ----------

function loadSettings() {
  try { return JSON.parse(localStorage.getItem(LS_KEY)); } catch { return null; }
}
function saveSettings(cfg) {
  localStorage.setItem(LS_KEY, JSON.stringify(cfg));
  state.cfg = cfg;
}
function openSettings() {
  $("cfgApikey").value = (state.cfg && state.cfg.apikey) || "";
  $("settings").classList.remove("hidden");
}
function closeSettings() { $("settings").classList.add("hidden"); }

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

// ---------- server config (read-only, shared) ----------

async function loadServerConfig() {
  try {
    const r = await fetch("/api/config");
    if (r.ok) state.server = await r.json();
  } catch { /* server unreachable; placeholders will show */ }
  initContextMeter();
}

function initContextMeter() {
  const w = (state.server && state.server.context_window) || 0;
  const bar = $("ctxProgress");
  bar.max = w || 100; bar.value = 0; bar.classList.remove("high");
  $("ctxText").textContent = w ? `0 / ${fmt(w)}` : "—";
}

function updateContext(used, window_) {
  const w = window_ || (state.server && state.server.context_window) || 0;
  const bar = $("ctxProgress");
  if (w) { bar.max = w; bar.value = Math.min(used, w); }
  const ratio = w ? used / w : 0;
  bar.classList.toggle("high", ratio >= 0.8);
  $("ctxText").textContent = w ? `${fmt(used)} / ${fmt(w)} · ${Math.round(ratio * 100)}%` : fmt(used);
}

function fmt(n) {
  if (n >= 1000) return (n / 1000).toFixed(n % 1000 ? 1 : 0) + "k";
  return String(n);
}

// ---------- rendering ----------

function setStatus(text, cls) {
  const s = $("status");
  s.textContent = text;
  s.className = "status " + (cls || "idle");
}

function addBlock(kind) {
  const div = document.createElement("div");
  div.className = "block " + kind;
  $("transcript").appendChild(div);
  $("transcript").scrollTop = $("transcript").scrollHeight;
  return div;
}

function ensureReasoning() {
  if (!state.reasoningEl) {
    state.reasoningEl = addBlock("reasoning");
    state.reasoningEl.innerHTML = '<div class="blk-label">思考</div><div class="blk-body"></div>';
  }
  return state.reasoningEl.querySelector(".blk-body");
}

function appendReasoning(text) { ensureReasoning().appendChild(document.createTextNode(text)); }
function finalizeReasoning() { state.reasoningEl = null; }

function renderToolCall(ev) {
  finalizeReasoning();
  const wrap = addBlock("tool");
  const head = document.createElement("div");
  head.className = "blk-label";
  const args = typeof ev.args === "string" ? ev.args : JSON.stringify(ev.args);
  head.innerHTML = `🔧 <code>${ev.tool}</code> <span class="args">${escapeHtml(args)}</span>`;
  const body = document.createElement("pre");
  body.className = "blk-body collapsed";
  wrap.appendChild(head);
  wrap.appendChild(body);
  head.addEventListener("click", () => body.classList.toggle("collapsed"));
  wrap.dataset.callId = ev.call_id;
  wrap._body = body;
}

function renderToolResult(ev) {
  finalizeReasoning();
  const wrap = document.querySelector(`.block.tool[data-call-id="${cssEscape(ev.call_id)}"]`);
  if (wrap && wrap._body) {
    const flag = ev.truncated ? " (已截断)" : "";
    const mark = ev.ok ? "" : "⚠️ ";
    wrap._body.textContent = mark + ev.summary + flag;
  } else {
    const b = addBlock("toolresult");
    b.textContent = (ev.ok ? "" : "⚠️ ") + ev.summary + (ev.truncated ? " (已截断)" : "");
  }
}

function renderClarification(ev) {
  finalizeReasoning();
  const box = $("clarifyBox");
  $("clarifyQ").textContent = "🤔 " + ev.question;
  $("clarifyInput").value = "";
  box.classList.remove("hidden");
  $("clarifyInput").focus();
  state.pendingClarify = { call_id: ev.call_id };
}

function renderAnswer(ev) {
  finalizeReasoning();
  const card = addBlock("answer");
  const body = document.createElement("div");
  body.className = "blk-body";
  body.innerHTML = markdownish(ev.text);
  card.appendChild(body);
  if (ev.evidence && ev.evidence.length) {
    const chips = document.createElement("div");
    chips.className = "evidence";
    chips.append("证据: ");
    for (const e of ev.evidence) {
      const c = document.createElement("span");
      c.className = "chip";
      c.textContent = `${e.file}:${e.line}`;
      chips.appendChild(c);
    }
    card.appendChild(chips);
  }
}

function renderTerminal(ev) {
  finalizeReasoning();
  if (ev.type === "error") { const b = addBlock("error"); b.textContent = "❌ " + ev.message; }
  if (ev.type === "cancelled") { const b = addBlock("cancelled"); b.textContent = "🛑 已停止"; }
}

// ---------- event handling ----------

function handleEvent(ev) {
  switch (ev.type) {
    case "step": appendReasoning(ev.delta); break;
    case "tool_call": renderToolCall(ev); break;
    case "tool_result": renderToolResult(ev); break;
    case "clarification": renderClarification(ev); break;
    case "context": updateContext(ev.used, ev.window); break;
    case "answer": renderAnswer(ev); break;
    case "error":
    case "cancelled": renderTerminal(ev); break;
  }
  $("transcript").scrollTop = $("transcript").scrollHeight;
}

// ---------- streaming ----------

async function ensureSession() {
  if (state.sessionId) return state.sessionId;
  const r = await fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ apikey: state.cfg.apikey }),
  });
  if (!r.ok) throw new Error(`创建会话失败 (${r.status}): ${await r.text()}`);
  state.sessionId = (await r.json()).session_id;
  return state.sessionId;
}

async function sendMessage() {
  if (!state.cfg || !state.cfg.apikey) { openSettings(); return; }
  const question = $("question").value.trim();
  if (!question || state.streaming) return;

  const uq = addBlock("user"); uq.textContent = question;
  $("question").value = "";
  toggleComposer(true);
  setStatus("思考中…", "busy");

  try {
    const sid = await ensureSession();
    const r = await fetch(`/api/sessions/${sid}/message`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!r.ok) throw new Error(`发起分析失败 (${r.status}): ${await r.text()}`);
    const { stream_url } = await r.json();
    await streamEvents(stream_url);
  } catch (e) {
    const b = addBlock("error"); b.textContent = "❌ " + e.message;
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
      if (TERMINAL.includes(ev.type)) { state.streaming = false; return; }
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
  if (state.sessionId) {
    await fetch(`/api/sessions/${state.sessionId}/cancel`, { method: "POST" });
  }
  if (state.abortCtrl) state.abortCtrl.abort();
});
$("clarifySendBtn").addEventListener("click", async () => {
  if (!state.pendingClarify) return;
  const text = $("clarifyInput").value.trim();
  if (!text) return;
  const ub = addBlock("user"); ub.textContent = text;
  await fetch(`/api/sessions/${state.sessionId}/answer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ call_id: state.pendingClarify.call_id, text }),
  });
  state.pendingClarify = null;
  $("clarifyBox").classList.add("hidden");
});
$("resetBtn").addEventListener("click", () => {
  state.sessionId = null;
  $("transcript").innerHTML = "";
  $("clarifyBox").classList.add("hidden");
  initContextMeter();
});

// ---------- tiny utils ----------

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function cssEscape(s) { return (window.CSS && CSS.escape) ? CSS.escape(s) : s.replace(/"/g, '\\"'); }
function markdownish(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br>");
}

// ---------- init ----------

state.cfg = loadSettings();
loadServerConfig();
if (!state.cfg || !state.cfg.apikey) openSettings();
