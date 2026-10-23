// Text -> Sign panel controller: translate via the Python NLP engine
// (/api/translate), compile the sign plan into a timeline, animate the avatar.
import * as THREE from '../vendor/three.module.min.js';
import { Avatar, Hand, HandshapeLibrary, createStage } from './avatar.js';
import { Timeline } from './timeline.js';

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const state = {
  lex: null, lib: null, stage: null, avatar: null, tl: null,
  plan: null, playing: false, t: 0, last: 0, speed: 1, loop: false, activeIdx: -1,
  recorder: null,
};

const api = (path) => (window.getApiBase ? window.getApiBase() : '') + path;

let initPromise = null;

// Idempotent and awaitable: the first call builds the scene, later calls
// just resize (the panel may have been hidden while the window changed size).
export function initTextToSign() {
  if (initPromise) { state.stage?.resize(); return initPromise; }
  initPromise = init().catch((e) => { initPromise = null; throw e; });
  return initPromise;
}

// Used by Sign -> Text's "Sign it back" button.
export async function signText(text) {
  await initTextToSign();
  $('t2s-input').value = text;
  await translateAndSign();
}

async function init() {
  try {
    const data = await (await fetch(api('/api/sign-data'))).json();
    // hand-authored signs + signs captured from real signers (ASL Citizen)
    state.lex = { ...data.lexicon, signs: { ...data.lexicon.signs, ...(data.captured || {}) } };
    state.lib = new HandshapeLibrary(data.handshapes);
  } catch (e) {
    showError('Could not load the sign lexicon. Start the server with <code>uv run python app.py</code> and reload.');
    throw e;
  }
  const canvas = $('t2s-canvas');
  state.stage = createStage(canvas);
  state.avatar = new Avatar(state.lib);
  state.stage.scene.add(state.avatar.group);
  const probeMat = new THREE.MeshBasicMaterial();
  state.tl = new Timeline(state.lib, state.lex, { r: new Hand(1, probeMat), l: new Hand(-1, probeMat) });
  state.tl.compile({ items: [], nmm: [] });
  state.stage.resize();
  new ResizeObserver(() => state.stage.resize()).observe(canvas);
  buildDictionary();
  bindControls();
  requestAnimationFrame(frame);
  // Deep links (handy for demos and for reproducible paper figures):
  //   ?say=<english>          translate and sign on load
  //   ?sign=<GLOSS>           play one dictionary sign
  //   &at=<ms>&view=<name>    freeze at a time / pick the camera
  const q = new URLSearchParams(location.search);
  if (q.get('view')) { $('t2s-view').value = q.get('view'); state.stage.setView(q.get('view')); }
  if (q.get('say')) { $('t2s-input').value = q.get('say'); await translateAndSign(); }
  else if (q.get('sign')) play({ items: [{ kind: 'sign', id: q.get('sign'), gloss: q.get('sign') }], nmm: [] });
  if (q.get('at') && state.plan) { state.t = +q.get('at'); state.playing = false; onStopped(); }
  document.body.dataset.t2sReady = '1';
}

// ------------------------------------------------------------ render loop
function frame(now) {
  const dt = state.last ? now - state.last : 0;
  state.last = now;
  // don't spend GPU time on a panel nobody can see
  if (document.hidden || !document.getElementById('panel-text-to-sign').classList.contains('active')) {
    if (state.playing) { state.playing = false; onStopped(); }
    requestAnimationFrame(frame);
    return;
  }
  const tl = state.tl;
  if (state.playing) {
    state.t += dt;
    if (state.t >= tl.duration) {
      if (state.loop && state.plan) state.t = 0;
      else { state.t = tl.duration; state.playing = false; onStopped(); }
    }
  }
  const t = state.t;
  state.avatar.setArm('r', tl.sample('r', t));
  state.avatar.setArm('l', tl.sample('l', t));
  const face = tl.face(t);
  face.tilt = Math.sin(now * 0.0011) * 0.012;           // idle life
  state.avatar.setFace(face);
  state.stage.render();
  updateProgress(t);
  requestAnimationFrame(frame);
}

function updateProgress(t) {
  const tl = state.tl;
  $('t2s-progress').style.width = tl.duration ? `${Math.min(100, (t / tl.duration) * 100)}%` : '0%';
  const seg = tl.activeSegment(t);
  const idx = seg ? seg.idx : -1;
  if (idx !== state.activeIdx) {
    state.activeIdx = idx;
    document.querySelectorAll('.t2s-chip.active').forEach((c) => c.classList.remove('active'));
    const chip = document.querySelector(`.t2s-chip[data-idx="${idx}"]`);
    if (chip) chip.classList.add('active');
  }
  // fingerspelling: show the word with the current letter highlighted
  let caption = seg ? esc(seg.gloss) : '';
  if (seg && seg.letters) {
    const cur = seg.letters.reduce((k, l, i) => (t >= l.t ? i : k), 0);
    caption = seg.letters.map((l, i) => (i === cur ? `<u>${esc(l.ch)}</u>` : esc(l.ch))).join('');
  }
  if (caption !== state.caption) { state.caption = caption; $('t2s-caption').innerHTML = caption; }
}

// ------------------------------------------------------------ playback
function play(plan, { restart = true } = {}) {
  state.plan = plan;
  state.tl.compile(plan, state.speed);
  if (restart) state.t = 0;
  setPlaying(true);
  $('t2s-hint').hidden = true;
}

function setPlaying(on) {
  state.playing = on;
  const btn = $('t2s-btn-play');
  btn.classList.toggle('is-playing', on);
  btn.setAttribute('aria-label', on ? 'Pause' : 'Play');
}

function onStopped() {
  setPlaying(false);
  if (state.recorder) state.recorder.stop();
}

function togglePlay() {
  if (!state.plan) return;
  if (state.playing) { state.playing = false; onStopped(); return; }
  if (state.t >= state.tl.duration) state.t = 0;
  setPlaying(true);
}

async function translateAndSign() {
  const text = $('t2s-input').value.trim();
  if (!text) return;
  const btn = $('t2s-btn-translate');
  btn.disabled = true;
  hideError();
  try {
    const res = await fetch(api('/api/translate'), {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }),
    });
    const out = await res.json();
    if (out.error) throw new Error(out.error);
    renderResult(out);
    if (out.plan.items.length) play(out.plan);
  } catch (e) {
    showError(`Couldn't translate: ${esc(e.message)}. Is the server running?`);
  } finally {
    btn.disabled = false;
  }
}

// ------------------------------------------------------------ result view
function renderResult(out) {
  const chips = out.plan.items.map((it, idx) => {
    if (it.kind === 'pause') return '<span class="t2s-sep" aria-hidden="true"></span>';
    const cls = it.kind === 'fs' ? 't2s-chip fs' : 't2s-chip';
    return `<button class="${cls}" data-idx="${idx}" title="${it.kind === 'fs' ? 'fingerspelled (no sign in lexicon)' : esc(state.lex.signs[it.id]?.desc || '')}">${esc(it.gloss)}</button>`;
  }).join('');
  const nmm = out.plan.nmm.map((s) => {
    const label = { wh: 'Brows down · question', yn: 'Brows up · yes/no question', neg: 'Head shake · negation' }[s.type];
    const a = out.plan.items[s.from]?.gloss, b = out.plan.items[s.to]?.gloss;
    return `<span class="t2s-nmm">${label} · ${esc(a)}${a !== b ? ` → ${esc(b)}` : ''}</span>`;
  }).join('');
  $('t2s-gloss').innerHTML = chips || '<span class="muted">Nothing to sign.</span>';
  $('t2s-nmm').innerHTML = nmm;
  $('t2s-stats').innerHTML = `<b>${out.stats.signs}</b> signs · <b>${out.stats.fingerspelled}</b> spelled · <b>${Math.round(out.stats.coverage * 100)}%</b> coverage`;

  $('t2s-trace').innerHTML = out.sentences.map((s, k) => `
    <div class="t2s-trace-block">
      <div class="t2s-trace-head">${esc(s.english)} <span class="sub">· ${{ wh: 'question', yn: 'yes/no question', statement: 'statement' }[s.type]} · ${s.tense}${s.negated ? ' · negated' : ''}</span></div>
      <div class="t2s-row"><span>Normalised</span><code>${esc(s.trace.expanded)}</code></div>
      <div class="t2s-row"><span>Tokens</span><code>${s.trace.tokens.map(esc).join(' · ')}</code></div>
      <div class="t2s-row"><span>Base forms</span><code>${s.trace.lemmas.map(esc).join(' ')}</code></div>
      <div class="t2s-row"><span>Dropped</span><code>${s.trace.dropped.map((d) => `${esc(d.word)}<i>${esc(d.reason)}</i>`).join(' ') || '—'}</code></div>
      <div class="t2s-row"><span>Grammar</span><code>${s.trace.rules.map(esc).join('; ') || '—'}</code></div>
      <div class="t2s-row"><span>Gloss</span><code class="t2s-gloss-line">${s.gloss.map(esc).join(' ')}</code></div>
    </div>`).join('');
  $('t2s-results').hidden = false;
  $('t2s-empty').hidden = true;
  document.querySelectorAll('.t2s-chip').forEach((chip) => chip.addEventListener('click', () => seekTo(+chip.dataset.idx)));
}

function seekTo(idx) {
  const seg = state.tl.segments.find((s) => s.idx === idx);
  if (!seg) return;
  state.t = seg.t0;
  setPlaying(true);
}

// ------------------------------------------------------------ dictionary
function buildDictionary() {
  const list = $('t2s-dict-list');
  const entries = Object.entries(state.lex.signs).sort(([a], [b]) => a.localeCompare(b));
  $('t2s-dict-count').textContent = `${entries.length} signs`;
  list.innerHTML = entries.map(([id, s]) => `
    <button class="t2s-dict-item" data-id="${esc(id)}" data-words="${esc((s.words || []).join(' '))}">
      <b>${esc(id)}</b><span>${esc(s.desc || '')}</span>
    </button>`).join('');
  list.querySelectorAll('.t2s-dict-item').forEach((b) => b.addEventListener('click', () => {
    play({ items: [{ kind: 'sign', id: b.dataset.id, gloss: b.dataset.id }], nmm: [] });
  }));
  $('t2s-dict-search').addEventListener('input', (e) => {
    const q = e.target.value.trim().toLowerCase();
    list.querySelectorAll('.t2s-dict-item').forEach((b) => {
      b.hidden = q && !b.dataset.id.toLowerCase().includes(q) && !b.dataset.words.includes(q);
    });
  });
}

// ------------------------------------------------------------ video export
function exportVideo() {
  if (!state.plan || state.recorder) return;
  const canvas = $('t2s-canvas');
  const stream = canvas.captureStream(30);
  const chunks = [];
  const rec = new MediaRecorder(stream, { mimeType: MediaRecorder.isTypeSupported('video/webm;codecs=vp9') ? 'video/webm;codecs=vp9' : 'video/webm' });
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  rec.onstop = () => {
    state.recorder = null;
    $('t2s-btn-export').disabled = false;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob(chunks, { type: 'video/webm' }));
    a.download = `ishaara-${Date.now()}.webm`;
    a.click();
  };
  state.recorder = rec;
  $('t2s-btn-export').disabled = true;
  state.loop = false;
  $('t2s-loop').checked = false;
  rec.start();
  play(state.plan);
}

// ------------------------------------------------------------ wiring
function bindControls() {
  $('t2s-btn-translate').addEventListener('click', translateAndSign);
  $('t2s-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) translateAndSign();
  });
  $('t2s-btn-play').addEventListener('click', togglePlay);
  $('t2s-btn-replay').addEventListener('click', () => state.plan && play(state.plan));
  $('t2s-btn-export').addEventListener('click', exportVideo);
  $('t2s-speed').addEventListener('change', (e) => {
    const old = state.tl.duration || 1, frac = state.t / old;
    state.speed = +e.target.value;
    if (state.plan) { state.tl.compile(state.plan, state.speed); state.t = frac * state.tl.duration; }
  });
  $('t2s-view').addEventListener('change', (e) => state.stage.setView(e.target.value));
  $('t2s-loop').addEventListener('change', (e) => { state.loop = e.target.checked; });
  document.querySelectorAll('.t2s-example').forEach((b) => b.addEventListener('click', () => {
    $('t2s-input').value = b.dataset.text;
    translateAndSign();
  }));
}

function showError(html) { const el = $('t2s-error'); el.innerHTML = html; el.hidden = false; }
function hideError() { $('t2s-error').hidden = true; }
