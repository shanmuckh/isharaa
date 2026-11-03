// Sign -> Text panel. Two recognisers share the camera, transcript and output:
//
//   Words (default): MediaPipe hand + body tracking in the browser, one clip
//     per sign, 200-word sequence model on the server (POST /api/sign).
//   Handshapes: the DETR detector on single frames (POST /api/detect), the
//     5 signs from Ishaara's own dataset. Two requests are kept in flight, and
//     class probabilities are smoothed over time.
import { speak } from '../speech/app.js';
import { Segmenter, drawSkeleton, loadTrackers, track } from './words.js';

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const api = (path) => (window.getApiBase ? window.getApiBase() : '') + path;

const SIGNS = {
  hello: { word: 'Hello', how: 'Open palm, wave' },
  iloveyou: { word: 'I love you', how: 'Thumb, index and pinky up' },
  thankyou: { word: 'Thank you', how: 'Flat hand forward from chin' },
  zero: { word: 'zero', how: 'Fingers and thumb form an O' },
  one: { word: 'one', how: 'Index finger up, fist closed' },
};

// handshape smoothing: accept a sign once its smoothed probability stays above
// COMMIT_P for COMMIT_FRAMES frames; accept it again only after RELEASE_P
const EMA = 0.45;
const COMMIT_P = 0.7;
const COMMIT_FRAMES = 6;
const RELEASE_P = 0.35;
// words: add the top guess automatically only when the model is this sure
const AUTO_ADD_P = 0.35;

const state = {
  mode: 'words', stream: null, running: false, ready: false, transcript: [], frameTimes: [],
  // handshapes
  classes: null, ema: null, streak: { cls: -1, n: 0 }, armed: true, seq: 0, applied: 0,
  // words
  trackers: null, segmenter: null, lastFrame: null, pending: 0, lastClipAdded: false, vocab: [],
};

export function initSignToText() {
  if (state.ready) return;
  state.ready = true;
  $('s2t-toggle').addEventListener('click', () => (state.running ? stop() : start()));
  $('s2t-start').addEventListener('click', start);
  $('s2t-clear').addEventListener('click', () => { state.transcript = []; state.lastClipAdded = false; renderTranscript(); });
  $('s2t-undo').addEventListener('click', () => { state.transcript.pop(); state.lastClipAdded = false; renderTranscript(); });
  $('s2t-sign-back').addEventListener('click', signBack);
  $('s2t-speak').addEventListener('click', () => speak(transcriptText()));
  $('s2t-camera').addEventListener('change', () => state.running && start());
  $('s2t-mode-words').addEventListener('click', () => setMode('words'));
  $('s2t-mode-shapes').addEventListener('click', () => setMode('shapes'));
  $('s2t-vocab-search').addEventListener('input', renderVocab);
  document.addEventListener('visibilitychange', () => { if (document.hidden && state.running) stop(); });
  renderProbs(Object.keys(SIGNS), null);
  fetch(api('/api/signs')).then((r) => r.json()).then((d) => { state.vocab = d.signs || []; renderVocab(); updateModelLine(); }).catch(() => {});
  fetch(api('/api/detector?load=1')).then((r) => r.json()).then((info) => {
    if (info.classes) { state.classes = info.classes; renderProbs(info.classes, null); }
    state.detectorInfo = info;
    updateModelLine();
  }).catch(() => {});
  loadTrackers().catch(() => {});        // fetch the tracking models early
  const q = new URLSearchParams(location.search);
  if (q.get('mode') === 'shapes') setMode('shapes');
  if (q.get('camera') === '1') start();
}

function setMode(mode) {
  if (mode === state.mode) return;
  const wasRunning = state.running;
  if (wasRunning) stop(true);
  state.mode = mode;
  $('s2t-mode-words').setAttribute('aria-selected', mode === 'words');
  $('s2t-mode-shapes').setAttribute('aria-selected', mode === 'shapes');
  $('s2t-words-pane').hidden = mode !== 'words';
  $('s2t-shapes-pane').hidden = mode !== 'shapes';
  $('s2t-how').textContent = mode === 'words'
    ? 'Raise your hands to sign a word, then lower them. Each word is recognised from your hand and arm movement.'
    : 'Hold one of the 5 handshapes steady in view. Lower your hand between signs to repeat one.';
  renderVocab();
  updateModelLine();
  if (wasRunning) start();
}

function updateModelLine() {
  const d = state.detectorInfo;
  $('s2t-model').textContent = state.mode === 'words'
    ? (state.vocab.length ? `${state.vocab.length} signs · ASL Citizen` : 'Word model not loaded')
    : (d?.weights ? `${d.weights.replace('pretrained/', '')} · ${d.device?.toUpperCase()}` : 'Loading…');
}

// ------------------------------------------------------------ camera
async function start() {
  hideError();
  stop(true);
  const deviceId = $('s2t-camera').value;
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { width: { ideal: 640 }, height: { ideal: 480 }, ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: 'user' }) },
    });
  } catch (e) {
    showError({
      NotAllowedError: 'Camera access was blocked. Click the camera icon in the address bar, allow access, then press Start again.',
      NotFoundError: 'No camera was found. Connect a camera and try again.',
      NotReadableError: 'The camera is being used by another app. Close it and try again.',
    }[e.name] || `Could not start the camera: ${esc(e.message)}`);
    return;
  }
  const video = $('s2t-video');
  video.srcObject = state.stream;
  await video.play();
  await listCameras();
  state.running = true;
  state.frameTimes = [];
  $('s2t-hint').hidden = true;
  $('s2t-toggle').classList.add('is-playing');
  $('s2t-toggle').setAttribute('aria-label', 'Stop camera');
  window.setCameraLive?.(true);
  if (state.mode === 'words') startWords();
  else startShapes();
}

function stop(silent = false) {
  state.running = false;
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = null;
  $('s2t-video').srcObject = null;
  clearOverlay();
  $('s2t-caption').textContent = '';
  if (silent) return;
  $('s2t-hint').hidden = false;
  $('s2t-toggle').classList.remove('is-playing');
  $('s2t-toggle').setAttribute('aria-label', 'Start camera');
  $('s2t-fps').textContent = '— fps';
  $('s2t-latency').textContent = '— ms';
  window.setCameraLive?.(false);
}

async function listCameras() {
  const sel = $('s2t-camera');
  const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
  const current = state.stream?.getVideoTracks()[0]?.getSettings().deviceId;
  sel.innerHTML = cams.map((c, i) => `<option value="${esc(c.deviceId)}">${esc(c.label || `Camera ${i + 1}`)}</option>`).join('');
  if (current) sel.value = current;
  sel.hidden = cams.length < 2;
}

function tick() {
  const now = performance.now();
  state.frameTimes.push(now);
  while (state.frameTimes.length && now - state.frameTimes[0] > 1000) state.frameTimes.shift();
  $('s2t-fps').textContent = `${state.frameTimes.length} fps`;
}

const panelVisible = () => $('panel-sign-to-text').classList.contains('active');

// ------------------------------------------------------------ words
async function startWords() {
  $('s2t-words-status').textContent = 'Loading hand tracking…';
  try {
    state.trackers = await loadTrackers();
  } catch (e) {
    showError(`Hand tracking could not start: ${esc(e.message)}`);
    stop();
    return;
  }
  if (!state.running) return;
  state.segmenter = new Segmenter(recognizeClip);
  $('s2t-words-status').textContent = 'Ready. Raise your hands and sign a word.';
  const video = $('s2t-video');
  let lastTs = -1;
  const step = () => {
    if (!state.running || state.mode !== 'words') return;
    if (panelVisible() && video.readyState >= 2) {
      const ts = performance.now();
      if (ts > lastTs) {
        lastTs = ts;
        const frame = track(state.trackers, video, ts);
        state.lastFrame = frame;
        const was = state.segmenter.state;
        const now = state.segmenter.push(frame);
        if (now === 'signing' && was !== 'signing') $('s2t-caption').textContent = '● Signing';
        tick();
        drawFrame(frame);
      }
    }
    if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(step);
    else requestAnimationFrame(step);
  };
  step();
}

async function recognizeClip(clip) {
  $('s2t-caption').textContent = '…';
  state.pending++;
  const t0 = performance.now();
  try {
    const r = await fetch(api('/api/sign'), {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frames: clip.map((f) => Array.from(f, (v) => Math.round(v * 1e4) / 1e4)) }),
    });
    const res = await r.json();
    if (!r.ok) throw new Error(res.error || r.statusText);
    $('s2t-latency').textContent = `${Math.round(performance.now() - t0)} ms`;
    const best = res.candidates[0];
    const sure = best.p >= AUTO_ADD_P;
    if (sure) addWord(best.word, true);
    else state.lastClipAdded = false;
    $('s2t-caption').textContent = sure ? best.word : '?';
    $('s2t-words-status').textContent = sure
      ? `Added “${best.word}”. Wrong? Pick the right word below.`
      : 'Not sure about that one. Pick the word below, or sign it again.';
    renderCandidates(res.candidates);
  } catch (e) {
    $('s2t-caption').textContent = '';
    showError(`Recognition failed: ${esc(e.message)}`);
  } finally {
    state.pending--;
  }
}

function renderCandidates(cands) {
  $('s2t-cands').innerHTML = cands.map((c, i) => `
    <button class="prob-row cand${i === 0 ? ' active' : ''}" data-word="${esc(c.word)}">
      <div class="prob-top"><b>${esc(c.word)}</b><span class="prob-pct">${Math.round(c.p * 100)}%</span></div>
      <div class="prob-bar"><div style="width:${Math.round(c.p * 100)}%"></div></div>
    </button>`).join('');
  $('s2t-cands').querySelectorAll('.cand').forEach((b) => b.addEventListener('click', () => {
    // correct the word this clip added, or add it if nothing was added
    if (state.lastClipAdded) state.transcript.pop();
    addWord(b.dataset.word, true);
    $('s2t-cands').querySelectorAll('.cand').forEach((x) => x.classList.toggle('active', x === b));
  }));
}

function drawFrame(frame) {
  const { ctx, map } = overlayContext();
  if (ctx) drawSkeleton(ctx, frame, map);
}

// ------------------------------------------------------------ handshapes (DETR)
const IN_FLIGHT = 2;

function startShapes() {
  state.ema = null;
  state.streak = { cls: -1, n: 0 };
  state.armed = true;
  state.seq = 0;
  state.applied = 0;
  for (let i = 0; i < IN_FLIGHT; i++) shapesWorker();
}

async function shapesWorker() {
  const video = $('s2t-video');
  const grab = document.createElement('canvas');
  grab.width = grab.height = 224;
  const ctx = grab.getContext('2d');
  while (state.running && state.mode === 'shapes') {
    if (!panelVisible()) { await new Promise((r) => setTimeout(r, 300)); continue; }
    if (video.readyState < 2) { await new Promise((r) => requestAnimationFrame(r)); continue; }
    const seq = ++state.seq;
    ctx.drawImage(video, 0, 0, 224, 224);
    const blob = await new Promise((r) => grab.toBlob(r, 'image/jpeg', 0.85));
    let res;
    try {
      const r = await fetch(api('/api/detect'), { method: 'POST', headers: { 'Content-Type': 'image/jpeg' }, body: blob });
      res = await r.json();
      if (!r.ok) throw new Error(res.error || r.statusText);
    } catch (e) {
      if (!state.running) return;
      showError(`Recognition stopped: ${esc(e.message)}`);
      stop();
      if (e instanceof TypeError) window.openModal?.();
      return;
    }
    if (!state.running || state.mode !== 'shapes') return;
    if (seq > state.applied) { state.applied = seq; onShapes(res); }
  }
}

function onShapes(res) {
  state.classes = res.classes;
  tick();
  $('s2t-latency').textContent = `${Math.round(res.latency_ms)} ms`;
  state.ema = state.ema ? state.ema.map((v, i) => v * (1 - EMA) + res.probs[i] * EMA) : res.probs.slice();
  const top = state.ema.indexOf(Math.max(...state.ema));
  const p = state.ema[top];
  const cls = res.classes[top];
  if (p >= COMMIT_P) {
    state.streak = state.streak.cls === top ? { cls: top, n: state.streak.n + 1 } : { cls: top, n: 1 };
    if (state.armed && state.streak.n >= COMMIT_FRAMES) {
      addWord(SIGNS[cls]?.word || cls, false);
      state.armed = false;
    }
  } else {
    state.streak = { cls: -1, n: 0 };
  }
  if (Math.max(...state.ema) < RELEASE_P) state.armed = true;
  $('s2t-caption').textContent = p >= 0.5 ? SIGNS[cls]?.word || cls : '';
  renderProbs(res.classes, state.ema, p >= 0.5 ? top : -1);
  drawBoxes(res.detections);
}

function drawBoxes(dets) {
  const { ctx, box } = overlayContext();
  if (!ctx) return;
  ctx.font = '600 13px -apple-system, BlinkMacSystemFont, system-ui, sans-serif';
  for (const d of dets.slice(0, 3)) {
    const { left, top, bw, bh } = box(d.box);
    ctx.strokeStyle = 'rgba(255,255,255,0.95)';
    ctx.lineWidth = 2;
    roundRect(ctx, left, top, bw, bh, 12);
    ctx.stroke();
    const label = `${SIGNS[d.class]?.word || d.class}  ${Math.round(d.score * 100)}%`;
    const tw = ctx.measureText(label).width + 16;
    ctx.fillStyle = 'rgba(255,255,255,0.95)';
    roundRect(ctx, left, Math.max(0, top - 28), tw, 22, 11);
    ctx.fill();
    ctx.fillStyle = '#000';
    ctx.fillText(label, left + 8, Math.max(0, top - 28) + 15);
  }
}

// ------------------------------------------------------------ shared drawing
// The preview is mirrored like a selfie, so overlays are mirrored too.
function overlayContext() {
  const video = $('s2t-video'), canvas = $('s2t-overlay');
  const w = canvas.clientWidth, h = canvas.clientHeight, dpr = window.devicePixelRatio || 1;
  if (!w || !h) return {};
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) { canvas.width = w * dpr; canvas.height = h * dpr; }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const vw = video.videoWidth || 4, vh = video.videoHeight || 3;
  const s = Math.min(w / vw, h / vh), dw = vw * s, dh = vh * s, ox = (w - dw) / 2, oy = (h - dh) / 2;
  return {
    ctx,
    map: (pt) => [ox + (1 - pt.x) * dw, oy + pt.y * dh],
    box: ([x1, y1, x2, y2]) => ({ left: ox + (1 - x2) * dw, top: oy + y1 * dh, bw: (x2 - x1) * dw, bh: (y2 - y1) * dh }),
  };
}

function clearOverlay() {
  const c = $('s2t-overlay');
  c.getContext('2d').clearRect(0, 0, c.width, c.height);
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.roundRect ? ctx.roundRect(x, y, w, h, r) : ctx.rect(x, y, w, h);
}

function renderProbs(classes, probs, active = -1) {
  const list = $('s2t-probs');
  if (list.children.length !== classes.length) {
    list.innerHTML = classes.map((c) => `
      <div class="prob-row" data-cls="${esc(c)}">
        <div class="prob-top"><b>${esc(SIGNS[c]?.word || c)}</b><span class="prob-pct">0%</span></div>
        <div class="prob-how">${esc(SIGNS[c]?.how || '')}</div>
        <div class="prob-bar"><div></div></div>
      </div>`).join('');
  }
  [...list.children].forEach((row, i) => {
    const p = probs ? probs[i] : 0;
    row.querySelector('.prob-bar > div').style.width = `${Math.round(p * 100)}%`;
    row.querySelector('.prob-pct').textContent = `${Math.round(p * 100)}%`;
    row.classList.toggle('active', i === active);
  });
}

function renderVocab() {
  const q = $('s2t-vocab-search').value.trim().toLowerCase();
  const all = state.mode === 'words' ? state.vocab.map((v) => v.word) : Object.values(SIGNS).map((s) => s.word);
  const shown = all.filter((w) => !q || w.toLowerCase().includes(q));
  $('s2t-vocab-count').textContent = `${all.length} signs`;
  $('s2t-vocab').innerHTML = shown.map((w) => `<span class="vocab-chip">${esc(w)}</span>`).join('') || '<span class="muted">No match.</span>';
}

// ------------------------------------------------------------ transcript
function addWord(word, fromClip) {
  state.transcript.push(word);
  state.lastClipAdded = fromClip;
  if ($('s2t-autospeak').checked) speak(word, { interrupt: false });
  renderTranscript(true);
}

function renderTranscript(flash = false) {
  const box = $('s2t-transcript');
  box.innerHTML = state.transcript.map((w, i) =>
    `<span class="t2s-chip${flash && i === state.transcript.length - 1 ? ' active' : ''}">${esc(w)}</span>`).join('');
  const empty = state.transcript.length === 0;
  $('s2t-empty').hidden = !empty;
  $('s2t-sign-back').disabled = $('s2t-speak').disabled = $('s2t-clear').disabled = $('s2t-undo').disabled = empty;
  if (flash) setTimeout(() => box.lastElementChild?.classList.remove('active'), 600);
}

function transcriptText() {
  return state.transcript.length ? state.transcript.join(' ').replace(/^./, (m) => m.toUpperCase()) + '.' : '';
}

async function signBack() {
  const text = transcriptText();
  if (!text) return;
  if (state.running) stop();
  window.switchMode('text-to-sign');
  const { signText } = await import('../text2sign/app.js');
  signText(text);
}

function showError(html) { const el = $('s2t-error'); el.innerHTML = html; el.hidden = false; }
function hideError() { $('s2t-error').hidden = true; }
