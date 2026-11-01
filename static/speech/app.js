// Speech in and out.
//   Speech -> Sign: microphone -> 16 kHz PCM -> POST /api/transcribe (local
//                   Whisper) -> the Text -> Sign engine and avatar.
//   Sign -> Speech: the browser's on-device voices read recognised signs aloud.
import { signText } from '../text2sign/app.js';

const $ = (id) => document.getElementById(id);
const api = (path) => (window.getApiBase ? window.getApiBase() : '') + path;

const TARGET_RATE = 16000;
const MAX_SECONDS = 30;
const NO_SPEECH_TIMEOUT = 8;     // give up if nothing is said for this long
const END_SILENCE = 1.2;         // seconds of quiet that end the utterance
const PARTIAL_EVERY = 1.5;       // live preview interval
const SPEECH_RMS = 0.015;

const mic = { state: 'idle', stream: null, ctx: null, node: null, chunks: [], rate: 48000,
  samples: 0, heard: false, lastVoice: 0, partialBusy: false, lastPartial: 0, session: 0, typed: '' };

// ------------------------------------------------------------ audio helpers
function toPcm16(chunks, rate) {
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const ratio = rate / TARGET_RATE;
  const out = new Int16Array(Math.floor(total / ratio));
  let chunk = 0, pos = 0;
  for (let i = 0; i < out.length; i++) {        // box filter, then decimate
    const n = Math.max(1, Math.floor((i + 1) * ratio) - Math.floor(i * ratio));
    let sum = 0;
    for (let k = 0; k < n; k++) {
      while (chunk < chunks.length && pos >= chunks[chunk].length) { chunk++; pos = 0; }
      if (chunk < chunks.length) sum += chunks[chunk][pos++];
    }
    out[i] = Math.max(-1, Math.min(1, sum / n)) * 32767;
  }
  return out;
}

async function transcribe(pcm) {
  const r = await fetch(api('/api/transcribe'), { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: pcm.buffer });
  const res = await r.json();
  if (!r.ok) throw new Error(res.error || r.statusText);
  return res;
}

// ------------------------------------------------------------ microphone
function setState(state, status = '') {
  mic.state = state;
  const btn = $('speech-mic');
  btn.dataset.state = state;
  btn.setAttribute('aria-pressed', state === 'listening');
  btn.querySelector('span').textContent = { idle: 'Speak', listening: 'Listening', working: 'Working…' }[state];
  btn.disabled = state === 'working';
  $('speech-status').textContent = status;
}

function showError(message) {
  const el = $('t2s-error');
  el.textContent = message;
  el.hidden = false;
}

async function startListening() {
  $('t2s-error').hidden = true;
  if (!navigator.mediaDevices?.getUserMedia) { showError('This browser cannot use the microphone here.'); return; }
  setState('working', 'Starting the microphone…');
  try {
    mic.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (e) {
    showError({
      NotAllowedError: 'Microphone access was blocked. Allow it from the icon in the address bar, then try again.',
      NotFoundError: 'No microphone was found.',
      NotReadableError: 'Another app is using the microphone. Close it and try again.',
    }[e.name] || `Could not start the microphone: ${e.message}`);
    setState('idle');
    return;
  }
  window.speechSynthesis?.cancel();     // never transcribe our own voice
  try {
    mic.ctx = new AudioContext();
    await mic.ctx.resume();
    await mic.ctx.audioWorklet.addModule(new URL('./capture-worklet.js', import.meta.url));
  } catch (e) {
    releaseMic();
    setState('idle');
    showError(`Could not start audio capture: ${e.message}`);
    return;
  }
  mic.rate = mic.ctx.sampleRate;
  Object.assign(mic, { chunks: [], samples: 0, heard: false, lastVoice: 0, lastPartial: 0, partialBusy: false, typed: $('t2s-input').value });
  mic.session++;
  const source = mic.ctx.createMediaStreamSource(mic.stream);
  mic.node = new AudioWorkletNode(mic.ctx, 'capture-processor');
  mic.node.port.onmessage = (e) => onAudio(e.data);
  source.connect(mic.node);
  // silent path to the output: some browsers only run nodes that reach it
  const mute = mic.ctx.createGain();
  mute.gain.value = 0;
  mic.node.connect(mute).connect(mic.ctx.destination);
  setState('listening', 'Speak now. It stops when you pause.');
  $('t2s-input').value = '';
}

function onAudio(chunk) {
  if (mic.state !== 'listening') return;
  mic.chunks.push(chunk);
  mic.samples += chunk.length;
  const seconds = mic.samples / mic.rate;
  const rms = Math.sqrt(chunk.reduce((s, v) => s + v * v, 0) / chunk.length);
  $('speech-mic').style.setProperty('--level', Math.min(1, rms * 12).toFixed(3));
  if (rms > SPEECH_RMS) { mic.heard = true; mic.lastVoice = seconds; }

  if (mic.heard && seconds - mic.lastVoice > END_SILENCE) finishListening();
  else if (!mic.heard && seconds > NO_SPEECH_TIMEOUT) cancelListening("I didn't hear anything. Tap Speak to try again.");
  else if (seconds > MAX_SECONDS) finishListening();
  else if (mic.heard && !mic.partialBusy && seconds - mic.lastPartial > PARTIAL_EVERY) livePreview(seconds);
}

async function livePreview(seconds) {
  mic.partialBusy = true;
  mic.lastPartial = seconds;
  const session = mic.session;
  try {
    const { text } = await transcribe(toPcm16(mic.chunks, mic.rate));
    if (mic.state === 'listening' && session === mic.session && text) $('t2s-input').value = text;
  } catch (e) { /* the final pass reports errors */ }
  mic.partialBusy = false;
}

function releaseMic() {
  mic.node?.port.close();
  mic.stream?.getTracks().forEach((t) => t.stop());
  mic.ctx?.close();
  mic.stream = mic.ctx = mic.node = null;
  $('speech-mic').style.setProperty('--level', 0);
}

function cancelListening(status = '') {
  releaseMic();
  $('t2s-input').value = mic.typed;     // give back what was in the box
  setState('idle', status);
}

async function finishListening() {
  const session = ++mic.session;        // outdates any live preview still in flight
  const pcm = toPcm16(mic.chunks, mic.rate);
  releaseMic();
  setState('working', 'Understanding…');
  try {
    const res = await transcribe(pcm);
    if (session !== mic.session) return;
    if (!res.text) { cancelListening("I couldn't make out any words. Tap Speak to try again."); return; }
    setState('idle', `Heard ${res.audio_s.toFixed(1)} s of speech in ${(res.latency_ms / 1000).toFixed(1)} s, on this computer`);
    await signText(res.text);
  } catch (e) {
    cancelListening();
    showError(`Speech recognition failed: ${e.message}`);
  }
}

// ------------------------------------------------------------ reading aloud
export function speak(text, { interrupt = true } = {}) {
  const synth = window.speechSynthesis;
  if (!synth || !text) return false;
  if (interrupt) synth.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = 'en-US';
  synth.speak(u);
  return true;
}

export function initSpeech() {
  $('speech-mic').addEventListener('click', () => {
    if (mic.state === 'idle') startListening();
    else if (mic.state === 'listening') (mic.heard ? finishListening() : cancelListening());
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && mic.state === 'listening') cancelListening();
  });
  if (!window.speechSynthesis) {
    $('s2t-speak').hidden = true;
    $('s2t-autospeak-wrap').hidden = true;
  }
  try { $('s2t-autospeak').checked = localStorage.getItem('ishaara-autospeak') === '1'; } catch (e) { /* per-visit only */ }
  $('s2t-autospeak').addEventListener('change', (e) => {
    try { localStorage.setItem('ishaara-autospeak', e.target.checked ? '1' : '0'); } catch (err) { /* per-visit only */ }
  });
}
