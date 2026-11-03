// Word mode: MediaPipe hand + body tracking in the browser, cut into one clip
// per sign, recognised on the server (POST /api/sign, 200 ASL Citizen signs).
//
// Landmarks use the ASL Citizen layout: 33 pose points, then the signer's left
// hand (21), then the right hand (21); a missing point is 0. MediaPipe's own
// "Left/Right" hand labels assume a mirrored selfie image, so instead each hand
// is matched to the nearest body wrist (pose 15 = left, 16 = right), which is
// the same relationship the training data has.
import { FilesetResolver, HandLandmarker, PoseLandmarker } from '../vendor/mediapipe/vision_bundle.mjs';

const BASE = new URL('../vendor/mediapipe/', import.meta.url).href;
let trackers = null;

export function loadTrackers() {
  if (!trackers) {
    trackers = (async () => {
      const files = await FilesetResolver.forVisionTasks(BASE + 'wasm');
      const make = (Task, model, extra, delegate) => Task.createFromOptions(files, {
        baseOptions: { modelAssetPath: BASE + 'models/' + model, delegate }, runningMode: 'VIDEO', ...extra,
      });
      const both = (delegate) => Promise.all([
        make(PoseLandmarker, 'pose_landmarker_lite.task', { numPoses: 1 }, delegate),
        make(HandLandmarker, 'hand_landmarker.task', { numHands: 2 }, delegate),
      ]);
      let pose, hands;
      try { [pose, hands] = await both('GPU'); } catch (e) { [pose, hands] = await both('CPU'); }
      return { pose, hands };
    })().catch((e) => { trackers = null; throw e; });
  }
  return trackers;
}

// ------------------------------------------------------------ one frame
export function track({ pose, hands }, video, ts) {
  const p = pose.detectForVideo(video, ts).landmarks[0];
  const h = hands.detectForVideo(video, ts).landmarks;
  const raw = new Float32Array(225);
  if (!p) return { raw, pose: null, hands: [], active: false };
  p.forEach((pt, i) => raw.set([pt.x, pt.y, pt.z], i * 3));
  const placed = [];
  for (const lm of h) {
    const d = (k) => Math.hypot(lm[0].x - p[k].x, lm[0].y - p[k].y);
    let slot = d(15) <= d(16) ? 0 : 1;                       // 0 = signer's left hand
    if (placed.some((q) => q.slot === slot)) slot = 1 - slot;  // never two hands in one slot
    placed.push({ lm, slot });
    lm.forEach((pt, i) => raw.set([pt.x, pt.y, pt.z], (33 + slot * 21 + i) * 3));
  }
  return { raw, pose: p, hands: placed, active: isActive(raw) };
}

// "signing" = a hand is raised above roughly the bottom of the chest.
// Works on the raw 75-point frame so recorded landmarks can be replayed.
export function isActive(raw) {
  const pt = (i) => [raw[i * 3], raw[i * 3 + 1]];
  const [l, r] = [pt(11), pt(12)];
  if (!l[0] && !r[0]) return false;
  const sw = Math.hypot(l[0] - r[0], l[1] - r[1]);
  const shoulderY = (l[1] + r[1]) / 2;
  return [33, 54].some((w) => { const [x, y] = pt(w); return (x || y) && y < shoulderY + 1.3 * sw; });
}

// ------------------------------------------------------------ clip cutting
// idle -> (hands raised for START frames) -> signing -> (hands down for END
// frames) -> clip. A short lead-in and tail are kept, like the dataset clips.
const START = 3;
const END = 9;
const LEAD_IN = 10;
const MAX_FRAMES = 180;
const MIN_ACTIVE = 6;

export class Segmenter {
  constructor(onClip) {
    this.onClip = onClip;
    this.reset();
  }

  reset() {
    this.state = 'idle';
    this.history = [];
    this.clip = [];
    this.up = 0;
    this.down = 0;
    this.activeFrames = 0;
  }

  push(frame) {
    if (this.state === 'idle') {
      this.history.push(frame.raw);
      if (this.history.length > LEAD_IN) this.history.shift();
      this.up = frame.active ? this.up + 1 : 0;
      if (this.up >= START) {
        this.state = 'signing';
        this.clip = this.history.slice();
        this.activeFrames = this.up;
        this.down = 0;
      }
      return this.state;
    }
    this.clip.push(frame.raw);
    if (frame.active) { this.activeFrames++; this.down = 0; } else this.down++;
    if (this.down >= END || this.clip.length >= MAX_FRAMES) {
      const clip = this.clip, ok = this.activeFrames >= MIN_ACTIVE;
      this.reset();
      if (ok) this.onClip(clip);
    }
    return this.state;
  }
}

// ------------------------------------------------------------ drawing
const HAND_LINKS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];
const ARM_LINKS = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16]];

export function drawSkeleton(ctx, frame, map) {
  if (!frame.pose) return;
  const line = (a, b) => { ctx.beginPath(); ctx.moveTo(...map(a)); ctx.lineTo(...map(b)); ctx.stroke(); };
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'rgba(255,255,255,0.35)';
  ctx.lineWidth = 3;
  for (const [a, b] of ARM_LINKS) line(frame.pose[a], frame.pose[b]);
  ctx.strokeStyle = frame.active ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.5)';
  ctx.lineWidth = 2;
  for (const { lm } of frame.hands) {
    for (const [a, b] of HAND_LINKS) line(lm[a], lm[b]);
    ctx.fillStyle = ctx.strokeStyle;
    for (const pt of lm) { const [x, y] = map(pt); ctx.beginPath(); ctx.arc(x, y, 2.4, 0, Math.PI * 2); ctx.fill(); }
  }
}
