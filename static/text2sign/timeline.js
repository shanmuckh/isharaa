// Sign plan -> animation timeline.
//
// A sign in the lexicon is a list of keyframes per hand. Each keyframe is a
// *delta* from the previous one (hs, loc, off, palm, fing and contact carry
// over), so a tap is just {"off": [...]} after the first frame.
//
//   hs       handshape name            loc    named location or [x,y,z]
//   off      [dx,dy,dz] offset (m)     palm   direction the palm faces
//   fing     direction of the fingers  contact which part of the hand is at loc
//   d        ms to reach this frame    hold   ms to stay
//   ease     smooth | lin | in | out   wiggle true = wiggle fingers
//
// The compiler resolves each keyframe to a wrist position + hand rotation
// (so the *contact point* lands on the location), lays signs end-to-end with
// co-articulated transitions, and the sampler interpolates with Catmull-Rom
// paths, quaternion slerp and handshape blending.
import * as THREE from '../vendor/three.module.min.js';
import { S, parseLoc, handQuat, lerpShape } from './avatar.js';

const REST = {
  r: { hs: 'relaxed', loc: 'restR', off: [0, 0, 0], palm: 'left+up', fing: 'out+left', contact: 'palm' },
  l: { hs: 'relaxed', loc: 'restL', off: [0, 0, 0], palm: 'right+up', fing: 'out+right', contact: 'palm' },
};
const CARRY = ['hs', 'loc', 'off', 'palm', 'fing', 'contact'];
const EASE = {
  smooth: (u) => 0.5 - 0.5 * Math.cos(Math.PI * u),
  lin: (u) => u,
  in: (u) => u * u,
  out: (u) => 1 - (1 - u) * (1 - u),
};

const swapLR = (d) => (Array.isArray(d) ? [-d[0], d[1], d[2]]
  : d.replace(/left|right/g, (m) => (m === 'left' ? 'right' : 'left')));

export function mirrorFrames(frames) {
  return frames.map((f) => {
    const g = { ...f };
    if (g.loc !== undefined) g.loc = Array.isArray(g.loc) ? swapLR(g.loc) : g.loc.replace(/R$|L$/, (m) => (m === 'R' ? 'L' : 'R'));
    if (g.off) g.off = swapLR(g.off);
    if (g.palm) g.palm = swapLR(g.palm);
    if (g.fing) g.fing = swapLR(g.fing);
    return g;
  });
}

// Mirror named locations: authoring names ending in R/L swap; symmetric
// names (chest, temple...) are defined for the right side and mirrored in x.
function resolveLoc(loc, off, mirrored) {
  if (!mirrored || Array.isArray(loc) || /[RL]$/.test(loc)) return parseLoc(loc, off);
  const p = parseLoc(loc, null);
  p[0] = -p[0];
  if (off) { p[0] += off[0]; p[1] += off[1]; p[2] += off[2]; }
  return p;
}

// Fingerspelling: letter -> keyframes (right hand at the "fs" location)
const FS_ORIENT = {
  G: { palm: 'in', fing: 'left' }, H: { palm: 'in', fing: 'left' },
  P: { hs: 'K', palm: 'left+in', fing: 'out+down' }, Q: { hs: 'G', palm: 'left+in', fing: 'down+out' },
};
function letterFrames(ch, i, prev) {
  const drift = Math.min(i * 0.006, 0.05);
  const base = { loc: 'fs', off: [drift, 0, 0], palm: 'out', fing: 'up', contact: 'palm' };
  if (ch === 'J') {
    return [
      { ...base, hs: 'I', d: 170 },
      { hs: 'I', off: [drift - 0.005, -0.035, 0.01], palm: 'left+out', d: 150, ease: 'in' },
      { hs: 'I', off: [drift - 0.035, -0.04, 0.0], palm: 'in', d: 150, ease: 'out', hold: 80 },
    ];
  }
  if (ch === 'Z') {
    const c = 'i';
    return [
      { ...base, hs: 'index', contact: c, off: [drift - 0.03, 0.03, 0], d: 170 },
      { off: [drift + 0.03, 0.03, 0], d: 110, ease: 'lin' },
      { off: [drift - 0.03, -0.03, 0], d: 130, ease: 'lin' },
      { off: [drift + 0.03, -0.03, 0], d: 110, ease: 'out', hold: 60 },
    ];
  }
  const hs = /[0-9]/.test(ch) ? { 0: 'O', 1: '1', 2: 'V', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: 'F' }[ch] : ch;
  const f = { ...base, hs, ...(FS_ORIENT[ch] || {}), d: 170, hold: 150 };
  if (prev === ch) {           // double letter: slide sideways
    return [{ ...f, off: [drift + 0.035, 0.0, 0.0], d: 200, hold: 130 }];
  }
  return [f];
}

export class Timeline {
  constructor(lib, lexicon, probes) {
    this.lib = lib;
    this.lex = lexicon.signs;
    this.probes = probes;           // {r: Hand, l: Hand} never rendered, used for FK
    this.reset();
  }

  reset() {
    this.tracks = { r: [], l: [] };
    this.segments = [];
    this.nmm = [];
    this.duration = 0;
  }

  // keyframe (fully specified) -> pose
  resolve(kf, side, mirrored) {
    const shape = this.lib.get(kf.hs);
    const pos = resolveLoc(kf.loc, kf.off, mirrored);
    const Q = handQuat(kf.palm, kf.fing);
    const probe = this.probes[side];
    probe.root.position.set(0, 0, 0);
    probe.root.quaternion.identity();
    probe.setShape(shape);
    const c = probe.contactLocal(kf.contact || 'palm').applyQuaternion(Q);
    return { W: S(pos).sub(c), Q, shape, wiggle: !!kf.wiggle };
  }

  // expand carry-over + repetitions for one hand of one sign
  expand(frames, rep) {
    let list = frames;
    if (rep && rep > 1 && frames.length > 1) {
      list = [frames[0]];
      for (let k = 0; k < rep; k++) list.push(...frames.slice(1));
    }
    const out = [];
    let cur = {};
    for (const f of list) {
      cur = { ...cur };
      for (const k of CARRY) if (f[k] !== undefined) cur[k] = f[k];
      if (f.loc !== undefined && f.off === undefined) cur.off = null;
      out.push({ ...cur, d: f.d ?? 260, hold: f.hold || 0, ease: f.ease || 'smooth', wiggle: f.wiggle });
    }
    return out;
  }

  push(side, t, pose, ease) {
    this.tracks[side].push({ t, ...pose, ease });
  }

  atRest(side) {
    const tr = this.tracks[side];
    return tr.length && tr[tr.length - 1].rest;
  }

  goRest(side, t, d = 380) {
    if (this.atRest(side)) return;
    const pose = this.resolve(REST[side], side, false);
    const tr = this.tracks[side];
    const dist = tr[tr.length - 1].W.distanceTo(pose.W);
    const start = Math.max(t, tr[tr.length - 1].t);
    this.push(side, start + Math.max(d, (160 + dist * 1000) / this.speed), { ...pose, rest: true }, 'smooth');
  }

  // Lay one hand's frames from time t0; returns end time.
  layHand(side, frames, t0, mirrored, speed, firstMin = 300) {
    const tr = this.tracks[side];
    let t = Math.max(t0, tr.length ? tr[tr.length - 1].t : 0);
    frames.forEach((f, k) => {
      const pose = this.resolve(f, side, mirrored);
      let d = f.d;
      if (k === 0) {
        // transition into the sign: duration grows with distance so the
        // hand peaks at ~1.3 m/s (natural signing) instead of teleporting
        const tr = this.tracks[side];
        const dist = tr.length ? tr[tr.length - 1].W.distanceTo(pose.W) : 0;
        d = Math.max(f.d, firstMin, firstMin ? 160 + dist * 1000 : 0);
      }
      d /= speed;
      t += d;
      this.push(side, t, pose, f.ease);
      if (f.hold) { t += f.hold / speed; this.push(side, t, pose, 'smooth'); }
    });
    return t;
  }

  // Captured clip: frames of [W(3), Q(4), index/middle/ring/pinky (4 each), thumb (5)]
  // in world space, null where the hand was not visible (it holds its pose).
  layClip(side, frames, fps, t0, speed) {
    const tr = this.tracks[side];
    const first = frames.findIndex(Boolean);
    if (first < 0) { this.goRest(side, t0, 380 / speed); return t0; }
    const pose = (f) => ({
      W: new THREE.Vector3(f[0], f[1], f[2]),
      Q: new THREE.Quaternion(f[3], f[4], f[5], f[6]),
      shape: { i: f.slice(7, 11), m: f.slice(11, 15), r: f.slice(15, 19), p: f.slice(19, 23), t: f.slice(23, 28) },
      wiggle: false,
    });
    const dt = 1000 / fps / speed;
    const p0 = pose(frames[first]);
    const from = tr.length ? tr[tr.length - 1] : null;
    const travel = from ? (160 + from.W.distanceTo(p0.W) * 1000) / speed : 300 / speed;
    const start = Math.max(t0, from ? from.t : 0) + Math.max(travel, 300 / speed);
    let end = start;
    frames.forEach((f, k) => {
      if (!f || k < first) return;
      end = start + (k - first) * dt;
      this.push(side, end, pose(f), k === first ? 'smooth' : 'lin');
    });
    return end;
  }

  // plan: {items: [{kind: 'sign'|'fs'|'pause', id, word, gloss}], nmm: [{type, from, to}]}
  compile(plan, speed = 1) {
    this.reset();
    this.speed = speed;
    for (const side of ['r', 'l']) this.push(side, 0, { ...this.resolve(REST[side], side, false), rest: true }, 'smooth');
    let t = 120;
    plan.items.forEach((item, idx) => {
      const t0 = t;
      if (item.kind === 'pause') {
        this.goRest('r', t, 420 / speed);
        this.goRest('l', t, 420 / speed);
        const lastR = this.tracks.r[this.tracks.r.length - 1].t, lastL = this.tracks.l[this.tracks.l.length - 1].t;
        t = Math.max(t + 560 / speed, lastR + 140 / speed, lastL + 140 / speed);
        return;
      }
      let ends = [];
      let letters = null;
      if (item.kind === 'fs') {
        // letters are laid one at a time so the caption can follow along
        letters = [];
        let tt = t;
        [...item.word.toUpperCase()].filter((c) => /[A-Z0-9]/.test(c)).forEach((ch, i, arr) => {
          letters.push({ ch, t: tt });
          tt = this.layHand('r', this.expand(letterFrames(ch, i, arr[i - 1])), tt, false, speed, i ? 0 : 300);
        });
        ends.push(tt);
        this.goRest('l', t, 380 / speed);
      } else {
        const sign = this.lex[item.id];
        if (!sign) throw new Error(`sign "${item.id}" missing from lexicon`);
        if (sign.clip) {
          // captured from a real signer: play the recorded performance
          ends.push(this.layClip('r', sign.r, sign.fps, t, speed));
          if (sign.l) ends.push(this.layClip('l', sign.l, sign.fps, t, speed));
          else this.goRest('l', t, 380 / speed);
          const t1 = Math.max(...ends);
          for (const side of ['r', 'l']) {
            const tr = this.tracks[side], last = tr[tr.length - 1];
            if (last.t < t1) tr.push({ ...last, t: t1, ease: 'smooth' });
          }
          t = t1 + 90 / speed;
          this.segments.push({ idx, t0, t1, gloss: item.gloss, letters: null });
          return;
        }
        const rFrames = this.expand(sign.r, sign.rep);
        ends.push(this.layHand('r', rFrames, t, false, speed));
        if (sign.l) {
          const mirrored = sign.l === 'mirror';
          const lFrames = mirrored ? this.expand(mirrorFrames(sign.r), sign.rep) : this.expand(sign.l, sign.lrep ?? sign.rep);
          ends.push(this.layHand('l', lFrames, t, mirrored, speed));
        } else {
          this.goRest('l', t, 380 / speed);
        }
      }
      const t1 = Math.max(...ends);
      // the shorter hand holds its last pose until the sign ends
      for (const side of ['r', 'l']) {
        const tr = this.tracks[side], last = tr[tr.length - 1];
        if (last.t < t1) tr.push({ ...last, t: t1, ease: 'smooth' });
      }
      t = t1 + 90 / speed;
      this.segments.push({ idx, t0, t1, gloss: item.gloss, letters });
    });
    this.goRest('r', t, 450 / speed);
    this.goRest('l', t, 450 / speed);
    this.duration = t + 600 / speed;
    for (const span of plan.nmm || []) {
      const a = this.segments.find((s) => s.idx >= span.from);
      const b = [...this.segments].reverse().find((s) => s.idx <= span.to);
      if (a && b) this.nmm.push({ type: span.type, t0: a.t0, t1: b.t1 });
    }
    return this;
  }

  // ------------------------------------------------------------ sampling
  sample(side, t) {
    const tr = this.tracks[side];
    if (t <= tr[0].t) return tr[0];
    if (t >= tr[tr.length - 1].t) return tr[tr.length - 1];
    let lo = 0, hi = tr.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (tr[mid].t <= t) lo = mid; else hi = mid; }
    const A = tr[lo], B = tr[hi];
    const u = (EASE[B.ease] || EASE.smooth)((t - A.t) / Math.max(B.t - A.t, 1e-6));
    const W = hermite(tr, lo, u);
    const Q = A.Q.clone().slerp(B.Q, u);
    const shape = lerpShape(A.shape, B.shape, u);
    const wiggle = B.wiggle || (A.wiggle && u < 1) ? t * 0.022 : 0;
    return { W, Q, shape, wiggle };
  }

  face(t) {
    let brow = 0, shake = 0, nod = 0;
    for (const s of this.nmm) {
      const ramp = 160;
      const k = Math.max(0, Math.min(1, (t - s.t0 + ramp) / ramp, (s.t1 + ramp - t) / ramp));
      if (k <= 0) continue;
      if (s.type === 'wh') brow = -k;
      if (s.type === 'yn') { brow = k; nod = 0.09 * k; }
      if (s.type === 'neg') shake = Math.sin(t * 0.016) * 0.16 * k;
    }
    return { brow, shake, nod };
  }

  activeSegment(t) {
    return this.segments.find((s) => t >= s.t0 && t <= s.t1 + 90) || null;
  }
}

// Catmull-Rom style tangents for non-uniform key times: velocity estimate
// v_k = (P[k+1] - P[k-1]) / (t[k+1] - t[k-1]), scaled by the segment duration
// so short segments next to long ones do not overshoot. Zeroed at holds so
// paused hands do not drift.
function velocity(tr, k) {
  const P = tr[k].W, prev = tr[k - 1], next = tr[k + 1];
  if (!prev || !next) return new THREE.Vector3();
  if (P.distanceToSquared(prev.W) < 1e-8 || P.distanceToSquared(next.W) < 1e-8) return new THREE.Vector3();
  return new THREE.Vector3().subVectors(next.W, prev.W).divideScalar(Math.max(next.t - prev.t, 1e-6));
}
function hermite(tr, k, u) {
  const P0 = tr[k].W, P1 = tr[k + 1].W;
  const dt = tr[k + 1].t - tr[k].t;
  const m0 = velocity(tr, k).multiplyScalar(dt), m1 = velocity(tr, k + 1).multiplyScalar(dt);
  const u2 = u * u, u3 = u2 * u;
  return new THREE.Vector3()
    .addScaledVector(P0, 2 * u3 - 3 * u2 + 1)
    .addScaledVector(m0, u3 - 2 * u2 + u)
    .addScaledVector(P1, -2 * u3 + 3 * u2)
    .addScaledVector(m1, u3 - u2);
}
