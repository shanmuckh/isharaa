// Retarget real signing (ASL Citizen landmarks) onto the avatar.
//
//   uv run python src/signs/export_motion.py      (world-space hand points per word)
//   node static/text2sign/tools/retarget.mjs      -> src/text2sign/data/clips.json
//
// Per frame and hand: wrist position, hand rotation from the palm (wrist,
// index MCP, middle MCP, pinky MCP), finger flexion/abduction from bone
// angles, and the thumb from the contact IK used for handshapes (target = the
// tracked thumb tip, warm-started from the previous frame for smooth motion).
// Frame record: [Wx,Wy,Wz, Qx,Qy,Qz,Qw, i0..i3, m0..m3, r0..r3, p0..p3, t0..t4]
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import * as THREE from '../../vendor/three.module.min.js';
import { HandshapeLibrary } from '../avatar.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, '..', '..', '..');
const motion = JSON.parse(fs.readFileSync(path.join(root, 'data', 'asl_citizen_200', 'motion_points.json'), 'utf8'));
const lib = new HandshapeLibrary(JSON.parse(fs.readFileSync(path.join(root, 'src', 'text2sign', 'data', 'handshapes.json'), 'utf8')));
const OUR_HAND = 0.0914;                                   // avatar wrist -> middle MCP (m)
const FINGERS = { i: [5, 6, 7, 8], m: [9, 10, 11, 12], r: [13, 14, 15, 16], p: [17, 18, 19, 20] };
const DEG = Math.PI / 180;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const V = (p) => new THREE.Vector3(...p);
const r4 = (x) => Math.round(x * 1e4) / 1e4;

function frameToPose(pts, side, seedThumb) {
  const P = pts.map(V);
  const Y = P[9].clone().sub(P[0]).normalize();
  const across = P[5].clone().sub(P[17]);                  // towards the thumb side
  const Z = across.clone().cross(Y).normalize().multiplyScalar(side);   // palm normal
  const X = Y.clone().cross(Z).normalize();
  const Q = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(X, Y, Z));
  const local = (v) => [v.dot(X), v.dot(Y), v.dot(Z)];
  const shape = {};
  for (const [f, [a, b, c, d]] of Object.entries(FINGERS)) {
    const b1 = P[b].clone().sub(P[a]), b2 = P[c].clone().sub(P[b]), b3 = P[d].clone().sub(P[c]);
    const [x, y, z] = local(b1.clone().normalize());
    shape[f] = [
      clamp(Math.atan2(z, y), -0.2, 1.7),                  // MCP flexion (towards the palm)
      clamp(b1.angleTo(b2), 0, 1.9),                       // PIP
      clamp(b2.angleTo(b3), 0, 1.5),                       // DIP
      clamp(Math.atan2(side * x, y), -25 * DEG, 25 * DEG), // abduction towards the thumb
    ];
  }
  // thumb: solve on the (right-handed) probe, mirroring the target for the left hand
  const scale = OUR_HAND / P[9].distanceTo(P[0]);
  const tip = local(P[4].clone().sub(P[0])).map((v) => v * scale);
  tip[0] *= side;
  shape.t = seedThumb.slice();
  shape.t = lib.solveThumb(shape, { at: tip, gap: 0 });
  return { W: P[0], Q, shape };
}

const out = {};
const soft = lib.get('relaxed').t;
for (const [gloss, m] of Object.entries(motion)) {
  const clip = { word: m.word, fps: m.fps, source: m.source };
  for (const [key, side] of [['r', 1], ['l', -1]]) {
    let seed = soft;
    const frames = m[key].map((pts) => {
      if (!pts) return null;
      const { W, Q, shape } = frameToPose(pts, side, seed);
      seed = shape.t;
      return [W.x, W.y, W.z, Q.x, Q.y, Q.z, Q.w, ...shape.i, ...shape.m, ...shape.r, ...shape.p, ...shape.t].map(r4);
    });
    clip[key] = frames.some(Boolean) ? frames : null;
  }
  out[gloss] = clip;
}
const dest = path.join(root, 'src', 'text2sign', 'data', 'clips.json');
fs.writeFileSync(dest, JSON.stringify({ _doc: 'Signs retargeted from ASL Citizen (CC BY-NC-SA 4.0) by static/text2sign/tools/retarget.mjs. Frame: [Wx,Wy,Wz,Qx,Qy,Qz,Qw, index(mcp,pip,dip,abd), middle, ring, pinky, thumb(yaw,abd,twist,mcp,ip)] in avatar world space; null = hand not visible.', clips: out }));
console.log(`retargeted ${Object.keys(out).length} signs -> ${path.relative(root, dest)} (${(fs.statSync(dest).size / 1e6).toFixed(1)} MB)`);
