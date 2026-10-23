// Kinematic validation of the sign lexicon (runs the real renderer code in
// node, no browser needed):   node static/text2sign/tools/validate.mjs
//
// For every sign: resolves all keyframes (handshape, location, orientation,
// contact point), compiles it into a timeline and checks that
//   - palm and finger directions are not (anti)parallel (degenerate rotation)
//   - every wrist target is within arm reach of its shoulder
//   - sampled motion has no NaNs and wrist speed stays below a sane limit
import { fileURLToPath } from 'url';
import fs from 'fs';
import path from 'path';
import * as THREE from '../../vendor/three.module.min.js';
import { Hand, HandshapeLibrary, BODY, S, parseDir } from '../avatar.js';
import { Timeline } from '../timeline.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.join(here, '..', '..', '..', 'src', 'text2sign', 'data');
const read = (f) => JSON.parse(fs.readFileSync(path.join(dataDir, f), 'utf8'));
const lex = read('lexicon.json');
const lib = new HandshapeLibrary(read('handshapes.json'));
const mat = new THREE.MeshBasicMaterial();
const tl = new Timeline(lib, lex, { r: new Hand(1, mat), l: new Hand(-1, mat) });
const reach = BODY.upperArm + BODY.foreArm;
const shoulder = { r: S(BODY.shoulderR), l: S(BODY.shoulderL) };
const MAX_SPEED = 2.6; // m/s

const problems = [];
for (const [id, sign] of Object.entries(lex.signs)) {
  try {
    for (const side of ['r', 'l']) {
      const frames = sign[side];
      if (!Array.isArray(frames)) continue;
      let palm, fing;
      for (const f of frames) {
        palm = f.palm ?? palm; fing = f.fing ?? fing;
        if (Math.abs(parseDir(palm).dot(parseDir(fing))) > 0.95) problems.push(`${id}/${side}: palm "${palm}" ~ parallel to fingers "${fing}"`);
      }
    }
    tl.compile({ items: [{ kind: 'sign', id, gloss: id }], nmm: [] });
    for (const side of ['r', 'l']) {
      for (const k of tl.tracks[side]) {
        const d = k.W.distanceTo(shoulder[side]);
        if (d > reach - 0.005) { problems.push(`${id}/${side}: target ${d.toFixed(3)} m from shoulder (reach ${reach.toFixed(2)})`); break; }
      }
      let prev = null;
      for (let t = 0; t <= tl.duration; t += 1000 / 60) {
        const W = tl.sample(side, t).W;
        if (![W.x, W.y, W.z].every(Number.isFinite)) { problems.push(`${id}/${side}: NaN at ${t | 0} ms`); break; }
        if (prev && W.distanceTo(prev) * 60 > MAX_SPEED) { problems.push(`${id}/${side}: wrist speed ${(W.distanceTo(prev) * 60).toFixed(2)} m/s at ${t | 0} ms`); break; }
        prev = W;
      }
    }
  } catch (e) {
    problems.push(`${id}: ${e.message}`);
  }
}
tl.compile({ items: [{ kind: 'fs', word: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', gloss: 'fs' }], nmm: [] });

const n = Object.keys(lex.signs).length;
if (problems.length) {
  console.log(problems.join('\n'));
  console.log(`\n${problems.length} problem(s) in ${n} signs`);
  process.exit(1);
}
console.log(`OK: ${n} signs, ${lib.names().length} handshapes, full fingerspelling alphabet compile cleanly`);
