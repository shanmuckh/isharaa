// Procedural signing avatar for Isharaa Text -> Sign.
//
// Coordinate conventions
//   Authoring ("signer") frame: +x = signer's right, +y = up, +z = forward
//   (towards the viewer). three.js world: the avatar faces +Z, so the
//   signer's right is world -X. `S()` converts authoring -> world.
//
//   Hand local frame (both hands): +Y = finger direction, +Z = palm normal
//   (the side the palm faces), +X = Y x Z. For the right hand the thumb sits
//   on +X, for the left hand on -X (geometry is mirrored, frame is not).
import * as THREE from '../vendor/three.module.min.js';

export const S = (v) => new THREE.Vector3(-v[0], v[1], v[2]);
const DEG = Math.PI / 180;

// ---------------------------------------------------------------- body
export const BODY = {
  shoulderR: [0.175, 1.425, 0.0],
  shoulderL: [-0.175, 1.425, 0.0],
  upperArm: 0.285,
  foreArm: 0.255,
  headCenter: [0, 1.645, 0.0],
  headRadius: 0.102,
};

// Named signing locations (authoring frame). Face/torso locations sit just
// in front of the surface so contact reads as a touch, not a collision.
export const LOCATIONS = {
  forehead:  [0.0, 1.695, 0.122],
  temple:    [0.085, 1.685, 0.085],
  eyes:      [0.03, 1.665, 0.14],
  nose:      [0.0, 1.635, 0.142],
  mouth:     [0.0, 1.595, 0.13],
  chin:      [0.0, 1.555, 0.118],
  cheek:     [0.078, 1.61, 0.105],
  ear:       [0.118, 1.645, 0.02],
  neck:      [0.0, 1.49, 0.075],
  chest:     [0.0, 1.33, 0.135],
  chestR:    [0.075, 1.33, 0.135],
  chestL:    [-0.075, 1.33, 0.135],
  shoulderR: [0.16, 1.40, 0.085],
  shoulderL: [-0.16, 1.40, 0.085],
  stomach:   [0.0, 1.16, 0.15],
  neutral:   [0.06, 1.25, 0.32],
  centre:    [0.0, 1.25, 0.32],
  base:      [-0.02, 1.215, 0.30],   // where the non-dominant hand waits
  fs:        [0.16, 1.37, 0.30],     // fingerspelling space
  sideR:     [0.30, 1.25, 0.24],
  sideL:     [-0.30, 1.25, 0.24],
  restR:     [0.11, 1.00, 0.20],
  restL:     [-0.11, 1.00, 0.20],
};

export const DIRS = {
  up: [0, 1, 0], down: [0, -1, 0], out: [0, 0, 1], in: [0, 0, -1],
  right: [1, 0, 0], left: [-1, 0, 0],
};

// "out+up" -> normalised vector; arrays pass through.
export function parseDir(d) {
  if (Array.isArray(d)) return new THREE.Vector3(...d).normalize();
  const v = new THREE.Vector3();
  for (const part of d.split('+')) {
    const base = DIRS[part.trim()];
    if (!base) throw new Error(`unknown direction "${part}"`);
    v.add(new THREE.Vector3(...base));
  }
  return v.normalize();
}

export function parseLoc(loc, off) {
  const src = Array.isArray(loc) ? loc : LOCATIONS[loc];
  if (!src) throw new Error(`unknown location "${loc}"`);
  const p = src.slice();
  if (off) { p[0] += off[0]; p[1] += off[1]; p[2] += off[2]; }
  return p;
}

// Build the world-space quaternion of a hand from authoring palm/finger dirs.
const _m = new THREE.Matrix4();
export function handQuat(palm, fing) {
  const z = S(parseDir(palm).toArray()).normalize();
  let y = S(parseDir(fing).toArray());
  y.sub(z.clone().multiplyScalar(y.dot(z)));              // Gram-Schmidt
  if (y.lengthSq() < 1e-6) y = Math.abs(z.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(0, 0, -1);
  y.normalize();
  const x = new THREE.Vector3().crossVectors(y, z).normalize();
  _m.makeBasis(x, y, z);
  return new THREE.Quaternion().setFromRotationMatrix(_m);
}

// ---------------------------------------------------------------- handshapes
const THENAR_MCP = new THREE.Vector3(0.038, 0.038, 0.016);

export class HandshapeLibrary {
  constructor(json) {
    this.fp = json.finger_presets;
    this.tp = json.thumb_presets;
    this.shapes = json.shapes;
    this.cache = new Map();
  }
  has(name) { return name in this.shapes; }
  names() { return Object.keys(this.shapes); }
  // -> {i:[mcp,pip,dip,abd], m, r, p, t:[yaw,abd,twist,mcp,ip]} in radians
  get(name) {
    if (this.cache.has(name)) return this.cache.get(name);
    const s = this.shapes[name];
    if (!s) throw new Error(`unknown handshape "${name}"`);
    const fan = { i: 1.5, m: 0.5, r: -0.5, p: -1.5 };
    const out = {};
    for (const f of ['i', 'm', 'r', 'p']) {
      const raw = typeof s[f] === 'string' ? this.fp[s[f]] : s[f];
      const a = [raw[0], raw[1], raw[2], (raw[3] || 0) + (s.sp || 0) * fan[f]];
      out[f] = a.map((x) => x * DEG);
    }
    const t = typeof s.t === 'string' ? this.tp[s.t] : s.t;
    out.t = t.map((x) => x * DEG);
    if (s.tt) out.t = this.solveThumb(out, s.tt);
    this.cache.set(name, out);
    return out;
  }

  // Contact-constrained thumb: find thumb angles that put the thumb tip on a
  // target (e.g. "i.tip" for O/F, "m.dip" for S) without passing through the
  // palm. Gradient-free pattern search over the 5 thumb DOF, seeded by the
  // preset so the solution stays in the anatomically plausible basin.
  solveThumb(shape, spec) {
    const hand = this.probe || (this.probe = new Hand(1, new THREE.MeshBasicMaterial()));
    hand.setShape(shape);
    hand.root.updateMatrixWorld(true);
    const pt = (o) => o.getWorldPosition(new THREE.Vector3());
    const ats = Array.isArray(spec.at) && typeof spec.at[0] === 'string' ? spec.at : [spec.at];
    const target = new THREE.Vector3();
    for (const at of ats) {
      if (Array.isArray(at)) { target.add(new THREE.Vector3(...at)); continue; }
      const [f, j] = at.split('.');
      const o = j === 'tip' ? hand.tips[f] : hand.joints[f][{ mcp: 0, pip: 1, dip: 2 }[j]];
      target.add(pt(o));
    }
    target.multiplyScalar(1 / ats.length);
    if (spec.off) target.add(new THREE.Vector3(...spec.off));
    const gap = spec.gap ?? (typeof ats[0] === 'string' && ats[0].endsWith('tip') ? 0.016 : 0);
    const seed = shape.t.slice();
    const lo = [-80, -5, -115, -10, -10].map((x) => x * DEG);
    const hi = [60, 90, -20, 80, 90].map((x) => x * DEG);
    const probeShape = { ...shape };
    const cost = (t) => {
      probeShape.t = t;
      hand.setShape(probeShape);
      hand.root.updateMatrixWorld(true);
      let c = (pt(hand.tips.t).distanceTo(target) - gap) ** 2 * 1e4;
      for (const o of [hand.joints.t[1], hand.joints.t[2], hand.tips.t]) {
        const p = pt(o);
        if (Math.abs(p.x) < 0.04 && p.y > 0.0 && p.y < 0.095) {
          const depth = 0.024 - p.z;
          if (depth > 0) c += depth * depth * 4e4;
        }
      }
      // the thumb metacarpal lives inside the thenar eminence: keep the MCP
      // joint close to the radial edge of the palm
      const mcpOff = pt(hand.joints.t[1]).distanceTo(THENAR_MCP) - 0.012;
      if (mcpOff > 0) c += mcpOff * mcpOff * 2e4;
      for (let j = 0; j < 5; j++) c += 0.02 * (t[j] - seed[j]) ** 2;
      return c;
    };
    let best = seed.slice(), bestC = cost(best);
    for (let step = 12 * DEG; step > 0.4 * DEG; ) {
      let improved = false;
      for (let j = 0; j < 5; j++) {
        for (const dir of [1, -1]) {
          const cand = best.slice();
          cand[j] = THREE.MathUtils.clamp(cand[j] + dir * step, lo[j], hi[j]);
          const c = cost(cand);
          if (c < bestC - 1e-9) { best = cand; bestC = c; improved = true; }
        }
      }
      if (!improved) step /= 2;
    }
    hand.setShape(shape);
    return best;
  }
}

export function lerpShape(a, b, u) {
  const o = {};
  for (const k of ['i', 'm', 'r', 'p', 't']) o[k] = a[k].map((x, j) => x + (b[k][j] - x) * u);
  return o;
}

// ---------------------------------------------------------------- hand
const FINGERS = {
  i: { x: 0.0280, base: 0.088, len: [0.042, 0.025, 0.019], r: 0.0099 },
  m: { x: 0.0090, base: 0.091, len: [0.046, 0.028, 0.020], r: 0.0101 },
  r: { x: -0.0100, base: 0.088, len: [0.043, 0.026, 0.019], r: 0.0097 },
  p: { x: -0.0280, base: 0.081, len: [0.034, 0.020, 0.017], r: 0.0086 },
};
const THUMB = { pos: [0.028, 0.012, 0.006], len: [0.038, 0.030, 0.024], r: 0.0108 };
const PALM = { w: 0.080, l: 0.090, t: 0.026 };

function capsule(len, r, mat) {
  const g = new THREE.CapsuleGeometry(r, Math.max(len - 2 * r * 0.6, 0.001), 4, 10);
  const mesh = new THREE.Mesh(g, mat);
  mesh.position.y = len / 2;
  mesh.castShadow = true;
  return mesh;
}

function roundedPalm(mat) {
  const { w, l, t } = PALM;
  const r = 0.016, b = 0.006;
  const s = new THREE.Shape();
  const x0 = -w / 2 + b, x1 = w / 2 - b, y0 = b, y1 = l - b;
  s.moveTo(x0 + r, y0);
  s.lineTo(x1 - r, y0); s.quadraticCurveTo(x1, y0, x1, y0 + r);
  s.lineTo(x1, y1 - r * 0.5); s.quadraticCurveTo(x1, y1, x1 - r * 0.5, y1);
  s.lineTo(x0 + r * 0.5, y1); s.quadraticCurveTo(x0, y1, x0, y1 - r * 0.5);
  s.lineTo(x0, y0 + r); s.quadraticCurveTo(x0, y0, x0 + r, y0);
  const g = new THREE.ExtrudeGeometry(s, {
    depth: t - 2 * b, bevelEnabled: true, bevelThickness: b, bevelSize: b, bevelSegments: 3, curveSegments: 6,
  });
  g.translate(0, 0, -(t - 2 * b) / 2);
  const mesh = new THREE.Mesh(g, mat);
  mesh.castShadow = true;
  return mesh;
}

export class Hand {
  // side: +1 right hand, -1 left hand
  constructor(side, mat) {
    this.side = side;
    this.root = new THREE.Group();
    this.root.add(roundedPalm(mat));
    this.joints = {};
    this.tips = {};
    for (const [f, d] of Object.entries(FINGERS)) {
      const chain = [];
      let parent = new THREE.Group();
      parent.position.set(d.x * side, d.base, 0);
      parent.rotation.order = 'ZXY';
      this.root.add(parent);
      chain.push(parent);
      d.len.forEach((L, k) => {
        parent.add(capsule(L, d.r * (1 - k * 0.09), mat));
        if (k < 2) {
          const j = new THREE.Group();
          j.position.y = L;
          parent.add(j);
          chain.push(j);
          parent = j;
        } else {
          const tip = new THREE.Object3D();
          tip.position.y = L - d.r * 0.4;
          parent.add(tip);
          this.tips[f] = tip;
        }
      });
      this.joints[f] = chain;
    }
    // Thumb: CMC (yaw/abduction/twist) -> MCP -> IP
    const cmc = new THREE.Group();
    cmc.position.set(THUMB.pos[0] * side, THUMB.pos[1], THUMB.pos[2]);
    cmc.rotation.order = 'ZXY';
    this.root.add(cmc);
    const tchain = [cmc];
    let parent = cmc;
    THUMB.len.forEach((L, k) => {
      parent.add(capsule(L, THUMB.r * (1 - k * 0.1), mat));
      if (k < 2) {
        const j = new THREE.Group();
        j.position.y = L;
        parent.add(j);
        tchain.push(j);
        parent = j;
      } else {
        const tip = new THREE.Object3D();
        tip.position.y = L - THUMB.r * 0.4;
        parent.add(tip);
        this.tips.t = tip;
      }
    });
    // Thenar pad so the thumb root does not look detached
    const pad = new THREE.Mesh(new THREE.SphereGeometry(0.019, 12, 10), mat);
    pad.scale.set(1, 1.5, 0.75);
    pad.position.set(0.019 * side, 0.026, 0.004);
    this.root.add(pad);
    this.joints.t = tchain;
    this.palmPoint = new THREE.Object3D();
    this.palmPoint.position.set(0, 0.048, PALM.t / 2);
    this.backPoint = new THREE.Object3D();
    this.backPoint.position.set(0, 0.048, -PALM.t / 2);
    this.root.add(this.palmPoint, this.backPoint);
  }

  // shape: output of HandshapeLibrary.get / lerpShape
  setShape(shape, wiggle = 0) {
    const s = this.side;
    ['i', 'm', 'r', 'p'].forEach((f, n) => {
      const a = shape[f], ch = this.joints[f];
      const w = wiggle ? Math.sin(wiggle + n * 1.3) * 22 * DEG : 0;
      ch[0].rotation.set(Math.max(a[0] + w, -0.2), 0, -a[3] * s);
      ch[1].rotation.set(a[1] + Math.max(w, 0) * 0.6, 0, 0);
      ch[2].rotation.set(a[2], 0, 0);
    });
    const t = shape.t, ch = this.joints.t;
    ch[0].rotation.set(t[1], t[2] * s, t[0] * s);
    ch[1].rotation.set(t[3], 0, 0);
    ch[2].rotation.set(t[4], 0, 0);
  }

  // Point of the hand used for contact, in hand-local coordinates for the
  // currently applied shape. kind: palm | back | tips | i | m | r | p | t | wrist
  contactLocal(kind) {
    this.root.updateMatrixWorld(true);
    const inv = this.root.matrixWorld.clone().invert();
    const get = (o) => o.getWorldPosition(new THREE.Vector3()).applyMatrix4(inv);
    if (kind === 'wrist') return new THREE.Vector3();
    if (kind === 'palm') return get(this.palmPoint);
    if (kind === 'back') return get(this.backPoint);
    if (kind === 'tips') return get(this.tips.i).add(get(this.tips.m)).multiplyScalar(0.5);
    if (this.tips[kind]) return get(this.tips[kind]);
    throw new Error(`unknown contact "${kind}"`);
  }
}

// ---------------------------------------------------------------- arm IK
// Analytic two-bone IK. Returns {E, W} world positions (W may be pulled in if
// the target is out of reach).
export function solveArm(Sh, T, a, b, pole) {
  const dir = T.clone().sub(Sh);
  let d = dir.length();
  dir.normalize();
  d = THREE.MathUtils.clamp(d, Math.abs(a - b) + 1e-3, a + b - 1e-3);
  const cosA = (a * a + d * d - b * b) / (2 * a * d);
  const sinA = Math.sqrt(Math.max(0, 1 - cosA * cosA));
  let p = pole.clone().sub(dir.clone().multiplyScalar(pole.dot(dir)));
  if (p.lengthSq() < 1e-6) p = new THREE.Vector3(0, 0, -1);
  p.normalize();
  const E = Sh.clone().add(dir.clone().multiplyScalar(a * cosA)).add(p.multiplyScalar(a * sinA));
  const W = Sh.clone().add(dir.multiplyScalar(d));
  return { E, W };
}

const UP = new THREE.Vector3(0, 1, 0);
function placeBone(mesh, A, B) {
  mesh.position.copy(A).add(B).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(UP, B.clone().sub(A).normalize());
}

// ---------------------------------------------------------------- avatar
export class Avatar {
  constructor(lib) {
    this.lib = lib;
    this.group = new THREE.Group();
    const skin = new THREE.MeshStandardMaterial({ color: 0xe6e3de, roughness: 0.55, metalness: 0.0 });
    const cloth = new THREE.MeshStandardMaterial({ color: 0x2b2b2e, roughness: 0.85 });
    const dark = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.4 });
    this.mats = { skin, cloth, dark };

    // torso (dark clothing gives the hands contrast, as interpreters do)
    const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.15, 0.36, 8, 20), cloth);
    torso.scale.set(1.18, 1, 0.68);
    torso.position.set(0, 1.18, 0);
    const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.05, 0.12, 16), skin);
    neck.position.set(0, 1.50, 0);
    this.group.add(torso, neck);
    for (const sh of [BODY.shoulderR, BODY.shoulderL]) {
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.062, 16, 12), cloth);
      m.position.copy(S(sh));
      this.group.add(m);
    }

    // head with simple features for non-manual markers
    this.head = new THREE.Group();
    this.head.position.copy(S([0, 1.56, 0]));
    const hc = S(BODY.headCenter).sub(this.head.position);
    const skull = new THREE.Mesh(new THREE.SphereGeometry(BODY.headRadius, 28, 22), skin);
    skull.scale.set(0.92, 1.12, 1.0);
    skull.position.copy(hc);
    this.head.add(skull);
    const hair = new THREE.Mesh(new THREE.SphereGeometry(BODY.headRadius * 1.03, 28, 16, 0, Math.PI * 2, 0, Math.PI * 0.42), dark);
    hair.scale.set(0.93, 1.12, 1.02);
    hair.position.copy(hc).add(new THREE.Vector3(0, 0.006, -0.008));
    hair.rotation.x = -0.25;
    this.head.add(hair);
    this.brows = [];
    for (const sx of [1, -1]) {
      const eye = new THREE.Mesh(new THREE.SphereGeometry(0.0085, 10, 8), dark);
      eye.position.copy(hc).add(new THREE.Vector3(0.034 * sx, 0.018, 0.093));
      const brow = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.0045, 0.006), dark);
      brow.position.copy(hc).add(new THREE.Vector3(0.034 * sx, 0.042, 0.094));
      brow.userData = { base: brow.position.clone(), sx };
      this.head.add(eye, brow);
      this.brows.push(brow);
    }
    const nose = new THREE.Mesh(new THREE.SphereGeometry(0.011, 10, 8), skin);
    nose.scale.set(0.8, 1.2, 1);
    nose.position.copy(hc).add(new THREE.Vector3(0, -0.006, 0.104));
    this.mouth = new THREE.Mesh(new THREE.BoxGeometry(0.034, 0.004, 0.004), dark);
    this.mouth.position.copy(hc).add(new THREE.Vector3(0, -0.044, 0.093));
    this.head.add(nose, this.mouth);
    this.group.add(this.head);

    // arms + hands
    this.arms = {};
    for (const [key, side, sh] of [['r', 1, BODY.shoulderR], ['l', -1, BODY.shoulderL]]) {
      const upper = new THREE.Mesh(new THREE.CapsuleGeometry(0.043, BODY.upperArm - 0.05, 6, 14), cloth);
      const fore = new THREE.Mesh(new THREE.CapsuleGeometry(0.033, BODY.foreArm - 0.04, 6, 14), cloth);
      const cuff = new THREE.Mesh(new THREE.CylinderGeometry(0.026, 0.028, 0.03, 14), skin);
      const elbow = new THREE.Mesh(new THREE.SphereGeometry(0.04, 12, 10), cloth);
      const hand = new Hand(side, skin);
      upper.castShadow = fore.castShadow = true;
      this.group.add(upper, fore, cuff, elbow, hand.root);
      this.arms[key] = { side, shoulder: S(sh), upper, fore, cuff, elbow, hand };
    }
  }

  // pose = {W: Vector3 (world wrist target), Q: Quaternion, shape, wiggle}
  setArm(key, pose) {
    const arm = this.arms[key];
    const s = arm.side;
    // elbow pole: low signing space -> elbow down, out and slightly back;
    // face height -> elbow drops in front of the chest (as signers do) rather
    // than winging out to the side
    const lift = THREE.MathUtils.clamp((pose.W.y - 1.2) / 0.4, 0, 1);
    const pole = S([s * (0.45 - 0.1 * lift), -1, -0.35 + 0.5 * lift]).normalize();
    const { E, W } = solveArm(arm.shoulder, pose.W, BODY.upperArm, BODY.foreArm, pole);
    placeBone(arm.upper, arm.shoulder, E);
    placeBone(arm.fore, E, W);
    arm.elbow.position.copy(E);
    const back = new THREE.Vector3(0, -0.012, 0).applyQuaternion(pose.Q);
    arm.cuff.position.copy(W).add(back);
    arm.cuff.quaternion.copy(pose.Q);
    arm.hand.root.position.copy(W);
    arm.hand.root.quaternion.copy(pose.Q);
    arm.hand.setShape(pose.shape, pose.wiggle || 0);
  }

  // nmm = {brow: -1..1 (furrow..raise), shake: rad, nod: rad, tilt: rad}
  setFace(nmm) {
    const b = nmm.brow || 0;
    for (const brow of this.brows) {
      brow.position.copy(brow.userData.base);
      brow.position.y += b * 0.0075;
      brow.rotation.z = b < 0 ? -b * 0.35 * brow.userData.sx : 0;
    }
    this.head.rotation.set((nmm.nod || 0) + (nmm.tilt || 0), nmm.shake || 0, 0);
  }
}

// ---------------------------------------------------------------- stage
export function createStage(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.shadowMap.enabled = true;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(26, 1, 0.05, 20);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x303034, 1.1));
  const key = new THREE.DirectionalLight(0xffffff, 2.0);
  key.position.set(-1.2, 2.6, 2.4);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  key.shadow.camera.left = -0.8; key.shadow.camera.right = 0.8;
  key.shadow.camera.top = 2.2; key.shadow.camera.bottom = 0.6;
  key.shadow.bias = -0.0005;
  const rim = new THREE.DirectionalLight(0xbfd4ff, 0.9);
  rim.position.set(1.5, 1.8, -1.5);
  scene.add(key, rim);

  const views = {
    front: { pos: [0, 1.42, 2.6], look: [0, 1.29, 0] },
    side:  { pos: [-1.55, 1.52, 1.75], look: [0, 1.37, 0.08] },
    close: { pos: [-0.14, 1.48, 1.30], look: [-0.08, 1.42, 0.12] },
  };
  const stage = {
    renderer, scene, camera,
    setView(name) {
      const v = views[name] || views.front;
      camera.position.set(...v.pos);
      camera.lookAt(...v.look);
    },
    resize() {
      const w = canvas.clientWidth || canvas.width, h = canvas.clientHeight || canvas.height;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    },
    render() { renderer.render(scene, camera); },
  };
  stage.setView('front');
  return stage;
}
