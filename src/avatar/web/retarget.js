// Retargeting: punctele MediaPipe ale unui cadru -> rotațiile oaselor avatarului VRM.
//
// Toate rotațiile sunt cuaternioni „delta” în spațiul scenei: delta × rotația de repaus (T-pose)
// = rotația osului în scenă. Pașii pentru un cadru:
//   1. trunchiul (linia umerilor) și capul (fața);
//   2. ținta încheieturii, calculată față de corp: aproape de față se raportează la față,
//      aproape de piept la piept, altfel la umăr, la scara avatarului (contactul se păstrează
//      chiar dacă avatarul are alte proporții decât semnatarul);
//   3. coliziuni: palma și vârfurile degetelor sunt împinse în afara volumelor corpului
//      (trunchi = elipsă, gât = capsulă, cap = sferă); mâinile nu trec una prin alta, iar
//      când se încrucișează, mâna dreaptă (dominantă) trece prin față;
//   4. IK cu două oase (umăr -> cot -> încheietură), cotul orientat în jos și ușor în afară;
//      antebrațele nu se intersectează; umărul se ridică puțin când brațul urcă;
//   5. orientarea: antebrațul preia răsucirea palmei, încheietura și fiecare falangă au
//      limite anatomice (flexie spre palmă și abducție laterală, fără răsucire).
// Coordonate: X = stânga semnatarului/avatarului, Y = sus, Z = spre privitor (avatarul stă cu fața spre +Z).
import * as THREE from 'three';
import { HANDSHAPES } from './signs.js';

const V3 = THREE.Vector3, Q = THREE.Quaternion;
const { clamp, degToRad } = THREE.MathUtils;

export const SIDES = ['left', 'right'];
// Punctele MediaPipe Hands ale fiecărui deget, de la bază la vârf
export const FINGER_POINTS = { Thumb: [1, 2, 3, 4], Index: [5, 6, 7, 8], Middle: [9, 10, 11, 12],
                               Ring: [13, 14, 15, 16], Little: [17, 18, 19, 20] };
export const FINGERS = Object.keys(FINGER_POINTS);
// Punctele MediaPipe Pose: umăr, cot, încheietură, deget mic, arătător
export const ARM_POINTS = { left: [11, 13, 15, 17, 19], right: [12, 14, 16, 18, 20] };
const EYES = [2, 5];   // ochiul stâng și drept (Pose)

export const fingerBones = (side, finger) => finger === 'Thumb'
  ? ['ThumbMetacarpal', 'ThumbProximal', 'ThumbDistal'].map(s => side + s)
  : ['Proximal', 'Intermediate', 'Distal'].map(s => side + finger + s);
export const BONE_ORDER = ['hips', 'spine', 'chest', 'upperChest', 'neck', 'head', ...SIDES.flatMap(s =>
  [s + 'Shoulder', s + 'UpperArm', s + 'LowerArm', s + 'Hand', ...FINGERS.flatMap(f => fingerBones(s, f))])];
const FK_ORDER = [...BONE_ORDER, 'leftEye', 'rightEye'];

// Limite anatomice, în grade: flexie (spre palmă; negativ = extensie spre dosul palmei)
// și abducție (depărtare laterală). Falangele mijlocii și distale au un singur grad de libertate.
export const LIMITS = {
  mcp: { flex: [-10, 90], abd: [-15, 15] },
  pip: { flex: [0, 100], abd: [0, 0] },
  dip: { flex: [0, 80], abd: [0, 0] },
  thumbCmc: { flex: [-30, 70], abd: [-45, 45] },
  thumbMcp: { flex: [-10, 70], abd: [-10, 10] },
  thumbIp: { flex: [-15, 85], abd: [0, 0] },
  wrist: { flex: [-70, 80], abd: [-30, 30] },
  // Răsucirea antebrațului față de cot (balama), de la T-pose cu palma în jos: pronație și
  // supinație (palma în sus). Măsurat pe cele 51 de semne (?check, forearmTwist): repausul = +22,
  // semnele cu palma în sus (what, now, again, ago) cer 100-117 (umărul avatarului nu se rotește
  // în IK, deci antebrațul preia și rotația lui); peste, doar inversări ale palmei din MediaPipe
  // (medicine -158 / +166, tired 160), care altfel dau răsuciri imposibile.
  forearm: { pron: 75, sup: 120 },
};
const jointLimits = (finger, i) => finger === 'Thumb'
  ? [LIMITS.thumbCmc, LIMITS.thumbMcp, LIMITS.thumbIp][i] : [LIMITS.mcp, LIMITS.pip, LIMITS.dip][i];
const THUMB_ROLL = degToRad(60);
// Degetele în repaus (fără date pentru mână): ușor îndoite, ca o mână relaxată
const RELAXED_FLEX = [15, 20, 10];

export const BODY = {
  palmRadius: 0.03,        // m: sfera palmei / încheieturii
  tipRadius: 0.01,         // m: vârfurile degetelor
  forearmRadius: 0.035,    // m: capsula antebrațului (între antebrațe și între cot și trunchi)
  neckRadius: 0.045,
  // cât de departe (în lățimi de umeri) contează fața ca reper pentru mână: doar la contact sau
  // aproape (0.6 trăgea în sus, la față, semnele din fața umărului - one, two, three, i - pentru
  // că avatarul are gâtul mult mai lung: ochii la ~0.75 lățimi de umeri deasupra umerilor, nu ~0.5)
  faceSigma: 0.3,
  chestSigma: 0.8,
  shoulderWeight: 0.3,     // ponderea constantă a umărului ca reper
  pole: new V3(0.6, -1, -0.35),   // direcția cotului (pentru brațul stâng; X se oglindește la dreptul)
  poleDataWeight: 0.3,     // cât din direcția cotului din date se păstrează
  dominantShare: 0.7,      // la separarea mâinilor, cât se mută mâna dreaptă (restul, stânga)
  shrugStart: 20,          // grade de ridicare a brațului peste orizontală de la care urcă umărul
  shrugGain: 0.3,
  shrugMax: 12,
  // Rotația trunchiului în jurul verticalei vine din diferența de adâncime (z) dintre umeri, pe
  // care MediaPipe o estimează slab (la unele semne cât lățimea umerilor => trunchi răsucit 45°):
  // se atenuează și se limitează.
  trunkDepthGain: 0.4,
  trunkYawMax: 15,         // grade
  torsoClearance: 0.025,   // m: palma rămâne cel puțin atât în fața trunchiului (fața/capul: contact permis)
  swivelMax: 60,           // grade: cât din răsucirea antebrațului peste limită poate prelua umărul
  reachFree: 0.15,         // m în fața pieptului fără compresie
  reachGain: 0.35,         // cât din distanța de peste reachFree se păstrează
  reachMax: 0.85,          // distanța umăr -> încheietură, din lungimea brațului
};

// ---------------------------------------------------------------- geometrie
export const vec = p => new V3(p[0], p[1], p[2]);
export const seg = (a, b) => new V3(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
export const mid = (a, b) => a.map((x, i) => (x + b[i]) / 2);
const perp = (v, axis) => { const n = axis.clone().normalize(); return v.clone().sub(n.multiplyScalar(v.dot(n))); };
const arcQuat = (from, to) => new Q().setFromUnitVectors(from.clone().normalize(), to.clone().normalize());
const median = xs => { const s = xs.filter(Number.isFinite).sort((a, b) => a - b); return s.length ? s[s.length >> 1] : NaN; };

function basis(a, b) {
  // reper ortonormat: x = a, z = a × b, y = z × x  (null dacă a și b sunt paralele)
  const x = a.clone().normalize();
  const z = new V3().crossVectors(x, b);
  if (z.lengthSq() < 1e-10) return null;
  z.normalize();
  return new THREE.Matrix4().makeBasis(x, new V3().crossVectors(z, x), z);
}

export function basisQuat(a0, b0, a, b) {
  // rotația care duce reperul (a0, b0) în reperul (a, b)
  const m0 = basis(a0, b0), m1 = basis(a, b);
  if (!m0 || !m1) return arcQuat(a0, a);
  return new Q().setFromRotationMatrix(m1.multiply(m0.transpose()));
}

function swingAngles(target, e1, normal, finger = false) {
  // Unghiurile (radiani) direcției `target` față de axa osului e1: flexia, ca unghi în planul
  // (e1, e2), e2 = spre palmă (negativ = extensie), și abducția (laterală, spre e3).
  const e2 = perp(normal, e1).normalize(), e3 = new V3().crossVectors(e1, e2);
  const t = target.clone().normalize();
  let flex = Math.atan2(t.dot(e2), t.dot(e1));
  // Deget: aproape de 180° semnul e ambiguu (falanga pliată complet, ex. pumn văzut din față,
  // unde MediaPipe pune toată îndoirea într-o singură articulație). Fără corecție, -170° ar fi
  // „îndoit pe spate”, limitat la 0 => deget complet drept. Un deget nu se hiperextinde peste
  // 90°, deci o asemenea valoare înseamnă o îndoire spre palmă.
  if (finger && flex < -Math.PI / 2) flex += 2 * Math.PI;
  return { flex, abd: Math.asin(clamp(t.dot(e3), -1, 1)) };
}

function swingFromAngles(e1, normal, { flex, abd }, limits) {
  // Rotația (fără răsucire) a axei osului e1 cu flexia și abducția date, în limitele anatomice
  const e2 = perp(normal, e1).normalize(), e3 = new V3().crossVectors(e1, e2);
  const f = clamp(flex, degToRad(limits.flex[0]), degToRad(limits.flex[1]));
  const a = clamp(abd, degToRad(limits.abd[0]), degToRad(limits.abd[1]));
  const d = e1.clone().multiplyScalar(Math.cos(a) * Math.cos(f))
    .addScaledVector(e2, Math.cos(a) * Math.sin(f)).addScaledVector(e3, Math.sin(a));
  return arcQuat(e1, d);
}

function limitedSwing(target, e1, normal, limits) {
  // Rotația (fără răsucire) care duce axa osului e1 spre `target`, cu unghiurile limitate
  return swingFromAngles(e1, normal, swingAngles(target, e1, normal), limits);
}

// Cuplajul anatomic al ultimelor două articulații ale degetului: îndoirea din vârf (DIP) ≈ 2/3
// din cea din mijloc (PIP); degetul nu poate îndoi vârful independent.
const DIP_PER_PIP = 2 / 3;

function chainReach(lengths, flexes) {
  // lanț plan: segmente cu lungimile date, fiecare rotit cu flexia lui față de cel dinainte;
  // întoarce distanța de la început la vârf / lungimea totală (1 = drept)
  let angle = 0, x = 0, y = 0;
  lengths.forEach((l, i) => { angle += flexes[i]; x += l * Math.cos(angle); y += l * Math.sin(angle); });
  return Math.hypot(x, y) / lengths.reduce((a, b) => a + b, 0);
}

function solveMonotone(f, target, lo, hi) {
  // x din [lo, hi] cu f(x) = target, pentru f descrescătoare (bisecție; capetele dacă nu se poate)
  if (f(lo) <= target) return lo;
  if (f(hi) >= target) return hi;
  for (let i = 0; i < 24; i++) {
    const m = (lo + hi) / 2;
    if (f(m) > target) lo = m; else hi = m;
  }
  return (lo + hi) / 2;
}

function fitFinger(hand, idx, lengths, limits) {
  // Flexiile unui deget (arătător...mic) din două măsuri sigure ale datelor, nu din unghiurile
  // fiecărei articulații: MediaPipe estimează slab adâncimea degetelor, deci la mâna strânsă
  // unghiurile ies cu semn greșit (degete „ridicate în plus”), laterale sau toată îndoirea într-o
  // singură articulație. Măsurile:
  //   1. cât de strâns e degetul: distanța bază -> vârf / lungimea degetului; depinde doar de
  //      articulația din mijloc și cea din vârf (cuplate, DIP_PER_PIP) => îndoirea din mijloc;
  //   2. cât de aproape e vârful de încheietura mâinii: distanța încheietură -> vârf / (metacarp +
  //      deget) => îndoirea de la baza degetului (pumn vs. palmă cu degetele drepte).
  const [w, b, p, d, t] = [hand[0], ...idx.map(i => hand[i])];
  const fingerLength = b.distanceTo(p) + p.distanceTo(d) + d.distanceTo(t);
  const curl = b.distanceTo(t) / fingerLength;
  const reach = w.distanceTo(t) / (w.distanceTo(b) + fingerLength);
  const rad = i => limits[i].flex.map(degToRad);
  const [pipLo, pipHi] = rad(1), [, dipHi] = rad(2), [mcpLo, mcpHi] = rad(0);
  const tail = pip => [0, pip, Math.min(DIP_PER_PIP * pip, dipHi)];
  const pip = solveMonotone(x => chainReach(lengths, tail(x)), curl, Math.max(pipLo, 0), pipHi);
  const [, , dip] = tail(pip);
  // metacarpul avatarului (încheietură -> baza degetului), în prelungirea palmei
  const full = [lengths.metacarpal, ...lengths];
  const mcp = solveMonotone(x => chainReach(full, [0, x, pip, dip]), reach, mcpLo, mcpHi);
  return [{ flex: mcp, abd: 0 }, { flex: pip, abd: 0 }, { flex: dip, abd: 0 }];
}

function redistributeCurl(angles, limits) {
  // Îndoirea totală a degetului se păstrează: ce depășește limita unei articulații (de obicei
  // cea din mijloc, la pumn) trece la articulațiile aceluiași deget care mai au loc, de la bază
  // spre vârf. MediaPipe estimează slab adâncimea degetelor, deci distribuția îndoirii între
  // articulații e nesigură, dar îndoirea totală (cât de strâns e degetul) e corectă.
  let excess = 0;
  angles.forEach((a, i) => {
    const max = degToRad(limits[i].flex[1]);
    if (a && a.flex > max) { excess += a.flex - max; a.flex = max; }
  });
  angles.forEach((a, i) => {
    if (!a || excess <= 0) return;
    const lo = degToRad(limits[i].flex[0]), hi = degToRad(limits[i].flex[1]);
    const start = Math.max(a.flex, lo), add = Math.min(Math.max(hi - start, 0), excess);
    a.flex = start + add;
    excess -= add;
  });
}

export const isHair = mesh => /hair/i.test([mesh.name, ...[mesh.material].flat().map(m => m?.name ?? '')].join(' '));

function keyedVector(keys, t) {
  // vectorul la momentul t din chei [[t, [x, y, z]], ...], interpolat lin (smoothstep)
  let i = 0;
  while (i < keys.length - 2 && keys[i + 1][0] <= t) i++;
  const [t0, a] = keys[i], [t1, b] = keys[Math.min(i + 1, keys.length - 1)];
  let f = t1 > t0 ? clamp((t - t0) / (t1 - t0), 0, 1) : 0;
  f = f * f * (3 - 2 * f);
  return a.map((x, k) => x + (b[k] - x) * f);
}

function rangeProgress(t, [t0, t1]) {
  // poziția lui t în intervalul [t0, t1] (0..1), sau null în afara lui
  return t < t0 || t > t1 ? null : (t - t0) / Math.max(t1 - t0, 1e-6);
}

function rangeWeight(t, [t0, t1], edge = 0.08) {
  // 1 în interiorul intervalului, cu intrare / ieșire lină pe `edge` la capete
  const s = x => { const k = clamp(x, 0, 1); return k * k * (3 - 2 * k); };
  return Math.min(s((t - t0 + edge) / edge), s((t1 + edge - t) / edge));
}

export function shapeAt(shape, t) {
  // forma mâinii la momentul t: un nume din HANDSHAPES sau chei [[t, nume], ...] interpolate
  if (!shape) return null;
  if (typeof shape === 'string') return HANDSHAPES[shape];
  let i = 0;
  while (i < shape.length - 1 && shape[i + 1][0] <= t) i++;
  const [t0, a] = shape[i], next = shape[Math.min(i + 1, shape.length - 1)];
  const A = HANDSHAPES[a], B = HANDSHAPES[next[1]];
  const f = next[0] > t0 ? clamp((t - t0) / (next[0] - t0), 0, 1) : 0;
  if (f === 0 || A === B) return A;
  const out = {};
  for (const finger of FINGERS) {
    const x = A[finger], y = B[finger];
    out[finger] = x && y ? x.map((v, k) => v + (y[k] - v) * f) : (f < 0.5 ? x : y);
  }
  return out;
}

function closestOnSegment(p, a, b) {
  const ab = b.clone().sub(a);
  const t = clamp(p.clone().sub(a).dot(ab) / Math.max(ab.lengthSq(), 1e-12), 0, 1);
  return a.clone().addScaledVector(ab, t);
}

function segmentDistance(a, b, c, d) {
  // distanța minimă între segmentele ab și cd (aproximare iterativă, suficientă aici)
  let p = a.clone(), q = c.clone();
  for (let i = 0; i < 4; i++) { q = closestOnSegment(p, c, d); p = closestOnSegment(q, a, b); }
  return { distance: p.distanceTo(q), p, q };
}

function pushFromCapsule(p, r, a, b, radius) {
  // deplasarea care scoate sfera (p, r) din capsula (a, b, radius); null dacă nu se ating
  const c = closestOnSegment(p, a, b), v = p.clone().sub(c), min = radius + r;
  const len = v.length();
  if (len >= min) return null;
  if (len < 1e-6) v.set(0, 0, 1);
  return v.normalize().multiplyScalar(min - len);
}

// ---------------------------------------------------------------- retargeter
export class Retargeter {
  constructor(vrm) {
    this.vrm = vrm;
    this.measure();
    this.rest = this.restPose();
  }

  node(name) { return this.vrm.humanoid.getNormalizedBoneNode(name); }

  measure() {
    // Poziții, rotații și direcții ale oaselor în T-pose, apoi volumele corpului (raze trimise
    // spre personaj), ca totul să se adapteze la orice personaj VRM.
    const vrm = this.vrm;
    vrm.humanoid.resetNormalizedPose();
    vrm.update(0);
    vrm.scene.updateMatrixWorld(true);
    this.pos = {}; this.restQ = {}; this.parent = {};
    const nameOf = new Map();
    for (const name of FK_ORDER) {
      const node = this.node(name);
      if (!node) continue;
      nameOf.set(node, name);
      this.pos[name] = node.getWorldPosition(new V3());
      this.restQ[name] = node.getWorldQuaternion(new Q());
    }
    for (const [node, name] of nameOf) {
      let p = node.parent;
      while (p && !nameOf.has(p)) p = p.parent;
      this.parent[name] = p ? nameOf.get(p) : null;
    }
    const P = this.pos;
    const dir = (a, b) => (P[a] && P[b]) ? P[b].clone().sub(P[a]).normalize() : null;
    this.dir = {}; this.side = {}; this.normal = {}; this.len = {};
    for (const s of SIDES) {
      this.dir[s + 'UpperArm'] = dir(s + 'UpperArm', s + 'LowerArm');
      this.dir[s + 'LowerArm'] = dir(s + 'LowerArm', s + 'Hand');
      this.dir[s + 'Hand'] = dir(s + 'Hand', s + 'MiddleProximal');
      this.side[s + 'Hand'] = dir(s + 'LittleProximal', s + 'IndexProximal');
      // normala palmei (spre interiorul palmei): în T-pose palmele privesc în jos
      const h = this.dir[s + 'Hand'], sd = this.side[s + 'Hand'];
      this.normal[s] = (s === 'left' ? new V3().crossVectors(h, sd) : new V3().crossVectors(sd, h)).normalize();
      for (const f of FINGERS) {
        const bones = fingerBones(s, f);
        bones.forEach((b, i) => { this.dir[b] = dir(b, bones[i + 1]) ?? this.dir[bones[i - 1]]; });
      }
      this.len[s] = {
        upper: P[s + 'UpperArm'].distanceTo(P[s + 'LowerArm']),
        fore: P[s + 'LowerArm'].distanceTo(P[s + 'Hand']),
        palm: P[s + 'Hand'].distanceTo(P[s + 'MiddleProximal']),
      };
    }
    this.shoulderWidth = P.leftUpperArm.distanceTo(P.rightUpperArm);
    if (P.leftUpperArm.x < P.rightUpperArm.x) console.warn('Avatarul nu stă cu fața spre +Z');

    const ray = new THREE.Raycaster();
    const hit = (origin, direction) => {
      ray.set(origin, direction);
      return ray.intersectObject(vrm.scene, true).find(i => i.object.isMesh)?.point ?? null;
    };
    // Trunchiul: elipsă în jurul pieptului (lățime din umeri, adâncime măsurată în față)
    const chest = P.chest, top = (P.neck ?? P.leftUpperArm).y;
    const chestFront = hit(new V3(0, P.leftUpperArm.y - 0.25 * this.shoulderWidth, 2), new V3(0, 0, -1));
    this.torso = {
      rx: 0.42 * this.shoulderWidth,
      rz: clamp(chestFront ? chestFront.z - chest.z : 0.1, 0.06, 0.2),
      yBottom: P.hips.y - 0.02 - chest.y, yTop: top - chest.y,
    };
    // Lățimea reală a trunchiului (cu tricoul), din lateral, sub axilă: pentru coliziunea cotului
    // (elipsa de mai sus e la lățimea articulațiilor umerilor, iar cotul intra în tricou).
    const bodyHit = (origin, direction) => {
      ray.set(origin, direction);
      return ray.intersectObject(vrm.scene, true).find(i => i.object.isMesh && !isHair(i.object))?.point ?? null;
    };
    const sideAt = y => [1, -1].map(sx => bodyHit(new V3(sx * 2, y, chest.z), new V3(-sx, 0, 0)))
      .map(h => h ? Math.abs(h.x - chest.x) : 0);
    const widths = [P.leftUpperArm.y - 0.12, chest.y, P.hips.y + 0.08].flatMap(sideAt).filter(w => w > 0.05 && w < 0.3);
    this.torso.rxOuter = widths.length ? Math.max(...widths) : this.torso.rx + 0.04;
    // Capul: sferă care atinge fața în față; raza din lățimea capului (cu păr)
    const eyes = (P.leftEye && P.rightEye) ? P.leftEye.clone().add(P.rightEye).multiplyScalar(0.5)
      : P.head.clone().add(new V3(0, 0.06, 0.05));
    const face = hit(new V3(0, eyes.y - 0.025, 2), new V3(0, 0, -1));
    const side = hit(new V3(2, eyes.y, eyes.z - 0.04), new V3(-1, 0, 0));
    const radius = clamp(side ? side.x : 0.1, 0.07, 0.14);
    this.head = { center: new V3(0, eyes.y, (face ? face.z : eyes.z + 0.02) - radius), radius };
    this.eyes = eyes;

    // Pentru landmarks(): nasul (pe fața personajului) și vârfurile degetelor (nodul copil al
    // falangei distale, dacă modelul îl are; altfel falanga distală prelungită), față de os.
    const nose = hit(new V3(0, eyes.y - 0.045, 2), new V3(0, 0, -1)) ?? eyes.clone().add(new V3(0, -0.045, 0.06));
    this.noseOffset = nose.sub(P.head);
    this.tipOffset = {};
    this.fingerLen = {};   // lungimile celor trei falange (pentru matchCurl)
    for (const s of SIDES) for (const f of FINGERS) {
      const [proximal, middle, distal] = fingerBones(s, f), node = this.node(distal);
      const child = node?.children.find(c => !nameOf.has(c));
      this.tipOffset[distal] = child ? child.getWorldPosition(new V3()).sub(P[distal])
        : this.dir[distal].clone().multiplyScalar(0.9 * P[distal].distanceTo(P[middle]));
      if (P[proximal] && P[middle] && P[distal]) {
        this.fingerLen[s + f] = [P[proximal].distanceTo(P[middle]), P[middle].distanceTo(P[distal]), this.tipOffset[distal].length()];
        this.fingerLen[s + f].metacarpal = P[s + 'Hand'].distanceTo(P[proximal]);
      }
    }
    // Suprafața reală din față a trunchiului (cu hainele) și a feței, ca hărți de adâncime: elipsa
    // și sfera sunt în interiorul tricoului / pieptului, iar mâna intra prin haine.
    const started = performance.now();
    this.surfaces = {
      torso: this.makeGrid('chest', [-0.12, 0.12], [P.hips.y - 0.06, P.head.y], 0.015),
      face: this.makeGrid('head', [-0.075, 0.075], [P.head.y - 0.02, eyes.y + 0.08], 0.01),
    };
    this.rasterize(Object.values(this.surfaces));
    console.info(`suprafețe corp măsurate în ${Math.round(performance.now() - started)} ms`);
    this.measureAnchors(hit);
    console.info('Volume corp', JSON.stringify({ torso: this.torso, head: { radius, center: this.head.center } }));
  }

  makeGrid(bone, [x0, x1], [y0, y1], step) {
    // grilă în planul x-y (T-pose, față de poziția osului) cu z-ul suprafeței din față; NaN = fără corp
    const origin = this.pos[bone], nx = Math.round((x1 - x0) / step) + 1, ny = Math.round((y1 - y0) / step) + 1;
    return { bone, origin, x0: x0 - origin.x, y0: y0 - origin.y, step, nx, ny, z: new Float32Array(nx * ny).fill(NaN) };
  }

  rasterize(grids) {
    // Suprafața din față = z-ul maxim al triunghiurilor personajului (fără păr) în fiecare celulă.
    // O singură trecere prin triunghiuri (vârfurile în poza curentă, T-pose, cu skinning):
    // mult mai rapid decât câte o rază pe celulă.
    const v = new V3();
    this.vrm.scene.traverse(mesh => {
      if (!mesh.isMesh || isHair(mesh) || !mesh.geometry?.attributes?.position) return;
      const pos = mesh.geometry.attributes.position, index = mesh.geometry.index, n = pos.count;
      const world = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        v.fromBufferAttribute(pos, i);
        if (mesh.isSkinnedMesh) mesh.applyBoneTransform(i, v);
        v.applyMatrix4(mesh.matrixWorld);
        world[3 * i] = v.x; world[3 * i + 1] = v.y; world[3 * i + 2] = v.z;
      }
      const triangles = index ? index.count / 3 : n / 3;
      for (let k = 0; k < triangles; k++) {
        const ids = [0, 1, 2].map(c => (index ? index.getX(3 * k + c) : 3 * k + c) * 3);
        for (const g of grids) {
          // vârfurile în unități de celulă ale grilei
          const P = ids.map(o => [(world[o] - g.origin.x - g.x0) / g.step, (world[o + 1] - g.origin.y - g.y0) / g.step, world[o + 2] - g.origin.z]);
          const i0 = Math.max(0, Math.ceil(Math.min(P[0][0], P[1][0], P[2][0]))), i1 = Math.min(g.nx - 1, Math.floor(Math.max(P[0][0], P[1][0], P[2][0])));
          const j0 = Math.max(0, Math.ceil(Math.min(P[0][1], P[1][1], P[2][1]))), j1 = Math.min(g.ny - 1, Math.floor(Math.max(P[0][1], P[1][1], P[2][1])));
          if (i0 > i1 || j0 > j1) continue;
          const d = (P[1][1] - P[2][1]) * (P[0][0] - P[2][0]) + (P[2][0] - P[1][0]) * (P[0][1] - P[2][1]);
          if (Math.abs(d) < 1e-12) continue;
          for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
            const a = ((P[1][1] - P[2][1]) * (i - P[2][0]) + (P[2][0] - P[1][0]) * (j - P[2][1])) / d;
            const b = ((P[2][1] - P[0][1]) * (i - P[2][0]) + (P[0][0] - P[2][0]) * (j - P[2][1])) / d;
            const c = 1 - a - b;
            if (a < -1e-6 || b < -1e-6 || c < -1e-6) continue;
            const z = a * P[0][2] + b * P[1][2] + c * P[2][2], cell = j * g.nx + i;
            if (!(g.z[cell] >= z)) g.z[cell] = z;
          }
        }
      }
    });
  }

  surfaceZ(grid, x, y) {
    // z-ul suprafeței în punctul (x, y) (coordonate față de os), interpolat; null în afara grilei
    const fx = (x - grid.x0) / grid.step, fy = (y - grid.y0) / grid.step;
    if (fx < 0 || fy < 0 || fx > grid.nx - 1 || fy > grid.ny - 1) return null;
    const i = Math.min(Math.floor(fx), grid.nx - 2), j = Math.min(Math.floor(fy), grid.ny - 2), u = fx - i, v = fy - j;
    const at = (a, b) => grid.z[b * grid.nx + a];
    const corners = [[at(i, j), (1 - u) * (1 - v)], [at(i + 1, j), u * (1 - v)], [at(i, j + 1), (1 - u) * v], [at(i + 1, j + 1), u * v]]
      .filter(([value]) => Number.isFinite(value));
    if (!corners.length) return null;
    const weight = corners.reduce((sum, [, w]) => sum + w, 0);
    return weight > 1e-6 ? corners.reduce((sum, [value, w]) => sum + value * w, 0) / weight : Math.max(...corners.map(c => c[0]));
  }

  surfacePush(p, r, body, margin = 0) {
    // deplasarea (înainte) care scoate sfera (p, r) din suprafața reală a trunchiului sau a feței
    let best = null;
    for (const grid of Object.values(this.surfaces)) {
      const q = body.eff[grid.bone].clone().invert();
      const local = p.clone().sub(body.pos[grid.bone]).applyQuaternion(q);
      const zs = this.surfaceZ(grid, local.x, local.y);
      if (zs === null) continue;
      const need = zs + r + margin - local.z;
      if (need > 0 && local.z > zs - 0.25 && (!best || need > best.need)) {
        best = { need, v: new V3(0, 0, need).applyQuaternion(body.eff[grid.bone]) };
      }
    }
    return best?.v ?? null;
  }

  handBodyPush(s, body, contact) {
    // după cinematica directă: deplasarea care scoate toată mâna (palmă, dos, degete) din corp
    const margin = contact ? 0.002 : 0.012, push = new V3();
    for (const name of ['palm', 'back', 'knuckles', 'wrist', 'thumb', 'index', 'middle', 'little', 'indexMid', 'littleMid']) {
      const v = this.surfacePush(this.handPoint(s, name, body), 0.008, body, margin);
      if (v && v.lengthSq() > push.lengthSq()) push.copy(v);
    }
    return push;
  }

  measureAnchors(hit) {
    // Repere pe suprafața corpului (pentru signs.js), măsurate pe personaj în T-pose și legate
    // de un os (capul, gâtul sau pieptul), ca să urmeze corpul. Pe părți: _l = stânga, _r = dreapta.
    const P = this.pos, eyes = this.eyes, { center, radius } = this.head;
    const frontAt = (y, x = 0) => hit(new V3(x, y, 2), new V3(0, 0, -1));
    // profilul feței din față: bărbia e acolo unde, sub nas, suprafața sare înapoi spre gât
    const noseY = eyes.y - 0.045, profile = [];
    for (let y = noseY; y > eyes.y - 0.2; y -= 0.004) { const h = frontAt(y); if (h) profile.push(h); }
    let chin = profile[profile.length - 1] ?? eyes.clone().add(new V3(0, -0.11, 0.05));
    for (let i = 1; i < profile.length; i++) {
      if (profile[i - 1].z - profile[i].z > 0.012) { chin = profile[i - 1]; break; }
    }
    const mouth = frontAt((noseY + chin.y) / 2) ?? chin.clone().add(new V3(0, 0.025, 0));
    const sphere = (x, y, z) => center.clone().addScaledVector(new V3(x, y, z).normalize(), radius);
    const anchors = {
      head: { forehead: frontAt(eyes.y + 0.045) ?? sphere(0, 0.45, 0.9), nose: frontAt(noseY), mouth, chin },
      neck: { neck: (P.neck ?? P.head).clone().add(new V3(0, 0.01, BODY.neckRadius)) },
      chest: {},
    };
    for (const [side, sx] of [['l', 1], ['r', -1]]) {
      Object.assign(anchors.head, {
        ['temple_' + side]: sphere(0.8 * sx, 0.3, 0.55), ['ear_' + side]: sphere(sx, -0.12, -0.05),
        ['cheek_' + side]: sphere(0.55 * sx, -0.3, 0.8), ['jaw_' + side]: sphere(0.55 * sx, -0.65, 0.5),
        ['mouth_' + side]: mouth.clone().add(new V3(0.025 * sx, 0, -0.005)),
      });
    }
    // trunchiul: punctul din față al elipsei, la înălțimea și lateral cât se cere
    // pe suprafața reală (cu hainele), din harta de adâncime a trunchiului
    const T = this.torso, chestP = P.chest;
    const torsoFront = (dy, x) => {
      const z = this.surfaceZ(this.surfaces.torso, x, dy);
      return chestP.clone().add(new V3(x, dy, z ?? T.rz * Math.sqrt(Math.max(0, 1 - (x / T.rx) ** 2))));
    };
    const chestY = P.leftUpperArm.y - 0.3 * this.shoulderWidth - chestP.y;
    const stomach = torsoFront(P.hips.y + 0.1 - chestP.y, 0);
    Object.assign(anchors.chest, {
      chest: torsoFront(chestY, 0), chest_l: torsoFront(chestY, 0.5 * T.rx), chest_r: torsoFront(chestY, -0.5 * T.rx),
      stomach, waist_r: new V3(-T.rx - 0.12, P.hips.y + 0.05, chestP.z + 0.08),
      shoulder_l: P.leftUpperArm.clone().add(new V3(0, 0.03, 0)), shoulder_r: P.rightUpperArm.clone().add(new V3(0, 0.03, 0)),
    });
    this.anchors = {};
    for (const [bone, list] of Object.entries(anchors)) {
      const b = P[bone] ? bone : 'head';
      for (const [name, point] of Object.entries(list)) if (point) this.anchors[name] = { bone: b, offset: point.clone().sub(P[b]) };
    }
  }

  anchor(name, body) {
    const a = this.anchors[name];
    return a.offset.clone().applyQuaternion(body.eff[a.bone]).add(body.pos[a.bone]);
  }

  handPoint(s, name, body) {
    // un punct al mâinii avatarului după cinematica directă: încheietura, palma (suprafața ei),
    // dosul palmei, baza degetelor sau vârful unui deget
    const pos = body.pos, eff = body.eff, H = s + 'Hand';
    const normal = this.normal[s].clone().applyQuaternion(eff[H]);
    const palm = pos[H].clone().lerp(pos[s + 'MiddleProximal'], 0.55);
    const tip = f => { const b = fingerBones(s, f)[2]; return this.tipOffset[b].clone().applyQuaternion(eff[b]).add(pos[b]); };
    switch (name) {
      case 'wrist': return pos[H].clone();
      case 'palm': return palm.addScaledVector(normal, 0.015);
      case 'back': return palm.addScaledVector(normal, -0.02);
      case 'knuckles': return pos[s + 'MiddleProximal'].clone();
      case 'heel': return pos[H].clone().lerp(pos[s + 'MiddleProximal'], 0.15).addScaledVector(normal, 0.015);
      case 'fingers': return pos[s + 'MiddleIntermediate'].clone().addScaledVector(normal, 0.012);
      case 'thumb': return tip('Thumb');
      case 'index': return tip('Index');
      case 'middle': return tip('Middle');
      case 'little': return tip('Little');
      case 'indexMid': return pos[s + 'IndexIntermediate'].clone();
      case 'littleMid': return pos[s + 'LittleIntermediate'].clone();
      default: return palm;
    }
  }

  // ---------------------------------------------------------------- cinematică directă
  fk(D) {
    // pozițiile oaselor pentru rotațiile delta D (osul fără delta urmează părintele)
    const pos = {}, eff = {};
    for (const name of FK_ORDER) {
      if (!this.pos[name]) continue;
      const p = this.parent[name];
      eff[name] = D.get(name) ?? (p ? eff[p] : new Q());
      pos[name] = p ? this.pos[name].clone().sub(this.pos[p]).applyQuaternion(eff[p]).add(pos[p]) : this.pos[name].clone();
    }
    return { pos, eff };
  }

  // ---------------------------------------------------------------- statistici ale unui semn
  motionStats(motion) {
    // lungimi tipice în date (pentru scări): lățimea umerilor, lungimea brațului și a palmei
    const frames = motion.frames.filter(f => f.pose);
    const stats = { shoulderWidth: median(frames.map(f => seg(f.pose[12], f.pose[11]).length())), arm: {}, palm: {} };
    for (const [k, s] of SIDES.entries()) {
      const [S, E, W] = ARM_POINTS[s];
      const active = frames.filter(f => f.weight[k] >= 1);
      const lengths = active.map(f => seg(f.pose[S], f.pose[E]).length() + seg(f.pose[E], f.pose[W]).length()).sort((a, b) => a - b);
      stats.arm[s] = lengths.length ? lengths[Math.floor(lengths.length * 0.75)] : 2.2 * stats.shoulderWidth;
      stats.palm[s] = median(active.map(f => f[s + '_hand'] ? seg(f[s + '_hand'][0], f[s + '_hand'][9]).length() : NaN));
    }
    for (const s of SIDES) {   // o mână fără date ia valorile celeilalte
      if (!Number.isFinite(stats.palm[s])) stats.palm[s] = stats.palm[s === 'left' ? 'right' : 'left'];
      if (!Number.isFinite(stats.palm[s])) stats.palm[s] = 0.3 * stats.shoulderWidth;
    }
    return stats;
  }

  // ---------------------------------------------------------------- poze
  restPose() {
    // Brațele jos, lângă corp, palmele spre coapse, degetele relaxate
    const D = new Map([['spine', new Q()], ['chest', new Q()], ['neck', new Q()], ['head', new Q()]]);
    for (const s of SIDES) {
      const sx = s === 'left' ? 1 : -1;
      D.set(s + 'Shoulder', new Q());
      this.orient(D, s, new V3(0.22 * sx, -1, 0.02), new V3(0.1 * sx, -1, 0.22),
                  new V3(0.06 * sx, -1, 0.3), new V3(0, 0.25, 1), null, new V3(sx, 0, 0));
    }
    return { D, mouth: 0 };
  }

  framePose(frame, stats, ctx = {}) {
    const D = new Map(this.rest.D);
    let mouth = 0;
    const p = frame.pose;
    if (p) {
      // linia umerilor: înclinare și rotire a trunchiului (rotirea atenuată și limitată, vezi BODY)
      const line = seg(p[12], p[11]);
      line.z *= BODY.trunkDepthGain;
      const maxZ = Math.tan(degToRad(BODY.trunkYawMax)) * Math.abs(line.x);
      line.z = clamp(line.z, -maxZ, maxZ);
      const R = arcQuat(new V3(1, 0, 0), line);
      D.set('spine', new Q().slerp(R, 0.5)).set('chest', R).set('leftShoulder', R).set('rightShoulder', R);
    }
    const f = frame.face;
    if (f) {
      const head = basisQuat(new V3(0, 1, 0), new V3(1, 0, 0), seg(f.chin, f.forehead), seg(f.eye_right, f.eye_left));
      D.set('head', head).set('neck', D.get('chest').clone().slerp(head, 0.5));
      const ratio = seg(f.lip_top, f.lip_bottom).length() / seg(f.chin, f.forehead).length();
      mouth = clamp((ratio - 0.02) / 0.08, 0, 1);
    }
    if (p) this.solveArms(D, frame, stats, ctx);
    return { D, mouth };
  }

  solveArms(D, frame, stats, ctx = {}) {
    const active = SIDES.filter((s, k) => frame.weight[k] > 0);
    if (!active.length) return;
    let body = this.fk(D);
    const volumes = this.volumes(body);
    const spec = ctx.spec ?? {}, t = ctx.t ?? 0, time = ctx.time ?? 0;
    const arms = {};
    for (const s of active) {
      arms[s] = this.handTarget(frame, s, stats, body, spec[s]?.free);
      this.applyHandSpec(arms[s], s, spec[s], t, time, body, ctx);
      this.pushOutOfBody(arms[s], volumes, spec[s]?.contact);
    }
    if (arms.left && arms.right) this.separateHands(arms.left, arms.right);

    // umărul urcă puțin când mâna e mult deasupra lui
    for (const s of active) {
      const toTarget = arms[s].target.clone().sub(body.pos[s + 'UpperArm']).normalize();
      const elevation = THREE.MathUtils.radToDeg(Math.asin(clamp(toTarget.y, -1, 1)));
      const shrug = clamp((elevation - BODY.shrugStart) * BODY.shrugGain, 0, BODY.shrugMax);
      const raise = new Q().setFromAxisAngle(new V3(0, 0, 1), degToRad(s === 'left' ? shrug : -shrug));
      D.set(s + 'Shoulder', raise.multiply(D.get('chest')));
    }
    body = this.fk(D);

    for (const s of active) this.solveIK(arms[s], s, body, volumes);
    if (arms.left && arms.right) {
      // antebrațele nu trec unul prin celălalt: antebrațul drept trece prin față
      for (let i = 0; i < 4; i++) {
        const { distance } = segmentDistance(arms.left.elbow, arms.left.wrist, arms.right.elbow, arms.right.wrist);
        const deficit = 2 * BODY.forearmRadius - distance;
        if (deficit <= 0) break;
        this.moveHand(arms.right, new V3(0, 0, deficit * 1.5));   // cotul se mută mai puțin decât mâna
        this.solveIK(arms.right, 'right', body, volumes);
      }
    }
    const finish = (s, swivel = true) => {
      const k = SIDES.indexOf(s), a = arms[s], solved = new Map(D);
      if (swivel) this.swivelForTwist(a, s, body, volumes);
      this.orient(solved, s, a.elbow.clone().sub(a.shoulder), a.wrist.clone().sub(a.elbow),
                  a.handDir, a.handSide, a.hand, a.pole, shapeAt(spec[s]?.shape, t));
      const w = frame.weight[k];
      for (const name of [s + 'UpperArm', s + 'LowerArm', s + 'Hand', ...FINGERS.flatMap(f => fingerBones(s, f))]) {
        if (solved.has(name)) D.set(name, this.rest.D.get(name).clone().slerp(solved.get(name), w));
      }
    };
    // cot fixat în signs.js (pole sau arc): fără swivel, care ar roti cotul în afara punctului fix
    for (const s of SIDES) if (arms[s]) finish(s, !spec[s]?.pole && !spec[s]?.arc);
    if (ctx.collect) for (const s of SIDES) if (arms[s]) ctx.collect[s] = arms[s].elbow.clone();

    // Corecții după ce mâinile au o poză: punctul de contact pe reper (traiectorii din signs.js),
    // atingerea mâinilor, mâna dreaptă scoasă din antebrațul stâng. Fiecare corecție mută ținta
    // încheieturii cu eroarea măsurată și refolosește IK-ul; două treceri ajung.
    const correct = (s, delta) => {
      if (delta.lengthSq() < 1e-8) return;
      this.moveHand(arms[s], delta);
      this.solveIK(arms[s], s, body, volumes);
      finish(s, false);
      body = this.fk(D);
    };
    for (let pass = 0; pass < 3; pass++) {
      for (const s of SIDES) {
        if (!arms[s]) continue;
        const other = s === 'left' ? 'right' : 'left';
        body = this.fk(D);
        const delta = new V3();
        const goal = this.pathGoal(spec[s], t, body, s, arms[other]);
        // arc din cot: cotul spre pivot (ex. sprijinit pe mâna cealaltă); IK refăcut chiar dacă mâna e
        // deja la țintă, ca cotul să ajungă exact în punctul fix
        if (goal?.pivot) {
          arms[s].fixedElbow = true;
          arms[s].pole = goal.pivot.clone().sub(body.pos[s + 'UpperArm']);
          this.solveIK(arms[s], s, body, volumes);
          finish(s, false);
          body = this.fk(D);
        }
        if (goal) delta.add(goal.point.add(arms[s].overlay ?? new V3()).sub(this.handPoint(s, goal.via, body)));
        const touch = spec.touch && s === 'right' && arms.left ? rangeWeight(t, spec.touch) : 0;
        if (touch > 0) {
          const gap = this.handGap(body);
          if (gap.distance > 0.012) delta.addScaledVector(gap.direction, (gap.distance - 0.012) * touch);
        }
        correct(s, delta);
        // apoi mâna (toate punctele ei) scoasă din corp și din antebrațul celălalt
        if (spec[s]?.arc) continue;
        const push = this.handBodyPush(s, body, spec[s]?.contact);
        if (s === 'right' && arms.left) { const f = this.forearmPush(body); if (f.lengthSq() > push.lengthSq()) push.copy(f); }
        correct(s, push);
      }
    }
  }

  bodyReport(D) {
    // pătrunderea maximă (cm) a punctelor mâinilor în suprafața reală a corpului, pentru verificare
    const body = this.fk(D);
    let worst = 0;
    for (const s of SIDES) for (const name of ['palm', 'back', 'knuckles', 'wrist', 'thumb', 'index', 'middle', 'little', 'indexMid', 'littleMid']) {
      const v = this.surfacePush(this.handPoint(s, name, body), 0, body);
      if (v) worst = Math.max(worst, v.length() * 100);
    }
    return worst;
  }

  applyHandSpec(arm, s, hs, t, time, body, ctx) {
    // Corecțiile din signs.js pentru o mână, înainte de IK: decalaj constant, mișcări adăugate
    // (cerc, bătăi, frecare, tremur, îndoirea încheieturii), orientarea impusă a palmei, pivot în cot.
    arm.overlay = new V3();   // decalajul + mișcările adăugate (se păstrează și la traiectorii)
    if (!hs) return;
    const R = body.eff.chest, toBody = v => new V3(...v).applyQuaternion(R);
    if (hs.offset) arm.overlay.add(toBody(hs.offset));
    for (const m of hs.motion ?? []) {
      const u = rangeProgress(t, m.t ?? [0, 1]);
      if (u === null) continue;
      const fade = rangeWeight(t, m.t ?? [0, 1]);
      if (m.type === 'circle') {
        const a = 2 * Math.PI * (m.turns ?? 2) * u, c = m.radius * (Math.cos(a) - 1), d = m.radius * Math.sin(a);
        arm.overlay.add(toBody(m.plane === 'xz' ? [c, 0, d] : [c, d, 0]).multiplyScalar(fade));
      } else if (m.type === 'tap' || m.type === 'rub') {
        const wave = m.type === 'tap' ? Math.sin(Math.PI * (m.count ?? 2) * u) ** 2 : Math.sin(2 * Math.PI * (m.count ?? 2) * u);
        arm.overlay.add(toBody(m.dir).multiplyScalar(m.amp * wave * fade));
      } else if (m.type === 'shake') {
        arm.overlay.add(toBody(m.dir ?? [1, 0, 0]).multiplyScalar(m.amp * Math.sin(2 * Math.PI * (m.freq ?? 6) * time) * fade));
      } else if (m.type === 'flap') {
        // îndoirea din încheietură (sus-jos), de `count` ori
        const angle = degToRad(m.amp) * Math.sin(Math.PI * (m.count ?? 2) * u) ** 2 * fade;
        arm.handDir.applyAxisAngle(arm.handSide.clone().normalize(), s === 'left' ? -angle : angle);
      } else if (m.type === 'twist') {
        // rotirea mâinii din încheietură (în jurul axei ei), înainte-înapoi de `count` ori
        const angle = degToRad(m.amp) * Math.sin(2 * Math.PI * (m.count ?? 2) * u) * fade;
        arm.handSide.applyAxisAngle(arm.handDir.clone().normalize(), angle);
      }
    }
    this.moveHand(arm, arm.overlay);
    if (hs.orient) {
      // dir: 'arc' = degetele în prelungirea antebrațului care se rotește (signs.js `arc`)
      // `at`: orientarea din date până aproape de t = at, apoi trecere lină spre cea impusă
      const w = hs.orient.at === undefined ? 1 : clamp((t - hs.orient.at + 0.2) / 0.2, 0, 1);
      const dir = hs.orient.dir === 'arc' ? this.arcDir(hs, t, R) : toBody(hs.orient.dir);
      arm.handDir = arm.handDir.clone().normalize().lerp(dir.normalize(), w);
      // side: un vector sau chei [[t, vector], ...] (palma se rotește lin pe durata semnului)
      const side = Array.isArray(hs.orient.side[0]) ? keyedVector(hs.orient.side, t) : hs.orient.side;
      arm.handSide = arm.handSide.clone().normalize().lerp(toBody(side).normalize(), w);
    }
    if (hs.pole) arm.pole = toBody(hs.pole);   // direcția fixă a cotului (fără salturi ale cotului)
    if (hs.pivot && ctx.pivot?.[s]) {
      // mișcarea doar din cot: cotul rămâne unde e la mijlocul semnului, mâna se mișcă pe sfera
      // cu raza antebrațului în jurul lui
      const E = ctx.pivot[s], L2 = this.len[s].fore;
      const goal = E.clone().addScaledVector(arm.target.clone().sub(E).normalize(), L2);
      this.moveHand(arm, goal.sub(arm.target));
      arm.pole = E.clone().sub(body.pos[s + 'UpperArm']);
    }
  }

  pathGoal(hs, t, body, side, otherArm) {
    const other = side === 'left' ? 'right' : 'left';
    // Punctul în care trebuie să ajungă o parte a mâinii (`via`) la momentul t, din traiectoria
    // din signs.js: chei [t, reper, decalaj (m, în reperul pieptului)], interpolate lin (smoothstep).
    // Reperele „other.*” sunt pe cealaltă mână / celălalt braț.
    if (!hs || (!hs.path && !hs.arc)) return null;
    const R = body.eff.chest;
    const where = (name, offset = [0, 0, 0]) => {
      let p;
      if (name.startsWith('other.')) {
        if (!otherArm) return null;
        const part = name.slice(6);
        p = part === 'elbow' ? body.pos[other + 'LowerArm'].clone()
          : part === 'forearm' ? body.pos[other + 'LowerArm'].clone().lerp(body.pos[other + 'Hand'], 0.5)
          : this.handPoint(other, part, body);
      } else {
        p = this.anchor(name, body);
      }
      return p.add(new V3(...offset).applyQuaternion(R));
    };
    if (hs.arc) {
      // antebrațul se rotește în jurul cotului: încheietura pe un arc. Cotul e fie la reperul `pivot`,
      // fie (cu `upper`) exact la umăr + lungimea brațului avatarului pe direcția `upper`: un punct pe
      // care cotul chiar îl poate atinge, deci rămâne fix (balama) pe toată mișcarea
      const pivot = hs.arc.upper
        ? body.pos[side + 'UpperArm'].clone().addScaledVector(new V3(...hs.arc.upper).applyQuaternion(R).normalize(), this.len[side].upper)
        : where(hs.arc.pivot, hs.arc.offset);
      return pivot && { point: pivot.clone().addScaledVector(this.arcDir(hs, t, R), this.len[side].fore), via: 'wrist', pivot };
    }
    const keys = hs.path;
    let i = 0;
    while (i < keys.length - 2 && keys[i + 1][0] <= t) i++;
    const [t0, n0, o0] = keys[i], [t1, n1, o1] = keys[Math.min(i + 1, keys.length - 1)];
    const a = where(n0, o0), b = where(n1, o1);
    if (!a || !b) return null;
    let f = t1 > t0 ? clamp((t - t0) / (t1 - t0), 0, 1) : 0;
    f = f * f * (3 - 2 * f);
    return { point: a.lerp(b, f), via: hs.via ?? 'palm' };
  }

  arcDir(hs, t, R) {
    const u = rangeProgress(t, hs.arc.t ?? [0, 1]) ?? (t < (hs.arc.t?.[0] ?? 0) ? 0 : 1);
    const k = u * u * (3 - 2 * u);
    const from = new V3(...hs.arc.from).normalize(), to = new V3(...hs.arc.to).normalize();
    return from.lerp(to, k).normalize().applyQuaternion(R);
  }

  handGap(body) {
    // cea mai mică distanță între mâna dreaptă și cea stângă (puncte ale palmei și degetelor)
    const names = ['palm', 'back', 'knuckles', 'thumb', 'index', 'middle', 'little', 'wrist'];
    let best = { distance: Infinity, direction: new V3() };
    for (const a of names) for (const b of names) {
      const pr = this.handPoint('right', a, body), pl = this.handPoint('left', b, body);
      const d = pr.distanceTo(pl);
      if (d < best.distance) best = { distance: d, direction: pl.sub(pr).normalize() };
    }
    return best;
  }

  forearmPush(body) {
    // mâna dreaptă nu trece prin antebrațul stâng (cu mâneca): deplasarea care o scoate afară
    const a = body.pos.leftLowerArm, b = body.pos.leftHand, push = new V3();
    for (const name of ['palm', 'back', 'index', 'middle', 'thumb', 'little', 'wrist']) {
      const p = this.handPoint('right', name, body);
      // mâneca e mai groasă spre cot
      const c = closestOnSegment(p, a, b), along = c.distanceTo(b) / Math.max(a.distanceTo(b), 1e-6);
      const v = pushFromCapsule(p, 0.008, c, c, BODY.forearmRadius + 0.02 * along);
      if (v && v.lengthSq() > push.lengthSq()) push.copy(v);
    }
    return push;
  }

  handTarget(frame, s, stats, body, free = false) {
    // Ținta încheieturii în scenă, ca medie ponderată a trei variante raportate la repere
    // (față, piept, umăr): reperul apropiat de mână contează cel mai mult.
    const p = frame.pose, [S, E, W, PINKY, INDEX] = ARM_POINTS[s];
    const wrist = vec(p[W]);
    let handData = frame[s + '_hand'];
    // z-ul mâinii e relativ la încheietură: îl aducem la z-ul încheieturii din corp
    if (handData) { const dz = p[W][2] - handData[0][2]; handData = handData.map(q => new V3(q[0], q[1], q[2] + dz)); }
    const dataPoints = handData ?? [wrist, vec(mid(p[PINKY], p[INDEX]))];

    const pos = body.pos;
    const faceD = vec(mid(p[EYES[0]], p[EYES[1]])), chestD = vec(mid(p[11], p[12])), shoulderD = vec(p[S]);
    const faceA = this.eyes.clone().sub(this.pos.head).applyQuaternion(body.eff.head).add(pos.head);
    const chestA = pos.leftUpperArm.clone().add(pos.rightUpperArm).multiplyScalar(0.5);
    const shoulderA = pos[s + 'UpperArm'];
    const bodyScale = this.shoulderWidth / stats.shoulderWidth;
    const armScale = (this.len[s].upper + this.len[s].fore) / stats.arm[s];
    const nearest = anchor => Math.min(...dataPoints.map(q => q.distanceTo(anchor))) / stats.shoulderWidth;
    const weights = [Math.exp(-((nearest(faceD) / BODY.faceSigma) ** 2)),
                     Math.exp(-((nearest(chestD) / BODY.chestSigma) ** 2)), BODY.shoulderWeight];
    const candidates = [faceA.clone().addScaledVector(wrist.clone().sub(faceD), bodyScale),
                        chestA.clone().addScaledVector(wrist.clone().sub(chestD), bodyScale),
                        shoulderA.clone().addScaledVector(wrist.clone().sub(shoulderD), armScale)];
    const total = weights.reduce((a, b) => a + b, 0);
    const target = new V3();
    candidates.forEach((c, i) => target.addScaledVector(c, weights[i] / total));
    // Adâncimea: z-ul MediaPipe Pose exagerează mult cât de în față e mâna (brațele ieșeau
    // întinse spre privitor). Distanța în fața pieptului se comprimă peste BODY.reachFree, iar
    // mâna rămâne la cel mult BODY.reachMax din lungimea brațului față de umăr (cotul îndoit).
    const chestFront = chestA.z + this.torso.rz, ahead = target.z - chestFront;
    // (`free` în signs.js: fără limitare, pentru semnele cu brațul întins înainte, ex. take)
    if (!free && ahead > BODY.reachFree) target.z = chestFront + BODY.reachFree + (ahead - BODY.reachFree) * BODY.reachGain;
    const fromShoulder = target.clone().sub(shoulderA), maxReach = BODY.reachMax * (this.len[s].upper + this.len[s].fore);
    if (!free && fromShoulder.length() > maxReach) target.copy(shoulderA).addScaledVector(fromShoulder.normalize(), maxReach);

    // punctele mâinii (pentru coliziuni), la scara palmei avatarului
    const handScale = this.len[s].palm / stats.palm[s];
    const points = handData
      ? [[0, BODY.palmRadius], ['palm', BODY.palmRadius], [4, BODY.tipRadius], [8, BODY.tipRadius],
         [12, BODY.tipRadius], [16, BODY.tipRadius], [20, BODY.tipRadius]].map(([i, r]) => {
          const q = i === 'palm' ? handData[0].clone().add(handData[5]).add(handData[17]).multiplyScalar(1 / 3) : handData[i];
          return { p: target.clone().addScaledVector(q.clone().sub(wrist), handScale), r };
        })
      : dataPoints.map(q => ({ p: target.clone().addScaledVector(q.clone().sub(wrist), handScale), r: BODY.palmRadius }));

    // direcția cotului: naturală (jos, ușor în afară și înapoi), plus puțin din date
    const sx = s === 'left' ? 1 : -1;
    const natural = BODY.pole.clone().setX(BODY.pole.x * sx).normalize();
    const dataPole = perp(seg(p[S], p[E]), seg(p[S], p[W]));
    const pole = dataPole.lengthSq() > 1e-10
      ? natural.multiplyScalar(1 - BODY.poleDataWeight).addScaledVector(dataPole.normalize(), BODY.poleDataWeight).normalize()
      : natural;
    return {
      target, points, pole, hand: handData,
      handDir: handData ? handData[9].clone().sub(handData[0]) : seg(p[W], mid(p[PINKY], p[INDEX])),
      handSide: handData ? handData[5].clone().sub(handData[17]) : seg(p[PINKY], p[INDEX]),
    };
  }

  moveHand(arm, delta) {
    arm.target.add(delta);
    for (const q of arm.points) q.p.add(delta);
  }

  volumes(body) {
    // volumele corpului pentru poza curentă a trunchiului și a capului
    const chestQ = body.eff.chest, chest = body.pos.chest;
    const head = this.head.center.clone().sub(this.pos.head).applyQuaternion(body.eff.head).add(body.pos.head);
    return { chestQ, chestInv: chestQ.clone().invert(), chest, head, neck: [body.pos.neck ?? body.pos.head, body.pos.head] };
  }

  penetration(p, r, vol, torsoMargin = 0) {
    // deplasarea minimă care scoate sfera (p, r) din corp, sau null; față de trunchi se mai
    // păstrează în plus `torsoMargin` (distanța minimă a palmei în fața pieptului)
    const T = this.torso;
    const q = p.clone().sub(vol.chest).applyQuaternion(vol.chestInv);
    let best = null;
    const consider = v => { if (v && (!best || v.lengthSq() > best.lengthSq())) best = v; };
    if (q.y > T.yBottom && q.y < T.yTop) {
      const ax = T.rx + r + torsoMargin, az = T.rz + r + torsoMargin, e = Math.hypot(q.x / ax, q.z / az);
      if (e < 1) {
        const target = e < 1e-6 ? new V3(0, q.y, az) : new V3(q.x / e, q.y, q.z / e);
        consider(target.sub(q).applyQuaternion(vol.chestQ));
      }
    }
    consider(pushFromCapsule(p, r, vol.neck[0], vol.neck[1], BODY.neckRadius));
    const toHead = p.clone().sub(vol.head), min = this.head.radius + r;
    if (toHead.length() < min) {
      if (toHead.lengthSq() < 1e-12) toHead.set(0, 0, 1);
      consider(toHead.normalize().multiplyScalar(min - p.distanceTo(vol.head)));
    }
    return best;
  }

  elbowInTorso(p, vol) {
    // cotul (cu mâneca) în trunchiul real: elipsa cu lățimea măsurată din lateral
    const T = this.torso, q = p.clone().sub(vol.chest).applyQuaternion(vol.chestInv), r = BODY.forearmRadius;
    if (q.y < T.yBottom || q.y > T.yTop) return false;
    return Math.hypot(q.x / (T.rxOuter + r), q.z / (T.rz + 0.03 + r)) < 1;
  }

  pushOutOfBody(arm, vol, contact = false) {
    // `contact`: semn cu mâna pe corp (signs.js) - fără distanța minimă față de piept
    for (let i = 0; i < 4; i++) {
      let worst = null;
      for (const q of arm.points) {
        const v = this.penetration(q.p, q.r, vol, q.r === BODY.palmRadius && !contact ? BODY.torsoClearance : 0);
        if (v && (!worst || v.lengthSq() > worst.lengthSq())) worst = v;
      }
      if (!worst) return;
      this.moveHand(arm, worst);
    }
  }

  separateHands(left, right) {
    // Mâinile se pot atinge (semne cu contact), dar nu trec una prin alta. Dacă se suprapun
    // mai mult lateral sau în adâncime decât pe verticală, mâna dreaptă trece prin față.
    for (let i = 0; i < 4; i++) {
      let worst = null;
      for (const a of left.points) for (const b of right.points) {
        const deficit = a.r + b.r - a.p.distanceTo(b.p);
        if (deficit > 0 && (!worst || deficit > worst.deficit)) worst = { deficit, a, b };
      }
      if (!worst) return;
      const n = worst.b.p.clone().sub(worst.a.p);
      if (n.lengthSq() < 1e-12 || Math.abs(n.clone().normalize().y) < 0.6) n.set(0, 0, 1);
      n.normalize();
      this.moveHand(right, n.clone().multiplyScalar(worst.deficit * BODY.dominantShare));
      this.moveHand(left, n.multiplyScalar(-worst.deficit * (1 - BODY.dominantShare)));
    }
  }

  solveIK(arm, s, body, vol) {
    // IK analitic cu două oase; cotul în planul dat de vectorul `pole`. Dacă cotul ar intra
    // în trunchi, se îndepărtează spre exterior.
    const shoulder = body.pos[s + 'UpperArm'], { upper: L1, fore: L2 } = this.len[s];
    const outward = new V3(s === 'left' ? 1 : -1, 0, 0);
    let pole = arm.pole.clone();
    for (let attempt = 0; attempt < 5; attempt++) {
      const toTarget = arm.target.clone().sub(shoulder);
      const d = clamp(toTarget.length(), Math.abs(L1 - L2) + 1e-3, L1 + L2 - 1e-3);
      const u = toTarget.normalize();
      const cosA = clamp((L1 * L1 + d * d - L2 * L2) / (2 * L1 * d), -1, 1);
      let side = perp(pole, u);
      if (side.lengthSq() < 1e-8) side = perp(new V3(0, -1, 0), u);
      side.normalize();
      arm.shoulder = shoulder.clone();
      arm.elbow = shoulder.clone().addScaledVector(u, L1 * cosA).addScaledVector(side, L1 * Math.sqrt(1 - cosA * cosA));
      arm.wrist = shoulder.clone().addScaledVector(u, d);
      arm.pole = pole.clone();
      if (arm.fixedElbow || (!this.penetration(arm.elbow, BODY.forearmRadius, vol) && !this.elbowInTorso(arm.elbow, vol))) return;
      pole.add(outward).normalize();
    }
  }

  armFrames(s, upper, fore, handSide, pole) {
    // Brațul: direcția umăr -> cot și planul în care se îndoaie cotul (cu `pole` ca rezervă
    // când brațul e aproape drept). Antebrațul, înainte de limite, preia răsucirea palmei.
    const a = upper.clone().normalize();
    let hint = perp(pole, a);
    if (hint.lengthSq() < 1e-8) hint = perp(new V3(0, 0, 1), a);
    const bend = perp(fore, a).addScaledVector(hint.normalize(), 0.05 * fore.length());
    const Du = basisQuat(this.dir[s + 'UpperArm'], new V3(0, 0, 1), upper, bend);
    const Dl = basisQuat(this.dir[s + 'LowerArm'], perp(this.side[s + 'Hand'], this.dir[s + 'LowerArm']),
                         fore, perp(handSide, fore));
    return { Du, Dl };
  }

  twistExcess(arm, s) {
    // cu cât depășește răsucirea antebrațului cerută de date limitele (grade, 0 = în limite)
    const upper = arm.elbow.clone().sub(arm.shoulder), fore = arm.wrist.clone().sub(arm.elbow);
    const { Du, Dl } = this.armFrames(s, upper, fore, arm.handSide, arm.pole);
    const t = this.forearmTwist(s, Du, fore, Dl).degrees;
    return t > LIMITS.forearm.sup ? t - LIMITS.forearm.sup : t < -LIMITS.forearm.pron ? t + LIMITS.forearm.pron : 0;
  }

  swivelForTwist(arm, s, body, vol) {
    // Ca la om: când antebrațul ar trebui să se răsucească peste limită (ex. palmele pe piept cu
    // coatele în lateral), restul rotației o face umărul: cotul se rotește în jurul axei
    // umăr -> încheietură (mâna rămâne pe loc). Se păstrează sensul care micșorează surplusul.
    let excess = this.twistExcess(arm, s), total = 0;
    // pași mici (jumătate din surplus): un pas mare poate împinge cotul în trunchi, iar IK-ul
    // îl mută atunci brusc pe partea cealaltă
    for (let i = 0; i < 6 && Math.abs(excess) > 1 && total < BODY.swivelMax; i++) {
      const step = degToRad(Math.min(0.5 * Math.abs(excess), BODY.swivelMax - total));
      const axis = arm.target.clone().sub(body.pos[s + 'UpperArm']).normalize();
      let best = null;
      for (const sign of [1, -1]) {
        const trial = { ...arm, pole: arm.pole.clone().applyAxisAngle(axis, sign * step) };
        this.solveIK(trial, s, body, vol);
        const e = this.twistExcess(trial, s);
        if (!best || Math.abs(e) < Math.abs(best.e)) best = { trial, e };
      }
      if (Math.abs(best.e) >= Math.abs(excess)) break;
      Object.assign(arm, best.trial);
      excess = best.e;
      total += THREE.MathUtils.radToDeg(step);
    }
  }

  orient(D, s, upper, fore, handDir, handSide, hand, pole, shape = null) {
    // Brațul și antebrațul (răsucirea limitată), încheietura și degetele
    const { Du, Dl: DlData } = this.armFrames(s, upper, fore, handSide, pole);
    const restSide = this.side[s + 'Hand'];
    const Dl = this.limitTwist(s, Du, fore, DlData);
    // Încheietura: doar flexie și înclinare laterală față de antebraț, în limite
    const wristLocal = handDir.clone().applyQuaternion(Dl.clone().invert());
    const Dh = Dl.clone().multiply(limitedSwing(wristLocal, this.dir[s + 'Hand'], this.normal[s], LIMITS.wrist));
    D.set(s + 'UpperArm', Du).set(s + 'LowerArm', Dl).set(s + 'Hand', Dh);
    // Degetele: unghiurile se măsoară față de palma din DATE (nu față de palma avatarului, care
    // poate fi limitată la încheietură), ca forma mâinii să nu depindă de limitele brațului.
    // Pentru fiecare deget: (1) unghiurile din date, pe lanțul NELIMITAT; (2) îndoirea totală se
    // păstrează (redistributeCurl); (3) rotațiile oaselor, cu limitele anatomice.
    const dataPalm = hand ? basisQuat(this.dir[s + 'Hand'], restSide, handDir, handSide) : Dh;
    for (const f of FINGERS) {
      const bones = fingerBones(s, f), limits = bones.map((_, i) => jointLimits(f, i));
      let angles;
      if (shape?.[f]) {
        // formă impusă (signs.js): [bază, mijloc, vârf, depărtare] / degetul mare [bază, apropiere, mijloc, vârf].
        // Axa laterală (e1 × e2) e spre degetul mic la mâna stângă și spre degetul mare la dreapta:
        // semnul se inversează la dreapta, ca valorile din signs.js să însemne același lucru.
        const a = shape[f].map(degToRad), sign = s === 'left' ? 1 : -1;
        // Degetul mare e rotit față de celelalte: articulațiile lui din mijloc și din vârf se îndoaie
        // spre degete (THUMB_ROLL față de normala palmei), nu drept în afara palmei.
        angles = f === 'Thumb'
          ? [{ flex: a[0], abd: sign * a[1] }, { flex: a[2], roll: sign * THUMB_ROLL }, { flex: a[3], roll: sign * THUMB_ROLL }]
          : [{ flex: a[0], abd: sign * a[3] }, { flex: a[1], abd: 0 }, { flex: a[2], abd: 0 }];
      } else if (hand && f !== 'Thumb' && this.fingerLen[s + f]) {
        const idx = FINGER_POINTS[f];
        angles = fitFinger(hand, idx, this.fingerLen[s + f], limits);
        // depărtarea laterală (din planul palmei, în limite); contează doar la degetul întins,
        // la mâna strânsă planul palmei din date e nesigur
        const proximal = hand[idx[1]].clone().sub(hand[idx[0]]).applyQuaternion(dataPalm.clone().invert());
        const abd = clamp(swingAngles(proximal, this.dir[bones[0]], this.normal[s]).abd,
                          degToRad(limits[0].abd[0]), degToRad(limits[0].abd[1]));
        angles[0].abd = abd * Math.max(0, Math.cos(angles[0].flex));
      } else if (hand) {
        // degetul mare (se mișcă și lateral, nu doar spre palmă): unghiurile față de palma din date
        const idx = FINGER_POINTS[f];
        let frame = dataPalm;
        angles = bones.map((b, i) => {
          if (!this.dir[b]) return null;
          const target = hand[idx[i + 1]].clone().sub(hand[idx[i]]).applyQuaternion(frame.clone().invert());
          frame = frame.clone().multiply(arcQuat(this.dir[b], target));
          return swingAngles(target, this.dir[b], this.normal[s], true);
        });
        redistributeCurl(angles, limits);
      } else {
        angles = RELAXED_FLEX.map(deg => ({ flex: degToRad(deg), abd: 0 }));
      }
      let parent = Dh;
      bones.forEach((b, i) => {
        if (!this.dir[b] || !angles[i]) return;
        let local;
        if (angles[i].roll !== undefined) {
          // îndoire în planul rotit cu `roll` în jurul osului (doar pentru formele impuse)
          const e1 = this.dir[b], e2 = perp(this.normal[s], e1).normalize(), e3 = new V3().crossVectors(e1, e2);
          const bend = e2.multiplyScalar(Math.cos(angles[i].roll)).addScaledVector(e3, Math.sin(angles[i].roll));
          local = arcQuat(e1, e1.clone().multiplyScalar(Math.cos(angles[i].flex)).addScaledVector(bend, Math.sin(angles[i].flex)));
        } else {
          local = swingFromAngles(this.dir[b], this.normal[s], angles[i], limits[i]);
        }
        parent = parent.clone().multiply(local);
        D.set(b, parent);
      });
    }
  }

  forearmTwist(s, Du, fore, Dl) {
    // Răsucirea antebrațului (grade, + = supinație) față de cot: comparată cu antebrațul care
    // doar se îndoaie din cot (balama), fără să se rotească față de braț.
    const axis = fore.clone().normalize();
    const hinge = arcQuat(this.dir[s + 'LowerArm'].clone().applyQuaternion(Du), axis).multiply(Du);
    const side = this.side[s + 'Hand'];
    const a = perp(side.clone().applyQuaternion(hinge), axis).normalize();
    const b = perp(side.clone().applyQuaternion(Dl), axis).normalize();
    const angle = Math.atan2(new V3().crossVectors(a, b).dot(axis), a.dot(b));
    // la brațul stâng (spre +X) supinația e o rotație negativă în jurul antebrațului; la drept, pozitivă
    return { hinge, axis, degrees: THREE.MathUtils.radToDeg(s === 'left' ? -angle : angle) };
  }

  limitTwist(s, Du, fore, Dl) {
    // Pronația / supinația antebrațului limitate la LIMITS.forearm (fără răsuciri imposibile când
    // MediaPipe inversează orientarea palmei sau la trecerea prin repaus)
    const { hinge, axis, degrees } = this.forearmTwist(s, Du, fore, Dl);
    if (this.twistLog) this.twistLog.push(degrees);
    const limited = clamp(degrees, -LIMITS.forearm.pron, LIMITS.forearm.sup);
    if (limited === degrees) return Dl;
    return new Q().setFromAxisAngle(axis, degToRad(s === 'left' ? -limited : limited)).multiply(hinge);
  }

  // ---------------------------------------------------------------- aplicare pe personaj
  apply(pose) {
    // rotația din scenă a fiecărui os = delta × repaus; osului i se dă rotația față de părinte
    const world = new Map();
    for (const name of BONE_ORDER) {
      const node = this.node(name);
      if (!node) continue;
      let parentQ = world.get(node.parent);
      if (!parentQ) {
        node.parent.updateWorldMatrix(true, false);
        parentQ = node.parent.getWorldQuaternion(new Q());
      }
      const delta = pose.D.get(name);
      const q = delta ? delta.clone().multiply(this.restQ[name]) : parentQ.clone();   // fără delta: urmează părintele
      node.quaternion.copy(parentQ.clone().invert().multiply(q));
      world.set(node, q);
    }
    this.vrm.expressionManager?.setValue('aa', pose.mouth);
  }

  // ---------------------------------------------------------------- puncte MediaPipe (back-translation)
  landmarks(D) {
    // Poza avatarului ca puncte MediaPipe, în coordonatele scenei (m): Pose 33 (doar nasul,
    // umerii, coatele și încheieturile; restul primesc poziția nasului) și Hands 21 pe fiecare
    // mână. src/avatar/evaluate_avatar.py le clasifică cu modelul de recunoaștere; punctele
    // Pose ale degetelor (17-22) le calculează acolo din punctele mâinii.
    const { pos, eff } = this.fk(D);
    const at = (bone, offset) => offset.clone().applyQuaternion(eff[bone]).add(pos[bone]);
    const nose = at('head', this.noseOffset);
    const pose = Array.from({ length: 33 }, () => nose);
    const hands = {};
    for (const s of SIDES) {
      const [S, E, W] = ARM_POINTS[s];
      pose[S] = pos[s + 'UpperArm']; pose[E] = pos[s + 'LowerArm']; pose[W] = pos[s + 'Hand'];
      hands[s] = [pos[s + 'Hand'], ...FINGERS.flatMap(f => {
        const bones = fingerBones(s, f);
        return [...bones.map(b => pos[b]), at(bones[2], this.tipOffset[bones[2]])];
      })];
    }
    const round = v => [v.x, v.y, v.z].map(x => Math.round(x * 1e4) / 1e4);
    return { pose: pose.map(round), left_hand: hands.left.map(round), right_hand: hands.right.map(round) };
  }

  // ---------------------------------------------------------------- verificare (?check)
  check(motion, stats) {
    // După aplicarea fiecărui cadru: pătrunderea maximă a mâinilor în corp, distanța minimă
    // dintre mâini și dintre antebrațe, unghiurile maxime ale falangelor (trebuie să respecte LIMITS).
    const report = { bodyPenetrationCm: 0, minHandGapCm: Infinity, minForearmGapCm: Infinity, maxFlex: {}, maxAbs: {} };
    const tips = s => ['Hand', 'ThumbDistal', 'IndexDistal', 'MiddleDistal', 'RingDistal', 'LittleDistal'].map(b => s + b);
    this.twistLog = [];
    this.restPose();
    report.restTwist = this.twistLog.map(Math.round);   // [stânga, dreapta]: ~0 = neutru
    const twists = this.twistLog = [];
    for (const frame of motion.frames) {
      const pose = this.framePose(frame, stats);
      this.apply(pose);
      const body = this.fk(pose.D), vol = this.volumes(body);
      for (const [k, s] of SIDES.entries()) {
        if (frame.weight[k] < 1) continue;
        for (const b of tips(s)) {
          const v = body.pos[b] && this.penetration(body.pos[b], 0, vol);
          if (v) report.bodyPenetrationCm = Math.max(report.bodyPenetrationCm, v.length() * 100);
        }
        for (const f of FINGERS) fingerBones(s, f).forEach((b, i) => {
          const parent = i === 0 ? s + 'Hand' : fingerBones(s, f)[i - 1];
          const local = pose.D.get(parent).clone().invert().multiply(pose.D.get(b));
          const e1 = this.dir[b], e2 = perp(this.normal[s], e1).normalize(), e3 = new V3().crossVectors(e1, e2);
          const d = e1.clone().applyQuaternion(local);
          const key = f === 'Thumb' ? ['thumbCmc', 'thumbMcp', 'thumbIp'][i] : ['mcp', 'pip', 'dip'][i];
          const flex = THREE.MathUtils.radToDeg(Math.atan2(d.dot(e2), d.dot(e1)));
          const abd = THREE.MathUtils.radToDeg(Math.asin(clamp(d.dot(e3), -1, 1)));
          report.maxFlex[key] = [Math.min(report.maxFlex[key]?.[0] ?? 0, Math.round(flex)), Math.max(report.maxFlex[key]?.[1] ?? 0, Math.round(flex))];
          report.maxAbs[key] = Math.max(report.maxAbs[key] ?? 0, Math.round(Math.abs(abd)));
        });
      }
      if (frame.weight[0] >= 1 && frame.weight[1] >= 1) {
        for (const a of tips('left')) for (const b of tips('right')) {
          report.minHandGapCm = Math.min(report.minHandGapCm, body.pos[a].distanceTo(body.pos[b]) * 100);
        }
        const { distance } = segmentDistance(body.pos.leftLowerArm, body.pos.leftHand, body.pos.rightLowerArm, body.pos.rightHand);
        report.minForearmGapCm = Math.min(report.minForearmGapCm, distance * 100);
      }
    }
    this.twistLog = null;
    // răsucirea antebrațului cerută de date (înainte de limitare), grade: minim, mediană, maxim
    const sorted = twists.sort((a, b) => a - b);
    report.forearmTwist = sorted.length ? [sorted[0], sorted[sorted.length >> 1], sorted[sorted.length - 1]].map(Math.round) : null;
    for (const k of ['bodyPenetrationCm', 'minHandGapCm', 'minForearmGapCm']) report[k] = Math.round(report[k] * 10) / 10;
    return report;
  }
}
