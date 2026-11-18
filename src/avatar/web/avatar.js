// Avatarul 3D care execută semnele: scena, încărcarea personajului VRM, redarea mișcărilor
// exportate de src/avatar/export_motion.py și netezirea în timp. Calculul rotațiilor oaselor
// (IK, coliziuni, limite anatomice) e în retarget.js.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { ARM_POINTS, BONE_ORDER, Retargeter, SIDES, isHair, mid, seg, vec } from './retarget.js';
import { HANDSHAPES, SIGNS } from './signs.js';
import { tightenShirt } from './outfit.js';

const V3 = THREE.Vector3;

const TRANSITION = 0.45;   // s: din repaus până la primul cadru al semnului și înapoi
const HOLD = 0.6;          // s: pauză în repaus după semn (la redarea în buclă)
// Netezire exponențială a rotațiilor (constanta de timp, s): elimină tremurul și smuciturile
// dintre cadre; întârzierea introdusă e de ordinul constantei.
const SMOOTHING = 0.05;
const HAND_EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10],
  [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];

const params = new URLSearchParams(location.search);
const ui = Object.fromEntries(['word', 'play', 'loop', 'speed', 'points', 'status', 'subtitle']
  .map(id => [id, document.getElementById(id)]));

// ---------------------------------------------------------------- scena
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
document.getElementById('scene').appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xeef1f5);
const camera = new THREE.PerspectiveCamera(30, innerWidth / innerHeight, 0.1, 20);
const controls = new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.HemisphereLight(0xffffff, 0x8a90a0, 1.8));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(0.5, 1.5, 2);
scene.add(sun);
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  if (retarget && !params.has('shape')) frameCamera();
});

let vrm = null, retarget = null, hairCut = null;

function updateHairCut() {
  if (hairCut) hairCut.plane.constant = -(hairCut.head.getWorldPosition(new V3()).y + hairCut.offset);
}

function shortenHair(model) {
  // Păr scurt (bob, până sub bărbie): părul lung al personajului de probă acoperea mâinile la
  // semnele de lângă umăr (back). Planul de tăiere e aplicat doar materialelor de păr și urcă /
  // coboară odată cu capul. ?hair=long păstrează părul original.
  renderer.localClippingEnabled = true;
  const plane = new THREE.Plane(new V3(0, 1, 0), 0);
  model.scene.traverse(o => {
    if (!o.isMesh || !isHair(o)) return;
    for (const m of [o.material].flat()) { m.clippingPlanes = [plane]; m.clipping = true; m.needsUpdate = true; }
  });
  const head = model.humanoid.getNormalizedBoneNode('head');
  // înălțimea tăieturii față de os: puțin sub bărbie
  return { plane, head, offset: retarget.anchors.chin.offset.y - 0.025 };
}

// ---------------------------------------------------------------- cronologia unui semn
function blendPose(a, b, f) {
  // Interpolare în spațiul articulațiilor: se interpolează rotația fiecărui os față de părinte
  // (ex. răsucirea antebrațului față de braț), nu rotația lui în scenă. Altfel brațul și
  // antebrațul se rotesc independent în tranziții și antebrațul trece prin răsuciri imposibile.
  const D = new Map();
  for (const name of BONE_ORDER) {
    const qa = a.D.get(name);
    if (!qa) continue;
    const qb = b.D.get(name) ?? qa, p = retarget.parent[name];
    const pa = a.D.get(p), pb = b.D.get(p), pd = D.get(p);
    if (pa && pb && pd) {
      const la = pa.clone().invert().multiply(qa), lb = pb.clone().invert().multiply(qb);
      D.set(name, pd.clone().multiply(la.slerp(lb, f)));
    } else {
      D.set(name, qa.clone().slerp(qb, f));
    }
  }
  return { D, mouth: a.mouth + (b.mouth - a.mouth) * f };
}

const POSE_ARM = { left: [13, 15, 17, 19, 21], right: [14, 16, 18, 20, 22] };   // cot, încheietură, degete (Pose)

function applyHolds(motion, spec) {
  // signs.js: `hold: t` = mâna stă pe loc toată durata, în poza din date de la momentul t;
  // `rest: true` = mâna nu participă; `active: true` = mâna participă tot semnul
  const n = motion.frames.length;
  for (const [k, s] of SIDES.entries()) {
    const hs = spec[s];
    if (!hs) continue;
    if (hs.hold !== undefined) {
      const src = motion.frames[Math.round(hs.hold * (n - 1))];
      for (const f of motion.frames) {
        if (f.pose && src.pose) for (const i of POSE_ARM[s]) f.pose[i] = src.pose[i].slice();
        f[s + '_hand'] = src[s + '_hand'];
        f.weight[k] = src.weight[k];
      }
    }
    if (hs.rest) for (const f of motion.frames) f.weight[k] = 0;
    if (hs.active) for (const f of motion.frames) f.weight[k] = 1;
  }
}

function resampleMotion(motion, fps = 30) {
  // Cadrele din date sunt ~10 pe secundă; la 30 se văd mișcările rapide adăugate din signs.js
  // (tremur, bătăi) și interpolarea dintre cadre e mai fină.
  const n = motion.frames.length;
  if (n < 2) return motion;
  const duration = (n - 1) / motion.fps, m = Math.max(2, Math.round(duration * fps) + 1);
  const mix = (a, b, f) => (a && b) ? a.map((p, i) => p.map((x, j) => x + (b[i][j] - x) * f)) : (f < 0.5 ? (a ?? b) : (b ?? a));
  const frames = [];
  for (let k = 0; k < m; k++) {
    const x = k / (m - 1) * (n - 1), i = Math.min(Math.floor(x), n - 2), f = x - i;
    const A = motion.frames[i], B = motion.frames[i + 1];
    const face = A.face && B.face ? Object.fromEntries(Object.keys(A.face).map(key => [key, mix([A.face[key]], [B.face[key]], f)[0]]))
      : (f < 0.5 ? A.face : B.face);
    frames.push({ pose: mix(A.pose, B.pose, f), face, left_hand: mix(A.left_hand, B.left_hand, f),
                  right_hand: mix(A.right_hand, B.right_hand, f), weight: A.weight.map((w, j) => w + (B.weight[j] - w) * f) });
  }
  return { ...motion, frames, fps: (m - 1) / duration };
}

function buildTimeline(original) {
  const spec = SIGNS[original.word] ?? {};
  const copy = JSON.parse(JSON.stringify(original));
  applyHolds(copy, spec);
  const motion = resampleMotion(copy);
  const stats = retarget.motionStats(motion);
  const rest = retarget.rest, n = motion.frames.length;
  const context = i => ({ spec, t: n > 1 ? i / (n - 1) : 0, time: i / motion.fps });
  let pivot = null;
  if (SIDES.some(s => spec[s]?.pivot)) {
    // cotul fix (signs.js `pivot`): poziția lui la mijlocul semnului
    const collect = {}, mid = Math.floor(n / 2);
    retarget.framePose(motion.frames[mid], stats, { ...context(mid), collect });
    pivot = collect;
  }
  const keys = [{ t: 0, pose: rest, ease: true }];
  motion.frames.forEach((frame, i) => keys.push({
    t: TRANSITION + i / motion.fps, pose: retarget.framePose(frame, stats, { ...context(i), pivot }), frame,
    ease: i === motion.frames.length - 1 }));
  const end = keys[keys.length - 1].t + TRANSITION;
  keys.push({ t: end, pose: rest }, { t: end + HOLD, pose: rest });
  return { keys, duration: end + HOLD, motion, stats };
}

function sample(timeline, t) {
  const keys = timeline.keys;
  let i = 0;
  while (i < keys.length - 2 && keys[i + 1].t <= t) i++;
  const a = keys[i], b = keys[i + 1];
  let f = THREE.MathUtils.clamp((t - a.t) / (b.t - a.t), 0, 1);
  if (a.ease) f = f * f * (3 - 2 * f);   // tranziție lină din/în repaus
  return { pose: blendPose(a.pose, b.pose, f), frame: f < 0.5 ? a.frame : b.frame };
}

// ---------------------------------------------------------------- punctele din date (verificare)
// Scheletul MediaPipe al înregistrării, desenat lângă avatar: brațe gri, mâna stângă
// a semnatarului albastră, dreapta portocalie. Arată dacă avatarul urmează corect datele.
const pointsGroup = new THREE.Group();
scene.add(pointsGroup);
const lineSets = ['#8a8f99', '#2f6fde', '#e8772e'].map(color => {
  const lines = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color }));
  lines.frustumCulled = false;   // geometria se schimbă la fiecare cadru
  pointsGroup.add(lines);
  return lines;
});

function drawPoints(frame, stats) {
  pointsGroup.visible = ui.points.checked && !!frame;
  if (!pointsGroup.visible || !frame.pose) return;
  const p = frame.pose, center = mid(p[11], p[12]), scale = retarget.shoulderWidth / stats.shoulderWidth;
  const place = q => vec(q).sub(vec(center)).multiplyScalar(scale);
  lineSets[0].geometry.setFromPoints([[11, 12], [11, 13], [13, 15], [12, 14], [14, 16]]
    .flatMap(([a, b]) => [place(p[a]), place(p[b])]));
  SIDES.forEach((s, k) => {
    const hand = frame[s + '_hand'];
    // z-ul mâinii e relativ la încheietură: îl aducem la z-ul încheieturii din corp
    const shift = hand ? p[ARM_POINTS[s][2]][2] - hand[0][2] : 0;
    const h = hand ? hand.map(q => [q[0], q[1], q[2] + shift]) : [];
    lineSets[k + 1].geometry.setFromPoints(hand ? HAND_EDGES.flatMap(([a, b]) => [place(h[a]), place(h[b])]) : []);
  });
}

// ---------------------------------------------------------------- back-translation (evaluare)
async function backtranslate(words) {
  // Redă fiecare semn exact ca pe ecran (tranziții + netezire, la 30 de cadre/s) și trimite
  // punctele MediaPipe ale avatarului la src/avatar/evaluate_avatar.py, care le clasifică.
  const fps = 30, dt = 1 / fps;
  for (const word of words) {
    const motion = await (await fetch(`/motion/${word}.json`)).json();
    const tl = buildTimeline(motion);
    const frames = [];
    let shown = null;
    for (let t = 0; t <= tl.duration; t += dt) {
      const s = sample(tl, t);
      shown = shown ? blendPose(shown, s.pose, 1 - Math.exp(-dt / SMOOTHING)) : s.pose;
      frames.push(retarget.landmarks(shown.D));
    }
    await fetch(`/backtranslate/${word}`, { method: 'POST', body: JSON.stringify({ word, fps, frames }) });
  }
}

// ---------------------------------------------------------------- redare
let timeline = null, t = 0, playing = false, shown = null, blinkTimer = 2;

async function loadWord(word) {
  const motion = await (await fetch(`/motion/${word}.json`)).json();
  const started = performance.now();
  timeline = buildTimeline(motion);
  console.info(`semnul ${word} calculat în ${Math.round(performance.now() - started)} ms`);
  ui.subtitle.textContent = motion.meaning;
  ui.status.textContent = `${motion.word}: ${motion.source}, încredere ${motion.confidence}, ${motion.frames.length} cadre`;
  t = 0;
  playing = true;
}

function frameCamera() {
  // Încadrează spațiul semnelor (de deasupra capului până sub șolduri, cu brațele întinse);
  // cu „puncte” bifat, și scheletul din date, din dreapta avatarului.
  const showPoints = ui.points.checked, P = retarget.pos;
  const top = P.head.y + 0.35, bottom = P.hips.y - 0.1, cy = (top + bottom) / 2;
  const halfWidth = showPoints ? 1.0 : 0.6, cx = showPoints ? 0.4 : 0;
  const tan = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
  const distance = Math.max((top - bottom) / 2 / tan, halfWidth / (tan * camera.aspect));
  camera.position.set(cx, cy, distance * 1.05);
  controls.target.set(cx, cy, 0);
  pointsGroup.position.set(0.85, P.leftUpperArm.y, 0);
  controls.update();
}

const timer = new THREE.Timer();
function animate() {
  requestAnimationFrame(animate);
  timer.update();
  const dt = Math.min(timer.getDelta(), 0.1);
  if (vrm) {
    if (timeline && playing) {
      t += dt * Number(ui.speed.value);
      if (t >= timeline.duration) {
        if (ui.loop.checked) t = 0;
        else { t = timeline.duration; playing = false; }
      }
    }
    const s = timeline ? sample(timeline, t) : { pose: retarget.rest, frame: null };
    // netezire: la cadru fix (?t) poza exactă, altfel apropiere exponențială de ținta curentă
    const alpha = params.has('t') || !shown ? 1 : 1 - Math.exp(-dt / SMOOTHING);
    shown = shown ? blendPose(shown, s.pose, alpha) : s.pose;
    retarget.apply(shown);
    drawPoints(s.frame, timeline?.stats);
    blinkTimer -= dt;   // clipit natural, la ~4 s
    if (blinkTimer < 0) blinkTimer = 3 + Math.random() * 2;
    vrm.expressionManager?.setValue('blink', blinkTimer < 0.15 ? 1 : 0);
    vrm.update(dt);
    updateHairCut();
  }
  renderer.render(scene, camera);
}

async function main() {
  animate();
  const loader = new GLTFLoader();
  loader.register(parser => new VRMLoaderPlugin(parser));
  const gltf = await loader.loadAsync('/avatar.vrm');
  const model = gltf.userData.vrm;
  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.combineSkeletons(gltf.scene);
  VRMUtils.rotateVRM0(model);   // modelele VRM 0.x stau invers; după asta toate privesc spre +Z
  model.scene.traverse(o => { o.frustumCulled = false; });
  scene.add(model.scene);
  if (params.get('shirt') !== 'loose') tightenShirt(model);   // ?shirt=loose: tricoul original
  retarget = new Retargeter(model);
  if (model.lookAt) model.lookAt.target = camera;   // privește spre pacient
  vrm = model;
  if (params.get('hair') !== 'long') hairCut = shortenHair(model);

  if (params.has('shape')) {
    // Previzualizare a unei forme a mâinii (signs.js): mâna dreaptă ridicată în fața umărului,
    // palma spre privitor (sau ?view=side: din lateral), camera aproape de mână.
    const D = new Map(retarget.rest.D);
    const vec3 = (name, fallback) => params.has(name) ? new V3(...params.get(name).split(',').map(Number)) : fallback;
    // antebrațul în prelungirea mâinii (încheietura dreaptă), ca orientarea dată să nu fie limitată
    retarget.orient(D, 'right', new V3(-0.25, -0.75, 0.55), vec3('dir', new V3(0.05, 1, 0.05)), vec3('dir', new V3(0, 1, 0)),
                    vec3('side', new V3(1, 0, 0)), null, new V3(-1, 0, 0), HANDSHAPES[params.get('shape')]);
    timeline = { keys: [{ t: 0, pose: { D, mouth: 0 } }, { t: 1, pose: { D, mouth: 0 } }], duration: 1 };
    retarget.apply({ D, mouth: 0 });
    vrm.scene.updateMatrixWorld(true);
    const hand = retarget.node('rightHand').getWorldPosition(new V3()).add(new V3(0, 0.06, 0));
    const side = params.get('view') === 'side';
    camera.position.copy(hand).add(side ? new V3(-0.5, 0, 0.05) : new V3(0, 0, 0.55));
    controls.target.copy(hand);
    controls.update();
    ui.status.textContent = 'forma ' + params.get('shape');
    document.body.dataset.ready = '1';
    return;
  }

  const index = await (await fetch('/motion/index.json')).json();
  for (const [word, info] of Object.entries(index)) ui.word.add(new Option(`${info.meaning} (${word})`, word));
  ui.word.value = params.get('word') ?? Object.keys(index)[0];
  ui.word.onchange = () => loadWord(ui.word.value);
  ui.play.onclick = () => { t = 0; playing = true; };
  // trecerea la semnul anterior / următor (butoane sau taste), cu redare automată
  const step = delta => {
    const n = ui.word.options.length;
    ui.word.selectedIndex = (ui.word.selectedIndex + delta + n) % n;
    document.activeElement?.blur();   // spațiul să nu mai „apese” butonul rămas selectat
    loadWord(ui.word.value);
  };
  document.getElementById('prev').onclick = () => step(-1);
  document.getElementById('next').onclick = () => step(1);
  addEventListener('keydown', e => {
    if (e.target.tagName === 'SELECT') return;   // în listă, săgețile o folosesc pe ea
    if (e.key === 'ArrowRight' || e.key === 'n' || e.key === 'N') step(1);
    else if (e.key === 'ArrowLeft' || e.key === 'p' || e.key === 'P') step(-1);
    else if (e.key === ' ') { e.preventDefault(); ui.play.onclick(); }
  });
  if (params.has('points')) ui.points.checked = true;
  ui.points.onchange = frameCamera;
  frameCamera();
  await loadWord(ui.word.value);
  if (params.has('t')) { t = Number(params.get('t')); playing = false; }   // cadru fix, pentru capturi
  if (params.has('check')) {
    // Verificare: pentru fiecare cuvânt din ?check=a,b (sau doar cel curent), raportul din consolă
    const words = params.get('check') ? params.get('check').split(',') : [ui.word.value];
    for (const word of words) {
      const motion = await (await fetch(`/motion/${word}.json`)).json();
      console.log('CHECK ' + word + ' ' + JSON.stringify(retarget.check(motion, retarget.motionStats(motion))));
    }
  }
  if (params.has('bodycheck')) {
    // ?bodycheck=a,b (gol = toate): pătrunderea maximă a mâinilor în corp (cm) pe durata semnului
    const words = params.get('bodycheck') ? params.get('bodycheck').split(',') : Object.keys(index);
    for (const word of words) {
      const tl = buildTimeline(await (await fetch(`/motion/${word}.json`)).json());
      let worst = 0, at = 0;
      for (const key of tl.keys) {
        const cm = retarget.bodyReport(key.pose.D);
        if (cm > worst) { worst = cm; at = key.t; }
      }
      console.log(`BODY ${word} ${worst.toFixed(1)} ${at.toFixed(2)}`);
    }
    console.log('BODY _done');
  }
  if (params.has('render')) {
    // ?render=a,b&at=0.3,0.6: imaginile semnelor la fracțiunile date din durata lor, trimise
    // serverului (POST /render/<cuvânt>_<at>); pentru verificare și pentru figuri
    const ats = (params.get('at') ?? '0.5').split(',').map(Number);
    ui.points.checked = false;
    frameCamera();
    if (params.has('zoom')) {
      // aproape de piept sau de față (pentru detalii: degete, contactul cu corpul)
      const target = (params.get('zoom') === 'face' ? retarget.pos.head : retarget.pos.chest).clone();
      camera.position.sub(controls.target).multiplyScalar(0.45).add(target);
      controls.target.copy(target);
      camera.lookAt(target);
    }
    if (params.get('view') === 'side') {
      // din trei sferturi, din dreapta avatarului: se vede adâncimea (mâna în fața corpului sau în el)
      const target = controls.target, d = camera.position.distanceTo(target);
      camera.position.set(target.x - 0.75 * d, target.y, target.z + 0.66 * d);
      camera.lookAt(target);
    }
    for (const word of params.get('render').split(',')) {
      const tl = buildTimeline(await (await fetch(`/motion/${word}.json`)).json());
      for (const at of ats) {
        const pose = sample(tl, TRANSITION + at * (tl.duration - 2 * TRANSITION - HOLD)).pose;
        retarget.apply(pose);
        vrm.update(0);
        updateHairCut();
        ui.subtitle.textContent = `${word} ${at}`;
        renderer.render(scene, camera);
        await fetch(`/render/${word}_${at}`, { method: 'POST', body: renderer.domElement.toDataURL('image/png') });
      }
    }
    await fetch('/render/_done', { method: 'POST', body: '' });
  }
  if (params.has('backtranslate')) {
    // ?backtranslate=a,b (gol = toate cuvintele); folosit de evaluate_avatar.py
    const words = params.get('backtranslate') ? params.get('backtranslate').split(',') : Object.keys(index);
    await backtranslate(words);
    await fetch('/backtranslate/_done', { method: 'POST', body: '{}' });
  }
  document.body.dataset.ready = '1';
}

main().catch(err => {
  ui.status.textContent = 'Eroare: ' + err.message;
  console.error(err);
  if (params.has('backtranslate')) fetch('/backtranslate/_error', { method: 'POST', body: JSON.stringify({ error: String(err.stack ?? err) }) });
});
