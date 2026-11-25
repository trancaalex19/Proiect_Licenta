// Avatarul 3D care execută semnele: scena, încărcarea personajului VRM, redarea mișcărilor
// exportate de src/avatar/export_motion.py și netezirea în timp. Calculul rotațiilor oaselor
// (IK, coliziuni, limite anatomice) e în retarget.js.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { ARM_POINTS, BONE_ORDER, Retargeter, SIDES, isHair, mid, seg, vec } from './retarget.js';
import { HANDSHAPES, SIGNS } from './signs.js';
import { SENTENCES } from './sentences.js';
import { tightenShirt } from './outfit.js';

const V3 = THREE.Vector3;

const TRANSITION = 0.45;   // s: din repaus până la primul cadru al semnului și înapoi
const HOLD = 0.6;          // s: pauză în repaus după semn (la redarea în buclă)
// Într-o propoziție, din poza finală a unui semn în cea de început a următorului: mâinile se
// deplasează cu ~HAND_SPEED (m/s), dar tranziția durează cel puțin / cel mult BETWEEN (s)
const HAND_SPEED = 1.0;
const BETWEEN = [0.25, 0.6];
// Netezire exponențială a rotațiilor (constanta de timp, s): elimină tremurul și smuciturile
// dintre cadre; întârzierea introdusă e de ordinul constantei.
const SMOOTHING = 0.05;
const HAND_EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10],
  [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];

const params = new URLSearchParams(location.search);
if (params.has('live')) document.body.classList.add('live');   // prototipul: interfața completă (index.html)
const ui = Object.fromEntries(['word', 'sentence', 'play', 'loop', 'speed', 'points', 'status', 'subtitle']
  .map(id => [id, document.getElementById(id)]));

// ---------------------------------------------------------------- scena
// în prototip canvasul e transparent: fundalul scenei (al interpretului) vine din pagină
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: params.has('live') });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
// canvasul ia mărimea containerului #scene (tot ecranul, sau doar scena în modul ?live)
const sceneEl = document.getElementById('scene');
const sceneSize = () => [sceneEl.clientWidth || innerWidth, sceneEl.clientHeight || innerHeight];
renderer.setSize(...sceneSize());
sceneEl.appendChild(renderer.domElement);
const scene = new THREE.Scene();
if (!params.has('live')) scene.background = new THREE.Color(0xeef1f5);
const camera = new THREE.PerspectiveCamera(30, sceneSize()[0] / sceneSize()[1], 0.1, 20);
const controls = new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.HemisphereLight(0xffffff, 0x8a90a0, 1.8));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(0.5, 1.5, 2);
scene.add(sun);
new ResizeObserver(() => {
  const [width, height] = sceneSize();
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
  if (retarget && !params.has('shape')) frameCamera();
}).observe(sceneEl);

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

async function timelineFor(item) {
  // pentru verificări (?render, ?bodycheck): un cuvânt sau o propoziție scrisă „a-b-c”
  const motions = await Promise.all(item.split('-').map(async w => (await fetch(`/motion/${w}.json`)).json()));
  return buildTimeline(...motions);
}

function signKeys(original) {
  // Pozele avatarului pentru fiecare cadru al unui semn (t de la 0, fără repaus)
  const spec = SIGNS[original.word] ?? {};
  const copy = JSON.parse(JSON.stringify(original));
  applyHolds(copy, spec);
  const motion = resampleMotion(copy);
  const stats = retarget.motionStats(motion);
  const n = motion.frames.length;
  const context = i => ({ spec, t: n > 1 ? i / (n - 1) : 0, time: i / motion.fps });
  let pivot = null;
  if (SIDES.some(s => spec[s]?.pivot)) {
    // cotul fix (signs.js `pivot`): poziția lui la mijlocul semnului
    const collect = {}, mid = Math.floor(n / 2);
    retarget.framePose(motion.frames[mid], stats, { ...context(mid), collect });
    pivot = collect;
  }
  const keys = motion.frames.map((frame, i) => ({
    t: i / motion.fps, pose: retarget.framePose(frame, stats, { ...context(i), pivot }), frame, stats }));
  // signs.js `enter` / `exit`: poza de început / de final cu mâna mutată (decalaj în m pe fiecare
  // parte), prin care trece tranziția din / spre repaus, ca brațul să ocolească trunchiul
  const waypoint = (offsets, i) => {
    if (!offsets) return null;
    const moved = JSON.parse(JSON.stringify(spec));
    for (const [side, o] of Object.entries(offsets)) {
      const base = moved[side]?.offset ?? [0, 0, 0];
      moved[side] = { ...moved[side], offset: base.map((x, j) => x + o[j]) };
    }
    return { pose: retarget.framePose(motion.frames[i], stats, { ...context(i), spec: moved, pivot }), frame: keys[i].frame, stats };
  };
  return { keys, length: (n - 1) / motion.fps, motion, stats, enter: waypoint(spec.enter, 0), exit: waypoint(spec.exit, n - 1) };
}

function transitionTime(a, b) {
  // cât durează trecerea între două poze: după cea mai mare distanță parcursă de o mână
  const pa = retarget.fk(a.D).pos, pb = retarget.fk(b.D).pos;
  const distance = Math.max(...SIDES.map(s => pa[s + 'Hand'].distanceTo(pb[s + 'Hand'])));
  return THREE.MathUtils.clamp(distance / HAND_SPEED, ...BETWEEN);
}

function buildTimeline(...originals) {
  // Unul sau mai multe semne (o propoziție): din repaus în primul semn, apoi din poza finală a
  // fiecărui semn direct în poza de început a următorului (fără repaus între ele; durata după
  // distanța parcursă de mâini, vezi HAND_SPEED),
  // iar la sfârșit înapoi în repaus. `spans` = intervalul fiecărui semn (pentru subtitrare).
  const rest = retarget.rest, keys = [{ t: 0, pose: rest, ease: true }], spans = [];
  let start = TRANSITION, first = null, last = null;
  originals.forEach((original, k) => {
    const sign = signKeys(original);
    first ??= sign;
    last = sign;
    if (k === 0 && sign.enter) keys.push({ ...sign.enter, t: TRANSITION / 2, ease: true });
    if (k > 0) start += transitionTime(keys[keys.length - 1].pose, sign.keys[0].pose);
    for (const key of sign.keys) keys.push({ ...key, t: start + key.t, word: k });
    keys[keys.length - 1].ease = true;   // tranziție lină spre următorul semn / repaus
    spans.push([start, start + sign.length]);
    start += sign.length;
  });
  if (last.exit) keys.push({ ...last.exit, t: start + TRANSITION / 2, ease: true });
  const end = start + TRANSITION;
  keys.push({ t: end, pose: rest }, { t: end + HOLD, pose: rest });
  return { keys, duration: end + HOLD, motion: first.motion, stats: first.stats, spans };
}

function sample(timeline, t) {
  const keys = timeline.keys;
  let i = 0;
  while (i < keys.length - 2 && keys[i + 1].t <= t) i++;
  const a = keys[i], b = keys[i + 1];
  let f = THREE.MathUtils.clamp((t - a.t) / (b.t - a.t), 0, 1);
  if (a.ease) f = f * f * (3 - 2 * f);   // tranziție lină din/în repaus
  const near = f < 0.5 ? a : b;
  return { pose: blendPose(a.pose, b.pose, f), frame: near.frame, stats: near.stats };
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
let sentence = null;   // propoziția redată: { gloss, text, meanings, current }

async function loadWord(word) {
  const motion = await (await fetch(`/motion/${word}.json`)).json();
  const started = performance.now();
  timeline = buildTimeline(motion);
  console.info(`semnul ${word} calculat în ${Math.round(performance.now() - started)} ms`);
  sentence = null;
  ui.subtitle.textContent = motion.meaning;
  ui.status.textContent = `${motion.word}: ${motion.source}, încredere ${motion.confidence}, ${motion.frames.length} cadre`;
  t = 0;
  playing = true;
}

async function loadSentence(gloss, text = null) {
  // Semnele propoziției unul după altul (buildTimeline); cuvintele fără semn sunt sărite
  const motions = [];
  for (const word of gloss) {
    const response = await fetch(`/motion/${word}.json`);
    if (response.ok) motions.push(await response.json());
    else console.warn(`fără semn pentru „${word}”`);
  }
  if (!motions.length) return;
  const started = performance.now();
  timeline = buildTimeline(...motions);
  console.info(`propoziția ${gloss.join(' ')} calculată în ${Math.round(performance.now() - started)} ms`);
  sentence = { gloss: motions.map(m => m.word), meanings: motions.map(m => m.meaning), text, current: null };
  ui.status.textContent = `propoziție: ${sentence.gloss.length} semne, ${timeline.duration.toFixed(1)} s`;
  t = 0;
  playing = true;
}

function showSentence(time) {
  // Subtitrarea: textul în română, iar dedesubt semnele (glose), cu cel curent evidențiat
  const k = timeline.spans.findIndex(([a, b]) => time >= a - BETWEEN[0] / 2 && time <= b + BETWEEN[0] / 2);
  if (k === sentence.current && ui.subtitle.childElementCount) return;
  sentence.current = k;
  const line = document.createElement('div');
  line.className = 'gloss';
  sentence.gloss.forEach((word, i) => {
    const span = document.createElement('span');
    span.textContent = sentence.meanings[i].split(' / ')[0];
    span.title = word;
    if (i === k) span.className = 'current';
    else if (k >= 0 && i < k) span.className = 'done';
    line.append(span);
  });
  ui.subtitle.replaceChildren(...(sentence.text ? [Object.assign(document.createElement('div'), { textContent: sentence.text })] : []), line);
}

// ---------------------------------------------------------------- prototip (?live)
// Pagina pornită de src/app/prototype.py e toată interfața: citește periodic /api/state și
// desenează rândul curent, camera pacientului (MJPEG de la /api/camera), ultimul semn recunoscut,
// conversația și propoziția în curs. La un mesaj nou al medicului, avatarul îi redă semnele.
const $ = id => document.getElementById(id);
const live = { lastId: 0, played: 0, queued: null };
const TURN_TEXT = {
  doctor: 'Medicul vorbește', patient: 'Pacientul semnează',
  idle: 'Oricine poate începe',
};
const MIC_TEXT = { repaus: 'Vorbiți', ascultare: 'Vă ascult…', vorbire: 'Înregistrez…', transcriere: 'Transcriu…',
                   redare: 'Se aude pacientul…' };
const command = name => fetch(`/api/command/${name}`, { method: 'POST' }).catch(() => {});
function el(tag, props = {}, ...children) {
  const element = Object.assign(document.createElement(tag), props);
  element.append(...children);
  return element;
}

const SPEAKER_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
  + 'stroke-linejoin="round" aria-hidden="true"><path d="M11 5 6 9H3v6h3l5 4z"/><path d="M15.5 8.5a5 5 0 0 1 0 7M18.5 5.5a9 9 0 0 1 0 13"/></svg>';

function renderMessage(m) {
  // un rând din fișa consultației: cine, ora, ce a spus, iar dedesubt semnele (în română)
  if (m.who === 'system') return el('div', { className: 'msg system' }, el('div', { className: 'text' }, m.text));
  const doctor = m.who === 'doctor';
  const signs = el('div', { className: 'signs' });
  if (m.meanings.length) signs.append(...m.meanings.map(word => el('span', {}, word)));
  else signs.append(el('span', { className: 'unknown' }, 'niciun semn din vocabular, doar text'));
  if (doctor && m.unknown?.length) signs.append(el('span', { className: 'unknown', title: 'Cuvinte fără semn în vocabular' },
                                                  `fără semn: ${m.unknown.join(', ')}`));
  if (!doctor) {
    const listen = el('button', { type: 'button', className: 'listen', title: 'Citește din nou propoziția, cu voce' });
    listen.innerHTML = SPEAKER_ICON;
    listen.append('Ascultă');
    listen.onclick = () => command(`say-${m.id}`);
    signs.append(listen);
  }
  const via = doctor && m.engine ? `transcris ${m.engine === 'groq' ? 'cu Groq' : 'local'}` : '';
  return el('div', { className: `msg ${m.who}` }, el('span', { className: 'who' }, doctor ? 'Medicul' : 'Pacientul'),
            el('time', { title: via }, m.time), el('div', { className: 'text' }, m.text), signs);
}

async function renderLive(state) {
  document.body.dataset.turn = state.turn;
  document.body.classList.toggle('cam-paused', state.camera.paused);
  $('turn').textContent = state.mic.transcribing ? 'Se transcrie ce a spus medicul'
    : state.voice.speaking ? 'Se aude propoziția pacientului' : TURN_TEXT[state.turn];

  // vocea pacientului (Piper): pornită / oprită din bara de sus; „Ascultă din nou” doar dacă există
  const voice = state.voice;
  document.body.classList.toggle('no-voice', !voice.available);
  $('voiceBtn').disabled = !voice.available;
  $('voiceBtn').setAttribute('aria-pressed', String(voice.enabled));
  $('voiceBtn').textContent = voice.available ? 'Vocea pacientului' : 'Vocea pacientului: indisponibilă';
  $('voiceBtn').title = voice.error ?? `Propoziția pacientului citită cu voce pentru medic (${voice.enabled ? 'pornită' : 'oprită'})`;
  const online = state.speech.engine === 'Groq';
  $('speechPill').textContent = online ? 'Transcriere online' : 'Transcriere locală';
  $('speechPill').title = online ? 'Groq (internet)' : 'Whisper pe acest calculator, fără internet';
  $('speechPill').classList.toggle('warn', !online);
  $('camPill').textContent = `Cameră ${state.camera.fps} cadre/s`;
  $('camPill').classList.toggle('warn', state.camera.fps < 15);
  for (const [id, side, name] of [['handL', 'left', 'stângă'], ['handR', 'right', 'dreaptă']]) {
    const seen = state.camera.hands[side];
    $(id).classList.toggle('off', !seen);
    $(id).textContent = seen ? `Mâna ${name}` : `Mâna ${name} nu se vede`;
  }

  // butonul medicului: starea microfonului și nivelul vocii
  const mic = state.mic.state !== 'repaus' ? state.mic.state
    : state.mic.transcribing ? 'transcriere' : state.voice.speaking ? 'redare' : 'repaus';
  $('speak').dataset.mic = mic;
  $('speakText').textContent = MIC_TEXT[mic];
  $('meter').firstElementChild.style.width = mic === 'vorbire' || mic === 'ascultare'
    ? `${Math.max(0, Math.min(1, (state.mic.level_db + 60) / 60)) * 100}%` : '0';

  // ultimul semn al pacientului: cel mai probabil, cât de sigur, și celelalte două variante
  const r = state.recognition;
  $('recog').hidden = !r;
  if (r) {
    const [best, ...others] = r.top;
    $('recog').classList.toggle('rejected', !r.accepted);
    $('recogWord').textContent = best.meaning;
    $('recogWord').title = best.word;
    $('recogPct').textContent = `${Math.round(best.p * 100)}%`;
    $('recogMeter').firstElementChild.style.width = `${best.p * 100}%`;
    $('recogNote').textContent = r.accepted ? 'Adăugat în propoziție.'
      : `Nesigur (sub ${Math.round(r.threshold * 100)}%), nu s-a adăugat. Repetați semnul.`;
    $('recogTop').replaceChildren(el('li', {}, 'Alte variante:'),
      ...others.map(t => el('li', { title: t.word }, el('b', {}, t.meaning), ` ${Math.round(t.p * 100)}%`)));
  }

  // propoziția pacientului în curs; bara = cât a trecut din pauza după care se trimite
  const pending = state.pending;
  $('pending').classList.toggle('active', pending.words.length > 0);
  $('pendingTitle').textContent = pending.words.length
    ? 'Pacientul semnează. Propoziția se trimite după o scurtă pauză.'
    : 'Propoziția pacientului apare aici pe măsură ce semnează.';
  $('pendingWords').replaceChildren(...pending.meanings.map(word => el('span', {}, word)));
  $('pendingBar').firstElementChild.style.width = `${pending.progress * 100}%`;

  // mesajele noi: în fir; ale medicului, și în semne
  const fresh = state.messages.filter(m => m.id > live.lastId);
  if (!fresh.length) return;
  live.lastId = fresh.at(-1).id;
  $('thread').querySelector('.empty')?.remove();
  $('thread').append(...fresh.map(renderMessage));
  $('thread').scrollTop = $('thread').scrollHeight;
  const doctor = fresh.filter(m => m.who === 'doctor').at(-1);
  if (doctor && doctor.id > live.played) await playDoctor(doctor);
}

async function playDoctor(doctor) {
  // personajul încă se încarcă: mesajul se redă imediat după (main -> startLive)
  if (!retarget) { live.queued = doctor; return; }
  live.played = doctor.id;
  if (doctor.gloss.length) await loadSentence(doctor.gloss, doctor.text);
  else { timeline = null; sentence = null; ui.subtitle.textContent = doctor.text; }   // doar subtitrare
}

async function pollLive() {
  let state = null;
  try { state = await (await fetch('/api/state', { cache: 'no-store' })).json(); }
  catch { $('turn').textContent = 'Fără legătură cu programul. Porniți python -m src.app.prototype.'; }
  if (state) await renderLive(state);
  setTimeout(pollLive, state?.poll_ms ?? 1000);
}

function startLive() {
  // pornește înaintea încărcării personajului: camera și conversația apar imediat
  $('cam').src = '/api/camera';
  // caseta camerei ia proporțiile imaginii (4:3 pe Pi, 16:9 pe laptop): fără benzi negre
  $('cam').addEventListener('load', e => {
    const { naturalWidth: w, naturalHeight: h } = e.target;
    if (w && h) $('camBox').style.setProperty('--cam-ratio', `${w} / ${h}`);
  }, { once: true });
  $('speak').onclick = () => command('speak');
  $('voiceBtn').onclick = () => command('voice');
  $('quit').onclick = () => { command('quit'); $('turn').textContent = 'Programul se oprește…'; };
  addEventListener('keydown', e => {
    if (e.key === ' ' && !e.repeat) { e.preventDefault(); command('speak'); }
  });
  ui.status.textContent = 'prototip';
  pollLive();
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
    drawPoints(s.frame, s.stats ?? timeline?.stats);
    if (sentence) showSentence(t);
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
  if (params.has('live')) startLive();
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
  ui.word.onchange = () => { ui.sentence.value = ''; loadWord(ui.word.value); };
  // propoziții de test (sentences.js); ?sentence=you,hurt,what = orice înșiruire de semne
  // (grupate pe teme)
  const groups = new Map();
  SENTENCES.forEach((s, i) => {
    const group = s.group ?? 'Altele';
    if (!groups.has(group)) groups.set(group, ui.sentence.appendChild(Object.assign(document.createElement('optgroup'), { label: group })));
    groups.get(group).append(new Option(s.text, i));
  });
  ui.sentence.onchange = () => {
    const s = SENTENCES[ui.sentence.value];
    if (s) loadSentence(s.gloss, s.text);
    document.activeElement?.blur();
  };
  ui.play.onclick = () => { t = 0; playing = true; };
  // trecerea la semnul anterior / următor (butoane sau taste), cu redare automată
  const step = delta => {
    const n = ui.word.options.length;
    ui.word.selectedIndex = (ui.word.selectedIndex + delta + n) % n;
    document.activeElement?.blur();   // spațiul să nu mai „apese” butonul rămas selectat
    ui.sentence.value = '';
    loadWord(ui.word.value);
  };
  document.getElementById('prev').onclick = () => step(-1);
  document.getElementById('next').onclick = () => step(1);
  addEventListener('keydown', e => {
    if (params.has('live')) return;              // în prototip, Space = vorbește medicul (startLive)
    if (e.target.tagName === 'SELECT') return;   // în listă, săgețile o folosesc pe ea
    if (e.key === 'ArrowRight' || e.key === 'n' || e.key === 'N') step(1);
    else if (e.key === 'ArrowLeft' || e.key === 'p' || e.key === 'P') step(-1);
    else if (e.key === ' ') { e.preventDefault(); ui.play.onclick(); }
  });
  if (params.has('points')) ui.points.checked = true;
  ui.points.onchange = frameCamera;
  frameCamera();
  if (params.has('sentence')) {
    // textul în română, dacă propoziția e una din sentences.js
    const gloss = params.get('sentence').split(','), i = SENTENCES.findIndex(s => s.gloss.join() === gloss.join());
    if (i >= 0) ui.sentence.value = i;
    await loadSentence(gloss, SENTENCES[i]?.text);
  }
  else if (!params.has('live')) await loadWord(ui.word.value);
  if (live.queued) playDoctor(live.queued);   // prototipul: mesajul medicului sosit în timpul încărcării
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
    // ?bodycheck=a,b (gol = toate; propoziții „a-b-c”): pătrunderea maximă a mâinilor și a brațelor
    // în corp (cm) și cel mai mare salt al cotului / încheieturii de la un cadru la altul (cm), pe
    // redarea la 30 de cadre/s, cu tot cu tranzițiile; ?trace scrie și pozițiile la fiecare cadru
    const words = params.get('bodycheck') ? params.get('bodycheck').split(',') : Object.keys(index);
    for (const word of words) {
      const tl = await timelineFor(word);
      let worst = 0, at = 0, arm = { worst: 0, where: '-' }, armAt = 0, jump = 0, jumpAt = 0, prev = null, armSign = 0;
      for (let time = 0; time <= tl.duration; time += 1 / 30) {
        const key = { ...sample(tl, time), t: time };
        const cm = retarget.bodyReport(key.pose.D);
        if (cm > worst) { worst = cm; at = key.t; }
        const a = retarget.armReport(key.pose.D);
        if (a.worst > arm.worst) { arm = a; armAt = key.t; }
        // brațele doar în timpul semnelor (fără tranziții)
        if (tl.spans.some(([s0, s1]) => time >= s0 && time <= s1) && a.worst > (armSign.worst ?? 0)) armSign = { ...a, t: time };
        const pos = retarget.fk(key.pose.D).pos;
        if (prev) for (const b of ['rightLowerArm', 'rightHand', 'leftLowerArm', 'leftHand']) {
          const d = pos[b].distanceTo(prev.pos[b]) * 100;
          if (d > jump) { jump = d; jumpAt = key.t; }
        }
        prev = { pos };
        if (params.has('trace')) {
          const f = v => v.toArray().map(x => x.toFixed(3)).join(',');
          console.log(`TRACE ${word} ${key.t.toFixed(3)} rE ${f(pos.rightLowerArm)} rW ${f(pos.rightHand)} lE ${f(pos.leftLowerArm)} lW ${f(pos.leftHand)}`);
        }
      }
      console.log(`BODY ${word} ${worst.toFixed(1)} ${at.toFixed(2)} ARM ${arm.worst.toFixed(1)} ${arm.where || '-'} ${armAt.toFixed(2)} SIGN ${(armSign.worst ?? 0).toFixed(1)} ${armSign.where || '-'} ${(armSign.t ?? 0).toFixed(2)} JUMP ${jump.toFixed(1)} ${jumpAt.toFixed(2)}`);
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
      const tl = await timelineFor(word);
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
