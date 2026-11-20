// Corecții per semn, aplicate peste mișcarea din date (retarget.js). Datele dau traiectoria
// generală a semnului; aici se fixează ce MediaPipe nu poate da sigur: forma exactă a mâinii,
// mișcările repetitive (cerc, frecare, bătăi, tremur), atingerile și locul exact pe corp.
//
// Forme ale mâinii (grade): pentru fiecare deget [flexie la bază, la mijloc, la vârf, depărtare
// laterală: + spre degetul mic, - spre degetul mare]; degetul mare: [flexie la bază (spre palmă),
// apropiere de degete (+ lipit de arătător, - depărtat), flexie la mijloc, la vârf].
// O formă poate defini doar unele degete; restul rămân din date.
const EXT = [0, 0, 0, 0];          // deget întins
const CURL = [88, 100, 65, 0];     // deget strâns în pumn
const HOOK = [10, 95, 70, 0];      // cârlig (X): baza dreaptă, mijlocul și vârful îndoite
const BENT = [85, 0, 0, 0];        // îndoit doar din bază (palmă „îndoită”)
const CLAW = [35, 70, 45, 0];      // gheară
const fist = { Index: CURL, Middle: CURL, Ring: CURL, Little: CURL };

export const THUMB = {
  out: [-10, -35, 0, 0],     // depărtat de palmă (L, Y)
  side: [-5, 25, 5, 0],      // lipit de arătător, în planul palmei (A, B)
  across: [30, 50, 60, 50],  // peste degetele strânse (S)
  tucked: [60, 25, 50, 40],  // strâns, ține degetele inelar / mic (V, W)
  up: [0, 35, 0, 0],         // întins în sus, pe lângă pumn (A cu degetul mare ridicat)
};

export const HANDSHAPES = {
  open: { Thumb: THUMB.out, Index: [0, 0, 0, -8], Middle: EXT, Ring: [0, 0, 0, 8], Little: [0, 0, 0, 15] },
  spread: { Thumb: THUMB.out, Index: [0, 0, 0, -12], Middle: EXT, Ring: [0, 0, 0, 12], Little: [0, 0, 0, 15] },
  B: { Thumb: THUMB.side, Index: EXT, Middle: EXT, Ring: EXT, Little: EXT },
  bentB: { Thumb: THUMB.side, Index: BENT, Middle: BENT, Ring: BENT, Little: BENT },
  S: { Thumb: THUMB.across, ...fist },
  A: { Thumb: THUMB.side, ...fist },
  thumbUp: { Thumb: THUMB.up, ...fist },
  one: { Thumb: THUMB.across, Index: EXT, Middle: CURL, Ring: CURL, Little: CURL },
  X: { Thumb: THUMB.across, Index: HOOK, Middle: CURL, Ring: CURL, Little: CURL },
  I: { Thumb: THUMB.across, Index: CURL, Middle: CURL, Ring: CURL, Little: EXT },
  Y: { Thumb: THUMB.out, Index: CURL, Middle: CURL, Ring: CURL, Little: [0, 0, 0, 15] },
  V: { Thumb: THUMB.tucked, Index: [0, 0, 0, -10], Middle: [0, 0, 0, 10], Ring: CURL, Little: CURL },
  W: { Thumb: THUMB.tucked, Index: [0, 0, 0, -10], Middle: EXT, Ring: [0, 0, 0, 10], Little: CURL },
  claw: { Thumb: [20, 30, 30, 30], Index: CLAW, Middle: CLAW, Ring: CLAW, Little: CLAW },
  eight: { Thumb: THUMB.out, Index: [0, 0, 0, -8], Middle: [80, 15, 5, 0], Ring: [0, 0, 0, 8], Little: [0, 0, 0, 15] },
  flatO: { Thumb: [35, 35, 20, 20], Index: [55, 25, 15, 0], Middle: [55, 25, 15, 0], Ring: [55, 25, 15, 0], Little: [55, 25, 15, 0] },
  C: { Thumb: [20, -10, 15, 10], Index: [30, 40, 20, 0], Middle: [30, 40, 20, 0], Ring: [30, 40, 20, 0], Little: [30, 40, 20, 0] },
  pinch: { Thumb: [25, 30, 25, 20], Index: [45, 35, 20, 0], Middle: CURL, Ring: CURL, Little: CURL },
  // pumn cu degetul mare ridicat perpendicular pe pumn (👍, „ok”)
  thumbsUp: { Thumb: [-5, -30, 0, 0], ...fist },
  // gheara mai deschisă (hot)
  clawOpen: { Thumb: [10, 10, 20, 20], Index: [25, 45, 25, 0], Middle: [25, 45, 25, 0], Ring: [25, 45, 25, 0], Little: [25, 45, 25, 0] },
  // palma întinsă, dar relaxată (arm)
  Brelaxed: { Thumb: [-5, 20, 5, 5], Index: [8, 12, 6, 0], Middle: [8, 12, 6, 0], Ring: [10, 14, 6, 0], Little: [12, 15, 6, 0] },
  // D: arătătorul întins; mijlociul, inelarul și degetul mic îndoite (~80°), vârfurile pe vârful degetului mare
  D: { Thumb: [35, 45, 30, 25], Index: EXT, Middle: [55, 75, 45, 0], Ring: [55, 75, 45, 0], Little: [55, 75, 45, 0] },
  // pumn complet închis, doar degetul mare întins (back)
  thumbFist: { Thumb: [0, 35, 0, 0], Index: [90, 100, 75, 0], Middle: [90, 100, 75, 0], Ring: [90, 100, 75, 0], Little: [90, 100, 75, 0] },
  relaxed: { Thumb: [0, -15, 10, 10], Index: [20, 20, 10, 0], Middle: [25, 25, 10, 0], Ring: [30, 25, 10, 0], Little: [35, 25, 10, 0] },
  clawTight: { Thumb: [30, 30, 40, 35], Index: [50, 85, 60, 0], Middle: [50, 85, 60, 0], Ring: [50, 85, 60, 0], Little: [50, 85, 60, 0] },
};
HANDSHAPES.flatO = { Thumb: [40, 45, 25, 20], Index: [60, 35, 25, 0], Middle: [60, 35, 25, 0], Ring: [60, 35, 25, 0], Little: [60, 35, 25, 0] };

// Corecțiile per semn (mâna dreaptă e cea dominantă). Pentru fiecare mână (`right` / `left`):
//   shape   forma mâinii: nume din HANDSHAPES sau chei [[t, nume], ...] (t = 0..1 din durata semnului)
//   path    traiectoria unui punct al mâinii (`via`: palm, back, wrist, knuckles, heel, fingers,
//           thumb, index, middle, little): chei [t, reper, decalaj [x, y, z] în m], unde reperul
//           e pe corpul avatarului (retarget.measureAnchors) sau pe cealaltă mână ("other.palm"...)
//   arc     antebrațul se rotește în jurul cotului aflat la reperul `pivot`, de la direcția `from` la `to`
//   offset  decalaj constant al mâinii [x, y, z] în m
//   motion  mișcări adăugate: circle (radius, turns, plane), tap / rub (dir, amp, count),
//           shake (dir, amp, freq), flap (îndoirea încheieturii: amp în grade, count); `t` = [t0, t1]
//   orient  orientarea impusă a palmei: dir (încheietură -> degete; 'arc' = în prelungirea antebrațului
//           din `arc`), side (degetul mic -> arătător); normala palmei (mâna dreaptă) = side × dir
//   pole    direcția fixă a cotului [x, y, z] (fără salturi ale cotului; ex. mâna la bărbie: jos și în față)
//   free    fără limitarea întinderii brațului în față (semne cu brațul întins, ex. take)
//   contact mâna atinge corpul (fără distanța minimă față de piept)
//   hold    mâna stă pe loc, în poza din date de la momentul t
//   pivot   mișcarea doar din cot (cotul rămâne unde e la mijlocul semnului)
//   rest / active   mâna nu participă / participă tot semnul
// La nivelul semnului: touch [t0, t1] = mâna dreaptă atinge mâna stângă în intervalul dat.
// Coordonatele sunt în reperul pieptului avatarului: X = stânga avatarului, Y = sus, Z = în față.
// cotul jos și ușor în față (mâna la bărbie / gură): nu lateral și sus
const ELBOW_FRONT = [-0.35, -1, 0.5];
// cotul jos, mai în lateral: antebrațul nu trece prin piept (better)
const ELBOW_SIDE = [-0.8, -1, 0.45];
const circleOnChest = { type: 'circle', radius: 0.035, turns: 2, t: [0.1, 0.95] };
export const SIGNS = {
  // pumn strâns, brațul pe loc; încheietura se îndoaie în jos de două ori
  yes: { right: { shape: 'S', hold: 0.5, motion: [{ type: 'flap', amp: 30, count: 2, t: [0.15, 0.95] }] }, left: { rest: true } },
  please: { right: { shape: 'B', contact: true, via: 'palm', path: [[0, 'chest_r', [0.02, 0, 0]]], motion: [circleOnChest] }, left: { rest: true } },
  // 👍 cu palma spre piept (văzut din profil: degetele spre stânga, degetul mare în sus), cercuri pe piept
  sorry: { right: { shape: 'thumbsUp', contact: true, via: 'indexMid', path: [[0, 'chest_r', [0.02, 0, 0]]],
    orient: { dir: [1, 0, 0.15], side: [0, 1, 0] }, motion: [circleOnChest] }, left: { rest: true } },
  understand: { right: { shape: [[0, 'S'], [0.55, 'S'], [0.75, 'one']], via: 'knuckles',
    path: [[0, 'temple_r', [-0.03, 0, 0.05]]] }, left: { rest: true } },
  // 👍 cu mâna dreaptă (palma spre stânga), bate de două ori în palma stângă, care stă fixă, cu palma în sus
  help: { left: { shape: 'B', active: true, via: 'palm', path: [[0, 'chest', [0, -0.13, 0.11]]], orient: { dir: [-0.3, 0, 1], side: [1, 0, 0] } },
    right: { shape: 'thumbsUp', via: 'littleMid', path: [[0, 'other.palm', [0, 0.012, 0]]], orient: { dir: [0.2, 0, 1], side: [0, 1, 0] },
      motion: [{ type: 'tap', dir: [0, 1, 0], amp: 0.04, count: 2, t: [0.15, 0.95] }] } },
  need: { right: { shape: 'X', motion: [{ type: 'flap', amp: 25, count: 2, t: [0.3, 1] }] }, left: { rest: true } },
  i: { right: { shape: 'I', via: 'wrist', path: [[0, 'chest_r', [-0.01, 0.03, 0.12]]] }, left: { rest: true } },
  // arată înainte cu arătătorul, fără altă mișcare
  you: { right: { shape: 'one', hold: 0.6, orient: { dir: [-0.1, -0.05, 1], side: [0.6, 0.8, 0] } }, left: { rest: true } },
  // palma vine de două ori spre piept
  my: { right: { shape: 'B', contact: true, via: 'palm',
    path: [[0, 'chest', [0, 0, 0.14]], [0.3, 'chest', [0, 0, 0]], [0.5, 'chest', [0, 0, 0.1]], [0.75, 'chest', [0, 0, 0]], [1, 'chest', [0, 0, 0]]] }, left: { rest: true } },
  child: { right: { shape: 'B', via: 'palm', path: [[0, 'waist_r', [0, 0, 0]]], orient: { dir: [-0.2, 0, 1], side: [1, 0, 0.2] },
    motion: [{ type: 'tap', dir: [0, -1, 0], amp: 0.04, count: 2, t: [0.2, 1] }] }, left: { rest: true } },
  sick: { right: { shape: 'eight', contact: true, via: 'middle', path: [[0, 'forehead', [0, 0, 0]]] },
    left: { shape: 'eight', contact: true, via: 'middle', path: [[0, 'stomach', [0, 0, 0]]], active: true } },
  headache: {
    right: { shape: 'one', via: 'index', path: [[0, 'forehead', [-0.05, 0.01, 0.06]]], orient: { dir: [1, 0, 0], side: [0, 1, 0] },
      motion: [{ type: 'rub', dir: [1, 0, 0], amp: 0.012, count: 2, t: [0.15, 0.95] }] },
    left: { shape: 'one', via: 'index', path: [[0, 'forehead', [0.05, 0.01, 0.06]]], orient: { dir: [-1, 0, 0], side: [0, 1, 0] },
      motion: [{ type: 'rub', dir: [-1, 0, 0], amp: 0.012, count: 2, t: [0.15, 0.95] }], active: true } },
  // mâna ușor deschisă, palma pe gât, freacă în sus și în jos
  sorethroat: { right: { shape: 'relaxed', contact: true, via: 'palm', path: [[0, 'neck', [0, -0.01, 0]]],
    orient: { dir: [1, 0.35, 0], side: [-0.35, 1, 0] },
    motion: [{ type: 'rub', dir: [0, 1, 0], amp: 0.015, count: 3, t: [0.1, 0.95] }] }, left: { rest: true } },
  // gheara bate în piept, cu o mică rotire din încheietură
  cough: { right: { shape: 'claw', contact: true, via: 'middle', path: [[0, 'chest', [0, 0, 0]]],
    motion: [{ type: 'flap', amp: 18, count: 3, t: [0.1, 0.95] }, { type: 'twist', amp: 20, count: 3, t: [0.1, 0.95] }] }, left: { rest: true } },
  cold: {
    right: { shape: 'S', via: 'knuckles', path: [[0, 'shoulder_r', [0, -0.09, 0.13]]],
      motion: [{ type: 'shake', dir: [1, 0, 0], amp: 0.01, freq: 7, t: [0.05, 1] }] },
    left: { shape: 'S', via: 'knuckles', path: [[0, 'shoulder_l', [0, -0.09, 0.13]]], active: true,
      motion: [{ type: 'shake', dir: [1, 0, 0], amp: 0.01, freq: 7, t: [0.05, 1] }] } },
  hot: { right: { shape: 'clawOpen' }, left: { rest: true } },
  vomit: { right: { offset: [0, 0.02, 0] }, left: { offset: [0, -0.06, 0.05] } },
  sleep: { right: { shape: [[0, 'spread'], [0.55, 'spread'], [0.9, 'flatO']] }, left: { rest: true } },
  head: { right: { shape: 'bentB', contact: true, via: 'index',
    path: [[0, 'jaw_r', [0, 0, 0]], [0.3, 'jaw_r', [0, 0, 0]], [0.75, 'ear_r', [0, 0.03, 0.02]], [1, 'ear_r', [0, 0.03, 0.02]]] }, left: { rest: true } },
  stomach: { right: { shape: 'B', contact: true, via: 'palm', path: [[0, 'stomach', [0, 0, 0]]], orient: { dir: [1, 0, 0], side: [0, 1, 0] },
    motion: [{ type: 'tap', dir: [0, 0, 1], amp: 0.04, count: 2, t: [0.15, 0.95] }] }, left: { rest: true } },
  // pumnul în fața umărului drept, puțin deasupra lui, cotul jos și în față (nu ridicat: cotul sus sărea
  // între soluții la fiecare bătaie și ieșea pielea prin mânecă); pumnul vertical (încheieturile degetelor
  // în sus), palma spre cap; degetul mare iese deasupra pumnului (perpendicular pe el) și arată spre spate
  back: { right: { shape: 'thumbsUp', via: 'knuckles', path: [[0, 'shoulder_r', [-0.02, 0.1, 0.1]]], pole: [-0.4, -0.7, 0.8],
    orient: { dir: [0, 1, 0.15], side: [0, 0, -1] }, motion: [{ type: 'tap', dir: [0, 0, -1], amp: 0.04, count: 2, t: [0.2, 1] }] }, left: { rest: true } },
  // antebrațul stâng orizontal în fața corpului, palma în jos; palma dreaptă alunecă pe el
  // (deasupra mânecii) de la cot spre încheietură
  arm: {
    left: { shape: 'B', active: true, via: 'wrist', path: [[0, 'chest', [-0.08, -0.12, 0.2]]], orient: { dir: [-1, 0, 0.15], side: [0, 0, -1] } },
    right: { shape: 'Brelaxed', via: 'palm', orient: { dir: [0.35, -0.1, 1], side: [1, 0, -0.3] },
      path: [[0, 'other.elbow', [0, 0.065, 0.01]], [0.2, 'other.elbow', [0, 0.065, 0.01]], [0.9, 'other.wrist', [0, 0.05, 0.01]], [1, 'other.wrist', [0, 0.05, 0.01]]] } },
  heart: {
    right: { shape: 'eight', contact: true, via: 'middle',
      path: [[0, 'chest_l', [0, 0.02, 0]], [0.25, 'chest_l', [-0.03, 0.05, 0]], [0.45, 'chest_l', [-0.05, 0.02, 0]], [0.8, 'chest_l', [0, -0.05, 0]], [1, 'chest_l', [0, -0.05, 0]]] },
    left: { shape: 'eight', contact: true, via: 'middle', active: true,
      path: [[0, 'chest_l', [0.005, 0.02, 0]], [0.25, 'chest_l', [0.035, 0.05, 0]], [0.45, 'chest_l', [0.055, 0.02, 0]], [0.8, 'chest_l', [0.005, -0.05, 0]], [1, 'chest_l', [0.005, -0.05, 0]]] } },
  now: { right: { shape: 'Y' }, left: { shape: 'Y' } },
  // pumn vertical lângă obraz, degetul mare spre spate: vârful lui urcă pe linia bărbiei, de la colțul gurii spre ureche; cotul jos, în față
  yesterday: { right: { shape: 'thumbsUp', via: 'thumb', orient: { dir: [-0.15, 1, 0.2], side: [0, 0, -1] }, pole: [-0.15, -0.5, 1],
    path: [[0, 'mouth_r', [0, -0.015, 0.005]], [0.3, 'mouth_r', [0, -0.015, 0.005]], [0.85, 'ear_r', [0, -0.015, 0.035]], [1, 'ear_r', [0, -0.015, 0.035]]] }, left: { rest: true } },
  // MORNING (ridicare, 2.9 s). Cotul drept fix (balama) la abdomenul superior, în fața corpului.
  // Stânga: antebraț orizontal, palma plată spre trunchi, vârfurile degetelor ating pliul cotului drept
  // din partea dinspre corp (antebrațul drept nu trece prin ele). Dreapta: B plat. Antebrațul pornește
  // jos, înainte (~15° peste orizontală), palma în sus; se ridică din cot ~90° până la vertical (ușor
  // spre exterior), palma ajunge spre față / piept (rotire naturală din cot, fără răsucire); apoi stă
  // nemișcat ~0.4 s (arcul se termină la t = 0.86).
  morning: {
    left: { shape: 'B', active: true, via: 'index', path: [[0, 'other.elbow', [0.045, 0, 0]]], orient: { dir: [-1, 0, 0.05], side: [0, 1, 0] } },
    right: { shape: 'B', arc: { upper: [-0.12, -0.75, 0.7], from: [0.05, 0.27, 1], to: [-0.13, 1, 0], t: [0.15, 0.86] },
      orient: { dir: 'arc', side: [-1, 0, 0] } } },
  night: { left: { hold: 0.5 }, right: { shape: 'bentB' }, touch: [0.5, 1] },
  // DAY (coborâre, 2.2 s). Cotul drept fix (balama) la nivelul plexului, în fața pieptului, sprijinit
  // deasupra degetelor stângi (~1.5 cm; pe mijlocul palmei, cotul stâng ajungea în piept).
  // Stânga: antebraț orizontal, palma plată în jos. Dreapta: D.
  // Antebrațul drept stă vertical (arătătorul în sus), apoi coboară din cot ~90° spre stânga avatarului
  // până la orizontal, paralel și imediat deasupra antebrațului stâng (degetele îndoite în jos); apoi stă
  // nemișcat ~0.4 s (arcul se termină la t = 0.82).
  day: {
    // cotul stâng în afară și în față: antebrațul stâng stă în fața tricoului (altfel cotul intra în piept)
    left: { shape: 'B', active: true, via: 'fingers', path: [[0, 'other.elbow', [0, -0.055, 0]]], orient: { dir: [-1, 0, 0.1], side: [0, 0, -1] },
      pole: [0.7, -0.5, 0.6] },
    right: { shape: 'D', arc: { upper: [-0.15, -0.6, 0.8], from: [-0.03, 1, 0.08], to: [1, 0, 0.1], t: [0.15, 0.82] },
      orient: { dir: 'arc', side: [0, 0, -1] } } },
  week: { left: { shape: 'B', active: true, via: 'palm', path: [[0, 'chest', [0.02, -0.17, 0.11]]], orient: { dir: [-0.3, 0, 1], side: [1, 0, 0] } },
    // degetele strânse ale mâinii drepte (palma în jos) alunecă pe palma stângă, de la podul palmei spre
    // degete (punctul de contact = vârful mijlociului îndoit, cel mai de jos: nu trece prin palma stângă)
    right: { shape: 'one', via: 'middle', orient: { dir: [0.4, 0, 1], side: [1, 0, 0] },
      path: [[0, 'other.heel', [0, 0.006, 0]], [0.15, 'other.heel', [0, 0.006, 0]], [0.8, 'other.fingers', [0, 0.006, 0]], [1, 'other.fingers', [0, 0.006, 0]]] } },
  month: { left: { hold: 0.5, shape: 'one' },
    right: { shape: 'one', via: 'index',
      path: [[0, 'other.index', [0, 0.04, 0.02]], [0.2, 'other.index', [0, 0.01, 0.02]], [0.85, 'other.knuckles', [0, -0.03, 0.02]], [1, 'other.knuckles', [0, -0.03, 0.02]]] } },
  ago: { right: { pivot: true }, left: { rest: true } },
  since: { right: { shape: 'one', offset: [0, -0.07, 0] }, left: { shape: 'one', offset: [0, -0.07, 0] } },
  two: { right: { shape: 'V' }, left: { rest: true } },
  // palma la bărbie, apoi spre dreapta și înainte, terminând cu degetul mare în sus, văzut din profil;
  // cotul jos și în lateral (ELBOW_SIDE: în față atingea pieptul și sărea)
  better: { right: { shape: [[0, 'B'], [0.4, 'B'], [0.8, 'thumbsUp']], contact: true, via: 'palm', pole: ELBOW_SIDE,
    orient: { dir: [0.2, 0, 1], side: [0, 1, 0], at: 0.8 },
    path: [[0, 'chin', [0, -0.01, 0]], [0.35, 'chin', [0, -0.01, 0]], [1, 'chin', [-0.14, 0.03, 0.17]]] }, left: { rest: true } },
  // palma stângă în sus, în fața corpului; mijlociul drept face cercuri mici în ea
  medicine: { left: { shape: 'B', active: true, via: 'palm', path: [[0, 'chest', [0.02, -0.14, 0.11]]], orient: { dir: [-0.3, 0, 1], side: [1, 0, 0] } },
    right: { shape: 'eight', via: 'middle', path: [[0, 'other.palm', [0, 0.005, 0]]],
      motion: [{ type: 'circle', plane: 'xz', radius: 0.012, turns: 2, t: [0.1, 0.9] }] } },
  take: { right: { shape: [[0, 'spread'], [0.45, 'spread'], [0.85, 'S']], free: true }, left: { rest: true } },
  // vârfurile adunate ating bărbia de două ori; încheietura nu se mișcă (orientare fixă)
  eat: { right: { shape: 'flatO', contact: true, via: 'index', path: [[0, 'chin', [0, 0, 0]]], pole: ELBOW_FRONT,
    orient: { dir: [0, 0.4, -1], side: [-1, 0, 0] },
    motion: [{ type: 'tap', dir: [0, 0, 1], amp: 0.05, count: 2, t: [0.15, 0.95] }] }, left: { rest: true } },
  // mâna în C se înclină spre gură, iar degetul mare rămâne pe loc, sub buza de jos
  drink: { right: { shape: 'C', via: 'thumb', path: [[0, 'chin', [0, 0.005, 0.012]]], pole: ELBOW_FRONT }, left: { rest: true } },
  // W cu degetele în sus, palma spre stânga; vârful arătătorului atinge bărbia de două ori;
  // cotul mult în lateral: antebrațul trece în diagonală prin fața pieptului, nu prin el
  water: { right: { shape: 'W', contact: true, via: 'index', path: [[0, 'chin', [-0.008, 0, 0]]], pole: [-1.5, -0.8, 0.3],
    orient: { dir: [0, 1, 0.1], side: [0, 0, -1] },
    motion: [{ type: 'tap', dir: [0, -0.4, 1], amp: 0.02, count: 2, t: [0.2, 1] }] }, left: { rest: true } },
};
