// Tricoul personajului de probă, făcut mai strâmt (la încărcare, în geometria lui, înainte de
// măsurarea corpului din retarget.js). Mânecile erau largi și legate de oase auxiliare (constrângeri
// „Aim” și oase-arc „Sec”), care urmăresc brațul doar parțial: brațul și mâna treceau prin mânecă.
//   1. vârfurile mânecilor se leagă direct de osul brațului (îl urmează exact);
//   2. mânecile se strâng spre braț: distanța față de axa brațului coboară aproape de piele;
//   3. trunchiul tricoului se strânge spre piele (mai puțin), pe o grilă înălțime x unghi.
// Totul în poza de legare (T-pose), cu pozițiile și oasele din aceeași poză.
import * as THREE from 'three';

const V3 = THREE.Vector3;
const SLEEVE = { gap: 0.008, keep: 0.3, bins: 8, from: 0.2, blend: 0.35 };   // m peste piele; cât din surplus se păstrează; tronsoane pe braț
const TORSO = { gap: 0.015, keep: 0.5, heightBin: 0.015, angleBins: 18 };

const materialNames = mesh => [mesh.material].flat().map(m => m?.name ?? '').join(' ');

function bindData(mesh) {
  // pozițiile vârfurilor și ale oaselor în poza de legare (același spațiu), osul dominant pe vârf
  const g = mesh.geometry, pos = g.attributes.position, si = g.attributes.skinIndex, sw = g.attributes.skinWeight;
  const bones = mesh.skeleton.bones.map((b, i) => new V3().setFromMatrixPosition(mesh.skeleton.boneInverses[i].clone().invert()));
  const points = [], main = [];
  for (let i = 0; i < pos.count; i++) {
    points.push(new V3().fromBufferAttribute(pos, i).applyMatrix4(mesh.bindMatrix));
    let best = 0, w = -1;
    for (let k = 0; k < 4; k++) if (sw.getComponent(i, k) > w) { w = sw.getComponent(i, k); best = si.getComponent(i, k); }
    main.push(best);
  }
  return { bones, points, main };
}

function boneIndex(mesh, vrm, humanoidName) {
  return mesh.skeleton.bones.indexOf(vrm.humanoid.getRawBoneNode(humanoidName));
}

function writeBack(mesh, points) {
  const pos = mesh.geometry.attributes.position, inv = mesh.bindMatrixInverse;
  points.forEach((p, i) => { const q = p.clone().applyMatrix4(inv); pos.setXYZ(i, q.x, q.y, q.z); });
  pos.needsUpdate = true;
  mesh.geometry.computeVertexNormals();
  mesh.geometry.computeBoundingSphere();
}

export function tightenShirt(vrm) {
  let shirt = null, skin = null;
  vrm.scene.traverse(o => {
    if (!o.isSkinnedMesh) return;
    if (/Tops/i.test(materialNames(o))) shirt = o;
    else if (/Body_00_SKIN/i.test(materialNames(o))) skin = o;
  });
  if (!shirt || !skin) return false;

  const shirtData = bindData(shirt), skinData = bindData(skin);
  const report = { sleeve: 0, torso: 0 };

  // 1. mânecile: oasele auxiliare ale mânecii -> osul brațului, dar doar de la SLEEVE.from din lungimea
  //    brațului încolo: lângă umăr osul auxiliar (care urmează brațul pe jumătate) acoperă umărul
  //    când brațul se mișcă în față; legată direct de braț, mâneca lăsa pielea umărului la vedere
  const si = shirt.geometry.attributes.skinIndex, sw = shirt.geometry.attributes.skinWeight;
  for (const [side, prefix] of [['left', '_L_'], ['right', '_R_']]) {
    const arm = boneIndex(shirt, vrm, side + 'UpperArm'), elbow = boneIndex(shirt, vrm, side + 'LowerArm');
    if (arm < 0 || elbow < 0) continue;
    const a = shirtData.bones[arm], axis = shirtData.bones[elbow].clone().sub(a), len = axis.length();
    axis.normalize();
    const helper = new Set(shirt.skeleton.bones.map((b, i) => (b.name.includes(prefix) && /Tops.*UpperArm/i.test(b.name)) ? i : -1).filter(i => i >= 0));
    for (let i = 0; i < si.count; i++) {
      // trecere treptată pe SLEEVE.blend din lungimea brațului (fără margine bruscă, unde mâneca
      // se întindea și își arăta interiorul când cotul stă mult în față)
      const f = THREE.MathUtils.clamp((shirtData.points[i].clone().sub(a).dot(axis) / len - SLEEVE.from) / SLEEVE.blend, 0, 1);
      if (f <= 0) continue;
      for (let k = 0; k < 4; k++) {
        if (!helper.has(si.getComponent(i, k))) continue;
        const w = sw.getComponent(i, k), free = [0, 1, 2, 3].find(j => sw.getComponent(i, j) === 0);
        if (f >= 1 || free === undefined) { si.setComponent(i, k, arm); continue; }
        // o parte din influența osului auxiliar trece la braț, într-un loc liber
        sw.setComponent(i, k, w * (1 - f));
        si.setComponent(i, free, arm);
        sw.setComponent(i, free, w * f);
      }
      if (f >= 0.5 && helper.has(shirtData.main[i])) shirtData.main[i] = arm;
    }
  }
  si.needsUpdate = true;
  sw.needsUpdate = true;

  // 2. mânecile se strâng spre axa brațului (umăr -> cot), la raza pielii brațului
  for (const side of ['left', 'right']) {
    const armShirt = boneIndex(shirt, vrm, side + 'UpperArm'), elbowShirt = boneIndex(shirt, vrm, side + 'LowerArm');
    const armSkinBone = vrm.humanoid.getRawBoneNode(side + 'UpperArm');
    if (armShirt < 0 || elbowShirt < 0) continue;
    const a = shirtData.bones[armShirt], b = shirtData.bones[elbowShirt], axis = b.clone().sub(a), len = axis.length();
    axis.normalize();
    const radial = p => { const t = THREE.MathUtils.clamp(p.clone().sub(a).dot(axis), 0, len); return p.clone().sub(a.clone().addScaledVector(axis, t)); };
    // raza pielii pe tronsoane de-a lungul brațului (percentila 95): la umăr (deltoid) e mai mare
    const tag = side === 'left' ? '_L_' : '_R_';
    const binOf = p => THREE.MathUtils.clamp(Math.floor(p.clone().sub(a).dot(axis) / len * SLEEVE.bins), 0, SLEEVE.bins - 1);
    const perBin = Array.from({ length: SLEEVE.bins }, () => []);
    skinData.points.forEach((p, i) => {
      const bone = skin.skeleton.bones[skinData.main[i]];
      if (bone === armSkinBone || (bone.name.includes(tag) && /UpperArm|Shoulder/.test(bone.name))) perBin[binOf(p)].push(radial(p).length());
    });
    const skinR = perBin.map(list => { list.sort((x, y) => x - y); return list.length ? list[Math.floor(list.length * 0.95)] : null; });
    if (skinR.every(r => r === null)) continue;
    shirtData.points.forEach((p, i) => {
      if (shirtData.main[i] !== armShirt) return;
      // primul tronson (lângă umăr) rămâne: acolo pielea se deformează odată cu umărul
      const bin = binOf(p), rs = skinR[bin];
      if (bin === 0 || rs === null) return;
      const R = rs + SLEEVE.gap, r = radial(p), d = r.length();
      if (d <= R) return;
      p.addScaledVector(r.normalize(), (R + (d - R) * SLEEVE.keep) - d);
      report.sleeve++;
    });
  }

  // 3. trunchiul: raza pielii pe celule (înălțime x unghi) în jurul axei verticale a șoldurilor
  const hips = vrm.humanoid.getRawBoneNode('hips'), hipIdx = skin.skeleton.bones.indexOf(hips);
  const center = skinData.bones[hipIdx] ?? new V3();
  const torsoBones = new Set(['hips', 'spine', 'chest', 'upperChest'].map(n => vrm.humanoid.getRawBoneNode(n)).filter(Boolean));
  const isTorso = (mesh, data, i) => {
    const bone = mesh.skeleton.bones[data.main[i]];
    return torsoBones.has(bone) || /Bust/i.test(bone.name);
  };
  const cellOf = p => {
    const angle = Math.atan2(p.x - center.x, p.z - center.z);
    return [Math.floor(p.y / TORSO.heightBin), Math.floor(((angle + Math.PI) / (2 * Math.PI)) * TORSO.angleBins) % TORSO.angleBins];
  };
  const key = (h, a) => `${h}:${(a + TORSO.angleBins) % TORSO.angleBins}`;
  const flat = p => Math.hypot(p.x - center.x, p.z - center.z);
  // toată pielea din zonă (și umerii, gâtul), ca tricoul să nu intre sub nicio parte a ei
  const skinMax = new Map();
  skinData.points.forEach(p => {
    const k = key(...cellOf(p));
    skinMax.set(k, Math.max(skinMax.get(k) ?? 0, flat(p)));
  });
  // raza pielii sub un punct al tricoului: maximul din celula lui și din cele vecine (fără „găuri”)
  const skinUnder = p => {
    const [h, a] = cellOf(p);
    let best = 0;
    for (let dh = -1; dh <= 1; dh++) for (let da = -1; da <= 1; da++) best = Math.max(best, skinMax.get(key(h + dh, a + da)) ?? 0);
    return best;
  };
  // doar sub piept: la umeri și axile pielea se deformează când brațul se mișcă în față și ar ieși
  // prin tricoul strâns
  const chestBone = shirt.skeleton.bones.indexOf(vrm.humanoid.getRawBoneNode('chest'));
  const topY = chestBone >= 0 ? shirtData.bones[chestBone].y : Infinity;
  shirtData.points.forEach((p, i) => {
    if (!isTorso(shirt, shirtData, i) || p.y > topY) return;
    const rs = skinUnder(p);
    if (!rs) return;   // fără piele dedesubt: neschimbat
    const d = flat(p), R = rs + TORSO.gap;
    if (d <= R) return;
    const k = (R + (d - R) * TORSO.keep) / d;
    p.x = center.x + (p.x - center.x) * k;
    p.z = center.z + (p.z - center.z) * k;
    report.torso++;
  });

  writeBack(shirt, shirtData.points);
  console.info(`tricou strâmtat: ${report.sleeve} vârfuri pe mâneci, ${report.torso} pe trunchi`);
  return true;
}
