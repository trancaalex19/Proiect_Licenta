"""
Mișcarea „tipică” a unui semn, din toate înregistrările proprii ale cuvântului, pentru avatar.

O singură înregistrare are cadre în care MediaPipe pierde mâna sau greșește degetele; avatarul
le-ar copia. Aici înregistrările se aliniază în timp cu DTW (Dynamic Time Warping) și se face
media lor cu DBA (DTW Barycenter Averaging, Petitjean et al., 2011), cu MEDIANA în locul mediei,
ca un cadru greșit dintr-o înregistrare să nu tragă rezultatul după el:

  1. fiecare înregistrare se normalizează față de umeri (centru și lățime, ca în preprocess) și
     se taie la cadrele cu mâini;
  2. secvența de start = medoidul (înregistrarea cu distanța DTW totală minimă față de celelalte);
     înregistrările mult mai depărtate decât restul (altă variantă, execuție ratată) se exclud;
  3. de ITERATIONS ori: fiecare înregistrare se aliniază cu DTW la secvența medie, iar fiecare
     cadru al mediei devine mediana cadrelor aliniate cu el. Pentru o mână contează doar cadrele
     în care MediaPipe chiar a detectat-o: mâna apare dacă e detectată în cel puțin jumătate din
     ele; încheietura și forma mâinii (degetele față de încheietură) se mediază separat.

Rezultatul se întoarce ca înregistrare brută (T, 1662) în coordonatele medoidului, deci intră
neschimbat în motion.raw_to_motion.
"""
import numpy as np

from src.avatar.motion import FACE_POINTS
from src.features.keypoints import FACE_END, L_SHOULDER, LH_END, LH_START, POSE_END, R_SHOULDER, RAW_DIM, RH_START
from src.features.preprocessing import fill_gaps, hand_shape

FACE_IDX = list(FACE_POINTS.values())
ARM_POINTS = [11, 12, 13, 14, 15, 16]   # umeri, coate, încheieturi (Pose)
ITERATIONS = 10
SHAPE_WEIGHT = 0.5      # cât contează forma mâinii la aliniere, față de poziția brațelor
OUTLIER_MADS = 3.0      # excludere: distanța față de medoid > mediana + 3 * MAD (robust)
MIN_HAND_VOTE = 0.5     # mâna apare dacă e detectată în cel puțin jumătate din cadrele aliniate


class Recording:
    """O înregistrare normalizată față de umeri și tăiată la cadrele cu mâini."""

    def __init__(self, raw, name=''):
        T = len(raw)
        pose = raw[:, :POSE_END].reshape(T, 33, 4)[:, :, :3]
        face = raw[:, POSE_END:FACE_END].reshape(T, 468, 3)[:, FACE_IDX]
        hands = np.stack([raw[:, LH_START:LH_END].reshape(T, 21, 3), raw[:, RH_START:].reshape(T, 21, 3)])
        detected = np.stack([np.any(h != 0, axis=(1, 2)) for h in hands], axis=1)   # (T, 2), fără interpolare
        pose, pose_ok = fill_gaps(pose, np.any(pose != 0, axis=(1, 2)))
        face, face_ok = fill_gaps(face, np.any(face != 0, axis=(1, 2)))
        filled = [fill_gaps(h, d) for h, d in zip(hands, detected.T)]
        present = np.stack([f[1] for f in filled], axis=1)
        hands = np.stack([f[0] for f in filled])

        active = np.where(present.any(axis=1))[0]
        if len(active) == 0:
            raise ValueError(f'{name}: nicio mână detectată')
        keep = slice(active[0], active[-1] + 1)
        shoulders = pose[pose_ok][:, [L_SHOULDER, R_SHOULDER]]
        self.center = np.median(shoulders.mean(axis=1), axis=0)                          # (3,)
        self.width = np.median(np.linalg.norm(shoulders[:, 0, :2] - shoulders[:, 1, :2], axis=1))
        self.name = name
        self.pose = self._norm(pose[keep])
        self.face = self._norm(face[keep]) if face_ok.any() else None
        self.hands = self._norm(hands[:, keep], hand=True)
        self.detected = detected[keep]
        self.present = present[keep]

    def _norm(self, points, hand=False):
        out = points.copy()
        out[..., :2] = (points[..., :2] - self.center[:2]) / self.width
        # z-ul punctelor Hands e deja relativ la încheietură; pentru Pose/Face se scade z-ul umerilor
        out[..., 2] = (points[..., 2] - (0 if hand else self.center[2])) / self.width
        return out

    def _denorm(self, points, hand=False):
        out = points * self.width
        out[..., :2] += self.center[:2]
        if not hand:
            out[..., 2] += self.center[2]
        return out

    def to_raw(self, pose, face, hands, present):
        """Secvența (normalizată) -> înregistrare brută (T, 1662) în coordonatele acestei înregistrări."""
        T = len(pose)
        raw = np.zeros((T, RAW_DIM), dtype=np.float32)
        raw[:, :POSE_END] = np.concatenate([self._denorm(pose), np.ones((T, 33, 1))], axis=2).reshape(T, -1)
        if face is not None:
            full = np.zeros((T, 468, 3), dtype=np.float32)
            full[:, FACE_IDX] = self._denorm(face)
            raw[:, POSE_END:FACE_END] = full.reshape(T, -1)
        for k, (start, end) in enumerate(((LH_START, LH_END), (RH_START, RAW_DIM))):
            h = self._denorm(hands[k], hand=True)
            h[~present[:, k]] = 0
            raw[:, start:end] = h.reshape(T, -1)
        return raw


def features(pose, hands, present):
    """Vectorii pe care se face alinierea DTW: brațele (x, y), încheieturile și forma mâinilor."""
    T = len(pose)
    parts = [pose[:, ARM_POINTS, :2].reshape(T, -1)]
    for k in range(2):
        ok = present[:, k]
        xy = hands[k][..., :2]
        parts += [np.where(ok[:, None], xy[:, 0], 0),
                  SHAPE_WEIGHT * hand_shape(xy, ok).reshape(T, -1),
                  ok[:, None].astype(float)]
    return np.concatenate(parts, axis=1)


def dtw(a, b):
    """Cost total și drumul de aliniere DTW între secvențele de vectori a (n, d) și b (m, d)."""
    cost = np.linalg.norm(a[:, None] - b[None], axis=2)
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    path, i, j = [], n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]])
        i, j = (i - 1, j - 1) if step == 0 else (i - 1, j) if step == 1 else (i, j - 1)
    return acc[n, m], path[::-1]


def canonical(raws, names=None):
    """
    Înregistrările brute ale unui cuvânt -> (raw (T, 1662) al mișcării tipice, raport dict).
    """
    names = names or [str(i) for i in range(len(raws))]
    recs = [Recording(np.asarray(r, dtype=np.float32), n) for r, n in zip(raws, names)]
    feats = [features(r.pose, r.hands, r.present) for r in recs]
    n = len(recs)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = dist[j, i] = dtw(feats[i], feats[j])[0]
    medoid = int(np.argmin(dist.sum(axis=1)))
    d = dist[medoid]
    others = np.delete(d, medoid)
    limit = np.median(others) + OUTLIER_MADS * 1.4826 * np.median(np.abs(others - np.median(others)))
    used = [i for i in range(n) if d[i] <= limit]
    base = recs[medoid]

    pose, face, hands, present = base.pose.copy(), None if base.face is None else base.face.copy(), \
        base.hands.copy(), base.present.copy()
    for _ in range(ITERATIONS):
        avg_feat = features(pose, hands, present)
        groups = [[] for _ in range(len(pose))]   # (înregistrare, cadru) aliniate cu fiecare cadru al mediei
        for i in used:
            for j, k in dtw(avg_feat, feats[i])[1]:
                groups[j].append((recs[i], k))
        new_pose = np.stack([np.median([r.pose[k] for r, k in g], axis=0) for g in groups])
        if face is not None:
            face = np.stack([np.median([r.face[k] for r, k in g if r.face is not None], axis=0)
                             if any(r.face is not None for r, _ in g) else face[j] for j, g in enumerate(groups)])
        new_hands, new_present = hands.copy(), np.zeros_like(present)
        for j, g in enumerate(groups):
            for s in range(2):
                seen = [r.hands[s][k] for r, k in g if r.detected[k, s]]
                if len(seen) >= MIN_HAND_VOTE * len(g):
                    seen = np.stack(seen)
                    wrist = np.median(seen[:, 0], axis=0)
                    new_hands[s, j] = wrist + np.median(seen - seen[:, :1], axis=0)
                    new_present[j, s] = True
        if np.array_equal(new_present, present) and np.allclose(new_pose, pose, atol=1e-4) \
                and np.allclose(new_hands, hands, atol=1e-4):
            break
        pose, hands, present = new_pose, new_hands, new_present

    for s in range(2):   # goluri izolate rămase după vot: interpolare, ca la preprocesare
        hands[s], present[:, s] = fill_gaps(hands[s], present[:, s])
    report = {'base': base.name, 'used': len(used), 'excluded': [recs[i].name for i in range(n) if i not in used],
              'frames': len(pose)}
    return base.to_raw(pose, face, hands, present), report
