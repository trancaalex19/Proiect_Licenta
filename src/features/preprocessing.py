"""
Transformă secvența brută (cadre x 1662) în trăsături compacte și normalizate
(cadre x 192), folosite de model. Aceeași funcție se aplică la antrenare și live.

Pași:
  1. Se elimină fața; se păstrează 13 puncte din partea superioară a corpului
     (pose) și cele 21 de puncte ale fiecărei mâini, doar coordonatele (x, y).
  2. Normalizare față de umeri: se scade centrul umerilor și se împarte la
     lățimea umerilor (mediana pe secvență), ca poziția și distanța față de
     cameră să nu mai conteze.
  3. Golurile scurte în detecția unei mâini (mâna „pierdută” câteva cadre între
     două detecții) se completează prin interpolare liniară. Mâinile care chiar
     lipsesc (înainte de prima / după ultima detecție) rămân 0 și primesc un
     indicator de prezență (0/1).
  4. Se elimină cadrele fără mâini de la început și sfârșit (timp mort) și
     secvența se reeșantionează la SEQUENCE_LENGTH cadre.
  5. Forma fiecărei mâini (hand_shape): degetele față de încheietură, la scara
     palmei. La scara umerilor, diferența dintre pumn și palmă e mică; aici e clară.

Aranjarea trăsăturilor într-un cadru (192 de valori):
  [55 puncte x (x, y)] = 110 valori, apoi [prezență mâna stângă, prezență mâna dreaptă],
  apoi forma mâinii stângi (20 x 2) și a mâinii drepte (20 x 2) = 80 de valori.
  Punctele: 13 pose (POSE_LANDMARKS), 21 mâna stângă, 21 mâna dreaptă.
  Forma mâinilor se calculează mereu din puncte (from_points), deci rămâne
  corectă și după augmentare.
"""
import numpy as np

from src.config import SEQUENCE_LENGTH

# Indici în vectorul brut
_POSE_END = 33 * 4
_FACE_END = _POSE_END + 468 * 3
_LH_END = _FACE_END + 21 * 3

# Puncte MediaPipe Pose păstrate: nas, umeri, coate, încheieturi, degete (pinky/index/thumb)
POSE_LANDMARKS = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
_L_SHOULDER, _R_SHOULDER = 11, 12

N_POSE = len(POSE_LANDMARKS)
N_HAND = 21
N_POINTS = N_POSE + 2 * N_HAND               # 55
LH_SLICE = slice(N_POSE, N_POSE + N_HAND)     # punctele mâinii stângi
RH_SLICE = slice(N_POSE + N_HAND, N_POINTS)   # punctele mâinii drepte
BASE_DIM = N_POINTS * 2 + 2                   # 112: poziții + prezența mâinilor
SHAPE_DIM = 2 * (N_HAND - 1) * 2              # 80: forma celor două mâini
FEATURE_DIM = BASE_DIM + SHAPE_DIM            # 192

# Încheietura (0) și baza degetelor arătător / mijlociu / inelar / mic: dau mărimea palmei
_KNUCKLES = [5, 9, 13, 17]
_SHAPE_CLIP = 3.0  # mâna văzută din profil poate da valori extreme

# Perechi stânga/dreapta în POSE_LANDMARKS (poziții în listă), pentru oglindire
POSE_MIRROR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]


def hand_shape(hand, present):
    """
    Forma mâinii, independentă de poziția și mărimea ei în imagine: cele 20 de puncte
    ale degetelor față de încheietură, împărțite la mărimea palmei (distanța medie
    încheietură - baza degetelor). (T, 21, 2) -> (T, 20, 2); 0 când mâna lipsește.
    Diferențiază semne cu aceeași mișcare dar altă formă a mâinii (ex. pumn / palmă).
    """
    relative = hand[:, 1:] - hand[:, :1]
    palm = np.linalg.norm(relative[:, [k - 1 for k in _KNUCKLES]], axis=2).mean(axis=1)
    ok = present & (palm > 1e-6)
    shape = np.zeros_like(relative)
    shape[ok] = np.clip(relative[ok] / palm[ok, None, None], -_SHAPE_CLIP, _SHAPE_CLIP)
    return shape


def to_points(features):
    """(T, 192) -> puncte (T, 55, 2) și prezența mâinilor (T, 2). Forma mâinilor se ignoră:
    se recalculează din puncte în from_points."""
    points = features[:, :N_POINTS * 2].reshape(-1, N_POINTS, 2)
    presence = features[:, N_POINTS * 2:BASE_DIM]
    return points, presence


def from_points(points, presence):
    """Inversul lui to_points: adaugă forma mâinilor, calculată din punctele lor."""
    present = presence > 0.5
    shapes = [hand_shape(points[:, LH_SLICE], present[:, 0]), hand_shape(points[:, RH_SLICE], present[:, 1])]
    T = len(points)
    return np.concatenate([points.reshape(T, -1), presence] + [s.reshape(T, -1) for s in shapes],
                          axis=1).astype(np.float32)


def fill_gaps(points, present, max_gap_ratio=0.5):
    """
    Interpolează liniar o parte a corpului (T, N, D) în cadrele în care lipsește,
    dar doar ÎNTRE două detecții și doar pentru goluri de cel mult
    max_gap_ratio * T cadre. Returnează (puncte, prezență) actualizate.
    """
    points, present = points.copy(), present.copy()
    idx = np.where(present)[0]
    if len(idx) < 2:
        return points, present
    max_gap = max(1, int(len(points) * max_gap_ratio))
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            t = (np.arange(a + 1, b) - a) / (b - a)
            points[a + 1:b] = points[a] + (points[b] - points[a]) * t[:, None, None]
            present[a + 1:b] = True
    return points, present


def _normalize(raw, fill_hand_gaps=True):
    """(T, 1662) -> (T, 112), normalizat față de umeri."""
    pose = raw[:, :_POSE_END].reshape(-1, 33, 4)[:, :, :2]
    lh = raw[:, _FACE_END:_LH_END].reshape(-1, 21, 3)[:, :, :2]
    rh = raw[:, _LH_END:].reshape(-1, 21, 3)[:, :, :2]

    pose_present = np.any(pose != 0, axis=(1, 2))
    lh_present = np.any(lh != 0, axis=(1, 2))
    rh_present = np.any(rh != 0, axis=(1, 2))
    if fill_hand_gaps:
        lh, lh_present = fill_gaps(lh, lh_present)
        rh, rh_present = fill_gaps(rh, rh_present)

    # Centru și scară la nivel de secvență (mediana), ca mișcarea corpului să rămână vizibilă
    if pose_present.any():
        shoulders = pose[pose_present][:, [_L_SHOULDER, _R_SHOULDER]]
        center = np.median(shoulders.mean(axis=1), axis=0)
        width = np.median(np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1))
    else:
        center, width = np.array([0.5, 0.5]), 0.0
    if width < 1e-6:
        width = 0.25  # lățime tipică a umerilor în cadru, ca rezervă

    def norm(pts, present):
        out = (pts - center) / width
        out[~present] = 0.0
        return out

    points = np.concatenate([
        norm(pose[:, POSE_LANDMARKS], pose_present),
        norm(lh, lh_present),
        norm(rh, rh_present),
    ], axis=1)
    presence = np.stack([lh_present, rh_present], axis=1).astype(np.float32)
    return from_points(points, presence)


def resample(features, length=SEQUENCE_LENGTH):
    """Reeșantionează uniform în timp (cel mai apropiat cadru) la `length` cadre."""
    idx = np.round(np.linspace(0, len(features) - 1, length)).astype(int)
    return features[idx]


def trim_inactive(features, min_frames=5):
    """Elimină cadrele fără nicio mână de la început și sfârșit."""
    _, presence = to_points(features)
    active = np.where(presence.any(axis=1))[0]
    if len(active) < min_frames:
        return features
    return features[active[0]:active[-1] + 1]


def preprocess(raw_sequence, fill_hand_gaps=True):
    """Secvență brută (T, 1662) -> trăsături model (SEQUENCE_LENGTH, 112)."""
    raw = np.asarray(raw_sequence, dtype=np.float32)
    features = _normalize(raw, fill_hand_gaps=fill_hand_gaps)
    features = trim_inactive(features)
    return resample(features)
