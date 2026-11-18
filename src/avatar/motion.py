"""
Transformă o înregistrare brută (T, 1662) în mișcarea pe care o execută avatarul:
punctele corpului, ale mâinilor și câteva puncte ale feței, în coordonatele scenei 3D.

Pagina (src/avatar/web/avatar.js) orientează fiecare os al avatarului după direcția dintre
două puncte (ex. braț = umăr -> cot), deci contează doar direcțiile, nu poziția sau scara.

Coordonatele scenei: X = stânga semnatarului (dreapta imaginii, camera nu e oglindită),
Y = în sus, Z = spre cameră. Avatarul stă cu fața spre +Z, deci mâna lui stângă e la +X,
ca un om care semnează în fața pacientului (nu în oglindă).
"""
import numpy as np

from src.config import CAMERA_ASPECT
from src.features.keypoints import FACE_END, LH_END, LH_START, POSE_END, RH_START
from src.features.preprocessing import fill_gaps

# Punctele feței folosite de avatar (indexuri MediaPipe Face Mesh, din 468):
# orientarea capului (frunte-bărbie, colțurile exterioare ale ochilor) și gura.
FACE_POINTS = {
    'forehead': 10,
    'chin': 152,
    'eye_right': 33,    # ochiul drept al semnatarului
    'eye_left': 263,
    'lip_top': 13,
    'lip_bottom': 14,
}
DECIMALS = 4   # precizie suficientă pentru direcții (coordonate ~0..1), JSON mic


def to_scene(points, aspect=CAMERA_ASPECT):
    """Coordonate MediaPipe (x spre dreapta, y în jos, z spre fundal) -> coordonatele scenei."""
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return np.stack([x - 0.5, -(y - 0.5) * aspect, -z], axis=-1)


def smooth(points, present):
    """Netezire temporală ușoară (0.25, 0.5, 0.25), doar unde vecinii există."""
    out = points.copy()
    for t in range(1, len(points) - 1):
        if present[t - 1:t + 2].all():
            out[t] = 0.25 * points[t - 1] + 0.5 * points[t] + 0.25 * points[t + 1]
    return out


def side_weight(present):
    """
    Cât de mult urmează brațul datele (1) față de poziția de repaus (0), pe cadre.
    Când mâna nu e detectată, brațul e de obicei jos, în afara cadrului, iar MediaPipe
    doar ghicește poziția lui; avatarul ține atunci brațul în repaus. Marginile se netezesc
    ca brațul să nu sară.
    """
    p = np.pad(present.astype(float), 1, mode='edge')
    return (p[:-2] + p[1:-1] + p[2:]) / 3


def _part(raw, start, end, n):
    points = raw[:, start:end].reshape(len(raw), n, -1)[:, :, :3]
    present = np.any(points != 0, axis=(1, 2))
    points, present = fill_gaps(points, present)
    return smooth(points, present), present


def raw_to_motion(raw, fps):
    """
    (T, 1662) -> dict serializabil JSON cu cadrele semnului (fără cadrele de la început și
    sfârșit în care nu se vede nicio mână). `fps` = cadre pe secundă ale înregistrării.
    """
    pose, pose_present = _part(raw, 0, POSE_END, 33)
    face, face_present = _part(raw, POSE_END, FACE_END, 468)
    lh, lh_present = _part(raw, LH_START, LH_END, 21)
    rh, rh_present = _part(raw, RH_START, raw.shape[1], 21)

    active = np.where(lh_present | rh_present)[0]
    if len(active) == 0:
        raise ValueError('nicio mână detectată în înregistrare')
    keep = slice(active[0], active[-1] + 1)
    weights = np.stack([side_weight(lh_present), side_weight(rh_present)], axis=1)[keep]

    def points_or_none(points, present):
        return [np.round(to_scene(p), DECIMALS).tolist() if ok else None for p, ok in zip(points, present)]

    face_selected = face[:, list(FACE_POINTS.values())]
    faces = points_or_none(face_selected, face_present)
    frames = []
    for i, (p, f, l, r, w) in enumerate(zip(
            points_or_none(pose, pose_present)[keep], faces[keep],
            points_or_none(lh, lh_present)[keep], points_or_none(rh, rh_present)[keep], weights)):
        frames.append({
            'pose': p,
            'face': dict(zip(FACE_POINTS, f)) if f is not None else None,
            'left_hand': l,
            'right_hand': r,
            'weight': np.round(w, 3).tolist(),   # [stânga, dreapta]
        })
    return {'fps': fps, 'frames': frames}
