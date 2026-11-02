"""
Augmentare pentru trăsăturile preprocesate (T, 112), aplicată DOAR pe setul de
antrenare, aleator la fiecare epocă (vezi src/ml/dataset.py).

Toate transformările lasă neatinse mâinile nedetectate (rămân 0).
"""
import numpy as np

from src.config import SEQUENCE_LENGTH
from src.features.preprocessing import (
    LH_SLICE, RH_SLICE, POSE_MIRROR_PAIRS,
    to_points, from_points, resample,
)


def mirror(features):
    """Oglindire stânga <-> dreapta (simulează un semnatar stângaci)."""
    points, presence = to_points(features.copy())
    points[:, :, 0] *= -1
    for a, b in POSE_MIRROR_PAIRS:
        points[:, [a, b]] = points[:, [b, a]]
    lh, rh = points[:, LH_SLICE].copy(), points[:, RH_SLICE].copy()
    points[:, LH_SLICE], points[:, RH_SLICE] = rh, lh
    return from_points(points, presence[:, ::-1])


def affine(features, rng, max_rotation_deg=10, scale_range=0.1, max_shift=0.1):
    """Rotație, scalare și deplasare mică (în unități de lățime a umerilor)."""
    points, presence = to_points(features.copy())
    mask = np.any(points != 0, axis=2, keepdims=True)
    angle = np.deg2rad(rng.uniform(-max_rotation_deg, max_rotation_deg))
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    scale = rng.uniform(1 - scale_range, 1 + scale_range)
    shift = rng.uniform(-max_shift, max_shift, size=2)
    points = np.where(mask, points @ rot.T * scale + shift, 0.0)
    return from_points(points, presence)


def add_noise(features, rng, sigma=0.02):
    """Zgomot gaussian mic pe coordonate (imperfecțiuni ale detecției)."""
    points, presence = to_points(features.copy())
    mask = np.any(points != 0, axis=2, keepdims=True)
    points = np.where(mask, points + rng.normal(0, sigma, points.shape), 0.0)
    return from_points(points, presence)


def time_warp(features, rng, min_keep=0.8):
    """Variație de viteză: decupează aleator 80-100% din secvență și reeșantionează."""
    n = len(features)
    keep = int(round(n * rng.uniform(min_keep, 1.0)))
    start = rng.integers(0, n - keep + 1)
    return resample(features[start:start + keep], SEQUENCE_LENGTH)


def augment(features, rng, p_mirror=0.0, warp=True, geometric=True, noise=True):
    """
    Aplică o combinație aleatoare de augmentări unei secvențe (fiecare poate fi oprită).
    Oglindirea e oprită implicit: în validarea încrucișată a scăzut acuratețea
    (majoritatea semnatarilor WLASL folosesc mâna dreaptă).
    """
    out = features
    if p_mirror > 0 and rng.random() < p_mirror:
        out = mirror(out)
    if warp:
        out = time_warp(out, rng)
    if geometric:
        out = affine(out, rng)
    if noise:
        out = add_noise(out, rng)
    return out
