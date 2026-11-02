"""
Încărcarea datelor pe baza împărțirii fixe din data/splits.json și generatorul
care aplică augmentarea aleator la fiecare epocă (doar pentru antrenare).
"""
import json
import os

import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical

from src.config import ACTIONS, KEYPOINTS_DIR, NUM_CLASSES, RECORDED_DIR, SEQUENCE_LENGTH, SPLITS_FILE
from src.features.augmentation import augment
from src.features.preprocessing import FEATURE_DIM, preprocess

LABEL_MAP = {action: i for i, action in enumerate(ACTIONS)}


def load_splits():
    if not os.path.exists(SPLITS_FILE):
        raise FileNotFoundError(
            f"Nu există {SPLITS_FILE}. Rulează mai întâi: python -m src.data.split_data")
    with open(SPLITS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_split(splits, split_name, fill_hand_gaps=True):
    """Încarcă și preprocesează videoclipurile ORIGINALE dintr-un set. Returnează (X, y)."""
    X, y = [], []
    for action in ACTIONS:
        for file in splits[split_name][action]:
            raw = np.load(os.path.join(KEYPOINTS_DIR, action, file))
            X.append(preprocess(raw, fill_hand_gaps=fill_hand_gaps))
            y.append(LABEL_MAP[action])
    return np.array(X, dtype=np.float32), np.array(y)


def recorded_files(action):
    """Fișierele înregistrate de utilizator pentru un cuvânt (data/recorded_keypoints/<cuvânt>/)."""
    folder = os.path.join(RECORDED_DIR, action)
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy'))


def load_recorded(fill_hand_gaps=True):
    """
    Încarcă înregistrările proprii (doar pentru ANTRENARE). Returnează (X, y);
    seturi goale dacă nu există înregistrări.
    """
    X, y = [], []
    for action in ACTIONS:
        for path in recorded_files(action):
            X.append(preprocess(np.load(path), fill_hand_gaps=fill_hand_gaps))
            y.append(LABEL_MAP[action])
    if not X:
        return np.zeros((0, SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32), np.zeros(0, dtype=int)
    return np.array(X, dtype=np.float32), np.array(y)


class AugmentedSequence(Sequence):
    """Generator Keras: la fiecare epocă, fiecare video primește o augmentare nouă."""

    def __init__(self, X, y, batch_size=16, seed=0, augment_data=True, augment_options=None):
        super().__init__()
        self.X, self.y = X, y
        self.batch_size = batch_size
        self.augment_data = augment_data
        self.augment_options = augment_options or {}  # argumente pentru augmentation.augment
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, i):
        idx = self.order[i * self.batch_size:(i + 1) * self.batch_size]
        batch = self.X[idx]
        if self.augment_data:
            batch = np.stack([augment(x, self.rng, **self.augment_options) for x in batch])
        return batch, to_categorical(self.y[idx], num_classes=NUM_CLASSES)

    def on_epoch_end(self):
        self.order = self.rng.permutation(len(self.X))
