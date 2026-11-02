"""
Împarte videoclipurile ORIGINALE în seturi train / val / test, stratificat pe clase.

Împărțirea se face la nivel de videoclip original (un fișier .npy = un video),
înainte de orice augmentare, ca nicio copie a unui video din val/test să nu
ajungă în antrenare. Rezultatul este salvat în data/splits.json.

Rulare (din rădăcina proiectului):
    python -m src.data.split_data
"""
import json
import os
import random

from src.config import ACTIONS, KEYPOINTS_DIR, SEED, SPLITS_FILE

VAL_RATIO = 0.15
TEST_RATIO = 0.15


def split_class(files, rng):
    """Împarte lista de fișiere a unei clase. Garantează minim 1 video în val și în test."""
    files = sorted(files)  # ordine deterministă, independentă de sistemul de fișiere
    rng.shuffle(files)
    n = len(files)
    if n < 3:
        raise ValueError(f"Prea puține videoclipuri ({n}) pentru train/val/test.")
    n_test = max(1, round(n * TEST_RATIO))
    n_val = max(1, round(n * VAL_RATIO))
    test = files[:n_test]
    val = files[n_test:n_test + n_val]
    train = files[n_test + n_val:]
    return train, val, test


def make_splits():
    rng = random.Random(SEED)
    splits = {'seed': SEED, 'train': {}, 'val': {}, 'test': {}}

    print(f"{'clasa':<10} {'total':>5} {'train':>5} {'val':>4} {'test':>4}")
    for action in ACTIONS:
        action_path = os.path.join(KEYPOINTS_DIR, action)
        # fișierele aug_* sunt rămășițe ale vechii augmentări pe disc și se ignoră
        originals = [f for f in os.listdir(action_path)
                     if f.endswith('.npy') and not f.startswith('aug_')]
        train, val, test = split_class(originals, rng)
        splits['train'][action] = train
        splits['val'][action] = val
        splits['test'][action] = test
        print(f"{action:<10} {len(originals):>5} {len(train):>5} {len(val):>4} {len(test):>4}")

    totals = {k: sum(len(v) for v in splits[k].values()) for k in ('train', 'val', 'test')}
    print(f"\nTotal: train={totals['train']}, val={totals['val']}, test={totals['test']}")

    with open(SPLITS_FILE, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2)
    print(f"Împărțirea a fost salvată în {SPLITS_FILE}")


if __name__ == '__main__':
    make_splits()
