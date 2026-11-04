"""
Evaluează un model pe semnele salvate de testul live (src/app/live_test.py).

Segmentele sunt salvate brute și complete, deci orice model nou (sau orice
schimbare de preprocesare) se poate compara pe exact aceleași încercări, fără
o nouă sesiune la cameră. Aceste date nu intră niciodată în antrenare.

Rulare (din rădăcina proiectului):
    python -m src.ml.evaluate_live                       # toate sesiunile
    python -m src.ml.evaluate_live --session 20261104    # doar sesiunile care încep cu acest text
    python -m src.ml.evaluate_live --model alt_model.h5
"""
import argparse
import glob
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np

from src.config import ACTIONS, LIVE_TEST_DIR, MODEL_PATH, NUM_CLASSES
from src.features.preprocessing import preprocess
from src.ml.model import fit_inputs

LABEL_MAP = {action: i for i, action in enumerate(ACTIONS)}


def predict_segment(model, segment):
    """Segment brut (T, 1662) -> probabilitățile celor 15 clase."""
    return model.predict(fit_inputs(model, preprocess(segment)[np.newaxis]), verbose=0)[0]


def load_live_test(session_prefix=''):
    """Fișierele salvate: data/live_test/<cuvânt>/<sesiune>_<k>.npy. Returnează (căi, etichete)."""
    paths, labels = [], []
    for action in ACTIONS:
        for path in sorted(glob.glob(os.path.join(LIVE_TEST_DIR, action, f'{session_prefix}*.npy'))):
            paths.append(path)
            labels.append(LABEL_MAP[action])
    return paths, np.array(labels, dtype=int)


def print_report(y_true, probabilities):
    """Acuratețe (top-1 și top-3), raport pe clase și matricea de confuzie."""
    from sklearn.metrics import classification_report, confusion_matrix

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    y_pred = probabilities.argmax(axis=1)
    top3 = np.mean([t in np.argsort(p)[-3:] for t, p in zip(y_true, probabilities)])
    print(f"\nAcuratețe live: {np.mean(y_pred == y_true):.1%} "
          f"({np.sum(y_pred == y_true)}/{len(y_true)} încercări), top-3: {top3:.1%}")
    print("\nRaport pe clase:")
    print(classification_report(y_true, y_pred, labels=range(NUM_CLASSES),
                                target_names=ACTIONS, zero_division=0))
    print("Matrice de confuzie (rânduri = cuvântul făcut, coloane = prezis):")
    print('          ' + ' '.join(f'{a[:4]:>4}' for a in ACTIONS))
    for action, row in zip(ACTIONS, confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))):
        print(f'{action:<10}' + ' '.join(f'{n:>4}' if n else '   .' for n in row))
    mistakes = [(ACTIONS[t], ACTIONS[p]) for t, p in zip(y_true, y_pred) if t != p]
    if mistakes:
        pairs, counts = np.unique(np.array(mistakes), axis=0, return_counts=True)
        order = np.argsort(-counts)[:5]
        print("\nCele mai dese confuzii: " +
              ', '.join(f"{pairs[i][0]}->{pairs[i][1]} ({counts[i]})" for i in order))


def main():
    parser = argparse.ArgumentParser(description='Evaluare pe testul live salvat')
    parser.add_argument('--model', default=MODEL_PATH)
    parser.add_argument('--session', default='', help='prefixul sesiunii (ex. 20261104)')
    args = parser.parse_args()

    paths, y_true = load_live_test(args.session)
    if not paths:
        print(f"Nu există încercări în {LIVE_TEST_DIR}. Rulează mai întâi: python -m src.app.live_test")
        return
    sessions = sorted({os.path.basename(p).rsplit('_', 1)[0] for p in paths})
    print(f"{len(paths)} încercări din {len(sessions)} sesiuni: {', '.join(sessions)}")

    from tensorflow.keras.models import load_model
    model = load_model(args.model)
    probabilities = [predict_segment(model, np.load(p)) for p in paths]
    print(f"Model: {args.model}")
    print_report(y_true, probabilities)


if __name__ == '__main__':
    main()
