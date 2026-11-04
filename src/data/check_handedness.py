"""
Caută exemplele făcute cu mâna stângă (semnatar stângaci sau înregistrare greșită).

Modelul și ghidul presupun un semnatar dreptaci. Un exemplu făcut cu mâna stângă
arată ca semnul în oglindă și îl încurcă pe model. Pentru fiecare exemplu
(WLASL și înregistrări proprii) se compară distanța până la media celorlalte
exemple ale aceluiași cuvânt, în varianta originală și în cea oglindită. Dacă
varianta oglindită e clar mai apropiată (raport < THRESHOLD), exemplul e suspect.

Semnele simetrice (ambele mâini fac același lucru: 'both' în src/app/sign_info.py, ex. book,
teacher) se sar: acolo oglindirea nu schimbă nimic real, iar scorul ar fi doar zgomot.

Exemplele confirmate (verificate vizual) se adaugă în data/cleaning_log.json, în lista
'mirrored', și sunt oglindite automat la încărcare (src/ml/dataset.py) și în ghid.
Rulat din nou, scriptul ține cont de listă: nu ar trebui să mai marcheze nimic.

Rulare (din rădăcina proiectului):
    python -m src.data.check_handedness
"""
import glob
import os

import numpy as np

from src.app.sign_info import SIGN_INFO
from src.config import ACTIONS, KEYPOINTS_DIR, RECORDED_DIR
from src.features.augmentation import mirror
from src.ml.dataset import load_example, mirrored_files

THRESHOLD = 0.85


def main():
    mirrored = mirrored_files()
    print(f"Deja oglindite (data/cleaning_log.json): {len(mirrored)}")
    suspects = 0
    symmetric = [a for a in ACTIONS if 'both' in SIGN_INFO[a]]
    print(f"Semne simetrice, sărite: {', '.join(symmetric)}")
    for action in ACTIONS:
        if action in symmetric:
            continue
        paths = sorted(glob.glob(os.path.join(KEYPOINTS_DIR, action, '*.npy'))) + \
                sorted(glob.glob(os.path.join(RECORDED_DIR, action, '*.npy')))
        X = np.array([load_example(action, p, mirrored) for p in paths])
        for i, path in enumerate(paths):
            reference = np.delete(X, i, axis=0).mean(axis=0)
            ratio = np.linalg.norm(mirror(X[i]) - reference) / np.linalg.norm(X[i] - reference)
            if ratio < THRESHOLD:
                suspects += 1
                source = 'proprie' if path.startswith(RECORDED_DIR) else 'WLASL'
                print(f"  SUSPECT  {action:<9} {source:<8} {os.path.basename(path):<32} "
                      f"oglindit / original = {ratio:.2f}")
    if suspects:
        print(f"\n{suspects} exemple par făcute cu mâna stângă. Verifică-le (ghidul / traiectoria mâinilor),\n"
              "apoi adaugă-le în data/cleaning_log.json la 'mirrored' sau refă înregistrarea.")
    else:
        print("Niciun exemplu suspect: toate par făcute cu mâna dreaptă.")


if __name__ == '__main__':
    main()
