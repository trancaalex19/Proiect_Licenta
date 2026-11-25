"""
Antrenarea finală a modelului LSTM.

  - date: videoclipurile originale din data/splits.json (train / val / test)
  - preprocesare: src/features/preprocessing.py (aceeași ca la predicția live)
  - augmentare: aleatoare la fiecare epocă, DOAR pe train
  - epoci: număr fix (FIXED_EPOCHS, ales prin validare încrucișată), sau, dacă e None,
    oprire timpurie, reducerea learning rate-ului și alegerea modelului după val
  - ansamblu: cu epoci fixe se antrenează câte un model pentru fiecare seed din
    ENSEMBLE_SEEDS, iar modelul salvat face media probabilităților lor
  - test: evaluat o singură dată, la final

Rulare (din rădăcina proiectului):
    python -m src.ml.train_lstm
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from src.config import ACTIONS, MODEL_PATH, NUM_CLASSES, SEED, USE_RECORDED_DATA
from src.ml.dataset import AugmentedSequence, load_recorded, load_split, load_splits
from src.ml.model import build_model, ensemble

MAX_EPOCHS = 300
BATCH_SIZE = 16
# Număr fix de epoci, ales din curba validării încrucișate (src/ml/cross_validate.py):
# modelul final se antrenează exact cum a fost evaluat acolo. Setul de val e prea mic
# pentru o oprire timpurie sigură; se raportează doar. None = oprire timpurie pe val.
FIXED_EPOCHS = 200
# Ansamblu (2026-11-25): același model antrenat cu seed-uri diferite dă rezultate live foarte
# diferite (date vechi: 78-90% pe sesiunea live 20261111, 85-92% pe 20261125), deci un singur
# model e o loterie. Cu cele 50 de înregistrări noi (my, please, stomach, water, drink), 3 seed-uri:
#   un model:              92% / 92-94%      (128 de unități LSTM în loc de 64: fără câștig)
#   media celor 3 modele:  95.1% / 97.0%     <- folosită (modelul vechi: 91.2% / 97.0%)
# Atenție: ambele sesiuni live au servit la această alegere; cifra finală pentru teză vine
# dintr-o sesiune live nouă.
ENSEMBLE_SEEDS = (1, 2, 3)


def main():
    tf.keras.utils.set_random_seed(SEED)

    # 1. Date (doar videoclipuri originale, împărțirea fixă)
    splits = load_splits()
    X_train, y_train = load_split(splits, 'train')
    X_val, y_val = load_split(splits, 'val')
    X_test, y_test = load_split(splits, 'test')
    print(f"Videoclipuri originale: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Înregistrările proprii intră DOAR în antrenare (val și test rămân semnatari WLASL)
    if USE_RECORDED_DATA:
        X_rec, y_rec = load_recorded()
        if len(X_rec):
            X_train, y_train = np.concatenate([X_train, X_rec]), np.concatenate([y_train, y_rec])
            counts = ', '.join(f"{ACTIONS[c]}={n}" for c, n in zip(*np.unique(y_rec, return_counts=True)))
            print(f"+ {len(X_rec)} înregistrări proprii în antrenare ({counts})")
    print(f"Forma unei secvențe: {X_train.shape[1:]} (cadre, trăsături)")

    # 2-3. Antrenare: augmentare aleatoare la fiecare epocă, doar pe train
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    validation = (X_val, to_categorical(y_val, NUM_CLASSES))
    if FIXED_EPOCHS:
        # ca în validarea încrucișată: epoci fixe, un model pentru fiecare seed, apoi media lor
        members = []
        for k, seed in enumerate(ENSEMBLE_SEEDS, 1):
            print(f"Modelul {k}/{len(ENSEMBLE_SEEDS)} (seed {seed}): {FIXED_EPOCHS} epoci fixe...", flush=True)
            tf.keras.utils.set_random_seed(seed)
            member = build_model()
            if k == 1:
                member.summary()
            member.fit(AugmentedSequence(X_train, y_train, batch_size=BATCH_SIZE, seed=seed),
                       validation_data=validation, epochs=FIXED_EPOCHS, verbose=2)
            print(f"  val: {np.mean(np.argmax(member.predict(X_val, verbose=0), axis=1) == y_val):.2%}")
            members.append(member)
        model = ensemble(members) if len(members) > 1 else members[0]
        model.save(MODEL_PATH)
    else:
        # oprire timpurie: callbacks pe val_loss, ca modelul salvat = modelul evaluat
        model = build_model()
        model.summary()
        print("Începem antrenarea modelului (oprire timpurie pe val)...")
        model.fit(
            AugmentedSequence(X_train, y_train, batch_size=BATCH_SIZE, seed=SEED),
            validation_data=validation,
            epochs=MAX_EPOCHS,
            callbacks=[
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-5, verbose=1),
                EarlyStopping(monitor='val_loss', patience=60, restore_best_weights=True),
                ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True),
            ],
            verbose=2,
        )

    # 4. Evaluare finală, o singură dată, pe setul de test
    val_acc = np.mean(np.argmax(model.predict(X_val, verbose=0), axis=1) == y_val)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    test_acc = np.mean(y_pred == y_test)
    print(f"\nAcuratețe pe VAL  ({len(y_val)} videoclipuri): {val_acc:.2%}")
    print(f"Acuratețe pe TEST ({len(y_test)} videoclipuri nevăzute): {test_acc:.2%}")
    print("\nRaport pe clase (test):")
    print(classification_report(y_test, y_pred, labels=range(NUM_CLASSES),
                                target_names=ACTIONS, zero_division=0))
    print("Matrice de confuzie (rânduri = adevărat, coloane = prezis):")
    print(confusion_matrix(y_test, y_pred, labels=range(NUM_CLASSES)))
    saved = (f"media a {len(ENSEMBLE_SEEDS)} modele, {FIXED_EPOCHS} epoci fiecare" if FIXED_EPOCHS
             else "cel mai bun după val_loss")
    print(f"\nModelul ({saved}) a fost salvat în {MODEL_PATH}")


if __name__ == '__main__':
    main()
