"""
Compară configurații prin validare încrucișată stratificată pe train+val
(videoclipuri originale). Setul de TEST nu este folosit aici deloc.

Fiecare fold se antrenează un număr fix de epoci (fără early stopping pe
fold-ul de evaluare, ca estimarea să nu fie optimistă).

Rulare (din rădăcina proiectului):
    python -m src.ml.cross_validate
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

from src.config import NUM_CLASSES, SEED
from src.features.preprocessing import BASE_DIM
from src.ml.dataset import AugmentedSequence, load_recorded, load_split, load_splits
from src.ml.model import build_model

N_SPLITS = 3
N_REPEATS = 2
DEFAULT_EPOCHS = 150

# Configurațiile comparate. 'augment': None = fără augmentare, altfel argumentele
# pentru src.features.augmentation.augment.
#
# Runda 1 (3x2 fold-uri, 150 epoci):
#   logreg (referință)            64.6% ± 8.7%
#   lstm fără augmentare          69.2% ± 8.1%
#   lstm cu augmentare completă   63.1% ± 5.8%
# Runda 2 (3x2 fold-uri, 250 epoci):
#   lstm fără aug                 62.7% ± 10.2%
#   lstm aug fără oglindire       70.1% ± 8.8%   <- ales (implicit în augment())
#   lstm aug ușoară (fără warp)   63.5% ± 8.6%
# Atenție: TF pe CPU nu e complet determinist; diferențe sub ~5 puncte nu sunt sigure.
#
# Runda 3 (3x2 fold-uri, 250 epoci), completarea golurilor mâinilor:
#   aug, fără completare          65.6% ± 11.4%
#   aug, cu completare            68.8% ± 11.2%  <- păstrat (implicit în preprocess())
#
# --- Setul de date curățat: 31 de videoclipuri WLASL eliminate (data/cleaning_log.json),
# --- reîmpărțit 62/17/17. Rundele de mai sus NU sunt comparabile cu cele de mai jos.
#
# Runda 4 (79 videoclipuri train+val, 3x2 fold-uri, 250 epoci): efectul înregistrărilor
# proprii (adăugate doar la antrenarea fiecărui fold; evaluarea rămâne pe semnatari WLASL):
#   fără înregistrări proprii     68.4% ± 12.9%
#   cu înregistrări proprii       73.4% ± 6.2%   <- folosite (USE_RECORDED_DATA în config)
#
# Runda 5 (aceleași date, cu înregistrări proprii): forma mâinii (src/features/preprocessing.py,
# hand_shape). Testul live a arătat confuzii între semne cu aceeași mișcare și altă formă
# a mâinii (need->yes, sorry->please, help->good). 'hand_shape': False = doar primele
# BASE_DIM trăsături (varianta veche). 78 videoclipuri train+val (după eliminarea no/video_1).
#   fără forma mâinii             73.1% ± 4.4%   (max 75.0% la ep229)
#   cu forma mâinii               78.2% ± 7.9%   (max 82.7% la ep157)  <- folosită (implicit)
#
# Runda 6: + 10 înregistrări proprii pentru „computer” (avea 0) și 10 pentru „stop”;
# curba pe epoci alege numărul fix de epoci pentru train_lstm (FIXED_EPOCHS).
#   cu forma mâinii               80.1% ± 5.6%   (ep100=80.8%, ep150=80.8%, ep200=84.6%;
#                                                 max 85.9% la ep168 -> FIXED_EPOCHS = 170)
#
# --- Runda 7: date schimbate (rundele de mai sus NU sunt comparabile): 10 videoclipuri „help”
# --- proaste eliminate (69 train+val), 3 exemple stângace oglindite (cleaning_log 'mirrored'),
# --- înregistrări proprii completate până la 24 de exemple pe cuvânt (275 în total).
#   cu forma mâinii               83.3% ± 10.5%  (ep50=83.3%, ep100=81.2%, ep150=87.0%, ep200=82.6%;
#                                                 max 88.4% la ep153 -> FIXED_EPOCHS = 150)
CONFIGS = {
    'cu forma mâinii': {'type': 'lstm', 'augment': {}, 'epochs': 250, 'use_recorded': True,
                        'hand_shape': True},
}


def run_logreg(X_tr, y_tr, X_ev, y_ev):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    clf.fit(X_tr.reshape(len(X_tr), -1), y_tr)
    return clf.score(X_ev.reshape(len(X_ev), -1), y_ev), None


def run_lstm(X_tr, y_tr, X_ev, y_ev, augment_options, epochs, seed):
    tf.keras.utils.set_random_seed(seed)
    model = build_model(feature_dim=X_tr.shape[-1])
    train_seq = AugmentedSequence(X_tr, y_tr, seed=seed, augment_data=augment_options is not None,
                                  augment_options=augment_options)
    history = model.fit(train_seq, epochs=epochs, verbose=0,
                        validation_data=(X_ev, to_categorical(y_ev, NUM_CLASSES)))
    curve = np.array(history.history['val_categorical_accuracy'])
    return curve[-1], curve


def load_train_val(splits, fill_hand_gaps=True):
    """Train + val concatenate (setul de test nu e folosit niciodată aici)."""
    X_tr, y_tr = load_split(splits, 'train', fill_hand_gaps=fill_hand_gaps)
    X_va, y_va = load_split(splits, 'val', fill_hand_gaps=fill_hand_gaps)
    return np.concatenate([X_tr, X_va]), np.concatenate([y_tr, y_va])


def main():
    splits = load_splits()
    _, y = load_train_val(splits)
    print(f"Validare încrucișată pe {len(y)} videoclipuri originale "
          f"({N_SPLITS} fold-uri x {N_REPEATS} repetări)")

    # Aceleași fold-uri pentru toate configurațiile (depind doar de etichete)
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    folds = list(cv.split(np.zeros(len(y)), y))

    for name, cfg in CONFIGS.items():
        fill = cfg.get('fill_hand_gaps', True)
        X, y = load_train_val(splits, fill_hand_gaps=fill)
        # Înregistrările proprii se adaugă doar la partea de antrenare a fiecărui fold
        if cfg.get('use_recorded', False):
            X_rec, y_rec = load_recorded(fill)
        else:
            X_rec, y_rec = X[:0], y[:0]
        if not cfg.get('hand_shape', True):
            X, X_rec = X[..., :BASE_DIM], X_rec[..., :BASE_DIM]
        scores, curves = [], []
        for k, (tr, ev) in enumerate(folds):
            X_tr = np.concatenate([X[tr], X_rec])
            y_tr = np.concatenate([y[tr], y_rec])
            if cfg['type'] == 'logreg':
                acc, curve = run_logreg(X_tr, y_tr, X[ev], y[ev])
            else:
                acc, curve = run_lstm(X_tr, y_tr, X[ev], y[ev], cfg['augment'],
                                      cfg.get('epochs', DEFAULT_EPOCHS), seed=SEED + k)
                curves.append(curve)
            scores.append(acc)
        line = f"{name:<26} acuratețe = {np.mean(scores):.1%} ± {np.std(scores):.1%}"
        if curves:
            mean_curve = np.mean(curves, axis=0)
            marks = ', '.join(f"ep{e}={mean_curve[e - 1]:.1%}" for e in (50, 100, 150, 200, 250)
                              if e <= len(mean_curve))
            line += f"   (medie pe epoci: {marks}; max={mean_curve.max():.1%} la ep{mean_curve.argmax() + 1})"
        print(line, flush=True)


if __name__ == '__main__':
    main()
