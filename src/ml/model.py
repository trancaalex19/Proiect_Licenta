"""Arhitectura rețelei, separată de scriptul de antrenare, și predicția pe un semn brut."""
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LSTM, Average, Dense, Dropout, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

from src.config import NUM_CLASSES, SEQUENCE_LENGTH
from src.features.preprocessing import FEATURE_DIM, preprocess


def fit_inputs(model, X):
    """
    Păstrează din trăsături doar câte așteaptă modelul. Trăsăturile noi se adaugă la
    final (ex. forma mâinii după primele 112), deci un model mai vechi rămâne utilizabil.
    """
    return X[..., :model.input_shape[-1]]


def predict_segment(model, segment):
    """Segment brut (T, 1662) -> probabilitățile claselor."""
    return model.predict(fit_inputs(model, preprocess(segment)[np.newaxis]), verbose=0)[0]


def build_model(units=64, dense_units=64, dropout=0.4, l2=1e-4, learning_rate=1e-3,
                feature_dim=FEATURE_DIM):
    """LSTM mic, potrivit pentru câteva sute de exemple de antrenare."""
    reg = regularizers.l2(l2)
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, feature_dim)),
        LSTM(units, return_sequences=True, kernel_regularizer=reg),
        Dropout(dropout),
        LSTM(units, kernel_regularizer=reg),
        Dropout(dropout),
        Dense(dense_units, activation='relu', kernel_regularizer=reg),
        Dropout(dropout),
        Dense(NUM_CLASSES, activation='softmax'),
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def ensemble(models):
    """
    Mai multe modele antrenate identic, cu alt seed -> un singur model care face media
    probabilităților lor. Se salvează și se încarcă ca orice model (load_model), deci
    predicția live și evaluările rămân neschimbate.
    """
    inputs = Input(shape=models[0].input_shape[1:])
    for k, model in enumerate(models):
        model._name = f'membru_{k + 1}'   # nume unice în modelul combinat
    return Model(inputs, Average()([model(inputs) for model in models]), name='ansamblu')
