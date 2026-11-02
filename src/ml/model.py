"""Arhitectura rețelei, separată de scriptul de antrenare."""
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from src.config import NUM_CLASSES, SEQUENCE_LENGTH
from src.features.preprocessing import FEATURE_DIM


def build_model(units=64, dense_units=64, dropout=0.4, l2=1e-4, learning_rate=1e-3):
    """LSTM mic, potrivit pentru ~100 de videoclipuri de antrenare."""
    reg = regularizers.l2(l2)
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, FEATURE_DIM)),
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
