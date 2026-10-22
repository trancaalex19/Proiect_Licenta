import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 1. Configurările de bază
DATA_PATH = os.path.join('data', 'processed_keypoints')
ACTIONS = np.array([
    'hello', 'thankyou', 'yes', 'no', 'help', 
    'please', 'sorry', 'teacher', 'student', 'book', 
    'computer', 'learn', 'need', 'good', 'stop'
])
SEQUENCE_LENGTH = 30 

# 2. Încărcarea datelor (va prelua inclusiv datele augmentate)
print("Încărcăm datele (inclusiv cele augmentate)...")
label_map = {label:num for num, label in enumerate(ACTIONS)}

sequences, labels = [], []

for action in ACTIONS:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        print(f"Avertisment: Folderul pentru '{action}' nu există!")
        continue
        
    for file in os.listdir(action_path):
        if file.endswith('.npy'):
            res = np.load(os.path.join(action_path, file))
            sequences.append(res)
            labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

print(f"Forma datelor X: {X.shape} (video-uri, cadre, coordonate)")
print(f"Forma etichetelor Y: {y.shape}")

# 3. Împărțirea în date de Antrenament (80%) și Testare (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Arhitectura Rețelei Neuronale (Optimizată cu Batch Normalization)
model = Sequential()

model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(ACTIONS.shape[0], activation='softmax')) 

# Compilarea modelului cu Learning Rate scăzut
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# 5. Callbacks (Sisteme de siguranță la antrenament)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
checkpoint = ModelCheckpoint('models/sign_language_model.h5', monitor='val_categorical_accuracy', save_best_only=True)

os.makedirs('models', exist_ok=True)

# 6. Antrenarea propriu-zisă
print("Începem antrenarea modelului...")
history = model.fit(
    X_train, y_train, 
    epochs=150, 
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

print("\nAntrenamentul a fost finalizat! Cel mai bun model a fost salvat în folderul 'models'.")