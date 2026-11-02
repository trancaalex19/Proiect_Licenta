"""
Recunoaștere live de pe cameră, împreună cu ghidul de semne.

Folosește exact aceeași extragere (src/features/keypoints.py) și aceeași
preprocesare (src/features/preprocessing.py) ca la antrenare.

Rulare (din rădăcina proiectului, cu venv activat):
    python -m src.app.live_predict               # camera din config + ghidul de semne
    python -m src.app.live_predict --camera 0    # altă cameră (0 = laptop)
    python -m src.app.live_predict --no-guide    # fără ghid

Taste: q = ieșire; în ghid: n / p = alt cuvânt-țintă, m = oglindește animația
"""
import argparse
import os
import time
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

from src.config import ACTIONS, CAMERA_INDEX, MODEL_PATH
from src.features.keypoints import extract_keypoints
from src.features.preprocessing import preprocess

WINDOW_NAME = 'SignAI - Testare Live (Apasa Q pentru a iesi)'
BUFFER_FRAMES = 60    # ~2 secunde la 30 fps; la antrenare un exemplu = tot semnul comprimat în 30 de cadre
THRESHOLD = 0.85      # modelul trebuie să fie 85% sigur pentru a afișa cuvântul
DISPLAY_WIDTH = 800   # imaginea camerei e redimensionată la afișare (camera telefonului are rezoluție mare)

# Desenăm doar ce folosește modelul: brațele (umeri, coate, încheieturi) și mâinile.
# Punctele feței din pose (0-10) nu mai sunt folosite de model, deci nu le mai afișăm.
ARM_CONNECTIONS = frozenset((a, b) for a, b in mp.solutions.holistic.POSE_CONNECTIONS
                            if 11 <= a <= 22 and 11 <= b <= 22)
OK_COLOR, LOST_COLOR = (60, 200, 90), (60, 60, 230)


def parse_args():
    parser = argparse.ArgumentParser(description='Recunoaștere live ASL')
    parser.add_argument('--camera', type=int, default=CAMERA_INDEX,
                        help=f'indexul camerei (implicit {CAMERA_INDEX}, din src/config.py)')
    parser.add_argument('--no-guide', action='store_true', help='nu deschide ghidul de semne')
    return parser.parse_args()


def draw_landmarks(image, results):
    """Brațele și mâinile, fără punctele feței."""
    if results.pose_landmarks:
        h, w = image.shape[:2]
        lm = results.pose_landmarks.landmark
        pts = {i: (int(lm[i].x * w), int(lm[i].y * h)) for i in range(11, 23)}
        for a, b in ARM_CONNECTIONS:
            cv2.line(image, pts[a], pts[b], (200, 200, 200), 2, cv2.LINE_AA)
        for p in pts.values():
            cv2.circle(image, p, 4, (245, 117, 16), -1, cv2.LINE_AA)
    for hand in (results.left_hand_landmarks, results.right_hand_landmarks):
        mp.solutions.drawing_utils.draw_landmarks(image, hand, mp.solutions.holistic.HAND_CONNECTIONS)


def draw_overlay(image, current_word, probabilities, results, fps):
    """Cuvântul recunoscut, starea mâinilor, FPS și primele 3 predicții."""
    width = image.shape[1]
    cv2.rectangle(image, (0, 0), (width, 40), (245, 117, 16), -1)
    cv2.putText(image, f"Predictie: {current_word}", (5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Starea detecției (stânga/dreapta = ale persoanei) și FPS, în colțul din dreapta jos
    h = image.shape[0]
    for k, (label, hand) in enumerate((("mana stanga", results.left_hand_landmarks),
                                       ("mana dreapta", results.right_hand_landmarks))):
        color = OK_COLOR if hand else LOST_COLOR
        cv2.putText(image, f"{label}: {'OK' if hand else 'pierduta'}", (width - 260, h - 45 + 25 * k),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    fps_color = OK_COLOR if fps >= 15 else LOST_COLOR
    cv2.putText(image, f"FPS: {fps:.0f}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2, cv2.LINE_AA)

    if probabilities is None:
        return
    for rank, i in enumerate(np.argsort(probabilities)[::-1][:3]):
        y = 70 + rank * 30
        p = probabilities[i]
        color = (60, 200, 90) if p > THRESHOLD else (255, 255, 255)
        cv2.rectangle(image, (5, y - 18), (5 + int(200 * p), y + 4), (80, 80, 80), -1)
        cv2.putText(image, f"{ACTIONS[i]} {p:.0%}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def main():
    args = parse_args()

    print("Încărcăm modelul neuronal. Te rog așteaptă...")
    model = load_model(MODEL_PATH)
    print("Model încărcat cu succes!")

    guide = None
    if not args.no_guide:
        from src.app.sign_guide import SignGuide
        guide = SignGuide(model=model)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"EROARE: Nu pot deschide camera {args.camera}. Încearcă --camera 0 sau verifică telefonul.")
        return
    print(f"Folosim camera {args.camera}.")

    buffer = []           # ultimele BUFFER_FRAMES cadre brute
    current_word = ""     # cuvântul recunoscut curent (peste prag)
    probabilities = None  # ultima ieșire a modelului
    windows_placed = False
    fps, last_time = 0.0, time.perf_counter()

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last_time, 1e-6)  # medie glisantă
            last_time = now

            # Transformăm imaginea pentru MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            draw_landmarks(image, results)  # feedback vizual: doar brațe și mâini

            # Logica de predicție
            buffer.append(extract_keypoints(results))
            buffer = buffer[-BUFFER_FRAMES:]
            hands_visible = results.left_hand_landmarks or results.right_hand_landmarks

            if len(buffer) == BUFFER_FRAMES and hands_visible:
                features = preprocess(np.array(buffer))  # (30, 112), ca la antrenare
                probabilities = model.predict(features[np.newaxis], verbose=0)[0]
                if probabilities.max() > THRESHOLD:
                    current_word = ACTIONS[int(np.argmax(probabilities))]

            # Afișare (redimensionată pentru camere cu rezoluție mare)
            if image.shape[1] > DISPLAY_WIDTH:
                scale = DISPLAY_WIDTH / image.shape[1]
                image = cv2.resize(image, (DISPLAY_WIDTH, int(image.shape[0] * scale)))
            draw_overlay(image, current_word, probabilities, results, fps)
            cv2.imshow(WINDOW_NAME, image)
            if guide is not None:
                guide.show(prediction=current_word)

            if not windows_placed:  # camera în stânga, ghidul în dreapta
                cv2.moveWindow(WINDOW_NAME, 0, 0)
                if guide is not None:
                    from src.app.sign_guide import WINDOW_NAME as GUIDE_WINDOW
                    cv2.moveWindow(GUIDE_WINDOW, image.shape[1] + 10, 0)
                windows_placed = True

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            if guide is not None and guide.handle_key(key):
                # cuvânt-țintă nou: pornim de la zero, ca predicția veche să nu apară drept „CORECT”
                buffer, current_word, probabilities = [], "", None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
