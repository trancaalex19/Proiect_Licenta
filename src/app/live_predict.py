"""
Recunoaștere live de pe cameră, împreună cu ghidul de semne.

Folosește exact aceeași extragere (src/features/keypoints.py) și aceeași
preprocesare (src/features/preprocessing.py) ca la antrenare.

Implicit, semnele se decupează întregi (src/app/segmenter.py): ridici mâinile
peste linia desenată, faci semnul, cobori mâinile, iar modelul clasifică tot
semnul o singură dată. Așa primește același tip de exemplu ca la antrenare.
Cu --continuu se folosește vechiul mod: o predicție la fiecare cadru, pe o
fereastră glisantă de BUFFER_FRAMES cadre.

Rulare (din rădăcina proiectului, cu venv activat):
    python -m src.app.live_predict               # camera din config + ghidul de semne
    python -m src.app.live_predict --camera 0    # altă cameră (0 = laptop)
    python -m src.app.live_predict --no-guide    # fără ghid
    python -m src.app.live_predict --continuu    # fereastră glisantă (modul vechi)

Taste: q = ieșire; în ghid: n / p = alt cuvânt-țintă, v = alt semnatar,
       m = oglindește videoclipul, l = încetinit
"""
import argparse
import os
import time
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from src.app.camera import (make_holistic, open_camera, place_windows, process_frame,
                            resize_for_display, smooth_fps)
from src.app.overlay import draw_landmarks, draw_overlay, draw_segmenter
from src.app.segmenter import SignSegmenter
from src.config import ACTIONS, CAMERA_INDEX, MODEL_PATH
from src.features.keypoints import extract_keypoints
from src.ml.model import predict_segment

WINDOW_NAME = 'SignAI - Testare Live (Apasa Q pentru a iesi)'
BUFFER_FRAMES = 60    # doar în modul --continuu: ~2 secunde la 30 fps
THRESHOLD = 0.85      # în modul --continuu: cuvântul se afișează doar peste acest prag


def parse_args():
    parser = argparse.ArgumentParser(description='Recunoaștere live ASL')
    parser.add_argument('--camera', type=int, default=CAMERA_INDEX,
                        help=f'indexul camerei (implicit {CAMERA_INDEX}, din src/config.py)')
    parser.add_argument('--no-guide', action='store_true', help='nu deschide ghidul de semne')
    parser.add_argument('--continuu', action='store_true',
                        help='predicție la fiecare cadru pe o fereastră glisantă (modul vechi)')
    return parser.parse_args()


def main():
    args = parse_args()

    print("Încărcăm modelul neuronal. Te rog așteaptă...")
    model = load_model(MODEL_PATH)
    print("Model încărcat cu succes!")

    guide = None
    if not args.no_guide:
        from src.app.sign_guide import SignGuide
        guide = SignGuide(model=model)

    cap = open_camera(args.camera, " Încearcă --camera 0 sau verifică telefonul.")
    if cap is None:
        return
    mode = "fereastră glisantă" if args.continuu else "semne întregi"
    print(f"Folosim camera {args.camera}, mod: {mode}.")

    segmenter = SignSegmenter()
    buffer = []           # modul --continuu: ultimele BUFFER_FRAMES cadre brute
    current_word = ""     # cuvântul recunoscut curent
    probabilities = None  # ultima ieșire a modelului
    windows_placed = False
    fps, last_time = 0.0, time.perf_counter()

    with make_holistic() as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            now = time.perf_counter()
            fps = smooth_fps(fps, now - last_time)
            last_time = now

            results = process_frame(holistic, frame)
            image = frame.copy()
            draw_landmarks(image, results)  # feedback vizual: doar brațe și mâini
            keypoints = extract_keypoints(results)

            if args.continuu:
                buffer = (buffer + [keypoints])[-BUFFER_FRAMES:]
                hands_visible = results.left_hand_landmarks or results.right_hand_landmarks
                if len(buffer) == BUFFER_FRAMES and hands_visible:
                    probabilities = predict_segment(model, np.array(buffer))
                    if probabilities.max() > THRESHOLD:
                        current_word = ACTIONS[int(np.argmax(probabilities))]
            else:
                segment = segmenter.update(keypoints, now)
                if segment is not None:  # semn încheiat: o singură predicție pe tot semnul
                    probabilities = predict_segment(model, segment)
                    current_word = ACTIONS[int(np.argmax(probabilities))]

            image = resize_for_display(image)
            if not args.continuu:
                draw_segmenter(image, segmenter)
            draw_overlay(image, current_word, probabilities, results, fps,
                         threshold=THRESHOLD if args.continuu else 0.5)
            cv2.imshow(WINDOW_NAME, image)
            if guide is not None:
                guide.show(prediction=current_word)

            if not windows_placed:  # camera în stânga, ghidul în dreapta
                place_windows(WINDOW_NAME, image, guide)
                windows_placed = True

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if guide is not None and guide.handle_key(key):
                # cuvânt-țintă nou: pornim de la zero, ca predicția veche să nu apară drept „CORECT”
                buffer, current_word, probabilities = [], "", None
                segmenter.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
