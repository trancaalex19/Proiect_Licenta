"""
Test live ghidat: măsoară acuratețea reală la utilizare, în aceleași condiții
ca src.app.live_predict (cameră, MediaPipe, semne întregi decupate automat).

Programul cere pe rând fiecare cuvânt, în ordine amestecată, de --rounds ori.
Pentru fiecare încercare decupează semnul, îl clasifică și salvează segmentul
brut complet în data/live_test/<cuvânt>/<sesiune>_<k>.npy. La final afișează
acuratețea și confuziile. Datele salvate NU intră niciodată în antrenare; orice
model nou se reevaluează pe ele cu:  python -m src.ml.evaluate_live

Rulare (din rădăcina proiectului, cu venv activat):
    python -m src.app.live_test                 # 3 runde x 15 cuvinte = 45 de încercări
    python -m src.app.live_test --rounds 2
    python -m src.app.live_test --camera 0 --no-guide

Taste: SPAȚIU = start / pauză, r = refă ultima încercare (doar dacă semnul a fost
decupat greșit, nu pentru că predicția a fost greșită), q = ieșire cu raport;
în ghid: v = alt semnatar, m = oglindește videoclipul, l = încetinit.
"""
import argparse
import os
import random
import time
from datetime import datetime
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

from src.app.live_predict import (draw_landmarks, draw_overlay, draw_segmenter, process_frame,
                                  resize_for_display)
from src.app.segmenter import SignSegmenter
from src.config import ACTIONS, CAMERA_INDEX, LIVE_TEST_DIR, MODEL_PATH
from src.features.keypoints import extract_keypoints
from src.ml.evaluate_live import LABEL_MAP, predict_segment, print_report

WINDOW_NAME = 'Test live (SPATIU = start/pauza, r = refa ultima, q = iesire)'
FEEDBACK_SECONDS = 1.5   # cât rămâne rezultatul pe ecran înainte de următorul cuvânt


def parse_args():
    parser = argparse.ArgumentParser(description='Test live ghidat')
    parser.add_argument('--rounds', type=int, default=3, help='de câte ori se cere fiecare cuvânt')
    parser.add_argument('--camera', type=int, default=CAMERA_INDEX)
    parser.add_argument('--no-guide', action='store_true', help='fără ghidul de semne')
    return parser.parse_args()


def make_plan(rounds):
    """Fiecare rundă conține toate cuvintele o dată, în ordine amestecată."""
    plan = []
    for _ in range(rounds):
        words = ACTIONS.copy()
        random.shuffle(words)
        plan += words
    return plan


def draw_task(image, word, index, total, state, last_result):
    h, w = image.shape[:2]
    cv2.rectangle(image, (0, 40), (w, 80), (40, 40, 40), -1)
    if state == 'paused':
        text = "PAUZA - apasa SPATIU"
    else:
        text = f"Fa semnul: {word.upper()}   ({index + 1}/{total})"
    cv2.putText(image, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    if state == 'feedback' and last_result:
        target, predicted, conf = last_result
        ok = target == predicted
        color = (60, 200, 90) if ok else (60, 60, 230)
        label = "CORECT" if ok else f"GRESIT (ai facut {target})"
        cv2.putText(image, f"{label}: {predicted} {conf:.0%}", (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, color, 3, cv2.LINE_AA)


def main():
    args = parse_args()
    model = load_model(MODEL_PATH)
    guide = None
    if not args.no_guide:
        from src.app.sign_guide import SignGuide
        guide = SignGuide(model=model)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"EROARE: Nu pot deschide camera {args.camera}.")
        return

    session = datetime.now().strftime('%Y%m%d_%H%M%S')
    plan = make_plan(args.rounds)
    index, state, state_start = 0, 'paused', time.perf_counter()
    saved = []          # (cale, etichetă, probabilități) pentru fiecare încercare
    redos = 0
    last_result, probabilities, current_word = None, None, ""
    segmenter = SignSegmenter()
    windows_placed = False
    fps, last_time = 0.0, time.perf_counter()
    if guide is not None:
        guide.set_target(plan[0])
    print(f"Sesiunea {session}: {len(plan)} încercări. Apasă SPAȚIU în fereastra camerei.")

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened() and index < len(plan):
            ret, frame = cap.read()
            if not ret:
                break
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last_time, 1e-6)
            last_time = now
            word = plan[index]

            results = process_frame(holistic, frame)
            image = frame.copy()
            draw_landmarks(image, results)
            keypoints = extract_keypoints(results)

            if state == 'feedback' and now - state_start >= FEEDBACK_SECONDS:
                index += 1
                state = 'waiting'
                segmenter.reset()
                if index < len(plan) and guide is not None:
                    guide.set_target(plan[index])
                continue
            if state == 'waiting':
                segment = segmenter.update(keypoints, now)
                if segment is not None:
                    path = os.path.join(LIVE_TEST_DIR, word, f"{session}_{index}.npy")
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    np.save(path, segment)
                    probabilities = predict_segment(model, segment)
                    predicted = ACTIONS[int(np.argmax(probabilities))]
                    current_word = predicted
                    saved.append((path, LABEL_MAP[word], probabilities))
                    last_result = (word, predicted, probabilities.max())
                    state, state_start = 'feedback', now

            image = resize_for_display(image)
            if state == 'waiting':
                draw_segmenter(image, segmenter)
            draw_overlay(image, current_word, probabilities, results, fps, threshold=0.5)
            draw_task(image, word, index, len(plan), state, last_result)
            cv2.imshow(WINDOW_NAME, image)
            if guide is not None:
                correct = sum(int(np.argmax(p)) == y for _, y, p in saved)
                guide.show(message=f"Corecte până acum: {correct}/{len(saved)}")

            if not windows_placed:
                cv2.moveWindow(WINDOW_NAME, 0, 0)
                if guide is not None:
                    from src.app.sign_guide import WINDOW_NAME as GUIDE_WINDOW
                    cv2.moveWindow(GUIDE_WINDOW, image.shape[1] + 10, 0)
                windows_placed = True

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                state = 'waiting' if state == 'paused' else 'paused'
                segmenter.reset()
            elif key == ord('r') and saved and state != 'feedback':
                # Refacem ultima încercare salvată: o ștergem și o cerem din nou
                path, _, _ = saved.pop()
                os.remove(path)
                index, redos = index - 1, redos + 1
                state, last_result, probabilities, current_word = 'paused', None, None, ""
                segmenter.reset()
                if guide is not None:
                    guide.set_target(plan[index])
            elif guide is not None and key in (ord('v'), ord('m'), ord('l')):
                guide.handle_key(key)  # alt semnatar / oglindă / încetinit (cuvântul rămâne cel cerut)

    cap.release()
    cv2.destroyAllWindows()

    if not saved:
        print("Nicio încercare salvată.")
        return
    print(f"\nSesiunea {session}: {len(saved)} încercări salvate în {LIVE_TEST_DIR} "
          f"(refăcute: {redos})")
    print_report([y for _, y, _ in saved], [p for _, _, p in saved])
    print("\nReevaluare cu alt model, pe aceleași încercări: python -m src.ml.evaluate_live")


if __name__ == '__main__':
    main()
