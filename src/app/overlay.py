"""
Desenare peste imaginea camerei, comună pentru live_predict și live_test:
brațele și mâinile, linia de pornire a segmentării și panoul cu predicția.
"""
import cv2
import mediapipe as mp
import numpy as np

from src.app.segmenter import START_LEVEL
from src.config import ACTIONS

# Desenăm doar ce folosește modelul: brațele (umeri, coate, încheieturi) și mâinile.
# Punctele feței din pose (0-10) nu mai sunt folosite de model, deci nu le mai afișăm.
ARM_CONNECTIONS = frozenset((a, b) for a, b in mp.solutions.holistic.POSE_CONNECTIONS
                            if 11 <= a <= 22 and 11 <= b <= 22)
OK_COLOR, LOST_COLOR = (60, 200, 90), (60, 60, 230)
SIGN_COLOR = (0, 0, 255)


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


def draw_segmenter(image, segmenter, hint="Ridica mainile peste linie, fa semnul, apoi coboara-le"):
    """Linia de pornire și starea segmentării (repaus / semn în curs)."""
    h, w = image.shape[:2]
    line = segmenter.line_y(START_LEVEL)
    signing = segmenter.state == 'semn'
    if line is not None and 0 < line < 1:
        y = int(line * h)
        color = SIGN_COLOR if signing else (255, 255, 255)
        for x in range(0, w, 30):  # linie punctată
            cv2.line(image, (x, y), (min(x + 15, w), y), color, 2, cv2.LINE_AA)
    if signing:
        cv2.rectangle(image, (0, 0), (w - 1, h - 1), SIGN_COLOR, 6)
        cv2.putText(image, f"Semn in curs... {segmenter.elapsed:.1f}s", (10, h - 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, SIGN_COLOR, 2, cv2.LINE_AA)
    else:
        cv2.putText(image, hint, (10, h - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def draw_overlay(image, current_word, probabilities, results, fps, threshold):
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
        color = (60, 200, 90) if p > threshold else (255, 255, 255)
        cv2.rectangle(image, (5, y - 18), (5 + int(200 * p), y + 4), (80, 80, 80), -1)
        cv2.putText(image, f"{ACTIONS[i]} {p:.0%}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
