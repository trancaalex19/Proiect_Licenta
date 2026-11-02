"""
Extragerea vectorului brut de keypoints dintr-un rezultat MediaPipe Holistic.
Folosit atât la extragerea din videoclipuri, cât și la predicția live,
ca formatul să fie garantat identic.
"""
import numpy as np

from src.config import SEQUENCE_LENGTH


def extract_keypoints(results):
    """
    Returnează un vector de 1662 de valori: pose (33x4: x, y, z, visibility),
    față (468x3), mâna stângă (21x3), mâna dreaptă (21x3).
    Părțile nedetectate sunt completate cu 0.
    """
    pose = np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])


def to_fixed_length(frames, length=SEQUENCE_LENGTH):
    """
    Standardizează o listă de cadre (vectori de 1662) la `length` cadre:
    eșantionare uniformă dacă sunt mai multe, padding cu ultimul cadru dacă sunt mai puține.
    Folosit identic pentru videoclipurile WLASL și pentru înregistrările proprii.
    """
    if len(frames) >= length:
        indices = np.linspace(0, len(frames) - 1, length, dtype=int)
        frames = [frames[i] for i in indices]
    else:
        frames = list(frames) + [frames[-1]] * (length - len(frames))
    return np.array(frames)
