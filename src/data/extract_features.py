"""
Extrage keypoints MediaPipe Holistic din videoclipurile organizate și salvează
câte o secvență brută (SEQUENCE_LENGTH x 1662) per video.

    data/raw_videos/<cuvânt>/*.mp4  ->  data/processed_keypoints/<cuvânt>/video_<i>.npy

Clasele care au deja .npy-uri NU se reextrag: numele video_<i> depind de ordinea
alfabetică a mp4-urilor rămase, deci după ștergerea unor videoclipuri o reextragere
ar schimba numele și ar invalida data/splits.json. Pentru a reextrage o clasă, șterge-i
folderul din data/processed_keypoints/ (și intrările ei din splits.json).

Rulare (din rădăcina proiectului):
    python -m src.data.extract_features
"""
import json
import os

import cv2
import mediapipe as mp
import numpy as np

from src.config import KEYPOINTS_DIR, RAW_VIDEOS_DIR, SEQUENCE_LENGTH, VIDEO_INDEX_FILE
from src.features.keypoints import extract_keypoints, to_fixed_length


def video_to_sequence(video_path, holistic):
    """Citește un video și returnează secvența de keypoints standardizată la SEQUENCE_LENGTH cadre."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        frames.append(extract_keypoints(holistic.process(image)))
    cap.release()

    if not frames:
        return None
    return to_fixed_length(frames, SEQUENCE_LENGTH)


def main():
    if not os.path.exists(RAW_VIDEOS_DIR):
        print(f"Eroare: Folderul {RAW_VIDEOS_DIR} nu există! Rulează mai întâi organize_videos.")
        return

    actions = [d for d in os.listdir(RAW_VIDEOS_DIR) if os.path.isdir(os.path.join(RAW_VIDEOS_DIR, d))]
    print(f"Am găsit {len(actions)} clase (cuvinte): {actions}")

    # cuvânt -> {video_<i>.npy: fișierul video sursă}; intrările claselor existente se păstrează
    video_index = {}
    if os.path.exists(VIDEO_INDEX_FILE):
        with open(VIDEO_INDEX_FILE, 'r', encoding='utf-8') as f:
            video_index = json.load(f)

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            action_raw_path = os.path.join(RAW_VIDEOS_DIR, action)
            action_out_path = os.path.join(KEYPOINTS_DIR, action)
            if os.path.isdir(action_out_path) and any(f.endswith('.npy') for f in os.listdir(action_out_path)):
                print(f"Sărit (deja extras): {action}")
                continue
            os.makedirs(action_out_path, exist_ok=True)

            video_files = sorted(f for f in os.listdir(action_raw_path) if f.endswith(('.mp4', '.avi', '.mov')))
            video_index[action] = {f"video_{i}.npy": f for i, f in enumerate(video_files)}
            for video_idx, video_file in enumerate(video_files):
                sequence = video_to_sequence(os.path.join(action_raw_path, video_file), holistic)
                if sequence is None:
                    print(f"Sărit (fără cadre): {action} / {video_file}")
                    continue
                save_path = os.path.join(action_out_path, f"video_{video_idx}.npy")
                np.save(save_path, sequence)
                print(f"Procesat: {action} / {video_file} -> salvat în {save_path}")

    # Corespondența .npy -> mp4 (numele video_<i> depind de ordinea alfabetică a mp4-urilor)
    with open(VIDEO_INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(video_index, f, indent=2, ensure_ascii=False)
    print(f"Corespondența .npy -> video a fost salvată în {VIDEO_INDEX_FILE}")
    print("Procesarea tuturor videoclipurilor s-a încheiat cu succes!")


if __name__ == '__main__':
    main()
