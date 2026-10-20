import cv2
import numpy as np
import os
import mediapipe as mp

# Inițializare MediaPipe
mp_holistic = mp.solutions.holistic

def extract_landmarks(results):
    """
    Extrage coordonatele X, Y, Z pentru mâini, față și corp.
    Dacă o mână nu este vizibilă în cadru, completează cu 0.
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, face, lh, rh])

RAW_DATA_PATH = os.path.join('data', 'raw_videos')
PROCESSED_DATA_PATH = os.path.join('data', 'processed_keypoints')

# Numărul fix de cadre pe care le procesăm din fiecare video (standardizare)
SEQUENCE_LENGTH = 30 

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Eroare: Folderul {RAW_DATA_PATH} nu există! Creează-l mai întâi.")
        exit()

    actions = [d for d in os.listdir(RAW_DATA_PATH) if os.path.isdir(os.path.join(RAW_DATA_PATH, d))]

    print(f"Am găsit {len(actions)} clase (cuvinte): {actions}")

    for action in actions:
        action_raw_path = os.path.join(RAW_DATA_PATH, action)
        action_processed_path = os.path.join(PROCESSED_DATA_PATH, action)
        os.makedirs(action_processed_path, exist_ok=True)

        video_files = [f for f in os.listdir(action_raw_path) if f.endswith(('.mp4', '.avi', '.mov'))]

        for video_idx, video_file in enumerate(video_files):
            video_path = os.path.join(action_raw_path, video_file)
            cap = cv2.VideoCapture(video_path)

            window_landmarks = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Conversie RGB și procesare MediaPipe
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)

                # Extragere vector de numere
                keypoints = extract_landmarks(results)
                window_landmarks.append(keypoints)

            cap.release()

            # Normalizăm secvența la fix 30 de cadre (prin tăiere sau interpolare simplă)
            if len(window_landmarks) > 0:
                if len(window_landmarks) >= SEQUENCE_LENGTH:
                    # Luăm uniform 30 de cadre
                    indices = np.linspace(0, len(window_landmarks) - 1, SEQUENCE_LENGTH, dtype=int)
                    final_sequence = [window_landmarks[i] for i in indices]
                else:
                    # Padding cu ultimul cadru dacă video-ul e prea scurt
                    final_sequence = window_landmarks
                    while len(final_sequence) < SEQUENCE_LENGTH:
                        final_sequence.append(window_landmarks[-1])

                # Salvare fișier numere
                save_path = os.path.join(action_processed_path, f"video_{video_idx}.npy")
                np.save(save_path, np.array(final_sequence))
                print(f"Procesat: {action} / {video_file} -> salvat în {save_path}")

print("Procesarea tuturor videoclipurilor s-a încheiat cu succes!")