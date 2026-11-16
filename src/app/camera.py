"""
Utilitare comune pentru programele care folosesc camera (live_predict, live_test,
record_samples): deschiderea camerei, MediaPipe Holistic, redimensionarea imaginii
pentru afișare și poziționarea ferestrelor.
"""
import cv2
import mediapipe as mp

DISPLAY_WIDTH = 800   # imaginea camerei e redimensionată la afișare (camera telefonului are rezoluție mare)


def open_camera(index, hint=""):
    """Deschide camera `index`. Returnează None (și afișează eroarea) dacă nu se poate deschide."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"EROARE: Nu pot deschide camera {index}.{hint}")
        return None
    return cap


def make_holistic():
    """MediaPipe Holistic cu aceleași praguri peste tot (extragere, înregistrare, live)."""
    return mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def process_frame(holistic, frame):
    """Rulează MediaPipe pe un cadru BGR și returnează rezultatele."""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    return holistic.process(image)


def resize_for_display(image):
    if image.shape[1] > DISPLAY_WIDTH:
        scale = DISPLAY_WIDTH / image.shape[1]
        image = cv2.resize(image, (DISPLAY_WIDTH, int(image.shape[0] * scale)))
    return image


def smooth_fps(fps, elapsed):
    """Medie glisantă a FPS: `elapsed` = secundele de la cadrul precedent."""
    return 0.9 * fps + 0.1 / max(elapsed, 1e-6)


def place_windows(camera_window, image, guide=None):
    """Camera în stânga, ghidul (dacă există) în dreapta ei."""
    cv2.moveWindow(camera_window, 0, 0)
    if guide is not None:
        from src.app.sign_guide import WINDOW_NAME as GUIDE_WINDOW
        cv2.moveWindow(GUIDE_WINDOW, image.shape[1] + 10, 0)
