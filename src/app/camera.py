"""
Utilitare comune pentru programele care folosesc camera (live_predict, live_test,
record_samples): deschiderea camerei, MediaPipe Holistic, redimensionarea imaginii
pentru afișare și poziționarea ferestrelor.

Camera poate fi un index (0, 1, ...) sau adresa unui flux video din rețea, de exemplu
telefonul ca cameră Wi-Fi: --camera http://192.168.1.23:8080/video (aplicația „IP Webcam”
pe Android) sau http://<ip>:4747/video (DroidCam). Telefonul și calculatorul / Pi-ul trebuie
să fie în aceeași rețea.
"""
import threading
import time

import cv2
import mediapipe as mp

from src.config import CAMERA_FRAME_SIZE, HOLISTIC_MODEL_COMPLEXITY

DISPLAY_WIDTH = 800   # imaginea camerei e redimensionată la afișare (camera telefonului are rezoluție mare)


STREAM_TIMEOUT_SECONDS = 5.0   # fluxul din rețea: fără imagine nouă atâta timp = camera s-a oprit


def camera_source(value):
    """Argumentul --camera: un număr = indexul camerei, altfel adresa unui flux (http://..., rtsp://...)."""
    return int(value) if str(value).isdigit() else value


class StreamCapture:
    """
    Flux video din rețea (telefonul ca cameră Wi-Fi), cu aceleași metode ca cv2.VideoCapture.
    Un fir citește continuu și păstrează doar ultima imagine: dacă programul procesează mai încet
    decât trimite telefonul (ex. pe Pi), imaginile vechi se aruncă în loc să se adune, deci
    camera nu rămâne în urmă. read() așteaptă o imagine nouă (nu o întoarce pe aceeași de două ori).
    """

    def __init__(self, url):
        self._cap = cv2.VideoCapture(url)
        self._frame, self._frame_id, self._returned_id = None, 0, 0
        self._ready = threading.Condition()
        self._running = self._cap.isOpened()
        if self._running:
            threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self._running:
            ok, frame = self._cap.read()
            with self._ready:
                if not ok:
                    self._running = False
                else:
                    self._frame, self._frame_id = frame, self._frame_id + 1
                self._ready.notify_all()

    def isOpened(self):
        return self._running

    def read(self):
        with self._ready:
            self._ready.wait_for(lambda: self._frame_id != self._returned_id or not self._running,
                                 timeout=STREAM_TIMEOUT_SECONDS)
            if self._frame_id == self._returned_id:
                return False, None
            self._returned_id = self._frame_id
            return True, self._frame

    def release(self):
        self._running = False
        time.sleep(0.05)
        self._cap.release()


def open_camera(index, hint=""):
    """Deschide camera `index` (număr sau adresă de flux). Returnează None (și afișează eroarea) dacă nu se poate deschide."""
    if isinstance(index, str):
        cap = StreamCapture(index)
        if not cap.isOpened():
            print(f"EROARE: Nu pot deschide fluxul {index}. Telefonul e în aceeași rețea și aplicația pornită?")
            return None
        return cap
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"EROARE: Nu pot deschide camera {index}.{hint}")
        return None
    if CAMERA_FRAME_SIZE is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_SIZE[1])
    return cap


def make_holistic():
    """MediaPipe Holistic cu aceleași praguri peste tot (extragere, înregistrare, live)."""
    return mp.solutions.holistic.Holistic(model_complexity=HOLISTIC_MODEL_COMPLEXITY,
                                          min_detection_confidence=0.5, min_tracking_confidence=0.5)


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
