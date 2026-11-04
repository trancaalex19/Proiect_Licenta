"""
Decupează semne întregi din fluxul camerei, ca la predicția live modelul să
primească același tip de exemplu ca la antrenare: un semn complet, de la
ridicarea mâinilor până la coborârea lor (nu o fereastră glisantă care prinde
bucăți de semn, tranziții sau repaus).

Regula (în timp real, nu în număr de cadre, ca să nu depindă de FPS):
  - repaus -> semn: o mână detectată are încheietura peste nivelul START_LEVEL
    timp de START_FRAMES cadre consecutive;
  - semn -> repaus: nicio mână nu mai e peste REST_LEVEL (sau nu se vede nicio
    mână) timp de END_SECONDS; segmentul se trimite la clasificare.
Înălțimea se măsoară în lățimi de umăr sub linia umerilor (+ = mai jos), deci nu
depinde de distanța față de cameră. Pragurile vin din înregistrările proprii:
în repaus încheietura stă la ~0.9, în timpul semnului în jur de 0.
Între START_LEVEL și REST_LEVEL e o zonă de histerezis, ca semnul să nu se
rupă când mâna coboară puțin.
"""
import time
from collections import deque

import numpy as np

# Indici în vectorul brut (vezi src/features/keypoints.py)
_POSE_END = 33 * 4
_FACE_END = _POSE_END + 468 * 3
_LH_END = _FACE_END + 21 * 3
_L_SHOULDER, _R_SHOULDER = 11, 12

START_LEVEL = 0.6       # mâna peste acest nivel = începe semnul
REST_LEVEL = 0.8        # mâinile sub acest nivel (sau nevăzute) = repaus
START_FRAMES = 2        # cadre consecutive cu mâna ridicată pentru a porni
END_SECONDS = 0.35      # cât timp trebuie să stea mâinile jos pentru a încheia semnul
MIN_SECONDS = 0.4       # segmentele mai scurte (mișcări accidentale) se ignoră
MAX_SECONDS = 6.0       # semnul se încheie forțat după atât (cel mai lung, „student”, ~3 s)
PRE_ROLL_SECONDS = 0.25   # cadre păstrate dinaintea pornirii (mâna care urcă)
POST_ROLL_SECONDS = 0.15  # cadre păstrate după ultima mână ridicată (mâna care coboară)
SMOOTHING = 0.2         # netezirea referinței umerilor (medie exponențială)


class SignSegmenter:
    def __init__(self):
        self.reference = None   # (y umeri, lățime umeri), netezite în timp
        self.reset()

    def reset(self):
        """Uită semnul în curs (referința umerilor se păstrează)."""
        self.state = 'repaus'
        self.frames = []                 # (timp, cadru brut) ale semnului în curs
        self.pre_roll = deque()          # ultimele cadre din repaus
        self._raised_frames = 0
        self._start_time = None
        self._last_active = None
        self._need_rest = False          # după o oprire forțată: întâi mâinile jos

    # ------------------------------------------------------------------
    def _update_reference(self, raw):
        pose = raw[:_POSE_END].reshape(33, 4)
        if not np.any(pose):
            return
        shoulders = pose[[_L_SHOULDER, _R_SHOULDER], :2]
        y, width = shoulders[:, 1].mean(), np.linalg.norm(shoulders[0] - shoulders[1])
        if width < 1e-6:
            return
        if self.reference is None:
            self.reference = np.array([y, width])
        else:
            self.reference += SMOOTHING * (np.array([y, width]) - self.reference)

    def hand_height(self, raw):
        """Cea mai ridicată încheietură detectată, în lățimi de umăr sub umeri; None dacă nu se vede."""
        if self.reference is None:
            return None
        wrists = [raw[start + 1] for start in (_FACE_END, _LH_END) if np.any(raw[start:start + 63])]
        if not wrists:
            return None
        y, width = self.reference
        return (min(wrists) - y) / width

    def line_y(self, level=START_LEVEL):
        """Poziția verticală (0-1 în imagine) a unui nivel, pentru desenarea liniei-ghid."""
        if self.reference is None:
            return None
        y, width = self.reference
        return y + level * width

    @property
    def elapsed(self):
        """Durata semnului în curs (secunde), 0 în repaus."""
        return 0.0 if self._start_time is None else time.perf_counter() - self._start_time

    # ------------------------------------------------------------------
    def update(self, raw, now=None):
        """
        Primește un cadru brut (1662). Returnează segmentul complet (T, 1662) în
        cadrul în care semnul s-a încheiat, altfel None.
        """
        now = time.perf_counter() if now is None else now
        raw = np.asarray(raw, dtype=np.float32)
        self._update_reference(raw)
        height = self.hand_height(raw)
        raised = height is not None and height < START_LEVEL
        resting = height is None or height > REST_LEVEL

        if self.state == 'repaus':
            if self._need_rest:
                self._need_rest = not resting
            self.pre_roll.append((now, raw))
            while self.pre_roll and now - self.pre_roll[0][0] > PRE_ROLL_SECONDS:
                self.pre_roll.popleft()
            self._raised_frames = self._raised_frames + 1 if raised and not self._need_rest else 0
            if self._raised_frames >= START_FRAMES:
                self.state = 'semn'
                self.frames = list(self.pre_roll)
                self._start_time = self._last_active = now
            return None

        # În timpul semnului
        self.frames.append((now, raw))
        if not resting:
            self._last_active = now
        ended = now - self._last_active >= END_SECONDS
        forced = now - self._start_time >= MAX_SECONDS
        if not (ended or forced):
            return None

        segment = [r for t, r in self.frames if t <= self._last_active + POST_ROLL_SECONDS]
        duration = self._last_active - self._start_time
        self.reset()
        self._need_rest = forced
        if duration < MIN_SECONDS:
            return None
        return np.array(segment)
