"""
Ghid de semne: arată cuvântul-țintă, ce înseamnă, un videoclip real din setul
de date (WLASL) cu un semnatar care face semnul și, sub el, ce face fiecare mână.

Videoclipul pornește de când semnatarul ridică mâinile (timpul mort de la
început și sfârșit e tăiat) și se repetă în buclă. Pe încheieturile semnatarului
apar marcajele D (mâna dreaptă) și S (mâna stângă), cu aceleași culori ca
textul de sub videoclip. Implicit e afișat oglindit, ca să poată fi copiat ca
într-o oglindă. Exemplul afișat pentru
fiecare cuvânt este cel pe care modelul antrenat îl recunoaște cu cea mai mare
încredere (cel mai „tipic”); celelalte videoclipuri sunt disponibile ca variante
(tasta v), pentru că multe semne ASL au variante.

Pornește automat împreună cu src.app.live_predict, unde afișează și dacă
modelul a recunoscut semnul tău („CORECT”). Poate rula și singur:
    python -m src.app.sign_guide

Taste: n / p = cuvântul următor / anterior, v = alt semnatar (variantă),
       m = oglindește (ca într-o oglindă), l = încetinit, q = ieșire
"""
import json
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.app.sign_info import SIGN_INFO
from src.config import (ACTIONS, KEYPOINTS_DIR, MODEL_PATH, NUM_CLASSES, RAW_VIDEOS_DIR, SPLITS_FILE,
                        VIDEO_INDEX_FILE)
from src.features.preprocessing import fill_gaps
from src.ml.dataset import load_example, mirrored_files
from src.ml.model import fit_inputs

WINDOW_NAME = 'Ghid semne (n/p = cuvant, v = varianta, m = oglinda, l = lent)'
WIDTH, HEIGHT = 560, 880
PANEL_X, PANEL_Y, PANEL_W, PANEL_H = 10, 120, WIDTH - 20, 360   # zona videoclipului
SLOW_SPEED = 0.5      # viteza în modul încetinit (tasta l)
HOLD_SECONDS = 0.8    # pauză pe ultimul cadru, înainte de reluare
MARGIN_SECONDS = 0.3  # cât se păstrează înainte / după ce mâinile sunt vizibile

# Culori BGR (OpenCV)
BG, PANEL_BG = (30, 30, 30), (48, 48, 48)
RIGHT_HAND_BGR, LEFT_HAND_BGR = (16, 117, 245), (230, 160, 40)
# Culori RGB (PIL)
WHITE, GRAY, ORANGE, GREEN = (255, 255, 255), (170, 170, 170), (245, 117, 16), (60, 200, 90)
RIGHT_HAND_RGB, LEFT_HAND_RGB = RIGHT_HAND_BGR[::-1], LEFT_HAND_BGR[::-1]

# În vectorul brut de 1662 (vezi src/features/keypoints.py), mâinile sunt la final
_HANDS_START = 33 * 4 + 468 * 3
_LH_START, _RH_START = _HANDS_START, _HANDS_START + 21 * 3
# Marcajele de pe videoclip: (literă, unde începe mâna în vectorul brut, culoare)
HAND_MARKERS = (('D', _RH_START, RIGHT_HAND_BGR), ('S', _LH_START, LEFT_HAND_BGR))


# ---------------------------------------------------------------------------
# Videoclipul unui exemplu
# ---------------------------------------------------------------------------

def _wrist_track(raw, hand_start, n_frames):
    """
    Poziția încheieturii unei mâini (x, y normalizate) în fiecare cadru video, sau None.
    Keypoints-urile salvate sunt len(raw) cadre eșantionate uniform din videoclip, deci
    poziția într-un cadru video se interpolează între cele două eșantioane vecine.
    """
    hand = raw[:, hand_start:hand_start + 21 * 3].reshape(len(raw), 21, 3)[:, :, :2]
    present = np.any(hand != 0, axis=(1, 2))
    hand, present = fill_gaps(hand, present)
    wrist = hand[:, 0]
    track = []
    for j in range(n_frames):
        k = j * (len(raw) - 1) / max(n_frames - 1, 1)   # poziția fracționară între eșantioane
        k0 = int(k)
        k1, w = min(k0 + 1, len(raw) - 1), k - k0
        if present[k0] and present[k1]:
            track.append(wrist[k0] * (1 - w) + wrist[k1] * w)
        elif present[k0] and w < 0.5:
            track.append(wrist[k0])
        elif present[k1] and w >= 0.5:
            track.append(wrist[k1])
        else:
            track.append(None)
    return track


class VideoClip:
    """Cadrele unui videoclip WLASL, redimensionate pentru panou și fără timp mort,
    plus poziția fiecărei mâini a semnatarului (pentru marcajele D / S).
    `left_handed`: semnatarul folosește mâna stângă; videoclipul se afișează întors,
    ca semnatarul să apară dreptaci, iar marcajele D / S se schimbă între ele."""

    def __init__(self, path, raw, left_handed=False):
        self.left_handed = left_handed
        cap = cv2.VideoCapture(path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            self.frames, self.wrists, self.duration = [], [], 0.0
            return

        # Unde ajunge un punct normalizat (x, y) din videoclip în panou
        h, w = frames[0].shape[:2]
        self.scale = min(PANEL_W / w, PANEL_H / h)
        self.size = (int(w * self.scale), int(h * self.scale))
        self.offset = ((PANEL_W - self.size[0]) // 2, (PANEL_H - self.size[1]) // 2)
        tracks = {label: _wrist_track(raw, start, len(frames)) for label, start, _ in HAND_MARKERS}

        # Timpul mort: keypoints-urile salvate (raw) sunt cadre eșantionate uniform din
        # tot videoclipul, deci prima / ultima mână detectată se poate duce înapoi în timp.
        hands = np.where(np.any(raw[:, _HANDS_START:] != 0, axis=1))[0]
        start, end = 0, len(frames)
        if len(frames) > 1 and len(hands) >= 3:
            positions = np.linspace(0, len(frames) - 1, len(raw))
            margin = int(MARGIN_SECONDS * self.fps)
            start = max(0, int(positions[hands[0]]) - margin)
            end = min(len(frames), int(positions[hands[-1]]) + margin + 1)

        self.frames = [self._fit(f) for f in frames[start:end]]
        # Pentru fiecare cadru: {literă: (x, y) în pixeli ai panoului} pentru mâinile vizibile
        self.wrists = [{label: self._to_panel(tracks[label][j]) for label in tracks
                        if tracks[label][j] is not None} for j in range(start, end)]
        self.duration = len(self.frames) / self.fps

    def _to_panel(self, point):
        return (int(point[0] * self.size[0] + self.offset[0]), int(point[1] * self.size[1] + self.offset[1]))

    def _fit(self, frame):
        """Redimensionează păstrând proporțiile și centrează pe fundalul panoului."""
        resized = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        panel = np.full((PANEL_H, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
        x, y = self.offset
        panel[y:y + self.size[1], x:x + self.size[0]] = resized
        return panel

    def frame_at(self, seconds, mirrored=False):
        """
        Cadrul de afișat după `seconds` secunde de redare (ultimul cadru în pauza de final),
        cu marcajele D / S pe încheieturile semnatarului.
        """
        j = min(int(seconds * self.fps), len(self.frames) - 1)
        flip = mirrored != self.left_handed   # un semnatar stângaci, întors, arată ca un dreptaci
        frame = cv2.flip(self.frames[j], 1) if flip else self.frames[j].copy()
        for label, _, color in HAND_MARKERS:
            # La un stângaci întors, mâna lui stângă face rolul mâinii drepte (D)
            source = {'D': 'S', 'S': 'D'}[label] if self.left_handed else label
            if source in self.wrists[j]:
                x, y = self.wrists[j][source]
                if flip:
                    x = PANEL_W - 1 - x
                cv2.circle(frame, (x, y), 13, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 13, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, label, (x - 7, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 2, cv2.LINE_AA)
        return frame


# ---------------------------------------------------------------------------
# Alegerea exemplelor
# ---------------------------------------------------------------------------

def _load_examples(model=None):
    """
    Pentru fiecare cuvânt: videoclipurile originale din train+val care există pe disc,
    ca (cale mp4, keypoints brute, stângaci?), ordonate după încrederea modelului în clasa
    corectă (cel mai tipic primul) și apoi după cât de des se văd mâinile.
    Semnatarii stângaci (lista 'mirrored' din data/cleaning_log.json) se evaluează oglindiți,
    exact ca la antrenare, și se afișează întorși.
    """
    mirrored = mirrored_files()
    with open(SPLITS_FILE, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    with open(VIDEO_INDEX_FILE, 'r', encoding='utf-8') as f:
        video_index = json.load(f)
    if model is None and os.path.exists(MODEL_PATH):
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH)
    if model is not None and model.output_shape[-1] != NUM_CLASSES:
        # Model antrenat pe alt vocabular: indicii claselor nu se potrivesc cu ACTIONS
        print(f"Atenție: modelul are {model.output_shape[-1]} clase, ACTIONS are {NUM_CLASSES}. "
              "Exemplele se aleg fără model, după cât de des se văd mâinile.")
        model = None

    examples = {}
    for label, action in enumerate(ACTIONS):
        files, videos = [], []
        for npy in splits['train'][action] + splits['val'][action]:
            video = os.path.join(RAW_VIDEOS_DIR, action, video_index[action].get(npy, ''))
            if os.path.isfile(video):
                files.append(npy)
                videos.append(video)
        paths = [os.path.join(KEYPOINTS_DIR, action, f) for f in files]
        raws = [np.load(p) for p in paths]
        left_handed = [(action, f) in mirrored for f in files]
        coverage = np.array([np.any(r[:, _HANDS_START:] != 0, axis=1).mean() for r in raws])
        if model is not None and raws:
            X = fit_inputs(model, np.stack([load_example(action, p, mirrored) for p in paths]))
            probs = model.predict(X, verbose=0)[:, label]
        else:
            probs = np.zeros(len(raws))
        order = list(np.lexsort((-coverage, -np.round(probs, 2))))
        # Exemplul fixat manual în sign_info.py (dacă există) trece primul
        pinned = SIGN_INFO[action].get('example')
        if pinned in files:
            order.remove(files.index(pinned))
            order.insert(0, files.index(pinned))
        examples[action] = [(videos[i], raws[i], left_handed[i]) for i in order]
    return examples


# ---------------------------------------------------------------------------
# Fereastra ghidului
# ---------------------------------------------------------------------------

def _load_font(size, bold=False):
    names = ['arialbd.ttf', 'segoeuib.ttf'] if bold else ['arial.ttf', 'segoeui.ttf']
    for name in names:
        try:
            return ImageFont.truetype(os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', name), size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap(draw, text, font, max_width):
    lines, line = [], ''
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if draw.textlength(candidate, font=font) <= max_width:
            line = candidate
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


class SignGuide:
    def __init__(self, model=None):
        print("Pregătim ghidul de semne...")
        self.examples = _load_examples(model)
        self.fonts = {
            'title': _load_font(44, bold=True), 'meaning': _load_font(26),
            'text': _load_font(19), 'small': _load_font(16), 'label': _load_font(19, bold=True),
        }
        self.index, self.variant = 0, 0
        self.mirrored = True   # implicit ca în oglindă: e mai ușor de copiat
        self.slow = False
        self._clips = {}
        self._background_key, self._background = None, None
        self._restart()

    @property
    def target(self):
        return ACTIONS[self.index]

    @property
    def clip(self):
        """Videoclipul curent (citit de pe disc o singură dată), sau None dacă nu există."""
        if not self.examples[self.target]:
            return None
        key = (self.target, self.variant)
        if key not in self._clips:
            self._clips[key] = VideoClip(*self.examples[self.target][self.variant])
        return self._clips[key]

    def _restart(self):
        """Redarea pornește de la început."""
        self._position = 0.0                   # secunde de videoclip redate
        self._last_tick = time.perf_counter()

    def handle_key(self, key):
        """Procesează o tastă. Returnează True dacă s-a schimbat cuvântul-țintă."""
        if key in (ord('n'), ord('p')):
            step = 1 if key == ord('n') else -1
            self.index = (self.index + step) % len(ACTIONS)
            self.variant = 0
            self._restart()
            return True
        if key == ord('v') and self.examples[self.target]:
            self.variant = (self.variant + 1) % len(self.examples[self.target])
            self._restart()
        elif key == ord('m'):
            self.mirrored = not self.mirrored
        elif key == ord('l'):
            self.slow = not self.slow
        return False

    def _render_background(self, prediction, message=None):
        """Textul (se schimbă rar): randat o singură dată, nu la fiecare cadru."""
        img = Image.new('RGB', (WIDTH, HEIGHT), BG[::-1])
        draw = ImageDraw.Draw(img)
        f, info = self.fonts, SIGN_INFO[self.target]

        draw.text((20, 15), self.target.upper(), font=f['title'], fill=ORANGE)
        draw.text((20, 72), info['meaning'], font=f['meaning'], fill=WHITE)
        draw.text((WIDTH - 90, 25), f"{self.index + 1}/{len(ACTIONS)}", font=f['meaning'], fill=GRAY)

        y = PANEL_Y + PANEL_H + 8
        n_var = len(self.examples[self.target])
        side = "oglindit" if self.mirrored else "normal"
        speed = "încetinit" if self.slow else "viteză normală"
        draw.text((20, y), f"semnatar {self.variant + 1}/{n_var} (v)  ·  {side} (m)  ·  {speed} (l)",
                  font=f['small'], fill=GRAY)
        y += 22
        if n_var and self.examples[self.target][self.variant][2]:
            hint = "Semnatar stângaci: videoclipul e întors, ca să-l copiezi cu mâna dreaptă."
        elif self.mirrored:
            hint = "Oglindit: copiază-l ca în oglindă; mâna ta dreaptă face ce face D."
        else:
            hint = "Normal: semnatarul te privește, mâna lui dreaptă (D) e în stânga ecranului."
        draw.text((20, y), hint, font=f['small'], fill=GRAY)

        # Ce face fiecare mână (culorile se potrivesc cu marcajele D / S de pe videoclip)
        y += 30
        if 'both' in info:
            sections = [('Ambele mâini', info['both'], WHITE)]
        elif 'right' in info:
            sections = [('D · Mâna dreaptă', info['right'], RIGHT_HAND_RGB),
                        ('S · Mâna stângă', info['left'], LEFT_HAND_RGB)]
        else:   # descrierea mâinilor nu e scrisă încă: doar videoclipul
            sections = [('Urmărește videoclipul', 'D = mâna dreaptă a semnatarului, S = cea stângă.', GRAY)]
        if 'then' in info:
            sections.append(('Apoi, ambele mâini', info['then'], WHITE))
        for title, text, color in sections:
            draw.text((20, y), title, font=f['label'], fill=color)
            y += 25
            for line in _wrap(draw, text, f['text'], WIDTH - 40):
                draw.text((20, y), line, font=f['text'], fill=WHITE)
                y += 24
            y += 8
        if 'note' in info:
            for line in _wrap(draw, info['note'], f['small'], WIDTH - 40):
                draw.text((20, y), line, font=f['small'], fill=ORANGE)
                y += 21

        bar_top = HEIGHT - 70
        if message is not None:
            draw.text((20, bar_top + 22), message, font=f['text'], fill=WHITE)
        elif prediction is None:
            draw.text((20, bar_top + 22), "Pornește live_predict pentru verificare.", font=f['text'], fill=GRAY)
        elif prediction == self.target:
            draw.rectangle((0, bar_top, WIDTH, HEIGHT), fill=GREEN)
            draw.text((20, bar_top + 18), f"CORECT!  Modelul a recunoscut „{prediction}”",
                      font=f['meaning'], fill=WHITE)
        else:
            draw.text((20, bar_top + 18), f"Modelul vede: {prediction or '—'}", font=f['meaning'], fill=GRAY)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def _video_frame(self):
        """Avansează redarea după timpul real scurs și returnează cadrul curent."""
        now = time.perf_counter()
        self._position += (now - self._last_tick) * (SLOW_SPEED if self.slow else 1.0)
        self._last_tick = now
        clip = self.clip
        if clip is None or not clip.frames:
            return None
        if self._position > clip.duration + HOLD_SECONDS:
            self._position = 0.0
        return clip.frame_at(self._position, self.mirrored)

    def render(self, prediction=None, message=None):
        key = (self.target, self.variant, self.mirrored, self.slow, prediction, message)
        if key != self._background_key:
            self._background_key, self._background = key, self._render_background(prediction, message)
        image = self._background.copy()
        frame = self._video_frame()
        if frame is not None:
            image[PANEL_Y:PANEL_Y + PANEL_H, PANEL_X:PANEL_X + PANEL_W] = frame
        return image

    def set_target(self, action):
        """Setează direct cuvântul-țintă (folosit de înregistrare și de testul live)."""
        self.index, self.variant = ACTIONS.index(action), 0
        self._restart()

    def show(self, prediction=None, message=None):
        """Desenează fereastra. `message` înlocuiește bara de jos."""
        cv2.imshow(WINDOW_NAME, self.render(prediction, message))


def main():
    guide = SignGuide()
    while True:
        guide.show()
        key = cv2.waitKey(15) & 0xFF
        if key == ord('q') or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
        guide.handle_key(key)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
