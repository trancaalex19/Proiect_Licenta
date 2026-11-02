"""
Ghid de semne: arată cuvântul-țintă, ce înseamnă, cum se execută și un avatar
animat (față, trunchi, brațe, mâini și urma mișcării) reconstruit din
keypoints-urile unui videoclip real din setul de date.

Exemplul afișat pentru fiecare cuvânt este cel pe care modelul antrenat îl
recunoaște cu cea mai mare încredere (cel mai „tipic”); celelalte videoclipuri
sunt disponibile ca variante (tasta v), pentru că multe semne ASL au variante.

Pornește automat împreună cu src.app.live_predict, unde afișează și dacă
modelul a recunoscut semnul tău („CORECT”). Poate rula și singur:
    python -m src.app.sign_guide

Taste: n / p = cuvântul următor / anterior, v = altă variantă,
       m = oglindește avatarul, q = ieșire
"""
import json
import os

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.app.sign_info import SIGN_INFO
from src.config import ACTIONS, KEYPOINTS_DIR, MODEL_PATH, SPLITS_FILE
from src.features.preprocessing import fill_gaps, preprocess

WINDOW_NAME = 'Ghid semne (n/p = cuvant, v = varianta, m = oglinda)'
WIDTH, HEIGHT = 560, 800
PANEL_X, PANEL_Y, PANEL_W, PANEL_H = 10, 120, WIDTH - 20, 400   # zona avatarului
UPSAMPLE = 3          # cadre interpolate între două cadre originale (mișcare fluidă)
HOLD_FRAMES = 20      # pauză la finalul animației, înainte de reluare
TRAIL_FRAMES = 18     # lungimea urmei lăsate de mâini
REAL_FACE_RATIO = 1.22  # înălțime / lățime pentru conturul feței MediaPipe la o față reală

# Culori BGR (OpenCV)
BG, PANEL_BG = (30, 30, 30), (48, 48, 48)
SKIN, BODY, TORSO = (205, 205, 205), (190, 190, 190), (70, 70, 70)
RIGHT_HAND, LEFT_HAND = (16, 117, 245), (230, 160, 40)
# Culori RGB (PIL)
WHITE, GRAY, ORANGE, GREEN = (255, 255, 255), (170, 170, 170), (245, 117, 16), (60, 200, 90)

# Indici în vectorul brut de 1662 (vezi src/features/keypoints.py)
_POSE_END, _FACE_END = 33 * 4, 33 * 4 + 468 * 3
_LH_END = _FACE_END + 21 * 3

_fm = mp.solutions.face_mesh_connections
FACE_CONNECTIONS = list(_fm.FACEMESH_FACE_OVAL | _fm.FACEMESH_LIPS | _fm.FACEMESH_LEFT_EYE |
                        _fm.FACEMESH_RIGHT_EYE | _fm.FACEMESH_LEFT_EYEBROW | _fm.FACEMESH_RIGHT_EYEBROW)
FACE_OVAL = sorted({i for c in _fm.FACEMESH_FACE_OVAL for i in c})
CHIN = 152
HAND_CONNECTIONS = list(mp.solutions.hands_connections.HAND_CONNECTIONS)
PALM = [0, 1, 5, 9, 13, 17]
ARMS = [(11, 13), (13, 15), (12, 14), (14, 16)]


# ---------------------------------------------------------------------------
# Pregătirea unui exemplu (o singură dată, la pornire)
# ---------------------------------------------------------------------------

def _fill_edges(points, present):
    """Completează începutul / sfârșitul cu cel mai apropiat cadru detectat."""
    idx = np.where(present)[0]
    if len(idx) == 0:
        return points, present
    points = points.copy()
    points[:idx[0]] = points[idx[0]]
    points[idx[-1] + 1:] = points[idx[-1]]
    return points, np.ones_like(present)


def _upsample(points, present, factor):
    """Interpolare liniară în timp (doar între cadre în care partea e prezentă)."""
    n = len(points)
    if n < 2:
        return points, present
    ts = np.linspace(0, n - 1, (n - 1) * factor + 1)
    i0 = np.minimum(np.floor(ts).astype(int), n - 2)
    w = (ts - i0)[:, None, None]
    i1 = i0 + 1
    both = present[i0] & present[i1]
    nearest = np.where(w[:, 0, 0] < 0.5, i0, i1)
    out = np.where(both[:, None, None], points[i0] * (1 - w) + points[i1] * w, points[nearest])
    return out, np.where(both, True, present[nearest])


class Clip:
    """Un exemplu gata de desenat: coordonate în pixeli ale panoului, pe cadre."""

    def __init__(self, raw):
        T = len(raw)
        pose = raw[:, :_POSE_END].reshape(T, 33, 4)[:, :, :2]
        face = raw[:, _POSE_END:_FACE_END].reshape(T, 468, 3)[:, :, :2]
        lh = raw[:, _FACE_END:_LH_END].reshape(T, 21, 3)[:, :, :2]
        rh = raw[:, _LH_END:].reshape(T, 21, 3)[:, :, :2]
        present = {k: np.any(v != 0, axis=(1, 2)) for k, v in
                   (('pose', pose), ('face', face), ('lh', lh), ('rh', rh))}

        # Golurile scurte se completează (mâinile nu mai „clipesc”)
        lh, present['lh'] = fill_gaps(lh, present['lh'])
        rh, present['rh'] = fill_gaps(rh, present['rh'])
        pose, present['pose'] = _fill_edges(*fill_gaps(pose, present['pose'], 1.0))
        face, present['face'] = _fill_edges(*fill_gaps(face, present['face'], 1.0))

        # Fără timp mort: de la prima la ultima detecție a unei mâini
        active = np.where(present['lh'] | present['rh'])[0]
        if len(active) >= 3:
            s, e = max(0, active[0] - 2), min(T, active[-1] + 3)
            pose, face, lh, rh = pose[s:e], face[s:e], lh[s:e], rh[s:e]
            present = {k: v[s:e] for k, v in present.items()}

        # Proporții: coordonatele MediaPipe sunt normalizate separat pe lățime și înălțime;
        # raportul de aspect al videoclipului se estimează din forma feței.
        aspect = 4 / 3
        if present['face'].any():
            oval = face[present['face']][:, FACE_OVAL]
            ratio = np.median((np.ptp(oval[:, :, 1], axis=1) + 1e-6) / (np.ptp(oval[:, :, 0], axis=1) + 1e-6))
            aspect = float(np.clip(ratio / REAL_FACE_RATIO, 1.0, 2.0))
        for arr in (pose, face, lh, rh):
            arr[:, :, 0] *= aspect

        # Mișcare fluidă
        pose, present['pose'] = _upsample(pose, present['pose'], UPSAMPLE)
        face, present['face'] = _upsample(face, present['face'], UPSAMPLE)
        lh, present['lh'] = _upsample(lh, present['lh'], UPSAMPLE)
        rh, present['rh'] = _upsample(rh, present['rh'], UPSAMPLE)

        # Încadrare constantă: lățimea umerilor = ~32% din panou, centrat pe umeri
        shoulders = pose[:, [11, 12]]
        center = np.median(shoulders.mean(axis=1), axis=0)
        shoulder_w = np.median(np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1))
        scale = 0.32 * PANEL_W / max(shoulder_w, 1e-3)
        visible = np.concatenate([face.reshape(-1, 2), pose[:, 11:23].reshape(-1, 2),
                                  lh[present['lh']].reshape(-1, 2), rh[present['rh']].reshape(-1, 2)])
        lo, hi = visible.min(axis=0), visible.max(axis=0)
        fit = min(0.92 * PANEL_W / max(hi[0] - lo[0], 1e-3), 0.92 * PANEL_H / max(hi[1] - lo[1], 1e-3))
        scale = min(scale, fit)
        offset_x = PANEL_W / 2 - center[0] * scale
        offset_y = PANEL_H / 2 - (lo[1] + hi[1]) / 2 * scale

        def px(a):
            return np.stack([a[..., 0] * scale + offset_x, a[..., 1] * scale + offset_y], axis=-1)

        self.pose, self.face, self.lh, self.rh = px(pose), px(face), px(lh), px(rh)
        self.present = present
        self.length = len(self.pose)


# ---------------------------------------------------------------------------
# Desen
# ---------------------------------------------------------------------------

def _pt(p):
    return int(round(p[0])), int(round(p[1]))


def _draw_hand(panel, hand, color):
    palm = np.array([_pt(hand[i]) for i in PALM], dtype=np.int32)
    fill = tuple(int(c * 0.45) for c in color)
    cv2.fillPoly(panel, [palm], fill, cv2.LINE_AA)
    for a, b in HAND_CONNECTIONS:
        cv2.line(panel, _pt(hand[a]), _pt(hand[b]), color, 3, cv2.LINE_AA)
    for i, p in enumerate(hand):
        cv2.circle(panel, _pt(p), 5 if i in (4, 8, 12, 16, 20) else 3, color, -1, cv2.LINE_AA)


def _draw_trail(panel, hand_track, present, frame, color):
    """Urma mâinii (centrul palmei) pe ultimele TRAIL_FRAMES cadre, care se estompează."""
    if not present[frame]:
        return  # fără urme rătăcite după ce mâna a coborât din cadru
    start = max(0, frame - TRAIL_FRAMES)
    for t in range(start + 1, frame + 1):
        if present[t] and present[t - 1]:
            alpha = (t - start) / TRAIL_FRAMES
            c = tuple(int(PANEL_BG[i] + (color[i] - PANEL_BG[i]) * alpha * 0.8) for i in range(3))
            a = hand_track[t - 1][PALM].mean(axis=0)
            b = hand_track[t][PALM].mean(axis=0)
            cv2.line(panel, _pt(a), _pt(b), c, 3, cv2.LINE_AA)


def draw_avatar(clip, frame, mirrored=False):
    panel = np.full((PANEL_H, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
    pose, face = clip.pose[frame], clip.face[frame]

    # Trunchi (umeri -> șolduri) și gât
    torso = np.array([_pt(pose[i]) for i in (11, 12, 24, 23)], dtype=np.int32)
    cv2.fillPoly(panel, [torso], TORSO, cv2.LINE_AA)
    neck = (pose[11] + pose[12]) / 2
    cv2.line(panel, _pt(neck), _pt(face[CHIN]), BODY, 6, cv2.LINE_AA)
    cv2.line(panel, _pt(pose[11]), _pt(pose[12]), BODY, 6, cv2.LINE_AA)

    # Față: contur, ochi, sprâncene, gură
    for a, b in FACE_CONNECTIONS:
        cv2.line(panel, _pt(face[a]), _pt(face[b]), SKIN, 2, cv2.LINE_AA)

    # Urma mișcării, apoi brațele și mâinile
    _draw_trail(panel, clip.rh, clip.present['rh'], frame, RIGHT_HAND)
    _draw_trail(panel, clip.lh, clip.present['lh'], frame, LEFT_HAND)
    for a, b in ARMS:
        cv2.line(panel, _pt(pose[a]), _pt(pose[b]), BODY, 8, cv2.LINE_AA)
    for i in (11, 12, 13, 14):
        cv2.circle(panel, _pt(pose[i]), 6, BODY, -1, cv2.LINE_AA)
    # Antebrațul continuă până la încheietura mâinii detectate
    for hand, key, elbow, color in ((clip.rh, 'rh', 14, RIGHT_HAND), (clip.lh, 'lh', 13, LEFT_HAND)):
        if clip.present[key][frame]:
            cv2.line(panel, _pt(pose[elbow + 2]), _pt(hand[frame][0]), BODY, 8, cv2.LINE_AA)
            _draw_hand(panel, hand[frame], color)

    if mirrored:
        panel = cv2.flip(panel, 1)
    return panel


# ---------------------------------------------------------------------------
# Alegerea exemplelor
# ---------------------------------------------------------------------------

def _load_examples(model=None):
    """
    Pentru fiecare cuvânt: toate videoclipurile originale din train+val, ordonate după
    încrederea modelului în clasa corectă (cel mai tipic primul) și apoi după cât de
    des se văd mâinile.
    """
    with open(SPLITS_FILE, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    if model is None and os.path.exists(MODEL_PATH):
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH)

    examples = {}
    for label, action in enumerate(ACTIONS):
        files = splits['train'][action] + splits['val'][action]
        raws = [np.load(os.path.join(KEYPOINTS_DIR, action, f)) for f in files]
        coverage = np.array([np.any(r[:, _FACE_END:] != 0, axis=1).mean() for r in raws])
        if model is not None:
            probs = model.predict(np.stack([preprocess(r) for r in raws]), verbose=0)[:, label]
        else:
            probs = np.zeros(len(raws))
        order = list(np.lexsort((-coverage, -np.round(probs, 2))))
        # Exemplul fixat manual în sign_info.py (dacă există) trece primul
        pinned = SIGN_INFO[action].get('example')
        if pinned in files:
            order.remove(files.index(pinned))
            order.insert(0, files.index(pinned))
        examples[action] = [raws[i] for i in order]
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
            'text': _load_font(20), 'small': _load_font(16),
        }
        self.index, self.variant = 0, 0
        self.mirrored = False
        self._clips = {}
        self._background_key, self._background = None, None
        self._reset_animation()

    @property
    def target(self):
        return ACTIONS[self.index]

    @property
    def clip(self):
        key = (self.target, self.variant)
        if key not in self._clips:
            self._clips[key] = Clip(self.examples[self.target][self.variant])
        return self._clips[key]

    def _reset_animation(self):
        self.frame, self.hold = 0, 0

    def handle_key(self, key):
        """Procesează o tastă. Returnează True dacă s-a schimbat cuvântul-țintă."""
        if key in (ord('n'), ord('p')):
            step = 1 if key == ord('n') else -1
            self.index = (self.index + step) % len(ACTIONS)
            self.variant = 0
            self._reset_animation()
            return True
        if key == ord('v'):
            self.variant = (self.variant + 1) % len(self.examples[self.target])
            self._reset_animation()
        elif key == ord('m'):
            self.mirrored = not self.mirrored
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
        draw.text((20, y), "● mâna dreaptă", font=f['small'], fill=RIGHT_HAND[::-1])
        draw.text((160, y), "● mâna stângă", font=f['small'], fill=LEFT_HAND[::-1])
        n_var = len(self.examples[self.target])
        side = "oglindit" if self.mirrored else "văzut din față"
        draw.text((300, y), f"varianta {self.variant + 1}/{n_var} · {side}", font=f['small'], fill=GRAY)

        y += 32
        draw.text((20, y), "Cum se face:", font=f['text'], fill=GRAY)
        y += 28
        for line in _wrap(draw, info['how'], f['text'], WIDTH - 40):
            draw.text((20, y), line, font=f['text'], fill=WHITE)
            y += 26

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

    def render(self, prediction=None, message=None):
        key = (self.target, self.variant, self.mirrored, prediction, message)
        if key != self._background_key:
            self._background_key, self._background = key, self._render_background(prediction, message)
        image = self._background.copy()
        image[PANEL_Y:PANEL_Y + PANEL_H, PANEL_X:PANEL_X + PANEL_W] = \
            draw_avatar(self.clip, min(self.frame, self.clip.length - 1), self.mirrored)
        return image

    def set_target(self, action):
        """Setează direct cuvântul-țintă (folosit de scriptul de înregistrare)."""
        self.index, self.variant = ACTIONS.index(action), 0
        self._reset_animation()

    def show(self, prediction=None, message=None):
        """Desenează fereastra și avansează animația cu un cadru. `message` înlocuiește bara de jos."""
        cv2.imshow(WINDOW_NAME, self.render(prediction, message))
        if self.frame < self.clip.length - 1:
            self.frame += 1
        elif self.hold < HOLD_FRAMES:
            self.hold += 1
        else:
            self._reset_animation()


def main():
    guide = SignGuide()
    while True:
        guide.show()
        key = cv2.waitKey(40) & 0xFF  # ~25 cadre/secundă
        if key == ord('q') or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
        guide.handle_key(key)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
