"""
Înregistrează exemple proprii pentru cuvintele care au puține videoclipuri,
ca toate clasele să ajungă la același număr de exemple.

Fiecare înregistrare = RECORD_SECONDS secunde de la cameră, din care se extrag
keypoints-urile cu ACEEAȘI funcție ca pentru WLASL și se salvează ca secvență
brută (SEQUENCE_LENGTH x 1662) în data/recorded_keypoints/<cuvânt>/rec_<dată>.npy.

Înregistrările se folosesc DOAR la antrenare (vezi src/ml/dataset.py), niciodată
în validare sau test.

Rulare (din rădăcina proiectului, cu venv activat):
    python -m src.data.record_samples                     # toate cuvintele, până la nivelul maxim
    python -m src.data.record_samples --target 20         # alt număr țintă de exemple per cuvânt
    python -m src.data.record_samples --words hello stop  # doar anumite cuvinte
    python -m src.data.record_samples --camera 0          # altă cameră
    python -m src.data.record_samples --seconds 5         # altă durată pentru toate cuvintele

Taste: SPAȚIU = start / pauză, r = șterge ultima înregistrare, s = sari peste cuvânt,
       v = alt semnatar în ghid, m = oglindește videoclipul, l = încetinit, q = ieșire
"""
import argparse
import os
import time
from datetime import datetime

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import cv2
import mediapipe as mp
import numpy as np

from src.config import ACTIONS, CAMERA_INDEX, KEYPOINTS_DIR, RECORDED_DIR, SEQUENCE_LENGTH
from src.features.keypoints import extract_keypoints, to_fixed_length

WINDOW_NAME = 'Inregistrare (SPATIU = start/pauza, r = sterge ultima, s = sari, q = iesire)'
COUNTDOWN_SECONDS = 3
RECORD_SECONDS = 3.0
# Semnele compuse durează mai mult (ex.: student = „learn” + semnul pentru „persoană").
# Durata nu schimbă formatul: fiecare înregistrare se reduce la SEQUENCE_LENGTH cadre,
# iar cadrele fără mâini de la început / sfârșit se elimină la preprocesare.
WORD_SECONDS = {'student': 5.0}
MIN_HAND_RATIO = 0.3   # minim 30% din cadre trebuie să aibă o mână, altfel se repetă
MIN_POSE_RATIO = 0.8   # umerii trebuie să se vadă (normalizarea se face față de umeri)
DISPLAY_WIDTH = 800

_FACE_END = 33 * 4 + 468 * 3

TIPS = [
    "Pornește cu mâinile jos, fă semnul, apoi coboară mâinile.",
    "Variază puțin distanța față de cameră între repetări.",
    "Mută-te puțin la stânga / dreapta între repetări.",
    "Variază puțin viteza: uneori mai rar, uneori mai repede.",
    "Ține umerii în cadru (de la talie în sus).",
]


# ---------------------------------------------------------------------------
# Numărarea exemplelor existente
# ---------------------------------------------------------------------------

def count_wlasl(action):
    folder = os.path.join(KEYPOINTS_DIR, action)
    return len([f for f in os.listdir(folder) if f.endswith('.npy') and not f.startswith('aug_')])


def count_recorded(action):
    folder = os.path.join(RECORDED_DIR, action)
    return len([f for f in os.listdir(folder) if f.endswith('.npy')]) if os.path.isdir(folder) else 0


def record_seconds(action, override=None):
    """Durata înregistrării pentru un cuvânt (opțiunea --seconds are prioritate)."""
    return override or WORD_SECONDS.get(action, RECORD_SECONDS)


def plan_recordings(words, target, seconds_override=None):
    """Câte înregistrări lipsesc pentru fiecare cuvânt; cele mai deficitare primele."""
    plan = []
    print(f"\n{'cuvânt':<10} {'WLASL':>5} {'proprii':>7} {'total':>5} {'lipsesc':>7} {'durată':>7}")
    for action in words:
        wlasl, rec = count_wlasl(action), count_recorded(action)
        missing = max(0, target - wlasl - rec)
        seconds = record_seconds(action, seconds_override)
        print(f"{action:<10} {wlasl:>5} {rec:>7} {wlasl + rec:>5} {missing:>7} {seconds:>6.0f}s")
        if missing:
            plan.append([action, missing])
    plan.sort(key=lambda item: -item[1])
    total = sum(m for _, m in plan)
    minutes = sum(m * (record_seconds(a, seconds_override) + COUNTDOWN_SECONDS + 1) for a, m in plan) / 60
    print(f"\nȚintă: {target} exemple per cuvânt -> {total} înregistrări (~{minutes:.0f} minute)\n")
    return plan


# ---------------------------------------------------------------------------
# Salvare și verificare
# ---------------------------------------------------------------------------

def check_quality(frames):
    """Verifică dacă înregistrarea e utilizabilă. Returnează (ok, motiv)."""
    raw = np.array(frames)
    hand_ratio = np.any(raw[:, _FACE_END:] != 0, axis=1).mean()
    pose_ratio = np.any(raw[:, :33 * 4] != 0, axis=1).mean()
    if pose_ratio < MIN_POSE_RATIO:
        return False, "Nu s-au văzut umerii. Stai de la talie în sus în cadru."
    if hand_ratio < MIN_HAND_RATIO:
        return False, "Nu s-au văzut mâinile destul. Mai încearcă o dată."
    return True, ""


def save_recording(action, frames):
    folder = os.path.join(RECORDED_DIR, action)
    os.makedirs(folder, exist_ok=True)
    name = f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.npy"
    path = os.path.join(folder, name)
    np.save(path, to_fixed_length(frames, SEQUENCE_LENGTH))
    return path


# ---------------------------------------------------------------------------
# Interfață
# ---------------------------------------------------------------------------

def draw_status(image, state, action, done, needed, seconds_left, progress, message):
    h, w = image.shape[:2]
    header = f"{action.upper()}   {done}/{needed}"
    cv2.rectangle(image, (0, 0), (w, 44), (245, 117, 16), -1)
    cv2.putText(image, header, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    if state == 'countdown':
        text = str(int(np.ceil(seconds_left)))
        cv2.putText(image, text, (w // 2 - 30, h // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 3.0,
                    (255, 255, 255), 6, cv2.LINE_AA)
        cv2.putText(image, "Pregateste-te: mainile jos", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
    elif state == 'recording':
        cv2.rectangle(image, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)
        cv2.rectangle(image, (0, h - 14), (int(w * progress), h), (0, 0, 255), -1)
        cv2.putText(image, "INREGISTRARE - fa semnul", (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2, cv2.LINE_AA)
    elif state == 'paused':
        cv2.putText(image, "PAUZA - apasa SPATIU", (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 255), 2, cv2.LINE_AA)
    if message:
        cv2.putText(image, message, (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)


def ascii(text):
    """OpenCV nu afișează diacritice."""
    return text.translate(str.maketrans('ăâîșşțţĂÂÎȘŞȚŢ', 'aaissttAAISSTT'))


def parse_args():
    parser = argparse.ArgumentParser(description='Înregistrare exemple proprii')
    parser.add_argument('--target', type=int, default=None,
                        help='exemple per cuvânt (implicit: maximul din WLASL)')
    parser.add_argument('--words', nargs='+', choices=ACTIONS, default=ACTIONS,
                        help='doar aceste cuvinte')
    parser.add_argument('--camera', type=int, default=CAMERA_INDEX)
    parser.add_argument('--seconds', type=float, default=None,
                        help=f'durata unei înregistrări (implicit {RECORD_SECONDS:.0f}s; {WORD_SECONDS})')
    return parser.parse_args()


def main():
    args = parse_args()
    target = args.target or max(count_wlasl(a) for a in ACTIONS)
    plan = plan_recordings(args.words, target, args.seconds)
    if not plan:
        print("Toate cuvintele au deja numărul țintă de exemple.")
        return

    from src.app.sign_guide import SignGuide
    guide = SignGuide()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"EROARE: Nu pot deschide camera {args.camera}.")
        return

    word_idx = 0
    done = 0                 # înregistrări salvate pentru cuvântul curent în această sesiune
    saved_paths = []         # pentru tasta r (ștergerea ultimei înregistrări)
    state, state_start = 'paused', time.perf_counter()
    frames, message, tip_idx = [], "Apasa SPATIU pentru a incepe.", 0
    guide.set_target(plan[0][0])
    windows_placed = False

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened() and word_idx < len(plan):
            action, needed = plan[word_idx]
            ret, frame = cap.read()
            if not ret:
                break
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = frame.copy()
            for hand in (results.left_hand_landmarks, results.right_hand_landmarks):
                mp.solutions.drawing_utils.draw_landmarks(image, hand, mp.solutions.holistic.HAND_CONNECTIONS)

            now = time.perf_counter()
            elapsed = now - state_start
            duration = record_seconds(action, args.seconds)

            if state == 'countdown' and elapsed >= COUNTDOWN_SECONDS:
                state, state_start, frames = 'recording', now, []
            elif state == 'recording':
                frames.append(extract_keypoints(results))
                if elapsed >= duration:
                    ok, reason = check_quality(frames)
                    if ok:
                        saved_paths.append(save_recording(action, frames))
                        done += 1
                        tip_idx = (tip_idx + 1) % len(TIPS)
                        message = f"Salvat ({done}/{needed}). " + ascii(TIPS[tip_idx])
                    else:
                        message = ascii(reason)
                    if done >= needed:  # cuvânt terminat -> următorul, cu pauză
                        word_idx += 1
                        done, saved_paths = 0, []
                        if word_idx < len(plan):
                            guide.set_target(plan[word_idx][0])
                            message = f"Gata '{action}'! Urmeaza '{plan[word_idx][0]}'. Apasa SPATIU."
                        state = 'paused'
                    else:
                        state = 'countdown'
                    state_start = now

            # Afișare
            if image.shape[1] > DISPLAY_WIDTH:
                scale = DISPLAY_WIDTH / image.shape[1]
                image = cv2.resize(image, (DISPLAY_WIDTH, int(image.shape[0] * scale)))
            seconds_left = COUNTDOWN_SECONDS - elapsed if state == 'countdown' else 0
            progress = min(1.0, elapsed / duration) if state == 'recording' else 0
            current = plan[min(word_idx, len(plan) - 1)]
            draw_status(image, state, current[0], done, current[1], seconds_left, progress, message)
            cv2.imshow(WINDOW_NAME, image)
            remaining = sum(m for _, m in plan[word_idx:]) - done
            guide.show(message=f"Înregistrări rămase în total: {remaining}")

            if not windows_placed:
                from src.app.sign_guide import WINDOW_NAME as GUIDE_WINDOW
                cv2.moveWindow(WINDOW_NAME, 0, 0)
                cv2.moveWindow(GUIDE_WINDOW, image.shape[1] + 10, 0)
                windows_placed = True

            # Taste
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if state == 'paused':
                    state, state_start, message = 'countdown', time.perf_counter(), ""
                else:
                    state, message = 'paused', "Pauza. Apasa SPATIU pentru a continua."
            elif key == ord('r') and saved_paths and state != 'recording':
                os.remove(saved_paths.pop())
                done -= 1
                state, message = 'paused', f"Ultima inregistrare a fost stearsa ({done}/{needed})."
            elif key == ord('s') and state != 'recording':
                word_idx += 1
                done, saved_paths = 0, []
                if word_idx < len(plan):
                    guide.set_target(plan[word_idx][0])
                state, message = 'paused', "Cuvant sarit. Apasa SPATIU."
            elif key in (ord('v'), ord('m'), ord('l')):
                guide.handle_key(key)

    cap.release()
    cv2.destroyAllWindows()
    print("\nSesiune încheiată. Situația actuală:")
    plan_recordings(args.words, target, args.seconds)
    print("Pentru a folosi înregistrările: python -m src.ml.train_lstm")


if __name__ == '__main__':
    main()
