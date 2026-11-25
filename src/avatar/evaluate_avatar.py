"""
Evaluare în buclă închisă (back-translation): recunoaște modelul nostru semnele făcute de avatar?

Pagina avatarului (?backtranslate) redă fiecare semn exact ca pe ecran (tranziții, netezire) și
trimite, cadru cu cadru, pozițiile oaselor ca puncte MediaPipe (Pose 33 + Hands 2 x 21). Aici
punctele se aduc în coordonatele imaginii înregistrării (aceeași lățime și același centru al
umerilor), mâna se consideră „detectată” doar când încheietura e în cadru (ca la cameră), apoi
trec prin `preprocess` și prin models/sign_language_model.h5, ca orice semn live.

Pentru comparație se clasifică și mișcarea exportată (datele din care se face animația, înainte
de retargeting): diferența dintre „date” și „avatar” e ce se pierde la transferul pe personaj.
Atenție: datele vin din înregistrările proprii, pe care modelul a fost antrenat, deci scorul
„date” e aproape mereu 100%; cifra utilă e cea a avatarului.

    python -m src.avatar.evaluate_avatar --label inainte
    python -m src.avatar.evaluate_avatar --label dupa --compare data/avatar_eval/inainte.csv
    python -m src.avatar.evaluate_avatar --words hurt yes --motion-dir alt/folder

Pornește un server local temporar și un browser Edge/Chrome fără fereastră (headless).
"""
import argparse
import csv
import functools
import http.server
import json
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np

from src.avatar.server import WEB_DIR, AvatarHandler, find_browser
from src.config import ACTIONS, AVATAR_EVAL_DIR, AVATAR_MOTION_DIR, CAMERA_ASPECT, LABEL_MAP, MODEL_PATH
from src.features.keypoints import FACE_END, LH_END, LH_START, POSE_END, RAW_DIM, RH_START
from src.features.preprocessing import preprocess
from src.ml.model import fit_inputs

# Mâna se vede în cadru doar deasupra acestei înălțimi a încheieturii (fracție din înălțimea
# imaginii). Din înregistrările proprii: încheietura cu mâna detectată e sub 0.91 în 95% din
# cadre, iar cu mâna nedetectată e peste 0.97 în 90% din cadre.
HAND_VISIBLE_Y = 0.93
# Punctele Pose ale degetelor (17/18 deget mic, 19/20 arătător, 21/22 degetul mare) nu sunt
# articulații precise: MediaPipe Pose le estimează undeva pe palmă. Le calculăm din cele 21 de
# puncte ale mâinii: încheietura + suma ponderată a (punct - încheietură). Ponderile sunt
# învățate pe înregistrările proprii (978, cadrele cu mâna detectată) prin cele mai mici pătrate
# cu ponderi nenegative (scipy.optimize.nnls); eroare mediană ~0.06 lățimi de umeri, față de
# 0.09-0.13 pentru încheieturile degetelor luate direct. Cheia = indexul punctului mâinii.
POSE_FROM_HAND = {
    17: {1: 0.029, 3: 0.052, 5: 0.014, 7: 0.092, 10: 0.042, 11: 0.065, 15: 0.044, 16: 0.027, 17: 0.308},
    19: {2: 0.158, 3: 0.069, 4: 0.012, 5: 0.264, 6: 0.124, 8: 0.005, 12: 0.059, 16: 0.011, 17: 0.149, 20: 0.072},
    21: {2: 0.075, 3: 0.143, 5: 0.231, 6: 0.125, 12: 0.018, 20: 0.112},
}
# Lungimile segmentelor degetelor (încheietură -> bază -> ... -> vârf), împărțite la mărimea
# palmei, mediana pe înregistrările proprii (3D MediaPipe Hands). Avatarul are alte proporții
# (degete mai lungi, baza degetului mare mai scurtă); camera virtuală le folosește pe acestea.
FINGER_CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
DATA_FINGER_LENGTHS = [[0.52, 0.45, 0.35, 0.28], [1.12, 0.50, 0.29, 0.22], [1.03, 0.53, 0.29, 0.21],
                       [0.95, 0.49, 0.27, 0.20], [0.91, 0.39, 0.21, 0.18]]
TIMEOUT_SECONDS = 300

# ---------------------------------------------------------------- conversii de coordonate
def scene_to_image(points, aspect=CAMERA_ASPECT):
    """Inversul lui motion.to_scene: coordonatele scenei -> coordonate MediaPipe ale imaginii."""
    points = np.asarray(points, dtype=np.float32)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return np.stack([x + 0.5, 0.5 - y / aspect, -z], axis=-1)


def to_raw(pose, left, right):
    """Puncte în coordonatele imaginii -> vectorul brut (T, 1662); None = parte nedetectată."""
    raw = np.zeros((len(pose), RAW_DIM), dtype=np.float32)
    for t, (p, l, r) in enumerate(zip(pose, left, right)):
        if p is not None:
            raw[t, :POSE_END] = np.concatenate([p, np.ones((33, 1))], axis=1).ravel()   # vizibilitate 1
        if l is not None:
            raw[t, LH_START:LH_END] = np.ravel(l)
        if r is not None:
            raw[t, RH_START:] = np.ravel(r)
    return raw


def motion_to_raw(motion):
    """Mișcarea exportată (data/avatar_motion/<cuvânt>.json) -> (T, 1662): datele animației."""
    conv = lambda points: None if points is None else scene_to_image(points)
    frames = motion['frames']
    return to_raw([conv(f['pose']) for f in frames], [conv(f['left_hand']) for f in frames],
                  [conv(f['right_hand']) for f in frames])


def body_measures(pose, active=None):
    """
    Măsuri ale corpului pentru cadre pose (T, 33, 3): centrul umerilor (median), lățimea
    umerilor în planul imaginii (x, y; z-ul MediaPipe e prea zgomotos), înălțimea nasului
    deasupra umerilor și lungimile brațelor (umăr-cot, cot-încheietură) pe fiecare parte,
    din cadrele în care mâna e activă (`active` (T, 2); altfel toate cadrele).
    """
    pose = np.asarray(pose, dtype=np.float32)
    center = (pose[:, 11] + pose[:, 12]) / 2
    m = {'center': np.median(center, axis=0),
         'width': np.median(np.linalg.norm(pose[:, 11, :2] - pose[:, 12, :2], axis=1)),
         'nose': np.median(np.linalg.norm(pose[:, 0, :2] - center[:, :2], axis=1))}
    for k, (S, E, W) in enumerate(((11, 13, 15), (12, 14, 16))):
        rows = pose if active is None or not active[:, k].any() else pose[active[:, k]]
        m[k] = (np.median(np.linalg.norm(rows[:, E, :2] - rows[:, S, :2], axis=1)),
                np.median(np.linalg.norm(rows[:, W, :2] - rows[:, E, :2], axis=1)))
    return m


def with_data_proportions(hand):
    """Mâna (T, 21, 3) cu direcțiile segmentelor păstrate, dar cu lungimile DATA_FINGER_LENGTHS."""
    palm = np.linalg.norm(hand[:, [5, 9, 13, 17]] - hand[:, :1], axis=2).mean(axis=1)[:, None]
    out = hand.copy()
    for chain, lengths in zip(FINGER_CHAINS, DATA_FINGER_LENGTHS):
        for a, b, length in zip(chain[:-1], chain[1:], lengths):
            d = hand[:, b] - hand[:, a]
            out[:, b] = out[:, a] + d / np.linalg.norm(d, axis=1, keepdims=True) * length * palm
    return out


def avatar_to_raw(result, motion):
    """
    Punctele avatarului (scena 3D, metri) -> (T, 1662), ca imaginea unei camere care vede
    avatarul cu proporțiile semnatarului din date. Avatarul are alt corp (umeri mai înguști,
    gât și brațe mai lungi); modelul ar vedea asta ca pe un alt semnatar, nu ca pe o mișcare
    greșită. De aceea:
      - umerii avatarului se suprapun peste cei din date (centru și lățime);
      - nasul și coatele păstrează direcția față de umăr, dar cu distanțele din date; la fel
        segmentele degetelor (direcția avatarului, lungimea din date);
      - încheieturile și mâinile rămân exact unde le pune avatarul (retargeting-ul le raportează
        deja la față / piept / umăr, la scara umerilor);
      - o mână e „detectată” doar dacă încheietura ar fi în cadru la un braț cu lungimea din date
        ținut în același unghi.
    """
    frames = result['frames']
    pose = np.array([f['pose'] for f in frames], dtype=np.float32)
    hands = [with_data_proportions(np.array([f[s + '_hand'] for f in frames], dtype=np.float32))
             for s in ('left', 'right')]
    data_frames = [f for f in motion['frames'] if f['pose'] is not None]
    data = body_measures([f['pose'] for f in data_frames], np.array([f['weight'] for f in data_frames]) >= 1)
    avatar = body_measures(pose)

    scale = data['width'] / avatar['width']   # metri avatar -> unități ale scenei din date
    place = lambda p: (p - avatar['center']) * scale + data['center']
    out = place(pose)
    left, right = (place(h) for h in hands)
    center = (out[:, 11] + out[:, 12]) / 2
    out[:, 0] = center + (out[:, 0] - center) * data['nose'] / (avatar['nose'] * scale)
    visible = []
    for k, ((S, E, W), hand) in enumerate(zip(((11, 13, 15), (12, 14, 16)), (left, right))):
        upper, fore = (d / (a * scale) for d, a in zip(data[k], avatar[k]))
        out[:, E] = out[:, S] + (out[:, E] - out[:, S]) * upper
        # unde ar fi încheietura cu brațul din date, la aceleași unghiuri -> e în cadru?
        wrist = out[:, E] + (out[:, W] - place(pose[:, E])) * fore
        visible.append(scene_to_image(wrist)[:, 1] < HAND_VISIBLE_Y)
        # mâna în afara cadrului (brațul în repaus) nu e raportată la corp de retargeting:
        # o punem unde ar fi cu brațul din date
        hidden = ~visible[-1]
        hand[hidden] += (wrist[hidden] - out[hidden, W])[:, None]
        out[hidden, W] = wrist[hidden]
        for point, weights in POSE_FROM_HAND.items():   # stânga = 17/19/21, dreapta = 18/20/22
            out[:, point + k] = hand[:, 0] + sum(w * (hand[:, i] - hand[:, 0]) for i, w in weights.items())

    out, left, right = (scene_to_image(p) for p in (out, left, right))
    # z-ul punctelor MediaPipe Hands e relativ la încheietură
    left[..., 2] -= left[:, :1, 2]
    right[..., 2] -= right[:, :1, 2]
    return to_raw(list(out), [h if ok else None for h, ok in zip(left, visible[0])],
                  [h if ok else None for h, ok in zip(right, visible[1])])


# ---------------------------------------------------------------- browserul și serverul
def run_backtranslation(words, motion_dir):
    """Pornește serverul și browserul headless; întoarce {cuvânt: punctele avatarului}."""
    results, done, errors = {}, threading.Event(), []

    class Handler(AvatarHandler):
        def do_POST(self):
            name = self.path.rsplit('/', 1)[-1]
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if name == '_done':
                done.set()
            elif name == '_error':
                errors.append(body['error'])
                done.set()
            else:
                results[name] = body
                print(f"  {name:<11} {len(body['frames'])} cadre", flush=True)
            self.send_response(204)
            self.end_headers()

    Handler.motion_dir = Path(motion_dir)
    Handler.extensions_map.update({'.js': 'text/javascript', '.vrm': 'model/gltf-binary'})
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Handler, directory=str(WEB_DIR)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/?backtranslate={','.join(words)}"
    with tempfile.TemporaryDirectory() as profile:
        browser = subprocess.Popen([find_browser(), '--headless=new', '--no-first-run', '--disable-extensions',
                                    '--enable-unsafe-swiftshader', f'--user-data-dir={profile}', url],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            finished = done.wait(TIMEOUT_SECONDS)
        finally:
            browser.terminate()
            browser.wait(10)
            server.shutdown()
    if errors:
        raise SystemExit('Eroare în pagina avatarului:\n' + errors[0])
    if not finished:
        raise SystemExit(f'Pagina nu a terminat în {TIMEOUT_SECONDS} s ({len(results)}/{len(words)} cuvinte).')
    return results


# ---------------------------------------------------------------- evaluare
def classify(model, raws):
    X = fit_inputs(model, np.stack([preprocess(r) for r in raws]))
    return model.predict(X, verbose=0)


def describe(probs, word):
    order = np.argsort(probs)[::-1]
    rank = int(np.where(order == LABEL_MAP[word])[0][0]) + 1
    return {'pred': ACTIONS[order[0]], 'pred_conf': float(probs[order[0]]),
            'target_conf': float(probs[LABEL_MAP[word]]), 'rank': rank}


def summary(rows, prefix):
    n = len(rows)
    top1 = sum(int(r[prefix + 'rank']) == 1 for r in rows)
    top3 = sum(int(r[prefix + 'rank']) <= 3 for r in rows)
    mean = np.mean([float(r[prefix + 'target_conf']) for r in rows]) * 100
    return f"top-1 {top1}/{n} ({100 * top1 / n:.1f}%), top-3 {top3}/{n} ({100 * top3 / n:.1f}%), încredere medie în semnul corect {mean:.1f}%"


def print_table(rows):
    print(f"\n{'Cuvânt':<11} {'Date':>6}   {'Avatar: prezis':<16} {'încr.':>6} {'corect':>7}  Status")
    for r in rows:
        status = 'Recunoscut' if r['avatar_rank'] == 1 else f"Greșit (corectul pe locul {r['avatar_rank']})"
        print(f"{r['word']:<11} {100 * r['data_target_conf']:5.0f}%   {r['avatar_pred']:<16} "
              f"{100 * r['avatar_pred_conf']:5.0f}% {100 * r['avatar_target_conf']:6.0f}%  {status}")
    print(f"\nDate:   {summary(rows, 'data_')}")
    print(f"Avatar: {summary(rows, 'avatar_')}")


def print_comparison(rows, baseline_file):
    with open(baseline_file, encoding='utf-8') as f:
        before = {r['word']: r for r in csv.DictReader(f)}
    common = [r for r in rows if r['word'] in before]
    print(f"\nÎnainte ({Path(baseline_file).stem}) vs după, avatar, încrederea în semnul corect:")
    print(f"{'Cuvânt':<11} {'înainte':>16}   {'după':>16}   schimbare")
    for r in common:
        b = before[r['word']]
        ok_b, ok_a = int(b['avatar_rank']) == 1, r['avatar_rank'] == 1
        change = 'REPARAT' if ok_a and not ok_b else 'STRICAT' if ok_b and not ok_a else ''
        fmt = lambda conf, pred, ok: f"{100 * float(conf):4.0f}% {'ok' if ok else '-> ' + pred:<10}"
        print(f"{r['word']:<11} {fmt(b['avatar_target_conf'], b['avatar_pred'], ok_b):>16}   "
              f"{fmt(r['avatar_target_conf'], r['avatar_pred'], ok_a):>16}   {change}")
    print(f"\nÎnainte: {summary([before[r['word']] for r in common], 'avatar_')}")
    print(f"După:    {summary(common, 'avatar_')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--words', nargs='+', choices=ACTIONS, metavar='CUVÂNT')
    parser.add_argument('--motion-dir', default=str(AVATAR_MOTION_DIR))
    parser.add_argument('--label', default='rezultat', help='numele fișierului data/avatar_eval/<label>.csv')
    parser.add_argument('--compare', help='CSV al unei evaluări anterioare, pentru tabelul înainte/după')
    args = parser.parse_args()

    motion_dir = Path(args.motion_dir)
    index = json.loads((motion_dir / 'index.json').read_text(encoding='utf-8'))
    words = [w for w in (args.words or ACTIONS) if w in index]
    print(f"Redau {len(words)} semne cu avatarul (browser headless)...")
    results = run_backtranslation(words, motion_dir)

    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)
    motions = {w: json.loads((motion_dir / f'{w}.json').read_text(encoding='utf-8')) for w in words}
    data_probs = classify(model, [motion_to_raw(motions[w]) for w in words])
    avatar_probs = classify(model, [avatar_to_raw(results[w], motions[w]) for w in words])

    rows = []
    for w, dp, ap in zip(words, data_probs, avatar_probs):
        d, a = describe(dp, w), describe(ap, w)
        rows.append({'word': w, 'source': motions[w].get('source', ''),
                     **{'data_' + k: v for k, v in d.items()}, **{'avatar_' + k: v for k, v in a.items()}})
    print_table(rows)

    AVATAR_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = AVATAR_EVAL_DIR / f'{args.label}.csv'
    with open(out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSalvat: {out}")
    if args.compare:
        print_comparison(rows, args.compare)


if __name__ == '__main__':
    main()
