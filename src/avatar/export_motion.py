"""
Exportă mișcarea fiecărui semn pentru avatar: data/avatar_motion/<cuvânt>.json + index.json.

Două metode (exemplele făcute cu mâna greșită, lista 'mirrored', sunt excluse la amândouă):
  dba  (implicit) mișcarea tipică din TOATE înregistrările proprii ale cuvântului: aliniere în
       timp cu DTW și mediană cadru cu cadru (src/avatar/canonical.py); dispar cadrele în care
       MediaPipe a pierdut mâna sau a greșit degetele într-o singură înregistrare;
  best o singură înregistrare: cea pe care modelul de recunoaștere o clasifică cel mai sigur.

    python -m src.avatar.export_motion                 # toate cele 51 de cuvinte, DBA
    python -m src.avatar.export_motion --method best --out alt/folder
    python -m src.avatar.export_motion --words hurt yes
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from src.app.sign_info import SIGN_INFO
from src.avatar.canonical import canonical
from src.avatar.motion import raw_to_motion
from src.config import ACTIONS, AVATAR_MOTION_DIR, LABEL_MAP, MODEL_PATH, SEQUENCE_LENGTH
from src.data.record_samples import record_seconds
from src.features.preprocessing import preprocess
from src.ml.dataset import mirrored_files, recorded_files
from src.ml.model import fit_inputs


def confidences(model, raws, action):
    """Încrederea modelului în `action` pentru fiecare înregistrare brută."""
    X = fit_inputs(model, np.stack([preprocess(r) for r in raws]))
    return model.predict(X, verbose=0)[:, LABEL_MAP[action]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--words', nargs='+', default=ACTIONS, choices=ACTIONS, metavar='CUVÂNT')
    parser.add_argument('--method', choices=['dba', 'best'], default='dba')
    parser.add_argument('--out', default=str(AVATAR_MOTION_DIR), help='folderul de ieșire')
    args = parser.parse_args()

    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)
    mirrored = mirrored_files()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_file = out_dir / 'index.json'
    index = json.loads(index_file.read_text(encoding='utf-8')) if index_file.exists() else {}
    for action in args.words:
        paths = [p for p in recorded_files(action) if (action, os.path.basename(p)) not in mirrored]
        if not paths:
            print(f"{action:<11} SĂRIT: nicio înregistrare proprie")
            continue
        raws = [np.load(p) for p in paths]
        if args.method == 'best':
            conf = confidences(model, raws, action)
            best = int(np.argmax(conf))
            raw, source, note = raws[best], os.path.basename(paths[best]), ''
        else:
            raw, report = canonical(raws, [os.path.basename(p) for p in paths])
            source = f"DBA din {report['used']} înregistrări (bază {report['base']})"
            note = f"  excluse: {', '.join(report['excluded'])}" if report['excluded'] else ''
        confidence = float(confidences(model, [raw], action)[0])
        # Înregistrările au durat record_seconds și au fost eșantionate uniform la SEQUENCE_LENGTH cadre
        motion = raw_to_motion(raw, fps=SEQUENCE_LENGTH / record_seconds(action))
        motion.update(word=action, meaning=SIGN_INFO[action]['meaning'], method=args.method,
                      source=source, confidence=round(confidence, 3))
        (out_dir / f'{action}.json').write_text(json.dumps(motion, ensure_ascii=False), encoding='utf-8')
        index[action] = {'meaning': motion['meaning'], 'method': args.method, 'source': source,
                         'confidence': motion['confidence'], 'frames': len(motion['frames'])}
        print(f"{action:<11} {source}  încredere {confidence:.2f}  {len(motion['frames'])} cadre{note}")

    ordered = {a: index[a] for a in ACTIONS if a in index}   # în ordinea vocabularului
    index_file.write_text(json.dumps(ordered, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"\n{len(ordered)} semne în {out_dir}")


if __name__ == '__main__':
    main()
