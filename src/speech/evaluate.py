"""
Evaluarea motoarelor de transcriere pe setul de test: fiecare data/test_doctor/<nume>.wav
are lângă el <nume>.txt cu textul corect (făcute cu src/speech/record_test_set.py).
Fiecare WAV trece prin Groq și prin modelul local; rezultatul e un CSV cu textul
fiecărui motor, WER-ul și timpul de procesare, plus mediile la final.

WER = (cuvinte înlocuite + șterse + inserate) / cuvintele din textul corect, calculat după
normalizare (litere mici, fără punctuație, ş/ţ cu sedilă = ș/ț cu virgulă), ca o virgulă
sau o majusculă să nu conteze ca greșeală. O transcriere eșuată are WER 1 (tot textul lipsă).

Rulare (din rădăcina proiectului):
    python -m src.speech.evaluate                          # Groq + local
    python -m src.speech.evaluate --engines local          # doar local (ex. pe Raspberry Pi, offline)
    python -m src.speech.evaluate --out data/speech_eval_pi.csv
"""
import argparse
import csv
import platform
import re
import statistics
import time
from pathlib import Path

import jiwer

from src.config import (GROQ_RATE_LIMIT_RETRIES, GROQ_RATE_LIMIT_WAIT_SECONDS, SPEECH_EVAL_FILE,
                        SPEECH_TEST_DIR, WHISPER_CPU_THREADS)

ENGINES = ['groq', 'local']
# variantele cu sedilă (din codificări vechi) devin cele corecte, cu virgulă
DIACRITIC_FIX = str.maketrans({'ş': 'ș', 'Ş': 'Ș', 'ţ': 'ț', 'Ţ': 'Ț'})


def normalize(text):
    """'Arătați-mi unde vă doare.' -> 'arătați mi unde vă doare'"""
    text = text.translate(DIACRITIC_FIX).lower()
    text = re.sub(r'[^\w\s]', ' ', text)            # punctuație și cratimă -> spațiu
    return ' '.join(text.split())


def load_test_set():
    """Perechile (wav, text corect), sortate după nume."""
    pairs = []
    for wav in sorted(SPEECH_TEST_DIR.glob('*.wav')):
        reference = wav.with_suffix('.txt')
        if reference.exists():
            pairs.append((wav, reference.read_text(encoding='utf-8').strip()))
        else:
            print(f"Atenție: {wav.name} nu are {reference.name}, îl sar.")
    return pairs


def make_engine(name):
    if name == 'groq':
        from src.speech.groq_engine import GroqEngine
        return GroqEngine()
    from src.speech.local_engine import LocalEngine
    return LocalEngine()


def transcribe(engine, wav_bytes):
    """Ca engine.transcribe, dar la limita Groq (429) așteaptă și reîncearcă (aici nu e timp real)."""
    for _ in range(GROQ_RATE_LIMIT_RETRIES):
        result = engine.transcribe(wav_bytes)
        if result.error != 'HTTP 429':
            return result
        print(f"   limita Groq (429): aștept {GROQ_RATE_LIMIT_WAIT_SECONDS} s...")
        time.sleep(GROQ_RATE_LIMIT_WAIT_SECONDS)
    return result


def wer(reference, hypothesis):
    return jiwer.wer(normalize(reference), normalize(hypothesis) if hypothesis else '')


def main():
    parser = argparse.ArgumentParser(description='Evaluare Groq / local pe data/test_doctor/')
    parser.add_argument('--engines', choices=ENGINES, nargs='+', default=ENGINES)
    parser.add_argument('--out', type=Path, default=SPEECH_EVAL_FILE)
    args = parser.parse_args()

    pairs = load_test_set()
    if not pairs:
        print(f"EROARE: niciun WAV cu text corect în {SPEECH_TEST_DIR.as_posix()}/ "
              "(înregistrează cu: python -m src.speech.record_test_set)")
        return
    print(f"{len(pairs)} fraze de test; motoare: {', '.join(args.engines)}; "
          f"{platform.system()} {platform.machine()}, cpu_threads={WHISPER_CPU_THREADS}")

    engines = {}
    for name in args.engines:
        try:
            engines[name] = make_engine(name)
        except RuntimeError as error:
            print(f"[{name}] EROARE: {error}")
    if 'local' in engines:                          # prima rulare e mai lentă: nu intră în timp
        engines['local'].transcribe(pairs[0][0].read_bytes())

    rows = []
    for wav, reference in pairs:
        wav_bytes = wav.read_bytes()
        row = {'fisier': wav.name, 'referinta': reference}
        for name, engine in engines.items():
            result = transcribe(engine, wav_bytes)
            row['durata_audio_s'] = round(result.audio_seconds, 2)
            row[f'text_{name}'] = result.text if result.ok else ''
            row[f'wer_{name}'] = round(wer(reference, row[f'text_{name}']), 3)
            row[f'timp_{name}_s'] = round(result.processing_seconds, 2)
            row[f'eroare_{name}'] = result.error or ''
        rows.append(row)
        print(f"{wav.name} „{reference}”")
        for name in engines:
            print(f"   {name:5s} WER {row[f'wer_{name}']:.2f}  {row[f'timp_{name}_s']:5.2f} s  "
                  f"{row[f'text_{name}'] or '[' + row[f'eroare_{name}'] + ']'}")

    # mediile: WER total (toate greșelile / toate cuvintele) și media WER pe frază
    summary = {}
    for name in engines:
        hypotheses = [normalize(r[f'text_{name}']) for r in rows]
        summary[name] = {
            'wer_total': jiwer.wer([normalize(r['referinta']) for r in rows], hypotheses),
            'wer_medie': statistics.mean(r[f'wer_{name}'] for r in rows),
            'timp_mediu': statistics.mean(r[f'timp_{name}_s'] for r in rows),
            'rtf': sum(r[f'timp_{name}_s'] for r in rows) / sum(r['durata_audio_s'] for r in rows),
            'erori': sum(bool(r[f'eroare_{name}']) for r in rows),
        }

    columns = ['fisier', 'durata_audio_s', 'referinta']
    for name in engines:
        columns += [f'text_{name}', f'wer_{name}', f'timp_{name}_s', f'eroare_{name}']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8-sig') as f:   # utf-8-sig: diacritice corecte în Excel
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        for label, key in (('MEDIE (pe frază)', 'wer_medie'), ('WER TOTAL', 'wer_total')):
            final = {'fisier': label, 'durata_audio_s': round(statistics.mean(r['durata_audio_s'] for r in rows), 2)}
            for name, s in summary.items():
                final[f'wer_{name}'] = round(s[key], 3)
                final[f'timp_{name}_s'] = round(s['timp_mediu'], 2)
            writer.writerow(final)

    print(f"\nRezultate: {args.out.as_posix()}")
    for name, s in summary.items():
        print(f"{name:5s}  WER total {s['wer_total']:.1%}  (medie pe frază {s['wer_medie']:.1%})  "
              f"timp mediu {s['timp_mediu']:.2f} s  RTF {s['rtf']:.2f}  erori {s['erori']}/{len(rows)}")


if __name__ == '__main__':
    main()
