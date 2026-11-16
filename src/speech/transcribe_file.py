"""
Transcrie un fișier WAV cu un motor anume, pentru testare.
Fără fișier dat, folosește ultima frază înregistrată (data/speech_recordings/).

Rulare (din rădăcina proiectului):
    python -m src.speech.transcribe_file                         # ultimul WAV, cu Groq
    python -m src.speech.transcribe_file cale/fraza.wav --engine local
    python -m src.speech.transcribe_file --engine groq local      # compară motoarele pe același WAV
    python -m src.speech.transcribe_file --show-prompt           # afișează prompt-ul cu vocabularul
"""
import argparse
import time
from pathlib import Path

from src.config import SPEECH_RECORDINGS_DIR, WHISPER_HF_MODEL
from src.speech.vocabulary import build_prompt

ENGINES = ['groq', 'local']
PROMPT_MAX_TOKENS = 224   # limita Whisper pentru prompt


def make_engine(name):
    if name == 'groq':
        from src.speech.groq_engine import GroqEngine
        return GroqEngine()
    if name == 'local':
        from src.speech.local_engine import LocalEngine
        return LocalEngine()
    raise ValueError(f'Motor necunoscut: {name}')


def show_prompt():
    """Prompt-ul și câți tokeni are (limita Whisper: 224). Numărarea cere transformers (requirements-dev)."""
    prompt = build_prompt()
    print(prompt)
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return
    for repo in (WHISPER_HF_MODEL, 'openai/whisper-large-v3'):
        tokens = len(AutoTokenizer.from_pretrained(repo)(prompt, add_special_tokens=False)['input_ids'])
        print(f"{repo}: {tokens} tokeni din {PROMPT_MAX_TOKENS}")


def latest_recording():
    recordings = sorted(SPEECH_RECORDINGS_DIR.glob('*.wav'))
    return recordings[-1] if recordings else None


def main():
    parser = argparse.ArgumentParser(description='Transcrie un fișier WAV')
    parser.add_argument('wav', nargs='?', type=Path, help='implicit: ultima frază înregistrată')
    parser.add_argument('--engine', choices=ENGINES, nargs='+', default=['groq'])
    parser.add_argument('--show-prompt', action='store_true', help='afișează prompt-ul și iese')
    args = parser.parse_args()

    if args.show_prompt:
        show_prompt()
        return
    path = args.wav or latest_recording()
    if path is None or not path.exists():
        print(f"EROARE: nu găsesc fișierul WAV ({path or SPEECH_RECORDINGS_DIR.as_posix() + '/ e gol'}).")
        return

    wav_bytes = path.read_bytes()
    print(f"Fișier:    {path.as_posix()}")
    for name in args.engine:
        load_start = time.perf_counter()
        try:
            engine = make_engine(name)
        except RuntimeError as error:
            print(f"\n[{name}] EROARE: {error}")
            continue
        load_seconds = time.perf_counter() - load_start
        result = engine.transcribe(wav_bytes)

        print(f"\n[{result.engine}] audio {result.audio_seconds:.1f} s, încărcare motor {load_seconds:.1f} s")
        if result.ok:
            confidence = '-' if result.confidence is None else f'{result.confidence:.2f}'
            print(f"Text:      {result.text}")
            print(f"Încredere: {confidence}")
        else:
            print(f"EROARE:    {result.error}")
        print(f"Procesare: {result.processing_seconds:.2f} s")


if __name__ == '__main__':
    main()
