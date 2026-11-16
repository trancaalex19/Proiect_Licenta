"""
Convertește modelul readerbench/whisper-ro (Hugging Face, PyTorch) în formatul
CTranslate2 int8 folosit de faster-whisper, în models/whisper-ro-ct2/.

Se rulează o singură dată, pe laptop: conversia cere torch și transformers
(requirements-dev.txt). Pe Raspberry Pi se copiază doar folderul rezultat.

Rulare (din rădăcina proiectului):
    python -m src.speech.convert_model            # descarcă (~1 GB, o dată) și convertește
    python -m src.speech.convert_model --force    # refă conversia peste cea existentă
"""
import argparse

from src.config import WHISPER_COMPUTE_TYPE, WHISPER_HF_MODEL, WHISPER_LOCAL_DIR

# Fișiere copiate lângă model: tokenizer-ul și setările spectrogramei, citite de faster-whisper
COPY_FILES = ['tokenizer.json', 'preprocessor_config.json']


def folder_size_mb(path):
    return sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / 1e6


def main():
    parser = argparse.ArgumentParser(description=f'Conversie {WHISPER_HF_MODEL} -> CTranslate2')
    parser.add_argument('--force', action='store_true', help='suprascrie conversia existentă')
    args = parser.parse_args()

    if (WHISPER_LOCAL_DIR / 'model.bin').exists() and not args.force:
        print(f"Modelul există deja în {WHISPER_LOCAL_DIR.as_posix()}/ (--force ca să-l refaci).")
        return
    try:
        from ctranslate2.converters import TransformersConverter
        converter = TransformersConverter(WHISPER_HF_MODEL, copy_files=COPY_FILES)
    except ImportError as error:
        print(f"EROARE: {error}. Conversia cere: pip install -r requirements-dev.txt")
        return

    print(f"Convertim {WHISPER_HF_MODEL} -> {WHISPER_LOCAL_DIR.as_posix()}/ ({WHISPER_COMPUTE_TYPE})...")
    converter.convert(str(WHISPER_LOCAL_DIR), quantization=WHISPER_COMPUTE_TYPE, force=args.force)
    print(f"Gata: {folder_size_mb(WHISPER_LOCAL_DIR):.0f} MB în {WHISPER_LOCAL_DIR.as_posix()}/")


if __name__ == '__main__':
    main()
