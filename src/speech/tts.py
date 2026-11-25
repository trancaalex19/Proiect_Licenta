"""
Text -> voce (sensul pacient -> medic): propoziția semnată de pacient, citită cu voce tare
pentru medic. Piper (rhasspy, ONNX, rulează local și pe Raspberry Pi), cu singura voce
românească disponibilă, ro_RO-mihai-medium (rhasspy/piper-voices, 63 MB, în models/piper/).

    python -m src.speech.tts --download              # o singură dată: descarcă vocea
    python -m src.speech.tts "Mă doare capul."        # sintetizează, redă și afișează timpii
    python -m src.speech.tts "Mă doare capul." --out data/speech_recordings/tts.wav
"""
import argparse
import queue
import threading
import time

import numpy as np

from src.config import TTS_LENGTH_SCALE, TTS_MODEL_DIR, TTS_VOICE, TTS_VOLUME


def voice_path(voice=TTS_VOICE, model_dir=TTS_MODEL_DIR):
    return model_dir / f'{voice}.onnx'


class PiperTTS:
    """Vocea Piper încărcată o dată; synthesize(text) -> audio int16 mono la `sample_rate`."""

    def __init__(self, voice=TTS_VOICE, model_dir=TTS_MODEL_DIR):
        from piper import PiperVoice, SynthesisConfig
        path = voice_path(voice, model_dir)
        if not path.exists():
            raise RuntimeError(f'lipsește vocea {path.as_posix()} (rulează python -m src.speech.tts --download)')
        self.voice = PiperVoice.load(path)
        self.sample_rate = self.voice.config.sample_rate
        self._config = SynthesisConfig(length_scale=TTS_LENGTH_SCALE, volume=TTS_VOLUME)

    def synthesize(self, text):
        chunks = [chunk.audio_int16_array for chunk in self.voice.synthesize(text, self._config)]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)


class SpeechOutput:
    """
    Firul vocii: ia textele din coadă, le sintetizează și le redă prin `play(samples, rate)`
    (în prototip SpeechRecorder.play: microfonul e oprit cât timp se aude vocea, ca aplicația
    să nu se înregistreze singură drept medic). Interfața nu așteaptă după el.
    """

    def __init__(self, tts, play):
        self.tts, self._play = tts, play
        self.speaking = False
        self._texts = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def say(self, text):
        if text.strip():
            self._texts.put(text)

    @property
    def busy(self):
        return self.speaking or not self._texts.empty()

    def _run(self):
        self.tts.synthesize('Bună ziua.')       # încălzire: prima sinteză durează ~3 s, apoi ~0.1 s
        while not self._stop.is_set():
            try:
                text = self._texts.get(timeout=0.1)
            except queue.Empty:
                continue
            self.speaking = True
            try:
                self._play(self.tts.synthesize(text), self.tts.sample_rate)
            except Exception as error:          # placa audio ocupată etc.: conversația merge mai departe
                print(f"EROARE la redarea vocii: {error}")
            finally:
                self.speaking = False


def download(voice=TTS_VOICE, model_dir=TTS_MODEL_DIR):
    from piper.download_voices import download_voice
    model_dir.mkdir(parents=True, exist_ok=True)
    download_voice(voice, model_dir)
    print(f"Vocea e în {voice_path(voice, model_dir).as_posix()}")


def main():
    parser = argparse.ArgumentParser(description='Text -> voce (Piper, română)')
    parser.add_argument('text', nargs='*')
    parser.add_argument('--download', action='store_true', help=f'descarcă vocea {TTS_VOICE}')
    parser.add_argument('--out', help='salvează WAV în loc să redea')
    args = parser.parse_args()
    if args.download:
        download()
    if not args.text:
        return

    started = time.perf_counter()
    tts = PiperTTS()
    loaded = time.perf_counter()
    text = ' '.join(args.text)
    samples = tts.synthesize(text)
    done = time.perf_counter()
    seconds = len(samples) / tts.sample_rate
    print(f"încărcare {loaded - started:.2f} s, sinteză {done - loaded:.2f} s pentru {seconds:.1f} s de voce "
          f"(RTF {(done - loaded) / max(seconds, 1e-6):.2f})")
    if args.out:
        from pathlib import Path
        from src.speech.audio import save_wav
        save_wav(Path(args.out), samples.tobytes(), tts.sample_rate)
        print(f"salvat în {args.out}")
    else:
        import sounddevice as sd
        sd.play(samples, tts.sample_rate)
        sd.wait()


if __name__ == '__main__':
    main()
