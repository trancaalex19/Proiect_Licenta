"""
Motorul offline: faster-whisper cu readerbench/whisper-ro convertit în CTranslate2 int8
(models/whisper-ro-ct2/, făcut de src/speech/convert_model.py). Merge fără internet,
pe laptop și pe Raspberry Pi. Modelul se încarcă o singură dată, la crearea motorului.
"""
import io
import time

from faster_whisper import WhisperModel

from src.config import (SPEECH_LANGUAGE, SPEECH_TEMPERATURE, WHISPER_BEAM_SIZE,
                        WHISPER_COMPUTE_TYPE, WHISPER_CPU_THREADS, WHISPER_DEVICE,
                        WHISPER_LOCAL_DIR, WHISPER_VAD_FILTER)
from src.speech.audio import wav_duration
from src.speech.transcript import Transcript, check_hallucination, confidence_from_segments
from src.speech.vocabulary import build_prompt


class LocalEngine:
    name = 'local'

    def __init__(self, prompt=None):
        if not (WHISPER_LOCAL_DIR / 'model.bin').exists():
            raise RuntimeError(f"Lipsește modelul local din {WHISPER_LOCAL_DIR.as_posix()}/. "
                               "Rulează: python -m src.speech.convert_model")
        self._model = WhisperModel(str(WHISPER_LOCAL_DIR), device=WHISPER_DEVICE,
                                   compute_type=WHISPER_COMPUTE_TYPE, cpu_threads=WHISPER_CPU_THREADS)
        self.prompt = build_prompt() if prompt is None else prompt

    def transcribe(self, wav_bytes):
        audio_seconds = wav_duration(wav_bytes)
        start = time.perf_counter()
        try:
            segments, _ = self._model.transcribe(
                io.BytesIO(wav_bytes),             # decodat și adus la 16 kHz de faster-whisper
                language=SPEECH_LANGUAGE,
                beam_size=WHISPER_BEAM_SIZE,
                temperature=SPEECH_TEMPERATURE,
                initial_prompt=self.prompt,
                vad_filter=WHISPER_VAD_FILTER,
            )
            segments = list(segments)              # transcrierea propriu-zisă se face aici
        except Exception as error:                 # orice problemă: întoarcem motivul, nu oprim aplicația
            return Transcript('', None, self.name, audio_seconds, time.perf_counter() - start,
                              error=f'{type(error).__name__}: {error}')
        text = ' '.join(segment.text.strip() for segment in segments).strip()
        return check_hallucination(Transcript(text, confidence_from_segments(segments), self.name,
                                              audio_seconds, time.perf_counter() - start))
