"""
Motorul online: Whisper large-v3 prin Groq API. Cheia se citește din .env (GROQ_API_KEY).
La orice eroare (timeout, limită 429, fără rețea) NU aruncă excepție: întoarce un
Transcript cu `error` completat, iar router-ul trece pe modelul local.
"""
import os
import time

import groq
from dotenv import load_dotenv

from src.config import (ENV_FILE, GROQ_MODEL, GROQ_RESPONSE_FORMAT, GROQ_TIMEOUT_SECONDS,
                        SPEECH_LANGUAGE, SPEECH_TEMPERATURE)
from src.speech.audio import wav_duration
from src.speech.transcript import Transcript, check_hallucination, confidence_from_segments
from src.speech.vocabulary import build_prompt


def describe_error(error):
    """Motivul erorii, scurt, pentru jurnal."""
    if isinstance(error, groq.APITimeoutError):
        return 'timeout'
    if isinstance(error, groq.APIConnectionError):
        return 'fără conexiune'
    if isinstance(error, groq.APIStatusError):
        return f'HTTP {error.status_code}'          # ex. 429 = limita planului gratuit
    return type(error).__name__


class GroqEngine:
    name = 'groq'

    def __init__(self, prompt=None):
        load_dotenv(ENV_FILE)
        api_key = os.environ.get('GROQ_API_KEY')
        if not api_key:
            raise RuntimeError(f"Lipsește GROQ_API_KEY în {ENV_FILE.as_posix()} (vezi .env.example).")
        # max_retries=0: la eroare nu reîncercăm, trecem imediat pe modelul local
        self._client = groq.Groq(api_key=api_key, timeout=GROQ_TIMEOUT_SECONDS, max_retries=0)
        self.prompt = build_prompt() if prompt is None else prompt

    def transcribe(self, wav_bytes):
        audio_seconds = wav_duration(wav_bytes)
        start = time.perf_counter()
        try:
            response = self._client.audio.transcriptions.create(
                file=('fraza.wav', wav_bytes, 'audio/wav'),
                model=GROQ_MODEL,
                language=SPEECH_LANGUAGE,
                temperature=SPEECH_TEMPERATURE,
                response_format=GROQ_RESPONSE_FORMAT,
                prompt=self.prompt,
            )
        except groq.APIError as error:
            return Transcript('', None, self.name, audio_seconds, time.perf_counter() - start,
                              error=describe_error(error))
        # cu verbose_json, răspunsul are și segmentele, cu avg_logprob pentru încredere
        segments = getattr(response, 'segments', None)
        return check_hallucination(Transcript(response.text.strip(), confidence_from_segments(segments),
                                              self.name, audio_seconds, time.perf_counter() - start))
