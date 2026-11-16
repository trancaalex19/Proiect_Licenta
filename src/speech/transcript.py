"""
Rezultatul comun al motoarelor de transcriere (Groq online, faster-whisper local):
ambele au `transcribe(wav_bytes) -> Transcript`, deci router-ul și evaluarea le
tratează la fel.
"""
import math
import zlib
from dataclasses import dataclass

from src.config import HALLUCINATION_COMPRESSION_RATIO


@dataclass
class Transcript:
    text: str
    confidence: float | None    # 0..1 (din probabilitățile modelului); None = necunoscută
    engine: str                 # 'groq' sau 'local'
    audio_seconds: float        # cât durează fraza
    processing_seconds: float   # cât a durat transcrierea
    error: str | None = None    # motivul, dacă transcrierea a eșuat

    @property
    def ok(self):
        return self.error is None


def _field(segment, name):
    """Segmentele vin ca dicționare (Groq) sau ca obiecte (faster-whisper)."""
    return segment[name] if isinstance(segment, dict) else getattr(segment, name)


def confidence_from_segments(segments):
    """
    Încrederea frazei: media lui exp(avg_logprob) pe segmente, ponderată cu durata lor.
    avg_logprob = media log-probabilității tokenilor; 0 = sigur, sub -0.5 = probabil greșit.
    """
    total = weight = 0.0
    for segment in segments or []:
        duration = max(_field(segment, 'end') - _field(segment, 'start'), 1e-3)
        total += math.exp(_field(segment, 'avg_logprob')) * duration
        weight += duration
    return round(total / weight, 3) if weight else None


def compression_ratio(text):
    """Cât de mult se comprimă textul (ca în Whisper): bucle de tip „a fost... a fost...” au valori mari."""
    data = text.encode('utf-8')
    return len(data) / len(zlib.compress(data)) if data else 0.0


def check_hallucination(transcript):
    """Textul repetitiv (halucinație) devine eroare: îl păstrăm pentru jurnal, dar nu e folosit."""
    if transcript.ok and compression_ratio(transcript.text) > HALLUCINATION_COMPRESSION_RATIO:
        transcript.error = 'text repetitiv (halucinație)'
    return transcript
