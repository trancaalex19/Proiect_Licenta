"""
Router-ul pentru vorbire -> text: alege motorul pentru fiecare frază.

  - 'auto': Groq când există internet, altfel modelul local. Dacă Groq dă orice eroare
    (timeout, limită 429, fără rețea, text repetitiv), aceeași frază merge pe modelul local.
  - 'offline_only': doar modelul local.

Internetul se verifică în fundal, la INTERNET_CHECK_SECONDS, nu la fiecare frază.
Modelul local se încarcă mereu, la pornire: e baza sistemului, care trebuie să meargă
100% fără internet. Fiecare încercare (și cele eșuate) intră în jurnalul CSV.
"""
import csv
import queue
import socket
import threading
from datetime import datetime

from src.config import (INTERNET_CHECK_HOST, INTERNET_CHECK_PORT, INTERNET_CHECK_SECONDS,
                        INTERNET_CHECK_TIMEOUT, SPEECH_LOG_FILE, SPEECH_MODE, USE_GROQ)
from src.speech.local_engine import LocalEngine

LOG_COLUMNS = ['timestamp', 'motor', 'durata_audio_s', 'durata_procesare_s', 'text', 'incredere', 'eroare']
# erori Groq după care considerăm internetul căzut până la următoarea verificare
CONNECTION_ERRORS = ('timeout', 'fără conexiune')


def internet_available():
    """O conexiune TCP scurtă la serverul Groq: rapid și fără să consume din limita API."""
    try:
        with socket.create_connection((INTERNET_CHECK_HOST, INTERNET_CHECK_PORT),
                                      timeout=INTERNET_CHECK_TIMEOUT):
            return True
    except OSError:
        return False


class SpeechRouter:
    def __init__(self, mode=SPEECH_MODE, use_groq=USE_GROQ, log_file=SPEECH_LOG_FILE):
        if mode not in ('auto', 'offline_only'):
            raise ValueError("mode trebuie să fie 'auto' sau 'offline_only'")
        self.mode = mode
        self.local = LocalEngine()
        self.groq = None
        self.groq_status = 'oprit (offline_only)' if mode == 'offline_only' else 'oprit (USE_GROQ = False)'
        if mode == 'auto' and use_groq:
            try:
                from src.speech.groq_engine import GroqEngine
                self.groq = GroqEngine()
                self.groq_status = 'activ'
            except RuntimeError as error:          # ex. lipsește cheia: mergem doar local
                self.groq_status = f'indisponibil: {error}'
        self.online = False
        self.last_fallback = None                  # de ce ultima frază n-a mers pe Groq
        self._log_file = log_file
        self._log_lock = threading.Lock()
        self._stop = threading.Event()
        if self.groq is not None:
            self.online = internet_available()
            threading.Thread(target=self._check_loop, daemon=True).start()

    def _check_loop(self):
        while not self._stop.wait(INTERNET_CHECK_SECONDS):
            self.online = internet_available()

    def close(self):
        self._stop.set()

    @property
    def groq_active(self):
        return self.groq is not None and self.online

    def transcribe(self, wav_bytes):
        """Transcrie o frază (WAV): Groq dacă se poate, altfel (sau la eroare) local."""
        self.last_fallback = None
        if self.groq_active:
            result = self.groq.transcribe(wav_bytes)
            self._log(result)
            if result.ok:
                return result
            self.last_fallback = result.error
            if result.error in CONNECTION_ERRORS:
                self.online = False                # nu mai așteptăm Groq până la verificarea următoare
        result = self.local.transcribe(wav_bytes)
        self._log(result)
        return result

    def _log(self, result):
        row = [datetime.now().isoformat(timespec='seconds'), result.engine,
               f'{result.audio_seconds:.2f}', f'{result.processing_seconds:.2f}', result.text,
               '' if result.confidence is None else result.confidence, result.error or '']
        with self._log_lock:
            is_new = not self._log_file.exists()
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            # utf-8-sig la creare: Excel afișează corect diacriticele
            with open(self._log_file, 'a', newline='', encoding='utf-8-sig' if is_new else 'utf-8') as f:
                writer = csv.writer(f)
                if is_new:
                    writer.writerow(LOG_COLUMNS)
                writer.writerow(row)


class TranscriptionWorker:
    """
    Firul de transcriere: ia frazele din coada microfonului (SpeechRecorder.utterances),
    le trece prin router și pune rezultatele în `results`. Interfața nu așteaptă după el.
    """

    def __init__(self, router, utterances):
        self.router = router
        self.results = queue.Queue()
        self.busy = False
        self._utterances = utterances
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    @property
    def waiting(self):
        """Câte fraze așteaptă transcrierea (dacă modelul local e mai lent decât vorbirea)."""
        return self._utterances.qsize()

    def _run(self):
        while not self._stop.is_set():
            try:
                utterance = self._utterances.get(timeout=0.1)
            except queue.Empty:
                continue
            self.busy = True
            result = self.router.transcribe(utterance.wav_bytes())
            self.results.put((result, self.router.last_fallback))
            self.busy = False
