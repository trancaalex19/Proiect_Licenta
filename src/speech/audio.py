"""
Captura audio pentru partea vorbire -> text: ascultă microfonul continuu și
decupează câte o frază (ce spune medicul) cu detecția vocii (webrtcvad).

Trei fire de execuție, ca interfața să nu înghețe:
  1. callback-ul sounddevice: doar pune bucățile de audio într-o coadă;
  2. firul VAD (SpeechRecorder): decide unde începe și unde se termină fraza;
  3. consumatorul (salvare WAV, transcriere): ia frazele gata din `recorder.utterances`.

Declanșare: 'button' = după apăsarea butonului (Space), fraza pornește la primul
cuvânt; 'vad' = pornește singură când cineva vorbește. Fraza se termină după
VAD_SILENCE_END_MS de liniște sau la MAX_UTTERANCE_SECONDS.
Half-duplex: cât timp se redă un sunet (play), microfonul este ignorat.
Modul 'button' pornește ascultarea imediat la apăsare (fără bip, vezi SPEECH_BEEP în config).
"""
import io
import queue
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import webrtcvad

from src.config import (AUDIO_CHANNELS, AUDIO_DTYPE, AUDIO_SAMPLE_RATE, MAX_UTTERANCE_SECONDS,
                        MICROPHONE_NAME, MIN_SPEECH_LEVEL_DB, MIN_UTTERANCE_SECONDS, PLAYBACK_TAIL_MS,
                        SPEECH_BEEP, SPEECH_BEEP_HZ, SPEECH_BEEP_SECONDS, SPEECH_BUTTON_WAIT_SECONDS,
                        SPEECH_TRIGGER, VAD_AGGRESSIVENESS, VAD_FRAME_MS, VAD_PRE_ROLL_MS,
                        VAD_SILENCE_END_MS, VAD_START_MS)

SAMPLE_WIDTH = np.dtype(AUDIO_DTYPE).itemsize           # int16 -> 2 octeți
FRAME_SAMPLES = AUDIO_SAMPLE_RATE * VAD_FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH * AUDIO_CHANNELS
_MUTED = b''                                            # marcaj în coadă: microfon oprit


def _frames(ms):
    """Câte cadre VAD înseamnă `ms` milisecunde."""
    return max(1, round(ms / VAD_FRAME_MS))


@dataclass
class Utterance:
    """O frază decupată: audio brut int16 mono."""
    pcm: bytes
    sample_rate: int
    started_at: float       # time.time() la începutul frazei
    cut: bool               # True = oprită la lungimea maximă, nu de liniște

    @property
    def duration(self):
        return len(self.pcm) / (self.sample_rate * SAMPLE_WIDTH * AUDIO_CHANNELS)

    def wav_bytes(self):
        return to_wav_bytes(self.pcm, self.sample_rate)


def to_wav_bytes(pcm, sample_rate=AUDIO_SAMPLE_RATE):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(AUDIO_CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def wav_duration(wav_bytes):
    """Durata unui WAV (în secunde), citită din antet."""
    with wave.open(io.BytesIO(wav_bytes), 'rb') as wav:
        return wav.getnframes() / wav.getframerate()


def save_wav(path, pcm, sample_rate=AUDIO_SAMPLE_RATE):
    """Salvează audio brut ca WAV (`path` = pathlib.Path)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(to_wav_bytes(pcm, sample_rate))
    return path


def list_input_devices():
    """(index, nume, sistem audio) pentru fiecare microfon."""
    apis = sd.query_hostapis()
    return [(i, d['name'], apis[d['hostapi']]['name'])
            for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0]


def find_input_device(name=MICROPHONE_NAME):
    """Indexul microfonului al cărui nume conține `name`; None = microfonul implicit."""
    if not name:
        return None
    matches = [(i, api) for i, device_name, api in list_input_devices()
               if name.lower() in device_name.lower()]
    if not matches:
        raise ValueError(f"Nu găsesc microfonul „{name}”. "
                         "Lista: python -m src.speech.record_test --list-mics")
    # același microfon apare o dată pentru fiecare sistem audio: îl preferăm pe cel implicit
    default_api = sd.query_hostapis(sd.default.hostapi)['name']
    preferred = [i for i, api in matches if api == default_api]
    return (preferred or [matches[0][0]])[0]


def level_db(frame):
    """Nivelul unui cadru, în dB față de maxim (0 = cel mai tare, -90 = liniște)."""
    samples = np.frombuffer(frame, dtype=AUDIO_DTYPE).astype(np.float32)
    rms = np.sqrt(np.mean(samples ** 2)) if samples.size else 0.0
    return float(20 * np.log10(max(rms, 1.0) / np.iinfo(AUDIO_DTYPE).max))


def speech_level_db(pcm):
    """Cât de tare e o frază: nivelul cadrelor ei cele mai tari (percentila 90), în dB."""
    levels = [level_db(pcm[i:i + FRAME_BYTES]) for i in range(0, len(pcm) - FRAME_BYTES + 1, FRAME_BYTES)]
    return float(np.percentile(levels, 90)) if levels else -90.0


def beep(hz=SPEECH_BEEP_HZ, seconds=SPEECH_BEEP_SECONDS, sample_rate=AUDIO_SAMPLE_RATE):
    """Bip scurt, cu capetele atenuate (fără pocnituri)."""
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    fade = np.clip(np.minimum(t, seconds - t) / 0.01, 0, 1)
    return (0.3 * np.sin(2 * np.pi * hz * t) * fade).astype(np.float32)


class SpeechRecorder:
    """
    Ascultă microfonul și pune frazele complete (Utterance) în coada `utterances`.
    Stări: 'repaus' (modul buton, neapăsat), 'ascultare' (așteaptă vocea), 'vorbire'.
    """

    def __init__(self, trigger=SPEECH_TRIGGER, device_name=MICROPHONE_NAME):
        if trigger not in ('button', 'vad'):
            raise ValueError("trigger trebuie să fie 'button' sau 'vad'")
        self.trigger = trigger
        self.device = find_input_device(device_name)
        self.utterances = queue.Queue()
        self.level_db = -90.0
        self.rejected_quiet = 0                 # fraze ignorate pentru că erau prea slabe
        self._vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self._incoming = queue.Queue()
        self._muted = threading.Event()
        self._arm = threading.Event()
        self._stream = None
        self._worker = None
        self._reset()

    # --- pornire / oprire --------------------------------------------------
    def start(self):
        self._stream = sd.RawInputStream(samplerate=AUDIO_SAMPLE_RATE, channels=AUDIO_CHANNELS,
                                         dtype=AUDIO_DTYPE, blocksize=FRAME_SAMPLES,
                                         device=self.device, callback=self._callback)
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        self._stream.start()
        return self

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._worker is not None:
            self._incoming.put(None)            # oprește firul VAD
            self._worker.join(timeout=2)
            self._worker = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    @property
    def device_name(self):
        return sd.query_devices(self.device, 'input')['name']

    @property
    def muted(self):
        return self._muted.is_set()

    # --- comenzi -------------------------------------------------------------
    def press(self):
        """Butonul: de acum așteaptă fraza (cu SPEECH_BEEP, după un bip cu microfonul oprit)."""
        if self.trigger != 'button' or self.state != 'repaus' or self.muted:
            return
        if SPEECH_BEEP:
            threading.Thread(target=self._beep_then_listen, daemon=True).start()
        else:
            self._arm.set()

    def _beep_then_listen(self):
        self.play(beep())
        self._arm.set()

    def play(self, samples, sample_rate=AUDIO_SAMPLE_RATE):
        """Redă un sunet cu microfonul oprit (half-duplex), ca sunetul să nu fie înregistrat."""
        self._muted.set()
        try:
            sd.play(samples, sample_rate)
            # wait() revine când datele au plecat spre placa audio, dar sunetul iese din
            # difuzor cu întârzierea ei (latency): microfonul rămâne oprit și atât
            latency = sd.get_stream().latency
            sd.wait()
            time.sleep(latency + PLAYBACK_TAIL_MS / 1000)
        finally:
            self._muted.clear()

    # --- procesarea audio ----------------------------------------------------
    def _callback(self, indata, frames, time_info, status):
        # firul audio: doar copiem datele, toată procesarea e în _run
        self._incoming.put(_MUTED if self._muted.is_set() else bytes(indata))

    def _run(self):
        pending = bytearray()
        while True:
            data = self._incoming.get()
            if data is None:
                return
            if data == _MUTED:                  # se redă un sunet: fraza începută se pierde
                pending.clear()
                if self.state == 'vorbire':
                    self._reset()
                continue
            pending.extend(data)
            while len(pending) >= FRAME_BYTES:
                frame = bytes(pending[:FRAME_BYTES])
                del pending[:FRAME_BYTES]
                self.process_frame(frame, self._vad.is_speech(frame, AUDIO_SAMPLE_RATE))

    def process_frame(self, frame, voiced):
        """Un cadru de VAD_FRAME_MS: actualizează starea; la final de frază o pune în coadă."""
        self.level_db = level_db(frame)
        if self._arm.is_set():
            self._arm.clear()
            if self.state == 'repaus':
                self.state = 'ascultare'
        if self.state == 'repaus':
            return

        if self.state == 'ascultare':
            self._pre_roll.append(frame)
            self._voiced_run = self._voiced_run + 1 if voiced else 0
            if self._voiced_run >= _frames(VAD_START_MS):
                self.state = 'vorbire'
                self._audio = list(self._pre_roll)
                self._voiced_total = self._voiced_run
                self._started_at = time.time() - len(self._audio) * VAD_FRAME_MS / 1000
            elif self.trigger == 'button':
                self._waited += 1
                if self._waited >= _frames(SPEECH_BUTTON_WAIT_SECONDS * 1000):
                    self._reset()               # nu a vorbit nimeni după apăsare
            return

        # 'vorbire'
        self._audio.append(frame)
        self._voiced_total += voiced
        self._silence = 0 if voiced else self._silence + 1
        too_long = len(self._audio) >= _frames(MAX_UTTERANCE_SECONDS * 1000)
        if self._silence >= _frames(VAD_SILENCE_END_MS) or too_long:
            self._finish(cut=self._silence < _frames(VAD_SILENCE_END_MS))

    def _finish(self, cut):
        # din liniștea de la final păstrăm doar cât pre-roll-ul, ca finalul să sune natural
        drop = max(0, self._silence - _frames(VAD_PRE_ROLL_MS))
        audio = self._audio[:len(self._audio) - drop]
        speech_seconds = self._voiced_total * VAD_FRAME_MS / 1000   # doar cadrele cu voce
        pcm = b''.join(audio)
        if speech_seconds >= MIN_UTTERANCE_SECONDS:
            if speech_level_db(pcm) >= MIN_SPEECH_LEVEL_DB:
                self.utterances.put(Utterance(pcm, AUDIO_SAMPLE_RATE, self._started_at, cut))
            else:
                self.rejected_quiet += 1        # zgomot/liniște: Whisper ar inventa text
        self._reset()

    def _reset(self):
        self.state = 'repaus' if self.trigger == 'button' else 'ascultare'
        self._pre_roll = deque(maxlen=_frames(VAD_PRE_ROLL_MS) + _frames(VAD_START_MS))
        self._audio = []
        self._silence = 0
        self._voiced_run = 0
        self._voiced_total = 0
        self._waited = 0
        self._started_at = 0.0
