"""
Semiprototipul end-to-end (pe laptop): medicul vorbește, pacientul surd semnează, totul într-o
singură fereastră (pagina avatarului, deschisă ca aplicație) și un singur proces Python.

  Medic -> pacient: Space / butonul „Vorbește” -> microfon (SpeechRecorder) -> SpeechRouter
                    (Groq / local) -> text -> text_to_gloss -> avatarul redă semnele
                    (mecanismul pentru propoziții), cu textul medicului ca subtitrare.
  Pacient -> medic: camera -> SignSegmenter -> predict_segment (ca în live_predict); semnele peste
                    PATIENT_MIN_CONFIDENCE se adună, iar după PATIENT_PAUSE_SECONDS fără semne
                    propoziția (gloss_to_text) ajunge în firul conversației și e citită cu voce
                    pentru medic (Piper, src/speech/tts.py).
  Half-duplex: cât timp medicul vorbește (după Space, până se încheie fraza), camera ignoră semnele;
  cât timp se aude vocea pacientului, microfonul e oprit (SpeechRecorder.play).

Python face tot calculul (camera, MediaPipe, modelele, microfonul); pagina doar afișează:
/api/state (starea, JSON), /api/camera (imaginea camerei, MJPEG), /api/command/speak|voice|say-<id>|quit.

Rulare (din rădăcina proiectului, cu venv activat):
    python -m src.app.prototype                        # camera și modul vorbirii din config
    python -m src.app.prototype --camera 0             # camera laptopului
    python -m src.app.prototype --mode offline_only    # doar modelul local de vorbire
    python -m src.app.prototype --no-browser           # fără fereastră (adresa e în terminal)
Pe Raspberry Pi: deploy/pi/run.sh (vezi deploy/pi/CITESTE.md).
În fereastră: Space = vorbește medicul. Închiderea ferestrei (sau Ctrl+C în terminal) oprește tot.
Fiecare mesaj (și cuvintele fără semn) se scrie în data/prototype_log.csv.
"""
import argparse
import csv
import os
import queue
import subprocess
import tempfile
import threading
import time
import webbrowser
from datetime import datetime
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import cv2
import numpy as np
import sounddevice as sd

from src.app.camera import camera_source, make_holistic, open_camera, process_frame, smooth_fps
from src.app.overlay import draw_landmarks
from src.app.segmenter import START_LEVEL, SignSegmenter
from src.app.sign_info import SIGN_INFO
from src.config import (ACTIONS, AVATAR_HOST, AVATAR_PORT, CAMERA_INDEX, MICROPHONE_NAME, MODEL_PATH, ON_LINUX,
                        PATIENT_MAX_WORDS, PATIENT_MIN_CONFIDENCE, PATIENT_PAUSE_SECONDS, PROTOTYPE_HISTORY,
                        PROTOTYPE_JPEG_QUALITY, PROTOTYPE_LOG_FILE, PROTOTYPE_MIRROR_CAMERA,
                        PROTOTYPE_POLL_MS, PROTOTYPE_STREAM_HEIGHT, PROTOTYPE_WINDOW_SIZE, SPEECH_MODE,
                        TTS_ENABLED)
from src.features.keypoints import extract_keypoints
from src.translate.gloss_to_text import gloss_to_text
from src.translate.text_to_gloss import text_to_gloss

LOG_COLUMNS = ['timestamp', 'directie', 'text', 'glose', 'metoda', 'scor', 'necunoscute', 'motor']
PATIENT_BGR = (11, 98, 168)        # chihlimbar, aceeași culoare ca pacientul în pagină


def meaning(gloss):
    """Sensul în română al unui semn: 'hurt' -> 'Durere'."""
    return SIGN_INFO[gloss]['meaning'].split(' / ')[0]


class LiveState:
    """
    Starea comună dintre firul camerei (care o scrie) și serverul paginii (care o citește):
    firul conversației, starea de moment (microfon, cameră, recunoaștere), ultima imagine
    a camerei și comenzile venite din pagină.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frame_ready = threading.Condition()
        self._jpeg, self._frame_id = None, 0
        self._next_id = 1
        self.messages = []
        self.now = {}               # starea de moment, refăcută la fiecare cadru
        self.commands = queue.Queue()
        self.closed = False

    # --- conversația -----------------------------------------------------------
    def add_message(self, who, text, gloss, **extra):
        """Un mesaj nou în firul conversației; întoarce id-ul lui."""
        with self._lock:
            message = {'id': self._next_id, 'who': who, 'text': text, 'gloss': list(gloss),
                       'meanings': [meaning(g) for g in gloss], 'time': datetime.now().strftime('%H:%M'), **extra}
            self._next_id += 1
            self.messages = (self.messages + [message])[-PROTOTYPE_HISTORY:]
            return message['id']

    def message_text(self, message_id):
        with self._lock:
            return next((m['text'] for m in self.messages if m['id'] == message_id), None)

    def set_now(self, now):
        with self._lock:
            self.now = now

    # --- pentru server (AvatarHandler.live) ----------------------------------------
    def snapshot(self):
        with self._lock:
            return {'poll_ms': PROTOTYPE_POLL_MS, 'messages': self.messages, **self.now}

    def command(self, name):
        self.commands.put(name)

    def set_frame(self, jpeg):
        with self._frame_ready:
            self._jpeg, self._frame_id = jpeg, self._frame_id + 1
            self._frame_ready.notify_all()

    def frames(self):
        """Fiecare imagine nouă a camerei (pentru fluxul MJPEG), până la închidere."""
        last = 0
        while not self.closed:
            with self._frame_ready:
                self._frame_ready.wait_for(lambda: self._frame_id != last or self.closed, timeout=1.0)
                jpeg, last = self._jpeg, self._frame_id
            if jpeg is not None and not self.closed:
                yield jpeg

    def close(self):
        self.closed = True
        with self._frame_ready:
            self._frame_ready.notify_all()


class PatientSentence:
    """
    Adună semnele recunoscute ale pacientului într-o propoziție. Semnul intră doar cu
    încredere >= min_confidence; propoziția e gata după `pause` secunde fără semn nou
    (sau la max_words semne). Fără cameră: se testează cu probabilități date.
    """

    def __init__(self, min_confidence=PATIENT_MIN_CONFIDENCE, pause=PATIENT_PAUSE_SECONDS,
                 max_words=PATIENT_MAX_WORDS):
        self.min_confidence, self.pause, self.max_words = min_confidence, pause, max_words
        self.words = []
        self.last_sign = None       # momentul ultimului semn acceptat

    def add(self, probabilities, now):
        """Un semn clasificat; întoarce cuvântul dacă a intrat în propoziție, altfel None."""
        k = int(np.argmax(probabilities))
        if probabilities[k] < self.min_confidence:
            return None
        self.words.append(ACTIONS[k])
        self.last_sign = now
        return ACTIONS[k]

    def progress(self, now, signing):
        """Cât din pauza de final a trecut (0..1), pentru bara din pagină."""
        if not self.words or signing:
            return 0.0
        return min(1.0, (now - self.last_sign) / self.pause)

    def ready(self, now, signing):
        """Propoziția e gata: pauză lungă fără semne (și nu e un semn în curs) sau prea lungă."""
        return bool(self.words) and (len(self.words) >= self.max_words or self.progress(now, signing) >= 1.0)

    def take(self):
        words, self.words, self.last_sign = self.words, [], None
        return words


def log_message(direction, text, gloss, method='', score='', unknown=(), engine=''):
    is_new = not PROTOTYPE_LOG_FILE.exists()
    PROTOTYPE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig la creare: Excel afișează corect diacriticele
    with open(PROTOTYPE_LOG_FILE, 'a', newline='', encoding='utf-8-sig' if is_new else 'utf-8') as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(LOG_COLUMNS)
        writer.writerow([datetime.now().isoformat(timespec='seconds'), direction, text, ' '.join(gloss),
                         method, score, ' '.join(unknown), engine])


def handle_transcript(result, fallback, live):
    """Un rezultat al transcrierii -> glose -> firul conversației (avatarul îl redă)."""
    via = f' (Groq a eșuat: {fallback})' if fallback else ''
    if not result.ok:
        print(f"[medic, {result.engine}{via}] EROARE: {result.error}")
        live.add_message('system', 'Nu am înțeles fraza. Apăsați Space și repetați.', [])
        return
    translation = text_to_gloss(result.text)
    live.add_message('doctor', result.text, translation.gloss, method=translation.method,
                     engine=result.engine, unknown=translation.unknown)
    print(f"[medic, {result.engine}{via}] {result.text}\n"
          f"    -> semne: {' '.join(translation.gloss) or '-'} ({translation.method}, "
          f"scor frază {translation.score:.2f})"
          + (f"; fără semn: {', '.join(translation.unknown)}" if translation.unknown else ''))
    log_message('medic', result.text, translation.gloss, translation.method, f'{translation.score:.2f}',
                translation.unknown, result.engine)


def send_patient(patient, live, voice):
    """Propoziția pacientului: în firul conversației și, dacă vocea e pornită, citită pentru medic."""
    words = patient.take()
    text = gloss_to_text(words)
    live.add_message('patient', text, words)
    print(f"[pacient] {' '.join(words)}  ->  {text}")
    log_message('pacient', text, words)
    if voice.enabled:
        voice.say(text)


class Voice:
    """Vocea pacientului (Piper): poate lipsi (fără model) și se poate opri din pagină."""

    def __init__(self, recorder):
        self.output, self.enabled, self.error = None, False, None
        if not TTS_ENABLED:
            self.error = 'oprită din config (TTS_ENABLED)'
            return
        try:
            from src.speech.tts import PiperTTS, SpeechOutput
            self.output = SpeechOutput(PiperTTS(), recorder.play).start()
            self.enabled = True
        except (RuntimeError, ImportError) as error:
            self.error = str(error)

    @property
    def available(self):
        return self.output is not None

    @property
    def speaking(self):
        return self.output is not None and self.output.busy

    def say(self, text):
        if self.output is not None:
            self.output.say(text)

    def toggle(self):
        self.enabled = self.available and not self.enabled

    def stop(self):
        if self.output is not None:
            self.output.stop()


def start_server(live, host, port):
    """Serverul paginii, într-un fir separat, cu /api/ legat de starea comună."""
    from src.avatar.server import AvatarHandler, make_server

    class Handler(AvatarHandler):
        pass
    Handler.live = live

    server, url = make_server(host, port, Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, url + '?live'


def open_app_window(url, profile_dir):
    """
    Pagina ca fereastră de aplicație (Edge / Chrome --app: fără bară de adrese și file), cu profil
    separat, ca procesul să fie al nostru: când se închide fereastra, se oprește și prototipul.
    Fără Edge / Chrome: browserul implicit (atunci programul se oprește din terminal).
    """
    from src.avatar.server import find_browser
    try:
        browser = find_browser()
    except SystemExit:
        webbrowser.open(url)
        return None
    width, height = PROTOTYPE_WINDOW_SIZE
    flags = [f'--app={url}', f'--user-data-dir={profile_dir}', '--no-first-run', '--no-default-browser-check',
             '--disable-extensions', f'--window-size={width},{height}']
    if ON_LINUX:
        # Chromium pe Pi: fără fereastra „deblochează keyring-ul” la pornire; ecran mic -> fereastra pe tot ecranul
        flags += ['--password-store=basic', '--start-maximized']
    return subprocess.Popen([browser, *flags], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def camera_jpeg(image, results, segmenter, signing_allowed):
    """Imaginea pentru pagină: brațe și mâini, linia de pornire a semnului, micșorată și oglindită."""
    draw_landmarks(image, results)
    line = segmenter.line_y(START_LEVEL)
    if signing_allowed and line is not None and 0 < line < 1:
        h, w = image.shape[:2]
        y = int(line * h)
        thickness = max(2, h // 240)
        for x in range(0, w, w // 24):                        # linie punctată
            cv2.line(image, (x, y), (min(x + w // 48, w), y), PATIENT_BGR if segmenter.state == 'semn'
                     else (255, 255, 255), thickness, cv2.LINE_AA)
    h, w = image.shape[:2]
    scale = PROTOTYPE_STREAM_HEIGHT / h
    image = cv2.resize(image, (int(w * scale), PROTOTYPE_STREAM_HEIGHT), interpolation=cv2.INTER_AREA)
    if PROTOTYPE_MIRROR_CAMERA:
        image = cv2.flip(image, 1)
    ok, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, PROTOTYPE_JPEG_QUALITY])
    return jpeg.tobytes() if ok else None


def moment_state(recorder, worker, router, segmenter, patient, recognition, results, fps, now, voice):
    """Starea de moment pentru pagină (se reface la fiecare cadru)."""
    doctor_active = recorder.state != 'repaus'
    transcribing = worker.busy or worker.waiting > 0
    signing = segmenter.state == 'semn'
    turn = ('doctor' if doctor_active or transcribing else
            'patient' if signing or patient.words or voice.speaking else 'idle')
    return {
        'turn': turn,
        'mic': {'state': recorder.state, 'level_db': round(recorder.level_db, 1), 'transcribing': transcribing},
        'voice': {'available': voice.available, 'enabled': voice.enabled, 'speaking': voice.speaking,
                  'error': voice.error},
        'speech': {'engine': 'Groq' if router.groq_active else 'local', 'online': router.online},
        'camera': {'fps': round(fps), 'signing': signing, 'paused': doctor_active,
                   'hands': {'left': results.left_hand_landmarks is not None,
                             'right': results.right_hand_landmarks is not None}},
        'recognition': recognition,
        'pending': {'words': list(patient.words), 'meanings': [meaning(w) for w in patient.words],
                    'progress': round(patient.progress(now, signing), 3)},
    }


def recognition_info(probabilities, accepted):
    """Ultimul semn clasificat: primele 3 variante, pentru pagină."""
    top = np.argsort(probabilities)[::-1][:3]
    return {'accepted': accepted is not None, 'threshold': PATIENT_MIN_CONFIDENCE,
            'top': [{'word': ACTIONS[i], 'meaning': meaning(ACTIONS[i]), 'p': round(float(probabilities[i]), 3)}
                    for i in top]}


def main():
    parser = argparse.ArgumentParser(description='Prototip end-to-end: vorbire -> avatar, semne -> text')
    parser.add_argument('--camera', type=camera_source, default=CAMERA_INDEX)
    parser.add_argument('--mode', choices=['auto', 'offline_only'], default=SPEECH_MODE)
    parser.add_argument('--mic', default=MICROPHONE_NAME, help='parte din numele microfonului')
    parser.add_argument('--host', default=AVATAR_HOST)
    parser.add_argument('--port', type=int, default=AVATAR_PORT)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()

    from tensorflow.keras.models import load_model
    from src.ml.model import predict_segment
    from src.speech.audio import SpeechRecorder
    from src.speech.router import SpeechRouter, TranscriptionWorker

    print("Încărcăm modelul de semne...")
    model = load_model(MODEL_PATH)
    print("Încărcăm modelul local de vorbire...")
    router = SpeechRouter(mode=args.mode)
    print(f"Groq: {router.groq_status}. Internet: {'da' if router.online else 'nu'}.")
    try:
        recorder = SpeechRecorder(trigger='button', device_name=args.mic).start()
    except (ValueError, sd.PortAudioError) as error:
        print(f"EROARE la microfon: {error}")
        router.close()
        return
    worker = TranscriptionWorker(router, recorder.utterances).start()
    voice = Voice(recorder)
    print(f"Vocea pacientului: {'pornită' if voice.enabled else f'indisponibilă ({voice.error})'}.")
    cap = open_camera(args.camera, " Încearcă --camera 0 sau verifică telefonul.")
    if cap is None:
        voice.stop()
        recorder.stop()
        worker.stop()
        router.close()
        return

    live = LiveState()
    server, url = start_server(live, args.host, args.port)
    profile = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    window = None if args.no_browser else open_app_window(url, profile.name)
    print(f"Gata: {url}  (Space = vorbește medicul; închide fereastra sau Ctrl+C pentru ieșire)")
    print(f"Jurnal: {PROTOTYPE_LOG_FILE.as_posix()}")

    segmenter, patient = SignSegmenter(), PatientSentence()
    recognition = None
    fps, last_time = 0.0, time.perf_counter()
    try:
        with make_holistic() as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                now = time.perf_counter()
                fps = smooth_fps(fps, now - last_time)
                last_time = now

                command = None if live.commands.empty() else live.commands.get()
                if command == 'quit' or (window is not None and window.poll() is not None):
                    break
                doctor_active = recorder.state != 'repaus'
                if command == 'speak' and not doctor_active and not voice.speaking:
                    if patient.words:                       # pacientul n-a terminat: întâi propoziția lui
                        send_patient(patient, live, voice)
                    if not voice.speaking:                  # cu vocea pornită, medicul apasă după ce se aude
                        recorder.press()
                    recognition = None
                elif command == 'voice':
                    voice.toggle()
                elif command and command.startswith('say-') and command[4:].isdigit():
                    text = live.message_text(int(command[4:]))   # „Ascultă din nou” pe un mesaj
                    if text:
                        voice.say(text)

                while not worker.results.empty():          # medic -> avatar
                    handle_transcript(*worker.results.get(), live)

                results = process_frame(holistic, frame)
                if doctor_active:
                    segmenter.reset()                       # half-duplex: semnele se ignoră
                else:
                    segment = segmenter.update(extract_keypoints(results), now)
                    if segment is not None:                 # semn întreg: o singură predicție
                        probabilities = predict_segment(model, segment)
                        recognition = recognition_info(probabilities, patient.add(probabilities, now))
                    if patient.ready(now, signing=segmenter.state == 'semn'):
                        send_patient(patient, live, voice)

                live.set_now(moment_state(recorder, worker, router, segmenter, patient, recognition,
                                          results, fps, now, voice))
                jpeg = camera_jpeg(frame, results, segmenter, not doctor_active)
                if jpeg is not None:
                    live.set_frame(jpeg)
    except KeyboardInterrupt:
        pass
    finally:
        live.close()
        cap.release()
        voice.stop()
        recorder.stop()
        worker.stop()
        router.close()
        server.shutdown()
        if window is not None and window.poll() is None:
            window.terminate()
            try:
                window.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        profile.cleanup()


if __name__ == '__main__':
    main()
