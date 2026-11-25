"""
Verificarea sistemului înainte de prototip (gândită pentru Raspberry Pi, merge și pe laptop):
fiecare componentă e încercată separat și se măsoară timpul ei, ca o problemă să se vadă
înainte de pornirea întregului prototip.

    python -m src.app.system_check                 # toate verificările
    python -m src.app.system_check --camera 0 --mic USB
    python -m src.app.system_check --skip camera,groq --silent

Verifică: sistemul (RAM, temperatură, limitare de putere pe Pi), fișierele modelelor, modelul de
semne (încărcare + o predicție pe o înregistrare proprie), camera + MediaPipe (FPS), microfonul
(1 s, nivelul), vocea Piper (sinteză + redare), Whisper local și Groq pe o frază din
data/test_doctor, traducerea text -> glose și browserul pentru fereastra prototipului.
Raportul se salvează și în data/system_check/<calculator>_<data>.txt.
"""
import argparse
import io
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
import wave
from datetime import datetime
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np

from src.config import (AUDIO_CHANNELS, AUDIO_DTYPE, AUDIO_SAMPLE_RATE, AVATAR_MODEL_FILE, AVATAR_MOTION_DIR,
                        CAMERA_INDEX, DATA_DIR, ENV_FILE, HOLISTIC_MODEL_COMPLEXITY, MICROPHONE_NAME, MODEL_PATH,
                        RECORDED_DIR, SPEECH_TEST_DIR, TTS_MODEL_DIR, TTS_VOICE, WHISPER_LOCAL_DIR)

CHECKS = ['sistem', 'fisiere', 'semne', 'camera', 'microfon', 'voce', 'whisper', 'groq', 'traducere', 'browser']
REPORT_DIR = Path(DATA_DIR) / 'system_check'


class Report:
    """Rezultatele: fiecare verificare are o stare (OK / ATENȚIE / EROARE) și detalii."""

    def __init__(self):
        self.lines, self.summary = [], []

    def log(self, text=''):
        print(text, flush=True)
        self.lines.append(text)

    def result(self, name, status, detail):
        self.summary.append((name, status, detail))
        self.log(f"  -> {status}: {detail}")

    def save(self):
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORT_DIR / f"{socket.gethostname()}_{datetime.now():%Y%m%d_%H%M%S}.txt"
        path.write_text('\n'.join(self.lines) + '\n', encoding='utf-8')
        return path


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def check_sistem(report, args):
    report.log(f"  {platform.system()} {platform.release()} {platform.machine()}, Python {platform.python_version()}")
    model = read_text('/proc/device-tree/model')
    if model:
        report.log(f"  Placa: {model.rstrip(chr(0))}")
    warnings = []
    meminfo = read_text('/proc/meminfo')
    if meminfo:
        mem = {line.split(':')[0]: int(line.split()[1]) for line in meminfo.splitlines() if line.split()[1:]}
        total, free = mem['MemTotal'] / 1024 ** 2, mem.get('MemAvailable', 0) / 1024 ** 2
        report.log(f"  RAM: {total:.1f} GB, disponibil {free:.1f} GB")
        if total < 3.5:
            warnings.append(f"doar {total:.1f} GB RAM (prototipul are nevoie de ~2.5 GB + browserul)")
    temp = read_text('/sys/class/thermal/thermal_zone0/temp')
    if temp:
        celsius = int(temp) / 1000
        report.log(f"  Temperatura procesorului: {celsius:.0f} °C")
        if celsius > 75:
            warnings.append(f"procesorul are {celsius:.0f} °C (răcire? peste 80 °C încetinește)")
    if shutil.which('vcgencmd'):
        out = subprocess.run(['vcgencmd', 'get_throttled'], capture_output=True, text=True).stdout.strip()
        report.log(f"  {out}  (0x0 = alimentare și temperatură în regulă)")
        if out and not out.endswith('=0x0'):
            warnings.append(f"Pi-ul a fost limitat ({out}): sursă prea slabă sau supraîncălzire")
    free_disk = shutil.disk_usage('.').free / 1024 ** 3
    report.log(f"  Spațiu liber pe disc: {free_disk:.1f} GB")
    if sys.version_info[:2] != (3, 11):
        warnings.append(f"Python {platform.python_version()} (pachetele sunt fixate pentru 3.11)")
    report.result('sistem', 'ATENȚIE' if warnings else 'OK', '; '.join(warnings) or 'în regulă')


def check_fisiere(report, args):
    files = {
        'model semne': Path(MODEL_PATH),
        'Whisper local': WHISPER_LOCAL_DIR / 'model.bin',
        'voce Piper': TTS_MODEL_DIR / f'{TTS_VOICE}.onnx',
        'avatar (VRM)': AVATAR_MODEL_FILE,
        'mișcări avatar': AVATAR_MOTION_DIR / 'index.json',
        'cheie Groq (.env)': ENV_FILE,
    }
    missing = []
    for name, path in files.items():
        exists = path.exists()
        size = f"{path.stat().st_size / 1024 ** 2:.1f} MB" if exists else 'LIPSEȘTE'
        report.log(f"  {name:18} {path.as_posix():40} {size}")
        if not exists:
            missing.append(name)
    has_key = ENV_FILE.exists() and any(line.startswith('GROQ_API_KEY=') and len(line.strip()) > 14
                                        for line in ENV_FILE.read_text(encoding='utf-8').splitlines())
    if ENV_FILE.exists() and not has_key:
        missing.append('GROQ_API_KEY e gol în .env (doar modul offline)')
    report.result('fisiere', 'EROARE' if any(m != 'cheie Groq (.env)' and 'GROQ' not in m for m in missing)
                  else 'ATENȚIE' if missing else 'OK', ', '.join(missing) or 'toate există')


def check_semne(report, args):
    start = time.perf_counter()
    from tensorflow.keras.models import load_model
    from src.ml.model import predict_segment
    from src.config import ACTIONS
    model = load_model(MODEL_PATH)
    load_s = time.perf_counter() - start
    example = next(iter(sorted((Path(RECORDED_DIR) / 'hurt').glob('*.npy'))), None)
    if example is None:
        report.result('semne', 'ATENȚIE', f"model încărcat în {load_s:.1f} s; nu am găsit o înregistrare de test")
        return
    segment = np.load(example)
    predict_segment(model, segment)                 # prima predicție include pregătirea TensorFlow
    times = []
    for _ in range(5):
        t = time.perf_counter()
        probabilities = predict_segment(model, segment)
        times.append(time.perf_counter() - t)
    word = ACTIONS[int(np.argmax(probabilities))]
    detail = (f"încărcare {load_s:.1f} s, o predicție {1000 * np.median(times):.0f} ms; "
              f"{example.parent.name}/{example.name} -> {word} ({probabilities.max():.0%})")
    report.result('semne', 'OK' if word == 'hurt' else 'ATENȚIE', detail)


def check_camera(report, args):
    import cv2
    from src.app.camera import make_holistic, open_camera, process_frame
    cap = open_camera(args.camera)
    if cap is None:
        devices = sorted(Path('/dev').glob('video*')) if Path('/dev').exists() else []
        report.result('camera', 'EROARE', f"camera {args.camera} nu se deschide"
                      + (f"; există: {', '.join(d.name for d in devices)}" if devices else ''))
        return
    ok, frame = cap.read()
    if not ok:
        cap.release()
        report.result('camera', 'EROARE', f"camera {args.camera} se deschide, dar nu dă imagini")
        return
    report.log(f"  Camera {args.camera}: imagine {frame.shape[1]}x{frame.shape[0]}; "
               f"MediaPipe model_complexity={HOLISTIC_MODEL_COMPLEXITY}")
    t = time.perf_counter()
    for _ in range(30):
        cap.read()
    camera_fps = 30 / (time.perf_counter() - t)
    report.log(f"  Doar camera: {camera_fps:.1f} FPS. Acum MediaPipe {args.seconds:.0f} s: ridică mâinile în fața camerei.")
    frames = hands = 0
    with make_holistic() as holistic:
        process_frame(holistic, frame)              # prima rulare încarcă modelele
        t = time.perf_counter()
        while time.perf_counter() - t < args.seconds:
            ok, frame = cap.read()
            if not ok:
                break
            results = process_frame(holistic, frame)
            frames += 1
            hands += results.left_hand_landmarks is not None or results.right_hand_landmarks is not None
        fps = frames / (time.perf_counter() - t)
    cap.release()
    cv2.destroyAllWindows()
    detail = f"camera {camera_fps:.0f} FPS, cu MediaPipe {fps:.1f} FPS; mâini văzute în {hands}/{frames} cadre"
    # Segmentarea e pe timp (nu pe cadre), dar sub ~8 FPS mișcările rapide pierd din formă
    report.result('camera', 'OK' if fps >= 8 else 'ATENȚIE', detail + ('' if fps >= 8 else
                  '; FPS mic: încearcă HOLISTIC_MODEL_COMPLEXITY = 0 în src/config.py'))


def check_microfon(report, args):
    import sounddevice as sd
    from src.speech.audio import find_input_device, list_input_devices
    for index, name, api in list_input_devices():
        report.log(f"  [{index}] {name} ({api})")
    device = find_input_device(args.mic)
    name = sd.query_devices(device, 'input')['name']
    report.log(f"  Folosit: {name}. Înregistrez {args.mic_seconds:.0f} s: vorbește acum.")
    with sd.InputStream(samplerate=AUDIO_SAMPLE_RATE, channels=AUDIO_CHANNELS, dtype=AUDIO_DTYPE,
                        device=device) as stream:
        audio, _ = stream.read(int(AUDIO_SAMPLE_RATE * args.mic_seconds))
    samples = audio.astype(np.float32).ravel() / 32768
    frames = samples[: len(samples) // 480 * 480].reshape(-1, 480)      # bucăți de 30 ms
    levels = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
    loud = float(np.percentile(levels, 90))
    detail = f"{name}, 16 kHz merge; nivel (percentila 90) {loud:.0f} dB"
    report.result('microfon', 'OK' if loud > -50 else 'ATENȚIE',
                  detail + ('' if loud > -50 else ' = aproape liniște (microfon greșit sau volum mic?)'))


def check_voce(report, args):
    from src.speech.tts import PiperTTS
    t = time.perf_counter()
    tts = PiperTTS()
    load_s = time.perf_counter() - t
    tts.synthesize('Bună ziua.')                    # prima sinteză e mai lentă (încălzire)
    text = 'Mă doare capul de ieri.'
    t = time.perf_counter()
    audio = tts.synthesize(text)
    synth_s = time.perf_counter() - t
    seconds = len(audio) / tts.sample_rate
    detail = f"încărcare {load_s:.1f} s, sinteză {synth_s:.2f} s pentru {seconds:.1f} s de voce (RTF {synth_s / seconds:.2f})"
    if not args.silent:
        import sounddevice as sd
        report.log(f"  Redau „{text}” pe {sd.query_devices(kind='output')['name']}...")
        sd.play(audio, tts.sample_rate)
        sd.wait()
        detail += '; ai auzit vocea? (dacă nu: ieșirea audio din setările Pi-ului)'
    report.result('voce', 'OK', detail)


def test_phrase():
    wav = next(iter(sorted(SPEECH_TEST_DIR.glob('*.wav'))), None)
    if wav is None:
        return None, None
    reference = wav.with_suffix('.txt')
    return wav, reference.read_text(encoding='utf-8').strip() if reference.exists() else ''


def wav_seconds(data):
    with wave.open(io.BytesIO(data)) as w:
        return w.getnframes() / w.getframerate()


def check_whisper(report, args):
    wav, reference = test_phrase()
    if wav is None:
        report.result('whisper', 'ATENȚIE', f"nu există WAV-uri în {SPEECH_TEST_DIR.as_posix()}")
        return
    from src.speech.local_engine import LocalEngine
    t = time.perf_counter()
    engine = LocalEngine()
    load_s = time.perf_counter() - t
    result = engine.transcribe(wav.read_bytes())
    if not result.ok:
        report.result('whisper', 'EROARE', result.error)
        return
    report.log(f"  Corect: {reference}\n  Local:  {result.text}")
    rtf = result.processing_seconds / result.audio_seconds
    report.result('whisper', 'OK' if rtf < 1.5 else 'ATENȚIE',
                  f"încărcare {load_s:.1f} s, {result.processing_seconds:.1f} s pentru {result.audio_seconds:.1f} s "
                  f"de audio (RTF {rtf:.2f}" + (')' if rtf < 1.5 else '; lent, medicul așteaptă după fiecare frază)'))


def check_groq(report, args):
    from src.speech.router import internet_available
    if not internet_available():
        report.result('groq', 'ATENȚIE', 'fără internet: prototipul folosește doar modelul local')
        return
    wav, reference = test_phrase()
    from src.speech.groq_engine import GroqEngine
    try:
        engine = GroqEngine()
    except RuntimeError as error:
        report.result('groq', 'ATENȚIE', f"internet da, dar {error}")
        return
    result = engine.transcribe(wav.read_bytes())
    if not result.ok:
        report.result('groq', 'ATENȚIE', f"eroare Groq: {result.error} (se trece pe local)")
        return
    report.log(f"  Groq:   {result.text}")
    report.result('groq', 'OK', f"{result.processing_seconds:.1f} s pentru {result.audio_seconds:.1f} s de audio")


def check_traducere(report, args):
    from src.translate.gloss_to_text import gloss_to_text
    from src.translate.text_to_gloss import text_to_gloss
    translation = text_to_gloss('Vă doare capul?')
    sentence = gloss_to_text(['me', 'headache', 'since', 'yesterday'])
    report.result('traducere', 'OK', f"„Vă doare capul?” -> {' '.join(translation.gloss)}; "
                  f"me headache since yesterday -> „{sentence}”")


def check_browser(report, args):
    from src.avatar.server import find_browser
    try:
        browser = find_browser()
    except SystemExit:
        report.result('browser', 'EROARE', 'nu am găsit Chromium / Chrome / Edge (sudo apt install chromium)')
        return
    display = os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')
    if platform.system() == 'Linux' and not display:
        report.result('browser', 'ATENȚIE', f"{browser}, dar nu există ecran (DISPLAY): rulează din desktopul Pi-ului")
        return
    report.result('browser', 'OK', browser)


def main():
    parser = argparse.ArgumentParser(description='Verificarea componentelor prototipului (laptop sau Pi)')
    # număr = indexul camerei, altfel adresa unui flux (telefonul ca cameră Wi-Fi), ca în src/app/camera.py;
    # aici fără importul lui camera.py, ca un MediaPipe lipsă să cadă doar la verificarea camerei
    parser.add_argument('--camera', type=lambda v: int(v) if v.isdigit() else v, default=CAMERA_INDEX)
    parser.add_argument('--mic', default=MICROPHONE_NAME, help='parte din numele microfonului')
    parser.add_argument('--seconds', type=float, default=8.0, help='cât rulează MediaPipe pe cameră')
    parser.add_argument('--mic-seconds', type=float, default=3.0)
    parser.add_argument('--silent', action='store_true', help='nu reda vocea Piper în difuzor')
    parser.add_argument('--skip', default='', help=f"verificări sărite, separate prin virgulă: {','.join(CHECKS)}")
    args = parser.parse_args()
    skip = {s.strip() for s in args.skip.split(',') if s.strip()}

    report = Report()
    report.log(f"Verificare sistem {socket.gethostname()}, {datetime.now():%Y-%m-%d %H:%M}")
    for name in CHECKS:
        if name in skip:
            continue
        report.log(f"\n[{name}]")
        t = time.perf_counter()
        try:
            globals()[f'check_{name}'](report, args)
        except Exception as error:                  # o componentă căzută nu oprește celelalte verificări
            report.log(traceback.format_exc().rstrip())
            report.result(name, 'EROARE', f"{type(error).__name__}: {error}")
        report.lines[-1] += f"  [{time.perf_counter() - t:.1f} s]"

    report.log('\nRezumat:')
    for name, status, detail in report.summary:
        report.log(f"  {status:8} {name:10} {detail}")
    errors = sum(status == 'EROARE' for _, status, _ in report.summary)
    report.log(f"\n{'Totul e pregătit pentru prototip.' if not errors else f'{errors} erori: vezi detaliile de mai sus.'}")
    print(f"Raport salvat: {report.save().as_posix()}")


if __name__ == '__main__':
    main()
