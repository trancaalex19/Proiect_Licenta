"""
Test pentru captura audio (etapa 2 a părții vorbire -> text): înregistrează fraze
de la microfon și le salvează ca WAV în data/speech_recordings/, fără transcriere.
Salvarea se face într-un fir separat, exact cum va rula mai târziu transcrierea.

Rulare (din rădăcina proiectului):
    python -m src.speech.record_test                   # Space = înregistrează o frază
    python -m src.speech.record_test --trigger vad     # pornește singur când vorbești
    python -m src.speech.record_test --list-mics       # lista microfoanelor
    python -m src.speech.record_test --mic "Realtek"   # alt microfon (parte din nume)
Taste în fereastră: Space = vorbește (după ce fereastra arată „Te ascult...”), q = ieșire.
"""
import argparse
import queue
import threading
from datetime import datetime

import cv2
import sounddevice as sd

from src.config import SPEECH_BUTTON_KEY, SPEECH_RECORDINGS_DIR, SPEECH_TRIGGER, MICROPHONE_NAME
from src.speech.audio import SpeechRecorder, list_input_devices, save_wav
from src.speech.window import draw

WINDOW_NAME = 'Test microfon'


def parse_args():
    parser = argparse.ArgumentParser(description='Test pentru captura audio (fraze -> WAV)')
    parser.add_argument('--trigger', choices=['button', 'vad'], default=SPEECH_TRIGGER)
    parser.add_argument('--mic', default=MICROPHONE_NAME, help='parte din numele microfonului')
    parser.add_argument('--list-mics', action='store_true', help='afișează microfoanele și iese')
    return parser.parse_args()


def save_worker(recorder, stop, last):
    """Firul consumator: salvează fiecare frază gata ca WAV (mai târziu: transcriere)."""
    while not stop.is_set():
        try:
            utterance = recorder.utterances.get(timeout=0.1)
        except queue.Empty:
            continue
        name = datetime.fromtimestamp(utterance.started_at).strftime('%Y%m%d_%H%M%S_%f')[:-3]
        path = save_wav(SPEECH_RECORDINGS_DIR / f'{name}.wav', utterance.pcm, utterance.sample_rate)
        note = ' (oprită la lungimea maximă)' if utterance.cut else ''
        print(f"Salvat {path.as_posix()}: {utterance.duration:.1f} s{note}")
        last['text'] = f'{path.name}  ({utterance.duration:.1f} s)'
        last['count'] += 1


def status_lines(recorder, last):
    return [f'Microfon: {recorder.device_name[:45]}',
            f'Mod: {recorder.trigger}   Salvate: {last["count"]}   Ignorate (prea slabe): {recorder.rejected_quiet}',
            f'Ultima: {last["text"]}',
            'SPACE = vorbeste   q = iesire']


def main():
    args = parse_args()
    if args.list_mics:
        for index, name, api in list_input_devices():
            print(f"{index:3d}  {api:20s}  {name}")
        return

    try:
        recorder = SpeechRecorder(trigger=args.trigger, device_name=args.mic)
        recorder.start()
    except (ValueError, sd.PortAudioError) as error:
        print(f"EROARE la microfon: {error}")
        return
    print(f"Microfon: {recorder.device_name}. Fraze salvate în {SPEECH_RECORDINGS_DIR.as_posix()}/")

    stop, last = threading.Event(), {'text': '-', 'count': 0}
    saver = threading.Thread(target=save_worker, args=(recorder, stop, last), daemon=True)
    saver.start()
    try:
        while True:
            cv2.imshow(WINDOW_NAME, draw(recorder, status_lines(recorder, last)))
            key = cv2.waitKey(30) & 0xFF
            if key == ord(SPEECH_BUTTON_KEY):
                recorder.press()
            elif key == ord('q') or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        recorder.stop()
        stop.set()
        saver.join(timeout=2)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
