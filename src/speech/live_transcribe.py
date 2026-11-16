"""
Test pentru router (etapa 5 a părții vorbire -> text): vorbești la microfon, iar textul
apare în terminal și în fereastră, cu motorul folosit (Groq sau local) și starea
internetului. Pentru testul offline: oprește Wi-Fi-ul în timp ce rulează programul.
Toate încercările se scriu în data/speech_log.csv.

Rulare (din rădăcina proiectului):
    python -m src.speech.live_transcribe                       # mod din config (auto)
    python -m src.speech.live_transcribe --mode offline_only   # doar modelul local
    python -m src.speech.live_transcribe --trigger vad         # fără buton
Taste în fereastră: Space = vorbește, q = ieșire.
"""
import argparse

import cv2
import sounddevice as sd

from src.config import MICROPHONE_NAME, SPEECH_BUTTON_KEY, SPEECH_LOG_FILE, SPEECH_MODE, SPEECH_TRIGGER
from src.speech.audio import SpeechRecorder
from src.speech.router import SpeechRouter, TranscriptionWorker
from src.speech.window import ascii_text, draw, wrap

WINDOW_NAME = 'Vorbire -> text'


def status_lines(router, worker, last):
    internet = 'da' if router.online else 'NU'
    engine = 'Groq' if router.groq_active else 'local'
    lines = [f'Mod: {router.mode}   Internet: {internet}   Urmatoarea fraza: {engine}',
             f'Groq: {ascii_text(router.groq_status)[:55]}']
    if worker.busy or worker.waiting:
        lines.append(f'Transcriu...  (in asteptare: {worker.waiting})')
    else:
        lines.append(ascii_text(last['info']))
    lines += wrap(ascii_text(last['text'])) + ['SPACE = vorbeste   q = iesire']
    return lines


def report(result, fallback, last):
    """Afișează rezultatul în terminal (cu diacritice) și îl pregătește pentru fereastră."""
    via = f' (Groq a eșuat: {fallback})' if fallback else ''
    confidence = '-' if result.confidence is None else f'{result.confidence:.2f}'
    if result.ok:
        print(f"[{result.engine}{via}] {result.text}   "
              f"(audio {result.audio_seconds:.1f} s, procesare {result.processing_seconds:.2f} s, "
              f"încredere {confidence})")
        last['text'] = result.text
    else:
        print(f"[{result.engine}{via}] EROARE: {result.error}")
        last['text'] = ''
    last['info'] = f'{result.engine}{via}: {result.processing_seconds:.1f} s, incredere {confidence}'


def main():
    parser = argparse.ArgumentParser(description='Vorbire -> text în timp real (router Groq / local)')
    parser.add_argument('--mode', choices=['auto', 'offline_only'], default=SPEECH_MODE)
    parser.add_argument('--trigger', choices=['button', 'vad'], default=SPEECH_TRIGGER)
    parser.add_argument('--mic', default=MICROPHONE_NAME, help='parte din numele microfonului')
    args = parser.parse_args()

    print("Încărcăm modelul local...")
    router = SpeechRouter(mode=args.mode)
    print(f"Groq: {router.groq_status}. Internet: {'da' if router.online else 'nu'}.")
    try:
        recorder = SpeechRecorder(trigger=args.trigger, device_name=args.mic).start()
    except (ValueError, sd.PortAudioError) as error:
        print(f"EROARE la microfon: {error}")
        router.close()
        return
    worker = TranscriptionWorker(router, recorder.utterances).start()
    print(f"Gata. Jurnal: {SPEECH_LOG_FILE.as_posix()}")

    last = {'text': '', 'info': '-'}
    try:
        while True:
            while not worker.results.empty():
                report(*worker.results.get(), last)
            cv2.imshow(WINDOW_NAME, draw(recorder, status_lines(router, worker, last)))
            key = cv2.waitKey(30) & 0xFF
            if key == ord(SPEECH_BUTTON_KEY):
                recorder.press()
            elif key == ord('q') or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        recorder.stop()
        worker.stop()
        router.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
