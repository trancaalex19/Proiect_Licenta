"""
Înregistrarea setului de test pentru vorbire -> text (etapa 6): arată pe rând frazele
din data/test_doctor/fraze.txt; pentru fiecare apeși Space și o citești. Se salvează
data/test_doctor/00k.wav + 00k.txt (textul corect), folosite de src/speech/evaluate.py.
Continuă de unde ai rămas: frazele care au deja WAV sunt sărite (cu --again: toate, din nou).

Rulare (din rădăcina proiectului):
    python -m src.speech.record_test_set
    python -m src.speech.record_test_set --again    # reînregistrează toate frazele (suprascrie)
Taste în fereastră: Space = citește fraza, b = refă fraza anterioară, n = sari peste, q = ieșire.
"""
import argparse
import queue

import cv2
import sounddevice as sd

from src.config import MICROPHONE_NAME, SPEECH_BUTTON_KEY, SPEECH_TEST_DIR, SPEECH_TEST_PHRASES_FILE
from src.speech.audio import SpeechRecorder, save_wav
from src.speech.window import ascii_text, draw, wrap

WINDOW_NAME = 'Set de test'


def load_phrases():
    """Frazele din fraze.txt (fără rândurile goale și comentariile cu #)."""
    lines = SPEECH_TEST_PHRASES_FILE.read_text(encoding='utf-8').splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith('#')]


def example_name(k):
    """Fraza k (de la 0) -> '001', '002', ..."""
    return f'{k + 1:03d}'


def first_missing(phrases, start=0, again=False):
    """Următoarea frază de înregistrat, de la `start`: prima fără WAV (sau, cu again, chiar `start`)."""
    if again:
        return min(start, len(phrases))
    for k in range(start, len(phrases)):
        if not (SPEECH_TEST_DIR / f'{example_name(k)}.wav').exists():
            return k
    return len(phrases)


def status_lines(phrases, k, recorder):
    if k >= len(phrases):
        return ['Gata, toate frazele sunt inregistrate!',
                'Evaluare: python -m src.speech.evaluate', '', 'q = iesire']
    return ([f'Fraza {k + 1} din {len(phrases)}  (ignorate, prea slabe: {recorder.rejected_quiet})']
            + wrap(ascii_text(f'"{phrases[k]}"'))
            + ['SPACE = citeste   b = reface anterioara   n = sari   q = iesire'])


def main():
    parser = argparse.ArgumentParser(description='Înregistrarea frazelor de test (WAV + text corect)')
    parser.add_argument('--mic', default=MICROPHONE_NAME, help='parte din numele microfonului')
    parser.add_argument('--again', action='store_true', help='reînregistrează toate frazele (suprascrie)')
    args = parser.parse_args()

    phrases = load_phrases()
    k = first_missing(phrases, again=args.again)
    print(f"{len(phrases)} fraze în {SPEECH_TEST_PHRASES_FILE.as_posix()}; începem cu fraza {k + 1}.")
    try:
        recorder = SpeechRecorder(trigger='button', device_name=args.mic).start()
    except (ValueError, sd.PortAudioError) as error:
        print(f"EROARE la microfon: {error}")
        return

    shown = None
    try:
        while True:
            if k != shown and k < len(phrases):
                print(f"\nFraza {k + 1}: „{phrases[k]}”  (Space, apoi citește)")
                shown = k
            try:
                utterance = recorder.utterances.get_nowait()
            except queue.Empty:
                utterance = None
            if utterance is not None and k < len(phrases):
                name = example_name(k)
                save_wav(SPEECH_TEST_DIR / f'{name}.wav', utterance.pcm, utterance.sample_rate)
                (SPEECH_TEST_DIR / f'{name}.txt').write_text(phrases[k] + '\n', encoding='utf-8')
                print(f"Salvat {name}.wav ({utterance.duration:.1f} s)")
                k = first_missing(phrases, k + 1, args.again)

            cv2.imshow(WINDOW_NAME, draw(recorder, status_lines(phrases, k, recorder)))
            key = cv2.waitKey(30) & 0xFF
            if key == ord(SPEECH_BUTTON_KEY) and k < len(phrases):
                recorder.press()
            elif key == ord('b') and k > 0 and recorder.state == 'repaus':
                k -= 1                              # următoarea înregistrare o suprascrie
            elif key == ord('n') and k < len(phrases) and recorder.state == 'repaus':
                k = first_missing(phrases, k + 1, args.again)
            elif key == ord('q') or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        recorder.stop()
        cv2.destroyAllWindows()
    done = sum((SPEECH_TEST_DIR / f'{example_name(i)}.wav').exists() for i in range(len(phrases)))
    print(f"\nÎnregistrate: {done} din {len(phrases)} fraze.")


if __name__ == '__main__':
    main()
