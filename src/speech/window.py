"""
Fereastra de stare comună programelor de test pentru vorbire (record_test,
record_test_set, live_transcribe): starea microfonului, nivelul lui și câteva rânduri de text.
"""
import unicodedata

import cv2
import numpy as np

WIDTH = 640
TEXT_WIDTH = 60     # caractere pe rând
# textul din fereastră e fără diacritice: cv2.putText nu le poate desena
STATE_TEXT = {
    'repaus': ('Apasa SPACE si vorbeste', (200, 200, 200)),
    'ascultare': ('Te ascult...', (0, 200, 255)),
    'vorbire': ('Inregistrez', (60, 220, 60)),
}


def ascii_text(text):
    """„mă doare” -> „ma doare” (pentru fereastră; în terminal diacriticele rămân)."""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()


def wrap(text, width=TEXT_WIDTH, max_lines=3):
    """Împarte textul pe rânduri de cel mult `width` caractere."""
    lines, line = [], ''
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f'{line} {word}'.strip()
    lines.append(line)
    return lines[:max_lines]


def draw(recorder, lines):
    """Starea microfonului, bara de nivel și rândurile date (fără diacritice)."""
    image = np.full((145 + 25 * len(lines), WIDTH, 3), 30, dtype=np.uint8)
    text, color = STATE_TEXT[recorder.state]
    if recorder.muted:
        text, color = 'Bip (microfon oprit)', (180, 120, 255)
    cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)

    # nivelul microfonului: -60 dB (liniște) ... 0 dB (maxim)
    fill = int(np.clip((recorder.level_db + 60) / 60, 0, 1) * (WIDTH - 40))
    cv2.rectangle(image, (20, 80), (WIDTH - 20, 105), (80, 80, 80), 1)
    cv2.rectangle(image, (20, 80), (20 + fill, 105), color, -1)

    for k, line in enumerate(lines):
        cv2.putText(image, line, (20, 135 + 25 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (220, 220, 220), 1, cv2.LINE_AA)
    return image
