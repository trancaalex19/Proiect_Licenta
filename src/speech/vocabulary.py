"""
Prompt-ul cu vocabularul medical, dat lui Whisper (Groq și local) ca să recunoască
mai sigur cuvintele care contează: cele pentru care avem semne. Se construiește din
sensurile în română din src/app/sign_info.py, deci un cuvânt nou ajunge automat aici.
"""
import re

from src.app.sign_info import SIGN_INFO
from src.config import ACTIONS

PROMPT_INTRO = 'Consultație la medicul de familie.'
# Cuvintele scurte și uzuale (da, nu, eu, cap, apă...) Whisper le recunoaște oricum;
# le lăsăm deoparte ca prompt-ul să încapă în limita de 224 de tokeni.
PROMPT_MIN_WORD_LENGTH = 4
# Cuvinte uzuale pe care medicul le spune în loc de cele din sign_info (ex. „burtă” pentru
# „stomac”); fără ele, modelul local scrie „burda”.
SPOKEN_SYNONYMS = ['burtă', 'abdomen', 'pastile']


def vocabulary_words():
    """Cuvintele în română, în ordinea din ACTIONS, fără dubluri (ex. 'Durere / A durea')."""
    words = []
    for action in ACTIONS:
        meaning = re.sub(r'\(.*?\)', '', SIGN_INFO[action]['meaning'])   # fără explicații
        for word in meaning.split('/'):
            word = word.strip()
            word = word[0].lower() + word[1:] if word else word          # 'Durere' -> 'durere'
            word = word[2:] if word.startswith('a ') else word            # 'a durea' -> 'durea' (tokeni mai puțini)
            if word and word not in words:
                words.append(word)
    return words


def build_prompt():
    """
    Ex.: 'Consultație la medicul de familie. Cuvinte: da, nu, te rog, ...'
    Whisper acceptă cel mult 224 de tokeni în prompt (verificat cu tokenizer-ul Whisper:
    python -m src.speech.transcribe_file --show-prompt).
    """
    words = [w for w in vocabulary_words() if len(w) >= PROMPT_MIN_WORD_LENGTH] + SPOKEN_SYNONYMS
    return f"{PROMPT_INTRO} Cuvinte: {', '.join(words)}."
