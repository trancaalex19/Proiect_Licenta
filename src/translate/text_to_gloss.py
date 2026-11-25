"""
Text în română (ce a spus medicul, din transcriere) -> lista de glose ASL din vocabular,
pentru avatar. Funcție pură (fără microfon, fără rețea): text_to_gloss(text).

  Pas 1: potrivire aproximativă cu frazele de test din src/avatar/web/sentences.js (câmpul
         text). Dacă cea mai apropiată frază trece de TRANSLATE_SENTENCE_THRESHOLD, se
         folosesc glosele ei (scrise de mână, deci în ordinea ASL). Doar dacă are aceleași
         cuvinte-cheie ca textul (fără „you”): altfel „Nu ați mâncat azi?” ar lua glosele lui
         „Ați mâncat azi?” (scor 0.90) și negația s-ar pierde; la fel „două” / „trei” zile.
  Pas 2: cuvinte-cheie, în ordinea din text: sensurile în română din src/app/sign_info.py,
         SPOKEN_SYNONYMS din src/speech/vocabulary.py și câteva forme flexionate uzuale
         (EXTRA_FORMS: „doare”, „capul”, „tușiți”...).
Cuvintele fără semn se ignoră și se întorc în `unknown` (apelantul le scrie în jurnal).

Verificare rapidă:
    python -m src.translate.text_to_gloss "Vă doare capul de ieri?"
"""
import argparse
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from src.app.sign_info import SIGN_INFO
from src.config import (ACTIONS, TRANSLATE_PREFIX_MIN_CHARS, TRANSLATE_SENTENCE_THRESHOLD,
                        TRANSLATE_SENTENCES_FILE)
from src.speech.vocabulary import SPOKEN_SYNONYMS

# Cuvinte din SPOKEN_SYNONYMS -> semnul lor (fiecare sinonim trebuie să aibă un semn aici)
SYNONYM_GLOSS = {'burtă': 'stomach', 'abdomen': 'stomach', 'pastile': 'medicine'}

# Forme spuse de medic care nu apar ca atare în sign_info (flexiuni, sinonime, persoana a 2-a).
# Au prioritate față de sensurile din sign_info (ex. „nu” într-o frază = negația „not”).
EXTRA_FORMS = {
    'you': ['vă', 'dumneavoastră', 'dumneata', 'tu', 'îți', 'ți'],
    'please': ['vă rog'],
    'sorry': ['scuze', 'scuzați'],
    'thankyou': ['mersi'],
    'not': ['nu'],
    'help': ['ajut', 'ajutați', 'ajutor'],
    'need': ['nevoie', 'trebuie'],
    'understand': ['înțeles', 'înțelegeți'],
    'again': ['repeta'],
    'me': ['mine'],
    'my': ['mei', 'mele'],
    'sick': ['boală'],
    'hurt': ['doare', 'dor', 'durut', 'dureri'],
    'sorethroat': ['durere în gât', 'doare gâtul', 'gâtul', 'gât'],
    'cough': ['tușiți', 'tușit', 'tușește', 'tușesc'],
    'cold': ['răcit', 'răciți'],
    'hot': ['cald', 'temperatură'],
    'vomit': ['vărsat', 'vomă'],
    'tired': ['oboseală'],
    'sleep': ['somn', 'odihniți', 'odihnă'],
    'allergy': ['alergic'],
    'head': ['capul'],
    'stomach': ['stomacul'],
    'now': ['astăzi'],
    'night': ['seara', 'seară'],
    'day': ['zile', 'ziua'],
    'week': ['săptămâni'],
    'month': ['luni'],
    'two': ['două'],
    'one': ['una'],
    'many': ['des'],
    'medicine': ['pastilă', 'pastila', 'tratament'],
    'take': ['luați', 'luat', 'iau', 'luăm'],
    'eat': ['mănâncă', 'mâncare'],
    'drink': ['beți', 'băut', 'bea'],
}

# Formule de politețe: fără semn, dar altfel ar deveni semne greșite („ziua” -> day)
IGNORED_PHRASES = ['bună ziua', 'bună seara', 'luați loc', 'la revedere']

# Cuvinte de legătură: fără semn, dar nici nu se trec în jurnal ca necunoscute
STOPWORDS = set('''a ai al ale am are au aveți ați ca care cu cum de din e este fi fost i în
    la le lui mai mă o pe pentru sau să se sunt sunteți și un unei unui vreo vreun într dintr'''.split())
# Semnul implicit în verbul la persoana a 2-a („ați vomitat” = you vomit): nu contează la pasul 1
IMPLICIT_GLOSS = {'you'}


@dataclass
class GlossResult:
    gloss: list                         # glosele, în ordinea semnelor (pot lipsi: listă goală)
    method: str                         # 'frază' (pasul 1), 'cuvinte' (pasul 2) sau 'nimic'
    score: float = 0.0                  # asemănarea cu cea mai apropiată frază din sentences.js
    sentence: str = None                # fraza din sentences.js folosită (doar la 'frază')
    unknown: list = field(default_factory=list)   # cuvintele fără semn (ignorate)


def fold(text):
    """Pentru comparare: litere mici, fără diacritice, fără punctuație. „Vă doare?” -> „va doare”."""
    text = unicodedata.normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
    return ' '.join(re.findall(r'[a-z0-9]+', text))


STOPWORDS_FOLDED = {fold(word) for word in STOPWORDS}


@lru_cache(maxsize=1)
def load_sentences(path=TRANSLATE_SENTENCES_FILE):
    """Frazele din sentences.js: listă de (text, glose). Fișierul are câte o frază pe rând."""
    pattern = re.compile(r"gloss:\s*\[([^\]]*)\][^}]*?text:\s*'((?:[^'\\]|\\.)*)'")
    sentences = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        match = pattern.search(line)
        if match:
            gloss = re.findall(r"'([^']+)'", match.group(1))
            sentences.append((match.group(2).replace("\\'", "'"), tuple(gloss)))
    return tuple(sentences)


@lru_cache(maxsize=1)
def keyword_table():
    """{(cuvinte...): glosă}, cu cuvintele fără diacritice; ex. ('durere', 'de', 'cap'): 'headache'."""
    table = {}
    for action in ACTIONS:                               # sensurile din sign_info (primul câștigă)
        meaning = re.sub(r'\(.*?\)', '', SIGN_INFO[action]['meaning'])
        for form in meaning.split('/'):
            words = fold(form).split()
            if words and words[0] == 'a' and len(words) > 1:   # „a durea” -> „durea”
                words = words[1:]
            if words:
                table.setdefault(tuple(words), action)
    missing = set(SPOKEN_SYNONYMS) - set(SYNONYM_GLOSS)
    assert not missing, f'SPOKEN_SYNONYMS fără semn în SYNONYM_GLOSS: {missing}'
    extra = [(form, gloss) for form, gloss in SYNONYM_GLOSS.items()]
    extra += [(form, gloss) for gloss, forms in EXTRA_FORMS.items() for form in forms]
    for form, gloss in extra:
        assert gloss in ACTIONS, gloss
        table[tuple(fold(form).split())] = gloss
    for phrase in IGNORED_PHRASES:
        table[tuple(fold(phrase).split())] = None
    return table


def _word_matches(word, key):
    """Cuvântul din text se potrivește cu cel din dicționar: identic, sau începe cu el dacă are
    cel puțin TRANSLATE_PREFIX_MIN_CHARS litere („medicament” -> „medicamentul”)."""
    return word == key or (len(key) >= TRANSLATE_PREFIX_MIN_CHARS and word.startswith(key))


def keywords_to_gloss(text):
    """Pasul 2: (glose, cuvinte necunoscute). La fiecare poziție se ia potrivirea cea mai lungă."""
    # cuvintele originale (pentru jurnal) și variantele fără diacritice (pentru căutare)
    original = [w for w in re.findall(r'[^\W_]+', text.lower()) if fold(w)]
    words = [fold(w) for w in original]
    table = keyword_table()
    if words == ['nu']:
        return ['no'], []                                # „Nu.” singur = răspunsul „nu”
    gloss, unknown, i = [], [], 0
    while i < len(words):
        best = None                                      # (nr. cuvinte, nr. litere, glosă)
        for key, action in table.items():
            n = len(key)
            if all(i + k < len(words) and _word_matches(words[i + k], key[k]) for k in range(n)):
                candidate = (n, sum(map(len, key)), action)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
        if best is None:
            if words[i] not in STOPWORDS_FOLDED:
                unknown.append(original[i])
            i += 1
            continue
        if best[2] is not None and (not gloss or gloss[-1] != best[2]):   # fără același semn de două ori la rând
            gloss.append(best[2])
        i += best[0]
    return gloss, unknown


def _key_signs(text):
    """Semnele-cheie ale unui text (pasul 2), ca mulțime, fără cele implicite."""
    return set(keywords_to_gloss(text)[0]) - IMPLICIT_GLOSS


def best_sentence(text, sentences=None):
    """Pasul 1: (scor, text, glose) pentru cea mai apropiată frază din sentences.js care are
    aceleași semne-cheie ca textul."""
    target, signs = fold(text), _key_signs(text)
    best = (0.0, None, ())
    for sentence, gloss in (sentences if sentences is not None else load_sentences()):
        score = SequenceMatcher(None, target, fold(sentence)).ratio()
        if score > best[0] and _key_signs(sentence) == signs:
            best = (score, sentence, gloss)
    return best


def text_to_gloss(text, threshold=TRANSLATE_SENTENCE_THRESHOLD, sentences=None):
    """Textul medicului -> GlossResult (glose din vocabular, în ordinea semnelor)."""
    if not fold(text):
        return GlossResult([], 'nimic')
    score, sentence, gloss = best_sentence(text, sentences)
    if sentence is not None and score >= threshold:
        return GlossResult(list(gloss), 'frază', score, sentence)
    gloss, unknown = keywords_to_gloss(text)
    return GlossResult(gloss, 'cuvinte' if gloss else 'nimic', score, None, unknown)


def main():
    parser = argparse.ArgumentParser(description='Text în română -> glose ASL')
    parser.add_argument('text', nargs='+')
    args = parser.parse_args()
    result = text_to_gloss(' '.join(args.text))
    print(f"glose: {' '.join(result.gloss) or '-'}   (metoda: {result.method}, scor frază {result.score:.2f})")
    if result.sentence:
        print(f"fraza: {result.sentence}")
    if result.unknown:
        print(f"necunoscute: {', '.join(result.unknown)}")


if __name__ == '__main__':
    main()
