"""
Glosele semnate de pacient -> o propoziție în română, pentru medic (afișată și citită cu voce,
src/speech/tts.py). Funcție pură, cu reguli simple, fără model:

  - fiecare semn are forma spusă de pacient, la persoana I („tușesc”, „am febră”) sau a III-a
    când vorbește despre copil („copilul meu are febră”);
  - durere + parte a corpului, în orice ordine -> „mă doare capul”;
  - număr + zi/săptămână/lună -> „două zile”; urmate de ago -> „acum două săptămâni”; since -> „de”;
  - negația ASL (not, de obicei la final) -> „nu” înaintea primului verb;
  - cu yesterday / ago, verbele la trecut; „am nevoie de”, „am alergie la”; virgulă după da, nu...
De ce: înșiruirea sensurilor („Eu, durere de cap, de ieri”) e greu de înțeles când e citită de
vocea sintetică (Whisper a auzit „e udurele de cap”), iar propoziția normală e înțeleasă perfect.

    python -m src.translate.gloss_to_text me headache since yesterday
"""
import argparse

# Semnele care sunt verbe (forma persoana I, forma persoana a III-a); conțin deja subiectul
VERBS = {
    'hurt': ('mă doare', 'îl doare'), 'headache': ('mă doare capul', 'îl doare capul'),
    'sorethroat': ('mă doare gâtul', 'îl doare gâtul'), 'cough': ('tușesc', 'tușește'),
    'cold': ('mi-e frig', 'îi e frig'), 'hot': ('am febră', 'are febră'), 'vomit': ('am vomitat', 'a vomitat'),
    'tired': ('sunt obosit', 'este obosit'), 'sleep': ('dorm', 'doarme'), 'allergy': ('am alergie', 'are alergie'),
    'sick': ('sunt bolnav', 'este bolnav'), 'need': ('am nevoie', 'are nevoie'), 'take': ('iau', 'ia'),
    'eat': ('mănânc', 'mănâncă'), 'drink': ('beau', 'bea'), 'understand': ('înțeleg', 'înțelege'),
    'better': ('mă simt mai bine', 'se simte mai bine'),
}
# Trecutul (când propoziția are yesterday / ago): „Ieri am vomitat”, nu „Ieri vomit”
PAST = {
    'hurt': ('m-a durut', 'l-a durut'), 'headache': ('m-a durut capul', 'l-a durut capul'),
    'sorethroat': ('m-a durut gâtul', 'l-a durut gâtul'), 'cough': ('am tușit', 'a tușit'),
    'cold': ('mi-a fost frig', 'i-a fost frig'), 'hot': ('am avut febră', 'a avut febră'),
    'tired': ('am fost obosit', 'a fost obosit'), 'sleep': ('am dormit', 'a dormit'),
    'sick': ('am fost bolnav', 'a fost bolnav'), 'take': ('am luat', 'a luat'), 'eat': ('am mâncat', 'a mâncat'),
    'drink': ('am băut', 'a băut'),
}
# Verbe urmate de prepoziție înaintea unui substantiv: „am nevoie de medicament”, „am alergie la apă”
VERB_PREPOSITION = {'need': 'de', 'allergy': 'la'}
NOUNS = {'medicine': 'medicament', 'water': 'apă'}   # forma de după prepoziție
# Cuvinte după care urmează virgulă dacă propoziția continuă („Îmi pare rău, nu înțeleg.”)
INTERJECTIONS = {'yes', 'no', 'please', 'thankyou', 'sorry'}
TIME_FIRST = {'yesterday', 'now', 'morning', 'night'}   # timpul pus la început (ordinea ASL)
BODY = {'head': 'capul', 'stomach': 'stomacul', 'back': 'spatele', 'arm': 'brațul', 'heart': 'pieptul'}
# zi, săptămână, lună sunt feminine: „o zi”, „două zile”
NUMBERS = {'one': 'o', 'two': 'două', 'three': 'trei'}
UNITS = {'day': ('zi', 'zile'), 'week': ('săptămână', 'săptămâni'), 'month': ('lună', 'luni')}
OTHER = {
    'yes': 'da', 'no': 'nu', 'please': 'vă rog', 'thankyou': 'mulțumesc', 'sorry': 'îmi pare rău',
    'again': 'din nou', 'help': 'ajutor', 'me': 'eu', 'i': 'eu', 'you': 'dumneavoastră', 'my': 'meu',
    'child': 'copilul', 'now': 'acum', 'yesterday': 'ieri', 'morning': 'dimineața', 'night': 'noaptea',
    'day': 'ziua', 'week': 'săptămâna', 'month': 'luna', 'ago': 'în urmă', 'since': 'de',
    'one': 'unu', 'two': 'doi', 'three': 'trei', 'many': 'mult', 'medicine': 'medicamentul',
    'water': 'apă', 'what': 'ce', 'head': 'capul', 'stomach': 'stomacul', 'back': 'spatele',
    'arm': 'brațul', 'heart': 'pieptul', 'not': 'nu',
}


def gloss_to_text(gloss):
    """['me', 'headache', 'since', 'yesterday'] -> 'Mă doare capul de ieri.'"""
    words = [g for g in gloss if g != 'not']
    negate = 'not' in gloss and len(words) > 0
    if not words:
        return 'Nu.' if gloss else ''
    third = 'child' in words                 # vorbește despre copil: verbele la persoana a III-a
    # „de ieri” (since yesterday) = încă durează -> prezent; „ieri” / „acum două săptămâni” -> trecut
    past = 'ago' in words or ('yesterday' in words and 'since' not in words)
    forms = {**VERBS, **PAST} if past else VERBS
    out, verb_done, i = [], False, 0

    def verb(text):
        nonlocal verb_done
        out.append(f'nu {text}' if negate and not verb_done else text)
        verb_done = True

    while i < len(words):
        w, nxt = words[i], words[i + 1] if i + 1 < len(words) else None
        after = words[i + 2] if i + 2 < len(words) else None
        if w in ('me', 'i') and (nxt in VERBS or nxt in BODY):
            i += 1                             # „eu mă doare” -> „mă doare”
        elif w == 'my' and nxt == 'child':
            out.append('copilul meu')
            i += 2
        elif w == 'hurt' and nxt in BODY or w in BODY and nxt == 'hurt':
            part = BODY[nxt if w == 'hurt' else w]
            verb(f"{forms['hurt'][third]} {part}")
            i += 2
        elif w in NUMBERS and nxt in UNITS:
            unit = UNITS[nxt][0 if w == 'one' else 1]
            phrase = f'{NUMBERS[w]} {unit}'
            if after == 'ago':
                out.append(f'acum {phrase}')
                i += 3
            else:
                out.append(phrase)
                i += 2
        elif w == 'since' and nxt is None:
            out.append('de când')              # întrebare: „de când?”
            i += 1
        elif w in VERB_PREPOSITION and nxt in NOUNS:
            verb(f'{VERBS[w][third]} {VERB_PREPOSITION[w]} {NOUNS[nxt]}')
            i += 2
        elif w in VERBS:
            verb(forms[w][third])
            i += 1
        else:
            out.append(OTHER.get(w, w) + (',' if w in INTERJECTIONS and nxt is not None else ''))
            i += 1
    if negate and not verb_done:
        out.insert(0, 'nu')
    # pauză după timpul de la început („Ieri, am vomitat”): vocea sintetică altfel le leagă („eram”)
    if len(out) > 1 and (words[0] in TIME_FIRST or out[0].startswith('acum ')) and not out[0].endswith(','):
        out[0] += ','
    text = ' '.join(out)
    end = '?' if 'what' in words or words[-1] == 'since' else '.'
    return text[0].upper() + text[1:] + end


def main():
    parser = argparse.ArgumentParser(description='Glose ASL -> propoziție în română')
    parser.add_argument('gloss', nargs='+')
    print(gloss_to_text(parser.parse_args().gloss))


if __name__ == '__main__':
    main()
