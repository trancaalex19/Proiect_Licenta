"""
Informații despre cele 15 semne ASL: ce înseamnă în română și ce face fiecare mână.
Descrierile sunt pentru o persoană dreptace (stângacii inversează rolul mâinilor).
Semnele au variante regionale; ghidul arată videoclipul unui semnatar real din WLASL.

Câmpuri:
  'right' / 'left' = ce face mâna dreaptă / stângă (semne în care mâinile au roluri diferite)
  'both'           = ce fac ambele mâini (semne în care fac același lucru)
  'then'           = a doua parte a unui semn compus
  'note'           = sfat, de obicei diferența față de un semn ușor de confundat
  'example'        = (opțional) videoclipul afișat implicit, un fișier din
                     data/processed_keypoints/<cuvânt>/. Fără el, ghidul alege exemplul
                     pe care modelul îl recunoaște cu cea mai mare încredere.
"""

RESTING = 'Nu participă: stă jos, relaxată.'

SIGN_INFO = {
    'hello': {
        'meaning': 'Salut / Bună ziua',
        'right': 'Palma deschisă, degetele lipite, lângă tâmplă. Se mișcă în afară și ușor '
                 'înainte, ca un salut militar.',
        'left': RESTING,
    },
    'thankyou': {
        'meaning': 'Mulțumesc',
        'right': 'Palma deschisă, vârful degetelor atinge bărbia. Coboară înainte, spre '
                 'persoana căreia îi mulțumești, și se termină cu palma în sus.',
        'left': RESTING,
    },
    'yes': {
        'meaning': 'Da',
        'right': 'Pumn, în fața pieptului. Îndoaie pumnul din încheietură în sus și în jos de '
                 'câteva ori, ca un cap care spune „da”.',
        'left': RESTING,
        'note': 'Diferența față de NEED: aici mâna e pumn strâns, nu arătătorul îndoit.',
    },
    'no': {
        'meaning': 'Nu',
        'right': 'Arătătorul și mijlociul întinse și lipite, deasupra degetului mare. '
                 'Le închide pe degetul mare de două ori, ca un cioc care se închide.',
        'left': RESTING,
    },
    'help': {
        'meaning': 'Ajutor / A ajuta',
        'right': 'Pumn cu degetul mare în sus, așezat pe palma stângă.',
        'left': 'Palma deschisă, în sus, sub pumnul drept. Ridică pumnul: ambele mâini urcă împreună.',
        'note': 'Diferența față de GOOD: aici dreapta e pumn și mâinile urcă; nu pornește de la bărbie.',
    },
    'please': {
        'meaning': 'Te rog',
        'right': 'Palma DESCHISĂ, lipită de piept. Face cercuri pe piept.',
        'left': RESTING,
        'note': 'Diferența față de SORRY e doar forma mâinii: aici palma e deschisă.',
    },
    'sorry': {
        'meaning': 'Îmi pare rău',
        'right': 'PUMN, lipit de piept. Face cercuri pe piept.',
        'left': RESTING,
        'note': 'Diferența față de PLEASE e doar forma mâinii: aici mâna e pumn.',
    },
    'teacher': {
        'meaning': 'Profesor',
        'both': 'Degetele strânse, cu vârfurile atingându-se, lângă tâmple. Se împing '
                'înainte de două ori („a preda”).',
        'then': 'Ambele palme deschise, față în față, coboară de-a lungul corpului („persoană”).',
    },
    'student': {
        'meaning': 'Student / Elev',
        'right': 'Întâi „learn”: degetele „iau” ceva din palma stângă (se strâng) și mâna urcă la frunte.',
        'left': 'Palma deschisă, în sus, în fața corpului, cât timp dreapta „ia” din ea.',
        'then': 'Ambele palme deschise, față în față, coboară de-a lungul corpului („persoană”).',
    },
    'book': {
        'meaning': 'Carte',
        'both': 'Palmele lipite în fața ta, ca o carte închisă. Se deschid ca o carte; '
                'marginile degetelor mici rămân lipite.',
    },
    'computer': {
        'meaning': 'Calculator',
        'right': 'Degetele curbate în forma literei „C”. Se mișcă în arcuri mici pe antebrațul '
                 'stâng, de la încheietură spre cot.',
        'left': 'Antebrațul ținut orizontal în fața corpului, nemișcat.',
        'note': 'În WLASL apar și alte variante (mâna lângă tâmplă, ambele mâini ridicate): '
                'apasă v ca să le vezi.',
        # WLASL amestecă 3 variante pentru „computer”; afișăm varianta descrisă mai sus
        'example': 'video_7.npy',
    },
    'learn': {
        'meaning': 'A învăța',
        'right': 'Degetele deschise, pe palma stângă, „iau” ceva (se strâng) și mâna urcă la frunte.',
        'left': 'Palma deschisă, în sus, în fața corpului, nemișcată.',
    },
    'need': {
        'meaning': 'A avea nevoie',
        'right': 'Doar ARĂTĂTORUL, îndoit ca un cârlig, cu palma în jos, în fața pieptului. '
                 'Coboară din încheietură de câteva ori.',
        'left': RESTING,
        'note': 'Diferența față de YES: aici e arătătorul îndoit, nu pumnul.',
    },
    'good': {
        'meaning': 'Bun / Bine',
        'right': 'Palma deschisă, vârful degetelor la bărbie. Coboară și se așază, cu palma '
                 'în sus, pe palma stângă.',
        'left': 'Palma deschisă, în sus, în fața corpului; primește mâna dreaptă.',
    },
    'stop': {
        'meaning': 'Stop / Oprește-te',
        'right': 'Palma verticală, cu degetul mic în jos. Lovește scurt, perpendicular, palma stângă.',
        'left': 'Palma deschisă, în sus, în fața corpului, nemișcată.',
    },
}
