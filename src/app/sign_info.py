"""
Informații despre cele 15 semne ASL: ce înseamnă în română și cum se execută.
Descrierile sunt pentru o persoană dreptace (stângacii inversează rolul mâinilor).
Semnele au variante regionale; animația din ghid arată un semnatar real din WLASL.

Cheia opțională 'example' fixează videoclipul afișat implicit (un fișier din
data/processed_keypoints/<cuvânt>/). Fără ea, ghidul alege exemplul pe care
modelul îl recunoaște cu cea mai mare încredere.
"""

SIGN_INFO = {
    'hello': {
        'meaning': 'Salut / Bună ziua',
        'how': 'Mâna dreaptă deschisă, cu degetele lipite, lângă tâmplă. '
               'Mișc-o în afară și ușor înainte, ca un salut militar.',
    },
    'thankyou': {
        'meaning': 'Mulțumesc',
        'how': 'Mâna deschisă, cu vârful degetelor atingând bărbia. '
               'Coboară mâna înainte, spre persoana căreia îi mulțumești, cu palma în sus.',
    },
    'yes': {
        'meaning': 'Da',
        'how': 'Fă pumn. Îndoaie pumnul din încheietură în sus și în jos de câteva ori, '
               'ca un cap care spune „da”.',
    },
    'no': {
        'meaning': 'Nu',
        'how': 'Arătătorul și mijlociul întinse, deasupra degetului mare. '
               'Închide-le de două ori pe degetul mare, ca un cioc care se închide.',
    },
    'help': {
        'meaning': 'Ajutor / A ajuta',
        'how': 'Mâna stângă deschisă, cu palma în sus. Pune pe ea pumnul drept, cu degetul mare '
               'în sus, și ridică ambele mâini împreună.',
    },
    'please': {
        'meaning': 'Te rog',
        'how': 'Palma dreaptă deschisă pe piept. Fă cercuri pe piept.',
    },
    'sorry': {
        'meaning': 'Îmi pare rău',
        'how': 'Pumnul drept pe piept. Fă cercuri pe piept.',
    },
    'teacher': {
        'meaning': 'Profesor',
        'how': 'Ambele mâini cu degetele strânse lângă tâmple; împinge-le înainte de două ori '
               '(„a preda”). Apoi palmele deschise, față în față, coboară de-a lungul corpului '
               '(semnul pentru „persoană”).',
    },
    'student': {
        'meaning': 'Student / Elev',
        'how': 'Fă semnul pentru „learn” (a învăța), apoi semnul pentru „persoană”: '
               'palmele deschise, față în față, coboară de-a lungul corpului.',
    },
    'book': {
        'meaning': 'Carte',
        'how': 'Palmele lipite în fața ta, ca o carte închisă. Deschide-le, păstrând '
               'marginile degetelor mici lipite, ca și cum ai deschide o carte.',
    },
    'computer': {
        'meaning': 'Calculator',
        'how': 'Mâna dreaptă în formă de „C”. Mișc-o în arcuri mici pe antebrațul stâng, '
               'de la încheietură spre cot. În WLASL apar și alte variante (mâna lângă tâmplă, '
               'ambele mâini ridicate): apasă v ca să le vezi.',
        # WLASL amestecă 3 variante pentru „computer”; afișăm varianta descrisă mai sus
        'example': 'video_7.npy',
    },
    'learn': {
        'meaning': 'A învăța',
        'how': 'Mâna stângă deschisă, cu palma în sus. Cu mâna dreaptă „iei” ceva din palmă '
               '(degetele se strâng) și duci mâna la frunte.',
    },
    'need': {
        'meaning': 'A avea nevoie',
        'how': 'Arătătorul drept îndoit ca un cârlig, cu palma în jos. '
               'Coboară mâna din încheietură de câteva ori.',
    },
    'good': {
        'meaning': 'Bun / Bine',
        'how': 'Mâna dreaptă deschisă, cu vârful degetelor la bărbie. Coboar-o și '
               'pune-o, cu palma în sus, pe palma stângă.',
    },
    'stop': {
        'meaning': 'Stop / Oprește-te',
        'how': 'Mâna stângă deschisă, cu palma în sus. Lovește-o scurt, perpendicular, '
               'cu marginea mâinii drepte (degetul mic în jos).',
    },
}
