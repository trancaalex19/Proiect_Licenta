"""
Informații despre semnele ASL: ce înseamnă în română și ce face fiecare mână.
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
    'need': {
        'meaning': 'A avea nevoie',
        'right': 'Doar ARĂTĂTORUL, îndoit ca un cârlig, cu palma în jos, în fața pieptului. '
                 'Coboară din încheietură de câteva ori.',
        'left': RESTING,
        'note': 'Diferența față de YES: aici e arătătorul îndoit, nu pumnul.',
    },
    'not': {
        'meaning': 'Nu (negație)',
        'right': 'Pumn cu degetul mare în sus, vârful degetului mare sub bărbie. '
                 'Împinge degetul mare înainte, de sub bărbie.',
        'left': RESTING,
    },
    'again': {
        'meaning': 'Din nou / Repetați',
        'right': 'Mâna îndoită (degetele lipite, curbate), cu palma în sus. Se întoarce și '
                 'aterizează cu vârful degetelor în palma stângă.',
        'left': 'Palma deschisă, în sus, în fața corpului. Stă pe loc.',
        'note': 'Diferența față de MEDICINE: aici mâna se întoarce și lovește palma o dată, '
                'acolo mijlociul freacă palma în cercuri.',
    },
    'understand': {
        'meaning': 'A înțelege',
        'right': 'Pumn lângă frunte, palma spre tine. Arătătorul sare brusc în sus, '
                 'ca un „bec care se aprinde”.',
        'left': RESTING,
    },
    'me': {
        'meaning': 'Eu / Pe mine',
        'right': 'Arătătorul întins arată spre propriul piept și îl atinge.',
        'left': RESTING,
        'note': 'Diferența față de I: aici arătătorul atinge pieptul, acolo e degetul mic ridicat.',
        # me_35544 (video_0): MediaPipe etichetează mâna dreaptă ca stângă în câteva cadre
        'example': 'video_1.npy',
    },
    'i': {
        'meaning': 'Eu',
        'right': 'Pumn cu DEGETUL MIC ridicat, în fața pieptului, palma spre stânga.',
        'left': RESTING,
        'note': 'Diferența față de ONE: aici e degetul mic, nu arătătorul.',
    },
    'you': {
        'meaning': 'Tu / Dumneavoastră',
        'right': 'Arătătorul întins arată înainte, spre persoana cu care vorbești.',
        'left': RESTING,
    },
    'my': {
        'meaning': 'Meu / Mea',
        'right': 'Palma deschisă, lipită de piept (o singură atingere, fără cercuri).',
        'left': RESTING,
        'note': 'Diferența față de PLEASE: aici palma doar se așază pe piept, fără să facă cercuri.',
    },
    'child': {
        'meaning': 'Copil',
        'right': 'Palma în jos, întinsă lateral la înălțimea taliei, ca pe capul unui copil. '
                 'Coboară ușor de una-două ori.',
        'left': RESTING,
    },
    'sick': {
        'meaning': 'Bolnav',
        'right': 'Degetul MIJLOCIU îndoit atinge fruntea.',
        'left': 'Degetul MIJLOCIU îndoit atinge stomacul, în același timp.',
    },
    'hurt': {
        'meaning': 'Durere / A durea',
        'both': 'Arătătoarele întinse, unul spre celălalt, în fața pieptului. Se apropie și se '
                'răsucesc de câteva ori (unde e durerea).',
    },
    'headache': {
        'meaning': 'Durere de cap',
        'both': 'Arătătoarele întinse, unul spre celălalt, în fața FRUNȚII. Se apropie și se '
                'răsucesc de câteva ori.',
        'note': 'E semnul HURT făcut în dreptul capului.',
    },
    'sorethroat': {
        'meaning': 'Durere de gât',
        'right': 'Degetul mare și arătătorul (ca un clește) pe gât, sub bărbie; '
                 'alunecă ușor în jos pe gât.',
        'left': RESTING,
        'note': 'Diferența față de ONE: mâna stă lipită de gât, nu ridicată în fața umărului.',
    },
    'cough': {
        'meaning': 'Tuse',
        'right': 'Degetele îndoite (ca o gheară) bat în piept de câteva ori.',
        'left': RESTING,
        'note': 'Diferența față de MY: aici degetele sunt îndoite și bat repetat.',
    },
    'cold': {
        'meaning': 'Răceală / Frig',
        'both': 'Pumni strânși, în fața umerilor, coatele lipite de corp. Tremură scurt, '
                'ca atunci când îți e frig.',
    },
    'hot': {
        'meaning': 'Febră / Cald',
        'right': 'Degetele îndoite (ca o gheară) în fața gurii. Mâna se răsucește rapid și '
                 'se aruncă înainte și în jos, ca și cum ai scăpa de ceva fierbinte.',
        'left': RESTING,
    },
    'vomit': {
        'meaning': 'A vomita',
        'right': 'Mâna deschisă, degetele răsfirate, la gură.',
        'left': 'Mâna deschisă, degetele răsfirate, în fața mâinii drepte. '
                'Ambele mâini pornesc împreună înainte și în jos, din gură.',
    },
    'tired': {
        'meaning': 'Obosit',
        'both': 'Mâinile îndoite, vârfurile degetelor pe piept. Se lasă în jos (se rotesc '
                'spre podea), ca un corp care se prăbușește de oboseală.',
    },
    'sleep': {
        'meaning': 'A dormi',
        'right': 'Mâna deschisă în fața feței, degetele răsfirate. Coboară spre bărbie și se '
                 'strânge (vârfurile degetelor se adună), ca niște ochi care se închid.',
        'left': RESTING,
    },
    'allergy': {
        'meaning': 'Alergie',
        'right': 'Arătătorul atinge nasul.',
        'left': RESTING,
        'then': 'Apoi ambele arătătoare, unul spre celălalt în fața pieptului, '
                'se depărtează spre lateral.',
    },
    'head': {
        'meaning': 'Cap',
        'right': 'Mâna îndoită (degetele lipite) atinge întâi partea de sus a tâmplei, '
                 'apoi coboară și atinge falca, pe aceeași parte.',
        'left': RESTING,
    },
    'stomach': {
        'meaning': 'Stomac',
        'right': 'Palma deschisă, orizontală, bate ușor pe stomac.',
        'left': RESTING,
    },
    'back': {
        'meaning': 'Spate',
        'right': 'Pumn cu degetul mare întins, ridicat lângă umărul drept. Degetul mare arată '
                 'peste umăr, în spate, de câteva ori.',
        'left': RESTING,
    },
    'arm': {
        'meaning': 'Braț',
        'right': 'Palma deschisă alunecă pe antebrațul stâng, de la cot spre încheietură.',
        'left': 'Antebrațul ținut orizontal în fața corpului, palma în jos.',
    },
    'heart': {
        'meaning': 'Inimă / Piept',
        'both': 'Degetele mijlocii desenează pe piept, în partea stângă, conturul unei inimi.',
        'note': 'Unii semnatari doar ating pieptul (partea stângă) cu mijlociul drept.',
    },
    'now': {
        'meaning': 'Acum / Azi',
        'both': 'Mâinile cu degetul mare și cel mic întinse (ca un „telefon”), palmele în sus, '
                'în fața corpului. Coboară scurt, amândouă deodată.',
        'note': 'Diferența față de MANY: aici forma mâinii nu se schimbă (degetul mare + cel mic).',
    },
    'yesterday': {
        'meaning': 'Ieri',
        'right': 'Degetul mare întins (pumn) atinge obrazul lângă colțul gurii, apoi se mută '
                 'înapoi, spre ureche.',
        'left': RESTING,
    },
    'morning': {
        'meaning': 'Dimineață',
        'right': 'Palma deschisă, spre tine. Antebrațul se ridică de la orizontal la vertical, '
                 'ca soarele care răsare.',
        'left': 'Antebrațul orizontal în fața corpului, mâna în îndoitura cotului drept.',
    },
    'night': {
        'meaning': 'Noapte',
        'right': 'Mâna îndoită, palma în jos, coboară în arc peste încheietura stângă, '
                 'ca soarele care apune.',
        'left': 'Antebrațul orizontal în fața corpului, palma în jos. Stă pe loc.',
        'note': 'Diferența față de AGAIN: aici mâna dreaptă trece PESTE încheietura stângă, '
                'nu aterizează în palmă.',
    },
    'day': {
        'meaning': 'Zi',
        'right': 'Arătătorul întins în sus, cotul sprijinit pe mâna stângă. Antebrațul coboară '
                 'în arc până ajunge culcat pe brațul stâng.',
        'left': 'Antebrațul orizontal în fața corpului, palma în jos, sub cotul drept.',
    },
    'week': {
        'meaning': 'Săptămână',
        'right': 'Arătătorul întins alunecă peste palma stângă, de la podul palmei spre degete.',
        'left': 'Palma deschisă, în sus (sau spre dreapta), în fața pieptului.',
        'note': 'Diferența față de AGAIN: aici arătătorul alunecă drept peste palmă, nu se întoarce.',
    },
    'month': {
        'meaning': 'Lună',
        'right': 'Arătătorul întins, orizontal, alunecă în jos pe spatele arătătorului stâng.',
        'left': 'Arătătorul întins în SUS, vertical, în fața pieptului.',
    },
    'ago': {
        'meaning': 'În urmă',
        'right': 'Palma deschisă ridicată lângă umăr. Se mișcă înapoi, peste umăr.',
        'left': RESTING,
    },
    'since': {
        'meaning': 'De când',
        'both': 'Arătătoarele întinse, lângă umărul drept / piept. Se mișcă în arc înainte '
                'și în jos, până arată înainte.',
    },
    'one': {
        'meaning': 'Unu',
        'right': 'Doar ARĂTĂTORUL întins în sus, palma spre tine, în fața umărului.',
        'left': RESTING,
    },
    'two': {
        'meaning': 'Doi',
        'right': 'ARĂTĂTORUL și MIJLOCIUL întinse în sus (V), palma spre tine.',
        'left': RESTING,
        'note': 'Diferența față de ONE: două degete, nu unul.',
    },
    'three': {
        'meaning': 'Trei',
        'right': 'DEGETUL MARE, ARĂTĂTORUL și MIJLOCIUL întinse, palma spre tine.',
        'left': RESTING,
        'note': 'Diferența față de WATER: aici degetul mare e întins și mâna stă în fața '
                'umărului, nu la bărbie.',
    },
    'many': {
        'meaning': 'Mult / Mulți',
        'both': 'Pumni strânși, palmele în sus, în fața corpului. Degetele se deschid brusc '
                '(răsfirate), de câteva ori.',
        'note': 'Diferența față de NOW: aici mâinile se deschid din pumn.',
    },
    'better': {
        'meaning': 'Mai bine',
        'right': 'Palma deschisă atinge bărbia, apoi pleacă spre dreapta și în sus, '
                 'închizându-se în pumn cu degetul mare în sus.',
        'left': RESTING,
        'note': 'Diferența față de ONE: pornește de la bărbie și se termină cu degetul MARE în sus.',
    },
    'medicine': {
        'meaning': 'Medicament',
        'right': 'Degetul MIJLOCIU îndoit, în palma stângă. Freacă palma în cercuri mici.',
        'left': 'Palma deschisă, în sus, în fața corpului.',
        'note': 'Diferența față de AGAIN: aici mijlociul rămâne în palmă și face cercuri.',
    },
    'take': {
        'meaning': 'A lua',
        'right': 'Mâna deschisă, întinsă înainte. Se închide în pumn în timp ce vine spre tine, '
                 'ca și cum ai apuca ceva.',
        'left': RESTING,
    },
    'eat': {
        'meaning': 'A mânca',
        'right': 'Vârfurile degetelor adunate (ca un cioc), atinge gura de câteva ori.',
        'left': RESTING,
    },
    'drink': {
        'meaning': 'A bea',
        'right': 'Mâna în formă de C, ca și cum ai ține un pahar, la gură. Se înclină spre gură, '
                 'ca și cum ai bea.',
        'left': RESTING,
    },
    'water': {
        'meaning': 'Apă',
        'right': 'ARĂTĂTORUL, MIJLOCIUL și INELARUL întinse (W), degetul mare ține degetul mic. '
                 'Arătătorul atinge bărbia de două ori.',
        'left': RESTING,
        'note': 'Diferența față de THREE: aici degetul mare e strâns, iar mâna stă la bărbie.',
    },
    'what': {
        'meaning': 'Ce',
        'both': 'Palmele deschise, în sus, în fața corpului. Se mișcă ușor lateral, '
                'ca un gest de „ce?”.',
    },
}
