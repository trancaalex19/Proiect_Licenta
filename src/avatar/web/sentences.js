// Propoziții de test (medic -> pacient): glosele ASL în ordinea semnelor și textul în română,
// grupate pe teme (în pagină: lista „Propoziție”). Deocamdată scrise de mână; etapa text -> glose
// le va genera din ce spune medicul. Doar cuvinte din vocabular (data/avatar_motion/index.json).
// Ordinea ASL: timpul la început (yesterday, now, morning...), negația la sfârșit (not).
export const SENTENCES = [
  // durere: unde și de când
  { group: 'Durere', gloss: ['you', 'hurt', 'what'], text: 'Ce vă doare?' },
  { group: 'Durere', gloss: ['you', 'head', 'hurt'], text: 'Vă doare capul?' },
  { group: 'Durere', gloss: ['you', 'stomach', 'hurt'], text: 'Vă doare stomacul?' },
  { group: 'Durere', gloss: ['you', 'back', 'hurt'], text: 'Vă doare spatele?' },
  { group: 'Durere', gloss: ['you', 'arm', 'hurt'], text: 'Vă doare brațul?' },
  { group: 'Durere', gloss: ['you', 'heart', 'hurt'], text: 'Vă doare în piept?' },
  { group: 'Durere', gloss: ['you', 'headache', 'since'], text: 'De când vă doare capul?' },
  { group: 'Durere', gloss: ['you', 'sorethroat', 'since'], text: 'De când vă doare gâtul?' },
  { group: 'Durere', gloss: ['you', 'hurt', 'since', 'three', 'day'], text: 'Vă doare de trei zile?' },
  // simptome
  { group: 'Simptome', gloss: ['yesterday', 'you', 'vomit'], text: 'Ați vomitat ieri?' },
  { group: 'Simptome', gloss: ['you', 'vomit', 'many'], text: 'Ați vomitat de multe ori?' },
  { group: 'Simptome', gloss: ['you', 'cough', 'many'], text: 'Tușiți mult?' },
  { group: 'Simptome', gloss: ['you', 'cough', 'since', 'two', 'week'], text: 'Tușiți de două săptămâni?' },
  { group: 'Simptome', gloss: ['yesterday', 'you', 'hot'], text: 'Ați avut febră ieri?' },
  { group: 'Simptome', gloss: ['you', 'cold'], text: 'Vă este frig?' },
  { group: 'Simptome', gloss: ['you', 'tired'], text: 'Sunteți obosit?' },
  { group: 'Simptome', gloss: ['night', 'you', 'sleep'], text: 'Dormiți noaptea?' },
  { group: 'Simptome', gloss: ['now', 'you', 'eat'], text: 'Ați mâncat azi?' },
  { group: 'Simptome', gloss: ['you', 'allergy', 'medicine'], text: 'Aveți alergie la vreun medicament?' },
  { group: 'Simptome', gloss: ['two', 'week', 'ago', 'you', 'sick'], text: 'Ați fost bolnav acum două săptămâni?' },
  { group: 'Simptome', gloss: ['child', 'sick'], text: 'Copilul este bolnav?' },
  { group: 'Simptome', gloss: ['child', 'hot', 'since', 'yesterday'], text: 'Copilul are febră de ieri?' },
  // tratament și sfaturi
  { group: 'Tratament', gloss: ['you', 'take', 'medicine', 'morning', 'night'], text: 'Luați medicamentul dimineața și seara.' },
  { group: 'Tratament', gloss: ['morning', 'you', 'take', 'medicine', 'one'], text: 'Luați o pastilă dimineața.' },
  { group: 'Tratament', gloss: ['you', 'take', 'medicine', 'three', 'day'], text: 'Luați medicamentul trei zile.' },
  { group: 'Tratament', gloss: ['you', 'need', 'drink', 'water'], text: 'Trebuie să beți apă.' },
  { group: 'Tratament', gloss: ['you', 'drink', 'water', 'many'], text: 'Beți multă apă.' },
  { group: 'Tratament', gloss: ['you', 'need', 'sleep', 'many'], text: 'Trebuie să vă odihniți mult.' },
  { group: 'Tratament', gloss: ['now', 'you', 'eat', 'not'], text: 'Azi nu mâncați.' },
  { group: 'Tratament', gloss: ['one', 'week', 'you', 'better'], text: 'Într-o săptămână vă simțiți mai bine.' },
  { group: 'Tratament', gloss: ['now', 'you', 'better'], text: 'Acum vă simțiți mai bine?' },
  // comunicare
  { group: 'Comunicare', gloss: ['you', 'understand'], text: 'Ați înțeles?' },
  { group: 'Comunicare', gloss: ['you', 'need', 'help'], text: 'Aveți nevoie de ajutor?' },
  { group: 'Comunicare', gloss: ['me', 'help', 'you'], text: 'Vă ajut eu.' },
  { group: 'Comunicare', gloss: ['please', 'again'], text: 'Repetați, vă rog.' },
  { group: 'Comunicare', gloss: ['sorry', 'i', 'understand', 'not'], text: 'Îmi pare rău, nu am înțeles.' },
  { group: 'Comunicare', gloss: ['yes', 'thankyou'], text: 'Da, mulțumesc.' },
];
