"""
Configurația centrală a proiectului: clase, căi și constante comune.
Toate scripturile importă de aici, ca să nu existe copii care se pot desincroniza.
Căile sunt relative la rădăcina proiectului (scripturile se rulează din rădăcină).
"""
import os
from pathlib import Path

# Cele 51 de cuvinte (scenariul: pacient surd la medicul de familie).
# 'me' și 'i' sunt clase separate: în ASL au semne diferite.
# ORDINEA contează: indexul din listă este eticheta clasei.
ACTIONS = [
    # dialog
    'yes', 'no', 'not', 'please', 'thankyou', 'sorry', 'again', 'understand', 'help', 'need',
    # persoane
    'me', 'i', 'you', 'my', 'child',
    # simptome
    'sick', 'hurt', 'headache', 'sorethroat', 'cough', 'cold', 'hot', 'vomit', 'tired', 'sleep', 'allergy',
    # corp
    'head', 'stomach', 'back', 'arm', 'heart',
    # timp
    'now', 'yesterday', 'morning', 'night', 'day', 'week', 'month', 'ago', 'since',
    # cantitate și stare
    'one', 'two', 'three', 'many', 'better',
    # tratament
    'medicine', 'take', 'eat', 'drink', 'water',
    # întrebări
    'what',
]
NUM_CLASSES = len(ACTIONS)
LABEL_MAP = {action: i for i, action in enumerate(ACTIONS)}   # cuvânt -> indexul clasei

# Căi
DATA_DIR = 'data'
WLASL_JSON = os.path.join(DATA_DIR, 'WLASL_v0.3.json')
RAW_VIDEOS_DIR = os.path.join(DATA_DIR, 'raw_videos')
KEYPOINTS_DIR = os.path.join(DATA_DIR, 'processed_keypoints')
SPLITS_FILE = os.path.join(DATA_DIR, 'splits.json')
# Videoclipuri eliminate și exemple făcute cu mâna stângă (lista 'mirrored', oglindite la încărcare)
CLEANING_LOG_FILE = os.path.join(DATA_DIR, 'cleaning_log.json')
# Ce mp4 corespunde fiecărui video_<i>.npy (scris de extract_features.py; folosit de ghid)
VIDEO_INDEX_FILE = os.path.join(DATA_DIR, 'video_index.json')
# Înregistrări proprii (src/data/record_samples.py): folosite DOAR la antrenare,
# niciodată în validare sau test, ca evaluarea să rămână pe semnatari WLASL nevăzuți.
RECORDED_DIR = os.path.join(DATA_DIR, 'recorded_keypoints')
USE_RECORDED_DATA = True
# Test live (src/app/live_test.py): semne întregi decupate din camera live, cu cuvântul
# cerut. NU se folosesc niciodată la antrenare; măsoară acuratețea reală la utilizare.
LIVE_TEST_DIR = os.path.join(DATA_DIR, 'live_test')
MODEL_PATH = os.path.join('models', 'sign_language_model.h5')

# Folderul local cu toate mp4-urile WLASL descărcate (folosit doar de organize_videos.py)
WLASL_VIDEOS_SOURCE = r'C:\Desktop\dataset licenta\videos'

# Camera pentru predicția live: 0 = camera laptopului, 1 = telefonul conectat ca webcam
CAMERA_INDEX = 1

# Secvențe
SEQUENCE_LENGTH = 30   # cadre per exemplu
# Dimensiunea vectorului brut (1662) și pozițiile din el: src/features/keypoints.py (RAW_DIM)

# Reproductibilitate
SEED = 42


# ---------------------------------------------------------------------------
# Vorbire -> text (ce spune medicul), codul în src/speech/.
# Merge pe laptop și pe Raspberry Pi 4 schimbând doar valorile de aici.
# Căile noi folosesc pathlib (relative la rădăcina proiectului), fără căi Windows.
# ---------------------------------------------------------------------------

# Motorul de transcriere
# 'auto' = Groq când există internet, altfel modelul local; 'offline_only' = doar local
SPEECH_MODE = 'auto'
USE_GROQ = True                 # False = Groq dezactivat complet (ex. fără cheie)
SPEECH_LANGUAGE = 'ro'
SPEECH_TEMPERATURE = 0.0
# Prompt-ul cu vocabularul medical se construiește din src/app/sign_info.py ('meaning')

# Online: Groq API (cheia GROQ_API_KEY din fișierul .env, vezi .env.example)
ENV_FILE = Path('.env')
GROQ_MODEL = 'whisper-large-v3'
GROQ_RESPONSE_FORMAT = 'verbose_json'
GROQ_TIMEOUT_SECONDS = 4.0      # peste atât (sau la orice eroare) trecem pe modelul local

# Offline: faster-whisper cu readerbench/whisper-ro convertit în CTranslate2 int8
WHISPER_HF_MODEL = 'readerbench/whisper-ro'
WHISPER_LOCAL_DIR = Path('models') / 'whisper-ro-ct2'
WHISPER_DEVICE = 'cpu'
WHISPER_COMPUTE_TYPE = 'int8'
WHISPER_CPU_THREADS = 4         # Raspberry Pi 4 are 4 nuclee
WHISPER_BEAM_SIZE = 2
WHISPER_VAD_FILTER = False      # liniștea e tăiată deja de VAD-ul nostru (webrtcvad)

# Verificarea internetului, în fundal (nu la fiecare frază)
INTERNET_CHECK_SECONDS = 30
INTERNET_CHECK_HOST = 'api.groq.com'
INTERNET_CHECK_PORT = 443
INTERNET_CHECK_TIMEOUT = 2.0

# Captura audio (formatul cerut de Whisper și de webrtcvad)
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_DTYPE = 'int16'
# Microfonul, după o parte din nume (lista: python -m sounddevice); None = microfonul implicit
MICROPHONE_NAME = None

# Declanșarea unei fraze: 'button' = apeși tasta, apoi vorbești; 'vad' = pornește singur la voce
SPEECH_TRIGGER = 'button'
SPEECH_BUTTON_KEY = ' '                 # Space (în fereastra programului)
SPEECH_BUTTON_WAIT_SECONDS = 5.0        # după apăsare: dacă nu vorbește nimeni atât, se renunță
# Bip după apăsare? Implicit NU: în testul din 2026-11-16 vorbitorul începea imediat, iar
# începutul frazei cădea în timpul bipului (microfon oprit). Semnalul e vizual („Te ascult...”).
SPEECH_BEEP = False
SPEECH_BEEP_HZ = 880
SPEECH_BEEP_SECONDS = 0.12
PLAYBACK_TAIL_MS = 150                  # half-duplex: după sunet (+ întârzierea plăcii audio), încă atât (ecou)

# Detecția vocii (webrtcvad)
VAD_AGGRESSIVENESS = 2          # 0 (sensibil) ... 3 (strict)
VAD_FRAME_MS = 30               # webrtcvad acceptă doar 10, 20 sau 30 ms
VAD_START_MS = 210              # atâta voce continuă înseamnă că fraza a început
VAD_PRE_ROLL_MS = 300           # audio păstrat dinaintea vocii, ca să nu tăiem începutul cuvântului
VAD_SILENCE_END_MS = 700        # fraza se termină după atâta liniște
MIN_UTTERANCE_SECONDS = 0.5     # mai scurt de atât = zgomot, se ignoră
MAX_UTTERANCE_SECONDS = 15.0
# Fraza se ignoră dacă și cele mai tari părți ale ei (percentila 90 a cadrelor) sunt sub acest
# nivel: e zgomot sau liniște, pe care Whisper ar inventa text. Vocea la microfonul laptopului
# ajunge la -37...-39 dB; pentru alt microfon se verifică nivelul în record_test.
MIN_SPEECH_LEVEL_DB = -50.0

# Protecție împotriva halucinațiilor Whisper: textul care se comprimă mai mult de atât e o buclă
# de repetiții („a fost... a fost...”) și se tratează ca eroare (același prag ca în Whisper).
# (O limită de tokeni după durata audio a fost încercată: încetinea transcrierea locală cu ~50%.)
HALLUCINATION_COMPRESSION_RATIO = 2.4

# Fișiere
SPEECH_RECORDINGS_DIR = Path(DATA_DIR) / 'speech_recordings'   # frazele înregistrate (WAV)
SPEECH_LOG_FILE = Path(DATA_DIR) / 'speech_log.csv'            # jurnalul transcrierilor
SPEECH_TEST_DIR = Path(DATA_DIR) / 'test_doctor'               # WAV-uri + textul corect, pentru evaluare
SPEECH_TEST_PHRASES_FILE = SPEECH_TEST_DIR / 'fraze.txt'      # frazele de citit (src/speech/record_test_set.py)
SPEECH_EVAL_FILE = Path(DATA_DIR) / 'speech_eval.csv'          # rezultatele evaluării (src/speech/evaluate.py)

# Evaluare: dacă Groq răspunde 429 (limita planului gratuit), așteptăm și reîncercăm
GROQ_RATE_LIMIT_WAIT_SECONDS = 10
GROQ_RATE_LIMIT_RETRIES = 3

# ---------------------------------------------------------------------------
# Avatar 3D care semnează (text al medicului -> semne ASL), în src/avatar/
# ---------------------------------------------------------------------------
AVATAR_MOTION_DIR = Path(DATA_DIR) / 'avatar_motion'           # mișcarea fiecărui semn (JSON, src/avatar/export_motion.py)
AVATAR_MODEL_FILE = Path('models') / 'avatar' / 'avatar.vrm'   # personajul 3D (VRM); nu e în git
AVATAR_EVAL_DIR = Path(DATA_DIR) / 'avatar_eval'               # rezultatele back-translation (src/avatar/evaluate_avatar.py)
AVATAR_HOST = '127.0.0.1'   # 0.0.0.0 = pagina se poate deschide și de pe o tabletă din aceeași rețea
AVATAR_PORT = 8765
# Raportul înălțime/lățime al camerei la înregistrările proprii (telefon, 16:9). Coordonatele
# MediaPipe sunt fracții din lățime (x) și din înălțime (y); fără raport, mișcarea s-ar turti.
# Estimat din date: palma (rigidă) are lungimi cel mai constante în timp la 0.5625.
CAMERA_ASPECT = 9 / 16
