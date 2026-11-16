"""
Configurația centrală a proiectului: clase, căi și constante comune.
Toate scripturile importă de aici, ca să nu existe copii care se pot desincroniza.
Căile sunt relative la rădăcina proiectului (scripturile se rulează din rădăcină).
"""
import os

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
