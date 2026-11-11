"""
Copiază videoclipurile WLASL pentru cuvintele din ACTIONS în data/raw_videos/<cuvânt>/.

O clasă poate aduna mai multe glose WLASL (WLASL_GLOSSES). La clasele care unesc mai
multe glose, numele fișierului începe cu gloss-ul (ex.: i_12345.mp4, pain_67890.mp4),
ca să se vadă de unde vine fiecare videoclip.

Clasele care au deja folder în data/raw_videos/ NU se ating: acolo sunt date curățate
manual (videoclipuri proaste șterse, vezi data/cleaning_log.json), iar o recopiere le-ar
aduce înapoi. Pentru a recopia o clasă, șterge-i întâi folderul.

Rulare (din rădăcina proiectului):
    python -m src.data.organize_videos
"""
import json
import os
import shutil

from src.config import ACTIONS, RAW_VIDEOS_DIR, WLASL_JSON, WLASL_VIDEOS_SOURCE

# Glosele WLASL din care se formează fiecare clasă (implicit: gloss-ul cu numele clasei)
WLASL_GLOSSES = {
    'thankyou': ['thank you'],
    'hurt': ['hurt', 'pain'],
    'sorethroat': ['sore throat'],
}


def glosses_for(action):
    return WLASL_GLOSSES.get(action, [action])


def organize_dataset():
    if not os.path.exists(WLASL_JSON):
        print(f"EROARE: Nu găsesc fișierul {WLASL_JSON}. Pune-l în folderul data/!")
        return

    if not os.path.exists(WLASL_VIDEOS_SOURCE):
        print(f"EROARE: Folderul sursă {WLASL_VIDEOS_SOURCE} nu există. Modifică calea în src/config.py!")
        return

    with open(WLASL_JSON, 'r', encoding='utf-8') as f:
        wlasl_data = json.load(f)

    # gloss -> lista de video_id (un gloss poate apărea în mai multe intrări)
    instances = {}
    for entry in wlasl_data:
        instances.setdefault(entry['gloss'], []).extend(i['video_id'] for i in entry['instances'])

    os.makedirs(RAW_VIDEOS_DIR, exist_ok=True)
    total_copiate = 0

    for action in ACTIONS:
        word_folder = os.path.join(RAW_VIDEOS_DIR, action)
        if os.path.isdir(word_folder) and os.listdir(word_folder):
            print(f"SĂRIT: {action} are deja videoclipuri în {word_folder} (nu se atinge).")
            continue

        glosses = glosses_for(action)
        prefix = len(glosses) > 1
        os.makedirs(word_folder, exist_ok=True)

        for gloss in glosses:
            if gloss not in instances:
                print(f"EROARE: gloss-ul '{gloss}' (clasa {action}) nu există în {WLASL_JSON}.")
                continue
            copiate, lipsa = 0, []
            for video_id in instances[gloss]:
                src_path = os.path.join(WLASL_VIDEOS_SOURCE, f"{video_id}.mp4")
                # WLASL nu are mereu TOATE videoclipurile descărcate
                if not os.path.exists(src_path):
                    lipsa.append(video_id)
                    continue
                dst_name = f"{gloss.replace(' ', '')}_{video_id}.mp4" if prefix else f"{video_id}.mp4"
                shutil.copy(src_path, os.path.join(word_folder, dst_name))
                copiate += 1
            total_copiate += copiate
            print(f"OK: {action} <- '{gloss}': {copiate} copiate din {len(instances[gloss])} listate"
                  + (f"; lipsă din arhivă: {', '.join(lipsa)}" if lipsa else ""))

    print(f"\nGATA! Am copiat {total_copiate} videoclipuri.")


if __name__ == "__main__":
    organize_dataset()
