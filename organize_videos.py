import json
import os
import shutil

# 1. Calea către fișierul JSON descărcat (asigură-te că e în folderul proiectului)
JSON_FILE = 'WLASL_v0.3.json'

# 2. Calea către folderul UNDE AI DESCARCAT TU toate mp4-urile amestecate 
# (Schimbă cu calea ta reală, ex: 'C:/Users/tranc/Downloads/WLASL/videos')
VIDEOS_SOURCE_FOLDER = r'C:\Users\tranc\OneDrive\Desktop\dataset licenta\videos'

# 3. Calea unde vrem să le mutăm organizat pentru proiect
OUTPUT_FOLDER = os.path.join('data', 'raw_videos')

# Lista noastră de 15 cuvinte (scrise exact cum apar în WLASL)
TARGET_WORDS = [
    'hello', 'thank you', 'yes', 'no', 'help', 
    'please', 'sorry', 'teacher', 'student', 'book', 
    'computer', 'learn', 'need', 'good', 'stop'
]

def organize_dataset():
    if not os.path.exists(JSON_FILE):
        print(f"EROARE: Nu găsesc fișierul {JSON_FILE}. Pune-l în folderul proiectului!")
        return

    if not os.path.exists(VIDEOS_SOURCE_FOLDER):
        print(f"EROARE: Folderul sursă {VIDEOS_SOURCE_FOLDER} nu există. Modifică calea în cod!")
        return

    # Citim dicționarul
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        wlasl_data = json.load(f)

    # Creăm structura de foldere
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    total_copiate = 0

    for entry in wlasl_data:
        gloss = entry['gloss']
        
        if gloss in TARGET_WORDS:
            # Creăm folderul pentru cuvânt (eliminăm spațiile, ex: 'thank you' -> 'thankyou')
            folder_name = gloss.replace(' ', '')
            word_folder = os.path.join(OUTPUT_FOLDER, folder_name)
            os.makedirs(word_folder, exist_ok=True)
            
            # Căutăm ID-urile video pentru acest cuvânt
            for instance in entry['instances']:
                video_id = instance['video_id']
                video_filename = f"{video_id}.mp4"
                
                src_path = os.path.join(VIDEOS_SOURCE_FOLDER, video_filename)
                dst_path = os.path.join(word_folder, video_filename)
                
                # WLASL nu are mereu TOATE videoclipurile descărcate. Verificăm dacă există.
                if os.path.exists(src_path):
                    shutil.copy(src_path, dst_path)
                    print(f"OK: Salvat {video_filename} în folderul -> {folder_name}")
                    total_copiate += 1
                else:
                    print(f"LIPSĂ: Videoclipul {video_filename} pentru '{gloss}' nu este în arhiva ta.")

    print(f"\nGATA! Am organizat cu succes {total_copiate} videoclipuri pentru cele 15 cuvinte.")

if __name__ == "__main__":
    organize_dataset()