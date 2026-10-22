import os
import numpy as np

# Calea către datele tale extrase
DATA_PATH = os.path.join('data', 'processed_keypoints')
ACTIONS = np.array([
    'hello', 'thankyou', 'yes', 'no', 'help', 
    'please', 'sorry', 'teacher', 'student', 'book', 
    'computer', 'learn', 'need', 'good', 'stop'
])

def add_noise(data, noise_level=0.01):
    """Adaugă zgomot aleator pentru a simula imperfecțiunile camerei."""
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def scale_data(data, scale_factor=0.9):
    """Mărește sau micșorează scheletul (persoana e mai aproape/departe)."""
    return data * scale_factor

def augment_dataset():
    print("Începem augmentarea (înmulțirea) datelor...")
    
    for action in ACTIONS:
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_path):
            continue
            
        files = [f for f in os.listdir(action_path) if f.endswith('.npy') and not f.startswith('aug_')]
        
        print(f"Procesăm '{action}': am găsit {len(files)} fișiere originale.")
        
        for file in files:
            original_path = os.path.join(action_path, file)
            data = np.load(original_path)
            
            # 1. Varianta cu Zgomot
            noisy_data = add_noise(data, noise_level=0.015)
            np.save(os.path.join(action_path, f"aug_noise_{file}"), noisy_data)
            
            # 2. Varianta Scalată (mai mic)
            scaled_data = scale_data(data, scale_factor=0.85)
            np.save(os.path.join(action_path, f"aug_scale_{file}"), scaled_data)
            
            # 3. Varianta combinată (Scalat + Zgomot)
            combined_data = add_noise(scale_data(data, scale_factor=1.15), noise_level=0.01)
            np.save(os.path.join(action_path, f"aug_combo_{file}"), combined_data)

    print("Augmentarea a fost finalizată! Setul tău de date este acum de 4 ori mai mare.")

if __name__ == '__main__':
    augment_dataset()