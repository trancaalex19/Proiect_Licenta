import cv2
import mediapipe as mp

# Inițializăm modulele MediaPipe pentru detecția corpului și desenare
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def draw_landmarks(image, results):
    # Desenăm conexiunile feței
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    
    # Desenăm postura (umeri, brațe)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
        
    # Desenăm mâna stângă
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))

    # Desenăm mâna dreaptă
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

# SETARE CAMERĂ: Dacă folosești DroidCam, indexul este de obicei 1 sau 2. 
# Dacă laptopul nu are altă cameră web, ar putea fi 0.
video_source = 2
cap = cv2.VideoCapture(video_source)

print("Pornim camera... Apasă tasta 'q' în fereastra video pentru a o închide.")

# Pornim modelul AI
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Nu se poate citi imaginea de la cameră. Verifică indexul (video_source) sau DroidCam.")
            break

        # Convertim imaginea din BGR (formatul OpenCV) în RGB (formatul cerut de MediaPipe)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Facem detecția
        results = holistic.process(image)
        
        # Convertim înapoi pentru a afișa culorile corect
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Desenăm punctele și liniile
        draw_landmarks(image, results)

        # Afișăm fereastra
        cv2.imshow('Testare Camera - MediaPipe', image)

        # Ieșire la apăsarea tastei 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()``