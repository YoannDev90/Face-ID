import cv2
import numpy as np
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine
from mtcnn import MTCNN
import json
import time
import hashlib

# Initialisation des modèles
detector = MTCNN()
embedder = FaceNet()

def preprocess_image(img):
    img = cv2.resize(img, (160, 160))
    return img

def extract_features(img):
    img = preprocess_image(img)
    img = np.expand_dims(img, axis=0)
    embeddings = embedder.embeddings(img)
    return embeddings[0]

def generate_key(features):
    feature_bytes = features.tobytes()
    return hashlib.sha256(feature_bytes).hexdigest()[:16]

def capture_from_webcam():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame

def register_new_face():
    img = capture_from_webcam()
    faces = detector.detect_faces(img)
    if len(faces) == 0:
        print("Aucun visage détecté")
        return
    
    x, y, w, h = faces[0]['box']
    face = img[y:y+h, x:x+w]
    features = extract_features(face)
    key = generate_key(features)
    
    name = input("Entrez le nom de la personne : ")
    
    user_data = {
        "features": features.tolist(),
        "key": key,
        "name": name
    }
    
    with open('authorized_face.json', 'w') as f:
        json.dump(user_data, f)
    
    print(f"Nouveau visage autorisé enregistré pour {name}")

def continuous_authentication():
    print("Démarrage de l'authentification continue. Appuyez sur 'q' pour quitter.")
    cap = cv2.VideoCapture(0)
    
    with open('authorized_face.json', 'r') as f:
        authorized_data = json.load(f)
    authorized_features = np.array(authorized_data['features'])
    authorized_name = authorized_data['name']
    
    while True:
        ret, frame = cap.read()
        faces = detector.detect_faces(frame)
        
        info_frame = np.zeros((150, 400, 3), dtype=np.uint8)
        
        if len(faces) > 0:
            face = faces[0]
            x, y, w, h = face['box']
            face_img = frame[y:y+h, x:x+w]
            features = extract_features(face_img)
            similarity = 1 - cosine(features, authorized_features)
            
            status = "Autorisé" if similarity > 0.7 else "Non autorisé"
            color = (0, 255, 0) if status == "Autorisé" else (0, 0, 255)
            
            # Dessiner le rectangle autour du visage
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Afficher les informations
            cv2.putText(info_frame, f"Nom: {authorized_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(info_frame, f"Confiance: {face['confidence']:.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(info_frame, f"Similarité: {similarity:.4f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(info_frame, f"Statut: {status}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(info_frame, "Aucun visage détecté", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Face Recognition', frame)
        cv2.imshow('Info', info_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main_menu():
    while True:
        print("\n--- Menu Principal ---")
        print("1. Enregistrer un nouveau visage")
        print("2. Authentification continue")
        print("3. Quitter")
        choice = input("Choisissez une option: ")

        if choice == '1':
            register_new_face()
        elif choice == '2':
            continuous_authentication()
        elif choice == '3':
            break
        else:
            print("Option non valide")

if __name__ == "__main__":
    main_menu()
