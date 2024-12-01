import cv2
import numpy as np
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine
from mtcnn import MTCNN
from scipy.fftpack import dct
import os

# Initialisation des modèles
detector = MTCNN()
embedder = FaceNet()

def preprocess_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    img = cv2.resize(img, (160, 160))  # FaceNet attend des images 160x160
    return img

def extract_features(img):
    img = preprocess_image(img)
    img = np.expand_dims(img, axis=0)
    embeddings = embedder.embeddings(img)
    return embeddings[0]

def two_step_verification(img1, img2):
    features1 = extract_features(img1)
    features2 = extract_features(img2)
    similarity_cnn = 1 - cosine(features1, features2)
    
    dct1 = dct(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).flatten())
    dct2 = dct(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).flatten())
    similarity_dct = 1 - cosine(dct1[:10], dct2[:10])
    
    return similarity_cnn > 0.7 and similarity_dct > 0.9  # Ajustez ces seuils selon vos besoins

def capture_from_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imwrite('temp_capture.jpg', frame)
            break
    cap.release()
    cv2.destroyAllWindows()
    return cv2.imread('temp_capture.jpg')

def register_new_face(source='webcam'):
    if source == 'webcam':
        img = capture_from_webcam()
    elif source == 'file':
        file_path = input("Entrez le chemin du fichier image: ")
        img = cv2.imread(file_path)
    else:
        print("Source non reconnue")
        return

    faces = detector.detect_faces(img)
    if len(faces) == 0:
        print("Aucun visage détecté")
        return
    
    x, y, w, h = faces[0]['box']
    face = img[y:y+h, x:x+w]
    face = preprocess_image(face)
    
    cv2.imwrite('authorized_face.jpg', face)
    print("Nouveau visage autorisé enregistré")

def authenticate(img_path, authorized_img_path='authorized_face.jpg'):
    img = cv2.imread(img_path)
    authorized_img = cv2.imread(authorized_img_path)
    
    faces = detector.detect_faces(img)
    authorized_faces = detector.detect_faces(authorized_img)
    
    if len(faces) == 0 or len(authorized_faces) == 0:
        return False
    
    x, y, w, h = faces[0]['box']
    face = img[y:y+h, x:x+w]
    
    ax, ay, aw, ah = authorized_faces[0]['box']
    authorized_face = authorized_img[ay:ay+ah, ax:ax+aw]
    
    return two_step_verification(face, authorized_face)

def main_menu():
    while True:
        print("\n--- Menu Principal ---")
        print("1. Enregistrer un nouveau visage")
        print("2. Authentifier")
        print("3. Quitter")
        choice = input("Choisissez une option: ")

        if choice == '1':
            source = input("Choisissez la source (webcam/file): ")
            register_new_face(source)
        elif choice == '2':
            test_img_path = input("Entrez le chemin de l'image à tester: ")
            if authenticate(test_img_path):
                print("Authentification réussie")
            else:
                print("Authentification échouée")
        elif choice == '3':
            break
        else:
            print("Option non valide")

if __name__ == "__main__":
    main_menu()
