import streamlit as st
import sqlite3
import face_recognition
import numpy as np
import cv2
import pickle
from deepface import DeepFace
import time

# DATABASE
conn = sqlite3.connect('Infos_Client.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS Clients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        nom TEXT NOT NULL,
        prenom TEXT NOT NULL,
        ville TEXT NOT NULL,       
        email TEXT NOT NULL,
        password TEXT NOT NULL,
        face_encoding BLOB
        
    )
''')

conn.commit()
conn.close()

# INSCRIPTION 

import streamlit as st
import sqlite3
import face_recognition
import cv2
import numpy as np

def Sauve_Infos(username, nom, prenom, ville, email, password, face_encoding):
    conn = sqlite3.connect('Infos_Client.db')
    cursor = conn.cursor()
    
    face_encoding_bytes = face_encoding.tobytes()  
    cursor.execute('''
        INSERT INTO Clients (username, nom, prenom, ville, email, password, face_encoding)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (username, nom, prenom, ville, email, password, face_encoding_bytes))
    
    conn.commit()
    conn.close()

def Inscription():
    st.markdown("""<p style = 'color: green'>Page d Inscription</p>""", unsafe_allow_html=True)

    username = st.text_input("Nom d'utilisateur")
    nom = st.text_input("Nom")
    prenom = st.text_input("Prénom")
    ville = st.text_input("Ville")
    email = st.text_input("Email")
    password = st.text_input("Mot de passe", type="password")
    uploaded_image = st.file_uploader("Téléchargez une photo pour la reconnaissance faciale", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = face_recognition.load_image_file(uploaded_image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(image_rgb)

        if face_encodings: 
            face_encoding = face_encodings[0] 
            st.image(uploaded_image, caption="Image téléchargée", use_column_width=True)

            if st.button("S'inscrire"):
                Sauve_Infos(username, nom, prenom, ville, email, password, face_encoding)
                st.success('Utilisateur inscrit avec succès')
            else:
                st.error('Aucun visage détecté dans l image. Veuillez télécharger une autre photo')

if __name__ == "__main__":
    Inscription()

# CONNEXION 

def Verification_1(username, password):
    conn = sqlite3.connect('Infos_Client.db')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM Clients WHERE username=? AND password=?', (username, password))
    count = cursor.fetchone()[0]
    conn.close()
    
    if count > 0:
        return True
    else:
        return False
    

def Verification_2(email):
    conn = sqlite3.connect('Infos_Client.db')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM Clients WHERE email=?', (email))
    count = cursor.fetchone()[0]
    conn.close()
    
    if count > 0:
        return True
    else:
        return False

def Verification_faciale(face_encoding):
    conn = sqlite3.connect('Infos_Client.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT face_encoding FROM Clients')
    face_encodings = cursor.fetchall()

    for encoding in face_encodings:
        stored_face_encoding = np.array(pickle.loads(encoding[0]), dtype='float')
        #stored_face_encoding = np.frombuffer(encoding[0], dtype=np.float64)  
        distance = np.linalg.norm(stored_face_encoding - face_encoding)  

        if distance < 0.5:  
            conn.close()
            return True  

    conn.close()
    return False  

def Resultat(Check_user):
    if Check_user: 
        st.success("Bienvenue, Connexion reussie 😊😊😊")
    else:
        st.error("Malheureusement nous n'avons pas pu vous retrouver 😖😖😖, verifier encore vos infos 🥲")

def Connexion():
    st.title("Page de connexion")

    st.markdown("""<p style = 'style = 'color: green' >Bienvenue, Choisissez un mode d authentification 😎😉</p>""", unsafe_allow_html=True)
    method = st.selectbox("__________Méthode de securité___________", ["Mot de Passe", "Reconnaissance faciale", "Compte Google", "Compte Facebook"])
    
    if method == "Mot de Passe":
        st.markdown("""<p  style = 'style = 'color: green'>Connexion avec Mot de Passe</p>""", unsafe_allow_html=True)
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        if st.button("Se connecter"):
            Check_user = Verification_1(username, password)
            Resultat(Check_user)
            
    
    elif method == "Reconnaissance faciale":
        st.markdown("""<p  style = 'style = 'color: green'>Connexion avec Reconnaissance faciale</p>""", unsafe_allow_html=True)
        capture = cv2.VideoCapture(0)
        while True:
            ret, image = capture.read()
            if ret:
                image_resized = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
                face_locations = face_recognition.face_locations(image_resized)
                face_encodings = face_recognition.face_encodings(image_resized, face_locations)

                for face_encoding, loc in zip(face_encodings, face_locations):
                    y1, x2, y2, x1 = loc
                    y1, x2, y2, x1 = 4*y1, 4*x2, 4*y2, 4*x1
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, "recognizing...", (x1, y1-25), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
                    try:
                        face_image=image[y1:y2,x1:x2]
                        analyse=DeepFace.analyze(face_image,actions=['emotion'],enforce_detection=True)
                        if isinstance(analyse,list):
                            analyse=analyse[0]
                        emotion=analyse.get('dominant_emotion','None')
                        cv2.putText(image,emotion,(x1,y2+25),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
                    except:
                        print('erreur') 

                    Check_user = Verification_faciale(face_encoding)

                    if Check_user:
                        st.success("Bienvenue, Connexion reussie 😊😊😊")
                        #st.write(f"during the process you were: {emotion}")
                        capture.release()
                        cv2.destroyAllWindows()
                        return 
                    else: 
                        st.error("Malheureusement nous n'avons pas pu vous retrouver 😖😖😖")
                        capture.release()
                        cv2.destroyAllWindows()
                        return 


                cv2.imshow('Reconnaissance faciale', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        capture.release()
        cv2.destroyAllWindows()

    elif method == "Compte Google":
        st.markdown("""<p  style = 'style = 'color: green'>Connexion avec Google</p>""", unsafe_allow_html=True)
        email = st.text_input("Adresse Email: ", type="password")
        if st.button("Se connecter"):
            Check_user = Verification_2(email)
            Resultat(Check_user)

    elif method == "Compte Facebook":
        st.markdown("""<p  style = 'style = 'color: green'>Connexion avec Facebook</p>""", unsafe_allow_html=True)
        email = st.text_input("Adresse Email: ", type="password")
        if st.button("Se connecter"):
            Check_user = Verification_2(email)
            Resultat(Check_user)

if __name__ == "__main__":
    Connexion()
