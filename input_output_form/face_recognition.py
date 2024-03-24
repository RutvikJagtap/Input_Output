import cv2
import numpy as np
import pymongo
from deepface import DeepFace
from base64 import b64decode
import base64
import pandas as pd
import pickle

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["input_output"]
collection = db["Person"]

known_faces = {}
for doc in collection.find():
    name = doc["firstname"]
    image_base64 = doc["image"]
    nparr = np.frombuffer(image_base64, dtype=np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    embedding = DeepFace.represent(img_np, model_name='Facenet')
    dic = embedding[0]
    emb = dic['embedding']
    emp = np.array(emb)
    known_faces[name] = emb


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
     # Changed to BGR for correct conversion

    # Detect faces
    faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml').detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face = frame[y:y+h, x:x+w]
        resized_face = cv2.resize(face, (160, 160))

        # Normalize pixel values to [0, 1]
        normalized_face = resized_face / 255.0

        # Expand dimensions to match the input shape of the model
        input_face = np.expand_dims(normalized_face, axis=0)

        # Calculate the face embedding using the Facenet model
        embedding = DeepFace.represent(face, model_name='Facenet', enforce_detection=False)
        dic = embedding[0]
        embedding = dic['embedding']
        embedding = np.array(embedding)
        if face.size == 0:
            continue
        # Compare the embedding with known face embeddings
        for name, known_embedding in known_faces.items():
            similarity_score = np.dot(embedding, known_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(known_embedding))
            if similarity_score > 0.7:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()