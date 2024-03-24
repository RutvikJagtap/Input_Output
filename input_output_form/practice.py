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
    print(len(nparr))
    img_np = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    print(len(img_np))