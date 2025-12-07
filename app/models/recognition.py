# /app/models/recognition.py
import onnxruntime as ort
import numpy as np
import cv2
import os

MODEL_PATH = "app/models/arcface.onnx"

class ArcFaceRecognizer:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"[❌] No se encontró el modelo en {MODEL_PATH}")
        
        print("[📦] Cargando modelo ArcFace ONNX...")
        self.session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, face_img):
        # Redimensionar a 112x112 y normalizar
        img = cv2.resize(face_img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0   # normalización recomendada ArcFace
        img = np.expand_dims(img, axis=0)  # (1,112,112,3) -> NHWC
        return img

    def get_embedding(self, face_img):
        img = self.preprocess(face_img)
        emb = self.session.run(None, {self.input_name: img})[0]
        emb = emb / np.linalg.norm(emb)  # normalizar vector L2
        return emb.flatten()

    def compare(self, emb1, emb2, threshold=0.9):
        # Similitud coseno
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return sim, sim > threshold
