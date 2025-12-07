# camera/camera_enroll.py
import cv2
import os
import sys
import pickle
import json
import numpy as np
from datetime import datetime
import pytz  # zona horaria

# Añadir raíz del proyecto al path para imports relativos
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from app.models.recognition import ArcFaceRecognizer
from app.models.detection import detect_and_crop_face
from pattern.pattern_auth import PatternAuth  # tu pattern/pattern_auth.py

# Rutas
EMBEDDINGS_PATH = os.path.join(ROOT, "models", "embeddings_arcface.pkl")
USERS_INFO_PATH = os.path.join(ROOT, "models", "users_info.json")
LOGS_DIR = os.path.join(ROOT, "logs")
LOG_FILE = os.path.join(LOGS_DIR, "access_logs.csv")
os.makedirs(LOGS_DIR, exist_ok=True)


def append_log(ts, user_id_or_name, result, detail=""):
    """Escribe una entrada en el log. user_id_or_name puede ser ID o nombre ya resuelto."""
    header = "ts,user_id,result,detail\n"
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(header)
    # Encerrar detail entre comillas para evitar problemas con comas
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f'{ts},{user_id_or_name},{result},"{detail}"\n')




def load_embeddings():
    """Carga embeddings previos si existen."""
    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_embeddings(data):
    """Guarda embeddings en disco."""
    ddir = os.path.dirname(EMBEDDINGS_PATH)
    if ddir and not os.path.exists(ddir):
        os.makedirs(ddir, exist_ok=True)
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(data, f)


def save_user_info(user_id, user_name):
    """Guarda el nombre del usuario en users_info.json"""
    users = {}
    if os.path.exists(USERS_INFO_PATH):
        try:
            with open(USERS_INFO_PATH, "r", encoding="utf-8") as f:
                users = json.load(f)
        except Exception:
            users = {}

    users[str(user_id)] = user_name  # user_id como string
    with open(USERS_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4, ensure_ascii=False)


def enroll_user(user_id, user_name=None):
    """Captura embeddings y patrón para un nuevo usuario."""
    embeddings = load_embeddings()

    recognizer = ArcFaceRecognizer()
    pattern_auth = PatternAuth(data_path=os.path.join(ROOT, "pattern", "pattern_data.pkl"))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[❌] No se pudo acceder a la cámara. Asegúrate de que no esté siendo usada por otra aplicación.")
        return

    print("[📷] Cámara lista. Presiona 'c' para capturar, 'q' para terminar.")
    user_embeddings = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[❌] Error leyendo frame de la cámara.")
            break

        display = frame.copy()
        cv2.putText(display, f"ID:{user_id} Capturas:{len(user_embeddings)} (c=cap,q=fin)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Enroll - Presiona 'c' para capturar", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            face = detect_and_crop_face(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if face is not None:
                try:
                    emb = recognizer.get_embedding(face)
                    user_embeddings.append(emb)
                    print(f"[📸] Embedding capturado ({len(user_embeddings)})")
                except Exception as e:
                    print("[❌] Error extrayendo embedding:", e)
            else:
                print("[⚠️] No se detectó cara en este frame.")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not user_embeddings:
        print("[❌] No se capturaron embeddings válidos.")
        return

    # Guardar embeddings
    if user_id in embeddings:
        embeddings[user_id].extend(user_embeddings)
    else:
        embeddings[user_id] = user_embeddings
    save_embeddings(embeddings)

    # Guardar nombre en JSON
    if user_name:
        save_user_info(user_id, user_name)
        print(f"[💾] Guardado nombre: {user_name} (ID {user_id})")

    print(f"[✅] Usuario {user_id} registrado con {len(user_embeddings)} embeddings.")

    # Registro del patrón facial/ocular
    print("[🔁] Ahora registrarás el patrón (movimiento facial/ocular).")
    success = pattern_auth.enroll_pattern(user_id, attempts=2, duration=5)

    if success:
        print("[✅] Patrón guardado correctamente.")
    else:
        print("[⚠️] No se registró el patrón correctamente.")

    # Hora local
    tz = pytz.timezone("America/Lima")
    lima_time = datetime.now(tz).isoformat(timespec="seconds")

    append_log(lima_time, f"{user_id}", "ENROLL_OK", f"n_embs={len(user_embeddings)}")
    print("[📁] Enrolamiento completo y log guardado con hora local de Lima.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 1:
        uid = args[0]
        name = args[1] if len(args) >= 2 else None
        enroll_user(uid, name)
    else:
        uid = input("Ingrese ID de usuario: ").strip()
        name = input("Ingrese nombre (opcional): ").strip()
        enroll_user(uid, name if name else None)
