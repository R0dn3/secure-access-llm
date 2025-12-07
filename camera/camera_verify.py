# camera/camera_verify.py
import cv2
import os
import sys
import pickle
import json
import numpy as np
from datetime import datetime
import pytz

# Añadir raíz del proyecto al path para imports relativos
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from app.models.recognition import ArcFaceRecognizer
from app.models.detection import detect_faces
try:
    from app.models.liveness import estimate_spoof_score
    HAS_LIVENESS = True
except Exception:
    HAS_LIVENESS = False

from pattern.pattern_auth import PatternAuth

# Rutas
EMBEDDINGS_PATH = os.path.join(ROOT, "models", "embeddings_arcface.pkl")
USERS_INFO_PATH = os.path.join(ROOT, "models", "users_info.json")
LOGS_DIR = os.path.join(ROOT, "logs")
LOG_FILE = os.path.join(LOGS_DIR, "access_logs.csv")
os.makedirs(LOGS_DIR, exist_ok=True)

# Umbrales
SIMILARITY_THRESHOLD = 0.7
SPOOF_THRESHOLD = 0.5
TIME_THRESHOLD = 5  # ⏱️ segundos obligatorios por estado


def append_log(ts, user_id_or_name, result, detail=""):
    header = "ts,user_id,result,detail\n"
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(header)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f'{ts},{user_id_or_name},{result},"{detail}"\n')


def load_embeddings_mean():
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError("No hay embeddings guardados.")
    with open(EMBEDDINGS_PATH, "rb") as f:
        data = pickle.load(f)
    users = {}
    for u, el in data.items():
        arr = np.vstack(el)
        m = np.mean(arr, axis=0)
        users[u] = m / (np.linalg.norm(m) + 1e-8)
    return users


def load_users_info(path=USERS_INFO_PATH):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {str(k): v for k, v in data.items()}
        except Exception:
            return {}
    return {}


def get_display_name(user_id, users_info):
    if user_id is None:
        return "DESCONOCIDO"
    return users_info.get(str(user_id), str(user_id))


def verify_loop():
    users = load_embeddings_mean()
    recognizer = ArcFaceRecognizer()
    pattern_auth = PatternAuth(data_path=os.path.join(ROOT, "pattern", "pattern_data.pkl"))
    users_info = load_users_info()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[❌] No se pudo abrir la cámara.")
        return

    prev_gray = None
    last_best_user = None
    last_status = None
    status_start_time = None
    decision_taken = False
    spoof_logged = False

    tz = pytz.timezone("America/Lima")

    print("[✅] Cámara lista. Sistema 100% automático.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)
        annotated = frame.copy()

        for (x, y, w, h) in faces:
            x2, y2 = x + w, y + h
            roi = frame[y:y2, x:x2]
            if roi.size == 0:
                continue

            try:
                emb = recognizer.get_embedding(roi)
            except Exception:
                continue

            best_user, best_score = None, -1.0
            for u, uemb in users.items():
                sim = float(np.dot(emb, uemb) / (np.linalg.norm(emb) * np.linalg.norm(uemb) + 1e-8))
                if sim > best_score:
                    best_score, best_user = sim, u

            if HAS_LIVENESS:
                try:
                    spoof_score = estimate_spoof_score(roi)
                except Exception:
                    spoof_score = 0.0
            else:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                motion = 0.0 if prev_gray is None else float(np.mean(cv2.absdiff(prev_gray, gray))) / 255.0
                prev_gray = gray
                spoof_score = max(0.0, min(1.0, 1.0 - motion))

            # DECISIÓN POR ESTADO
            if best_score >= SIMILARITY_THRESHOLD and spoof_score <= SPOOF_THRESHOLD:
                color = (0, 255, 0)
                display_name = get_display_name(best_user, users_info)
                label = f"{display_name}"
                result = "VALIDO"
                last_best_user = best_user
            elif best_score >= SIMILARITY_THRESHOLD and spoof_score > SPOOF_THRESHOLD:
                color = (0, 255, 255)
                display_name = get_display_name(best_user, users_info)
                label = f"{display_name} SPOOF"
                result = "DENEGADO_SPOOF"
            else:
                color = (0, 0, 255)
                label = f"DESCONOCIDO"
                result = "DENEGADO"

            now = datetime.now(tz)

            # ✅ LOG INSTANTÁNEO DE SPOOF (aunque dure 0.2s)
            if result == "DENEGADO_SPOOF" and not spoof_logged:
                append_log(
                    now.isoformat(timespec="seconds"),
                    get_display_name(best_user, users_info),
                    "DENEGADO_SPOOF",
                    f"score={best_score:.4f},spoof={spoof_score:.4f}"
                )
                spoof_logged = True
                
            # CONTROL DE TIEMPO ESTABLE
            if result != last_status:
                status_start_time = now
                last_status = result
                spoof_logged = False   # ✅ se resetea al cambiar de estado

            elapsed = (now - status_start_time).total_seconds()

            # DIBUJO
            cv2.rectangle(annotated, (x, y), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.putText(
                annotated,
                f"Tiempo estable: {elapsed:.1f}s",
                (x, y2 + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            # DECISIÓN FINAL A LOS 5 SEGUNDOS
            if elapsed >= TIME_THRESHOLD and not decision_taken:
                decision_taken = True

                # 🟢 CASO VERDE → PASA A PATRÓN
                if result == "VALIDO":
                    cap.release()
                    cv2.destroyAllWindows()

                    display_name = get_display_name(last_best_user, users_info)
                    print(f"[🔑] Acceso confirmado ({TIME_THRESHOLD}s). Iniciando patrón para {display_name}...")

                    # ✅ LOG DEL RECONOCIMIENTO EXITOSO
                    append_log(
                        now.isoformat(timespec="seconds"),
                        display_name,
                        "VALIDO",
                        f"score={best_score:.4f},spoof={spoof_score:.4f}"
                    )

                    ok = pattern_auth.verify_pattern(last_best_user, duration=6)
                    now_str = now.isoformat(timespec="seconds")

                    if ok:
                        print("[✅] Patrón correcto. Acceso permitido.")
                        append_log(now_str, display_name, "PERMITIDO_PATRON", "pattern=OK")
                    else:
                        print("[❌] Patrón incorrecto. Acceso denegado.")
                        append_log(now_str, display_name, "DENEGADO_PATRON", "pattern=FAIL")

                    return

                # 🔴🟡 CASO ROJO O SPOOF → BLOQUEO
                else:
                    cap.release()
                    cv2.destroyAllWindows()
                    
                    usuario = get_display_name(best_user, users_info)

                    print("[❌] Usuario rechazado por 5 segundos en estado NO válido.")
                    append_log(
                        now.isoformat(timespec="seconds"),
                        usuario,
                        result,
                        f"score={best_score:.4f},spoof={spoof_score:.4f}"
                    )
                    return

        cv2.imshow("Verify - Modo Automático por Tiempo", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        verify_loop()
    except FileNotFoundError as e:
        print("❌", e)
