import json
import numpy as np
import pytz
from datetime import datetime
from .db import SessionLocal, User, AccessLog

# ==========================
# CONFIGURACIÓN DE UMBRALES
# ==========================
RECOG_THRESHOLD = 0.5      # cosine distance threshold (lower => more similar)
SPOOF_THRESHOLD = 0.6      # spoof_score <= this means 'live'

# ==========================
# FUNCIONES DE UTILIDAD
# ==========================
def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula la distancia coseno entre dos embeddings."""
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# ==========================
# REGISTRO DE USUARIO (ENROLL)
# ==========================
def enroll(user_id: str, name: str, embedding: np.ndarray):
    """Registra o actualiza un usuario con su embedding facial."""
    db = SessionLocal()
    try:
        e = json.dumps(embedding.tolist())
        u = db.query(User).filter(User.user_id == user_id).first()
        if u is None:
            u = User(user_id=user_id, name=name, embedding=e)
            db.add(u)
        else:
            u.name = name
            u.embedding = e
        db.commit()
    finally:
        db.close()

# ==========================
# VERIFICACIÓN FACIAL (VERIFY)
# ==========================
def verify(embedding: np.ndarray, spoof_score: float):
    """
    Verifica la identidad de un usuario comparando embeddings y puntaje de spoofing.
    Guarda los resultados en la base de datos con hora local (Perú).
    """
    db = SessionLocal()
    try:
        users = db.query(User).all()
        best_uid, best_score = None, 1e9
        emb = embedding

        # Buscar la mejor coincidencia (menor distancia)
        for u in users:
            ref = np.array(json.loads(u.embedding), dtype='float32')
            d = cosine_distance(emb, ref)
            if d < best_score:
                best_score, best_uid = d, u.user_id

        # Determinar resultado
        status, reason = "denied", None
        if spoof_score > SPOOF_THRESHOLD:
            reason = f"spoof_score demasiado alto ({spoof_score:.2f})"
        elif best_score <= RECOG_THRESHOLD and best_uid is not None:
            status = "granted"
        else:
            reason = f"sin coincidencia cercana (distancia mínima={best_score:.3f})"

        # ==========================
        # HORA LOCAL (LIMA, PERÚ)
        # ==========================
        tz = pytz.timezone("America/Lima")
        lima_time = datetime.now(tz)
        formatted_time = lima_time.strftime("%Y-%m-%dT%H:%M:%S-05:00")

        # ✅ Guardar como texto con hora local real
        log = AccessLog(
            ts=formatted_time,
            user_id=best_uid,
            status=status,
            score=best_score,
            spoof_score=spoof_score,
            reason=reason
        )
        db.add(log)
        db.commit()

        return status, best_uid, float(best_score), float(spoof_score), reason

    finally:
        db.close()
