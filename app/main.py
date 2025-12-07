import os
import cv2
import numpy as np
import base64
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from .models.detection import detect_and_crop_face
from .models.recognition import ArcFaceRecognizer   # ✅ actualizado
from .models.liveness import estimate_spoof_score
from .utils.db import init_db, SessionLocal, AccessLog
from .utils.access import enroll as enroll_user, verify as verify_user
from .utils.notifications import notify_teams, notify_email
from .utils.llm import summarize_logs_as_report
from datetime import datetime

app = FastAPI(title="Secure Access MCP/LLM")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar base de datos
init_db()

# ✅ Inicializar una sola instancia del recognizer
recognizer = ArcFaceRecognizer()

# -----------------------------
# Función para leer imágenes en base64
# -----------------------------
def read_image_from_b64(image_b64: str):
    _, encoded = image_b64.split(",", 1)
    nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# -----------------------------
# 1. Enroll → Registro de usuario
# -----------------------------
@app.post("/enroll")
async def enroll(request: Request):
    try:
        data = await request.json()
        user_id = data.get("user_id")
        name = data.get("name")
        img = read_image_from_b64(data["image_b64"])
        face = detect_and_crop_face(img)
        if face is None:
            return {"ok": False, "msg": "No face detected"}

        emb = recognizer.get_embedding(face)   # ✅ usar ArcFace
        enroll_user(user_id, name, emb)
        return {"ok": True, "msg": "User enrolled successfully"}
    except Exception as e:
        return {"ok": False, "msg": str(e)}

# -----------------------------
# 2. Verify → Verificación de usuario
# -----------------------------
@app.post("/verify")
async def verify(request: Request):
    try:
        data = await request.json()
        img = read_image_from_b64(data["image_b64"])
        face = detect_and_crop_face(img)
        if face is None:
            return {"status": "denied", "reason": "No face detected"}

        emb = recognizer.get_embedding(face)   # ✅ usar ArcFace
        spoof_score = estimate_spoof_score(img)
        status, uid, score, sscore, reason = verify_user(emb, spoof_score)

        # Notificación si hay intento sospechoso
        if status == "denied":
            title = "🚨 Access Denied"
            text = f"Reason: {reason}\nscore={score:.3f}, spoof_score={sscore:.2f}"
            notify_teams(title, text)
            notify_email(subject=title, body=text)

        return {
            "status": status,
            "user_id": uid,
            "score": score,
            "spoof_score": sscore,
            "reason": reason
        }
    except Exception as e:
        return {"status": "error", "reason": str(e)}

# -----------------------------
# 3. Logs → Consultar registros de acceso
# -----------------------------
@app.get("/logs")
def get_logs(since_iso: str = None, until_iso: str = None):
    db = SessionLocal()
    try:
        q = db.query(AccessLog)
        if since_iso:
            q = q.filter(AccessLog.ts >= datetime.fromisoformat(since_iso))
        if until_iso:
            q = q.filter(AccessLog.ts <= datetime.fromisoformat(until_iso))
        rows = q.order_by(AccessLog.ts.desc()).limit(500).all()
        return [
            {
                "ts": r.ts.isoformat(),
                "user_id": r.user_id,
                "status": r.status,
                "score": r.score,
                "spoof_score": r.spoof_score,
                "reason": r.reason,
            }
            for r in rows
        ]
    finally:
        db.close()

# -----------------------------
# 4. Report → Generar reporte básico
# -----------------------------
@app.post("/report")
def report():
    db = SessionLocal()
    try:
        logs_data = db.query(AccessLog).order_by(AccessLog.ts.desc()).limit(500).all()
        logs_dict = [
            {
                "ts": r.ts.isoformat(),
                "user_id": r.user_id,
                "status": r.status,
                "score": r.score,
                "spoof_score": r.spoof_score,
                "reason": r.reason,
            }
            for r in logs_data
        ]
        summary = summarize_logs_as_report(logs_dict)
        return {"summary": summary, "count": len(logs_data)}
    finally:
        db.close()
