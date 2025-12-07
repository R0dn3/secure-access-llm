# Secure Access Control — Face Recognition + Anti-Spoofing + MCP/LLM (Starter)

This is a **starter project** for a system that performs:
- **Face Detection** (MediaPipe)
- **Face Recognition** (MobileNetV2 embeddings as a lightweight baseline; replaceable with FaceNet)
- **Anti-Spoofing/Liveness** (blink detection via MediaPipe FaceMesh + texture cue)
- **Access Control Orchestration** (FastAPI backend + SQLite)
- **MCP/LLM integration stubs** (Teams/Outlook notifications, Azure OpenAI report summaries)

> ⚠️ This is a **teaching/PoC scaffold**. You can swap in stronger recognizers (FaceNet, ArcFace) and PAD models later.

## Quick start

```bash
# 1) Create env (Python 3.10+ recommended)
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# 2) Install deps (CPU-friendly set; feel free to pin versions)
pip install -r requirements.txt

# 3) Launch API (FastAPI)
uvicorn app.main:app --reload --port 8000

# 4) Launch UI (Streamlit) in another terminal
streamlit run ui/app.py
```

## Environment variables (optional)

Create a `.env` file (or export as env vars) to enable notifications/LLM:
```
TEAMS_WEBHOOK_URL= https://outlook.office.com/webhook/...
SMTP_HOST=smtp.office365.com
SMTP_PORT=587
SMTP_USER=your_outlook_user@domain.com
SMTP_PASS=app_password_or_secret
ALERT_EMAIL=security@domain.com

AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini   # or your deployed model name
```

## Project layout

```
app/
  main.py                # FastAPI app and endpoints
  schemas.py             # Pydantic models
  models/
    detection.py         # Face detection/cropping (MediaPipe)
    recognition.py       # Embedding extractor (MobileNetV2 baseline)
    liveness.py          # Blink-based + texture liveness
  utils/
    db.py                # SQLite via SQLAlchemy
    access.py            # Decision logic
    notifications.py     # Teams + Outlook email
    llm.py               # Azure OpenAI helper (summaries/reports)
ui/
  app.py                 # Streamlit client (webcam + API)
data/                    # embeddings/db files (created at runtime)
requirements.txt
README.md
```

## Notes

- Replace `app/models/recognition.py` with **FaceNet/ArcFace** for production.
- Replace/extend `app/models/liveness.py` with a **CNN PAD model** for higher robustness.
- MCP integration: this scaffold uses **Microsoft 365 hooks** (Teams/Outlook) and an **Azure OpenAI** client to emulate Copilot-style capabilities.
