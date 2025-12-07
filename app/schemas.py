from pydantic import BaseModel
from typing import List, Optional

class EnrollRequest(BaseModel):
    user_id: str
    name: str
    image_b64: str  # base64-encoded RGB image

class VerifyRequest(BaseModel):
    image_b64: str

class VerifyResponse(BaseModel):
    status: str  # 'granted' | 'denied'
    user_id: Optional[str] = None
    score: float
    spoof_score: float
    reason: Optional[str] = None

class ReportRequest(BaseModel):
    since_iso: Optional[str] = None  # ISO8601 start time
    until_iso: Optional[str] = None  # ISO8601 end time

class LogEntry(BaseModel):
    ts: str
    user_id: Optional[str]
    status: str
    score: float
    spoof_score: float
    reason: Optional[str] = None
