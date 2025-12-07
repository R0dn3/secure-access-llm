from sqlalchemy import create_engine, Column, Integer, Float, String, LargeBinary, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "access.db")
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data"), exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, unique=True, index=True)
    name = Column(String)
    embedding = Column(Text)  # JSON string list of floats

class AccessLog(Base):
    __tablename__ = "access_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(DateTime, default=datetime.utcnow, index=True)
    user_id = Column(String, nullable=True)
    status = Column(String)  # granted | denied
    score = Column(Float)
    spoof_score = Column(Float)
    reason = Column(Text, nullable=True)

def init_db():
    Base.metadata.create_all(engine)
