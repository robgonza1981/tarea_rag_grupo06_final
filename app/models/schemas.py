# app/models/schemas.py
from typing import List, Optional
from pydantic import BaseModel

# ==== REQUESTS ====
class AskRequest(BaseModel):
    """Solo lo que el usuario debe enviar."""
    question: str


# ==== RESPONSES (PROD) ====
class AskResponse(BaseModel):
    """Respuesta simple para producción."""
    answer: str


# ==== RESPONSES (DEBUG) ====
class Hit(BaseModel):
    text: Optional[str] = None
    source: Optional[str] = None
    score: Optional[float] = None
    chunk_id: Optional[int] = None

class AskDebugResponse(BaseModel):
    """Respuesta extendida para diagnóstico."""
    answer: str
    k: int
    hits: List[Hit] = []

    # app/models/schemas.py

class QdrantStats(BaseModel):
    collection: str
    exists: bool
    points: int
    vectors: int | None = None

