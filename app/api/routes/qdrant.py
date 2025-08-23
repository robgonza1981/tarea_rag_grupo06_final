# app/api/routes/qdrant.py
from fastapi import APIRouter
from qdrant_client import QdrantClient
from app.settings import settings
from app.models.schemas import QdrantStats

router = APIRouter()

@router.get("/qdrant/stats", response_model=QdrantStats)
def qdrant_stats():
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=60)
    cols = [c.name for c in client.get_collections().collections]
    exists = settings.QDRANT_COLLECTION in cols
    points = None
    if exists:
        s = client.get_collection(settings.QDRANT_COLLECTION)
        points = s.points_count
    return QdrantStats(
        collection=settings.QDRANT_COLLECTION,
        exists=exists,
        points=points,
        collections=cols
    )
