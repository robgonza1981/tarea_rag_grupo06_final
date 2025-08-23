# app/core/deps.py
from functools import lru_cache
from qdrant_client import QdrantClient
from app.settings import settings
from app.rag.retriever import QdrantRetriever
from app.rag.chains import build_chain

@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=60)

@lru_cache(maxsize=1)
def get_retriever() -> QdrantRetriever:
    return QdrantRetriever(top_k=settings.TOP_K)

@lru_cache(maxsize=1)
def get_chain():
    return build_chain()
