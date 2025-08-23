# app/rag/retriever.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchText
from langchain_openai import OpenAIEmbeddings
from app.settings import settings

def _norm(text: Any) -> str:
    s = "" if text is None else str(text)
    return " ".join(s.strip().split())

class QdrantRetriever:
    """
    Recupera documentos desde Qdrant usando embeddings OpenAI.
    Guarda texto completo en el payload (campo 'text') para que el LLM lo use como contexto.
    """

    def __init__(self, top_k: Optional[int] = None):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=60,
        )
        self.embed = OpenAIEmbeddings(model=settings.EMBED_MODEL, disallowed_special=())
        self.top_k = top_k or settings.TOP_K
        self.collection = settings.QDRANT_COLLECTION

    def get(self, query: str, k: Optional[int] = None) -> List[Any]:
        """
        Busca en Qdrant con vector similarity. Si `k` viene None usa self.top_k.
        Devuelve la lista de hits (objetos de qdrant_client) con .payload y .score.
        """
        q = _norm(query)
        top_k = int(k or self.top_k)

        # Vector de consulta
        qv = self.embed.embed_query(q)

        # (Opcional) filtro por texto si quieres priorizar coincidencias literales:
        # text_filter = Filter(
        #     must=[FieldCondition(key="text", match=MatchText(text=q))]
        # )

        res = self.client.search(
            collection_name=self.collection,
            query_vector=qv,
            limit=top_k,
            # filter=text_filter,  # si decides activarlo, descomenta el bloque de Filter arriba
        )
        return res

    def pack_hits(self, hits: List[Any]) -> List[Dict[str, Any]]:
        """Convierte los hits de Qdrant a dicts simples (para respuestas debug)."""
        out = []
        for h in hits or []:
            payload = getattr(h, "payload", None) or {}
            out.append({
                "text": payload.get("text"),
                "source": payload.get("source"),
                "score": float(getattr(h, "score", 0.0) or 0.0),
                "chunk_id": payload.get("chunk_id"),
            })
        return out