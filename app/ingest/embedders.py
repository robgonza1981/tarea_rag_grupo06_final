# app/ingest/embedders.py
from __future__ import annotations
from typing import List
from langchain_core.embeddings import Embeddings
from openai import OpenAI
import os


class OpenAIEmbedder(Embeddings):
    """
    Wrapper mínimo que implementa la interfaz de LangChain (Embeddings)
    usando el SDK oficial 'openai' (>=1.x).
    Evita importar 'langchain_openai' y sus dependencias de chat_models.
    """

    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY no está definido.")
        self.client = OpenAI(api_key=api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # OpenAI recomienda limpiar entradas vacías o demasiado largas,
        # pero aquí asumimos que ya filtras/recortas antes.
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(model=self.model, input=[text])
        return resp.data[0].embedding
