from __future__ import annotations

# Compat pydantic/chatopenai
from langchain_openai import ChatOpenAI
from langchain_core.caches import BaseCache as _LC_BaseCache  # noqa: F401
from langchain_core.callbacks.base import (
    BaseCallbackHandler as _LC_BaseCallbackHandler,  # noqa: F401
    Callbacks as _LC_Callbacks,                      # noqa: F401
)
from langchain_core.callbacks.manager import (
    BaseCallbackManager as _LC_BaseCallbackManager,  # noqa: F401
)
import importlib as _importlib
_co_base = _importlib.import_module("langchain_openai.chat_models.base")  # noqa: F401

from typing import List
from operator import itemgetter
import os

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor

from qdrant_client import QdrantClient
from app.ingest.embedders import OpenAIEmbedder
from app.settings import settings

ABSTAIN = "No tengo información suficiente en los documentos para responder con certeza."

# ---- “perillas” vía env (con defaults razonables) ----
AGENT_MAX_SNIPPET = int(os.getenv("AGENT_MAX_SNIPPET", "220"))   # chars por fragmento de contexto
AGENT_TOP_K       = int(os.getenv("AGENT_TOP_K", "2"))           # cuántos fragmentos pasar al LLM
AGENT_MAX_TOKENS  = int(os.getenv("AGENT_MAX_TOKENS", "80"))     # tope de tokens de salida


def _native_search(query: str, k: int) -> List[str]:
    """Busca pasajes relevantes en Qdrant (búsqueda nativa con QdrantClient)."""
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=60)
    embedder = OpenAIEmbedder(model=settings.EMBED_MODEL, api_key=settings.OPENAI_API_KEY)
    vec = embedder.embed_query(query)

    res = client.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=vec,
        limit=k,
        with_payload=True,
        score_threshold=getattr(settings, "MIN_SCORE", 0.0) or None,  # si es 0, no filtra
    )
    out: List[str] = []
    for p in res:
        pl = p.payload or {}
        txt = pl.get("text") or pl.get("page_content") or pl.get("content") or ""
        src = pl.get("source") or pl.get("file") or pl.get("doc") or "doc"
        out.append(f"[{src}] {str(txt)[:AGENT_MAX_SNIPPET]}")
    return out


def build_agent_runnable(k: int = 4):
    """Runnable que acepta {'input': str, 'chat_history': List[BaseMessage]} y retorna str."""
    # si viene AGENT_TOP_K en env, lo usamos por sobre el parámetro
    k = int(os.getenv("AGENT_TOP_K", str(k)))

    @tool("kb_search", return_direct=False)
    def kb_search(query: str) -> str:
        """
        Busca en la base de conocimiento (Qdrant) y devuelve como máximo
        AGENT_TOP_K fragmentos cortos, cada uno con su fuente entre corchetes.
        Si no hay resultados, retorna cadena vacía.
        """
        hits = _native_search(query, k)
        if not hits:
            return ""
        return "\n\n".join(hits)

    llm = ChatOpenAI(
        model=settings.GEN_MODEL,
        temperature=0,
        api_key=settings.OPENAI_API_KEY,
        streaming=True,
        max_tokens=AGENT_MAX_TOKENS,  # <-- recorta la salida
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un agente que DEBE usar la herramienta 'kb_search' antes de responder. "
                f"Si el contexto no respalda la respuesta, contesta exactamente: '{ABSTAIN}'. "
                "Responde en español con **máximo una o dos frases** (25–35 palabras). "
                "Solo si la pregunta requiere listar tipos/pasos, entrega **hasta 3 viñetas** "
                "muy breves (6–10 palabras cada una). No repitas texto del contexto innecesariamente."
            ),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm, [kb_search], prompt)

    # AgentExecutor retorna un dict; lo convertimos a string para el chat playground
    executor = AgentExecutor(
        agent=agent,
        tools=[kb_search],
        verbose=False,
        handle_parsing_errors=True,
        return_intermediate_steps=False,
    )

    return executor | itemgetter("output")
