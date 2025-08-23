# app/server_langserve.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import re
import unicodedata
import time
import random

from fastapi import FastAPI, APIRouter
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from qdrant_client import QdrantClient
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from app.rag.chains import (
    build_chain,
    build_lc_runnable_rag,
    ABSTAIN,
    quick_can_answer,
)
from app.rag.tools import build_agent_runnable
from app.settings import settings

# ================== FastAPI base ==================
app = FastAPI(title="FEN RAG · LangServe", version="1.8")

BASE_DIR = Path(__file__).resolve().parent
UI_DIR = BASE_DIR / "ui"
app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui/")

@app.get("/health", tags=["Info"])
def health():
    return {"ok": True}

# ---- Estado del servicio y de Qdrant (para demo/profesor)
@app.get("/status", tags=["Info"])
def status():
    return {
        "ok": True,
        "gen_model": settings.GEN_MODEL,
        "embed_model": settings.EMBED_MODEL,
        "collection": settings.QDRANT_COLLECTION,
        "top_k": settings.TOP_K,
        "min_score": getattr(settings, "MIN_SCORE", 0.0),
    }

@app.get("/qdrant/info", tags=["Info"])
def qdrant_info():
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=15)
    col = client.get_collection(settings.QDRANT_COLLECTION)
    cnt = client.count(collection_name=settings.QDRANT_COLLECTION).count
    return {
        "collection": settings.QDRANT_COLLECTION,
        "vectors_count": cnt,
        "vector_size": col.config.params.vectors.size,
        "distance": str(col.config.params.vectors.distance),
        "payload_indexes": list((col.payload_schema or {}).keys()),
    }

# ================== Modelos ==================
class RagInput(BaseModel):
    question: str = Field("", description="Pregunta del usuario")

class RagOutput(BaseModel):
    answer: str = Field(..., description="Respuesta del sistema")

class ChatUISchema(BaseModel):
    input: str = ""
    messages: List[dict] = Field(default_factory=list)

# ================== Utilidades ==================
def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if isinstance(p, dict):
                t = p.get("text")
                parts.append(t if isinstance(t, str) else str(t))
            else:
                parts.append(str(p))
        return " ".join(x for x in parts if x)
    return str(content)

def _dict_to_base_message(m: dict) -> Optional[BaseMessage]:
    role = (m.get("type") or m.get("_type") or "").lower()
    content = _extract_text_from_content(m.get("content", "")) if m is not None else ""
    if role in ("human", "human_message", "user"):
        return HumanMessage(content=content)
    if role in ("ai", "assistant"):
        return AIMessage(content=content)
    return None

def _last_user_text(messages: List[dict]) -> str:
    last = ""
    for m in messages:
        role = (m.get("type") or m.get("_type") or "").lower()
        if role in ("human", "human_message", "user"):
            last = _extract_text_from_content(m.get("content", "")) if m is not None else ""
    return (last or "").strip()

def _safe_str(o: Any) -> str:
    s = "" if o is None else str(o)
    return s.strip()

# ================== Sugerencias ==================
SUGGESTION_CATALOG: List[Tuple[str, str]] = [
    ("Consultar una orden de venta",           "buscar orden de venta 7.12.10"),
    ("Solicitar una nota de crédito",          "nota de crédito 7.10.20.2"),
    ("Proceso de contratación de personal KSB", "proceso contratación Workday"),
    ("Anulación de compras",                   "anulación de compras 25.15"),
    ("Crear una orden de compra",              "crear orden de compra 5.1.1"),
]
_SUG_CACHE: Dict[str, Any] = {"ts": 0.0, "labels": []}

# --- NUEVO: map label->probe y normalizador para reescritura ---
_LABEL2PROBE: Dict[str, str] = {label: probe for (label, probe) in SUGGESTION_CATALOG}

def _normalize(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def _rewrite_from_suggestion(text: str) -> str:
    t = _normalize(text)
    for label, probe in SUGGESTION_CATALOG:
        if _normalize(label) == t:
            return probe
    return text

def _render_suggestions(labels: List[str]) -> str:
    if not labels:
        return ""
    bullets = "\n".join(f"• {x}" for x in labels)
    return f"\n\nSugerencias:\n{bullets}"

@app.on_event("startup")
def _warm_suggestions_cache() -> None:
    try:
        labels: List[str] = []
        for label, probe in SUGGESTION_CATALOG:
            try:
                if quick_can_answer(probe, top_k=settings.TOP_K, strict=True):
                    labels.append(label)
            except Exception:
                pass
        _SUG_CACHE["labels"] = labels
        _SUG_CACHE["ts"] = time.time()
    except Exception:
        _SUG_CACHE["labels"] = []
        _SUG_CACHE["ts"] = time.time()

def _get_suggestions(limit: int = 3) -> List[str]:
    cached = list(_SUG_CACHE.get("labels") or [])
    labels = list(cached)
    if len(labels) < limit:
        for lbl, _ in SUGGESTION_CATALOG:
            if lbl not in labels:
                labels.append(lbl)
            if len(labels) >= limit:
                break
    if len(labels) > limit:
        return random.sample(labels, k=limit)
    return labels[:limit]

# ================== Smalltalk ==================
_GREET_PAT = re.compile(r"\b(hola|holi|buenas|hi|hello|ola|oli)\b", re.I)
_GREET_CHATTY_PAT = re.compile(r"(hola como estas|hola cómo estás|como estas|cómo estás|que tal|qué tal|como va|cómo va|como andas|cómo andas)", re.I)
_BYE_PAT = re.compile(r"\b(adios|adiós|chao|bye|hasta luego|nos vemos|hasta pronto|nos vimos|nada mas|es todo)\b", re.I)
_THANKS_PAT = re.compile(r"^(gracias|muchas gracias|thanks|ty|ok|vale|genial|perfecto|super|listo|genial gracias)$", re.I)

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def _is_smalltalk(text: str) -> Optional[str]:
    t = _strip_accents((text or "").strip().lower())
    if not t:
        return "greet"
    # gating: mensajes largos no pasan por smalltalk
    if len(t.split()) > 6:
        return None
    if _GREET_CHATTY_PAT.search(t):
        return "greet_chatty"
    if _GREET_PAT.search(t):
        return "greet"
    if _BYE_PAT.search(t):
        return "bye"
    if _THANKS_PAT.fullmatch(t):
        return "thanks"
    return None

_GREET_TEMPLATES = [
    "¡Hola! ¿Qué tal? ¿En qué puedo ayudarte hoy?",
    "¡Hola! Encantado de ayudarte. ¿Qué te gustaría saber?",
    "¡Hola! Estoy aquí para apoyarte con ventas, compras y RR.HH. ¿Qué necesitas?",
    "¡Hola! ¿En qué puedo ayudarte hoy?",
]
_GREET_CHATTY_TEMPLATES = [
    "¡Hola! Muy bien, gracias por preguntar. ¿En qué puedo ayudarte hoy?",
    "¡Hola! Todo en orden por aquí. ¿Qué necesitas?",
    "¡Hola! Listo para ayudarte. ¿Qué te gustaría saber?",
]
_BYE_TEMPLATES = [
    "¡Hasta luego! Si necesitas algo más sobre los procesos de KSB, aquí estaré.",
    "¡Nos vemos! Cuando quieras retomamos.",
]
_THANKS_TEMPLATES = [
    "¡De nada! ¿Te ayudo con algo más?",
    "¡Con gusto! Si quieres, puedo sugerirte más consultas.",
]

def _smalltalk_reply(kind: str) -> str:
    sugs = _get_suggestions()
    if kind == "greet_chatty":
        return f"{random.choice(_GREET_CHATTY_TEMPLATES)}{_render_suggestions(sugs)}"
    if kind == "greet":
        return f"{random.choice(_GREET_TEMPLATES)}{_render_suggestions(sugs)}"
    if kind == "bye":
        return random.choice(_BYE_TEMPLATES)
    return random.choice(_THANKS_TEMPLATES)

# ================== Runnables LangServe ==================
rag_fn   = build_chain(debug=False)
rag_json = RunnableLambda(rag_fn).with_types(input_type=RagInput, output_type=RagOutput)

rag_core = build_lc_runnable_rag()  # {"question": str} -> str

def _route_and_answer(payload: Dict[str, Any]) -> str:
    messages = payload.get("messages") or []
    text = (payload.get("input") or _last_user_text(messages) or "").strip()

    # smalltalk rápido
    sk = _is_smalltalk(text)
    if sk:
        return _smalltalk_reply(sk)

    # reescritura si coincide con un label de sugerencia
    text = _rewrite_from_suggestion(text)

    # RAG normal
    out: str = rag_core.invoke({"question": text}) or ""
    out = out.strip()
    if not out or out == ABSTAIN:
        return ABSTAIN + _render_suggestions(_get_suggestions())
    return out

rag_chat = RunnableLambda(_route_and_answer).with_types(
    input_type=ChatUISchema,
    output_type=str,
)

# agente
agent_core = build_agent_runnable(k=settings.TOP_K)
agent_chat = (
    RunnableLambda(lambda payload: {
        "input": (payload.get("input") or _last_user_text(payload.get("messages") or []) or "").strip(),
        "chat_history": [
            m for m in (_dict_to_base_message(m) for m in (payload.get("messages") or []))
            if m is not None
        ],
    })
    | agent_core
    | RunnableLambda(_safe_str)
).with_types(
    input_type=ChatUISchema,
    output_type=str,
)

# ---- Montamos LangServe playgrounds (para tu UI)
add_routes(app, rag_json, path="/rag-json", playground_type="default")
add_routes(app, rag_chat, path="/rag",   playground_type="chat")
add_routes(app, rag_chat, path="/chat",  playground_type="chat")  # <-- ¡RAG en /chat!
add_routes(app, agent_chat, path="/agent", playground_type="chat")

# Ocultamos estos endpoints del esquema para que /docs no reviente
for r in list(app.routes):
    try:
        if any(getattr(r, "path", "").startswith(p) for p in ("/rag-json", "/rag", "/chat", "/agent")):
            r.include_in_schema = False
    except Exception:
        pass

# ================== Wrappers documentados (para /docs) ==================
api = APIRouter(prefix="/api", tags=["API"])

@api.post("/rag-json", response_model=RagOutput, summary="RAG (JSON, sin streaming)")
def api_rag_json(body: RagInput) -> RagOutput:
    return RagOutput(**rag_fn(body.model_dump()))

@api.post("/rag", response_model=str, summary="RAG (chat, string)")
def api_rag(body: ChatUISchema) -> str:
    return _route_and_answer(body.model_dump())

@api.post("/agent", response_model=str, summary="Agente (herramientas)")
def api_agent(body: ChatUISchema) -> str:
    inner = {
        "input": (body.input or _last_user_text(body.messages) or "").strip(),
        "chat_history": [
            m for m in (_dict_to_base_message(m) for m in body.messages) if m is not None
        ],
    }
    out = agent_core.invoke(inner)
    return _safe_str(out)

app.include_router(api)
