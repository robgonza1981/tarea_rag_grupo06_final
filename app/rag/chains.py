# app/rag/chains.py
from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import OrderedDict
import unicodedata
import re
import importlib as _importlib

# ===== Parche Pydantic / OpenAI (compat) =====
from langchain_openai import ChatOpenAI
from langchain_core.caches import BaseCache as _LC_BaseCache  # noqa: F401
from langchain_core.callbacks.base import (
    BaseCallbackHandler as _LC_BaseCallbackHandler,  # noqa: F401
    Callbacks as _LC_Callbacks,                      # noqa: F401
)
from langchain_core.callbacks.manager import (
    BaseCallbackManager as _LC_BaseCallbackManager,  # noqa: F401
)

_co_base = _importlib.import_module("langchain_openai.chat_models.base")
for _name, _obj in [
    ("BaseCache", _LC_BaseCache),
    ("BaseCallbackHandler", _LC_BaseCallbackHandler),
    ("BaseCallbackManager", _LC_BaseCallbackManager),
    ("Callbacks", _LC_Callbacks),
]:
    if not hasattr(_co_base, _name):
        setattr(_co_base, _name, _obj)

try:
    ChatOpenAI.model_rebuild(force=True)
except Exception:
    pass
# =============================================

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from app.settings import settings
from app.ingest.embedders import OpenAIEmbedder

ABSTAIN = "No tengo información suficiente en los documentos para responder con certeza."

# --------------------------- Modelos para LangServe UI ---------------------------
class RAGInput(BaseModel):
    question: str = Field("", title="QUESTION", description="Escribe tu consulta")

class RAGOutput(BaseModel):
    answer: str

# --------------------------- Utilidades / estructuras ---------------------------
@dataclass
class PointHit:
    text: str
    score: float
    source: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None  # metadatos completos

def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")

# Normalización base (siglas y typos)
_RE_PL = [
    (re.compile(r"\bnc\b", re.I), "nota de credito"),
    (re.compile(r"\bov\b", re.I), "orden de venta"),
    (re.compile(r"\boc\b", re.I), "orden de compra"),
    (re.compile(r"\bot\b", re.I), "orden de trabajo"),
    (re.compile(r"\bfi\b", re.I), "factura interna"),
    (re.compile(r"\bdte\b", re.I), "documento tributario electronico"),
    (re.compile(r"nota\s+de\s+credi\w*", re.I), "nota de credito"),
    (re.compile(r"anular\s+compra[s]?", re.I), "anulacion de compras"),
]
def _normalize_base(q: str) -> str:
    q2 = _strip_accents(q.lower())
    for pat, repl in _RE_PL:
        q2 = pat.sub(repl, q2)
    q2 = re.sub(r"\s+", " ", q2).strip()
    return q2

# --------------------------- Expansión de consulta (dominio) ---------------------------
def _expand_domain_hints(qnorm: str) -> str:
    tokens: List[str] = []

    # Duplicar/clonar OV
    if (("orden de venta" in qnorm or re.search(r"\bov\b", qnorm))
        and re.search(r"\b(copia[r]?|duplicar|clonar|replicar)\b", qnorm)):
        tokens += [
            "duplicar orden de venta",
            "clonar orden de venta",
            "generar copia",
            "modulo 7.10.20.36",
            "qad 7.10.20.36",
            "archivo de confirmacion del pedido",
            "confirmacion del pedido",
        ]

    # Consultar/buscar/revisar/ver OV -> empuja 7.12.10
    if (("orden de venta" in qnorm or re.search(r"\bov\b", qnorm))
        and re.search(r"\b(consultar|buscar|revisar|ver)\b", qnorm)):
        tokens += ["7.12.10", "liberar orden de venta", "numero de cotizacion", "consulta orden de venta"]

    # Nueva factura a partir de OV
    if ("nueva factura" in qnorm) or re.search(r"\bfactura\s+nueva\b", qnorm):
        tokens += ["facturacion", "a partir de orden de venta", "referenciando la factura"]

    # Nota de Crédito
    if ("nota de credito" in qnorm) and re.search(r"\b(solicitar|emitir|generar)\b", qnorm):
        tokens += ["7.10.20.2", "referenciando la factura", "nota de credito contra factura"]

    # *** NUEVO: DTE / envío / rechazo / plazos ***
    if ("dte" in qnorm or "documento tributario electronico" in qnorm
        or "documentos tributarios electronicos" in qnorm
        or ("rechaz" in qnorm or "plazo" in qnorm or "llegan" in qnorm or "envio" in qnorm)):
        tokens += [
            "dte", "bcn", "sii", "xml", "pdf",
            "rechazo", "rechazar", "plazo 8 dias", "8 dias", "ocho dias",
            "envia a bcn", "distribuye", "envio a sii", "recepcion dte",
        ]

    # *** NUEVO: Validaciones antes de liberar OV (stock rápido) ***
    if (("orden de venta" in qnorm or "ov" in qnorm)
        and (re.search(r"\bvalidaci(ón|on|ones|ar)\b", qnorm) or "liberar" in qnorm or "stock rapido" in qnorm)):
        tokens += [
            "7.12.10",          # liberar OV
            "7.12.20.7",        # aprobacion comercial
            "3.6.30.31",        # disponibilidad
            "aprobacion comercial",
            "disponibilidad",
            "stock rapido",
        ]

    # Refuerzos cortos
    if "orden de venta" in qnorm and "ov" not in qnorm:
        tokens += ["ov"]
    if "nota de credito" in qnorm and "nc" not in qnorm:
        tokens += ["nc"]

    return (qnorm + " " + " ".join(tokens)).strip() if tokens else qnorm

# --------------------------- Rerank heurístico ---------------------------
_CODE_RE = re.compile(r"\b\d+(?:\.\d+){1,4}\b")  # 7.10.20.36, 7.1.1, etc.

def _keywords_for_rerank(qnorm: str) -> List[str]:
    kws = set()
    for w in qnorm.split():
        if len(w) > 3:
            kws.add(w)
    for m in re.findall(r"\d+(?:\.\d+){1,4}", qnorm):
        kws.add(m)
    if ("orden de venta" in qnorm or re.search(r"\bov\b", qnorm)) and re.search(r"\b(copia|duplicar|clonar)\b", qnorm):
        kws.update({"copia","duplicar","clonar","orden de venta","ov","7.10.20.36","confirmacion","pedido"})
    # señales DTE
    if ("dte" in qnorm or "documento tributario electronico" in qnorm):
        kws.update({"dte","bcn","sii","xml","pdf","rechazo","rechazar","8","dias","8 dias"})
    # señales validación/liberar OV
    if ("validacion" in qnorm or "liberar" in qnorm or "stock rapido" in qnorm) and ("orden de venta" in qnorm or "ov" in qnorm):
        kws.update({"7.12.10","7.12.20.7","3.6.30.31","aprobacion","disponibilidad","liberar","validacion"})
    return list(kws)

def _heuristic_rerank(qnorm: str, hits: List[PointHit]) -> List[PointHit]:
    kws = _keywords_for_rerank(qnorm)

    def score(h: PointHit) -> float:
        txt = _strip_accents((h.text or "").lower())
        base = float(h.score)
        overlap = sum(1 for k in kws if k in txt)
        bonus = 0.0
        if _CODE_RE.search(txt): bonus += 0.10
        if ("modulo" in txt or "módulo" in txt or "menu" in txt or "menú" in txt or "ruta" in txt): bonus += 0.05
        if ("orden de venta" in qnorm or "ov" in qnorm) and ("orden de venta" in txt or "ov" in txt): bonus += 0.04
        if "factura" in qnorm and "factura" in txt: bonus += 0.03
        if (h.payload or {}).get("strategy") == "hier": bonus += 0.03
        # DTE bonuses
        if ("dte" in qnorm or "documento tributario electronico" in qnorm):
            if ("bcn" in txt or "sii" in txt or "xml" in txt or "pdf" in txt): bonus += 0.08
            if ("rechaz" in txt and ("8 dias" in txt or "ocho dias" in txt or "8 dias corridos" in txt)): bonus += 0.10
        # validación/liberar OV
        if ("validacion" in qnorm or "liberar" in qnorm or "stock rapido" in qnorm):
            if ("aprobacion" in txt or "aprobación" in txt): bonus += 0.08
            if "disponibilidad" in txt or "3.6.30.31" in txt: bonus += 0.08
            if "7.12.10" in txt or "7.12.20.7" in txt: bonus += 0.08
        return base + 0.25 * overlap + bonus

    return sorted(hits, key=score, reverse=True)

# --------------------------- Búsqueda nativa + expansión + fallback ---------------------------
def _qdrant_native_search(question_norm: str, top_k: int, min_score: float) -> List[PointHit]:
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=60)
    embedder = OpenAIEmbedder(model=settings.EMBED_MODEL, api_key=settings.OPENAI_API_KEY)

    # PASO 1: consulta expandida (dirigida por dominio)
    q1 = _expand_domain_hints(question_norm)
    qvec1 = embedder.embed_query(q1)
    cand_limit = max(top_k * 3, 12)
    res = client.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=qvec1,
        limit=cand_limit,
        with_payload=True,
        score_threshold=min_score if (min_score and min_score > 0) else None,
    )

    # PASO 2: fallback si quedamos cortos o con score bajo
    need_fallback = (not res) or (len(res) == 0) or (float(getattr(res[0], "score", 0.0) or 0.0) < max(0.15, min_score - 0.05))
    if need_fallback:
        keep: List[str] = []
        for w in question_norm.split():
            if len(w) > 3 or re.match(r"\d", w):  # tokens "fuertes"
                keep.append(w)
        # enriquecer con sinónimos por dominio
        if ("dte" in question_norm or "documento tributario electronico" in question_norm):
            keep += ["dte","bcn","sii","xml","pdf","rechazo","8 dias"]
        if ("orden de venta" in question_norm or "ov" in question_norm):
            keep += ["7.12.10","7.12.20.7","3.6.30.31"]
        broaden = " ".join(keep).strip() or question_norm

        qvec2 = embedder.embed_query(broaden)
        res = client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=qvec2,
            limit=cand_limit,
            with_payload=True,
            score_threshold=None,  # laxo
        )

    hits: List[PointHit] = []
    for p in (res or []):
        payload = p.payload or {}
        txt = payload.get("text") or payload.get("page_content") or payload.get("content") or ""
        src = payload.get("source") or payload.get("file") or payload.get("doc") or None
        hits.append(PointHit(text=str(txt), score=float(getattr(p, "score", 0.0) or 0.0), source=src, payload=payload))

    return _heuristic_rerank(question_norm, hits)[:top_k]

# --------------------------- Contexto cohesionado por sección ---------------------------
def _format_context(hits: List[PointHit], max_chars: int) -> str:
    groups: "OrderedDict[Tuple[str, str], List[PointHit]]" = OrderedDict()
    for h in hits:
        md = h.payload or {}
        key = (str(md.get("doc_id")), str(md.get("section_title")))
        groups.setdefault(key, []).append(h)

    for k in groups:
        groups[k].sort(key=lambda hh: int((hh.payload or {}).get("position", 0)))

    parts: List[str] = []
    used = 0
    for (_, _), grp in groups.items():
        head_md = grp[0].payload or {}
        head = f"[{head_md.get('doc_title','?')} › {head_md.get('section_title','?')}]"
        body = "\n\n".join((g.text or "").strip() for g in grp if (g.text or "").strip())
        block = f"{head}\n{body}".strip()
        if not block:
            continue
        need = len(block) + (2 if parts else 0)
        if used + need > max_chars:
            block = block[: max(0, max_chars - used)]
        parts.append(block)
        used += len(block) + (2 if parts else 0)
        if used >= max_chars:
            break
    return "\n\n".join(parts)

def _extract_codes(text: str) -> List[str]:
    return list(dict.fromkeys(re.findall(r"\b\d+(?:\.\d+){1,4}\b", text or "")))

def _ensure_codes_in_answer(answer: str, context: str) -> str:
    codes_ctx = _extract_codes(context)
    if not codes_ctx:
        return answer
    missing = [c for c in codes_ctx if c not in (answer or "")]
    if not missing:
        return answer
    tail = "Códigos de módulo/menú vistos: " + ", ".join(missing)
    if answer.endswith((":", "·", "-", "•")):
        return f"{answer}\n{tail}"
    return f"{answer}\n\n{tail}"

# --------------------------- Prompt del RAG ---------------------------
_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un asistente experto y SOLO puedes responder con información del CONTEXTO. "
            f"Si el contexto no tiene soporte suficiente, responde EXACTAMENTE: \"{ABSTAIN}\". "
            "Responde en español, breve y preciso. Si hay pasos/menús o CÓDIGOS DE MÓDULO/RUTAS (p.ej. 7.10.20.36, 7.1.1), "
            "inclúyelos LITERALMENTE en viñetas o texto. No inventes.",
        ),
        ("human", "CONTEXT:\n{context}\n\nPREGUNTA: {question}\n\nRESPUESTA:"),
    ]
)

# --------------------------- 1) Func chain para /rag-json ---------------------------
def build_chain(debug: bool = False):
    llm = ChatOpenAI(model=settings.GEN_MODEL, temperature=0, streaming=False)

    def _invoke(inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = (inputs.get("question") or "").strip()
        if not question:
            return {"answer": "Falta el parámetro 'question'."}

        qnorm = _normalize_base(question)
        hits = _qdrant_native_search(qnorm, settings.TOP_K, getattr(settings, "MIN_SCORE", 0.0))
        context = _format_context(hits, getattr(settings, "MAX_CONTEXT_CHARS", 4000))

        msgs = _PROMPT.format_messages(question=question, context=context)
        resp = llm.invoke(msgs)
        answer = (getattr(resp, "content", None) or "").strip() or ABSTAIN
        answer = _ensure_codes_in_answer(answer, context)

        if debug:
            return {
                "answer": answer,
                "hits": [
                    {
                        "score": h.score,
                        "source": h.source,
                        "doc_id": (h.payload or {}).get("doc_id"),
                        "section": (h.payload or {}).get("section_title"),
                        "pos": (h.payload or {}).get("position"),
                        "text_preview": (h.text[:240] + "…") if len(h.text) > 240 else h.text,
                    }
                    for h in hits
                ],
            }
        return {"answer": answer}

    return _invoke

# --------------------------- 2) Runnable para /rag (stream) ---------------------------
def build_lc_runnable_rag():
    llm = ChatOpenAI(model=settings.GEN_MODEL, temperature=0, streaming=True)

    def _to_prompt_inputs(d: Dict[str, Any]) -> Dict[str, str]:
        q = (d.get("question") or "").strip()
        qnorm = _normalize_base(q)
        hits = _qdrant_native_search(qnorm, settings.TOP_K, getattr(settings, "MIN_SCORE", 0.0))
        context = _format_context(hits, getattr(settings, "MAX_CONTEXT_CHARS", 4000))
        return {"question": q, "context": context}

    chain = RunnableLambda(_to_prompt_inputs) | _PROMPT | llm | StrOutputParser()
    return chain.with_types(input_type=RAGInput, output_type=str)

# --------------------------- 3) Chequeo rápido para sugerencias ---------------------------
def quick_can_answer(
    query: str,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
    strict: bool = False,
) -> bool:
    """True si hay evidencia suficiente en el índice para sugerir este tópico."""
    qnorm = _normalize_base(query or "")
    k = top_k or settings.TOP_K
    thr = settings.__dict__.get("MIN_SCORE", 0.0) if min_score is None else float(min_score)
    hits = _qdrant_native_search(qnorm, k, thr)
    if not hits:
        return False

    if not strict:
        return any((h.text or "").strip() and float(h.score) >= thr for h in hits)

    top = hits[0]
    txt = _strip_accents((top.text or "").lower())
    kw = _keywords_for_rerank(qnorm)
    overlap = sum(1 for k in kw if k in txt)
    has_code = bool(_CODE_RE.search(txt))
    from_hier = ((top.payload or {}).get("strategy") == "hier")

    return (
        float(top.score) >= (thr + 0.10)
        and ((overlap >= 2) or has_code)
        and from_hier
    )
