# app/services/rag_service.py
from __future__ import annotations
from typing import Any, Callable, Dict, Optional

from app.settings import settings
from app.rag.chains import build_chain


class RAGService:
    """
    Orquesta el chain de RAG. El chain realiza:
      - retrieve en Qdrant
      - formateo de contexto
      - llamada al LLM (OpenAI SDK)
      - retorno: {"answer"} y, en modo debug, {"answer","hits"}
    """

    def __init__(self, chain: Optional[Callable] = None, debug_default: bool = False):
        """
        Si 'chain' es None, construye uno con build_chain(debug=debug_default).
        Si 'chain' tiene .invoke, se adapta para exponer una función callable.
        """
        self._debug_default = debug_default

        if chain is None:
            self._chain_fn = build_chain(debug=debug_default)  # callable(inputs)->dict
        else:
            self._chain_fn = self._wrap_callable(chain)

    @staticmethod
    def _wrap_callable(obj: Any) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Acepta:
          - un callable normal: lambda inputs: {...}
          - un Runnable de LangChain con .invoke(inputs)
        Devuelve siempre una función callable(inputs)->dict
        """
        if callable(obj) and not hasattr(obj, "invoke"):
            return obj
        if hasattr(obj, "invoke"):
            return lambda inputs: obj.invoke(inputs)
        raise TypeError("El parámetro 'chain' debe ser un callable o tener método .invoke(inputs).")

    def ask(self, question: str, k: Optional[int] = None, debug: bool = False) -> Dict[str, Any]:
        # Normalización mínima
        q = (question or "").strip()

        # Respuesta amable si viene vacío
        if not q:
            msg = "¡Hola! ¿En qué te puedo ayudar hoy?"
            return {"answer": msg} if not debug else {"answer": msg, "k": 0, "hits": []}

        # Si el modo debug solicitado difiere del default con el que se construyó,
        # construimos un chain temporal con ese modo para esta invocación.
        if debug != self._debug_default:
            chain_fn = build_chain(debug=debug)
        else:
            chain_fn = self._chain_fn

        # Ejecutar el chain (callable(inputs) -> dict)
        out = chain_fn({"question": q}) or {}

        answer = (out.get("answer") or "").strip()
        if not answer:
            answer = "No tengo información suficiente para responder."

        if not debug:
            return {"answer": answer}

        # En modo debug, devolvemos 'hits' si están disponibles; si no, lista vacía
        return {
            "answer": answer,
            "k": k or settings.TOP_K,
            "hits": out.get("hits", []),
        }
