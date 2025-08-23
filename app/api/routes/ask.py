# app/api/routes/ask.py
from fastapi import APIRouter
from app.models.schemas import (
    AskRequest,
    AskResponse,
    AskDebugResponse,
)
from app.services.rag_service import RAGService
from app.rag.chains import build_chain

router = APIRouter(tags=["ask"])

# Instancia única del servicio para este router
_service = RAGService(build_chain())

@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest):
    """
    Endpoint de producción: SOLO devuelve `answer`.
    En OpenAPI/Swagger el usuario verá solo `question` como campo de entrada.
    """
    out = _service.ask(body.question, debug=False)
    return AskResponse(answer=out["answer"])

@router.post("/ask_debug", response_model=AskDebugResponse)
def ask_debug(body: AskRequest):
    """
    Endpoint de diagnóstico: devuelve `answer`, `k` y `hits`.
    Útil para evaluar recuperación y fuentes.
    """
    out = _service.ask(body.question, debug=True)
    return AskDebugResponse(**out)
