from __future__ import annotations
import argparse, json, math
from pathlib import Path
from statistics import mean
from typing import Dict, Any, List, Optional

from qdrant_client import QdrantClient
from openai import OpenAI

from app.settings import settings
from app.ingest.embedders import OpenAIEmbedder
from app.rag.chains import build_chain

# Límite para no pasarle al juez textos enormes
MAX_CONTEXT_CHARS = getattr(settings, "MAX_CONTEXT_CHARS", 6000)

SYSTEM = """Eres un evaluador muy estricto. Vas a calificar la respuesta del asistente SOLO en función del CONTEXTO dado.
- Responde en español, breve y preciso. Si hay pasos/menús o CÓDIGOS DE MÓDULO/RUTAS (p.ej. 7.10.20.36, 7.1.1), tenlo en cuenta.
- Inclúyelos LITERALMENTE en viñetas. No inventes.
- Si la respuesta afirma algo que NO esté explícita o implícitamente respaldado por el contexto, bájale precisión y conciencia de contexto.
- Si omite elementos clave que sí están en el contexto, bájale completitud.
- Si divaga o añade info irrelevante al foco de la pregunta, bájale relevancia.
Devuelve SIEMPRE un JSON válido con estas claves (numéricas 0-10):
{
  "precision": <float>,
  "relevancia": <float>,
  "completitud": <float>,
  "conciencia_contexto": <float>,
  "comentario": "<breve justificación>"
}
No incluyas ningún texto fuera del JSON.
"""

USER_TPL = """PREGUNTA:
{question}

RESPUESTA DEL ASISTENTE:
{answer}

CONTEXTOS RETRIEVADOS (top-{k}):
{contexts}

INSTRUCCIONES:
- Califica de 0 a 10 (puedes usar decimales).
- Si el contexto es vacío o insuficiente, penaliza precisión y conciencia de contexto si la respuesta inventa.
- Devuelve SOLO el JSON pedido, sin texto extra.
"""

def _join_contexts(hits: List[Any], max_chars: int) -> str:
    blocks = []
    used = 0
    for r in hits:
        payload = (r.payload or {})
        txt = (payload.get("text") or payload.get("chunk") or payload.get("content") or "").strip()
        src = (payload.get("source") or "?").strip()
        if not txt:
            continue
        snippet = txt
        if used + len(snippet) > max_chars:
            snippet = snippet[: max(0, max_chars - used)]
        blocks.append(f"[source: {src}]\n{snippet}".strip())
        used += len(snippet)
        if used >= max_chars:
            break
    return "\n\n---\n\n".join(blocks) if blocks else "(sin contexto)"

def _qdrant_search(client: QdrantClient, embedder: OpenAIEmbedder, question: str, k: int) -> List[Any]:
    qv = embedder.embed_query(question)
    res = client.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=qv,
        limit=k,
        with_payload=True,
        score_threshold=settings.MIN_SCORE if settings.MIN_SCORE > 0 else None,
    )
    return res or []

def _as_float(d: Dict[str, Any], key: str) -> float:
    try:
        return float(d.get(key, 0))
    except Exception:
        return 0.0

def run(path_questions: Path, collection: str, k: int, model: Optional[str], verbose: bool=False):
    # 1) Fijar colección y K (para ser coherentes con la cadena si la usamos)
    settings.QDRANT_COLLECTION = collection
    settings.TOP_K = k

    # 2) Cargar preguntas respondibles
    questions = json.loads(path_questions.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise ValueError("El archivo de preguntas debe ser un JSON array.")

    # 3) Inicializar cliente LLM y retrieval
    judge_model = model or getattr(settings, "GEN_MODEL", "gpt-4.1-mini")
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    qclient = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=60)
    embedder = OpenAIEmbedder(model=settings.EMBED_MODEL, api_key=settings.OPENAI_API_KEY)

    # 4) Cadena de generación “real” (para producir la respuesta a evaluar)
    chain = build_chain(debug=True)  # debug=True para poder inspeccionar hits si lo necesitas

    rows: List[Dict[str, Any]] = []
    for i, row in enumerate(questions, 1):
        q = (row.get("q") or row.get("question") or "").strip()
        if not q:
            continue

        # a) generar respuesta del sistema actual
        out = chain({"question": q}) or {}
        answer = (out.get("answer") or "").strip()

        # b) recuperar contexto (top-k) desde Qdrant
        hits = _qdrant_search(qclient, embedder, q, k)
        contexts = _join_contexts(hits, MAX_CONTEXT_CHARS)

        # c) pedir juicio al LLM con sistema + usuario
        msg = USER_TPL.format(question=q, answer=answer, contexts=contexts, k=k)
        comp = client.chat.completions.create(
            model=judge_model,
            temperature=0,
            response_format={"type": "json_object"},  # ayuda a que devuelva JSON estricto
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": msg},
            ],
        )
        content = comp.choices[0].message.content or "{}"
        try:
            score = json.loads(content)
        except Exception:
            score = {}

        precision = _as_float(score, "precision")
        relevancia = _as_float(score, "relevancia")
        completitud = _as_float(score, "completitud")
        contexto   = _as_float(score, "conciencia_contexto")
        overall = mean([precision, relevancia, completitud, contexto])

        rows.append({
            "question": q,
            "answer": answer,
            "precision": precision,
            "relevancia": relevancia,
            "completitud": completitud,
            "conciencia_contexto": contexto,
            "overall": overall,
            "comment": score.get("comentario", ""),
        })

        if verbose:
            print(f"[{i}/{len(questions)}] {q}")
            print(f"  → overall={overall:.2f}  (P={precision:.2f}, R={relevancia:.2f}, C={completitud:.2f}, Ctx={contexto:.2f})")
            print("  sample ctx:", contexts[:200].replace("\n", " "), "...\n")

    # Agregados finales
    prec = mean(r["precision"] for r in rows) if rows else 0
    rel  = mean(r["relevancia"] for r in rows) if rows else 0
    comp = mean(r["completitud"] for r in rows) if rows else 0
    ctx  = mean(r["conciencia_contexto"] for r in rows) if rows else 0
    overall = mean([prec, rel, comp, ctx]) if rows else 0

    print("\n====================================")
    print("📊 RESULTADOS DE LA PRUEBA")
    print(f"Puntuación General: {overall:.2f}/10")
    print(f"Precisión: {prec:.2f}/10")
    print(f"Relevancia: {rel:.2f}/10")
    print(f"Completitud: {comp:.2f}/10")
    print(f"Conciencia de Contexto: {ctx:.2f}/10")

def _parse():
    ap = argparse.ArgumentParser(description="Eval LLM-as-a-judge con contexto real recuperado.")
    ap.add_argument("answerable_json", help="app/eval/questions_answerable.json")
    ap.add_argument("--collection", required=True, help="Colección en Qdrant a usar (ej: docs_recursive).")
    ap.add_argument("-k", "--top-k", type=int, default=None, help="Top-K para contexto (default: settings.TOP_K).")
    ap.add_argument("--model", default=None, help="Modelo juez (default: settings.GEN_MODEL).")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse()
    if args.top_k:
        settings.TOP_K = int(args.top_k)
    run(Path(args.answerable_json), args.collection, settings.TOP_K, args.model, args.verbose)
