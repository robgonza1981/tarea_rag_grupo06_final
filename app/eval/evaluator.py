# app/eval/evaluator.py
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient

from app.settings import settings
from app.ingest.embedders import OpenAIEmbedder
from app.rag.chains import build_chain

ABSTENTION_PHRASE = "No tengo información suficiente en los documentos para responder con certeza."


# --------------------------
# Carga y normalización
# --------------------------

def _load_json_array(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} no es un JSON de lista/array.")
    return data


def _norm_answerable(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Soporta campos:
      - q | question
      - expect (opcional, string de referencia)
      - expect_any (opcional, lista de keywords; basta con alguna)
      - expect_all (opcional, lista de keywords; deben estar todas)
      - gold_sources (opcional, lista de substrings para matchear 'source' de Qdrant)
    Devuelve: {"q": str, "expect": Optional[str], "expect_any": List[str], "expect_all": List[str], "gold_sources": List[str]}
    """
    out: List[Dict[str, Any]] = []
    for r in rows:
        q = (r.get("q") or r.get("question") or "").strip()
        expect = r.get("expect")
        if isinstance(expect, str):
            expect = expect.strip()

        def _listify(x):
            if not x:
                return []
            if isinstance(x, str):
                return [x]
            return list(x)

        expect_any = _listify(r.get("expect_any"))
        expect_all = _listify(r.get("expect_all"))
        gold = _listify(r.get("gold_sources"))

        out.append({
            "q": q,
            "expect": expect,
            "expect_any": expect_any,
            "expect_all": expect_all,
            "gold_sources": gold,
        })
    return out


def _norm_unanswerable(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        q = (r.get("q") or r.get("question") or "").strip()
        out.append({"q": q})
    return out


# --------------------------
# Utilidades de matching
# --------------------------

def _source_matches_gold(hit_source: Optional[str], gold_sources: List[str]) -> bool:
    if not hit_source:
        return False
    hs = hit_source.replace("\\", "/").lower()
    for g in gold_sources:
        if not g:
            continue
        if g.lower() in hs:
            return True
    return False


def _substring_ok(answer: str, expect: Optional[str]) -> bool:
    if not expect:
        return ABSTENTION_PHRASE not in (answer or "")
    return expect.lower() in (answer or "").lower()


def _cos_sim(u: List[float], v: List[float]) -> float:
    num = sum(a * b for a, b in zip(u, v))
    den = math.sqrt(sum(a * a for a in u)) * math.sqrt(sum(b * b for b in v))
    return (num / den) if den else 0.0


def _similarity_ok(embedder: OpenAIEmbedder, answer: str, expect: Optional[str], thr: float) -> Tuple[bool, float]:
    if not expect:
        return (ABSTENTION_PHRASE not in (answer or ""), 0.0)
    av = embedder.embed_query(answer or "")
    ev = embedder.embed_query(expect or "")
    sim = _cos_sim(av, ev)
    return (sim >= thr, sim)


def _match_keywords(
    embedder: OpenAIEmbedder,
    answer: str,
    keywords: List[str],
    mode: str,
    thr: float,
) -> Tuple[bool, bool, List[Dict[str, Any]]]:
    """
    Retorna:
      any_ok: True si alguna keyword matchea
      all_ok: True si todas las keywords matchean
      details: [{kw, substring, sim, ok}, ...]
    """
    if not keywords:
        return True, True, []

    details: List[Dict[str, Any]] = []
    oks: List[bool] = []

    for kw in keywords:
        sub = kw.lower() in (answer or "").lower()
        sim_ok, sim_val = (False, None)
        if mode in ("similarity", "both"):
            sim_ok, sim_val = _similarity_ok(embedder, answer, kw, thr)

        if mode == "substring":
            ok = sub
        elif mode == "similarity":
            ok = sim_ok
        else:  # both
            ok = sub or sim_ok

        oks.append(ok)
        details.append({"kw": kw, "substring": sub, "sim": sim_val, "ok": ok})

    any_ok = any(oks)
    all_ok = all(oks)
    return any_ok, all_ok, details


# --------------------------
# Búsqueda en Qdrant
# --------------------------

def _qdrant_search(client: QdrantClient, embedder: OpenAIEmbedder, question: str, k: int) -> List[Any]:
    qvec = embedder.embed_query(question)
    res = client.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=qvec,
        limit=k,
        with_payload=True,
        score_threshold=settings.MIN_SCORE if settings.MIN_SCORE > 0 else None,
    )
    return res


# --------------------------
# Evaluación principal
# --------------------------

def run_eval(
    collection: str,
    ans_path: Path,
    unans_path: Path,
    out_csv: Path,
    top_k: Optional[int] = None,
    verbose: bool = False,
    expect_mode: str = "both",           # "substring" | "similarity" | "both"
    expect_sim_threshold: float = 0.78,  # umbral de similitud semántica
) -> None:
    """
    - recall@k en respondibles (si hay 'gold_sources', matchea; si no, cuenta hit>0)
    - answer_ok en respondibles:
        * Opción A: campo 'expect' (substring/similitud según expect_mode)
        * Opción B: 'expect_any'/'expect_all' (por keyword), si existen
      (Si hay expect_any/all, sustituyen a 'expect'.)
    - abstention_rate en no respondibles: respuesta debe contener ABSTENTION_PHRASE
    """
    # 1) Ajustar colección y K objetivo (también para el chain de generación)
    settings.QDRANT_COLLECTION = collection
    K = int(top_k or settings.TOP_K)
    settings.TOP_K = K  # para que el chain use el mismo K

    # 2) Cargar datasets
    ans_rows = _norm_answerable(_load_json_array(ans_path))
    unans_rows = _norm_unanswerable(_load_json_array(unans_path))

    if not ans_rows and not unans_rows:
        print("⚠️  No hay preguntas en los JSON provistos.")
        return

    # 3) Clientes reutilizables
    qclient = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=60)
    embedder = OpenAIEmbedder(model=settings.EMBED_MODEL, api_key=settings.OPENAI_API_KEY)
    chain_normal = build_chain(debug=False)  # callable(inputs)->dict

    # 4) Resultados fila a fila
    rows_out: List[Dict[str, Any]] = []

    # --- Answerable ---
    ans_hits_flags: List[float] = []
    ans_answer_ok_flags: List[float] = []

    for i, item in enumerate(ans_rows, start=1):
        q = item["q"]
        expect = item.get("expect")
        expect_any = item.get("expect_any", [])
        expect_all = item.get("expect_all", [])
        gold_sources = item.get("gold_sources", [])

        # Retrieval
        res = _qdrant_search(qclient, embedder, q, K)
        any_hit = False
        top_source = None
        top_score = None
        if res:
            top_source = (res[0].payload or {}).get("source")
            top_score = float(getattr(res[0], "score", 0.0) or 0.0)
            if gold_sources:
                any_hit = any(_source_matches_gold((r.payload or {}).get("source"), gold_sources) for r in res)
            else:
                any_hit = True  # si no hay gold, basta con tener ≥1 hit

        # Generación
        out = chain_normal({"question": q}) or {}
        ans = (out.get("answer") or "").strip()
        abstained = ABSTENTION_PHRASE in ans

        # --- Comprobación de expectativa ---
        substring_ok = None
        sim_val = None
        kw_any_ok = None
        kw_all_ok = None
        kw_detail: List[Dict[str, Any]] = []

        # Si hay keywords, se usan en lugar de 'expect'
        if expect_any or expect_all:
            if expect_any:
                any_ok, _, det_any = _match_keywords(embedder, ans, expect_any, expect_mode, expect_sim_threshold)
            else:
                any_ok, det_any = True, []

            if expect_all:
                _, all_ok, det_all = _match_keywords(embedder, ans, expect_all, expect_mode, expect_sim_threshold)
            else:
                all_ok, det_all = True, []

            kw_any_ok = any_ok
            kw_all_ok = all_ok
            kw_detail = det_any + det_all
            expect_ok = any_ok and all_ok
        else:
            # Solo 'expect' string
            substring_ok = _substring_ok(ans, expect)
            sim_ok = False
            if expect and expect_mode in ("similarity", "both"):
                sim_ok, sim_val = _similarity_ok(embedder, ans, expect, expect_sim_threshold)
            if expect_mode == "substring":
                expect_ok = substring_ok
            elif expect_mode == "similarity":
                expect_ok = sim_ok
            else:
                expect_ok = (substring_ok or sim_ok)

        answer_ok = (not abstained) and expect_ok

        ans_hits_flags.append(1.0 if any_hit else 0.0)
        ans_answer_ok_flags.append(1.0 if answer_ok else 0.0)

        if verbose:
            print(f"\n[Answerable {i}/{len(ans_rows)}] Q: {q}")
            if gold_sources:
                print(f"  gold_sources: {gold_sources}")
            if res:
                print("  hits:")
                for r in res:
                    src = (r.payload or {}).get("source")
                    sc = float(getattr(r, "score", 0.0) or 0.0)
                    print(f"    - {src} | score={sc:.3f}")
            else:
                print("  hits: (sin resultados)")
            if expect_any or expect_all:
                print(f"  kw_any_ok: {kw_any_ok} | kw_all_ok: {kw_all_ok}")
            else:
                print(f"  substring_ok: {substring_ok} | sim={sim_val}")
            print(f"  any_hit: {any_hit} | abstained: {abstained} | answer_ok: {answer_ok}")
            if expect:
                print(f"  expect: {expect}")
            if expect_any:
                print(f"  expect_any: {expect_any}")
            if expect_all:
                print(f"  expect_all: {expect_all}")
            print(f"  answer: {ans[:240]}{'…' if len(ans) > 240 else ''}")

        rows_out.append({
            "collection": collection,
            "type": "answerable",
            "question": q,
            "any_hit": any_hit,
            "abstained": abstained,
            "answer_ok": answer_ok,
            "top_score": top_score,
            "top_source": top_source,
            "expect_mode": expect_mode,
            "expect_sim": sim_val,
            "substring_ok": substring_ok,
            "kw_any_ok": kw_any_ok,
            "kw_all_ok": kw_all_ok,
            "kw_detail": json.dumps(kw_detail, ensure_ascii=False) if kw_detail else None,
        })

    # --- Unanswerable ---
    unans_abstain_flags: List[float] = []

    for j, item in enumerate(unans_rows, start=1):
        q = item["q"]

        out = chain_normal({"question": q}) or {}
        ans = (out.get("answer") or "").strip()
        abstained = ABSTENTION_PHRASE in ans
        unans_abstain_flags.append(1.0 if abstained else 0.0)

        if verbose:
            print(f"\n[Unanswerable {j}/{len(unans_rows)}] Q: {q}")
            print(f"  abstained: {abstained}")
            print(f"  answer: {ans[:240]}{'…' if len(ans) > 240 else ''}")

        rows_out.append({
            "collection": collection,
            "type": "unanswerable",
            "question": q,
            "any_hit": None,
            "abstained": abstained,
            "answer_ok": abstained,  # OK si se abstiene
            "top_score": None,
            "top_source": None,
            "expect_mode": None,
            "expect_sim": None,
            "substring_ok": None,
            "kw_any_ok": None,
            "kw_all_ok": None,
            "kw_detail": None,
        })

    # 5) Agregados
    recall_at_k = mean(ans_hits_flags) if ans_hits_flags else 0.0
    answer_ok_rate = mean(ans_answer_ok_flags) if ans_answer_ok_flags else 0.0
    abstention_rate = mean(unans_abstain_flags) if unans_abstain_flags else 0.0

    print("\n========== RESULTADOS ==========")
    print(f"Collection         : {collection}")
    print(f"K                  : {K}")
    print(f"Answerable         : {len(ans_rows)} preguntas")
    print(f"  - recall@{K}     : {recall_at_k:.3f}  ({int(sum(ans_hits_flags))}/{len(ans_hits_flags) if ans_hits_flags else 0})")
    print(f"  - answer_ok_rate : {answer_ok_rate:.3f}  ({int(sum(ans_answer_ok_flags))}/{len(ans_answer_ok_flags) if ans_answer_ok_flags else 0})")
    print(f"Unanswerable       : {len(unans_rows)} preguntas")
    print(f"  - abstention_rate: {abstention_rate:.3f}  ({int(sum(unans_abstain_flags))}/{len(unans_abstain_flags) if unans_abstain_flags else 0})")

    # 6) CSV salida
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "collection", "type", "question",
        "any_hit", "abstained", "answer_ok",
        "top_score", "top_source", "expect_mode", "expect_sim",
        "substring_ok", "kw_any_ok", "kw_all_ok", "kw_detail",
    ]

    def _write_csv(path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8-sig") as f:  # <- BOM para Excel
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_out)

    try:
        _write_csv(out_csv)
        print(f"\nOK -> {out_csv}")
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = out_csv.with_name(f"{out_csv.stem}_{ts}{out_csv.suffix}")
        _write_csv(alt)
        print(f"\n⚠️ Archivo bloqueado ({out_csv}). Guardé en: {alt}")

# --------------------------
# CLI
# --------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluación RAG: retrieval + calidad de respuesta + abstención.")
    p.add_argument("collection", help="Colección en Qdrant a evaluar.")
    p.add_argument("answerable_json", help="JSON de preguntas respondibles.")
    p.add_argument("unanswerable_json", help="JSON de preguntas no respondibles.")
    p.add_argument("--out", default=None, help="CSV de salida (por defecto: eval_<collection>.csv).")
    p.add_argument("-k", "--top-k", type=int, default=None, help="K para retrieval y generación (default: settings.TOP_K).")
    p.add_argument("-v", "--verbose", action="store_true", help="Detalle por pregunta.")
    p.add_argument("--expect-mode", choices=["substring", "similarity", "both"], default="both",
                   help="Criterio de validación de 'expect' o keywords.")
    p.add_argument("--expect-sim-threshold", type=float, default=0.78,
                   help="Umbral de similitud (0-1) para 'similarity' o 'both'.")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    coll = args.collection
    ans_p = Path(args.answerable_json)
    unans_p = Path(args.unanswerable_json)
    out_p = Path(args.out) if args.out else Path(f"eval_{coll}.csv")

    if not ans_p.exists():
        raise FileNotFoundError(f"No existe el archivo respondibles: {ans_p}")
    if not unans_p.exists():
        raise FileNotFoundError(f"No existe el archivo no respondibles: {unans_p}")

    run_eval(
        collection=coll,
        ans_path=ans_p,
        unans_path=unans_p,
        out_csv=out_p,
        top_k=args.top_k,
        verbose=args.verbose,
        expect_mode=args.expect_mode,
        expect_sim_threshold=args.expect_sim_threshold,
    )
