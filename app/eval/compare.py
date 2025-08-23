# app/eval/compare.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

def _as_bool(x) -> bool:
    return str(x).strip().lower() in ("true", "1", "yes", "y")

def summarize(csv_path: Path) -> Dict[str, float]:
    with csv_path.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    ans = [r for r in rows if (r.get("type") or "").strip().lower() == "answerable"]
    unans = [r for r in rows if (r.get("type") or "").strip().lower() == "unanswerable"]

    def safe_mean(vals: List[bool]) -> float:
        return mean(vals) if vals else 0.0

    recall = safe_mean([_as_bool(r.get("any_hit", "")) for r in ans])
    answer_ok = safe_mean([_as_bool(r.get("answer_ok", "")) for r in ans])
    abstention = safe_mean([_as_bool(r.get("abstained", "")) for r in unans])

    coll = next(( (r.get("collection") or "").strip() for r in rows if (r.get("collection") or "").strip() ), csv_path.stem)

    return {
        "collection": coll,
        "n_answerable": len(ans),
        "n_unanswerable": len(unans),
        "recall_at_k": recall,
        "answer_ok_rate": answer_ok,
        "abstention_rate": abstention,
        "path": str(csv_path),
    }

def print_table(results: List[Dict[str, float]]) -> None:
    headers = ["Collection", "Ans", "Unans", "recall@k", "answer_ok", "abstention", "archivo"]
    rows = [[
        s["collection"],
        s["n_answerable"],
        s["n_unanswerable"],
        f"{s['recall_at_k']:.3f}",
        f"{s['answer_ok_rate']:.3f}",
        f"{s['abstention_rate']:.3f}",
        s["path"],
    ] for s in results]
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    def fmt(cols): return " | ".join(str(c).ljust(w) for c, w in zip(cols, widths))
    print("\n=== RESUMEN VERSUS ===\n")
    print(fmt(headers)); print(fmt(["-"*w for w in widths]))
    for r in rows: print(fmt(r))
    if len(results) >= 2:
        base = results[0]
        print(f"\nΔ vs. '{base['collection']}' (baseline)")
        for s in results[1:]:
            print(f"- {s['collection']}: Δrecall={s['recall_at_k']-base['recall_at_k']:+.3f} | "
                  f"Δanswer_ok={s['answer_ok_rate']-base['answer_ok_rate']:+.3f} | "
                  f"Δabstention={s['abstention_rate']-base['abstention_rate']:+.3f}")
    print("")

def _latest_match(patterns: List[str]) -> Optional[Path]:
    # Busca por patrón en varios lugares habituales
    here = Path(__file__).parent           # app/eval
    roots = [here, here.parent / "eval", Path("eval"), Path(".")]
    candidates: List[Path] = []
    for pat in patterns:
        for base in roots:
            candidates.extend(base.glob(pat))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None

def main():
    p = argparse.ArgumentParser(description="Compara CSVs de evaluación.")
    p.add_argument("files", nargs="*", help="Rutas a CSVs. Si se omiten, autodetecta.")
    args = p.parse_args()

    if args.files:
        files = [Path(x) for x in args.files]
    else:
        rec = _latest_match(["eval_docs_recursive*.csv"])
        sem = _latest_match(["eval_docs_semantic*.csv"])
        if not rec or not sem:
            miss = []
            if not rec: miss.append("eval_docs_recursive*.csv")
            if not sem: miss.append("eval_docs_semantic*.csv")
            raise SystemExit("Faltan CSVs (no encontrados por patrón): " + ", ".join(miss))
        files = [rec, sem]

    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise SystemExit("Faltan CSVs: " + ", ".join(missing))

    results = [summarize(p) for p in files]
    print_table(results)

if __name__ == "__main__":
    main()
