from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, OptimizersConfigDiff, HnswConfigDiff, PayloadSchemaType

from app.ingest.embedders import OpenAIEmbedder
from app.ingest.chunking import build_chunks
from app.settings import settings

def _read_text_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def _iter_files(root: Path) -> List[Path]:
    exts = {".txt", ".md"}
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)

def safe_recreate_collection(client: QdrantClient, name: str, dim: int):
    try:
        client.delete_collection(name)
    except Exception:
        pass
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=32, ef_construct=256, full_scan_threshold=20000),
        optimizers_config=OptimizersConfigDiff(default_segment_number=2, indexing_threshold=0),
    )
    for field, typ in [
        ("source",        PayloadSchemaType.KEYWORD),
        ("doc_id",        PayloadSchemaType.KEYWORD),
        ("doc_title",     PayloadSchemaType.TEXT),
        ("section_title", PayloadSchemaType.TEXT),
        ("section_path",  PayloadSchemaType.KEYWORD),
        ("strategy",      PayloadSchemaType.KEYWORD),
        ("section_level", PayloadSchemaType.INTEGER),
        ("position",      PayloadSchemaType.INTEGER),
    ]:
        try:
            client.create_payload_index(collection_name=name, field_name=field, field_schema=typ)
        except Exception:
            pass

def upsert_chunks(client: QdrantClient, name: str, chunks: List[Dict], embedder: OpenAIEmbedder, batch: int = 64):
    texts = [c["text"] for c in chunks]
    vectors = embedder.embed_documents(texts)
    points = []
    for i, c in enumerate(chunks):
        points.append({
            "id": c["id"],
            "vector": vectors[i],
            "payload": c["metadata"] | {"text": c["text"]},
        })
    client.upsert(collection_name=name, points=points)

def main():
    ap = argparse.ArgumentParser(description="Ingesta a Qdrant con estrategia seleccionable")
    ap.add_argument("--path", required=True, help="Carpeta con .txt/.md (recursivo)")
    ap.add_argument("--collection", default=settings.QDRANT_COLLECTION)
    ap.add_argument("--recreate", type=int, default=1, help="1 = recrea colección (destructivo)")
    ap.add_argument("--strategy", choices=["hier", "flat"], default="hier", help="Chunking: jerárquico (hier) o semántico plano (flat)")
    ap.add_argument("--target-chars", type=int, default=900)
    ap.add_argument("--overlap-chars", type=int, default=150)
    args = ap.parse_args()

    root = Path(args.path)
    assert root.exists(), f"No existe: {root}"

    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=60)
    if args.recreate:
        safe_recreate_collection(client, args.collection, dim=settings.EMBED_DIM)

    embedder = OpenAIEmbedder(model=settings.EMBED_MODEL, api_key=settings.OPENAI_API_KEY)

    all_files = _iter_files(root)
    if not all_files:
        print("No hay archivos .txt/.md en", root)
        return

    total_chunks = 0
    for fp in all_files:
        txt = _read_text_file(fp)
        chunks = build_chunks(
            txt, source_path=str(fp),
            strategy=args.strategy,
            target_chars=args.target_chars,
            overlap_chars=args.overlap_chars,
        )
        dicts = [{"id": c.id, "text": c.text, "metadata": c.metadata} for c in chunks]
        upsert_chunks(client, args.collection, dicts, embedder)
        total_chunks += len(dicts)
        print(f"OK: {fp.name} -> {len(dicts)} chunks (strategy={args.strategy})")

    print(f"Listo. Total chunks: {total_chunks}")

if __name__ == "__main__":
    main()
