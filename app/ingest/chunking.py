from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple
import re, time, hashlib, unicodedata, uuid
from pathlib import Path

# ==== Heurísticas de secciones (para 'hier') ==================================
_HEADING_PATTERNS = [
    r"^#{1,6}\s+.+$",
    r"^[A-ZÁÉÍÓÚÑ0-9][A-ZÁÉÍÓÚÑ0-9\s\.\-_/]{4,}$",
    r"^\d+(\.\d+){0,3}\s+.+$",
]
_HEADING_RE = re.compile("|".join(f"(?:{p})" for p in _HEADING_PATTERNS))

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def _slug(s: str, max_len: int = 80) -> str:
    s2 = _strip_accents(s).lower()
    s2 = re.sub(r"[^a-z0-9]+", "-", s2).strip("-")
    return s2[:max_len] or "sec"

@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict

# ---- util de tamaño ----
def _chunk_paragraphs(par_lines: List[str], target_chars: int, overlap_chars: int) -> List[str]:
    paras = "\n".join(par_lines).split("\n\n")
    chunks: List[str] = []
    cur = ""
    def push():
        nonlocal cur
        t = cur.strip()
        if t: chunks.append(t)
        cur = ""
    for p in paras:
        p2 = p.strip()
        if not p2: continue
        if len(cur) + len(p2) + 2 <= target_chars:
            cur = (cur + "\n\n" + p2) if cur else p2
        else:
            if cur:
                tail = cur[-overlap_chars:] if len(cur) > overlap_chars else cur
                push()
                cur = tail + "\n\n" + p2
            else:
                for i in range(0, len(p2), target_chars):
                    chunks.append(p2[i:i+target_chars])
                cur = ""
    push()
    return chunks

# ---- split secciones (hier) ----
def _split_into_sections(text: str) -> List[Tuple[str, List[str]]]:
    lines = text.splitlines()
    blocks: List[Tuple[str, List[str]]] = []
    current_heading = ""
    buf: List[str] = []
    def flush():
        nonlocal buf, current_heading
        if buf:
            blocks.append((current_heading, buf))
            buf = []
    for ln in lines:
        if _HEADING_RE.match(ln.strip()):
            flush()
            current_heading = ln.strip()
        else:
            buf.append(ln)
    flush()
    return blocks or [("", lines)]

def _heading_level(h: str) -> int:
    h = (h or "").strip()
    if h.startswith("#"):
        return min(6, len(h) - len(h.lstrip("#")))
    m = re.match(r"^(\d+)(\.\d+){0,3}\s+", h)
    if m:  # 1 / 1.2 / 1.2.3 / 1.2.3.4
        return h.split()[0].count(".") + 1
    return 1 if h else 0

# ---- builder: HIER ----
def build_hier_chunks(
    doc_text: str, source_path: str, target_chars: int = 900, overlap_chars: int = 150
) -> List[Chunk]:
    ts = int(time.time())
    src = str(Path(source_path).as_posix())
    doc_title = Path(source_path).stem
    doc_id = _slug(doc_title) or hashlib.md5(src.encode()).hexdigest()[:10]

    sections = _split_into_sections(doc_text)
    results: List[Chunk] = []
    global_idx = 0

    for (heading, lines) in sections:
        section_title = heading.strip("# ").strip() if heading else doc_title
        section_path = [doc_title] + ([section_title] if heading else [])
        level = _heading_level(heading)
        for piece in _chunk_paragraphs(lines, target_chars, overlap_chars):
            piece_strip = piece.strip()
            if not piece_strip: continue
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{src}|{global_idx}|{ts}"))
            meta = {
                "source": src,
                "doc_id": doc_id,
                "doc_title": doc_title,
                "section_title": section_title,
                "section_path": section_path,
                "section_level": level,
                "position": global_idx,
                "n_chars": len(piece_strip),
                "ts_ingested": ts,
                "strategy": "hier",
                "chunk_size": target_chars,
                "chunk_overlap": overlap_chars,
            }
            results.append(Chunk(id=chunk_id, text=piece_strip, metadata=meta))
            global_idx += 1
    return results

# ---- builder: FLAT (semántico puro por tamaño, ignora headings) ----
def build_flat_chunks(
    doc_text: str, source_path: str, target_chars: int = 700, overlap_chars: int = 120
) -> List[Chunk]:
    ts = int(time.time())
    src = str(Path(source_path).as_posix())
    doc_title = Path(source_path).stem
    doc_id = _slug(doc_title) or hashlib.md5(src.encode()).hexdigest()[:10]

    results: List[Chunk] = []
    global_idx = 0
    lines = doc_text.splitlines()
    for piece in _chunk_paragraphs(lines, target_chars, overlap_chars):
        piece_strip = piece.strip()
        if not piece_strip: continue
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{src}|{global_idx}|{ts}"))
        meta = {
            "source": src,
            "doc_id": doc_id,
            "doc_title": doc_title,
            "section_title": doc_title,
            "section_path": [doc_title],
            "section_level": 1,
            "position": global_idx,
            "n_chars": len(piece_strip),
            "ts_ingested": ts,
            "strategy": "flat",
            "chunk_size": target_chars,
            "chunk_overlap": overlap_chars,
        }
        results.append(Chunk(id=chunk_id, text=piece_strip, metadata=meta))
        global_idx += 1
    return results

# ---- selector común ----
def build_chunks(
    doc_text: str, source_path: str, strategy: str, target_chars: int, overlap_chars: int
) -> List[Chunk]:
    strategy = (strategy or "hier").lower()
    if strategy == "flat":
        return build_flat_chunks(doc_text, source_path, target_chars, overlap_chars)
    return build_hier_chunks(doc_text, source_path, target_chars, overlap_chars)
