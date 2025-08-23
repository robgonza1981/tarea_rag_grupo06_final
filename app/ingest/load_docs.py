# app/ingest/load_docs.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional
from pypdf import PdfReader
from docx import Document
import mimetypes
import uuid

def _doc_id(path: Path) -> str:
    return f"doc-{uuid.uuid5(uuid.NAMESPACE_URL, str(path.resolve()))}"

def _mime(path: Path) -> Optional[str]:
    return mimetypes.guess_type(str(path))[0]

def load_txt(path: Path) -> List[Dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [{
        "source": str(path),
        "doc_id": _doc_id(path),
        "doc_type": _mime(path) or "text/plain",
        "page": None,
        "text": text,
    }]

def load_pdf(path: Path) -> List[Dict]:
    reader = PdfReader(str(path))
    pages: List[Dict] = []
    for i, p in enumerate(reader.pages):
        try:
            txt = p.extract_text() or ""
        except Exception:
            txt = ""
        pages.append({
            "source": f"{path}#page={i+1}",
            "doc_id": _doc_id(path),
            "doc_type": _mime(path) or "application/pdf",
            "page": i + 1,
            "text": txt,
        })
    return pages

def load_docx(path: Path) -> List[Dict]:
    doc = Document(str(path))
    text = "\n".join(p.text for p in doc.paragraphs)
    return [{
        "source": str(path),
        "doc_id": _doc_id(path),
        "doc_type": _mime(path) or "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "page": None,
        "text": text,
    }]

ALLOWED = {".txt", ".md", ".pdf", ".docx"}

def load_sources(paths: List[Path]) -> List[Dict]:
    docs: List[Dict] = []
    for p in paths:
        if not p.exists():
            continue
        suf = p.suffix.lower()
        if suf in [".txt", ".md"]:
            docs.extend(load_txt(p))
        elif suf == ".pdf":
            docs.extend(load_pdf(p))
        elif suf == ".docx":
            docs.extend(load_docx(p))
        else:
            pass
    return docs
