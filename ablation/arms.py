"""
Arm construction for the structure-ablation experiment.

Arm A — the current pipeline, unchanged. Extraction and chunking are IMPORTED
        from ingest.py rather than reimplemented, so this arm cannot drift from
        what main actually runs.

Arm B — flat extraction (page.get_text("text")), no table detection, no markdown
        rendering, fixed token window over each page.

DESIGN NOTE — page boundaries in Arm B.
    Arm B keeps page boundaries (it windows within each page, not across the
    whole document). The original brief said Arm B should carry "no structural
    information at all", which read literally means concatenating the document
    and windowing across page breaks. The scope cut narrowed the experiment to
    "table structure preservation vs flattening, nothing more", so page
    boundaries are held CONSTANT across both arms and table handling is the only
    manipulated variable. Switching to document-level windowing is a one-line
    change in chunk_arm_b (see PAGE_SCOPED).

Neither builder touches ChromaDB or BM25. They return chunk lists only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import fitz
except ImportError:
    import pymupdf as fitz

from transformers import AutoTokenizer

# Arm A extraction + chunking, taken straight from the live pipeline.
from ingest import (
    extract_text_from_pdf as _arm_a_extract,
    chunk_text as _arm_a_chunk_prose,
    chunk_table as _arm_a_chunk_table,
    extract_metadata_from_filename,
    generate_chunk_id,
)

DATA_RAW_DIR = Path("data/raw")

# Arm B uses the tokenizer of the model that will embed the chunks, so "token
# window" means tokens as the embedder counts them.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_TOK = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

PAGE_SCOPED = True  # see DESIGN NOTE above


@dataclass
class ArmChunk:
    """Chunk record shared by both arms. Mirrors ingest.DocumentChunk fields
    that downstream indexing and evaluation actually read."""
    chunk_id: str
    text: str
    source_file: str
    source_path: str
    page_number: int
    chunk_index: int
    total_chunks_in_doc: int = 0
    doc_category: str = ""
    doc_section: str = ""
    doc_subject: str = ""
    word_count: int = 0
    char_count: int = 0
    is_table: bool = False


def list_pdfs() -> list[Path]:
    """Sorted so insertion order into Chroma/BM25 is deterministic.

    Sorting does not change chunk_ids: generate_chunk_id hashes
    filename::page::within_page_index, none of which depend on file order.
    """
    return sorted(DATA_RAW_DIR.rglob("*.pdf"), key=lambda p: str(p).replace("\\", "/"))


# ─── Arm A ──────────────────────────────────────────────────────────────────
def chunk_arm_a(pdf_path: Path) -> list[ArmChunk]:
    """Exactly the chunking in ingest.process_all_documents, for one PDF."""
    filename = pdf_path.name
    meta = extract_metadata_from_filename(filename, pdf_path.parent.name)
    pages = _arm_a_extract(pdf_path)

    out: list[ArmChunk] = []
    for page_data in pages:
        page_chunks: list[tuple[str, bool]] = []
        for block in page_data["blocks"]:
            if block["is_table"]:
                page_chunks += [(t, True) for t in _arm_a_chunk_table(block["text"])]
            else:
                page_chunks += [(t, False) for t in _arm_a_chunk_prose(block["text"])]
        for chunk_idx, (text, is_table) in enumerate(page_chunks):
            out.append(ArmChunk(
                chunk_id=generate_chunk_id(filename, page_data["page_number"], chunk_idx),
                text=text,
                source_file=filename,
                source_path=str(pdf_path),
                page_number=page_data["page_number"],
                chunk_index=len(out),
                doc_category=meta["doc_category"],
                doc_section=meta["doc_section"],
                doc_subject=meta["doc_subject"],
                word_count=len(text.split()),
                char_count=len(text),
                is_table=is_table,
            ))
    for c in out:
        c.total_chunks_in_doc = len(out)
    return out


# ─── Arm B ──────────────────────────────────────────────────────────────────
def extract_flat(pdf_path: Path) -> list[dict]:
    """Flat extraction: whole page as plain text, no table detection.

    Identical to ingest_original.extract_text_from_pdf — the pre-v2 path.
    """
    pages = []
    try:
        doc = fitz.open(str(pdf_path))
        for page_num in range(len(doc)):
            text = doc[page_num].get_text("text")
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r" {2,}", " ", text).strip()
            if text and len(text) > 20:
                pages.append({"page_number": page_num + 1, "text": text})
        doc.close()
    except Exception as e:
        print(f"  ! flat extraction failed on {pdf_path.name}: {e}")
    return pages


def window_by_tokens(text: str, window: int, overlap: int) -> list[str]:
    """Fixed token window over `text`, returning slices of the ORIGINAL string.

    Uses offset mapping rather than tokenizer.decode() because this tokenizer is
    uncased — decoding would lowercase every Arm B chunk and introduce a second
    difference between the arms on top of the one being measured.
    """
    enc = _TOK(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    if not offsets:
        return []
    if len(offsets) <= window:
        return [text.strip()] if text.strip() else []

    stride = window - overlap
    if stride <= 0:
        raise ValueError(f"overlap {overlap} must be < window {window}")

    out = []
    for start in range(0, len(offsets), stride):
        sl = offsets[start:start + window]
        if not sl:
            break
        piece = text[sl[0][0]:sl[-1][1]].strip()
        if piece:
            out.append(piece)
        if start + window >= len(offsets):
            break
    return out


def chunk_arm_b(pdf_path: Path, window: int, overlap: int) -> list[ArmChunk]:
    """Flat text, fixed token window. No table detection, no markdown, no
    is_table flag ever set."""
    filename = pdf_path.name
    meta = extract_metadata_from_filename(filename, pdf_path.parent.name)
    pages = extract_flat(pdf_path)

    out: list[ArmChunk] = []
    if PAGE_SCOPED:
        units = [(p["page_number"], p["text"]) for p in pages]
    else:
        units = [(pages[0]["page_number"] if pages else 1,
                  "\n".join(p["text"] for p in pages))]

    for page_number, text in units:
        for chunk_idx, piece in enumerate(window_by_tokens(text, window, overlap)):
            out.append(ArmChunk(
                chunk_id=generate_chunk_id(filename, page_number, chunk_idx),
                text=piece,
                source_file=filename,
                source_path=str(pdf_path),
                page_number=page_number,
                chunk_index=len(out),
                doc_category=meta["doc_category"],
                doc_section=meta["doc_section"],
                doc_subject=meta["doc_subject"],
                word_count=len(piece.split()),
                char_count=len(piece),
                is_table=False,
            ))
    for c in out:
        c.total_chunks_in_doc = len(out)
    return out


def build_arm(arm: str, window: int = 0, overlap: int = 0) -> list[ArmChunk]:
    pdfs = list_pdfs()
    out: list[ArmChunk] = []
    for p in pdfs:
        out.extend(chunk_arm_a(p) if arm == "A" else chunk_arm_b(p, window, overlap))
    return out


def token_len(text: str) -> int:
    return len(_TOK(text, add_special_tokens=False)["input_ids"])


def as_dicts(chunks: list[ArmChunk]) -> list[dict]:
    return [asdict(c) for c in chunks]
