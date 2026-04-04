"""
Aviation RAG - Document Ingestion Pipeline
==========================================
Processes aviation PDFs → chunks with metadata → ChromaDB vectorstore + BM25 index

Built without LangChain — from scratch for full architectural control.

Author: Prabhat (AAI Junior Executive → MS CS Aspirant)
Domain: Indian Civil Aviation Regulatory Documents
"""

import os
import re
import json
import pickle
import logging
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

import fitz  # PyMuPDF
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest")

# ─── Configuration ──────────────────────────────────────────────────────────
DATA_RAW_DIR = Path("data/raw")
DATA_PROCESSED_DIR = Path("data/processed")
VECTORSTORE_DIR = Path("vectorstore")
BM25_INDEX_PATH = DATA_PROCESSED_DIR / "bm25_index.pkl"
CHUNKS_METADATA_PATH = DATA_PROCESSED_DIR / "chunks_metadata.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 512       # characters per chunk
CHUNK_OVERLAP = 64     # overlap between chunks
COLLECTION_NAME = "aviation_docs"


# ─── Data Structures ───────────────────────────────────────────────────────
@dataclass
class DocumentChunk:
    """Single chunk of text with rich metadata for retrieval."""
    chunk_id: str
    text: str
    source_file: str
    source_path: str
    page_number: int
    chunk_index: int
    total_chunks_in_doc: int = 0
    doc_category: str = ""       # dgca_cars, aai_circulars, icao, notams, aip
    doc_section: str = ""        # e.g., "Section 2", "Air Safety"
    doc_subject: str = ""        # extracted subject/title
    word_count: int = 0
    char_count: int = 0


# ─── Metadata Extraction ──────────────────────────────────────────────────
def extract_metadata_from_filename(filename: str, parent_folder: str) -> dict:
    """
    Extract structured metadata from filename and folder structure.
    
    Examples:
        car_s2_b_p1_minimum_equipment_list.pdf → Section 2, Series B, Part I
        aic_2025_06_ground_handling_services.pdf → AIC 2025, No. 6
        circular_air_safety_2025_01_topic.pdf → Air Safety Circular
    """
    meta = {
        "doc_category": parent_folder,
        "doc_section": "",
        "doc_subject": "",
    }
    
    name = filename.lower().replace(".pdf", "").replace("_", " ")
    
    # CAR documents
    if "car" in name and parent_folder == "dgca_cars":
        section_match = re.search(r"s(\d+)", name)
        if section_match:
            meta["doc_section"] = f"Section {section_match.group(1)}"
        
        # Extract subject from remaining parts
        parts = name.split()
        subject_parts = [p for p in parts if not re.match(r"^(car|s\d|p\d+|[a-z]|rev\d*)$", p)]
        meta["doc_subject"] = " ".join(subject_parts).strip().title()
    
    # AIC documents
    elif "aic" in name or parent_folder == "aai_circulars":
        year_match = re.search(r"(\d{4})", name)
        if year_match:
            meta["doc_section"] = f"Year {year_match.group(1)}"
        
        parts = name.split()
        subject_parts = [p for p in parts if not re.match(r"^(aic|circular|\d+)$", p)]
        meta["doc_subject"] = " ".join(subject_parts).strip().title()
    
    # ICAO documents
    elif parent_folder == "icao":
        meta["doc_section"] = "ICAO"
        doc_match = re.search(r"doc(\d+)", name)
        if doc_match:
            meta["doc_section"] = f"ICAO Doc {doc_match.group(1)}"
        meta["doc_subject"] = name.replace("icao", "").strip().title()
    
    # Fallback
    if not meta["doc_subject"]:
        meta["doc_subject"] = name.title()
    
    return meta


# ─── PDF Processing ────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract text from PDF page by page using PyMuPDF.
    Handles text-based and partially scanned PDFs.
    
    Returns list of {page_number, text} dicts.
    """
    pages = []
    try:
        doc = fitz.open(str(pdf_path))
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            
            # Clean the extracted text
            text = re.sub(r"\n{3,}", "\n\n", text)  # Remove excessive newlines
            text = re.sub(r" {2,}", " ", text)        # Remove excessive spaces
            text = text.strip()
            
            if text and len(text) > 20:  # Skip nearly empty pages
                pages.append({
                    "page_number": page_num + 1,
                    "text": text,
                })
        doc.close()
    except Exception as e:
        log.warning(f"  ⚠ Failed to process {pdf_path.name}: {e}")
    
    return pages


# ─── Chunking ──────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Recursively split text into overlapping chunks.
    Tries to split on paragraph → sentence → word boundaries.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    
    chunks = []
    
    # Try splitting on double newlines (paragraphs) first
    separators = ["\n\n", "\n", ". ", " "]
    
    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            current_chunk = ""
            for part in parts:
                candidate = current_chunk + sep + part if current_chunk else part
                if len(candidate) <= chunk_size:
                    current_chunk = candidate
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    # Start new chunk with overlap from previous
                    if overlap > 0 and current_chunk:
                        overlap_text = current_chunk[-overlap:]
                        current_chunk = overlap_text + sep + part
                    else:
                        current_chunk = part
                    
                    # If single part exceeds chunk_size, force split
                    if len(current_chunk) > chunk_size:
                        while len(current_chunk) > chunk_size:
                            chunks.append(current_chunk[:chunk_size].strip())
                            current_chunk = current_chunk[chunk_size - overlap:]
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            return chunks if chunks else [text[:chunk_size]]
    
    # Fallback: hard split
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk.strip())
    
    return chunks


# ─── Main Ingestion Pipeline ──────────────────────────────────────────────
def generate_chunk_id(source_file: str, page: int, chunk_idx: int) -> str:
    """Generate deterministic unique ID for a chunk."""
    raw = f"{source_file}::page{page}::chunk{chunk_idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def process_all_documents() -> list[DocumentChunk]:
    """
    Process all PDFs in data/raw/ subdirectories.
    Returns list of DocumentChunk objects.
    """
    all_chunks: list[DocumentChunk] = []
    pdf_files = list(DATA_RAW_DIR.rglob("*.pdf"))
    
    if not pdf_files:
        log.error(f"No PDF files found in {DATA_RAW_DIR}/")
        log.error("Make sure your PDFs are in data/raw/dgca_cars/, data/raw/aai_circulars/, etc.")
        return []
    
    log.info(f"Found {len(pdf_files)} PDF files across {DATA_RAW_DIR}/")
    log.info("─" * 60)
    
    for idx, pdf_path in enumerate(pdf_files, 1):
        parent_folder = pdf_path.parent.name  # e.g., "dgca_cars"
        filename = pdf_path.name
        
        log.info(f"[{idx}/{len(pdf_files)}] Processing: {filename}")
        
        # Extract text from PDF
        pages = extract_text_from_pdf(pdf_path)
        if not pages:
            log.warning(f"  ⚠ No text extracted from {filename}")
            continue
        
        # Extract metadata from filename
        meta = extract_metadata_from_filename(filename, parent_folder)
        
        # Chunk each page
        doc_chunks = []
        for page_data in pages:
            page_chunks = chunk_text(page_data["text"])
            for chunk_idx, chunk_text_content in enumerate(page_chunks):
                chunk_id = generate_chunk_id(filename, page_data["page_number"], chunk_idx)
                
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text_content,
                    source_file=filename,
                    source_path=str(pdf_path),
                    page_number=page_data["page_number"],
                    chunk_index=len(doc_chunks),
                    doc_category=meta["doc_category"],
                    doc_section=meta["doc_section"],
                    doc_subject=meta["doc_subject"],
                    word_count=len(chunk_text_content.split()),
                    char_count=len(chunk_text_content),
                )
                doc_chunks.append(chunk)
        
        # Update total chunks count
        for c in doc_chunks:
            c.total_chunks_in_doc = len(doc_chunks)
        
        all_chunks.extend(doc_chunks)
        log.info(f"  ✓ {len(pages)} pages → {len(doc_chunks)} chunks")
    
    log.info("─" * 60)
    log.info(f"Total: {len(all_chunks)} chunks from {len(pdf_files)} documents")
    
    return all_chunks


def build_vectorstore(chunks: list[DocumentChunk]) -> None:
    """
    Build ChromaDB vectorstore from document chunks.
    Uses sentence-transformers for dense embeddings.
    """
    log.info(f"Building ChromaDB vectorstore with {EMBEDDING_MODEL}...")
    
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize ChromaDB with persistent storage
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    
    # Use sentence-transformers embedding function
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    
    # Delete existing collection if it exists
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Batch insert (ChromaDB has a batch limit)
    BATCH_SIZE = 100
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_num in range(total_batches):
        start = batch_num * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(chunks))
        batch = chunks[start:end]
        
        ids = [c.chunk_id for c in batch]
        documents = [c.text for c in batch]
        metadatas = [
            {
                "source_file": c.source_file,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "doc_category": c.doc_category,
                "doc_section": c.doc_section,
                "doc_subject": c.doc_subject,
                "word_count": c.word_count,
                "total_chunks_in_doc": c.total_chunks_in_doc,
            }
            for c in batch
        ]
        
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        
        log.info(f"  Batch {batch_num + 1}/{total_batches} — {len(batch)} chunks embedded & stored")
    
    log.info(f"✓ ChromaDB vectorstore built: {collection.count()} chunks indexed")


def build_bm25_index(chunks: list[DocumentChunk]) -> None:
    """
    Build BM25 sparse search index for hybrid retrieval.
    """
    log.info("Building BM25 sparse search index...")
    
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Tokenize all chunks
    tokenized_corpus = []
    for chunk in chunks:
        tokens = word_tokenize(chunk.text.lower())
        tokens = [t for t in tokens if t.isalnum() and len(t) > 1]
        tokenized_corpus.append(tokens)
    
    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save BM25 index
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    
    # Save chunks metadata for BM25 result mapping
    chunks_data = [asdict(c) for c in chunks]
    with open(CHUNKS_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    
    log.info(f"✓ BM25 index saved: {BM25_INDEX_PATH}")
    log.info(f"✓ Chunks metadata saved: {CHUNKS_METADATA_PATH}")


# ─── Entry Point ───────────────────────────────────────────────────────────
def run_ingestion():
    """Run the complete ingestion pipeline."""
    log.info("=" * 60)
    log.info("  AVIATION RAG — Document Ingestion Pipeline")
    log.info("=" * 60)
    
    # Step 1: Process all PDFs
    chunks = process_all_documents()
    if not chunks:
        log.error("No chunks generated. Check your PDF files.")
        return
    
    # Step 2: Build ChromaDB vectorstore (dense embeddings)
    build_vectorstore(chunks)
    
    # Step 3: Build BM25 index (sparse search)
    build_bm25_index(chunks)
    
    # Summary
    log.info("=" * 60)
    log.info("  INGESTION COMPLETE")
    log.info(f"  Documents processed: {len(set(c.source_file for c in chunks))}")
    log.info(f"  Total chunks: {len(chunks)}")
    log.info(f"  Vectorstore: {VECTORSTORE_DIR}/")
    log.info(f"  BM25 index: {BM25_INDEX_PATH}")
    log.info("=" * 60)


if __name__ == "__main__":
    run_ingestion()
