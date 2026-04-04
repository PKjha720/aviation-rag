"""
Aviation RAG - Retrieval & Generation Engine
=============================================
Hybrid search (Dense + BM25) → Cross-encoder Reranking → Groq LLM Generation

Architecture:
    Query → [Dense Search (ChromaDB)] ──┐
                                         ├─ Reciprocal Rank Fusion → Cross-Encoder Rerank → Top-K
    Query → [Sparse Search (BM25)]   ───┘
    
    Top-K Chunks + Query → Groq LLM → Answer with Citations

Author: Prabhat (AAI Junior Executive → MS CS Aspirant)
"""

import os
import json
import pickle
import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from groq import Groq
from dotenv import load_dotenv
import nltk
from nltk.tokenize import word_tokenize

load_dotenv()
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

log = logging.getLogger("rag_engine")

# ─── Configuration ──────────────────────────────────────────────────────────
VECTORSTORE_DIR = Path("vectorstore")
DATA_PROCESSED_DIR = Path("data/processed")
BM25_INDEX_PATH = DATA_PROCESSED_DIR / "bm25_index.pkl"
CHUNKS_METADATA_PATH = DATA_PROCESSED_DIR / "chunks_metadata.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "aviation_docs"

GROQ_MODEL = "llama-3.3-70b-versatile"

# Retrieval params
DENSE_TOP_K = 20          # candidates from vector search
SPARSE_TOP_K = 20         # candidates from BM25
RERANK_TOP_K = 5          # final chunks after reranking
RRF_K = 60                # RRF constant


# ─── Data Structures ───────────────────────────────────────────────────────
@dataclass
class RetrievedChunk:
    """A chunk retrieved with its relevance scores."""
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    doc_category: str
    doc_section: str
    doc_subject: str
    dense_rank: int = 0
    sparse_rank: int = 0
    rrf_score: float = 0.0
    rerank_score: float = 0.0


@dataclass
class RAGResponse:
    """Complete RAG response with answer and sources."""
    answer: str
    sources: list[dict]
    query: str
    num_chunks_used: int
    retrieval_mode: str


# ─── Retrieval Engine ──────────────────────────────────────────────────────
class AviationRAGEngine:
    """
    Production-grade RAG engine for aviation regulatory documents.
    
    Features:
        - Hybrid search: Dense (ChromaDB) + Sparse (BM25)
        - Reciprocal Rank Fusion for score combination
        - Cross-encoder reranking for precision
        - Groq LLM generation with source citations
        - Category-aware filtering
    """
    
    def __init__(self):
        log.info("Initializing Aviation RAG Engine...")
        self._load_vectorstore()
        self._load_bm25()
        self._load_reranker()
        self._init_llm()
        log.info("✓ RAG Engine ready")
    
    def _load_vectorstore(self):
        """Load ChromaDB collection with embedding function."""
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self.chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        self.collection = self.chroma_client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
        )
        log.info(f"  ✓ ChromaDB loaded: {self.collection.count()} chunks")
    
    def _load_bm25(self):
        """Load BM25 index and chunks metadata."""
        with open(BM25_INDEX_PATH, "rb") as f:
            self.bm25: BM25Okapi = pickle.load(f)
        
        with open(CHUNKS_METADATA_PATH, "r", encoding="utf-8") as f:
            self.chunks_metadata: list[dict] = json.load(f)
        
        log.info(f"  ✓ BM25 index loaded: {len(self.chunks_metadata)} chunks")
    
    def _load_reranker(self):
        """Load cross-encoder model for reranking."""
        self.reranker = CrossEncoder(RERANKER_MODEL)
        log.info(f"  ✓ Reranker loaded: {RERANKER_MODEL}")
    
    def _init_llm(self):
        """Initialize Groq client."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            log.warning("  ⚠ GROQ_API_KEY not set — generation will fail")
            self.groq_client = None
        else:
            self.groq_client = Groq(api_key=api_key)
            log.info(f"  ✓ Groq LLM ready: {GROQ_MODEL}")
    
    # ── Dense Search ───────────────────────────────────────────────────────
    def _dense_search(self, query: str, top_k: int = DENSE_TOP_K,
                      category_filter: str = None) -> list[dict]:
        """Vector similarity search using ChromaDB."""
        where_filter = None
        if category_filter:
            where_filter = {"doc_category": category_filter}
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        
        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "rank": i + 1,
            })
        return hits
    
    # ── Sparse Search ──────────────────────────────────────────────────────
    def _sparse_search(self, query: str, top_k: int = SPARSE_TOP_K,
                       category_filter: str = None) -> list[dict]:
        """BM25 keyword search."""
        tokens = word_tokenize(query.lower())
        tokens = [t for t in tokens if t.isalnum() and len(t) > 1]
        
        scores = self.bm25.get_scores(tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1]
        
        hits = []
        rank = 1
        for idx in top_indices:
            if rank > top_k:
                break
            if scores[idx] <= 0:
                break
            
            chunk_meta = self.chunks_metadata[idx]
            
            # Apply category filter if specified
            if category_filter and chunk_meta.get("doc_category") != category_filter:
                continue
            
            hits.append({
                "chunk_id": chunk_meta["chunk_id"],
                "text": chunk_meta["text"],
                "metadata": {
                    "source_file": chunk_meta["source_file"],
                    "page_number": chunk_meta["page_number"],
                    "doc_category": chunk_meta["doc_category"],
                    "doc_section": chunk_meta["doc_section"],
                    "doc_subject": chunk_meta["doc_subject"],
                },
                "bm25_score": float(scores[idx]),
                "rank": rank,
            })
            rank += 1
        
        return hits
    
    # ── Hybrid Search with RRF ─────────────────────────────────────────────
    def _reciprocal_rank_fusion(self, dense_hits: list[dict],
                                 sparse_hits: list[dict],
                                 k: int = RRF_K) -> list[RetrievedChunk]:
        """
        Combine dense and sparse search results using Reciprocal Rank Fusion.
        RRF score = sum(1 / (k + rank)) across all result lists.
        """
        chunk_scores: dict[str, dict] = {}
        
        # Score dense results
        for hit in dense_hits:
            cid = hit["chunk_id"]
            if cid not in chunk_scores:
                chunk_scores[cid] = {
                    "chunk_id": cid,
                    "text": hit["text"],
                    "metadata": hit["metadata"],
                    "rrf_score": 0.0,
                    "dense_rank": 0,
                    "sparse_rank": 0,
                }
            chunk_scores[cid]["rrf_score"] += 1.0 / (k + hit["rank"])
            chunk_scores[cid]["dense_rank"] = hit["rank"]
        
        # Score sparse results
        for hit in sparse_hits:
            cid = hit["chunk_id"]
            if cid not in chunk_scores:
                chunk_scores[cid] = {
                    "chunk_id": cid,
                    "text": hit["text"],
                    "metadata": hit["metadata"],
                    "rrf_score": 0.0,
                    "dense_rank": 0,
                    "sparse_rank": 0,
                }
            chunk_scores[cid]["rrf_score"] += 1.0 / (k + hit["rank"])
            chunk_scores[cid]["sparse_rank"] = hit["rank"]
        
        # Sort by RRF score
        sorted_chunks = sorted(chunk_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
        
        # Convert to RetrievedChunk objects
        results = []
        for item in sorted_chunks:
            meta = item["metadata"]
            results.append(RetrievedChunk(
                chunk_id=item["chunk_id"],
                text=item["text"],
                source_file=meta.get("source_file", ""),
                page_number=meta.get("page_number", 0),
                doc_category=meta.get("doc_category", ""),
                doc_section=meta.get("doc_section", ""),
                doc_subject=meta.get("doc_subject", ""),
                dense_rank=item["dense_rank"],
                sparse_rank=item["sparse_rank"],
                rrf_score=item["rrf_score"],
            ))
        
        return results
    
    # ── Cross-Encoder Reranking ────────────────────────────────────────────
    def _rerank(self, query: str, chunks: list[RetrievedChunk],
                top_k: int = RERANK_TOP_K) -> list[RetrievedChunk]:
        """Rerank candidate chunks using cross-encoder for precision."""
        if not chunks:
            return []
        
        # Prepare query-document pairs
        pairs = [(query, chunk.text) for chunk in chunks]
        
        # Score with cross-encoder
        scores = self.reranker.predict(pairs)
        
        # Attach scores and sort
        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)
        
        reranked = sorted(chunks, key=lambda x: x.rerank_score, reverse=True)
        return reranked[:top_k]
    
    # ── Complete Retrieval Pipeline ────────────────────────────────────────
    def retrieve(self, query: str, mode: str = "hybrid",
                 category_filter: str = None,
                 top_k: int = RERANK_TOP_K) -> list[RetrievedChunk]:
        """
        Full retrieval pipeline.
        
        Modes:
            - "hybrid": Dense + Sparse + RRF + Reranking (best quality)
            - "dense": Vector search + Reranking
            - "sparse": BM25 keyword search + Reranking
        """
        if mode == "hybrid":
            dense_hits = self._dense_search(query, category_filter=category_filter)
            sparse_hits = self._sparse_search(query, category_filter=category_filter)
            candidates = self._reciprocal_rank_fusion(dense_hits, sparse_hits)
        elif mode == "dense":
            dense_hits = self._dense_search(query, category_filter=category_filter)
            candidates = [
                RetrievedChunk(
                    chunk_id=h["chunk_id"], text=h["text"],
                    source_file=h["metadata"].get("source_file", ""),
                    page_number=h["metadata"].get("page_number", 0),
                    doc_category=h["metadata"].get("doc_category", ""),
                    doc_section=h["metadata"].get("doc_section", ""),
                    doc_subject=h["metadata"].get("doc_subject", ""),
                    dense_rank=h["rank"],
                )
                for h in dense_hits
            ]
        elif mode == "sparse":
            sparse_hits = self._sparse_search(query, category_filter=category_filter)
            candidates = [
                RetrievedChunk(
                    chunk_id=h["chunk_id"], text=h["text"],
                    source_file=h["metadata"].get("source_file", ""),
                    page_number=h["metadata"].get("page_number", 0),
                    doc_category=h["metadata"].get("doc_category", ""),
                    doc_section=h["metadata"].get("doc_section", ""),
                    doc_subject=h["metadata"].get("doc_subject", ""),
                    sparse_rank=h["rank"],
                )
                for h in sparse_hits
            ]
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        
        # Rerank top candidates
        reranked = self._rerank(query, candidates[:30], top_k=top_k)
        return reranked
    
    # ── LLM Generation ─────────────────────────────────────────────────────
    def generate(self, query: str, chunks: list[RetrievedChunk],
                 chat_history: list[dict] = None) -> RAGResponse:
        """
        Generate answer using Groq LLM with retrieved context and citations.
        """
        if not self.groq_client:
            return RAGResponse(
                answer="⚠️ GROQ_API_KEY not configured. Please set it in .env file.",
                sources=[], query=query, num_chunks_used=0, retrieval_mode=""
            )
        
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}] Document: {chunk.source_file} | "
                f"Page: {chunk.page_number} | "
                f"Category: {chunk.doc_category} | "
                f"Section: {chunk.doc_section}\n"
                f"{chunk.text}"
            )
        context = "\n\n---\n\n".join(context_parts)
        
        # System prompt
        system_prompt = """You are an expert Aviation Regulatory Assistant specializing in Indian civil aviation regulations (DGCA CARs, AICs, Circulars) and ICAO standards.

Your role:
1. Answer questions accurately based ONLY on the provided source documents
2. Always cite your sources using [Source N] notation
3. If the answer spans multiple sources, cite all relevant ones
4. If the provided sources don't contain enough information, say so clearly
5. Use precise aviation terminology
6. Structure complex answers with clear headings when appropriate
7. When discussing regulations, mention the specific CAR/AIC/Circular number

Important: Never fabricate information. If unsure, indicate the limitation."""

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add chat history if available
        if chat_history:
            for msg in chat_history[-6:]:  # Last 3 exchanges
                messages.append(msg)
        
        # User message with context
        user_message = f"""Based on the following aviation regulatory documents, answer the question.

RETRIEVED DOCUMENTS:
{context}

QUESTION: {query}

Provide a comprehensive answer with [Source N] citations. If the sources don't fully cover the question, indicate what additional information might be needed."""

        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=2048,
                top_p=0.9,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"⚠️ Generation error: {str(e)}"
        
        # Build sources list
        sources = [
            {
                "source_number": i + 1,
                "file": chunk.source_file,
                "page": chunk.page_number,
                "category": chunk.doc_category,
                "section": chunk.doc_section,
                "subject": chunk.doc_subject,
                "relevance_score": round(chunk.rerank_score, 4),
                "preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
            }
            for i, chunk in enumerate(chunks)
        ]
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            query=query,
            num_chunks_used=len(chunks),
            retrieval_mode="hybrid",
        )
    
    # ── Full Pipeline ──────────────────────────────────────────────────────
    def query(self, question: str, mode: str = "hybrid",
              category_filter: str = None,
              chat_history: list[dict] = None) -> RAGResponse:
        """
        End-to-end RAG pipeline: Retrieve → Rerank → Generate.
        """
        # Retrieve
        chunks = self.retrieve(question, mode=mode, category_filter=category_filter)
        
        if not chunks:
            return RAGResponse(
                answer="No relevant documents found for your query. Try rephrasing or removing category filters.",
                sources=[], query=question, num_chunks_used=0, retrieval_mode=mode,
            )
        
        # Generate
        response = self.generate(question, chunks, chat_history=chat_history)
        response.retrieval_mode = mode
        
        return response
    
    # ── Stats ──────────────────────────────────────────────────────────────
    def get_stats(self) -> dict:
        """Get vectorstore and index statistics."""
        # Count documents per category
        categories = {}
        for chunk in self.chunks_metadata:
            cat = chunk.get("doc_category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        
        unique_docs = set(c["source_file"] for c in self.chunks_metadata)
        
        return {
            "total_chunks": self.collection.count(),
            "total_documents": len(unique_docs),
            "chunks_per_category": categories,
            "embedding_model": EMBEDDING_MODEL,
            "reranker_model": RERANKER_MODEL,
            "llm_model": GROQ_MODEL,
        }
