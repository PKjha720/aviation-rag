"""
Build one index per arm and run retrieval over both question pools.

Both arms go through the UNMODIFIED retrieval stack in rag_engine.py: Chroma
dense search (all-MiniLM-L6-v2, cosine), BM25 Okapi, RRF k=60, cross-encoder
ms-marco-MiniLM-L-6-v2 over the top 30. The only thing that differs between arms
is the chunk list fed to the index builders. rag_engine's path constants are
patched at module level before the engine is instantiated; no retrieval code is
edited.

Seeds pinned; PDF order was already sorted in arms.py so Chroma insertion order
is deterministic.

Reads:  ablation/out/arm_a_chunks.json, arm_b_chunks.json
        ablation/out/fresh_pool_v2.json   (headline pool, n=39, ARM-B-BIASED)
        ablation/out/anchors.json         (old 80, CONTAMINATED - DO NOT REPORT)
Writes: ablation/index_A/, ablation/index_B/   (gitignored)
        ablation/out/retrieval_A.json, retrieval_B.json
"""

import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

import rag_engine
from rag_engine import AviationRAGEngine

OUT = Path("ablation/out")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION = "aviation_docs"
TOP_K = 10
MODES = ["hybrid", "dense", "sparse"]   # "hybrid" = full pipeline (RRF + rerank)


def build_index(arm: str, chunks: list[dict]) -> Path:
    root = Path(f"ablation/index_{arm}")
    vs, bm25_p, meta_p = root / "vectorstore", root / "bm25_index.pkl", root / "chunks_metadata.json"
    if vs.exists() and bm25_p.exists() and meta_p.exists():
        print(f"[{arm}] index exists at {root}, reusing")
        return root
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Dense: mirrors ingest.build_vectorstore exactly.
    client = chromadb.PersistentClient(path=str(vs))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.get_or_create_collection(name=COLLECTION, embedding_function=ef,
                                          metadata={"hnsw:space": "cosine"})
    B = 100
    for i in range(0, len(chunks), B):
        batch = chunks[i:i + B]
        col.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{
                "source_file": c["source_file"], "page_number": c["page_number"],
                "chunk_index": c["chunk_index"], "doc_category": c["doc_category"],
                "doc_section": c["doc_section"], "doc_subject": c["doc_subject"],
                "word_count": c["word_count"], "total_chunks_in_doc": c["total_chunks_in_doc"],
                "is_table": c["is_table"],
            } for c in batch],
        )
        if (i // B) % 20 == 0:
            print(f"[{arm}] embedded {min(i + B, len(chunks))}/{len(chunks)}  ({time.time()-t0:.0f}s)")
    print(f"[{arm}] chroma count = {col.count()}")

    # Sparse: mirrors ingest.build_bm25_index exactly. BM25 row i <-> chunks[i].
    corpus = []
    for c in chunks:
        toks = [t for t in word_tokenize(c["text"].lower()) if t.isalnum() and len(t) > 1]
        corpus.append(toks)
    with open(bm25_p, "wb") as f:
        pickle.dump(BM25Okapi(corpus), f)
    meta_p.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    print(f"[{arm}] index built in {time.time()-t0:.0f}s -> {root}")
    return root


def engine_for(root: Path) -> AviationRAGEngine:
    # Patch the module-level path constants rag_engine reads inside _load_*.
    rag_engine.VECTORSTORE_DIR = root / "vectorstore"
    rag_engine.BM25_INDEX_PATH = root / "bm25_index.pkl"
    rag_engine.CHUNKS_METADATA_PATH = root / "chunks_metadata.json"
    rag_engine.COLLECTION_NAME = COLLECTION
    return AviationRAGEngine()


def run(arm: str, engine: AviationRAGEngine, pools: dict) -> dict:
    out = {}
    for pool_name, qs in pools.items():
        for q in qs:
            qid = q["question_id"]
            rec = {"pool": pool_name, "question": q["question"]}
            for mode in MODES:
                hits = engine.retrieve(q["question"], mode=mode, top_k=TOP_K)
                rec[mode] = [{"chunk_id": h.chunk_id, "text": h.text,
                              "source_file": h.source_file, "page_number": h.page_number,
                              "rerank_score": h.rerank_score} for h in hits]
            out[qid] = rec
        print(f"[{arm}] retrieved {pool_name}: {len(qs)} queries x {len(MODES)} modes")
    return out


def main():
    arms = {}
    for arm, fname in (("A", "arm_a_chunks.json"), ("B", "arm_b_chunks.json")):
        arms[arm] = json.loads((OUT / fname).read_text(encoding="utf-8"))
        print(f"[{arm}] {len(arms[arm])} chunks")

    fresh = json.loads((OUT / "fresh_pool_v2.json").read_text(encoding="utf-8"))
    old80 = json.loads((OUT / "anchors.json").read_text(encoding="utf-8"))
    pools = {"FRESH": fresh, "OLD80_CONTAMINATED": old80}
    print(f"pools: FRESH={len(fresh)}  OLD80_CONTAMINATED={len(old80)}")

    for arm in ("A", "B"):
        root = build_index(arm, arms[arm])
        eng = engine_for(root)
        res = run(arm, eng, pools)
        (OUT / f"retrieval_{arm}.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        print(f"[{arm}] wrote {OUT / f'retrieval_{arm}.json'}")
        del eng
    print("DONE")


if __name__ == "__main__":
    main()
