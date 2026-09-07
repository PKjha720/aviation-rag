# ✈️ AeroRAG — Aviation Regulatory Intelligence System

> Hybrid retrieval-augmented generation for Indian civil aviation regulatory documents — built from scratch, no LangChain, with a full ablation study and the failure cases reported.

[![Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?logo=streamlit)](https://aviation-rag-by-prabhat.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22382810.svg)](https://doi.org/10.5281/zenodo.22382810)

**📄 Paper:** *AeroRAG: A Domain-Adaptive Hybrid Retrieval-Augmented Generation System for Civil Aviation Regulatory Intelligence* — Prabhat Kumar Jha, 2026. DOI: [10.5281/zenodo.22382810](https://doi.org/10.5281/zenodo.22382810)

---

## ⚠️ Correction notice (v2)

**Version 1 reported that cross-encoder reranking degrades retrieval on numerical-threshold
queries, and attributed this to the reranker demoting tabular content. That claim does not
survive re-testing and is withdrawn.**

Two errors produced it:

1. **The v1 ingestion pipeline destroyed table structure.** `page.get_text("text")` flattens
   tables into newline-separated lines indistinguishable from prose. No table ever reached
   the reranker *as a table*, so the reranker cannot have been demoting tables.
2. **The effect was inside sampling noise.** The reported 0.72 → 0.64 drop was a difference
   of two questions out of 25.

After rebuilding ingestion to preserve tables and re-running with stratified sampling
(n=40 per slice), the reranker shows **no** significant effect on table queries
(+0.050, p=0.81). The real finding is different, larger, and reported below.

---

## 🎯 What This Does

Ask natural-language questions about Indian aviation regulations and get cited answers drawn
from official DGCA documents, AICs, circulars, and ICAO standards.

**Example queries:**
- *"What are the requirements for aircraft fuelling procedures?"*
- *"What is the calibration schedule for a barometer?"*
- *"What does DGCA mandate regarding flight data recorders?"*

---

## 📊 Results

### Headline: table-borne evidence is retrieved far worse than prose

80 questions, stratified: 40 whose gold evidence is a table, 40 whose gold evidence is prose.
**R@5 = Recall@5.** Two-sided Fisher exact test.

| Retrieval mode | Table R@5 | Prose R@5 | Gap | p |
|---|---|---|---|---|
| Dense only | **0.500** | 0.875 | −0.375 | **0.0006** |
| Sparse (BM25) | 0.675 | 0.925 | −0.250 | **0.0103** |
| Hybrid (RRF) | 0.625 | 0.900 | −0.275 | **0.0075** |
| Full (hybrid + rerank) | 0.675 | 0.925 | −0.250 | **0.0103** |

Every configuration is significantly worse on table evidence. **Dense retrieval is worst** —
`all-MiniLM-L6-v2` finds half of table answers versus seven-eighths of prose answers. BM25
partially compensates, consistent with tables being dense in exact numerals and identifiers
that lexical matching weights heavily and a bi-encoder smooths over.

### Reranker effect (v1's claim, retested)

| Slice | Hybrid R@5 | Full R@5 | Δ | p |
|---|---|---|---|---|
| Table | 0.625 | 0.675 | +0.050 | 0.81 |
| Prose | 0.900 | 0.925 | +0.025 | 1.00 |

No effect on either slice. **The bottleneck is retrieval, not reranking.**

### Four-way ablation by query category

20 questions per category, n=80 total.

| Query Category | Dense R@5 | Sparse R@5 | Hybrid R@5 | Full R@5 |
|---|---|---|---|---|
| (A) Direct Regulation | 0.65 | 0.75 | 0.75 | 0.75 |
| (B) Numerical Threshold | 0.65 | 0.75 | **0.80** | 0.70 |
| (C) Procedural | 0.70 | **0.90** | 0.75 | **0.90** |
| (D) Cross-Document | 0.75 | 0.80 | 0.75 | **0.85** |
| **Overall** | 0.69 | **0.80** | 0.76 | **0.80** |
| **MRR@10** | 0.66 | **0.73** | 0.655 | 0.725 |

---

## 🧱 What ingestion loses before retrieval runs

Diagnosing the v1 error surfaced three distinct forms of evidence loss occurring *before* any
retrieval component executes. None is visible in a Recall@5 number, because evaluation
questions are generated from chunks that made it into the index.

| Failure | Scale | Status |
|---|---|---|
| Tables flattened into prose by `get_text("text")` | 368 tables corpus-wide | **Fixed** — extracted via `find_tables()`, emitted as markdown, flagged `is_table` |
| Pages that are scanned images with no text layer | 419 / 1,969 pages (21.3%) | **Open** — needs OCR |
| Documents contributing zero chunks | 4 of 43 PDFs (30 pages) | **Open** — fully scanned |

The four absent documents are `aic_2024_02_open_sky_policy_cargo`,
`aic_2025_15_ophthalmological_disorders`, `aic_2025_16_hypertension_civil_aircrew`, and
`aic_2026_01_aic_01_of_2026`. Two further documents — the Varanasi and Port Blair airport
tariff schedules, 317 pages combined — are over 80% scanned and contribute almost nothing
despite being the most table-dense sources in the corpus.

**Table detection needs a validity filter.** `find_tables()` fires on numbered lists and
indented prose: 57.4% of its initial detections had fewer than two populated columns.
Requiring ≥2 columns populated in >50% of rows, and ≥3 rows, cut 786 candidate table chunks
to 368 genuine ones. Rejected regions fall back to the prose path rather than being dropped.

---

## 🔧 Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| PDF Parsing | PyMuPDF (`find_tables` + `get_text("blocks")`) | Table structure preserved; table regions excluded from the prose pass |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Lightweight, free — **and the weakest link on tables** |
| Vector Store | ChromaDB | Persistent, no external service |
| Sparse Search | BM25 Okapi | Catches exact identifiers and numerals dense search misses |
| Score Fusion | Reciprocal Rank Fusion | Parameter-free — no annotated data needed |
| Reranking | Cross-Encoder (ms-marco-MiniLM-L-6-v2) | Precision gate on final candidates |
| LLM | Groq (`openai/gpt-oss-120b`) | See reproducibility note below |
| Frontend | Streamlit | Rapid development, free hosting |

**No LangChain, no LlamaIndex.** Every stage is implemented explicitly so each can be ablated
independently.

> **Reproducibility note.** v1 used `llama-3.3-70b-versatile` for generation and
> `llama-3.1-8b-instant` for benchmark construction. Groq retired both; as of September 2026
> neither is available. v2 uses `openai/gpt-oss-120b` and `qwen/qwen3.8-27b`. Retrieval
> metrics are unaffected — Recall@5 and MRR involve no LLM.

---

## ⚠️ Limitations

- **Question style is confounded with evidence type.** Table questions were generated by a
  prompt demanding a specific value lookup; prose questions by a general prompt. Part of the
  25-point gap may reflect value-lookup questions being intrinsically harder rather than table
  evidence being harder to retrieve. This design cannot separate the two. A human-written
  benchmark with matched question styles across both slices is required to settle it.
- **Synthetic evaluation only.** Questions are LLM-generated from sampled chunks; ground truth
  is the source chunk. This measures retrieval recall on synthetic queries, not end-to-end
  answer quality. Real-world numbers on naturalistic queries would be lower.
- **No generation-quality evaluation in v2.** Answer faithfulness and citation accuracy are
  not measured — the LLM-judge step was disabled to stay within API quota.
- **n=40 per slice.** Table-vs-prose gaps are significant; per-category figures (n=20) are
  not. Read those as trends only.
- **No external baseline.** Configurations are compared against each other, not against
  LangChain or a commercial search tool.
- **No domain fine-tuning.** Embedding model and reranker are both off-the-shelf.
- **Single-annotator benchmark.** No inter-annotator agreement metrics.
- **21% of the corpus is unreadable.** Until OCR is added, any claim about coverage of Indian
  aviation regulation is a claim about the text-layer subset only.

---

## 📚 Document Corpus

| Source | Type | PDFs | Notes |
|--------|------|------|-------|
| DGCA CARs | Regulatory | 15 | Airworthiness, operations, licensing |
| AICs & Circulars | Advisory / Operational | 23 | Tariffs, policies, medical standards, safety |
| ICAO | International | 5 | Safety management, global safety plan |

**43 PDFs → 39 contribute text → 10,572 chunks, of which 368 (3.5%) are table-derived.**
Four PDFs yield nothing; see the ingestion section above.

---

## 🚀 Quick Start

```bash
git clone https://github.com/PKjha720/aviation-rag.git
cd aviation-rag

python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
pip install pandas            # needed by evaluate_rag.py

echo "GROQ_API_KEY=your_key_here" > .env
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Run ingestion
```bash
python ingest.py
```
Expect `table-derived chunks: 368 (3.5%)` at the end.

### Launch app
```bash
streamlit run app.py
```

### Reproduce the evaluation
```bash
python evaluate_rag.py
```
Generates the stratified 80-question benchmark and runs the four-way ablation. Outputs land
in `eval_results/` — including `results_v2.csv` and the table-vs-prose summary. Set
`SKIP_ANSWER_JUDGING = False` to re-enable LLM answer scoring (costs substantially more API
quota).

---

## 🔬 Key Design Decisions

### Why hybrid search?
Dense retrieval handles paraphrase; BM25 handles exact identifiers, acronyms and numerals.
Regulatory corpora need both, and the ablation shows neither alone wins across all query
categories.

### Why RRF over learned fusion?
No large annotated aviation QA dataset exists to learn interpolation weights from. RRF is
parameter-free and robust without calibration data.

### Why treat tables as atomic units?
A table row is meaningless without its header. Splitting a table mid-way produces chunks where
the numbers survive but the thing they measure does not. `chunk_table()` splits by row and
repeats the header on every piece.

---

## 🔭 Next Steps

1. **OCR the 419 scanned pages.** Largest single coverage gain available.
2. **Human-written benchmark with matched question styles** across table and prose slices, to
   remove the confound above.
3. **Table-aware embedding.** The dense gap (0.500 vs 0.875) is the sharpest signal in the
   study and points at linearisation or structure-aware encoding of table content.
4. **NLI-based faithfulness gate** between retrieval and generation.

---

## 📄 License

MIT — see [LICENSE](LICENSE).

## ✍️ Citation

```bibtex
@misc{jha2026aerorag,
  title  = {AeroRAG: A Domain-Adaptive Hybrid Retrieval-Augmented Generation
            System for Civil Aviation Regulatory Intelligence},
  author = {Jha, Prabhat Kumar},
  year   = {2026},
  doi    = {10.5281/zenodo.22382810},
  url    = {https://doi.org/10.5281/zenodo.22382810}
}
```
