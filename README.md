# ✈️ AeroRAG — Aviation Regulatory Intelligence

Hybrid retrieval-augmented generation over Indian civil aviation regulation — DGCA Civil Aviation Requirements, Aeronautical Information Circulars, operational circulars, and ICAO standards. Built without LangChain or LlamaIndex, so every stage can be ablated independently.

[![Live Demo](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?logo=streamlit)](https://aviation-rag-by-prabhat.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22382810.svg)](https://doi.org/10.5281/zenodo.22382810)

**Paper:** *AeroRAG: A Domain-Adaptive Hybrid Retrieval-Augmented Generation System for Civil Aviation Regulatory Intelligence* — [10.5281/zenodo.22382810](https://doi.org/10.5281/zenodo.22382810)

---

## TL;DR

Ask a regulatory question, get a cited answer. But the interesting part of this repo isn't the system — it's the measurement.

**Tables are where regulation keeps its binding values, and retrieval handles them badly.** Queries whose answer lives in a table retrieve at **0.675** Recall@5 against **0.925** for prose (p = 0.010). Dense retrieval alone drops to **0.500** (p = 0.0006). Neither score fusion nor cross-encoder reranking closes the gap.

**And the standard PDF extraction path destroys tables before retrieval ever runs.** `page.get_text("text")` flattens a table into newline-separated lines with no column boundaries — a tariff row and a paragraph become the same object. This corpus holds 368 genuine tables; without structure-aware ingestion, none of them exist in the index *as tables*.

**None of this shows up in a normal benchmark.** If you build eval questions by sampling indexed chunks — the standard cheap approach — you can only ask about content that survived ingestion. Evidence destroyed at ingestion is never a candidate, never a gold chunk, never a miss. It leaves no trace in any metric.

---

## What it does

Natural-language questions over official Indian aviation regulatory documents, answered with source and page citations.

```
"What is the calibration schedule for a barometer?"
"What are the requirements for aircraft fuelling procedures?"
"What does DGCA mandate regarding flight data recorders?"
```

Try it: **[aviation-rag-by-prabhat.streamlit.app](https://aviation-rag-by-prabhat.streamlit.app/)**

---

## Architecture

```
INGESTION (offline)                    QUERY (online)
─────────────────────                  ─────────────────────
43 PDFs                                user query
   ↓                                      ↓         ↓
PyMuPDF                              ChromaDB    BM25 Okapi
 ├─ find_tables()  → markdown         top-20      top-20
 └─ get_text()     → prose               ↓         ↓
   ↓                                  Reciprocal Rank Fusion (k=60)
chunking                                    ↓
 ├─ tables: by row, header repeated   cross-encoder rerank → top-5
 └─ prose:  512 chars, 64 overlap           ↓
   ↓                                  citation-enforced generation
ChromaDB + BM25 index
```

Every component is explicit. No framework abstraction sits between the code and the retrieval logic, which is what made the ingestion defect findable at all.

| Stage | Choice | Rationale |
|---|---|---|
| PDF parsing | PyMuPDF `find_tables()` + `get_text("blocks")` | Tables survive as markdown; their page regions are excluded from the prose pass to prevent double-indexing |
| Embeddings | `all-MiniLM-L6-v2` | Lightweight and free — **and the weakest link on tabular content** |
| Vector store | ChromaDB | Persistent, no external service |
| Sparse search | BM25 Okapi | Recovers exact identifiers, acronyms and numerals dense search smooths over |
| Fusion | Reciprocal Rank Fusion (k=60) | Parameter-free; no annotated aviation QA data exists to learn weights from |
| Reranking | `ms-marco-MiniLM-L-6-v2` cross-encoder | Precision gate on the final candidate set |
| Generation | Groq `openai/gpt-oss-120b` | Citation-enforced prompting |
| Frontend | Streamlit | Free hosting |

> **Reproducibility note.** Hosted model availability shifts. This system was originally built against `llama-3.3-70b-versatile`, which Groq has since retired; that configuration can no longer be run. Models listed here were verified available in September 2026. Retrieval metrics involve no LLM and are unaffected by generation-model changes.

---

## Results

### Recall@5 by evidence type

80 questions, stratified: 40 whose gold evidence is a table, 40 whose gold evidence is prose. Two-sided Fisher exact test.

| Retrieval mode | Table R@5 | Prose R@5 | Gap | p |
|---|---|---|---|---|
| Dense only | **0.500** | 0.875 | −0.375 | **0.0006** |
| Sparse (BM25) | 0.675 | 0.925 | −0.250 | **0.0103** |
| Hybrid (RRF) | 0.625 | 0.900 | −0.275 | **0.0075** |
| Full pipeline | 0.675 | 0.925 | −0.250 | **0.0103** |

Every configuration is significantly worse on table evidence. Dense retrieval is worst: `all-MiniLM-L6-v2` finds half of table answers against seven-eighths of prose answers. BM25 partially compensates, which fits — tables are dense in exact numerals and identifiers that lexical matching weights heavily and a bi-encoder distributes across all tokens.

A bi-encoder has to compress a linearised table into one fixed-length vector, and the numbers that make the table useful are exactly the tokens such models represent least distinctively.

### Does reranking help?

| Slice | Hybrid R@5 | Full R@5 | Δ | p |
|---|---|---|---|---|
| Table | 0.625 | 0.675 | +0.050 | 0.81 |
| Prose | 0.900 | 0.925 | +0.025 | 1.00 |

Not measurably, on either slice. MRR@10 rises from 0.655 to 0.725, so the cross-encoder is improving **rank precision, not coverage** — it moves the right chunk up the list rather than into it. That still matters, because only the top 5 reach the generator.

**The bottleneck is retrieval, not reranking.**

### By query category

20 questions per category. Underpowered — read as trends, not results.

| Query Category | Dense | Sparse | Hybrid | Full |
|---|---|---|---|---|
| (A) Direct Regulation | 0.65 | 0.75 | 0.75 | 0.75 |
| (B) Numerical Threshold | 0.65 | 0.75 | **0.80** | 0.70 |
| (C) Procedural | 0.70 | **0.90** | 0.75 | **0.90** |
| (D) Cross-Document | 0.75 | 0.80 | 0.75 | **0.85** |
| **Overall** | 0.69 | **0.80** | 0.76 | **0.80** |
| **MRR@10** | 0.66 | **0.73** | 0.655 | 0.725 |

---

## What ingestion loses before retrieval runs

Three distinct forms of evidence loss, all occurring before any retrieval component executes, none visible in a recall metric.

| Failure | Scale | Status |
|---|---|---|
| Tables flattened into prose by `get_text("text")` | 368 tables corpus-wide | **Fixed** — extracted via `find_tables()`, emitted as markdown, flagged `is_table` |
| Pages that are scanned images with no text layer | 419 / 1,969 pages (21.3%) | **Open** — needs OCR |
| Documents contributing zero chunks | 4 of 43 PDFs (30 pages) | **Open** — fully scanned |

The four absent documents are `aic_2024_02_open_sky_policy_cargo`, `aic_2025_15_ophthalmological_disorders`, `aic_2025_16_hypertension_civil_aircrew`, and `aic_2026_01_aic_01_of_2026`. Two more — the Varanasi and Port Blair airport tariff schedules, 317 pages between them — are over 80% scanned, despite being the most table-dense sources in the corpus. The system has never been able to answer a question about either.

**Table detection needs a validity filter.** `find_tables()` fires readily on numbered lists and indented prose. Of 786 initial detections, 57.4% had fewer than two columns populated in a majority of rows — typically one populated column flanked by empty ones, as an enumerated list produces. Requiring ≥2 columns populated in >50% of rows and ≥3 rows cuts 786 candidates to 368 genuine tables. Rejected regions fall back to the prose path rather than being discarded.

**Tables are chunked by row, never mid-row, with the header repeated on every piece.** A row stripped of its header keeps its numbers but loses what they measure: split a calibration schedule badly and "Once in a year" is no longer attached to "Barometer".

---

## Limitations

Read these before citing any number above.

- **Question style is confounded with evidence type.** This is the most serious one. Table questions were generated by a prompt demanding a specific value lookup; prose questions by a general prompt. Part of the 25-point gap may reflect value-lookup questions being intrinsically harder, rather than table evidence being harder to retrieve. This design cannot separate the two. Settling it needs a human-written benchmark with question styles matched across both slices.
- **Synthetic evaluation only.** Questions are LLM-generated from sampled chunks; ground truth is the source chunk, which favours retrievers matching surface wording. Real-world figures on naturalistic queries would be lower.
- **No generation-quality evaluation.** Faithfulness, hallucination rate and citation accuracy are unmeasured — the LLM-judge stage is disabled by default to stay within API quota.
- **n = 40 per slice.** The table-vs-prose gaps are significant. The per-category figures (n = 20) are not.
- **21% of the corpus is unreadable.** Until OCR is added, any claim about coverage of Indian aviation regulation is a claim about the text-layer subset.
- **No external baseline.** Configurations are compared against each other, not against a LangChain pipeline or a commercial search tool.
- **No domain fine-tuning.** Embedding model and reranker are both off-the-shelf.
- **Single-annotator benchmark.** No inter-annotator agreement metrics.

---

## Corpus

| Source | Type | PDFs | Content |
|---|---|---|---|
| DGCA CARs | Regulatory | 15 | Airworthiness, operations, licensing |
| AICs & circulars | Advisory / operational | 23 | Tariffs, policies, medical standards, safety |
| ICAO | International standards | 5 | Safety management, global safety plan |

**43 PDFs, 1,969 pages → 39 documents with extractable text → 10,572 chunks, of which 368 (3.5%) are table-derived.**

All source PDFs are committed to `data/raw/`, so the corpus is reproducible without re-downloading anything.

---

## Quick Start

```bash
git clone https://github.com/PKjha720/aviation-rag.git
cd aviation-rag

python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
pip install pandas                # needed by evaluate_rag.py

echo "GROQ_API_KEY=your_key_here" > .env
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

**Build the indices** (no API key needed — parsing, embeddings and BM25 only):

```bash
python ingest.py
```

Expect `table-derived chunks: 368 (3.5%)` at the end. First run downloads `all-MiniLM-L6-v2` (~90 MB).

**Run the app:**

```bash
streamlit run app.py
```

**Reproduce the evaluation:**

```bash
python evaluate_rag.py
```

Generates the stratified 80-question benchmark, runs the four-way ablation, and prints the table-vs-prose comparison. Set `SKIP_ANSWER_JUDGING = False` to re-enable LLM answer scoring — it costs substantially more API quota.

---

## Repository layout

```
ingest.py           PDF → tables + prose → chunks → ChromaDB + BM25
rag_engine.py       retrieval, RRF fusion, reranking, generation
app.py              Streamlit interface
evaluate_rag.py     stratified benchmark construction + four-way ablation
data/raw/           43 source PDFs
eval_results/       benchmark, per-query results, charts
```

`data/processed/` and `vectorstore/` are gitignored — regenerate them with `ingest.py`.

---

## Next steps

1. **OCR the 419 scanned pages.** Largest single coverage gain available.
2. **Human-written benchmark with matched question styles**, to remove the confound above.
3. **Structure-aware table encoding.** The dense-retrieval gap (0.500 vs 0.875) is the sharpest signal here and points at linearisation schemes that preserve header–value binding, or encoders trained with table structure as an explicit signal.
4. **NLI-based faithfulness gate** between retrieval and generation.

---

## License

MIT — see [LICENSE](LICENSE).

## Citation

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
