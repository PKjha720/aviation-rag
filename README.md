# ✈️ AeroRAG — Aviation Regulatory Intelligence System

> Hybrid retrieval-augmented generation for Indian civil aviation regulatory documents — built from scratch, no LangChain, with a full ablation study and the failure cases reported.

[![Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?logo=streamlit)](https://aviation-rag-by-prabhat.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22382810.svg)](https://doi.org/10.5281/zenodo.22382810)

**📄 Paper:** *AeroRAG: A Domain-Adaptive Hybrid Retrieval-Augmented Generation System for Civil Aviation Regulatory Intelligence* — Prabhat Kumar Jha, 2026. Preprint: [`paper/AeroRAG.pdf`](paper/AeroRAG.pdf) · DOI: [10.5281/zenodo.22382810](https://doi.org/10.5281/zenodo.22382810)

---

## 🎯 What This Does

Ask natural-language questions about Indian aviation regulations and get cited answers drawn from official DGCA documents, AICs, circulars, and ICAO standards.

**Example queries:**
- *"What are the requirements for aircraft fuelling procedures?"*
- *"What is the process for registration and de-registration of aircraft in India?"*
- *"What does DGCA mandate regarding flight data recorders?"*
- *"Compare GPWS installation requirements across different aircraft categories"*

---

## 🧩 Why This Domain Is Hard

Four retrieval failure modes are structural to regulatory text, and the architecture is built around them:

| Failure mode | What breaks | How it's addressed |
|---|---|---|
| **Terminology divergence** | Section-specific acronyms (MTOW, MEL, CVR/FDR, GPWS) are out-of-vocabulary for general bi-encoders | BM25 provides exact acronym recall |
| **Numerical precision** | Dense embeddings smooth over numeric tokens; queries need exact thresholds | BM25 treats numerals as high-weight independent terms — *still the weakest category, see Limitations* |
| **Cross-document synthesis** | A CAR must be read alongside the circular that amends it | Citation-enforced prompting across all retrieved context blocks |
| **Structural boundary sensitivity** | Fixed-size chunking severs a regulation identifier from the clause it governs | Separator-priority chunking treats clause boundaries as hard split points |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                │
└──────────────────────┬───────────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌──────────────────┐    ┌──────────────────┐
│   Dense Search   │    │  Sparse Search   │
│   (ChromaDB +    │    │    (BM25         │
│  Sentence-BERT)  │    │    Okapi)        │
│   Top-20         │    │   Top-20         │
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌──────────────────────┐
         │  Reciprocal Rank     │
         │  Fusion (RRF, k=60)  │
         │  Combined ranking    │
         └──────────┬───────────┘
                    ▼
         ┌──────────────────────┐
         │  Cross-Encoder       │
         │  Reranking           │
         │  (ms-marco-MiniLM)   │
         │  Top-5 precision     │
         └──────────┬───────────┘
                    ▼
         ┌──────────────────────┐
         │  Groq LLM            │
         │  (Llama 3.3 70B)     │
         │  + Source Citations  │
         └──────────────────────┘
```

**Latency profile:** dual retrieval ~120 ms · RRF + reranking ~400 ms · generation ~1.8 s · **~2.3 s median end to end**

---

## 🔧 Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| PDF Parsing | PyMuPDF | Fast page-level extraction from the native text layer |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Lightweight, good quality, free |
| Vector Store | ChromaDB | Persistent, no external service needed |
| Sparse Search | BM25 Okapi | Catches exact identifiers and numerals that dense search misses |
| Score Fusion | Reciprocal Rank Fusion | Parameter-free — no annotated data needed to calibrate weights |
| Reranking | Cross-Encoder (ms-marco-MiniLM-L-6-v2) | Precision gate on the final candidates |
| LLM | Groq (Llama 3.3 70B) | 70B-class generation without multi-GPU hardware |
| Frontend | Streamlit | Rapid development, free hosting |

**No LangChain, no LlamaIndex.** Every stage is implemented explicitly so that each one can be ablated independently and audited by someone who doesn't know any framework's internals.

---

## 📊 Results

Four-way ablation on a 100-question benchmark (25 questions per category). **R@5 = Recall@5.**

| Query Category | Dense R@5 | Sparse R@5 | Hybrid R@5 | Full AeroRAG R@5 |
|---|---|---|---|---|
| (A) Direct Regulation | 0.72 | **0.92** | 0.80 | **0.92** |
| (B) Numerical Threshold | 0.60 | 0.56 | **0.72** | 0.64 |
| (C) Procedural | 0.80 | 0.80 | **0.84** | **0.84** |
| (D) Cross-Document | 0.88 | **0.96** | **0.96** | **0.96** |
| **Overall** | 0.75 | 0.81 | 0.83 | **0.84** |
| **MRR@10** | 0.690 | 0.753 | 0.665 | **0.773** |

### What the numbers actually say

**Cross-encoder reranking buys rank precision, not coverage.** Recall@5 moves 0.83 → 0.84 (noise), but MRR@10 moves 0.665 → 0.773. That's the largest gap in the table. It matters because only the top-5 chunks reach the generator — a correct chunk ranked sixth is as useless as one never retrieved.

**⚠️ The cross-encoder actively hurts on numerical queries.** On category B the full pipeline (0.64) *underperforms* fusion alone (0.72). The likely cause: `ms-marco-MiniLM` is trained on natural-language passages and demotes chunks that are dense numerical tables, because they don't resemble its training distribution. This is the most interesting result in the study and it points directly at document-structure-aware chunking — treating tables as atomic units — as the next thing to build.

**BM25 alone matches the full pipeline on regulation lookups (0.92).** Identifiers like *"CAR Section 7, Series F, Part I"* are high-IDF tokens that BM25 weights heavily, while dense embeddings distribute attention across all tokens and dilute the signal. Adding dense candidates to the RRF pool actually *dilutes* the clean lexical signal here (0.80); the cross-encoder then recovers it.

**Statistical caveat:** with 25 queries per category, exact binomial 95% CIs run ±0.12–0.19 (e.g. 0.64 on n=25 → [0.43, 0.82]). Single-category differences of 0.04 are inside sampling noise. The overall figures (n=100) are tighter: 0.84 ±0.07. Read these as component-level trends, not statistically significant differences.

---

## ⚠️ Limitations

Stated plainly, because the point of releasing this is that people can check it.

- **Synthetic evaluation only.** Questions were generated from sampled corpus chunks with Llama 3.1 8B and manually filtered (~15% replacement rate); ground truth is the source chunk. This measures **retrieval recall on synthetic queries**, not end-to-end answer quality on real user questions. Real-world numbers on naturalistic queries would be lower.
- **No generation-quality evaluation.** Faithfulness, hallucination rate, and citation accuracy are not measured. RAGAS-style evaluation plus human review by domain experts is the necessary next step.
- **No external baseline.** Configurations are compared against each other, not against a LangChain pipeline or a commercial search tool.
- **No domain fine-tuning.** Both the embedding model and the reranker are off-the-shelf, trained on general web and MS MARCO data.
- **Fixed chunking parameters.** The 512-character target was chosen empirically. Section-aware dynamic chunking would preserve numbered sub-clauses better.
- **Single-annotator benchmark.** The 100-question set was built by one person. No inter-annotator agreement metrics.
- **No automated faithfulness gate.** Citation-enforced prompting reduces hallucination empirically, but nothing verifies that a generated claim is entailed by the retrieved context. For any safety-consequential deployment, an NLI-based gate between retrieval and generation is required.
- **Modest corpus.** 39 documents. The four failure modes are structural properties of regulatory text that persist as the corpus grows, but scale is not tested here.

---

## 📚 Document Corpus

| Source | Type | Count | Description |
|--------|------|-------|-------------|
| DGCA CARs | Regulatory | 14 | Civil Aviation Requirements — airworthiness, operations, licensing |
| AICs | Advisory | 10 | Aeronautical Information Circulars — tariffs, policies, medical standards |
| DGCA Circulars | Operational | 10 | Air safety, cabin safety, dangerous goods, aircraft engineering |
| ICAO | International | 5 | Safety management, global aviation safety plan |

**Total: 39 documents → 9,445 chunks after ingestion.** The publicly deployed demo indexes additional documents added after the evaluation was frozen; all experiments above use the 39-document corpus.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Setup

```bash
git clone https://github.com/PKjha720/aviation-rag.git
cd aviation-rag

python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Run Ingestion
```bash
# Place aviation PDFs in data/raw/ subdirectories
python ingest.py
```

### Launch App
```bash
streamlit run app.py
```

### Reproduce the Evaluation
```bash
# Generates the 100-question benchmark and runs the 4-way ablation
python evaluate_rag.py
```

Outputs land in `eval_results/` — per-question results, the summary table, and both charts from the paper.

---

## 🔬 Key Design Decisions

### Why hybrid search?
Dense retrieval handles paraphrase and semantic equivalence; BM25 handles exact identifiers, acronyms, and numerals. Regulatory corpora need both, and the ablation shows neither alone wins across all four query categories.

### Why RRF over learned fusion?
There is no large annotated aviation QA dataset to learn interpolation weights from. RRF is parameter-free and robust without calibration data, which is exactly the constraint of a low-resource domain.

### Why cross-encoder reranking?
It contributes rank precision rather than coverage (MRR 0.665 → 0.773). **But see the numerical-threshold result above** — it is not a free win, and an off-the-shelf reranker can demote content it wasn't trained on.

### Why no LangChain?
Explicit implementation of every stage means no abstraction-layer overhead, direct component-level logging, and a system a domain auditor can inspect without learning a framework. For safety-critical applications, auditability is not optional.

### Why Groq?
Running Llama 3.3 70B locally needs multi-GPU hardware. Groq's LPU inference delivers the same model at ~1.8 s generation latency on the free tier.

---

## 📁 Project Structure

```
aviation-rag/
├── app.py                    # Streamlit frontend
├── ingest.py                 # Document processing pipeline
├── rag_engine.py             # Retrieval + generation engine
├── evaluate_rag.py           # Evaluation pipeline (100-Q ablation)
├── requirements.txt
├── .env.example
├── .gitignore
├── paper/
│   └── AeroRAG.pdf           # Preprint
├── .streamlit/
│   └── config.toml
├── data/
│   ├── raw/                  # Source PDFs (not in git)
│   │   ├── dgca_cars/
│   │   ├── aai_circulars/
│   │   ├── dgca_circulars/
│   │   ├── icao/
│   │   └── notams/
│   └── processed/            # BM25 index + metadata
│       ├── bm25_index.pkl
│       └── chunks_metadata.json
├── eval_results/
│   ├── results.csv
│   ├── summary_table.csv
│   ├── recall_chart.png
│   ├── mrr_chart.png
│   └── eval_dataset.json
├── vectorstore/              # ChromaDB persistent storage
└── README.md
```

---

## 📝 Citation

```bibtex
@misc{jha2026aerorag,
  title        = {AeroRAG: A Domain-Adaptive Hybrid Retrieval-Augmented Generation
                  System for Civil Aviation Regulatory Intelligence},
  author       = {Jha, Prabhat Kumar},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.22382810},
  url          = {https://doi.org/10.5281/zenodo.22382810},
  howpublished = {Preprint. Code: \url{https://github.com/PKjha720/aviation-rag}}
}
```

---

## 👤 Author

**Prabhat Kumar Jha** — AI Operations Specialist 2, Bread Financial · previously Junior Executive (Technical), Airports Authority of India

Working at the intersection of domain expertise and retrieval systems. This project was carried out independently of my duties at either organisation.

- 📧 prabhatbit2016@gmail.com
- 🆔 ORCID: [0009-0003-0851-2481](https://orcid.org/0009-0003-0851-2481)
- 🔗 [LinkedIn](https://linkedin.com/in/prabhat-kumar-jha-46a777100/)
- 🌐 [Portfolio](https://pkjha720.github.io/portfolio)
- 💻 [GitHub](https://github.com/PKjha720)

---

## 🙏 Acknowledgements

Thanks to the maintainers of PyMuPDF, sentence-transformers, ChromaDB, rank-bm25, and Groq for making production-quality components freely accessible. No funding was received for this work.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
