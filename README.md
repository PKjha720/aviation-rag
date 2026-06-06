# ✈️ Aviation Regulatory Intelligence System (AeroRAG)

> Production-grade RAG system for Indian civil aviation regulatory documents — built from scratch without LangChain.

[![Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?logo=streamlit)](https://aviation-rag-by-prabhat.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<!-- Uncomment when paper is submitted/accepted:
[![Paper](https://img.shields.io/badge/Paper-NeurIPS_2026_Workshop-blue)](LINK_TO_PAPER)
-->

**📄 Paper:** *AeroRAG: A Domain-Adaptive Hybrid Retrieval-Augmented Generation System for Civil Aviation Regulatory Intelligence* — submitted to NeurIPS 2026 Workshop (link forthcoming).

---

## 🎯 What This Does

This system lets you **ask natural language questions about Indian aviation regulations** and get accurate, cited answers from official DGCA documents, AICs, circulars, and ICAO standards.

**Example queries:**
- *"What are the requirements for aircraft fuelling procedures?"*
- *"What is the process for registration and de-registration of aircraft in India?"*
- *"What does DGCA mandate regarding flight data recorders?"*
- *"Compare GPWS installation requirements across different aircraft categories"*

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
         │  Fusion (RRF)        │
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
         │  + Source Citations   │
         └──────────────────────┘
```

---

## 🔧 Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| PDF Parsing | PyMuPDF | Fast, handles scanned + text PDFs |
| Embeddings | sentence-transformers (MiniLM-L6-v2) | Lightweight, good quality, free |
| Vector Store | ChromaDB | Persistent, no external service needed |
| Sparse Search | BM25 (Okapi) | Catches exact keyword matches that dense search misses |
| Score Fusion | Reciprocal Rank Fusion | Principled way to combine dense + sparse rankings |
| Reranking | Cross-Encoder (ms-marco-MiniLM) | Precision boost on final candidates |
| LLM | Groq (Llama 3.3 70B) | Free, blazing fast inference |
| Frontend | Streamlit | Professional UI, rapid development |
| Deployment | Streamlit Cloud | Free hosting, GitHub integration |

**No LangChain.** Every component is built from scratch for full architectural control and understanding.

---

## 📊 Results (from paper)

| Query Category | Dense R@5 | Sparse R@5 | Hybrid R@5 | Full AeroRAG R@5 |
|---|---|---|---|---|
| (A) Direct Regulation | 0.72 | **0.92** | 0.80 | **0.92** |
| (B) Numerical Threshold | 0.60 | 0.56 | **0.72** | 0.64 |
| (C) Procedural | 0.80 | 0.80 | **0.84** | **0.84** |
| (D) Cross-Document | 0.88 | **0.96** | **0.96** | **0.96** |
| **Overall** | 0.75 | 0.81 | 0.83 | **0.84** |
| **MRR@10** | 0.690 | 0.753 | 0.665 | **0.773** |

Full details and ablation analysis in the paper.

---

## 📊 Document Corpus

| Source | Type | Count | Description |
|--------|------|-------|-------------|
| DGCA CARs | Regulatory | 14 | Civil Aviation Requirements across airworthiness, operations, licensing |
| AICs | Advisory | 10 | Aeronautical Information Circulars — tariffs, policies, medical standards |
| DGCA Circulars | Operational | 10 | Air safety, cabin safety, dangerous goods, aircraft engineering |
| ICAO | International | 5 | Safety management, global aviation safety plan |

**Total: 39 documents → 9,445 chunks after ingestion**

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Setup

```bash
# Clone the repo
git clone https://github.com/PKjha720/aviation-rag.git
cd aviation-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Run Ingestion
```bash
# Place your aviation PDFs in data/raw/ subdirectories
python ingest.py
```

### Launch App
```bash
streamlit run app.py
```

### Run Evaluation
```bash
# Generates 100-question benchmark and runs 4-way ablation
python evaluate_rag.py
```

---

## 🔬 Key Design Decisions

### Why Hybrid Search?
Dense search (vector similarity) excels at semantic understanding but can miss exact terminology. BM25 catches specific regulation numbers, aircraft codes, and technical terms. Combining both via RRF gives the best of both worlds.

### Why Cross-Encoder Reranking?
Bi-encoder embeddings are fast but approximate. Cross-encoder processes query-document pairs jointly, providing much higher precision for the final top-5 results. Our ablation shows this contributes primarily to rank precision (MRR@10) rather than raw coverage (Recall@5).

### Why No LangChain?
Building from scratch makes each component's role explicit and auditable. For safety-critical aviation applications, auditability is not optional.

### Why Aviation Domain?
Aviation regulations are an ideal RAG challenge: dense technical language, cross-referenced documents, precise terminology, and real-world safety implications where accuracy is non-negotiable.

---

## 📁 Project Structure

```
aviation-rag/
├── app.py                    # Streamlit frontend
├── ingest.py                 # Document processing pipeline
├── rag_engine.py             # Retrieval + generation engine
├── evaluate_rag.py           # Evaluation pipeline (100-Q ablation)
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
├── .gitignore
├── .streamlit/
│   └── config.toml           # UI theme configuration
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
├── eval_results/             # Evaluation outputs
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

If you use AeroRAG in your research, please cite:

```bibtex
@inproceedings{jha2026aerorag,
  title={AeroRAG: A Domain-Adaptive Hybrid Retrieval-Augmented Generation
         System for Civil Aviation Regulatory Intelligence},
  author={Jha, Prabhat Kumar},
  booktitle={NeurIPS 2026 Workshop},
  year={2026},
  url={https://github.com/PKjha720/aviation-rag}
}
```

---

## 👤 Author

**Prabhat Kumar Jha** — Junior Executive (Technical), Airports Authority of India

Building at the intersection of aviation domain expertise and AI/ML engineering.

- 📧 prabhatbit2016@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/prabhat-kumar-jha-46a777100/)
- 🌐 [Portfolio](https://pkjha720.github.io/portfolio)
- 💻 [GitHub](https://github.com/PKjha720)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
