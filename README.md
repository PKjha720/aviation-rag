# ✈️ Aviation Regulatory Intelligence System

> Production-grade RAG system for Indian civil aviation regulatory documents — built from scratch without LangChain.

[![Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?logo=streamlit)](https://aviation-rag-by-prabhat.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

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

## 📊 Document Corpus

| Source | Type | Count | Description |
|--------|------|-------|-------------|
| DGCA CARs | Regulatory | ~15 | Civil Aviation Requirements across airworthiness, operations, licensing |
| AICs | Advisory | ~13 | Aeronautical Information Circulars — tariffs, policies, medical standards |
| DGCA Circulars | Operational | ~12 | Air safety, cabin safety, dangerous goods, aircraft engineering |
| ICAO | International | ~5-8 | Safety management, global aviation safety plan |

**Total: ~300MB of aviation regulatory documents**

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/aviation-rag.git
cd aviation-rag

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env       # Windows
# cp .env.example .env       # Mac/Linux
# Edit .env and add your GROQ_API_KEY
```

### Run Ingestion
```bash
# Place your aviation PDFs in data/raw/ subdirectories
# Then run:
python ingest.py
```

### Launch App
```bash
streamlit run app.py
```

---

## 🔬 Key Design Decisions

### Why Hybrid Search?
Dense search (vector similarity) excels at semantic understanding but can miss exact terminology. BM25 catches specific regulation numbers, aircraft codes, and technical terms. Combining both via RRF gives the best of both worlds.

### Why Cross-Encoder Reranking?
Bi-encoder embeddings are fast but approximate. Cross-encoder processes query-document pairs jointly, providing much higher precision for the final top-5 results. This is the same architecture used by Google Search.

### Why No LangChain?
Building from scratch demonstrates understanding of each component's role. It also eliminates the abstraction overhead, making the system faster, more debuggable, and more impressive to technical reviewers.

### Why Aviation Domain?
Aviation regulations are the perfect RAG challenge: dense technical language, cross-referenced documents, precise terminology, and real-world safety implications where accuracy is non-negotiable.

---

## 📁 Project Structure

```
aviation-rag/
├── app.py                    # Streamlit frontend
├── ingest.py                 # Document processing pipeline
├── rag_engine.py             # Retrieval + generation engine
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
├── .gitignore
├── .streamlit/
│   └── config.toml           # UI theme configuration
├── data/
│   ├── raw/                  # Source PDFs (not in git)
│   │   ├── dgca_cars/
│   │   ├── aai_circulars/
│   │   ├── icao/
│   │   └── notams/
│   └── processed/            # BM25 index + metadata
│       ├── bm25_index.pkl
│       └── chunks_metadata.json
├── vectorstore/              # ChromaDB persistent storage
└── README.md
```

---

## 👤 Author

**Prabhat** — Junior Executive (Technical), Airports Authority of India

Building at the intersection of aviation domain expertise and AI/ML engineering. Currently preparing for MS CS (Fall 2027) with focus on ML Systems and Information Retrieval.

- Domain: Indian Civil Aviation (DGCA, AAI)
- Focus: RAG Systems, ML Infrastructure, NLP
- GRE: 339/340 (170Q + 169V)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
