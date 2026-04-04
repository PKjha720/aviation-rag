"""
Aviation RAG — Streamlit Interface
====================================
Production-grade aviation regulatory document Q&A system.

Features:
    - Chat interface with conversation memory
    - Hybrid search (Dense + Sparse + Reranking)
    - Source citations with expandable previews
    - Document category filtering
    - System analytics dashboard
"""
import os, subprocess
from pathlib import Path

# Rebuild vectorstore on cloud if needed
if not Path("data/processed/bm25_index.pkl").exists() or not Path("vectorstore/chroma.sqlite3").exists():
    if os.path.exists("/mount/src"):  # Only on Streamlit Cloud
        subprocess.run(["python", "ingest.py"], check=True)

import streamlit as st
import time

from dotenv import load_dotenv

load_dotenv()

# ─── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aviation RAG — Regulatory Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""<style>section[data-testid="stSidebar"]{min-width:300px !important; max-width:300px !important; transform:none !important;}</style>""", unsafe_allow_html=True)
# ─── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global */
    .stApp {
        background-color: #0a0f1a;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 {
        color: #e2e8f0;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.5px;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 0.9rem;
        margin: 0.3rem 0 0 0;
    }
    
    /* Tech badges */
    .tech-stack {
        display: flex;
        gap: 8px;
        justify-content: center;
        flex-wrap: wrap;
        margin-top: 0.8rem;
    }
    .tech-badge {
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: #60a5fa;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 500;
    }

    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .stat-card .stat-value {
        color: #60a5fa;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .stat-card .stat-label {
        color: #94a3b8;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Source cards */
    .source-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    .source-card .source-header {
        color: #60a5fa;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .source-card .source-meta {
        color: #94a3b8;
        font-size: 0.75rem;
    }
    .source-card .source-preview {
        color: #cbd5e1;
        font-size: 0.8rem;
        margin-top: 0.4rem;
        padding: 0.5rem;
        background: #0f172a;
        border-radius: 4px;
        border-left: 3px solid #3b82f6;
    }

    /* Category pills */
    .category-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 500;
        margin-right: 4px;
    }
    .cat-dgca { background: #1e3a2f; color: #4ade80; border: 1px solid #166534; }
    .cat-aic { background: #1e2a3f; color: #60a5fa; border: 1px solid #1e40af; }
    .cat-icao { background: #2a1e3f; color: #a78bfa; border: 1px solid #5b21b6; }
    .cat-notam { background: #3f2a1e; color: #fb923c; border: 1px solid #9a3412; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e3a5f;
    }
    
    /* Chat */
    .stChatMessage {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
    }
    
    /* Remove streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
            
    /* Fix sidebar toggle button visibility */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"],
    .st-emotion-cache-1dp5vir,
    .st-emotion-cache-eczf16 {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        width: 2.5rem !important;
        height: 2.5rem !important;
        color: #ffffff !important;
        background-color: #3b82f6 !important;
        border: none !important;
        border-radius: 50% !important;
        z-index: 9999999 !important;
        position: fixed !important;
        top: 14px !important;
        left: 14px !important;
        cursor: pointer !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.5) !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Initialize Engine ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_engine():
    """Load RAG engine (cached across reruns)."""
    from rag_engine import AviationRAGEngine
    return AviationRAGEngine()


def get_category_pill(category: str) -> str:
    """Generate colored pill HTML for document category."""
    cat_class = {
        "dgca_cars": "cat-dgca",
        "aai_circulars": "cat-aic",
        "icao": "cat-icao",
        "notams": "cat-notam",
    }.get(category, "cat-aic")
    
    cat_label = {
        "dgca_cars": "DGCA CAR",
        "aai_circulars": "AIC/Circular",
        "icao": "ICAO",
        "notams": "NOTAM",
    }.get(category, category.upper())
    
    return f'<span class="category-pill {cat_class}">{cat_label}</span>'


# ─── Sidebar ───────────────────────────────────────────────────────────────
def render_sidebar(engine):
    """Render sidebar with stats, settings, and info."""
    with st.sidebar:
        st.markdown("## ✈️ Aviation RAG")
        st.markdown("---")
        
        # Stats
        stats = engine.get_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{stats['total_documents']}</div>
                <div class="stat-label">Documents</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{stats['total_chunks']}</div>
                <div class="stat-label">Chunks</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Search Settings
        st.markdown("### ⚙️ Search Settings")
        
        search_mode = st.selectbox(
            "Retrieval Mode",
            ["hybrid", "dense", "sparse"],
            index=0,
            help="Hybrid = Dense + Sparse + Reranking (best quality)"
        )
        
        category_options = ["All Categories"] + list(stats.get("chunks_per_category", {}).keys())
        category = st.selectbox(
            "Filter by Category",
            category_options,
            index=0,
        )
        category_filter = None if category == "All Categories" else category
        
        st.markdown("---")
        
        # Architecture info
        st.markdown("### 🏗️ Architecture")
        st.markdown(f"""
        - **Embeddings:** `{stats['embedding_model']}`
        - **Reranker:** Cross-Encoder
        - **LLM:** `{stats['llm_model']}`
        - **Vector DB:** ChromaDB
        - **Sparse:** BM25 (Okapi)
        - **Fusion:** Reciprocal Rank
        """)
        
        st.markdown("---")
        
        # Category breakdown
        st.markdown("### 📊 Document Breakdown")
        for cat, count in stats.get("chunks_per_category", {}).items():
            cat_display = {
                "dgca_cars": "🟢 DGCA CARs",
                "aai_circulars": "🔵 AICs/Circulars",
                "icao": "🟣 ICAO",
                "notams": "🟠 NOTAMs",
                "aip": "⚪ AIP",
            }.get(cat, cat)
            st.markdown(f"{cat_display}: **{count}** chunks")
        
        st.markdown("---")
        
        # Clear chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")
        st.markdown(
            '<p style="color:#64748b;font-size:0.7rem;text-align:center;">'
            'Built by Prabhat<br>AAI Junior Executive<br>'
            'Aviation Domain × AI/ML Engineering</p>',
            unsafe_allow_html=True
        )
    
    return search_mode, category_filter


# ─── Main Chat Interface ──────────────────────────────────────────────────
def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>✈️ Aviation Regulatory Intelligence System</h1>
        <p>AI-powered Q&A over DGCA CARs, AICs, Circulars & ICAO Standards</p>
        <div class="tech-stack">
            <span class="tech-badge">Hybrid Search</span>
            <span class="tech-badge">BM25 + Dense</span>
            <span class="tech-badge">Cross-Encoder Reranking</span>
            <span class="tech-badge">Groq LLM</span>
            <span class="tech-badge">ChromaDB</span>
            <span class="tech-badge">No LangChain</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sources(sources: list[dict]):
    """Render source citations in expandable cards."""
    if not sources:
        return
    
    with st.expander(f"📚 Sources ({len(sources)} documents referenced)", expanded=False):
        for src in sources:
            pill = get_category_pill(src.get("category", ""))
            score_display = f"{src.get('relevance_score', 0):.4f}"
            
            st.markdown(f"""
            <div class="source-card">
                <div class="source-header">
                    [Source {src['source_number']}] {src['file']}
                </div>
                <div class="source-meta">
                    {pill} Page {src['page']} · {src.get('section', '')} · Relevance: {score_display}
                </div>
                <div class="source-preview">
                    {src.get('preview', '')}
                </div>
            </div>
            """, unsafe_allow_html=True)


def main():
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Check for required files
    vectorstore_exists = Path("vectorstore").exists() and any(Path("vectorstore").iterdir())
    bm25_exists = Path("data/processed/bm25_index.pkl").exists()
    
    if not vectorstore_exists or not bm25_exists:
        render_header()
        st.error("⚠️ Vectorstore not found! Run ingestion first:")
        st.code("python ingest.py", language="bash")
        st.info("This will process all PDFs in data/raw/ and build the search indexes.")
        return
    
    # Load engine
    with st.spinner("Loading Aviation RAG Engine..."):
        engine = load_engine()
    
    # Render UI
    search_mode, category_filter = render_sidebar(engine)
    render_header()
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                render_sources(msg["sources"])
    
    # Chat input
    if prompt := st.chat_input("Ask about aviation regulations..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching & analyzing..."):
                start_time = time.time()
                
                response = engine.query(
                    question=prompt,
                    mode=search_mode,
                    category_filter=category_filter,
                    chat_history=st.session_state.chat_history,
                )
                
                elapsed = time.time() - start_time
            
            # Display answer
            st.markdown(response.answer)
            
            # Display metadata
            st.markdown(
                f'<p style="color:#64748b;font-size:0.75rem;margin-top:0.5rem;">'
                f'⚡ {elapsed:.1f}s · {response.retrieval_mode} search · '
                f'{response.num_chunks_used} sources</p>',
                unsafe_allow_html=True
            )
            
            # Display sources
            render_sources(response.sources)
        
        # Save to session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
        })
        
        # Update chat history for context
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": response.answer})
        
        # Keep only last 6 messages in history
        if len(st.session_state.chat_history) > 6:
            st.session_state.chat_history = st.session_state.chat_history[-6:]


if __name__ == "__main__":
    main()
