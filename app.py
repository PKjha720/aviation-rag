"""
Aviation RAG — Streamlit interface.

Hybrid retrieval (BM25 Okapi + dense bi-encoder, reciprocal rank fusion,
cross-encoder reranking) over Indian civil aviation regulatory documents.
"""

import os
import sys
import logging
from pathlib import Path

# ─── Auto-ingestion on cloud ──────────────────────────────────────────────
INDEX_PATHS = (Path("data/processed/bm25_index.pkl"), Path("vectorstore"))


def indexes_present() -> bool:
    """True when both the BM25 index and a non-empty vectorstore exist."""
    bm25, store = INDEX_PATHS
    return bm25.exists() and store.is_dir() and any(store.iterdir())


if not indexes_present() and os.path.exists("/mount/src"):  # Streamlit Cloud
    logging.info("Indexes absent on cloud instance, running ingestion")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ingest import run_ingestion

    run_ingestion()

# ─── Main App ─────────────────────────────────────────────────────────────
import html
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AeroRAG — Aviation Regulatory Intelligence",
    page_icon="✈",
    layout="wide",
    # "auto" keeps the sidebar open on desktop but collapsed on narrow screens,
    # where "expanded" would cover the conversation on load.
    initial_sidebar_state="auto",
)

# ─── Design tokens ─────────────────────────────────────────────────────────
# One palette, one type scale, one spacing scale. Every value below is drawn
# from these; nothing is hand-tuned at the call site.
STYLES = """
<style>
:root {
  --bg:            #0b0f16;
  --surface:       #121926;
  --surface-sunk:  #0d131d;
  --border:        #212c3d;
  --border-bright: #2d3b52;

  --text:          #dbe3ef;
  --text-muted:    #8a99af;
  /* Was #5c6a7e, which measured 3.20:1 on --surface: below the 4.5:1 AA floor
     for the 11-13px labels this token carries. Lifted in place, hue and
     saturation preserved, to 4.62:1 against the lightest surface it sits on. */
  --text-faint:    #74849a;

  /* The accent is the only vivid hue in the interface and belongs to chrome.
     Category colours below are data, so they are desaturated and kept off the
     accent's hue: blue must not mean "emphasis" and "AIC" at the same time. */
  --accent:        #4c8dff;

  --c-dgca:  #4cac5a;
  --c-aic:   #4e99b5;
  --c-icao:  #9d84c7;
  --c-notam: #b88f44;

  --mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;

  --fs-xs: 11px;  --fs-sm: 12px;  --fs-md: 13px;
  --fs-base: 14px; --fs-xl: 21px;

  --sp-1: 4px; --sp-2: 8px; --sp-3: 12px;
  --sp-4: 16px; --sp-5: 24px;

  --radius: 6px;
}

.stApp { background: var(--bg); }

/* Numerals in data positions must not jitter between reruns. */
.metric-value, .src-score, .statusline, .kv-value {
  font-family: var(--mono);
  font-variant-numeric: tabular-nums;
}

/* ── Sidebar ───────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: var(--surface-sunk);
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] > div { padding-top: var(--sp-4); }

.side-title {
  font-size: var(--fs-md);
  font-weight: 600;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: var(--text);
  margin: 0 0 var(--sp-1);
}
.side-sub {
  font-size: var(--fs-xs);
  color: var(--text-faint);
  margin: 0 0 var(--sp-4);
}
.side-label {
  font-size: var(--fs-xs);
  font-weight: 600;
  letter-spacing: .09em;
  text-transform: uppercase;
  color: var(--text-faint);
  margin: var(--sp-5) 0 var(--sp-2);
}

/* ── Metrics ───────────────────────────────────────────────────────────── */
.metrics { display: flex; gap: var(--sp-2); }
.metric {
  flex: 1;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: var(--sp-3);
}
.metric-value {
  display: block;
  font-size: var(--fs-xl);
  font-weight: 600;
  color: var(--text);
  line-height: 1.1;
}
.metric-label {
  display: block;
  font-size: var(--fs-xs);
  color: var(--text-faint);
  letter-spacing: .06em;
  text-transform: uppercase;
  margin-top: var(--sp-1);
}

/* ── Key/value rows (architecture) ─────────────────────────────────────── */
.kv { display: flex; justify-content: space-between; gap: var(--sp-3); padding: 5px 0; }
.kv + .kv { border-top: 1px solid var(--border); }
.kv-key { font-size: var(--fs-sm); color: var(--text-faint); white-space: nowrap; }
.kv-value {
  font-size: var(--fs-sm);
  color: var(--text-muted);
  text-align: right;
  word-break: break-all;
}

/* ── Index composition bars ────────────────────────────────────────────── */
.comp-row { margin-bottom: var(--sp-3); }
.comp-head {
  display: flex; justify-content: space-between;
  font-size: var(--fs-sm); margin-bottom: 5px;
}
.comp-name { color: var(--text-muted); }
.comp-count { font-family: var(--mono); font-variant-numeric: tabular-nums; color: var(--text-faint); }
.comp-track { height: 3px; background: var(--border); border-radius: 2px; overflow: hidden; }
.comp-fill { height: 100%; border-radius: 2px; }

/* ── Header ────────────────────────────────────────────────────────────── */
.app-head {
  border-bottom: 1px solid var(--border);
  padding-bottom: var(--sp-4);
  margin-bottom: var(--sp-5);
}
.app-head h1 {
  font-size: var(--fs-xl);
  font-weight: 600;
  color: var(--text);
  margin: 0;
  letter-spacing: -.01em;
}
.app-head p {
  font-size: var(--fs-md);
  color: var(--text-muted);
  margin: var(--sp-2) 0 0;
  max-width: 68ch;
}

/* ── Source cards ──────────────────────────────────────────────────────── */
.src {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: var(--sp-3) var(--sp-4);
  margin-bottom: var(--sp-2);
}
.src-top {
  display: flex; align-items: baseline;
  justify-content: space-between; gap: var(--sp-3);
}
.src-file {
  font-size: var(--fs-base);
  font-weight: 600;
  color: var(--text);
  word-break: break-word;
}
.src-index { color: var(--text-faint); font-family: var(--mono); margin-right: 6px; }
.src-score { font-size: var(--fs-sm); color: var(--text-muted); white-space: nowrap; }
/* A bare 0.891 means nothing without a name for the scale. */
.src-score-label { color: var(--text-faint); font-size: var(--fs-xs); margin-right: 3px; }
.src-meta {
  font-size: var(--fs-sm);
  color: var(--text-faint);
  margin-top: 5px;
  display: flex; align-items: center; gap: var(--sp-2); flex-wrap: wrap;
}
.src-quote {
  font-size: var(--fs-md);
  color: var(--text-muted);
  line-height: 1.55;
  margin-top: var(--sp-3);
  padding: var(--sp-2) var(--sp-3);
  background: var(--surface-sunk);
  border-left: 2px solid var(--border-bright);
  border-radius: 0 var(--radius) var(--radius) 0;
  white-space: pre-wrap;
}

/* ── Chips ─────────────────────────────────────────────────────────────── */
.chip {
  display: inline-block;
  font-size: var(--fs-xs);
  font-weight: 500;
  letter-spacing: .03em;
  padding: 1px 7px;
  border-radius: 3px;
  border: 1px solid currentColor;
  opacity: .9;
}
.chip-dgca  { color: var(--c-dgca); }
.chip-aic   { color: var(--c-aic); }
.chip-icao  { color: var(--c-icao); }
.chip-notam { color: var(--c-notam); }

/* ── Status line under an answer ───────────────────────────────────────── */
.statusline {
  font-size: var(--fs-sm);
  color: var(--text-faint);
  margin-top: var(--sp-3);
  padding-top: var(--sp-2);
  border-top: 1px solid var(--border);
}
.statusline span + span::before { content: "·"; margin: 0 var(--sp-2); opacity: .5; }

/* ── Streamlit surfaces ────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
  background: transparent;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: var(--sp-3) var(--sp-4);
}
[data-testid="stExpander"] details {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface-sunk);
}
[data-testid="stExpander"] summary { font-size: var(--fs-md); color: var(--text-muted); }

/* Streamlit's default avatars ship in its own brand red/orange, which fights
   this palette. Neutralise them and keep only a role distinction. */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
  background: var(--surface) !important;
  border: 1px solid var(--border);
  color: var(--text-faint) !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
  border-color: var(--accent);
  color: var(--accent) !important;
}

#MainMenu, footer { visibility: hidden; }
[data-testid="stAppDeployButton"], [data-testid="stToolbarActions"] { display: none; }
header[data-testid="stHeader"] { background: transparent; }

@media (max-width: 640px) {
  .metrics { flex-direction: column; }
  .src-top { flex-direction: column; gap: var(--sp-1); }
}
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)


# ─── Category presentation ────────────────────────────────────────────────
CATEGORIES = {
    "dgca_cars":     ("DGCA CAR",     "chip-dgca",  "var(--c-dgca)"),
    "aai_circulars": ("AIC/Circular", "chip-aic",   "var(--c-aic)"),
    "icao":          ("ICAO",         "chip-icao",  "var(--c-icao)"),
    "notams":        ("NOTAM",        "chip-notam", "var(--c-notam)"),
}


def category_meta(category: str) -> tuple[str, str, str]:
    """Display label, chip class and bar colour for a corpus category."""
    fallback = (category.replace("_", " ").upper(), "chip-aic", "var(--c-aic)")
    return CATEGORIES.get(category, fallback)


def chip(category: str) -> str:
    label, css_class, _ = category_meta(category)
    return f'<span class="chip {css_class}">{html.escape(label)}</span>'


# ─── Engine ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_engine():
    """Load the RAG engine once per process."""
    from rag_engine import AviationRAGEngine

    return AviationRAGEngine()


# ─── Sidebar ──────────────────────────────────────────────────────────────
def render_sidebar(engine) -> tuple[str, str | None]:
    with st.sidebar:
        st.markdown(
            '<p class="side-title">AeroRAG</p>'
            '<p class="side-sub">Indian civil aviation regulatory corpus</p>',
            unsafe_allow_html=True,
        )

        stats = engine.get_stats()

        st.markdown(
            '<div class="metrics">'
            f'<div class="metric"><span class="metric-value">{stats["total_documents"]:,}</span>'
            '<span class="metric-label">Documents</span></div>'
            f'<div class="metric"><span class="metric-value">{stats["total_chunks"]:,}</span>'
            '<span class="metric-label">Chunks</span></div>'
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown('<p class="side-label">Retrieval</p>', unsafe_allow_html=True)
        search_mode = st.selectbox(
            "Mode",
            ["hybrid", "dense", "sparse"],
            index=0,
            help=(
                "hybrid: BM25 + dense, fused by RRF, then cross-encoder reranked. "
                "dense: bi-encoder only. sparse: BM25 only."
            ),
            label_visibility="collapsed",
        )

        per_category = stats.get("chunks_per_category", {})
        options = ["All categories"] + list(per_category)
        selected = st.selectbox("Category filter", options, index=0)
        category_filter = None if selected == "All categories" else selected

        if per_category:
            st.markdown('<p class="side-label">Index composition</p>', unsafe_allow_html=True)
            largest = max(per_category.values())
            rows = []
            for name, count in sorted(per_category.items(), key=lambda kv: -kv[1]):
                label, _, colour = category_meta(name)
                width = (count / largest * 100) if largest else 0
                rows.append(
                    '<div class="comp-row">'
                    f'<div class="comp-head"><span class="comp-name">{html.escape(label)}</span>'
                    f'<span class="comp-count">{count:,}</span></div>'
                    f'<div class="comp-track"><div class="comp-fill" '
                    f'style="width:{width:.1f}%;background:{colour}"></div></div>'
                    "</div>"
                )
            st.markdown("".join(rows), unsafe_allow_html=True)

        st.markdown('<p class="side-label">Pipeline</p>', unsafe_allow_html=True)
        pipeline = [
            ("Embeddings", stats["embedding_model"]),
            ("Reranker", stats.get("reranker_model", "cross-encoder")),
            ("Generation", stats["llm_model"]),
            ("Vector store", "ChromaDB"),
            ("Lexical", "BM25 Okapi"),
            ("Fusion", "RRF (k=60)"),
        ]
        st.markdown(
            "".join(
                f'<div class="kv"><span class="kv-key">{html.escape(key)}</span>'
                f'<span class="kv-value">{html.escape(str(value))}</span></div>'
                for key, value in pipeline
            ),
            unsafe_allow_html=True,
        )

        st.markdown('<p class="side-label">Session</p>', unsafe_allow_html=True)
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

    return search_mode, category_filter


# ─── Main surfaces ────────────────────────────────────────────────────────
def render_header() -> None:
    st.markdown(
        '<div class="app-head">'
        "<h1>Aviation Regulatory Intelligence</h1>"
        "<p>Question answering over DGCA Civil Aviation Requirements, AAI circulars, "
        "aeronautical information circulars and ICAO standards. Every answer cites the "
        "source document and page it was drawn from.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_sources(sources: list[dict]) -> None:
    """Render retrieved chunks. All document-derived text is escaped: the corpus
    is third-party PDF text and will contain angle brackets and ampersands."""
    if not sources:
        return

    with st.expander(f"Sources — {len(sources)} retrieved", expanded=False):
        for src in sources:
            page = html.escape(str(src.get("page", "?")))
            section = str(src.get("section") or "").strip()
            meta = [chip(src.get("category", "")), f"p. {page}"]
            if section:
                meta.append(html.escape(section))

            st.markdown(
                '<div class="src">'
                '<div class="src-top">'
                f'<span class="src-file"><span class="src-index">'
                f'{int(src.get("source_number", 0)):02d}</span>'
                f'{html.escape(str(src.get("file", "unknown")))}</span>'
                '<span class="src-score"><span class="src-score-label">rel</span>'
                f'{src.get("relevance_score", 0):.3f}</span>'
                "</div>"
                f'<div class="src-meta">{"".join(f"<span>{part}</span>" for part in meta)}</div>'
                f'<div class="src-quote">{html.escape(str(src.get("preview", "")))}</div>'
                "</div>",
                unsafe_allow_html=True,
            )


def render_statusline(elapsed: float, mode: str, num_chunks: int) -> None:
    st.markdown(
        '<div class="statusline">'
        f"<span>{elapsed:.2f}s</span>"
        f"<span>{html.escape(mode)} retrieval</span>"
        f"<span>{num_chunks} chunks in context</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("chat_history", [])

    if not indexes_present():
        render_header()
        st.error("No search index found. Build it before starting the app.")
        st.code("python ingest.py", language="bash")
        st.caption(
            "Ingestion reads every PDF under data/raw/, extracts tables and prose "
            "separately, and writes the ChromaDB collection and BM25 index."
        )
        return

    with st.spinner("Loading retrieval engine"):
        engine = load_engine()

    search_mode, category_filter = render_sidebar(engine)
    render_header()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                render_sources(msg["sources"])

    if prompt := st.chat_input("Ask about aviation regulations"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating"):
                started = time.perf_counter()
                response = engine.query(
                    question=prompt,
                    mode=search_mode,
                    category_filter=category_filter,
                    chat_history=st.session_state.chat_history,
                )
                elapsed = time.perf_counter() - started

            st.markdown(response.answer)
            render_statusline(elapsed, response.retrieval_mode, response.num_chunks_used)
            render_sources(response.sources)

        st.session_state.messages.append(
            {"role": "assistant", "content": response.answer, "sources": response.sources}
        )
        st.session_state.chat_history.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response.answer},
            ]
        )
        st.session_state.chat_history = st.session_state.chat_history[-6:]


if __name__ == "__main__":
    main()
