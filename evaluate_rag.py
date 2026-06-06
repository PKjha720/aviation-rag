"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          AeroRAG  —  Evaluation Pipeline                                    ║
║                                                                              ║
║  WHAT THIS SCRIPT DOES (in order):                                           ║
║  1. Reads your existing chunked documents from data/processed/               ║
║  2. Auto-generates ~100 eval questions using Groq LLM                        ║
║  3. Runs 4 retrieval modes on every question:                                ║
║       (a) Dense-only    (b) Sparse-only                                      ║
║       (c) Hybrid-RRF    (d) Full pipeline (Hybrid + Reranking)               ║
║  4. Scores each: Recall@5, MRR@10, Answer Accuracy (LLM-judged)             ║
║  5. Saves results to eval_results/                                           ║
║       - results.csv          ← raw per-query numbers                         ║
║       - summary_table.csv    ← the table for your paper                      ║
║       - recall_chart.png     ← bar chart (paper-ready)                       ║
║       - mrr_chart.png        ← MRR bar chart                                 ║
║       - eval_dataset.json    ← the QA pairs (reusable)                       ║
║                                                                              ║
║  HOW TO RUN:                                                                 ║
║  Step 1: Put this file in your aviation-rag/ folder (same level as app.py)  ║
║  Step 2: Make sure your .env has GROQ_API_KEY set                            ║
║  Step 3: Make sure ingest.py has been run (data/processed/ must exist)       ║
║  Step 4: Run:  python evaluate_rag.py                                        ║
║                                                                              ║
║  TIME: ~15-20 minutes for 100 questions (Groq API calls)                     ║
║  COST: FREE — uses Groq free tier                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import sys
import json
import time
import random
import logging
import traceback
from pathlib import Path
from dataclasses import dataclass, asdict

# ── Third-party ───────────────────────────────────────────────────────────────
try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")          # no GUI needed — saves to file
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from groq import Groq
    from dotenv import load_dotenv
except ImportError as e:
    print(f"\n❌  Missing package: {e}")
    print("   Run:  pip install pandas matplotlib groq python-dotenv")
    sys.exit(1)

# ── Local imports ─────────────────────────────────────────────────────────────
# We import your existing RAG engine — no duplication
try:
    from rag_engine import AviationRAGEngine, RetrievedChunk
except ImportError:
    print("\n❌  Cannot import rag_engine.py")
    print("   Make sure this script is in the same folder as rag_engine.py")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval")

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path("eval_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
NUM_EVAL_QUESTIONS   = 100    # total questions to generate
QUESTIONS_PER_CHUNK  = 1      # questions generated per sampled chunk
GROQ_MODEL           = "llama-3.1-8b-instant"
RERANK_TOP_K         = 5      # final chunks used for retrieval eval

# Query categories — balanced across the eval set
CATEGORIES = {
    "direct_regulation": "Direct regulation lookup (specific CAR/AIC section or requirement)",
    "numerical":         "Numerical threshold or measurement (altitudes, speeds, weights, limits)",
    "procedural":        "Step-by-step procedure or process",
    "cross_document":    "Requires synthesising information from multiple documents or categories",
}
CAT_QUOTA = NUM_EVAL_QUESTIONS // len(CATEGORIES)   # 25 per category


# ═════════════════════════════════════════════════════════════════════════════
# PART 1: AUTO-GENERATE EVAL DATASET
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalQuestion:
    question_id:   str
    question:      str
    category:      str
    source_file:   str    # the chunk this was generated from
    source_page:   int
    ground_truth_chunk_id: str   # the chunk that should be retrieved


def load_chunks_metadata() -> list[dict]:
    """Load the chunks metadata JSON created by ingest.py."""
    path = Path("data/processed/chunks_metadata.json")
    if not path.exists():
        log.error("data/processed/chunks_metadata.json not found.")
        log.error("Run ingest.py first to build your indices.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    log.info(f"Loaded {len(chunks)} chunks from metadata.")
    return chunks


def generate_question_from_chunk(
    groq_client: Groq,
    chunk: dict,
    category: str,
    category_desc: str,
    retries: int = 3,
) -> str | None:
    """
    Ask Groq LLM to write one realistic aviation question from a chunk of text.
    Returns the question string, or None if it fails.
    """
    prompt = f"""You are creating an evaluation dataset for a Retrieval-Augmented Generation system
built for Indian civil aviation regulatory documents.

Your task: Read the aviation regulatory text below and write EXACTLY ONE realistic question
that a compliance officer or aviation engineer would ask, where the answer is directly found
in this text.

Requirements:
- Question type: {category_desc}
- The question must be answerable FROM THIS TEXT ALONE
- Use natural professional language (not "According to the text...")
- Do NOT include the answer
- Output only the question, nothing else — no numbering, no preamble

Aviation regulatory text:
\"\"\"
{chunk['text'][:800]}
\"\"\"

Question:"""

    for attempt in range(retries):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7,
            )
            question = response.choices[0].message.content.strip()
            # Basic sanity check — must end with ? and be non-trivial
            if "?" in question and len(question) > 20:
                return question
        except Exception as e:
            wait = 60 if "429" in str(e) else 3
            log.warning(f"  Attempt {attempt+1} failed — waiting {wait}s: {e}")
            time.sleep(wait)
    return None


def build_eval_dataset(groq_client: Groq, chunks: list[dict]) -> list[EvalQuestion]:
    """
    Generate NUM_EVAL_QUESTIONS questions balanced across 4 categories.
    Saves to eval_results/eval_dataset.json and returns the list.
    """
    # Check if we already have a saved dataset — skip regeneration
    saved_path = OUTPUT_DIR / "eval_dataset.json"
    if saved_path.exists():
        log.info(f"✓ Found existing eval dataset at {saved_path} — loading it.")
        log.info("  (Delete this file to regenerate fresh questions)")
        with open(saved_path, "r") as f:
            raw = json.load(f)
        return [EvalQuestion(**q) for q in raw]

    log.info("═" * 60)
    log.info("STEP 1: Generating evaluation dataset from your documents")
    log.info("═" * 60)

    # Filter out very short chunks — not useful for question generation
    good_chunks = [c for c in chunks if len(c.get("text", "")) > 200]
    log.info(f"  {len(good_chunks)} chunks qualify for question generation (>200 chars)")

    eval_questions: list[EvalQuestion] = []
    q_id = 0

    for cat_key, cat_desc in CATEGORIES.items():
        log.info(f"\n  Generating {CAT_QUOTA} questions for category: [{cat_key}]")
        generated = 0
        attempts   = 0
        max_attempts = CAT_QUOTA * 5   # allow retries

        # Shuffle chunks so we get variety across documents
        shuffled = random.sample(good_chunks, min(len(good_chunks), max_attempts))

        for chunk in shuffled:
            if generated >= CAT_QUOTA:
                break
            attempts += 1

            question_text = generate_question_from_chunk(
                groq_client, chunk, cat_key, cat_desc
            )
            if question_text is None:
                continue

            q_id += 1
            eval_questions.append(EvalQuestion(
                question_id            = f"Q{q_id:03d}",
                question               = question_text,
                category               = cat_key,
                source_file            = chunk.get("source_file", ""),
                source_page            = chunk.get("page_number", 0),
                ground_truth_chunk_id  = chunk.get("chunk_id", ""),
            ))
            generated += 1

            # Small delay to stay within Groq rate limits
            time.sleep(0.3)
            log.info(f"    [{generated}/{CAT_QUOTA}] {question_text[:80]}...")

    log.info(f"\n✓ Generated {len(eval_questions)} evaluation questions total.")

    # Save so you don't have to regenerate every time
    with open(saved_path, "w") as f:
        json.dump([asdict(q) for q in eval_questions], f, indent=2)
    log.info(f"  Saved to {saved_path}")

    return eval_questions


# ═════════════════════════════════════════════════════════════════════════════
# PART 2: RETRIEVAL EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class QueryResult:
    """Stores retrieval + answer results for one question under one mode."""
    question_id:    str
    category:       str
    mode:           str
    recall_at_5:    float    # 1.0 if ground truth chunk in top-5, else 0.0
    reciprocal_rank: float   # 1/rank if found in top-10, else 0.0
    answer_accuracy: float   # 1.0 / 0.5 / 0.0 from LLM judge
    retrieved_ids:  list     # chunk_ids actually retrieved


def compute_recall_at_k(
    retrieved_chunks: list[RetrievedChunk],
    ground_truth_chunk_id: str,
    k: int = 5,
) -> float:
    """
    Returns 1.0 if the ground-truth chunk appears in the top-k results.

    HOW THIS WORKS:
    We check if the chunk we KNOW is relevant (from the generation step)
    appears anywhere in the top-k retrieved chunks. This is the standard
    Recall@K metric used in information retrieval research.
    """
    retrieved_ids = [c.chunk_id for c in retrieved_chunks[:k]]
    return 1.0 if ground_truth_chunk_id in retrieved_ids else 0.0


def compute_reciprocal_rank(
    retrieved_chunks: list[RetrievedChunk],
    ground_truth_chunk_id: str,
    k: int = 10,
) -> float:
    """
    MRR = 1 / rank_of_first_relevant_result.
    If not found in top-k, returns 0.0.

    EXAMPLE: If the correct chunk appears at rank 3, MRR = 1/3 = 0.333
    """
    ids = [c.chunk_id for c in retrieved_chunks[:k]]
    if ground_truth_chunk_id in ids:
        rank = ids.index(ground_truth_chunk_id) + 1   # 1-indexed
        return 1.0 / rank
    return 0.0


def judge_answer_accuracy(
    groq_client: Groq,
    question: str,
    answer: str,
    ground_truth_chunk_text: str,
    retries: int = 2,
) -> float:
    """
    Uses Groq LLM as a judge to score the generated answer.

    Returns:
      1.0  — fully correct and grounded in the source text
      0.5  — partially correct (right topic, missing detail)
      0.0  — incorrect or hallucinated

    This is the standard LLM-as-judge evaluation pattern used in
    research papers (e.g., MT-Bench, RAGAS).
    """
    prompt = f"""You are an expert evaluator for a Question Answering system about Indian civil aviation regulations.

Question: {question}

Source text (ground truth): 
\"\"\"{ground_truth_chunk_text[:600]}\"\"\"

System answer: 
\"\"\"{answer[:500]}\"\"\"

Score the answer on this scale:
- 1.0: Answer is fully correct and directly supported by the source text
- 0.5: Answer is partially correct — right direction but missing key details or imprecise
- 0.0: Answer is incorrect, hallucinated, or not supported by the source text

Respond with ONLY a single number: 1.0 or 0.5 or 0.0"""

    for attempt in range(retries):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )
            score_str = response.choices[0].message.content.strip()
            score = float(score_str)
            if score in (0.0, 0.5, 1.0):
                return score
        except Exception as e:
            wait = 60 if "429" in str(e) else 2
            log.warning(f"  Judge attempt {attempt+1} failed — waiting {wait}s: {e}")
            time.sleep(wait)
    return 0.0   # default to 0 if judge fails


def retrieve_with_mode(
    engine: AviationRAGEngine,
    query: str,
    mode: str,
    top_k: int = RERANK_TOP_K,
) -> list[RetrievedChunk]:
    """
    Wrapper that handles our 4 retrieval modes.

    The 4 modes we compare:
    - "dense"         → vector search only, with reranking
    - "sparse"        → BM25 only, with reranking
    - "hybrid_rrf"    → dense + sparse + RRF, WITHOUT reranking
    - "full"          → dense + sparse + RRF + cross-encoder reranking  ← AeroRAG
    """
    if mode in ("dense", "sparse"):
        # engine.retrieve() already applies reranking to dense/sparse
        return engine.retrieve(query=query, mode=mode, top_k=top_k)

    elif mode == "hybrid_rrf":
        # We want RRF fusion WITHOUT the reranking step
        # So we call the internal methods directly
        dense_hits  = engine._dense_search(query)
        sparse_hits = engine._sparse_search(query)
        fused       = engine._reciprocal_rank_fusion(dense_hits, sparse_hits)
        # Take top_k directly from RRF without cross-encoder
        return fused[:top_k]

    elif mode == "full":
        # Full AeroRAG pipeline: hybrid + reranking
        return engine.retrieve(query=query, mode="hybrid", top_k=top_k)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_retrieval_evaluation(
    engine: AviationRAGEngine,
    groq_client: Groq,
    eval_questions: list[EvalQuestion],
    chunks_by_id: dict,
) -> list[QueryResult]:
    """
    Core evaluation loop.
    For each question × each retrieval mode → compute Recall@5, MRR, Accuracy.
    """
    MODES = ["dense", "sparse", "hybrid_rrf", "full"]
    results: list[QueryResult] = []

    log.info("\n" + "═" * 60)
    log.info("STEP 2: Running 4-way retrieval ablation")
    log.info(f"  {len(eval_questions)} questions × {len(MODES)} modes = {len(eval_questions)*len(MODES)} evaluations")
    log.info("═" * 60)

    total = len(eval_questions) * len(MODES)
    done  = 0

    for q in eval_questions:
        gt_chunk = chunks_by_id.get(q.ground_truth_chunk_id)
        gt_text  = gt_chunk["text"] if gt_chunk else ""

        for mode in MODES:
            done += 1
            log.info(f"  [{done}/{total}] {q.question_id} | mode={mode} | {q.question[:60]}...")

            # ── Retrieval ──────────────────────────────────────────────────
            try:
                retrieved = retrieve_with_mode(engine, q.question, mode)
            except Exception as e:
                log.warning(f"    Retrieval failed: {e}")
                retrieved = []

            retrieved_ids = [c.chunk_id for c in retrieved]

            # ── Metrics ────────────────────────────────────────────────────
            recall    = compute_recall_at_k(retrieved, q.ground_truth_chunk_id, k=5)
            mrr       = compute_reciprocal_rank(retrieved, q.ground_truth_chunk_id, k=10)

            # Answer accuracy: only run LLM judge on "full" mode to save API calls
            # For others, accuracy = recall (proxy) — common in ablation studies
            if mode == "full" and gt_text:
                # Generate an answer first, then judge it
                try:
                    rag_response = engine.query(q.question, mode="hybrid")
                    answer       = rag_response.answer
                    accuracy     = judge_answer_accuracy(
                        groq_client, q.question, answer, gt_text
                    )
                    time.sleep(0.5)   # rate limit buffer
                except Exception as e:
                    log.warning(f"    Answer generation/judging failed: {e}")
                    accuracy = recall   # fallback
            else:
                # For ablation modes, use recall as proxy for accuracy
                # (standard in retrieval-focused ablation papers)
                accuracy = recall

            results.append(QueryResult(
                question_id     = q.question_id,
                category        = q.category,
                mode            = mode,
                recall_at_5     = recall,
                reciprocal_rank = mrr,
                answer_accuracy = accuracy,
                retrieved_ids   = retrieved_ids,
            ))

            time.sleep(0.2)   # gentle rate limiting

    log.info(f"\n✓ Evaluation complete. {len(results)} results collected.")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# PART 3: AGGREGATE + EXPORT
# ═════════════════════════════════════════════════════════════════════════════

MODE_LABELS = {
    "dense":       "Dense-only",
    "sparse":      "Sparse-only",
    "hybrid_rrf":  "Hybrid (RRF)",
    "full":        "Full AeroRAG",
}

CAT_LABELS = {
    "direct_regulation": "(A) Direct Regulation",
    "numerical":         "(B) Numerical Threshold",
    "procedural":        "(C) Procedural",
    "cross_document":    "(D) Cross-Document",
}

COLORS = {
    "dense":      "#5B9BD5",
    "sparse":     "#ED7D31",
    "hybrid_rrf": "#70AD47",
    "full":       "#FFC000",
}


def build_summary_table(results: list[QueryResult]) -> pd.DataFrame:
    """
    Builds the summary table that goes into your paper.

    Columns: Query Category | Dense R@5 | Sparse R@5 | Hybrid R@5 | Full R@5
    + overall row + MRR row
    """
    df = pd.DataFrame([asdict(r) for r in results])

    rows = []
    for cat_key, cat_label in CAT_LABELS.items():
        cat_df = df[df["category"] == cat_key]
        row = {"Query Category": cat_label}
        for mode in ["dense", "sparse", "hybrid_rrf", "full"]:
            subset = cat_df[cat_df["mode"] == mode]
            row[MODE_LABELS[mode] + " R@5"] = round(subset["recall_at_5"].mean(), 2) if len(subset) else 0.0
        rows.append(row)

    # Overall row
    overall = {"Query Category": "Overall"}
    for mode in ["dense", "sparse", "hybrid_rrf", "full"]:
        subset = df[df["mode"] == mode]
        overall[MODE_LABELS[mode] + " R@5"] = round(subset["recall_at_5"].mean(), 2)
    rows.append(overall)

    # MRR row
    mrr_row = {"Query Category": "MRR@10"}
    for mode in ["dense", "sparse", "hybrid_rrf", "full"]:
        subset = df[df["mode"] == mode]
        mrr_row[MODE_LABELS[mode] + " R@5"] = round(subset["reciprocal_rank"].mean(), 3)
    rows.append(mrr_row)

    summary = pd.DataFrame(rows)
    # Rename columns cleanly
    summary.columns = [c.replace(" R@5", "") for c in summary.columns]
    summary.rename(columns={"Dense-only": "Dense R@5", "Sparse-only": "Sparse R@5",
                             "Hybrid (RRF)": "Hybrid R@5", "Full AeroRAG": "Full R@5"}, inplace=True)
    # Fix the metric rows
    summary.columns = list(summary.columns)

    return summary


def plot_recall_chart(summary_df: pd.DataFrame, save_path: Path):
    """
    Grouped bar chart: Recall@5 per category per retrieval mode.
    Publication-ready (300 DPI, clean axes).
    """
    categories = list(CAT_LABELS.values()) + ["Overall"]
    modes      = ["Dense-only", "Sparse-only", "Hybrid (RRF)", "Full AeroRAG"]
    mode_cols  = ["Dense R@5", "Sparse R@5", "Hybrid R@5", "Full R@5"]
    mode_colors = list(COLORS.values())

    # Filter out MRR row
    plot_df = summary_df[summary_df["Query Category"] != "MRR@10"].copy()

    x         = np.arange(len(plot_df))
    width     = 0.18
    fig, ax   = plt.subplots(figsize=(12, 6))

    for i, (mode_label, col, color) in enumerate(zip(modes, mode_cols, mode_colors)):
        offset = (i - 1.5) * width
        bars   = ax.bar(x + offset, plot_df[col], width, label=mode_label, color=color,
                        edgecolor="white", linewidth=0.8)
        # Value labels on bars
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8, color="#333333")

    ax.set_xlabel("Query Category", fontsize=12, labelpad=10)
    ax.set_ylabel("Recall@5", fontsize=12)
    ax.set_title("AeroRAG Retrieval Ablation — Recall@5 by Query Category",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Query Category"], fontsize=10, rotation=10, ha="right")
    ax.set_ylim(0, 1.12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"  Chart saved: {save_path}")


def plot_mrr_chart(results: list[QueryResult], save_path: Path):
    """
    Horizontal bar chart comparing MRR@10 across retrieval modes.
    """
    df    = pd.DataFrame([asdict(r) for r in results])
    modes = ["dense", "sparse", "hybrid_rrf", "full"]
    mrrs  = [df[df["mode"] == m]["reciprocal_rank"].mean() for m in modes]
    labels = [MODE_LABELS[m] for m in modes]
    colors = [COLORS[m] for m in modes]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars    = ax.barh(labels, mrrs, color=colors, edgecolor="white", height=0.5)

    for bar, val in zip(bars, mrrs):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=11, color="#333333")

    ax.set_xlabel("MRR@10", fontsize=12)
    ax.set_title("AeroRAG Retrieval Ablation — MRR@10", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_xlim(0, max(mrrs) * 1.25)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"  Chart saved: {save_path}")


def export_all_results(results: list[QueryResult], summary_df: pd.DataFrame):
    """Save everything to eval_results/."""
    # Raw per-query CSV
    raw_path = OUTPUT_DIR / "results.csv"
    pd.DataFrame([asdict(r) for r in results]).to_csv(raw_path, index=False)
    log.info(f"  Raw results  → {raw_path}")

    # Summary table CSV
    summ_path = OUTPUT_DIR / "summary_table.csv"
    summary_df.to_csv(summ_path, index=False)
    log.info(f"  Summary table → {summ_path}")

    # Print the table to console so you can see it immediately
    print("\n" + "═" * 70)
    print("  RESULTS TABLE (copy these numbers into your paper)")
    print("═" * 70)
    print(summary_df.to_string(index=False))
    print("═" * 70 + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║         AeroRAG  —  Evaluation Pipeline Starting            ║
╚══════════════════════════════════════════════════════════════╝
""")

    # ── Check API key ──────────────────────────────────────────────────────
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌  GROQ_API_KEY not found in environment.")
        print("   Add it to your .env file:  GROQ_API_KEY=gsk_...")
        sys.exit(1)

    groq_client = Groq(api_key=api_key)
    log.info("✓ Groq client initialised.")

    # ── Load chunks ────────────────────────────────────────────────────────
    chunks = load_chunks_metadata()

    # Build a lookup dict: chunk_id → chunk  (used for answer judging)
    chunks_by_id = {c["chunk_id"]: c for c in chunks}

    # ── Generate / load eval dataset ──────────────────────────────────────
    eval_questions = build_eval_dataset(groq_client, chunks)

    if len(eval_questions) == 0:
        log.error("No eval questions generated. Check your Groq API key and chunks.")
        sys.exit(1)

    # ── Load RAG engine ────────────────────────────────────────────────────
    log.info("\n" + "═" * 60)
    log.info("Loading AviationRAGEngine...")
    log.info("═" * 60)
    engine = AviationRAGEngine()

    # ── Run evaluation ─────────────────────────────────────────────────────
    results = run_retrieval_evaluation(engine, groq_client, eval_questions, chunks_by_id)

    # ── Aggregate & export ─────────────────────────────────────────────────
    log.info("\n" + "═" * 60)
    log.info("STEP 3: Aggregating results and generating charts")
    log.info("═" * 60)

    summary_df = build_summary_table(results)
    export_all_results(results, summary_df)

    plot_recall_chart(summary_df, OUTPUT_DIR / "recall_chart.png")
    plot_mrr_chart(results,       OUTPUT_DIR / "mrr_chart.png")

    print(f"""
✅  EVALUATION COMPLETE

Files saved in: {OUTPUT_DIR.resolve()}/
  📄  results.csv         — per-query raw numbers
  📄  summary_table.csv   — table for your paper
  📊  recall_chart.png    — bar chart (300 DPI, paper-ready)
  📊  mrr_chart.png       — MRR chart (300 DPI, paper-ready)
  📁  eval_dataset.json   — the 100 QA pairs (reusable)

Next steps:
  1. Open summary_table.csv → update Table 2 in your paper with real numbers
  2. Insert recall_chart.png and mrr_chart.png as Figure 2 and Figure 3
  3. Run again anytime — will reuse existing eval_dataset.json
     (delete eval_dataset.json to regenerate fresh questions)
""")


if __name__ == "__main__":
    main()
