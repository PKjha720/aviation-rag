"""
reslice_by_table.py  --  drop this in your aviation-rag/ folder and run:
    py reslice_by_table.py

Does NOT re-run retrieval and does NOT need GROQ_API_KEY.
It reuses the retrieved_ids already stored in eval_results/results.csv.

What it answers:
    Does the cross-encoder reranker hurt recall specifically on questions
    whose gold evidence sits in a TABLE?

Your summary_table.csv already hints at this (numerical: hybrid 0.72 -> full
0.64). But "numerical" is a question-category label, not a fact about the
chunk. This script replaces the proxy with the real variable.

Reads:  data/processed/chunks_metadata.json
        eval_results/eval_dataset.json
        eval_results/results.csv
Writes: eval_results/table_slice.csv
        eval_results/table_labels_sample.txt   <- HAND-CHECK THIS
"""

import ast
import json
import re
from pathlib import Path

import pandas as pd

TOP_K = 5


def looks_like_table(text: str) -> bool:
    """
    Heuristic for 'this chunk is mostly tabular'.
    PyMuPDF flattens tables into runs of short lines with several numbers,
    so that is what we look for. Tune it after reading the sample file.
    """
    if text.count("|") >= 6:
        return True
    if re.search(r"(?im)^\s*table\s+\d+", text):
        return True

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return False

    numeric_rows = sum(
        1
        for ln in lines
        if len(ln) < 120 and len(re.findall(r"\d+(?:\.\d+)?", ln)) >= 2
    )
    return numeric_rows >= 3 and numeric_rows / len(lines) > 0.5


def parse_ids(cell) -> list[str]:
    """retrieved_ids was written by pandas as the repr of a Python list."""
    if isinstance(cell, list):
        return cell
    if not isinstance(cell, str) or not cell.strip():
        return []
    try:
        val = ast.literal_eval(cell)
        return list(val) if isinstance(val, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def main():
    chunks = json.loads(
        Path("data/processed/chunks_metadata.json").read_text(encoding="utf-8")
    )
    text_by_id = {c["chunk_id"]: c.get("text", "") for c in chunks}

    evalset = json.loads(
        Path("eval_results/eval_dataset.json").read_text(encoding="utf-8")
    )
    gold_by_qid = {q["question_id"]: q["ground_truth_chunk_id"] for q in evalset}
    cat_by_qid = {q["question_id"]: q["category"] for q in evalset}

    is_table = {cid: looks_like_table(txt) for cid, txt in text_by_id.items()}
    n_tab = sum(is_table.values())
    print(f"corpus: {len(text_by_id)} chunks, {n_tab} labelled table-bearing "
          f"({n_tab / max(len(text_by_id), 1):.1%})\n")

    # ---- dump a sample so you can verify the labels by eye -----------------
    sample_path = Path("eval_results/table_labels_sample.txt")
    with sample_path.open("w", encoding="utf-8") as f:
        tab_ids = [c for c, v in is_table.items() if v][:15]
        pro_ids = [c for c, v in is_table.items() if not v][:15]
        for header, ids in (("LABELLED TABLE", tab_ids), ("LABELLED PROSE", pro_ids)):
            for cid in ids:
                f.write(f"\n{'=' * 70}\n{header}  {cid}\n{'=' * 70}\n")
                f.write(text_by_id[cid][:600] + "\n")
    print(f"-> hand-check 30 labels in {sample_path} BEFORE trusting anything below\n")

    # ---- recompute recall, sliced by table vs prose ------------------------
    df = pd.read_csv("eval_results/results.csv")
    df["gold"] = df["question_id"].map(gold_by_qid)
    df["gold_is_table"] = df["gold"].map(lambda g: is_table.get(g, False))
    df["recomputed_recall"] = [
        1.0 if row.gold in parse_ids(row.retrieved_ids)[:TOP_K] else 0.0
        for row in df.itertuples()
    ]

    # sanity: recomputed recall should match the stored column
    mismatch = (df["recomputed_recall"] != df["recall_at_5"]).sum()
    if mismatch:
        print(f"WARNING: {mismatch}/{len(df)} rows disagree with stored recall_at_5.")
        print("         retrieved_ids may be truncated. Investigate before quoting.\n")

    print(f"{'slice':<16}{'mode':<14}{'hits':>7}{'n':>6}{'recall':>9}")
    print("-" * 52)
    rows = []
    for label, want in (("TABLE", True), ("PROSE", False)):
        for mode in ("hybrid_rrf", "full"):
            s = df[(df.gold_is_table == want) & (df["mode"] == mode)]["recall_at_5"]
            if len(s) == 0:
                continue
            print(f"{label:<16}{mode:<14}{int(s.sum()):>7}{len(s):>6}{s.mean():>9.3f}")
            rows.append({"slice": label, "mode": mode,
                         "hits": int(s.sum()), "n": len(s), "recall": s.mean()})
        print("-" * 52)

    # ---- the headline delta ----------------------------------------------
    def rec(want, mode):
        s = df[(df.gold_is_table == want) & (df["mode"] == mode)]["recall_at_5"]
        return (s.mean(), len(s)) if len(s) else (float("nan"), 0)

    (th, nt), (tf, _) = rec(True, "hybrid_rrf"), rec(True, "full")
    (ph, npr), (pf, _) = rec(False, "hybrid_rrf"), rec(False, "full")
    print(f"\nreranker effect on TABLE queries : {tf - th:+.3f}  (n={nt})")
    print(f"reranker effect on PROSE queries : {pf - ph:+.3f}  (n={npr})")
    print("\nIf n is under ~40 per slice, treat this as a hypothesis, not a result.")

    # cross-tab against the old proxy, so you can see how good it was
    df["category"] = df["question_id"].map(cat_by_qid)
    print("\nhow well 'numerical' proxied for 'table':")
    print(pd.crosstab(df.drop_duplicates("question_id")["category"],
                      df.drop_duplicates("question_id")["gold_is_table"]))

    pd.DataFrame(rows).to_csv("eval_results/table_slice.csv", index=False)
    print("\nwritten: eval_results/table_slice.csv")


if __name__ == "__main__":
    main()
