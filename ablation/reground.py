"""
Build the re-grounded judgment anchors and run the reachability ceiling.

The ceiling matters more than it sounds. A judgment is only usable if the arm
can physically produce a chunk that covers the gold evidence. If Arm B has no
chunk covering gold Q at 60%, then Arm B's Recall@10 for Q is zero no matter how
good retrieval is, and the "retrieval delta" would really be an ingestion delta
wearing a retrieval label.

Reported per arm, per threshold:
  per-chunk ceiling  — some single chunk covers the gold at >= t
  union ceiling      — all chunks on the gold's page together cover it at >= t
                       (isolates windowing loss from extraction loss)

Writes: ablation/out/anchors.json
        ablation/out/ceiling.csv
        ablation/out/ceiling_report.txt
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from arms import build_arm, as_dicts
from judgments import THRESHOLDS, normalize, coverage, union_coverage, best_coverage

random.seed(0)
np.random.seed(0)

OUT = Path("ablation/out")
OUT.mkdir(parents=True, exist_ok=True)


def load_arm_b():
    cfg = json.loads((OUT / "arm_b_config.json").read_text())
    print(f"Building Arm B (window={cfg['window']}, overlap={cfg['overlap']})...")
    chunks = as_dicts(build_arm("B", cfg["window"], cfg["overlap"]))
    print(f"  Arm B built: {len(chunks)} chunks")
    (OUT / "arm_b_chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    return chunks, cfg


def main():
    arm_a = json.loads((OUT / "arm_a_chunks.json").read_text(encoding="utf-8"))
    arm_b, cfg = load_arm_b()

    ev = json.loads(Path("eval_results/eval_dataset_v2.json").read_text(encoding="utf-8"))
    a_by_id = {c["chunk_id"]: c for c in arm_a}

    # ── anchors: the chunking-invariant judgment ──────────────────────────
    anchors = []
    for q in ev:
        g = a_by_id.get(q["ground_truth_chunk_id"])
        if g is None:
            print(f"  !! {q['question_id']}: gold chunk id not in Arm A — skipped")
            continue
        anchors.append({
            "question_id": q["question_id"],
            "question": q["question"],
            "category": q["category"],
            "gold_is_table": q["gold_is_table"],
            "gold_chunk_id_armA": q["ground_truth_chunk_id"],
            "gold_text": g["text"],
            "gold_norm_len": len(normalize(g["text"])),
            "source_file": g["source_file"],
            "page_number": g["page_number"],
        })
    (OUT / "anchors.json").write_text(
        json.dumps(anchors, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT/'anchors.json'} — {len(anchors)} anchors "
          f"({sum(a['gold_is_table'] for a in anchors)} table / "
          f"{sum(not a['gold_is_table'] for a in anchors)} prose)")

    # ── index chunks by (file, page) for the ceiling ──────────────────────
    def by_page(chunks):
        d = defaultdict(list)
        for c in chunks:
            d[(c["source_file"], c["page_number"])].append(c)
        return d

    a_pages, b_pages = by_page(arm_a), by_page(arm_b)

    def by_file(chunks):
        d = defaultdict(list)
        for c in chunks:
            d[c["source_file"]].append(c)
        return d

    b_files = by_file(arm_b)

    rows = []
    print("\nComputing reachability ceiling...")
    for i, a in enumerate(anchors, 1):
        gn = normalize(a["gold_text"])
        key = (a["source_file"], a["page_number"])
        rec = {
            "question_id": a["question_id"],
            "gold_is_table": a["gold_is_table"],
            "category": a["category"],
            "source_file": a["source_file"],
            "page_number": a["page_number"],
            "gold_norm_len": len(gn),
        }
        for arm, pages, allchunks in (("A", a_pages, None), ("B", b_pages, b_files)):
            cand = pages.get(key, [])
            norms = [normalize(c["text"]) for c in cand]
            best, _ = best_coverage(gn, norms)
            uni = union_coverage(gn, norms)
            # If the page yields nothing, the page attribution may differ.
            # Fall back to scanning the whole source file before concluding.
            if best < 0.40 and allchunks is not None:
                fb = [normalize(c["text"]) for c in allchunks.get(a["source_file"], [])]
                fb_best, _ = best_coverage(gn, fb)
                if fb_best > best:
                    best = fb_best
                    rec[f"arm{arm}_page_fallback_used"] = True
            rec[f"arm{arm}_n_page_chunks"] = len(cand)
            rec[f"arm{arm}_best_chunk_cov"] = round(best, 4)
            rec[f"arm{arm}_union_page_cov"] = round(uni, 4)
        rows.append(rec)
        if i % 20 == 0:
            print(f"  {i}/{len(anchors)}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ceiling.csv", index=False)

    # ── report ─────────────────────────────────────────────────────────────
    lines = []
    w = lines.append
    w("REACHABILITY CEILING — can each arm produce a chunk covering the gold?")
    w("=" * 78)
    w(f"Arm A: current pipeline (table-preserving + page boundaries)")
    w(f"Arm B: flat extraction, {cfg['window']}-token window, {cfg['overlap']}-token overlap")
    w(f"n = {len(df)} questions ({int(df.gold_is_table.sum())} table-gold, "
      f"{int((~df.gold_is_table).sum())} prose-gold)")
    w("")
    w("Gold text is taken from the Arm A chunk that generated the question, so")
    w("Arm A's per-chunk ceiling is 1.000 BY CONSTRUCTION and is not evidence of")
    w("anything. Arm B's ceiling is the number that matters.")
    w("")

    for slice_name, mask in (("ALL", pd.Series(True, index=df.index)),
                             ("TABLE-GOLD", df.gold_is_table),
                             ("PROSE-GOLD", ~df.gold_is_table)):
        sub = df[mask]
        w("-" * 78)
        w(f"{slice_name}  (n={len(sub)})")
        w(f"  {'thresh':>7} | {'A per-chunk':>12} {'B per-chunk':>12} | "
          f"{'A union/pg':>11} {'B union/pg':>11}")
        for t in THRESHOLDS:
            a_pc = (sub.armA_best_chunk_cov >= t).mean()
            b_pc = (sub.armB_best_chunk_cov >= t).mean()
            a_un = (sub.armA_union_page_cov >= t).mean()
            b_un = (sub.armB_union_page_cov >= t).mean()
            w(f"  {t:>7.2f} | {a_pc:>12.3f} {b_pc:>12.3f} | {a_un:>11.3f} {b_un:>11.3f}")
        w(f"  median best per-chunk coverage: A={sub.armA_best_chunk_cov.median():.3f}  "
          f"B={sub.armB_best_chunk_cov.median():.3f}")
        w(f"  median union-over-page coverage: A={sub.armA_union_page_cov.median():.3f}  "
          f"B={sub.armB_union_page_cov.median():.3f}")
    w("-" * 78)

    txt = "\n".join(lines)
    (OUT / "ceiling_report.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)
    print(f"\nwrote {OUT/'ceiling.csv'} and {OUT/'ceiling_report.txt'}")


if __name__ == "__main__":
    main()
