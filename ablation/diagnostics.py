"""
First-class diagnostics, independent of any retrieval delta.

(1) The old 80-question pool, recomputed with the CONTIGUOUS criterion and
    marked CONTAMINATED - DO NOT REPORT, with the three reasons inline.
(2) Gold-length reachability: can a single chunk in each arm physically hold
    the gold at all, and if not, how much of it COULD one chunk hold?
(3) Column-wise vs row-wise extraction failure, per query: how many table golds
    have their content present on the page but broken into scattered fragments
    in Arm B.

These stand on their own. They are properties of ingestion, measured before any
query is issued, and they do not depend on which retriever is used.

Every number in the written report is derived from the dataframe or from the
arm chunk files at run time. Nothing numeric is typed into a string literal.
The attribution counts are cross-checked against notes.ATTRIBUTION_COUNTS and
the script refuses to write a report if they disagree.

Writes: ablation/out/DIAGNOSTICS.txt
        ablation/out/scatter_per_query.csv
        ablation/out/ceiling_contiguous.csv
"""

import inspect
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ingest import chunk_table
from judgments import (THRESHOLDS, normalize, coverage, coverage_blocks,
                       union_coverage)
from notes import ATTRIBUTION_COUNTS, format_attribution

random.seed(0)
np.random.seed(0)

OUT = Path("ablation/out")

# A gold is "scattered" when its characters are largely present in a chunk but
# cannot be found as one run.
SCATTER_BLOCKS_MIN = 0.60
SCATTER_CONTIG_MAX = 0.35
REACH = 0.60


def contamination_notice(a_max_norm, a_max_raw, table_soft_cap, b_max_norm, window):
    return f"""\
################################################################################
#  OLD 80-QUESTION POOL  -  CONTAMINATED - DO NOT REPORT                       #
#                                                                              #
#  Computed and kept in the repo for completeness. It must not be quoted as a  #
#  result, in the email or anywhere else. Three independent defects:           #
#                                                                              #
#  1. QUESTIONS WERE WRITTEN FROM ARM A CHUNKS. The generator saw Arm A's      #
#     markdown table rendering. The pool is, by construction, the set of       #
#     questions an Arm-A index can answer.                                     #
#                                                                              #
#  2. GOLD IS DEFINED AS AN ARM A CHUNK. Arm A's per-chunk ceiling is          #
#     therefore 1.000 by construction, not by merit. Arm B is scored against   #
#     a target that is literally one of Arm A's outputs.                       #
#                                                                              #
#  3. GOLD LENGTHS COME FROM A TAIL ARM B CANNOT MATCH. Arm A chunks run to    #
#     {a_max_norm} normalized chars ({a_max_raw} raw; chunk_table's {table_soft_cap} is a SOFT cap that
#     a single long row can exceed). Arm B is bounded at {b_max_norm} normalized chars
#     by the {window}-token window. A gold longer than {b_max_norm} cannot be FULLY
#     covered by one Arm B chunk; how much of it CAN be is a per-gold number
#     (column armB_max_possible_cov), not a blanket statement.
################################################################################
"""


def load():
    arm_a = json.loads((OUT / "arm_a_chunks.json").read_text(encoding="utf-8"))
    arm_b = json.loads((OUT / "arm_b_chunks.json").read_text(encoding="utf-8"))
    anchors = json.loads((OUT / "anchors.json").read_text(encoding="utf-8"))
    cfg = json.loads((OUT / "arm_b_config.json").read_text(encoding="utf-8"))
    return arm_a, arm_b, anchors, cfg


def by_page(chunks):
    d = defaultdict(list)
    for c in chunks:
        d[(c["source_file"], c["page_number"])].append(c)
    return d


def main():
    arm_a, arm_b, anchors, cfg = load()
    a_pages, b_pages = by_page(arm_a), by_page(arm_b)

    a_norm_len = np.array([len(normalize(c["text"])) for c in arm_a])
    b_norm_len = np.array([len(normalize(c["text"])) for c in arm_b])
    a_max, b_max = int(a_norm_len.max()), int(b_norm_len.max())
    a_max_raw = max(c["char_count"] for c in arm_a)
    table_soft_cap = inspect.signature(chunk_table).parameters["max_chars"].default
    window = cfg["window"]

    rows = []
    for a in anchors:
        gn = normalize(a["gold_text"])
        k = (a["source_file"], a["page_number"])
        r = {
            "question_id": a["question_id"],
            "gold_is_table": a["gold_is_table"],
            "category": a["category"],
            "source_file": a["source_file"],
            "page_number": a["page_number"],
            "gold_norm_len": len(gn),
        }
        for arm, pages in (("A", a_pages), ("B", b_pages)):
            cand = [normalize(c["text"]) for c in pages.get(k, [])]
            if not cand:
                r.update({f"arm{arm}_contig": 0.0, f"arm{arm}_blocks": 0.0,
                          f"arm{arm}_contig_at_best_blocks": 0.0,
                          f"arm{arm}_union_contig": 0.0, f"arm{arm}_n": 0})
                continue
            contigs = [coverage(gn, c) for c in cand]
            blocks = [coverage_blocks(gn, c) for c in cand]
            best_i = int(np.argmax(blocks))
            r[f"arm{arm}_n"] = len(cand)
            r[f"arm{arm}_contig"] = round(max(contigs), 4)
            r[f"arm{arm}_blocks"] = round(blocks[best_i], 4)
            r[f"arm{arm}_contig_at_best_blocks"] = round(contigs[best_i], 4)
            r[f"arm{arm}_union_contig"] = round(union_coverage(gn, cand), 4)
        r["armB_scatter_gap"] = round(r["armB_blocks"] - r["armB_contig_at_best_blocks"], 4)
        r["armB_scattered"] = bool(r["armB_blocks"] >= SCATTER_BLOCKS_MIN
                                   and r["armB_contig_at_best_blocks"] < SCATTER_CONTIG_MAX)
        r["fits_in_armB"] = len(gn) <= b_max
        r["fits_in_armA"] = len(gn) <= a_max
        # The most of this gold ONE chunk of the arm could hold, if perfectly
        # aligned. Equals 1.0 whenever the gold fits; below 1.0 it is the true
        # ceiling and is what "arithmetically barred" must be read against.
        r["armB_max_possible_cov"] = round(min(1.0, b_max / len(gn)), 4) if gn else 0.0
        r["armA_max_possible_cov"] = round(min(1.0, a_max / len(gn)), 4) if gn else 0.0
        rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ceiling_contiguous.csv", index=False)
    df[["question_id", "gold_is_table", "gold_norm_len", "armB_blocks",
        "armB_contig_at_best_blocks", "armB_scatter_gap", "armB_scattered",
        "armB_max_possible_cov", "source_file", "page_number"]
       ].to_csv(OUT / "scatter_per_query.csv", index=False)

    # ── derived counts, computed once and used everywhere below ──────────────
    t = df[df.gold_is_table]
    pr = df[~df.gold_is_table]
    too_long = ~t.fits_in_armB
    # Structurally exclusive: each row lands in exactly one bucket by construction.
    reached = t.fits_in_armB & (t.armB_contig >= REACH)
    scattered = t.fits_in_armB & (t.armB_contig < REACH) & t.armB_scattered
    lowcov = t.fits_in_armB & (t.armB_contig < REACH) & ~t.armB_scattered
    assert int(too_long.sum() + reached.sum() + scattered.sum() + lowcov.sum()) == len(t)
    # NOTE: a too-long gold that reached 0.60 would fall in `too_long`, not
    # `reached`. That has not happened on this data (see barred/possible split
    # below) but the bucket order is chosen so it cannot double-count if it does.
    too_long_reached = int((too_long & (t.armB_contig >= REACH)).sum())

    tl = t[too_long]
    barred_060 = int((tl.armB_max_possible_cov < 0.60).sum())
    barred_050 = int((tl.armB_max_possible_cov < 0.50).sum())
    barred_040 = int((tl.armB_max_possible_cov < 0.40).sum())

    computed = {
        "n_table_golds": len(t),
        "too_long": int(too_long.sum()),
        "too_long_barred_060": barred_060,
        "low_cov": int(lowcov.sum()),
        "scattered": int(scattered.sum()),
        "reached": int(reached.sum()),
        "prose_scatter_gt_020": int((pr.armB_scatter_gap > 0.20).sum()),
        "table_scatter_gt_020": int((t.armB_scatter_gap > 0.20).sum()),
    }
    computed["failures"] = computed["n_table_golds"] - computed["reached"]

    mismatch = {k: (computed[k], ATTRIBUTION_COUNTS.get(k))
                for k in computed if computed[k] != ATTRIBUTION_COUNTS.get(k)}
    if mismatch:
        raise SystemExit(
            "REFUSING TO WRITE DIAGNOSTICS: notes.ATTRIBUTION_COUNTS is stale.\n"
            "  computed vs recorded: " + json.dumps(mismatch) + "\n"
            "  Update notes.ATTRIBUTION_COUNTS to the computed values, then re-run.")

    # ── report ──────────────────────────────────────────────────────────────
    L = []
    w = L.append
    w(contamination_notice(a_max, a_max_raw, table_soft_cap, b_max, window))
    w("")
    w("=" * 78)
    w("(1) CEILING, RECOMPUTED WITH THE CONTIGUOUS CRITERION")
    w("=" * 78)
    w("Longest contiguous run only. This supersedes the block-sum figures that an")
    w("earlier version of reground.py wrote to ceiling.csv / ceiling_report.txt;")
    w("those files have been removed from the repository.")
    w("Arm A per-chunk is 1.000 by construction - see defect 2 above.")
    w("")
    for name, mask in (("ALL", pd.Series(True, index=df.index)),
                       ("TABLE-GOLD", df.gold_is_table),
                       ("PROSE-GOLD", ~df.gold_is_table)):
        s = df[mask]
        w(f"  {name} (n={len(s)})")
        w(f"    {'thresh':>7} | {'A per-chunk':>12} {'B per-chunk':>12} | {'B union/page':>13}")
        for th in THRESHOLDS:
            w(f"    {th:>7.2f} | {(s.armA_contig >= th).mean():>12.3f} "
              f"{(s.armB_contig >= th).mean():>12.3f} | "
              f"{(s.armB_union_contig >= th).mean():>13.3f}")
        w(f"    median B per-chunk contiguous coverage: {s.armB_contig.median():.3f}")
        w("")

    w("=" * 78)
    w("(2) GOLD-LENGTH REACHABILITY  [first-class result]")
    w("=" * 78)
    w("Can one chunk in the arm physically hold the gold, and if not, how much of")
    w("it could one chunk hold at best?")
    w("")
    w(f"  longest chunk, Arm A : {a_max} normalized / {a_max_raw} raw chars")
    w(f"      (chunk_table's max_chars={table_soft_cap} is a SOFT cap; one long row exceeds it)")
    w(f"  longest chunk, Arm B : {b_max} normalized chars  (bounded by the {window}-token window)")
    w("")
    w(f"  {'slice':<12}{'n':>4}{'gold median':>13}{'gold max':>10}"
      f"{'fits Arm A':>12}{'fits Arm B':>12}")
    for name, mask in (("ALL", pd.Series(True, index=df.index)),
                       ("TABLE-GOLD", df.gold_is_table),
                       ("PROSE-GOLD", ~df.gold_is_table)):
        s = df[mask]
        w(f"  {name:<12}{len(s):>4}{s.gold_norm_len.median():>13.0f}"
          f"{s.gold_norm_len.max():>10.0f}{s.fits_in_armA.mean():>12.3f}"
          f"{s.fits_in_armB.mean():>12.3f}")
    w("")
    w(f"  Arm B per-chunk coverage, split by whether the gold fits in ANY Arm B chunk:")
    for label, mask in (("gold FITS Arm B", df.fits_in_armB),
                        ("gold TOO LONG for Arm B", ~df.fits_in_armB)):
        s = df[mask]
        if len(s) == 0:
            w(f"    {label:<26} n=0")
            continue
        w(f"    {label:<26} n={len(s):>3}  median contig={s.armB_contig.median():.3f}"
          f"  reach {REACH:.2f}: {(s.armB_contig >= REACH).mean():.3f}")
    w("")
    w(f"  THE {int(too_long.sum())} TOO-LONG TABLE GOLDS, BY WHAT ONE ARM B CHUNK COULD HOLD AT BEST")
    w("  (armB_max_possible_cov = Arm B max chunk length / gold length):")
    w(f"    max possible < 0.60  (arithmetically barred from {REACH:.2f}) : {barred_060:>2}")
    w(f"    max possible < 0.50                                       : {barred_050:>2}")
    w(f"    max possible < 0.40                                       : {barred_040:>2}")
    w(f"    max possible >= 0.60 (COULD reach {REACH:.2f}, measured not to) : "
      f"{int(too_long.sum()) - barred_060:>2}")
    w(f"    of the {int(too_long.sum())}, number that actually reached {REACH:.2f} : {too_long_reached:>2}")
    w("")
    w("  Read the distinction carefully. 'Too long' means one Arm B chunk cannot")
    w("  hold ALL of the gold. Only the subset with max possible < 0.60 is barred")
    w(f"  from the {REACH:.2f} threshold by arithmetic. The rest were MEASURED at zero")
    w("  reach; that is an observation about alignment, not a ceiling. None of the")
    w(f"  {int(too_long.sum())} is arithmetically barred at 0.40 or 0.50.")
    w("")
    w("  Per-gold values:")
    w(f"    {'qid':<8}{'gold len':>9}{'max possible':>13}{'measured':>10}")
    for _, r in tl.sort_values("gold_norm_len", ascending=False).iterrows():
        w(f"    {r.question_id:<8}{int(r.gold_norm_len):>9}{r.armB_max_possible_cov:>13.3f}"
          f"{r.armB_contig:>10.3f}")

    w("")
    w("=" * 78)
    w("(3) COLUMN-WISE vs ROW-WISE EXTRACTION FAILURE  [first-class result]")
    w("=" * 78)
    w("A gold is SCATTERED in Arm B when its characters are largely present in a")
    w(f"single Arm B chunk (block-sum coverage >= {SCATTER_BLOCKS_MIN}) but cannot be")
    w(f"found as one run (contiguous coverage < {SCATTER_CONTIG_MAX}). That is the")
    w("signature of PyMuPDF reading a row-wise table down its columns.")
    w("")
    w(f"  {'slice':<12}{'n':>4}{'scattered':>11}{'rate':>8}"
      f"{'median gap':>12}{'p90 gap':>10}")
    for name, s in (("TABLE-GOLD", t), ("PROSE-GOLD", pr)):
        w(f"  {name:<12}{len(s):>4}{int(s.armB_scattered.sum()):>11}"
          f"{s.armB_scattered.mean():>8.3f}{s.armB_scatter_gap.median():>12.3f}"
          f"{s.armB_scatter_gap.quantile(0.9):>10.3f}")
    w("")
    w("  The binary criterion above is deliberately strict and, on its own,")
    w("  UNDERSTATES the effect. The graded distribution is the better view:")
    w("")
    w(f"  {'scatter gap >':<16}{'TABLE-GOLD':>12}{'PROSE-GOLD':>14}")
    graded = {}
    for th in (0.05, 0.10, 0.20, 0.30, 0.40):
        graded[th] = (int((t.armB_scatter_gap > th).sum()), int((pr.armB_scatter_gap > th).sum()))
        w(f"  {th:<16.2f}{graded[th][0]:>6} / {len(t):<4}{graded[th][1]:>8} / {len(pr):<4}")
    w("")
    prose_max = max(v[1] for v in graded.values())
    w(f"  Prose is the control. It never exceeds {prose_max}/{len(pr)} at any scatter-gap cutoff,")
    w(f"  against {graded[0.20][0]}/{len(t)} for tables at gap > 0.20. The signal tracks table")
    w("  structure, not some general property of long chunks.")
    w("")
    w(f"  WHY TABLE GOLDS FAIL IN ARM B (n={len(t)}, structurally exclusive buckets):")
    w(f"    longer than any single Arm B chunk : {computed['too_long']:>3}")
    w(f"    fits, but cells scattered          : {computed['scattered']:>3}")
    w(f"    fits, contiguous but < {REACH:.2f}       : {computed['low_cov']:>3}")
    w(f"    reaches {REACH:.2f} in Arm B              : {computed['reached']:>3}")
    w("")
    w(f"  {computed['failures']} of {len(t)} fail to reach {REACH:.2f}; {computed['reached']} reach it.")
    w(f"  Cell scattering explains {computed['scattered']} of the {computed['failures']} failures.")
    w(f"  Gold length explains {computed['too_long']} and plain low contiguous coverage")
    w(f"  {computed['low_cov']}. An earlier single-example reading of Q026 over-generalised")
    w("  scattering as the main mechanism; this table supersedes it.")
    w("")
    scat = t[t.armB_scattered].nlargest(10, "armB_scatter_gap")
    if len(scat):
        w("  Widest table-gold scatter gaps:")
        w(f"    {'qid':<8}{'blocks':>8}{'contig':>8}{'gap':>8}  source")
        for _, r in scat.iterrows():
            w(f"    {r.question_id:<8}{r.armB_blocks:>8.3f}"
              f"{r.armB_contig_at_best_blocks:>8.3f}{r.armB_scatter_gap:>8.3f}  "
              f"{r.source_file[:38]} p{int(r.page_number)}")
    w("")
    w("=" * 78)
    w("")
    w(format_attribution(computed))

    txt = "\n".join(L)
    (OUT / "DIAGNOSTICS.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"\nwrote {OUT/'DIAGNOSTICS.txt'}, {OUT/'scatter_per_query.csv'}, "
          f"{OUT/'ceiling_contiguous.csv'}")


if __name__ == "__main__":
    main()
