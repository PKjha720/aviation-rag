"""
First-class diagnostics, independent of any retrieval delta.

(1) The old 80-question pool, recomputed with the CONTIGUOUS criterion and
    marked CONTAMINATED - DO NOT REPORT, with the three reasons inline.
(2) Gold-length reachability: can a single chunk in each arm physically hold
    the gold at all?
(3) Column-wise vs row-wise extraction failure, per query: how many table golds
    have their content present on the page but broken into scattered fragments
    in Arm B.

These stand on their own. They are properties of ingestion, measured before any
query is issued, and they do not depend on which retriever is used.

Writes: ablation/out/DIAGNOSTICS.txt
        ablation/out/scatter_per_query.csv
        ablation/out/ceiling_contiguous.csv
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from judgments import (THRESHOLDS, normalize, coverage, coverage_blocks,
                       union_coverage)

random.seed(0)
np.random.seed(0)

OUT = Path("ablation/out")

# A gold is "scattered" when its characters are largely present in a chunk but
# cannot be found as one run.
SCATTER_BLOCKS_MIN = 0.60
SCATTER_CONTIG_MAX = 0.35

CONTAMINATION_NOTICE = """\
################################################################################
#  OLD 80-QUESTION POOL  —  CONTAMINATED - DO NOT REPORT                       #
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
#  3. GOLD LENGTHS COME FROM A TAIL ARM B CANNOT REACH. Arm A chunks run to    #
#     1375 normalized chars (chunk_table allows 1500); Arm B is bounded at     #
#     550 by the 92-token window. Golds longer than any Arm B chunk cannot be  #
#     covered by Arm B at any threshold. That is arithmetic, not retrieval.    #
################################################################################
"""


def load():
    arm_a = json.loads((OUT / "arm_a_chunks.json").read_text(encoding="utf-8"))
    arm_b = json.loads((OUT / "arm_b_chunks.json").read_text(encoding="utf-8"))
    anchors = json.loads((OUT / "anchors.json").read_text(encoding="utf-8"))
    return arm_a, arm_b, anchors


def by_page(chunks):
    d = defaultdict(list)
    for c in chunks:
        d[(c["source_file"], c["page_number"])].append(c)
    return d


def main():
    arm_a, arm_b, anchors = load()
    a_pages, b_pages = by_page(arm_a), by_page(arm_b)

    a_norm_len = np.array([len(normalize(c["text"])) for c in arm_a])
    b_norm_len = np.array([len(normalize(c["text"])) for c in arm_b])
    b_max = int(b_norm_len.max())
    a_max = int(a_norm_len.max())

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
        r["armB_scatter_gap"] = round(r.get("armB_blocks", 0) - r.get("armB_contig_at_best_blocks", 0), 4)
        r["armB_scattered"] = bool(
            r.get("armB_blocks", 0) >= SCATTER_BLOCKS_MIN
            and r.get("armB_contig_at_best_blocks", 1) < SCATTER_CONTIG_MAX)
        r["fits_in_armB"] = len(gn) <= b_max
        r["fits_in_armA"] = len(gn) <= a_max
        rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ceiling_contiguous.csv", index=False)
    df[["question_id", "gold_is_table", "gold_norm_len", "armB_blocks",
        "armB_contig_at_best_blocks", "armB_scatter_gap", "armB_scattered",
        "source_file", "page_number"]].to_csv(OUT / "scatter_per_query.csv", index=False)

    L = []
    w = L.append
    w(CONTAMINATION_NOTICE)
    w("")
    w("=" * 78)
    w("(1) CEILING, RECOMPUTED WITH THE CONTIGUOUS CRITERION")
    w("=" * 78)
    w("Supersedes the earlier block-sum numbers. Longest contiguous run only.")
    w("Arm A per-chunk is 1.000 by construction - see defect 2 above.")
    w("")
    for name, mask in (("ALL", pd.Series(True, index=df.index)),
                       ("TABLE-GOLD", df.gold_is_table),
                       ("PROSE-GOLD", ~df.gold_is_table)):
        s = df[mask]
        w(f"  {name} (n={len(s)})")
        w(f"    {'thresh':>7} | {'A per-chunk':>12} {'B per-chunk':>12} | {'B union/page':>13}")
        for t in THRESHOLDS:
            w(f"    {t:>7.2f} | {(s.armA_contig >= t).mean():>12.3f} "
              f"{(s.armB_contig >= t).mean():>12.3f} | "
              f"{(s.armB_union_contig >= t).mean():>13.3f}")
        w(f"    median B per-chunk contiguous coverage: {s.armB_contig.median():.3f}")
        w("")

    w("=" * 78)
    w("(2) GOLD-LENGTH REACHABILITY  [first-class result]")
    w("=" * 78)
    w("Can one chunk in the arm physically hold the gold, before retrieval?")
    w("")
    w(f"  longest chunk, Arm A : {a_max} normalized chars  (chunk_table cap 1500)")
    w(f"  longest chunk, Arm B : {b_max} normalized chars  (bounded by 92-token window)")
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
    w("  Arm B per-chunk coverage, split by whether the gold fits in ANY Arm B chunk:")
    for label, mask in (("gold FITS Arm B", df.fits_in_armB),
                        ("gold TOO LONG for Arm B", ~df.fits_in_armB)):
        s = df[mask]
        if len(s) == 0:
            w(f"    {label:<26} n=0")
            continue
        w(f"    {label:<26} n={len(s):>3}  median contig={s.armB_contig.median():.3f}"
          f"  reach 0.60: {(s.armB_contig >= 0.60).mean():.3f}")

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
    for name, mask in (("TABLE-GOLD", df.gold_is_table),
                       ("PROSE-GOLD", ~df.gold_is_table)):
        s = df[mask]
        w(f"  {name:<12}{len(s):>4}{int(s.armB_scattered.sum()):>11}"
          f"{s.armB_scattered.mean():>8.3f}{s.armB_scatter_gap.median():>12.3f}"
          f"{s.armB_scatter_gap.quantile(0.9):>10.3f}")
    w("")
    w("  The binary criterion above is deliberately strict and, on its own,")
    w("  UNDERSTATES the effect. The graded distribution is the better view:")
    w("")
    w(f"  {'scatter gap >':<16}{'TABLE-GOLD':>12}{'PROSE-GOLD':>14}")
    for th in (0.05, 0.10, 0.20, 0.30, 0.40):
        t = df[df.gold_is_table]
        pr = df[~df.gold_is_table]
        w(f"  {th:<16.2f}{int((t.armB_scatter_gap > th).sum()):>6} / {len(t):<4}"
          f"{int((pr.armB_scatter_gap > th).sum()):>8} / {len(pr):<4}")
    w("")
    w("  Prose is the control, and it is clean: 1/40 at every threshold against")
    w("  10/40 for tables at gap > 0.20. The signal tracks table structure, not")
    w("  some general property of long chunks.")
    w("")
    w("  WHY TABLE GOLDS FAIL IN ARM B (n=40, mutually exclusive):")
    t = df[df.gold_is_table]
    too_long = (~t.fits_in_armB)
    scattered = t.armB_scattered & t.fits_in_armB
    lowcov = (t.armB_contig < 0.60) & t.fits_in_armB & ~t.armB_scattered
    reached = (t.armB_contig >= 0.60)
    w(f"    too long for any Arm B chunk : {int(too_long.sum()):>3}")
    w(f"    fits, but cells scattered    : {int(scattered.sum()):>3}")
    w(f"    fits, contiguous but < 0.60  : {int(lowcov.sum()):>3}")
    w(f"    reaches 0.60 in Arm B        : {int(reached.sum()):>3}")
    w("")
    w("  Read this carefully before describing the mechanism. Cell scattering is")
    w("  REAL but is NOT the dominant failure: gold length (12) and plain low")
    w("  contiguous coverage (13) each account for more than scattering (2) under")
    w("  the strict definition. An earlier single-example reading of Q026")
    w("  over-generalised scattering as the main mechanism; this table supersedes")
    w("  it.")
    w("")
    scat = df[df.gold_is_table & df.armB_scattered].nlargest(10, "armB_scatter_gap")
    if len(scat):
        w("  Widest table-gold scatter gaps:")
        w(f"    {'qid':<8}{'blocks':>8}{'contig':>8}{'gap':>8}  source")
        for _, r in scat.iterrows():
            w(f"    {r.question_id:<8}{r.armB_blocks:>8.3f}"
              f"{r.armB_contig_at_best_blocks:>8.3f}{r.armB_scatter_gap:>8.3f}  "
              f"{r.source_file[:38]} p{int(r.page_number)}")
    w("")
    w("=" * 78)

    txt = "\n".join(L)
    (OUT / "DIAGNOSTICS.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"\nwrote {OUT/'DIAGNOSTICS.txt'}, {OUT/'scatter_per_query.csv'}, "
          f"{OUT/'ceiling_contiguous.csv'}")


if __name__ == "__main__":
    main()
