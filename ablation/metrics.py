"""
Score both arms on both pools and write the results.

Relevance is chunking-invariant: a retrieved chunk is relevant to a query when
its normalized text contains the gold span as one CONTIGUOUS run covering at
least t of the span, for t in {0.40, 0.50, 0.60, 0.70, 0.80}. Every metric is
reported at every t. Secondary: union-over-top-10 (each chunk's longest run,
unioned over gold positions). Upper bound: page-match.

Metrics at k=10: Recall@10 (hit), MRR@10, nDCG@10. With one relevant chunk
nDCG@10 is exactly 1/log2(rank+1) - a re-expression of MRR, not independent
evidence. Overlapping windows can make more than one chunk relevant, in which
case IDCG uses the number of relevant chunks in the whole arm.

Paired tests, same queries in both arms: McNemar exact (binomial on discordant
pairs) for Recall@10; paired bootstrap 95% CI (seed 0, 10 000 resamples) for
the MRR and nDCG deltas.

Pools:
  FRESH              headline. n=38 (20 table / 18 prose). ARM-B-BIASED (generated
                     from get_text("text")). Hand-check was model-annotated.
  OLD80_CONTAMINATED computed for the record only. DO NOT REPORT.

Reads:  ablation/out/retrieval_A.json, retrieval_B.json
        ablation/out/fresh_pool_v2.json, anchors.json
        ablation/out/arm_a_chunks.json, arm_b_chunks.json
        ablation/out/power_mde.csv
Writes: ablation/out/RESULTS.md
        ablation/out/results_per_query.csv
        ablation/out/results_summary.csv
"""

import json
import math
import random
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from judgments import THRESHOLDS, normalize, coverage, union_coverage
from notes import ATTRIBUTION, BIAS, PROMPT_ASYMMETRY, STRATUM_DROP

random.seed(0)
np.random.seed(0)
OUT = Path("ablation/out")
K = 10
MODES = [("hybrid", "Full pipeline (RRF + rerank)"), ("dense", "Dense only"), ("sparse", "Sparse (BM25) only")]
N_BOOT = 10_000


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial test on discordant pairs. b = A hit & B miss,
    c = A miss & B hit."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def boot_ci(diffs: np.ndarray, rng: np.random.Generator):
    if len(diffs) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(diffs), size=(N_BOOT, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main():
    ra = json.loads((OUT / "retrieval_A.json").read_text(encoding="utf-8"))
    rb = json.loads((OUT / "retrieval_B.json").read_text(encoding="utf-8"))
    fresh = json.loads((OUT / "fresh_pool_v2.json").read_text(encoding="utf-8"))
    old80 = json.loads((OUT / "anchors.json").read_text(encoding="utf-8"))
    arms = {"A": json.loads((OUT / "arm_a_chunks.json").read_text(encoding="utf-8")),
            "B": json.loads((OUT / "arm_b_chunks.json").read_text(encoding="utf-8"))}
    retr = {"A": ra, "B": rb}

    # All chunks per arm, normalized once. n_rel (for IDCG and reachability)
    # must be counted over the WHOLE arm, not the gold page: a retrieved chunk
    # from another page can partially cover a span (repeated headers, boiler-
    # plate), and if DCG credits it while IDCG does not, nDCG exceeds 1.
    all_norm = {arm: [normalize(c["text"]) for c in chunks] for arm, chunks in arms.items()}

    def corpus_covs(gn: str, norms: list[str]) -> list[float]:
        """Contiguous coverage of gn by every chunk, with an exact prefilter.
        A common substring of length >= 0.4*len(gn) must contain at least one
        aligned probe of length L = floor(0.2*len(gn)) taken at stride L, so a
        chunk containing no probe has coverage < 0.40 and is skipped. Substring
        tests are C-speed; SequenceMatcher runs only on survivors."""
        L = max(4, int(0.2 * len(gn)))
        probes = {gn[i:i + L] for i in range(0, len(gn) - L + 1, L)} | {gn[-L:]}
        out = []
        for cn in norms:
            if any(p in cn for p in probes):
                out.append(coverage(gn, cn))
            else:
                out.append(0.0)
        return out

    queries = []
    for q in fresh:
        queries.append({"pool": "FRESH", "qid": q["question_id"], "gold": q["gold_span"],
                        "stratum": q["gold_stratum"], "file": q["source_file"], "page": q["page_number"]})
    for q in old80:
        queries.append({"pool": "OLD80_CONTAMINATED", "qid": q["question_id"], "gold": q["gold_text"],
                        "stratum": "table" if q["gold_is_table"] else "prose",
                        "file": q["source_file"], "page": q["page_number"]})
    # Only score queries the retrieval run actually has (it ran on the 39-pool;
    # T013 was dropped afterwards and is simply not in `fresh`).
    queries = [q for q in queries if q["qid"] in ra and q["qid"] in rb]

    rows = []
    for q in queries:
        gn = normalize(q["gold"])
        for arm in ("A", "B"):
            cc_ = corpus_covs(gn, all_norm[arm])
            n_rel = {t: sum(1 for c in cc_ if c >= t) for t in THRESHOLDS}
            for mode, _ in MODES:
                hits = retr[arm][q["qid"]][mode][:K]
                covs = [coverage(gn, normalize(h["text"])) for h in hits]
                norms = [normalize(h["text"]) for h in hits]
                page_hit = any(h["source_file"] == q["file"] and h["page_number"] == q["page"] for h in hits)
                union = union_coverage(gn, norms)
                for t in THRESHOLDS:
                    rel = [c >= t for c in covs]
                    first = next((i + 1 for i, r in enumerate(rel) if r), None)
                    dcg = sum(1 / math.log2(i + 2) for i, r in enumerate(rel) if r)
                    idcg = sum(1 / math.log2(i + 2) for i in range(min(n_rel[t], K)))
                    assert dcg <= idcg + 1e-9, (q["qid"], arm, mode, t, dcg, idcg, n_rel[t])
                    rows.append({
                        "pool": q["pool"], "qid": q["qid"], "stratum": q["stratum"], "arm": arm,
                        "mode": mode, "t": t,
                        "recall10": int(first is not None),
                        "mrr10": (1 / first) if first else 0.0,
                        "ndcg10": (dcg / idcg) if idcg > 0 else 0.0,
                        "n_rel_in_arm": n_rel[t],
                        "reachable": int(n_rel[t] > 0),
                        "union10_hit": int(union >= t),
                        "page_hit": int(page_hit),
                        "first_rank": first or 0,
                        "best_cov_top10": max(covs) if covs else 0.0,
                    })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results_per_query.csv", index=False)

    # ── aggregate ────────────────────────────────────────────────────────
    rng = np.random.default_rng(0)
    summ = []
    for pool in ("FRESH", "OLD80_CONTAMINATED"):
        for stratum in ("all", "table", "prose"):
            for mode, _ in MODES:
                for t in THRESHOLDS:
                    sel = df[(df.pool == pool) & (df["mode"] == mode) & (df.t == t)]
                    if stratum != "all":
                        sel = sel[sel.stratum == stratum]
                    a = sel[sel.arm == "A"].set_index("qid").sort_index()
                    b = sel[sel.arm == "B"].set_index("qid").sort_index()
                    if len(a) == 0:
                        continue
                    assert list(a.index) == list(b.index)
                    n = len(a)
                    bb = int(((a.recall10 == 1) & (b.recall10 == 0)).sum())
                    cc = int(((a.recall10 == 0) & (b.recall10 == 1)).sum())
                    d_mrr = (a.mrr10 - b.mrr10).to_numpy()
                    d_ndcg = (a.ndcg10 - b.ndcg10).to_numpy()
                    ci_mrr, ci_ndcg = boot_ci(d_mrr, rng), boot_ci(d_ndcg, rng)
                    summ.append({
                        "pool": pool, "stratum": stratum, "mode": mode, "t": t, "n": n,
                        "A_recall10": a.recall10.mean(), "B_recall10": b.recall10.mean(),
                        "d_recall10": a.recall10.mean() - b.recall10.mean(),
                        "discordant_A_only": bb, "discordant_B_only": cc,
                        "mcnemar_p": mcnemar_exact(bb, cc),
                        "A_mrr10": a.mrr10.mean(), "B_mrr10": b.mrr10.mean(),
                        "d_mrr10": d_mrr.mean(), "d_mrr10_ci_lo": ci_mrr[0], "d_mrr10_ci_hi": ci_mrr[1],
                        "A_ndcg10": a.ndcg10.mean(), "B_ndcg10": b.ndcg10.mean(),
                        "d_ndcg10": d_ndcg.mean(), "d_ndcg10_ci_lo": ci_ndcg[0], "d_ndcg10_ci_hi": ci_ndcg[1],
                        "A_union10": a.union10_hit.mean(), "B_union10": b.union10_hit.mean(),
                        "A_page": a.page_hit.mean(), "B_page": b.page_hit.mean(),
                        "A_reachable": a.reachable.mean(), "B_reachable": b.reachable.mean(),
                    })
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "results_summary.csv", index=False)

    # ── sign stability across thresholds ─────────────────────────────────
    unstable = []
    for (pool, stratum, mode), g in S.groupby(["pool", "stratum", "mode"]):
        for metric in ("d_recall10", "d_mrr10", "d_ndcg10"):
            signs = set(np.sign(g[metric]).astype(int))
            signs.discard(0)
            if len(signs) > 1:
                unstable.append((pool, stratum, mode, metric, [f"{t:.1f}:{v:+.3f}" for t, v in zip(g.t, g[metric])]))

    # ── write RESULTS.md ─────────────────────────────────────────────────
    mde = pd.read_csv(OUT / "power_mde.csv")
    L = []
    w = L.append
    w("# Structure ablation — results")
    w("")
    w("Arm A: current pipeline (find_tables → markdown, page boundaries, 512-char prose / 64 overlap).")
    w("Arm B: flat get_text(\"text\"), no table detection, 92-token window / 12 overlap, page-scoped.")
    w("Identical retrieval stack for both: all-MiniLM-L6-v2 dense + BM25 + RRF(k=60) + ms-marco-MiniLM-L-6-v2 rerank over top 30; k=10.")
    w("Chunk-length parity: Arm A mean/median 414.2/453.0 chars, Arm B 409.5/455.0 (−1.1% / +0.4%).")
    w("")
    fresh_n = df[(df.pool == "FRESH") & (df.arm == "A") & (df["mode"] == "hybrid") & (df.t == 0.6)]
    n_t = int((fresh_n.stratum == "table").sum()); n_p = int((fresh_n.stratum == "prose").sum())
    w(f"**Headline pool: FRESH, n = {len(fresh_n)} ({n_t} table / {n_p} prose).** Read the four notices at the end before quoting anything.")
    w("")
    if unstable:
        w("## ⚠ SIGN INSTABILITY ACROSS THRESHOLDS")
        w("")
        w("The sign of the delta is NOT the same at all five coverage thresholds for:")
        w("")
        for pool, stratum, mode, metric, vals in unstable:
            w(f"- {pool} / {stratum} / {mode} / {metric}: {', '.join(vals)}")
        w("")
        w("Where this occurs the direction of the effect depends on how strictly relevance is defined and must not be reported as a single-signed result.")
        w("")
    else:
        w("Sign of every delta (Recall, MRR, nDCG) is stable across all five thresholds in every pool/stratum/mode cell.")
        w("")

    def table(pool, stratum, mode):
        g = S[(S.pool == pool) & (S.stratum == stratum) & (S["mode"] == mode)].sort_values("t")
        if g.empty:
            return
        n = int(g.n.iloc[0])
        w(f"### {dict(MODES)[mode]} — {stratum} (n = {n})")
        w("")
        w("| t | A R@10 | B R@10 | Δ R@10 | discordant A-only / B-only | McNemar p | A MRR | B MRR | Δ MRR [95% CI] | A nDCG | B nDCG | Δ nDCG [95% CI] |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in g.iterrows():
            w(f"| {r.t:.2f} | {r.A_recall10:.3f} | {r.B_recall10:.3f} | {r.d_recall10:+.3f} | {int(r.discordant_A_only)} / {int(r.discordant_B_only)} | {r.mcnemar_p:.3f} "
              f"| {r.A_mrr10:.3f} | {r.B_mrr10:.3f} | {r.d_mrr10:+.3f} [{r.d_mrr10_ci_lo:+.3f}, {r.d_mrr10_ci_hi:+.3f}] "
              f"| {r.A_ndcg10:.3f} | {r.B_ndcg10:.3f} | {r.d_ndcg10:+.3f} [{r.d_ndcg10_ci_lo:+.3f}, {r.d_ndcg10_ci_hi:+.3f}] |")
        w("")
        r6 = g[g.t == 0.6].iloc[0]
        w(f"Secondary at t=0.60 — union-over-top-10 hit: A {r6.A_union10:.3f}, B {r6.B_union10:.3f}. "
          f"Page-match (upper bound): A {r6.A_page:.3f}, B {r6.B_page:.3f}. "
          f"Reachable (≥1 relevant chunk exists in arm): A {r6.A_reachable:.3f}, B {r6.B_reachable:.3f}.")
        w("")

    w("## FRESH pool (headline)")
    w("")
    for stratum in ("all", "table", "prose"):
        table("FRESH", stratum, "hybrid")
    w("### Diagnostic modes (FRESH, all)")
    w("")
    for mode in ("dense", "sparse"):
        table("FRESH", "all", mode)

    w("## Power")
    w("")
    w("From `power_mde.csv` (McNemar exact, α=0.05 two-sided, 80% power). The detectable difference depends on the discordant rate actually observed above:")
    w("")
    w("| n | disc 10% | 20% | 30% | 40% |")
    w("|---|---|---|---|---|")
    for n_ in (42, 80, 90):
        vals = []
        for dr in (0.1, 0.2, 0.3, 0.4):
            r = mde[(mde.n == n_) & (mde.discordant_rate == dr)]
            vals.append(f"{r.mde_recall_points.iloc[0]:.3f}" if len(r) and not pd.isna(r.mde_recall_points.iloc[0]) else "no region")
        w(f"| {n_} | " + " | ".join(vals) + " |")
    w("")
    w(f"The FRESH pool is n={len(fresh_n)}, below the smallest row. At n≈40 with 20% discordance the MDE is about 0.19 Recall points; a delta smaller than that is not detectable regardless of sign.")
    w("")

    w("## OLD80 — CONTAMINATED - DO NOT REPORT")
    w("")
    w("Computed for the record. Questions were written from Arm A chunks, gold IS an Arm A chunk (Arm A ceiling 1.0 by construction), and 12/40 table golds are longer than any Arm B chunk. Any Arm A advantage here is largely built in.")
    w("")
    for stratum in ("all", "table", "prose"):
        table("OLD80_CONTAMINATED", stratum, "hybrid")

    w("## Notices")
    w("")
    w("```")
    w(BIAS); w(""); w(PROMPT_ASYMMETRY); w(""); w(STRATUM_DROP); w(""); w(ATTRIBUTION)
    w("```")
    w("")
    w("HAND-CHECK PROVENANCE: the 20-item hand-check (HANDCHECK_20.txt) was model-annotated by the AI assistant at the author's instruction on 2026-09-20 — not a human review. 19 accepted, 1 rejected (T013).")
    w("")
    w("nDCG@10 NOTE: with a single relevant chunk nDCG@10 = 1/log2(rank+1), a re-expression of the same rank MRR uses. It is not independent evidence. Overlapping windows can make >1 chunk relevant, in which case IDCG uses the count of relevant chunks in that arm.")
    w("")
    w("Files: results_per_query.csv (every query × arm × mode × threshold), results_summary.csv (every aggregate cell).")

    (OUT / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n... wrote {OUT/'RESULTS.md'}, results_per_query.csv ({len(df)} rows), results_summary.csv ({len(S)} rows)")


if __name__ == "__main__":
    main()
