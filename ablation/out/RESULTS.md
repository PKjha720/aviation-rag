# Structure ablation — results

Arm A: current pipeline (find_tables → markdown, page boundaries, 512-char prose / 64 overlap).
Arm B: flat get_text("text"), no table detection, 92-token window / 12 overlap, page-scoped.
Identical retrieval stack for both: all-MiniLM-L6-v2 dense + BM25 + RRF(k=60) + ms-marco-MiniLM-L-6-v2 rerank over top 30; k=10.
Chunk-length parity: Arm A mean/median 414.2/453.0 chars, Arm B 409.5/455.0 (−1.1% / +0.4%).

**Headline pool: FRESH, n = 33 (16 table / 17 prose).** Read the four notices at the end before quoting anything.

Sign of every delta (Recall, MRR, nDCG) is stable across all five thresholds in every pool/stratum/mode cell.

## FRESH pool (headline)

### Full pipeline (RRF + rerank) — all (n = 33)

| t | A R@10 | B R@10 | Δ R@10 | discordant A-only / B-only | McNemar p | A MRR | B MRR | Δ MRR [95% CI] | A nDCG | B nDCG | Δ nDCG [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.40 | 0.818 | 0.848 | -0.030 | 2 / 3 | 1.000 | 0.675 | 0.765 | -0.091 [-0.216, +0.017] | 0.634 | 0.665 | -0.031 [-0.135, +0.059] |
| 0.50 | 0.818 | 0.848 | -0.030 | 2 / 3 | 1.000 | 0.652 | 0.727 | -0.075 [-0.203, +0.040] | 0.647 | 0.678 | -0.031 [-0.143, +0.064] |
| 0.60 | 0.818 | 0.848 | -0.030 | 2 / 3 | 1.000 | 0.652 | 0.727 | -0.075 [-0.199, +0.040] | 0.655 | 0.692 | -0.036 [-0.148, +0.064] |
| 0.70 | 0.818 | 0.818 | +0.000 | 3 / 3 | 1.000 | 0.652 | 0.695 | -0.044 [-0.170, +0.067] | 0.675 | 0.709 | -0.034 [-0.162, +0.080] |
| 0.80 | 0.818 | 0.818 | +0.000 | 3 / 3 | 1.000 | 0.652 | 0.695 | -0.044 [-0.171, +0.067] | 0.672 | 0.709 | -0.036 [-0.163, +0.079] |

Secondary at t=0.60 — union-over-top-10 hit: A 0.909, B 0.879. Page-match (upper bound): A 0.909, B 1.000. Reachable (≥1 relevant chunk exists in arm): A 1.000, B 1.000.

### Full pipeline (RRF + rerank) — table (n = 16)

| t | A R@10 | B R@10 | Δ R@10 | discordant A-only / B-only | McNemar p | A MRR | B MRR | Δ MRR [95% CI] | A nDCG | B nDCG | Δ nDCG [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.40 | 0.625 | 0.688 | -0.062 | 2 / 3 | 1.000 | 0.414 | 0.609 | -0.196 [-0.417, +0.010] | 0.416 | 0.500 | -0.084 [-0.292, +0.099] |
| 0.50 | 0.625 | 0.688 | -0.062 | 2 / 3 | 1.000 | 0.367 | 0.531 | -0.165 [-0.406, +0.054] | 0.411 | 0.483 | -0.073 [-0.288, +0.113] |
| 0.60 | 0.625 | 0.688 | -0.062 | 2 / 3 | 1.000 | 0.367 | 0.531 | -0.165 [-0.396, +0.058] | 0.411 | 0.504 | -0.094 [-0.312, +0.101] |
| 0.70 | 0.625 | 0.625 | +0.000 | 3 / 3 | 1.000 | 0.367 | 0.466 | -0.099 [-0.335, +0.117] | 0.426 | 0.506 | -0.080 [-0.330, +0.147] |
| 0.80 | 0.625 | 0.625 | +0.000 | 3 / 3 | 1.000 | 0.367 | 0.466 | -0.099 [-0.345, +0.113] | 0.426 | 0.506 | -0.080 [-0.328, +0.141] |

Secondary at t=0.60 — union-over-top-10 hit: A 0.812, B 0.750. Page-match (upper bound): A 0.812, B 1.000. Reachable (≥1 relevant chunk exists in arm): A 1.000, B 1.000.

### Full pipeline (RRF + rerank) — prose (n = 17)

| t | A R@10 | B R@10 | Δ R@10 | discordant A-only / B-only | McNemar p | A MRR | B MRR | Δ MRR [95% CI] | A nDCG | B nDCG | Δ nDCG [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.40 | 1.000 | 1.000 | +0.000 | 0 / 0 | 1.000 | 0.920 | 0.912 | +0.008 [-0.063, +0.088] | 0.839 | 0.820 | +0.019 [+0.000, +0.050] |
| 0.50 | 1.000 | 1.000 | +0.000 | 0 / 0 | 1.000 | 0.920 | 0.912 | +0.008 [-0.063, +0.088] | 0.870 | 0.862 | +0.008 [-0.045, +0.069] |
| 0.60 | 1.000 | 1.000 | +0.000 | 0 / 0 | 1.000 | 0.920 | 0.912 | +0.008 [-0.063, +0.088] | 0.886 | 0.868 | +0.018 [-0.039, +0.078] |
| 0.70 | 1.000 | 1.000 | +0.000 | 0 / 0 | 1.000 | 0.920 | 0.912 | +0.008 [-0.063, +0.088] | 0.909 | 0.899 | +0.009 [-0.048, +0.070] |
| 0.80 | 1.000 | 1.000 | +0.000 | 0 / 0 | 1.000 | 0.920 | 0.912 | +0.008 [-0.063, +0.088] | 0.904 | 0.899 | +0.004 [-0.052, +0.065] |

Secondary at t=0.60 — union-over-top-10 hit: A 1.000, B 1.000. Page-match (upper bound): A 1.000, B 1.000. Reachable (≥1 relevant chunk exists in arm): A 1.000, B 1.000.

### Diagnostic modes (FRESH, all)

### Dense only — all (n = 33)

| t | A R@10 | B R@10 | Δ R@10 | discordant A-only / B-only | McNemar p | A MRR | B MRR | Δ MRR [95% CI] | A nDCG | B nDCG | Δ nDCG [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.40 | 0.758 | 0.818 | -0.061 | 2 / 4 | 0.688 | 0.625 | 0.732 | -0.108 [-0.256, +0.035] | 0.575 | 0.622 | -0.047 [-0.180, +0.080] |
| 0.50 | 0.727 | 0.788 | -0.061 | 3 / 5 | 0.727 | 0.594 | 0.689 | -0.095 [-0.252, +0.056] | 0.588 | 0.634 | -0.046 [-0.182, +0.086] |
| 0.60 | 0.727 | 0.788 | -0.061 | 3 / 5 | 0.727 | 0.594 | 0.689 | -0.095 [-0.252, +0.056] | 0.593 | 0.647 | -0.053 [-0.194, +0.083] |
| 0.70 | 0.727 | 0.758 | -0.030 | 4 / 5 | 1.000 | 0.594 | 0.658 | -0.063 [-0.223, +0.087] | 0.608 | 0.663 | -0.056 [-0.209, +0.088] |
| 0.80 | 0.727 | 0.758 | -0.030 | 4 / 5 | 1.000 | 0.594 | 0.658 | -0.063 [-0.225, +0.090] | 0.605 | 0.663 | -0.058 [-0.211, +0.089] |

Secondary at t=0.60 — union-over-top-10 hit: A 0.818, B 0.848. Page-match (upper bound): A 0.848, B 1.000. Reachable (≥1 relevant chunk exists in arm): A 1.000, B 1.000.

### Sparse (BM25) only — all (n = 33)

| t | A R@10 | B R@10 | Δ R@10 | discordant A-only / B-only | McNemar p | A MRR | B MRR | Δ MRR [95% CI] | A nDCG | B nDCG | Δ nDCG [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.40 | 0.818 | 0.848 | -0.030 | 2 / 3 | 1.000 | 0.683 | 0.765 | -0.082 [-0.205, +0.025] | 0.637 | 0.667 | -0.030 [-0.136, +0.061] |
| 0.50 | 0.818 | 0.848 | -0.030 | 2 / 3 | 1.000 | 0.660 | 0.727 | -0.067 [-0.192, +0.045] | 0.651 | 0.678 | -0.028 [-0.138, +0.069] |
| 0.60 | 0.818 | 0.848 | -0.030 | 2 / 3 | 1.000 | 0.660 | 0.727 | -0.067 [-0.192, +0.045] | 0.658 | 0.692 | -0.033 [-0.148, +0.067] |
| 0.70 | 0.818 | 0.818 | +0.000 | 3 / 3 | 1.000 | 0.660 | 0.695 | -0.035 [-0.163, +0.078] | 0.679 | 0.708 | -0.030 [-0.157, +0.085] |
| 0.80 | 0.818 | 0.818 | +0.000 | 3 / 3 | 1.000 | 0.660 | 0.695 | -0.035 [-0.167, +0.074] | 0.676 | 0.708 | -0.032 [-0.163, +0.082] |

Secondary at t=0.60 — union-over-top-10 hit: A 0.848, B 0.909. Page-match (upper bound): A 0.879, B 1.000. Reachable (≥1 relevant chunk exists in arm): A 1.000, B 1.000.

## Power

From `power_mde.csv` (McNemar exact, α=0.05 two-sided, 80% power). The detectable difference depends on the discordant rate actually observed above:

| n | disc 10% | 20% | 30% | 40% |
|---|---|---|---|---|
| 42 | no region | 0.189 | 0.228 | 0.250 |
| 80 | 0.095 | 0.141 | 0.178 | 0.213 |
| 90 | 0.082 | 0.130 | 0.173 | 0.194 |

The FRESH pool is n=33, below the smallest row. At n≈40 with 20% discordance the MDE is about 0.19 Recall points; a delta smaller than that is not detectable regardless of sign.

## OLD80 — CONTAMINATED - DO NOT REPORT

Computed for the record. Questions were written from Arm A chunks, gold IS an Arm A chunk (Arm A ceiling 1.0 by construction), and 12/40 table golds are longer than any Arm B chunk. Any Arm A advantage here is largely built in.

### Full pipeline (RRF + rerank) — all (n = 80)

| t | A R@10 | B R@10 | Δ R@10 | discordant A-only / B-only | McNemar p | A MRR | B MRR | Δ MRR [95% CI] | A nDCG | B nDCG | Δ nDCG [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.40 | 0.863 | 0.625 | +0.238 | 20 / 1 | 0.000 | 0.744 | 0.567 | +0.177 [+0.088, +0.270] | 0.749 | 0.514 | +0.235 [+0.152, +0.323] |
| 0.50 | 0.863 | 0.537 | +0.325 | 27 / 1 | 0.000 | 0.734 | 0.490 | +0.244 [+0.143, +0.350] | 0.762 | 0.500 | +0.262 [+0.165, +0.361] |
| 0.60 | 0.863 | 0.525 | +0.338 | 28 / 1 | 0.000 | 0.734 | 0.471 | +0.263 [+0.159, +0.367] | 0.763 | 0.484 | +0.278 [+0.180, +0.377] |
| 0.70 | 0.863 | 0.412 | +0.450 | 37 / 1 | 0.000 | 0.734 | 0.381 | +0.354 [+0.245, +0.463] | 0.765 | 0.388 | +0.377 [+0.269, +0.484] |
| 0.80 | 0.863 | 0.275 | +0.588 | 47 / 0 | 0.000 | 0.734 | 0.243 | +0.491 [+0.389, +0.597] | 0.765 | 0.251 | +0.514 [+0.412, +0.617] |

Secondary at t=0.60 — union-over-top-10 hit: A 0.863, B 0.662. Page-match (upper bound): A 0.912, B 0.912. Reachable (≥1 relevant chunk exists in arm): A 1.000, B 0.650.

### Full pipeline (RRF + rerank) — table (n = 40)

| t | A R@10 | B R@10 | Δ R@10 | discordant A-only / B-only | McNemar p | A MRR | B MRR | Δ MRR [95% CI] | A nDCG | B nDCG | Δ nDCG [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.40 | 0.800 | 0.325 | +0.475 | 19 / 0 | 0.000 | 0.575 | 0.274 | +0.300 [+0.155, +0.448] | 0.602 | 0.241 | +0.361 [+0.232, +0.497] |
| 0.50 | 0.800 | 0.200 | +0.600 | 24 / 0 | 0.000 | 0.575 | 0.183 | +0.391 [+0.241, +0.540] | 0.622 | 0.188 | +0.435 [+0.293, +0.577] |
| 0.60 | 0.800 | 0.175 | +0.625 | 25 / 0 | 0.000 | 0.575 | 0.158 | +0.416 [+0.267, +0.562] | 0.624 | 0.163 | +0.461 [+0.321, +0.600] |
| 0.70 | 0.800 | 0.125 | +0.675 | 27 / 0 | 0.000 | 0.575 | 0.108 | +0.466 [+0.325, +0.608] | 0.629 | 0.113 | +0.516 [+0.382, +0.651] |
| 0.80 | 0.800 | 0.100 | +0.700 | 28 / 0 | 0.000 | 0.575 | 0.083 | +0.491 [+0.353, +0.632] | 0.629 | 0.087 | +0.541 [+0.404, +0.670] |

Secondary at t=0.60 — union-over-top-10 hit: A 0.800, B 0.425. Page-match (upper bound): A 0.875, B 0.850. Reachable (≥1 relevant chunk exists in arm): A 1.000, B 0.325.

### Full pipeline (RRF + rerank) — prose (n = 40)

| t | A R@10 | B R@10 | Δ R@10 | discordant A-only / B-only | McNemar p | A MRR | B MRR | Δ MRR [95% CI] | A nDCG | B nDCG | Δ nDCG [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.40 | 0.925 | 0.925 | +0.000 | 1 / 1 | 1.000 | 0.912 | 0.859 | +0.053 [-0.044, +0.153] | 0.896 | 0.787 | +0.109 [+0.028, +0.196] |
| 0.50 | 0.925 | 0.875 | +0.050 | 3 / 1 | 0.625 | 0.894 | 0.797 | +0.097 [-0.025, +0.222] | 0.902 | 0.813 | +0.089 [-0.019, +0.204] |
| 0.60 | 0.925 | 0.875 | +0.050 | 3 / 1 | 0.625 | 0.894 | 0.784 | +0.110 [-0.013, +0.238] | 0.902 | 0.806 | +0.095 [-0.018, +0.210] |
| 0.70 | 0.925 | 0.700 | +0.225 | 10 / 1 | 0.012 | 0.894 | 0.653 | +0.241 [+0.081, +0.400] | 0.902 | 0.664 | +0.237 [+0.086, +0.394] |
| 0.80 | 0.925 | 0.450 | +0.475 | 19 / 0 | 0.000 | 0.894 | 0.403 | +0.491 [+0.338, +0.644] | 0.902 | 0.414 | +0.487 [+0.340, +0.643] |

Secondary at t=0.60 — union-over-top-10 hit: A 0.925, B 0.900. Page-match (upper bound): A 0.950, B 0.975. Reachable (≥1 relevant chunk exists in arm): A 1.000, B 0.975.

## Notices

```
BIAS LABEL — FRESH POOL
  Questions are generated from page.get_text("text"), which IS ARM B's
  extraction path. This pool is BIASED TOWARD ARM B. It is not neutral and must
  never be called unbiased or bias-free. The bias runs against the expected
  effect, so a surviving Arm A advantage is a lower bound, not an estimate. An
  Arm B advantage measured on this pool would be uninterpretable.

PROMPT ASYMMETRY — TABLE vs PROSE STRATUM
  The table stratum was regenerated with a prompt requiring the span to carry a
  row LABEL plus its VALUE, because the original generic prompt returned bare
  cell values (median 22 normalized chars, from replay_validation.py console
  output) that fell under the 40-char floor. The prose stratum keeps the
  original generic prompt, unchanged and un-rerun. Both strata see only raw flat
  page text, and both pass identical deterministic validation. Compare strata
  with this difference in mind.

STRATUM LABELS AND THE THREE DROPPED ITEMS
  page_stratum : which sampling frame the page came from (table-bearing page or
                 prose-only page). Provenance only.
  gold_stratum : whether the gold span sits inside an Arm A chunk that ingest
                 flagged is_table. This is the label every table-vs-prose
                 comparison uses.
  Three items sampled from table-bearing pages (T001, T009, T020) had prose
  golds. They are DROPPED from the pool. An earlier revision moved them into
  the prose stratum instead; that decision is RETRACTED, because it made the
  prose stratum a mix of two generator prompts and reintroduced exactly the
  asymmetry the strata are meant to avoid. Every remaining item has
  page_stratum == gold_stratum.

MECHANISM ATTRIBUTION — REQUIRED IN ANY SUMMARY OF THIS EXPERIMENT
  Of 40 table golds, 27 fail to reach 0.60 contiguous
  coverage in Arm B and 13 reach it. The 40 split into
  mutually exclusive categories:
      gold longer than any single Arm B chunk ... 12
          (of these, 7 are arithmetically barred from 0.60;
           the other 5 could reach it and were
           measured not to)
      fits, contiguous coverage < 0.60 ......... 13
      fits, but cells scattered ................  2
      reaches 0.60 in Arm B .................... 13
  Cell scattering explains 2 of the 27 failures. Gold length
  explains 12 and plain low coverage 13. Scattering is REAL
  (prose control: 1/40 at scatter gap > 0.20 against
  10/40 for tables) but it is NOT the mechanism. Do not
  describe it as the mechanism.
```

HAND-CHECK PROVENANCE: the 20-item hand-check (HANDCHECK_20.txt) was model-annotated by the AI assistant at the author's instruction on 2026-09-20 — not a human review. 19 accepted, 1 rejected (T013).

nDCG@10 NOTE: with a single relevant chunk nDCG@10 = 1/log2(rank+1), a re-expression of the same rank MRR uses. It is not independent evidence. Overlapping windows can make >1 chunk relevant, in which case IDCG uses the count of relevant chunks in that arm.

Files: results_per_query.csv (every query × arm × mode × threshold), results_summary.csv (every aggregate cell).
