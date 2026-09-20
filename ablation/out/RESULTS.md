# Structure ablation — results

Arm A: current pipeline (find_tables → markdown, page boundaries, 512-char prose / 64 overlap).
Arm B: flat get_text("text"), no table detection, 92-token window / 12 overlap, page-scoped.
Identical retrieval stack for both: all-MiniLM-L6-v2 dense + BM25 + RRF(k=60) + ms-marco-MiniLM-L-6-v2 rerank over top 30; k=10.
Chunk-length parity: Arm A mean/median 414.2/453.0 chars, Arm B 409.5/455.0 (−1.1% / +0.4%).

**Headline pool: FRESH, n = 38 (20 table / 18 prose).** Read the four notices at the end before quoting anything.

Sign of every delta (Recall, MRR, nDCG) is stable across all five thresholds in every pool/stratum/mode cell.

## FRESH pool (headline)

### Full pipeline (RRF + rerank) — all (n = 38)

| t | A R@10 | B R@10 | Δ R@10 | discordant A-only / B-only | McNemar p | A MRR | B MRR | Δ MRR [95% CI] | A nDCG | B nDCG | Δ nDCG [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.40 | 0.816 | 0.842 | -0.026 | 2 / 3 | 1.000 | 0.691 | 0.750 | -0.059 [-0.173, +0.044] | 0.637 | 0.651 | -0.014 [-0.109, +0.069] |
| 0.50 | 0.789 | 0.842 | -0.053 | 2 / 4 | 0.688 | 0.645 | 0.717 | -0.072 [-0.204, +0.047] | 0.641 | 0.658 | -0.017 [-0.120, +0.073] |
| 0.60 | 0.789 | 0.842 | -0.053 | 2 / 4 | 0.688 | 0.645 | 0.717 | -0.072 [-0.197, +0.049] | 0.648 | 0.691 | -0.043 [-0.159, +0.062] |
| 0.70 | 0.789 | 0.816 | -0.026 | 3 / 4 | 1.000 | 0.645 | 0.689 | -0.044 [-0.169, +0.072] | 0.665 | 0.706 | -0.041 [-0.168, +0.073] |
| 0.80 | 0.789 | 0.816 | -0.026 | 3 / 4 | 1.000 | 0.645 | 0.689 | -0.044 [-0.171, +0.071] | 0.663 | 0.706 | -0.043 [-0.169, +0.070] |

Secondary at t=0.60 — union-over-top-10 hit: A 0.895, B 0.895. Page-match (upper bound): A 0.895, B 1.000. Reachable (≥1 relevant chunk exists in arm): A 1.000, B 1.000.

### Full pipeline (RRF + rerank) — table (n = 20)

| t | A R@10 | B R@10 | Δ R@10 | discordant A-only / B-only | McNemar p | A MRR | B MRR | Δ MRR [95% CI] | A nDCG | B nDCG | Δ nDCG [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.40 | 0.650 | 0.700 | -0.050 | 2 / 3 | 1.000 | 0.481 | 0.600 | -0.119 [-0.321, +0.074] | 0.447 | 0.491 | -0.043 [-0.220, +0.118] |
| 0.50 | 0.600 | 0.700 | -0.100 | 2 / 4 | 0.688 | 0.393 | 0.537 | -0.144 [-0.375, +0.073] | 0.429 | 0.468 | -0.039 [-0.227, +0.129] |
| 0.60 | 0.600 | 0.700 | -0.100 | 2 / 4 | 0.688 | 0.393 | 0.537 | -0.144 [-0.379, +0.064] | 0.429 | 0.525 | -0.096 [-0.302, +0.095] |
| 0.70 | 0.600 | 0.650 | -0.050 | 3 / 4 | 1.000 | 0.393 | 0.485 | -0.092 [-0.325, +0.119] | 0.441 | 0.526 | -0.085 [-0.319, +0.127] |
| 0.80 | 0.600 | 0.650 | -0.050 | 3 / 4 | 1.000 | 0.393 | 0.485 | -0.092 [-0.317, +0.122] | 0.441 | 0.526 | -0.085 [-0.322, +0.126] |

Secondary at t=0.60 — union-over-top-10 hit: A 0.800, B 0.800. Page-match (upper bound): A 0.800, B 1.000. Reachable (≥1 relevant chunk exists in arm): A 1.000, B 1.000.

### Full pipeline (RRF + rerank) — prose (n = 18)

| t | A R@10 | B R@10 | Δ R@10 | discordant A-only / B-only | McNemar p | A MRR | B MRR | Δ MRR [95% CI] | A nDCG | B nDCG | Δ nDCG [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.40 | 1.000 | 1.000 | +0.000 | 0 / 0 | 1.000 | 0.925 | 0.917 | +0.008 [-0.060, +0.083] | 0.848 | 0.830 | +0.018 [+0.000, +0.047] |
| 0.50 | 1.000 | 1.000 | +0.000 | 0 / 0 | 1.000 | 0.925 | 0.917 | +0.008 [-0.060, +0.083] | 0.877 | 0.870 | +0.007 [-0.046, +0.065] |
| 0.60 | 1.000 | 1.000 | +0.000 | 0 / 0 | 1.000 | 0.925 | 0.917 | +0.008 [-0.060, +0.083] | 0.892 | 0.876 | +0.017 [-0.034, +0.074] |
| 0.70 | 1.000 | 1.000 | +0.000 | 0 / 0 | 1.000 | 0.925 | 0.917 | +0.008 [-0.060, +0.083] | 0.914 | 0.905 | +0.009 [-0.045, +0.066] |
| 0.80 | 1.000 | 1.000 | +0.000 | 0 / 0 | 1.000 | 0.925 | 0.917 | +0.008 [-0.060, +0.083] | 0.909 | 0.905 | +0.004 [-0.049, +0.062] |

Secondary at t=0.60 — union-over-top-10 hit: A 1.000, B 1.000. Page-match (upper bound): A 1.000, B 1.000. Reachable (≥1 relevant chunk exists in arm): A 1.000, B 1.000.

### Diagnostic modes (FRESH, all)

### Dense only — all (n = 38)

| t | A R@10 | B R@10 | Δ R@10 | discordant A-only / B-only | McNemar p | A MRR | B MRR | Δ MRR [95% CI] | A nDCG | B nDCG | Δ nDCG [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.40 | 0.737 | 0.789 | -0.053 | 2 / 4 | 0.688 | 0.599 | 0.695 | -0.096 [-0.240, +0.041] | 0.544 | 0.580 | -0.036 [-0.156, +0.079] |
| 0.50 | 0.684 | 0.763 | -0.079 | 3 / 6 | 0.508 | 0.569 | 0.658 | -0.089 [-0.243, +0.056] | 0.553 | 0.594 | -0.040 [-0.169, +0.079] |
| 0.60 | 0.684 | 0.763 | -0.079 | 3 / 6 | 0.508 | 0.569 | 0.658 | -0.089 [-0.239, +0.057] | 0.568 | 0.626 | -0.058 [-0.198, +0.070] |
| 0.70 | 0.684 | 0.737 | -0.053 | 4 / 6 | 0.754 | 0.569 | 0.630 | -0.061 [-0.215, +0.087] | 0.580 | 0.640 | -0.060 [-0.209, +0.080] |
| 0.80 | 0.684 | 0.737 | -0.053 | 4 / 6 | 0.754 | 0.569 | 0.630 | -0.061 [-0.218, +0.085] | 0.578 | 0.640 | -0.062 [-0.206, +0.076] |

Secondary at t=0.60 — union-over-top-10 hit: A 0.763, B 0.842. Page-match (upper bound): A 0.816, B 0.974. Reachable (≥1 relevant chunk exists in arm): A 1.000, B 1.000.

### Sparse (BM25) only — all (n = 38)

| t | A R@10 | B R@10 | Δ R@10 | discordant A-only / B-only | McNemar p | A MRR | B MRR | Δ MRR [95% CI] | A nDCG | B nDCG | Δ nDCG [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.40 | 0.816 | 0.816 | +0.000 | 3 / 3 | 1.000 | 0.675 | 0.726 | -0.051 [-0.163, +0.051] | 0.617 | 0.634 | -0.018 [-0.108, +0.060] |
| 0.50 | 0.789 | 0.789 | +0.000 | 3 / 3 | 1.000 | 0.629 | 0.684 | -0.055 [-0.167, +0.042] | 0.626 | 0.637 | -0.011 [-0.109, +0.074] |
| 0.60 | 0.789 | 0.789 | +0.000 | 3 / 3 | 1.000 | 0.629 | 0.684 | -0.055 [-0.165, +0.040] | 0.632 | 0.653 | -0.021 [-0.120, +0.066] |
| 0.70 | 0.789 | 0.763 | +0.026 | 4 / 3 | 1.000 | 0.629 | 0.657 | -0.028 [-0.137, +0.070] | 0.650 | 0.668 | -0.018 [-0.131, +0.083] |
| 0.80 | 0.789 | 0.763 | +0.026 | 4 / 3 | 1.000 | 0.629 | 0.657 | -0.028 [-0.138, +0.071] | 0.648 | 0.668 | -0.020 [-0.130, +0.082] |

Secondary at t=0.60 — union-over-top-10 hit: A 0.842, B 0.921. Page-match (upper bound): A 0.842, B 0.974. Reachable (≥1 relevant chunk exists in arm): A 1.000, B 1.000.

## Power

From `power_mde.csv` (McNemar exact, α=0.05 two-sided, 80% power). The detectable difference depends on the discordant rate actually observed above:

| n | disc 10% | 20% | 30% | 40% |
|---|---|---|---|---|
| 42 | no region | 0.189 | 0.228 | 0.250 |
| 80 | 0.095 | 0.141 | 0.178 | 0.213 |
| 90 | 0.082 | 0.130 | 0.173 | 0.194 |

The FRESH pool is n=38, below the smallest row. At n≈40 with 20% discordance the MDE is about 0.19 Recall points; a delta smaller than that is not detectable regardless of sign.

## OLD80 — CONTAMINATED - DO NOT REPORT

Computed for the record. Questions were written from Arm A chunks, gold IS an Arm A chunk (Arm A ceiling 1.0 by construction), and 12/40 table golds are longer than any Arm B chunk. Any Arm A advantage here is largely built in.

### Full pipeline (RRF + rerank) — all (n = 80)

| t | A R@10 | B R@10 | Δ R@10 | discordant A-only / B-only | McNemar p | A MRR | B MRR | Δ MRR [95% CI] | A nDCG | B nDCG | Δ nDCG [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.40 | 0.863 | 0.625 | +0.238 | 20 / 1 | 0.000 | 0.744 | 0.567 | +0.177 [+0.087, +0.269] | 0.749 | 0.514 | +0.235 [+0.152, +0.320] |
| 0.50 | 0.863 | 0.537 | +0.325 | 27 / 1 | 0.000 | 0.734 | 0.490 | +0.244 [+0.143, +0.346] | 0.762 | 0.500 | +0.262 [+0.164, +0.360] |
| 0.60 | 0.863 | 0.525 | +0.338 | 28 / 1 | 0.000 | 0.734 | 0.471 | +0.263 [+0.160, +0.367] | 0.763 | 0.484 | +0.278 [+0.180, +0.381] |
| 0.70 | 0.863 | 0.412 | +0.450 | 37 / 1 | 0.000 | 0.734 | 0.381 | +0.354 [+0.245, +0.464] | 0.765 | 0.388 | +0.377 [+0.270, +0.483] |
| 0.80 | 0.863 | 0.275 | +0.588 | 47 / 0 | 0.000 | 0.734 | 0.243 | +0.491 [+0.386, +0.596] | 0.765 | 0.251 | +0.514 [+0.414, +0.617] |

Secondary at t=0.60 — union-over-top-10 hit: A 0.863, B 0.662. Page-match (upper bound): A 0.912, B 0.912. Reachable (≥1 relevant chunk exists in arm): A 1.000, B 0.650.

### Full pipeline (RRF + rerank) — table (n = 40)

| t | A R@10 | B R@10 | Δ R@10 | discordant A-only / B-only | McNemar p | A MRR | B MRR | Δ MRR [95% CI] | A nDCG | B nDCG | Δ nDCG [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.40 | 0.800 | 0.325 | +0.475 | 19 / 0 | 0.000 | 0.575 | 0.274 | +0.300 [+0.157, +0.446] | 0.602 | 0.241 | +0.361 [+0.229, +0.499] |
| 0.50 | 0.800 | 0.200 | +0.600 | 24 / 0 | 0.000 | 0.575 | 0.183 | +0.391 [+0.246, +0.538] | 0.622 | 0.188 | +0.435 [+0.296, +0.576] |
| 0.60 | 0.800 | 0.175 | +0.625 | 25 / 0 | 0.000 | 0.575 | 0.158 | +0.416 [+0.268, +0.565] | 0.624 | 0.163 | +0.461 [+0.319, +0.600] |
| 0.70 | 0.800 | 0.125 | +0.675 | 27 / 0 | 0.000 | 0.575 | 0.108 | +0.466 [+0.323, +0.610] | 0.629 | 0.113 | +0.516 [+0.380, +0.647] |
| 0.80 | 0.800 | 0.100 | +0.700 | 28 / 0 | 0.000 | 0.575 | 0.083 | +0.491 [+0.348, +0.635] | 0.629 | 0.087 | +0.541 [+0.406, +0.674] |

Secondary at t=0.60 — union-over-top-10 hit: A 0.800, B 0.425. Page-match (upper bound): A 0.875, B 0.850. Reachable (≥1 relevant chunk exists in arm): A 1.000, B 0.325.

### Full pipeline (RRF + rerank) — prose (n = 40)

| t | A R@10 | B R@10 | Δ R@10 | discordant A-only / B-only | McNemar p | A MRR | B MRR | Δ MRR [95% CI] | A nDCG | B nDCG | Δ nDCG [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.40 | 0.925 | 0.925 | +0.000 | 1 / 1 | 1.000 | 0.912 | 0.859 | +0.053 [-0.044, +0.154] | 0.896 | 0.787 | +0.109 [+0.027, +0.197] |
| 0.50 | 0.925 | 0.875 | +0.050 | 3 / 1 | 0.625 | 0.894 | 0.797 | +0.097 [-0.022, +0.225] | 0.902 | 0.813 | +0.089 [-0.025, +0.204] |
| 0.60 | 0.925 | 0.875 | +0.050 | 3 / 1 | 0.625 | 0.894 | 0.784 | +0.110 [-0.013, +0.238] | 0.902 | 0.806 | +0.095 [-0.016, +0.212] |
| 0.70 | 0.925 | 0.700 | +0.225 | 10 / 1 | 0.012 | 0.894 | 0.653 | +0.241 [+0.081, +0.400] | 0.902 | 0.664 | +0.237 [+0.084, +0.392] |
| 0.80 | 0.925 | 0.450 | +0.475 | 19 / 0 | 0.000 | 0.894 | 0.403 | +0.491 [+0.338, +0.644] | 0.902 | 0.414 | +0.487 [+0.334, +0.637] |

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
