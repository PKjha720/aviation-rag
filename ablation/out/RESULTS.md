# Structure ablation — results

Arm A: current pipeline (find_tables → markdown, page boundaries, 512-char prose / 64 overlap).
Arm B: flat get_text("text"), no table detection, 92-token window / 12 overlap, page-scoped.
Identical retrieval stack for both: all-MiniLM-L6-v2 dense + BM25 + RRF(k=60) + ms-marco-MiniLM-L-6-v2 rerank over top 30; k=10.
Chunk-length parity: Arm A mean/median 414.2/453.0 chars, Arm B 409.5/455.0 (−1.1% / +0.4%).

**Headline pool: FRESH, n = 29 (13 table / 16 prose).** Read the four notices at the end before quoting anything.

## ⚠ SIGN INSTABILITY ACROSS THRESHOLDS

The sign of the delta is NOT the same at all five coverage thresholds for:

- FRESH / all / dense / d_ndcg10: 0.4:+0.007, 0.5:+0.008, 0.6:-0.001, 0.7:-0.003, 0.8:-0.006
- FRESH / all / hybrid / d_ndcg10: 0.4:-0.001, 0.5:+0.001, 0.6:-0.004, 0.7:-0.002, 0.8:-0.005
- FRESH / all / sparse / d_ndcg10: 0.4:-0.004, 0.5:+0.002, 0.6:-0.005, 0.7:-0.000, 0.8:-0.003

Where this occurs the direction of the effect depends on how strictly relevance is defined and must not be reported as a single-signed result.

## FRESH pool (headline)

### Full pipeline (RRF + rerank) — all (n = 29)

| t | A R@10 | B R@10 | Δ R@10 | discordant A-only / B-only | McNemar p | A MRR | B MRR | Δ MRR [95% CI] | A nDCG | B nDCG | Δ nDCG [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.40 | 0.828 | 0.828 | +0.000 | 2 / 2 | 1.000 | 0.690 | 0.759 | -0.069 [-0.195, +0.041] | 0.639 | 0.641 | -0.001 [-0.098, +0.081] |
| 0.50 | 0.828 | 0.828 | +0.000 | 2 / 2 | 1.000 | 0.664 | 0.716 | -0.051 [-0.180, +0.061] | 0.656 | 0.655 | +0.001 [-0.100, +0.090] |
| 0.60 | 0.828 | 0.828 | +0.000 | 2 / 2 | 1.000 | 0.664 | 0.716 | -0.051 [-0.183, +0.062] | 0.666 | 0.670 | -0.004 [-0.112, +0.091] |
| 0.70 | 0.828 | 0.793 | +0.034 | 3 / 2 | 1.000 | 0.664 | 0.679 | -0.015 [-0.141, +0.093] | 0.688 | 0.689 | -0.002 [-0.128, +0.110] |
| 0.80 | 0.828 | 0.793 | +0.034 | 3 / 2 | 1.000 | 0.664 | 0.679 | -0.015 [-0.145, +0.094] | 0.685 | 0.689 | -0.005 [-0.134, +0.112] |

Secondary at t=0.60 — union-over-top-10 hit: A 0.931, B 0.862. Page-match (upper bound): A 0.931, B 1.000. Reachable (≥1 relevant chunk exists in arm): A 1.000, B 1.000.

### Full pipeline (RRF + rerank) — table (n = 13)

| t | A R@10 | B R@10 | Δ R@10 | discordant A-only / B-only | McNemar p | A MRR | B MRR | Δ MRR [95% CI] | A nDCG | B nDCG | Δ nDCG [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.40 | 0.615 | 0.615 | +0.000 | 2 / 2 | 1.000 | 0.413 | 0.577 | -0.164 [-0.410, +0.054] | 0.405 | 0.433 | -0.028 [-0.234, +0.153] |
| 0.50 | 0.615 | 0.615 | +0.000 | 2 / 2 | 1.000 | 0.355 | 0.481 | -0.126 [-0.385, +0.115] | 0.404 | 0.411 | -0.007 [-0.227, +0.177] |
| 0.60 | 0.615 | 0.615 | +0.000 | 2 / 2 | 1.000 | 0.355 | 0.481 | -0.126 [-0.395, +0.113] | 0.404 | 0.437 | -0.033 [-0.259, +0.171] |
| 0.70 | 0.615 | 0.538 | +0.077 | 3 / 2 | 1.000 | 0.355 | 0.400 | -0.045 [-0.315, +0.180] | 0.422 | 0.438 | -0.016 [-0.290, +0.220] |
| 0.80 | 0.615 | 0.538 | +0.077 | 3 / 2 | 1.000 | 0.355 | 0.400 | -0.045 [-0.309, +0.183] | 0.422 | 0.438 | -0.016 [-0.293, +0.223] |

Secondary at t=0.60 — union-over-top-10 hit: A 0.846, B 0.692. Page-match (upper bound): A 0.846, B 1.000. Reachable (≥1 relevant chunk exists in arm): A 1.000, B 1.000.

### Full pipeline (RRF + rerank) — prose (n = 16)

| t | A R@10 | B R@10 | Δ R@10 | discordant A-only / B-only | McNemar p | A MRR | B MRR | Δ MRR [95% CI] | A nDCG | B nDCG | Δ nDCG [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.40 | 1.000 | 1.000 | +0.000 | 0 / 0 | 1.000 | 0.915 | 0.906 | +0.009 [-0.067, +0.094] | 0.829 | 0.809 | +0.020 [+0.000, +0.053] |
| 0.50 | 1.000 | 1.000 | +0.000 | 0 / 0 | 1.000 | 0.915 | 0.906 | +0.009 [-0.067, +0.094] | 0.861 | 0.853 | +0.008 [-0.052, +0.073] |
| 0.60 | 1.000 | 1.000 | +0.000 | 0 / 0 | 1.000 | 0.915 | 0.906 | +0.009 [-0.067, +0.094] | 0.879 | 0.860 | +0.019 [-0.041, +0.083] |
| 0.70 | 1.000 | 1.000 | +0.000 | 0 / 0 | 1.000 | 0.915 | 0.906 | +0.009 [-0.067, +0.094] | 0.903 | 0.893 | +0.010 [-0.051, +0.074] |
| 0.80 | 1.000 | 1.000 | +0.000 | 0 / 0 | 1.000 | 0.915 | 0.906 | +0.009 [-0.067, +0.094] | 0.898 | 0.893 | +0.005 [-0.052, +0.069] |

Secondary at t=0.60 — union-over-top-10 hit: A 1.000, B 1.000. Page-match (upper bound): A 1.000, B 1.000. Reachable (≥1 relevant chunk exists in arm): A 1.000, B 1.000.

### Diagnostic modes (FRESH, all)

### Dense only — all (n = 29)

| t | A R@10 | B R@10 | Δ R@10 | discordant A-only / B-only | McNemar p | A MRR | B MRR | Δ MRR [95% CI] | A nDCG | B nDCG | Δ nDCG [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.40 | 0.793 | 0.793 | +0.000 | 2 / 2 | 1.000 | 0.668 | 0.724 | -0.056 [-0.200, +0.082] | 0.609 | 0.602 | +0.007 [-0.102, +0.117] |
| 0.50 | 0.759 | 0.793 | -0.034 | 2 / 3 | 1.000 | 0.633 | 0.681 | -0.048 [-0.198, +0.099] | 0.626 | 0.618 | +0.008 [-0.112, +0.132] |
| 0.60 | 0.759 | 0.793 | -0.034 | 2 / 3 | 1.000 | 0.633 | 0.681 | -0.048 [-0.203, +0.095] | 0.632 | 0.633 | -0.001 [-0.127, +0.126] |
| 0.70 | 0.759 | 0.759 | +0.000 | 3 / 3 | 1.000 | 0.633 | 0.645 | -0.011 [-0.170, +0.133] | 0.648 | 0.651 | -0.003 [-0.150, +0.135] |
| 0.80 | 0.759 | 0.759 | +0.000 | 3 / 3 | 1.000 | 0.633 | 0.645 | -0.011 [-0.162, +0.133] | 0.645 | 0.651 | -0.006 [-0.151, +0.134] |

Secondary at t=0.60 — union-over-top-10 hit: A 0.862, B 0.828. Page-match (upper bound): A 0.862, B 1.000. Reachable (≥1 relevant chunk exists in arm): A 1.000, B 1.000.

### Sparse (BM25) only — all (n = 29)

| t | A R@10 | B R@10 | Δ R@10 | discordant A-only / B-only | McNemar p | A MRR | B MRR | Δ MRR [95% CI] | A nDCG | B nDCG | Δ nDCG [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.40 | 0.828 | 0.828 | +0.000 | 2 / 2 | 1.000 | 0.697 | 0.759 | -0.062 [-0.184, +0.041] | 0.639 | 0.643 | -0.004 [-0.098, +0.082] |
| 0.50 | 0.828 | 0.828 | +0.000 | 2 / 2 | 1.000 | 0.671 | 0.716 | -0.045 [-0.171, +0.066] | 0.657 | 0.655 | +0.002 [-0.099, +0.094] |
| 0.60 | 0.828 | 0.828 | +0.000 | 2 / 2 | 1.000 | 0.671 | 0.716 | -0.045 [-0.172, +0.069] | 0.666 | 0.670 | -0.005 [-0.113, +0.093] |
| 0.70 | 0.828 | 0.793 | +0.034 | 3 / 2 | 1.000 | 0.671 | 0.679 | -0.009 [-0.136, +0.105] | 0.689 | 0.689 | -0.000 [-0.129, +0.117] |
| 0.80 | 0.828 | 0.793 | +0.034 | 3 / 2 | 1.000 | 0.671 | 0.679 | -0.009 [-0.137, +0.102] | 0.686 | 0.689 | -0.003 [-0.130, +0.115] |

Secondary at t=0.60 — union-over-top-10 hit: A 0.862, B 0.897. Page-match (upper bound): A 0.897, B 1.000. Reachable (≥1 relevant chunk exists in arm): A 1.000, B 1.000.

## Power

From `power_mde.csv` (McNemar exact, α=0.05 two-sided, 80% power). The detectable difference depends on the discordant rate actually observed above:

| n | disc 10% | 20% | 30% | 40% |
|---|---|---|---|---|
| 42 | no region | 0.189 | 0.228 | 0.250 |
| 80 | 0.095 | 0.141 | 0.178 | 0.213 |
| 90 | 0.082 | 0.130 | 0.173 | 0.194 |

The FRESH pool is n=29, below the smallest row. At n≈40 with 20% discordance the MDE is about 0.19 Recall points; a delta smaller than that is not detectable regardless of sign.

## OLD80 — CONTAMINATED - DO NOT REPORT

Computed for the record. Questions were written from Arm A chunks, gold IS an Arm A chunk (Arm A ceiling 1.0 by construction), and 12/40 table golds are longer than any Arm B chunk. Any Arm A advantage here is largely built in.

### Full pipeline (RRF + rerank) — all (n = 80)

| t | A R@10 | B R@10 | Δ R@10 | discordant A-only / B-only | McNemar p | A MRR | B MRR | Δ MRR [95% CI] | A nDCG | B nDCG | Δ nDCG [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.40 | 0.863 | 0.625 | +0.238 | 20 / 1 | 0.000 | 0.744 | 0.567 | +0.177 [+0.086, +0.272] | 0.749 | 0.514 | +0.235 [+0.152, +0.323] |
| 0.50 | 0.863 | 0.537 | +0.325 | 27 / 1 | 0.000 | 0.734 | 0.490 | +0.244 [+0.142, +0.348] | 0.762 | 0.500 | +0.262 [+0.165, +0.362] |
| 0.60 | 0.863 | 0.525 | +0.338 | 28 / 1 | 0.000 | 0.734 | 0.471 | +0.263 [+0.158, +0.369] | 0.763 | 0.484 | +0.278 [+0.180, +0.380] |
| 0.70 | 0.863 | 0.412 | +0.450 | 37 / 1 | 0.000 | 0.734 | 0.381 | +0.354 [+0.244, +0.465] | 0.765 | 0.388 | +0.377 [+0.270, +0.486] |
| 0.80 | 0.863 | 0.275 | +0.588 | 47 / 0 | 0.000 | 0.734 | 0.243 | +0.491 [+0.387, +0.597] | 0.765 | 0.251 | +0.514 [+0.412, +0.613] |

Secondary at t=0.60 — union-over-top-10 hit: A 0.863, B 0.662. Page-match (upper bound): A 0.912, B 0.912. Reachable (≥1 relevant chunk exists in arm): A 1.000, B 0.650.

### Full pipeline (RRF + rerank) — table (n = 40)

| t | A R@10 | B R@10 | Δ R@10 | discordant A-only / B-only | McNemar p | A MRR | B MRR | Δ MRR [95% CI] | A nDCG | B nDCG | Δ nDCG [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.40 | 0.800 | 0.325 | +0.475 | 19 / 0 | 0.000 | 0.575 | 0.274 | +0.300 [+0.154, +0.448] | 0.602 | 0.241 | +0.361 [+0.225, +0.498] |
| 0.50 | 0.800 | 0.200 | +0.600 | 24 / 0 | 0.000 | 0.575 | 0.183 | +0.391 [+0.243, +0.537] | 0.622 | 0.188 | +0.435 [+0.294, +0.575] |
| 0.60 | 0.800 | 0.175 | +0.625 | 25 / 0 | 0.000 | 0.575 | 0.158 | +0.416 [+0.266, +0.567] | 0.624 | 0.163 | +0.461 [+0.316, +0.599] |
| 0.70 | 0.800 | 0.125 | +0.675 | 27 / 0 | 0.000 | 0.575 | 0.108 | +0.466 [+0.324, +0.607] | 0.629 | 0.113 | +0.516 [+0.384, +0.649] |
| 0.80 | 0.800 | 0.100 | +0.700 | 28 / 0 | 0.000 | 0.575 | 0.083 | +0.491 [+0.351, +0.632] | 0.629 | 0.087 | +0.541 [+0.407, +0.669] |

Secondary at t=0.60 — union-over-top-10 hit: A 0.800, B 0.425. Page-match (upper bound): A 0.875, B 0.850. Reachable (≥1 relevant chunk exists in arm): A 1.000, B 0.325.

### Full pipeline (RRF + rerank) — prose (n = 40)

| t | A R@10 | B R@10 | Δ R@10 | discordant A-only / B-only | McNemar p | A MRR | B MRR | Δ MRR [95% CI] | A nDCG | B nDCG | Δ nDCG [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.40 | 0.925 | 0.925 | +0.000 | 1 / 1 | 1.000 | 0.912 | 0.859 | +0.053 [-0.040, +0.153] | 0.896 | 0.787 | +0.109 [+0.029, +0.195] |
| 0.50 | 0.925 | 0.875 | +0.050 | 3 / 1 | 0.625 | 0.894 | 0.797 | +0.097 [-0.025, +0.222] | 0.902 | 0.813 | +0.089 [-0.022, +0.204] |
| 0.60 | 0.925 | 0.875 | +0.050 | 3 / 1 | 0.625 | 0.894 | 0.784 | +0.110 [-0.015, +0.237] | 0.902 | 0.806 | +0.095 [-0.016, +0.214] |
| 0.70 | 0.925 | 0.700 | +0.225 | 10 / 1 | 0.012 | 0.894 | 0.653 | +0.241 [+0.081, +0.401] | 0.902 | 0.664 | +0.237 [+0.083, +0.395] |
| 0.80 | 0.925 | 0.450 | +0.475 | 19 / 0 | 0.000 | 0.894 | 0.403 | +0.491 [+0.337, +0.647] | 0.902 | 0.414 | +0.487 [+0.337, +0.641] |

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

HAND-CHECK PROVENANCE: every pool item was reviewed by a model at the author's instruction on 2026-09-20; none was reviewed by a human. Three passes: (1) a 20-item check of the 39-item pool (HANDCHECK_20.txt), 19 accepted, 1 rejected (T013); (2) an independent model reviewer, blind to pass 1 and briefed to reject wherever it could, over the same 20, rejecting 5 more (T003 T004 T006 T010 P005), rule: reject if either pass rejects; (3) a single-pass review at the stricter standard of the 19 items no pass had seen (HANDCHECK_REMAINING_19.txt), rejecting 4 (T012 T016 T018 P016). Pool 39 -> 33 -> 29. All 39 items were reviewed at least once and 20 of them twice; of the 29 that remain, 14 were reviewed twice and 15 once.

nDCG@10 NOTE: with a single relevant chunk nDCG@10 = 1/log2(rank+1), a re-expression of the same rank MRR uses. It is not independent evidence. Overlapping windows can make >1 chunk relevant, in which case IDCG uses the count of relevant chunks in that arm.

Files: results_per_query.csv (every query × arm × mode × threshold), results_summary.csv (every aggregate cell).
