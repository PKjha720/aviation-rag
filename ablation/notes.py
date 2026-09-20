"""
Standing notes that every output of this experiment must carry.

The numbers live in one dict, ATTRIBUTION_COUNTS, and the prose is formatted
from it. diagnostics.py recomputes the same counts from data and REFUSES TO RUN
if they disagree with this dict, so a stale copy here becomes a crash rather
than a quietly wrong sentence.
"""

# Measured in diagnostics.py over all 40 table golds of the old pool. Recorded
# because an early single-example reading of Q026 got the mechanism wrong, and
# the wrong version is the more quotable one.
ATTRIBUTION_COUNTS = {
    "n_table_golds": 40,
    "too_long": 12,          # gold longer than any single Arm B chunk
    "too_long_barred_060": 7,  # of those, max possible coverage < 0.60 by arithmetic
    "low_cov": 13,           # fits, contiguous coverage < 0.60, not scattered
    "scattered": 2,          # fits, block-sum >= 0.60 and contiguous < 0.35
    "reached": 13,           # contiguous coverage >= 0.60
    "prose_scatter_gt_020": 1,
    "table_scatter_gt_020": 10,
}
ATTRIBUTION_COUNTS["failures"] = (ATTRIBUTION_COUNTS["n_table_golds"]
                                  - ATTRIBUTION_COUNTS["reached"])


def format_attribution(c: dict) -> str:
    return f"""\
MECHANISM ATTRIBUTION — REQUIRED IN ANY SUMMARY OF THIS EXPERIMENT
  Of {c['n_table_golds']} table golds, {c['failures']} fail to reach 0.60 contiguous
  coverage in Arm B and {c['reached']} reach it. The {c['n_table_golds']} split into
  mutually exclusive categories:
      gold longer than any single Arm B chunk ... {c['too_long']:>2}
          (of these, {c['too_long_barred_060']} are arithmetically barred from 0.60;
           the other {c['too_long'] - c['too_long_barred_060']} could reach it and were
           measured not to)
      fits, contiguous coverage < 0.60 ......... {c['low_cov']:>2}
      fits, but cells scattered ................ {c['scattered']:>2}
      reaches 0.60 in Arm B .................... {c['reached']:>2}
  Cell scattering explains {c['scattered']} of the {c['failures']} failures. Gold length
  explains {c['too_long']} and plain low coverage {c['low_cov']}. Scattering is REAL
  (prose control: {c['prose_scatter_gt_020']}/40 at scatter gap > 0.20 against
  {c['table_scatter_gt_020']}/40 for tables) but it is NOT the mechanism. Do not
  describe it as the mechanism.\
"""


ATTRIBUTION = format_attribution(ATTRIBUTION_COUNTS)

BIAS = """\
BIAS LABEL — FRESH POOL
  Questions are generated from page.get_text("text"), which IS ARM B's
  extraction path. This pool is BIASED TOWARD ARM B. It is not neutral and must
  never be called unbiased or bias-free. The bias runs against the expected
  effect, so a surviving Arm A advantage is a lower bound, not an estimate. An
  Arm B advantage measured on this pool would be uninterpretable.\
"""

# The two strata no longer share a generator prompt. Recorded so the asymmetry
# is never mistaken for a property of the documents.
PROMPT_ASYMMETRY = """\
PROMPT ASYMMETRY — TABLE vs PROSE STRATUM
  The table stratum was regenerated with a prompt requiring the span to carry a
  row LABEL plus its VALUE, because the original generic prompt returned bare
  cell values (median 22 normalized chars, from replay_validation.py console
  output) that fell under the 40-char floor. The prose stratum keeps the
  original generic prompt, unchanged and un-rerun. Both strata see only raw flat
  page text, and both pass identical deterministic validation. Compare strata
  with this difference in mind.\
"""

# Stratum labels were originally assigned by PAGE (did Arm A find a table
# anywhere on the page). Three table-page golds turned out to sit in prose
# chunks. Analysis now keys on the GOLD, not the page; page_stratum is retained
# as sampling provenance.
STRATUM_RELABEL = """\
STRATUM LABELS
  page_stratum : which sampling frame the page came from (table-bearing page or
                 prose-only page). Provenance only.
  gold_stratum : whether the gold span sits inside an Arm A chunk that ingest
                 flagged is_table. This is the label every table-vs-prose
                 comparison uses. Three items sampled from table-bearing pages
                 have prose golds and are counted as PROSE here; they were
                 generated with the table prompt, which prompt_version records.\
"""
