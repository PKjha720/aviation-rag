"""
Standing notes that every output of this experiment must carry.

Imported rather than retyped, so a summary cannot quietly drop one.
"""

# Measured in diagnostics.py over all 40 table golds of the old pool. Recorded
# here because an early single-example reading of Q026 got this wrong, and the
# wrong version is the more quotable one.
ATTRIBUTION = """\
MECHANISM ATTRIBUTION — REQUIRED IN ANY SUMMARY OF THIS EXPERIMENT
  Of 40 table-gold failures in Arm B (mutually exclusive categories):
      gold too long for any Arm B chunk ... 12
      fits, contiguous coverage < 0.60 .... 13
      fits, but cells scattered ...........  2
      reaches 0.60 in Arm B ............... 13
  Cell scattering explains 2 of 40. Gold length explains 12 and plain low
  coverage 13. Scattering is REAL (prose control: 1/40 at every scatter-gap
  threshold, against 10/40 for tables at gap > 0.20) but it is NOT the
  mechanism. Do not describe it as the mechanism.\
"""

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
  cell values (median 22 normalized chars) that fell under the 40-char floor.
  The prose stratum keeps the original generic prompt, unchanged and un-rerun.
  Both strata see only raw flat page text, and both pass identical deterministic
  validation. Compare strata with this difference in mind.\
"""
