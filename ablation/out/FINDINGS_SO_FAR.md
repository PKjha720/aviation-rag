# Findings so far — structure ablation

Branch `structure-ablation`. Written before any retrieval has been run.

**No retrieval delta has been computed.** No index has been built for either arm,
no query has been issued, and no nDCG, Recall or MRR figure exists for this
experiment. Nothing below depends on one.

What follows are results that are complete as they stand, because they are
properties of ingestion and of eval-set construction, measured before retrieval
enters. Each number names the file in `ablation/out/` it comes from.

The two arms, for reference:

- **Arm A** — the current pipeline, unchanged: PyMuPDF `find_tables()` rendering
  tables to markdown, page boundaries, 512-char prose chunks with 64-char
  overlap.
- **Arm B** — flat extraction via `get_text("text")`, no table detection, fixed
  92-token window with 12-token overlap.

Arm A was rebuilt from the PDFs using functions imported from `ingest.py` rather
than reimplemented, and reproduces the index on disk exactly: 10,572 chunks, 0
chunk ids differing, 0 shared ids whose text differs, 368 table chunks in both
(`ARM_A_VERIFICATION.txt`). Arm B's window was chosen so the two arms match on
chunk length: mean 409.5 vs 414.2 characters (−1.1%), median 455.0 vs 453.0
(+0.4%), from `parity_sweep.csv` and `length_distributions.txt`. Five windows
(84–100 tokens) satisfied the 10% band; 92 minimises combined error.

---

## 1. The original 80-question eval set cannot measure this experiment

`eval_results/eval_dataset_v2.json` contains 80 questions with one gold
`chunk_id` each. It has three independent defects with respect to a chunking
ablation, any one of which would be disqualifying.

**The judgments are addresses, not content.** `generate_chunk_id` in `ingest.py`
is `md5(f"{file}::page{page}::chunk{idx}")[:12]`. It hashes position. Change the
chunking and every id changes meaning: an id like `file::page7::chunk0` exists in
both arms and refers to different text in each. Scored naively, Arm B would post
a near-zero result that had nothing to do with chunking quality and that would
look entirely real.

**The questions were written from Arm A's chunks.** `build_eval_dataset` in
`evaluate_rag.py` samples a chunk and asks an LLM to write a question answerable
from it, using a dedicated `TABLE_PROMPT` for table chunks. The generator saw
Arm A's markdown rendering. The pool is, by construction, the set of questions an
Arm-A index can answer.

**The gold is itself an Arm A chunk**, so Arm A's per-chunk ceiling is 1.000 by
construction rather than by merit. Measured with the contiguous criterion, at a
60% coverage threshold Arm A reaches 1.000 and Arm B 0.650 overall, 0.325 on
table golds (`DIAGNOSTICS.txt`, `ceiling_contiguous.csv`). Arm A is being scored
against a target that is literally one of its own outputs.

These are recorded in `DIAGNOSTICS.txt` under a `CONTAMINATED - DO NOT REPORT`
banner. The pool is retained in the repository for completeness and must not be
quoted as a result.

### The general constraint

A chunking ablation cannot be run on an eval set whose questions were generated
by sampling one arm's chunks, and the problem is not fixed by re-keying the
judgments.

Sampling chunks to build questions means only content that survived *that*
ingestion can become a question. Evidence destroyed or fragmented by that
pipeline is never a candidate, never a gold, and never a miss — it leaves no
trace in any metric. When the same pipeline is then one arm of a comparison, the
eval set encodes that arm's view of the corpus as the definition of what is
askable.

Re-grounding the judgments on gold *text* rather than chunk id removes the
addressing defect but not this one. It is why the fresh pool exists, why its gold
is a short span rather than a chunk from either arm, and why a span must fit
inside a single chunk in **both** arms before it is accepted.

---

## 2. Gold-length reachability: 12 of 40 table golds are unreachable by arithmetic

Arm A's `chunk_table` permits chunks up to 1,500 characters and produces one of
1,375 normalized characters. Arm B is bounded by its 92-token window at 550
normalized characters. A gold longer than 550 cannot be covered by any single
Arm B chunk at any threshold.

From `ceiling_contiguous.csv`:

| slice | n | gold median | gold max | fits Arm A | fits Arm B |
|---|---|---|---|---|---|
| all | 80 | 387 | 1079 | 1.000 | 0.850 |
| table-gold | 40 | 397 | 1079 | 1.000 | 0.700 |
| prose-gold | 40 | 384 | 428 | 1.000 | 1.000 |

**12 of 40 table golds exceed Arm B's maximum chunk length.** Their Arm B
per-chunk coverage cannot reach 0.60, and does not: that subgroup reaches 0.60 at
a rate of exactly 0.000, against 0.765 for golds that do fit.

This is arithmetic, not retrieval. Corpus-level chunk-length parity holds (§ above),
but parity on the *corpus* does not imply parity on the *golds*: Arm A's length
distribution has a right tail that Arm B structurally cannot have, and the table
golds are drawn disproportionately from that tail. Any comparison on this pool
would be measuring that, in part, rather than structure.

n = 40 per slice. This is a deterministic property of the two chunkings, not a
sampled estimate, so sampling error does not apply to the 12/40 count itself —
but the count describes *this* pool of golds and does not generalise to another.

---

## 3. Scatter analysis: cell scattering is real but is not the mechanism

The intuition that `get_text("text")` reads tables column-wise, scattering a
row's cells, is visible in individual cases. Measured across all 40 table golds
it explains a small minority of failures.

Method: for each gold, compare the longest *contiguous* run of gold characters
found in the best Arm B chunk against the *sum of all* matching runs in that
chunk. A large gap means the content is present but broken into pieces, which is
the signature of column-wise reading. `coverage()` in `judgments.py` is the
contiguous measure and is the only one used for any relevance decision;
`coverage_blocks()` is diagnostic and never decides anything.

Why table golds fail in Arm B, from `ceiling_contiguous.csv` and
`scatter_per_query.csv`, n = 40, categories mutually exclusive:

| cause | count |
|---|---|
| gold too long for any Arm B chunk | 12 |
| fits, contiguous coverage below 0.60 | 13 |
| fits, but cells scattered | 2 |
| reaches 0.60 in Arm B | 13 |

**Cell scattering explains 2 of 40.** Gold length explains 12 and plain low
contiguous coverage 13. Any summary of this experiment must attribute the effect
that way and must not describe scattering as the mechanism.

Scattering is nonetheless real, and the prose control establishes that it tracks
table structure rather than some general property of long chunks:

| scatter gap > | table-gold | prose-gold |
|---|---|---|
| 0.05 | 17 / 40 | 1 / 40 |
| 0.10 | 11 / 40 | 1 / 40 |
| 0.20 | 10 / 40 | 1 / 40 |
| 0.30 | 9 / 40 | 1 / 40 |
| 0.40 | 4 / 40 | 1 / 40 |

The strict binary criterion (block-sum ≥ 0.60 and contiguous < 0.35) yields 2/40
and understates the phenomenon; the graded distribution above is the better view.
Both are reported so neither reading stands alone.

n = 40 per slice. At that size the table-versus-prose contrast at gap > 0.20
(10/40 against 1/40) is large enough to be worth stating, but the finer
distinctions between adjacent thresholds are not, and no significance test has
been run on these counts.

---

## 4. A converse asymmetry, observed but not established (n = 6)

Structure-preserving ingestion also breaks some spans that flat extraction keeps
whole. This is the converse of the table-scattering story and is worth recording,
but the evidence is thin and partly confounded.

During fresh-pool construction each candidate span must fit inside a single chunk
in both arms. The stage that rejects spans fitting no single **Arm A** chunk
(`V6_no_fit_armA`) fired:

- v1 generic prompt: **1** of 60 table pages (`fresh_pool_rejects.csv`)
- v2 label+value prompt: **6** of 105 real responses (`fresh_pool_v2_rejects.csv`)

The mechanism is plausible. Arm A splits a page into a markdown table block plus
a prose block and reorders table cells row-wise, so a run of text that is
contiguous in column-wise flat output can straddle that boundary or be reordered
away, fitting no single Arm A chunk.

Three reasons not to lean on this:

1. **n = 6.** Six events cannot support a rate, a comparison, or a claim about
   the corpus.
2. **It is confounded with the prompt change.** The v2 prompt asks for longer
   spans carrying a row label *and* its value, which are mechanically more likely
   to straddle a chunk boundary than the bare cell values v1 produced. The rise
   from 1 to 6 is at least partly caused by what was asked for, not by a property
   of the documents. The two counts also have different denominators (60 pages
   versus 105 responses).
3. **It was not designed for.** These are rejection-stage side effects, not a
   measurement anyone set out to make.

Stated as an observation worth a dedicated test, not as a finding.

---

## 5. Pool construction is itself constrained, independently of retrieval

The fresh pool is generated from `get_text("text")` — Arm B's extraction path —
and is therefore **biased toward Arm B**. It is not neutral and must not be
described as unbiased. The bias runs against the expected direction of the
effect, so a surviving Arm A advantage would be a lower bound rather than an
estimate; an Arm B advantage measured on this pool would be uninterpretable.
This label is carried in `notes.py` and injected into every artifact the pool
touches.

Current state (`POOL_STATUS.txt`): **42 accepted questions, 24 table and 18
prose**, against a target of 45 and 45.

The table sweep stopped at 119 of 199 table-bearing pages because Groq's free
tier daily token quota was exhausted (`Limit 200000, Used 199856`). Fourteen
pages attempted after that returned empty content; those are recorded as quota
failures, not as samples, and 80 pages were never attempted. The corpus was not
exhausted — the quota was. The 40-character span floor was not relaxed.

Two constraints surfaced that do not depend on how the run finishes:

**The span-length band binds hard.** Replaying validation offline across a grid
(`validation_yield.csv`, no API cost) shows that at the 40-character floor,
raising the cap from 98 to 240 characters moves total yield only from 31 to 50,
while spans that fit no single Arm B chunk rise from 4 to 9. Dropping the floor to
zero raises the count but 29 of those spans match more than one page, at which
point "relevant" stops being well defined.

**The two strata behave oppositely.** Table pages yield spans with a median of 22
normalized characters — the model quotes a bare cell — while prose pages yield a
median of 84. A single length band cannot suit both, which is why the table
stratum was regenerated with a different prompt, and why the two strata are no
longer generated by the same instruction. That asymmetry is recorded in
`notes.py` and must be kept in mind when comparing them.

**Row pairing.** Because the table prompt asks for a span containing a row label
and its value, and because flat text is column-ordered, a span can pass every
deterministic check while pairing a label from one row with a value from another.
`row_pairing_check.csv` reconstructs each page row-wise with `find_tables()` and
reports, over the 24 table items: 20 same row, 0 different row, 4 undetermined.
Of the four undetermined, three are prose golds on pages that merely contain a
table elsewhere, which is benign, and one is tabular content that could not be
placed. `find_tables()` is Arm A's own view of the page, so these verdicts are
evidence rather than ground truth, and nothing was dropped on their strength.

---

## Power, stated plainly

At n = 42 — and at the target n ≈ 90 — a paired comparison of binary retrieval
outcomes can detect only a large difference. On 80 paired queries a difference
below roughly 10 percentage points of Recall is not reliably distinguishable from
noise, and the per-stratum slices are weaker still. Nothing in this document is a
retrieval result, but the same limit will apply when one exists, and it should be
reported alongside any delta rather than after it.

One further caution for when metrics are computed: with a single binary gold per
query, nDCG@10 is a deterministic function of the gold's rank, `1/log2(rank+1)`.
It is a third view of the same rank, not a third independent piece of evidence,
and presenting nDCG, Recall and MRR as three corroborating results would
overstate what has been measured.

---

## Files

| file | contents |
|---|---|
| `ARM_A_VERIFICATION.txt` | Arm A rebuild vs the index on disk |
| `parity_sweep.csv`, `length_distributions.txt` | chunk-length parity, all windows tried |
| `anchors.json` | re-grounded text judgments for the original 80 |
| `ceiling_contiguous.csv`, `DIAGNOSTICS.txt` | reachability, gold length, scatter analysis |
| `scatter_per_query.csv` | per-query scatter gaps |
| `fresh_pool_v2.json`, `POOL_STATUS.txt` | the fresh pool and why it is incomplete |
| `fresh_pool_rejects.csv`, `fresh_pool_v2_rejects.csv` | every rejection with its stage |
| `validation_yield.csv` | offline replay across span-length bands |
| `row_pairing_check.csv` | row-pairing verdicts for the table stratum |
| `HANDCHECK_20.txt` | 20 questions awaiting human review |
