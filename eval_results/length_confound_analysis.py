"""
Length-confound analysis for the table-vs-prose Recall@5 result.

The README and the preprint report Recall@5 of 0.675 on table-derived gold
chunks against 0.925 on prose-derived ones (full pipeline), p = 0.010. This
script asks whether that gap survives when gold chunk LENGTH is held roughly
constant, by splitting the table slice at the embedding model's input limit.

Reads:  eval_results/results_v2.csv          per-query retrieval outcomes
        eval_results/eval_dataset_v2.json    the 80 questions and their gold ids
        eval_results/gold_chunks_v2.json     the 80 gold chunks (text, length)
Writes: eval_results/length_confound_analysis.txt

gold_chunks_v2.json is a committed extract of data/processed/chunks_metadata.json
(which is gitignored, 14 MB) so that a reader can verify every number below
without rebuilding the index. If it is absent and chunks_metadata.json is
present, this script regenerates it.

Run:  python eval_results/length_confound_analysis.py
Deps: pandas, transformers (already required by sentence-transformers).
"""

import json
import sys
from math import comb
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_v2.csv"
EVAL = HERE / "eval_dataset_v2.json"
GOLD = HERE / "gold_chunks_v2.json"
LIVE = HERE.parent / "data" / "processed" / "chunks_metadata.json"
OUT = HERE / "length_confound_analysis.txt"

# all-MiniLM-L6-v2 default max_seq_length, as loaded by sentence-transformers
# and therefore by Chroma's SentenceTransformerEmbeddingFunction. Tokens beyond
# this position are not embedded.
EMBED_MAX_TOKENS = 256
TOKENIZER = "sentence-transformers/all-MiniLM-L6-v2"
MODES = ["dense", "sparse", "hybrid_rrf", "full"]
LABEL = {"dense": "Dense", "sparse": "Sparse (BM25)", "hybrid_rrf": "Hybrid (RRF)",
         "full": "Full pipeline"}


def fisher_two_sided(a, b, c, d):
    """Exact two-sided Fisher test on [[a, b], [c, d]] by summing the
    probabilities of all tables at least as extreme as the observed one."""
    n, r1, c1 = a + b + c + d, a + b, a + c
    def p(x):
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)
    p_obs = p(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return sum(p(x) for x in range(lo, hi + 1) if p(x) <= p_obs + 1e-12)


def ensure_gold_chunks(ev):
    if GOLD.exists():
        return json.loads(GOLD.read_text(encoding="utf-8"))
    if not LIVE.exists():
        sys.exit(f"Neither {GOLD.name} nor {LIVE} is present. Run ingest.py first, "
                 f"or restore {GOLD.name}.")
    meta = {c["chunk_id"]: c for c in json.loads(LIVE.read_text(encoding="utf-8"))}
    keep = ["chunk_id", "text", "char_count", "is_table", "source_file", "page_number"]
    gold = [{k: meta[q["ground_truth_chunk_id"]][k] for k in keep} for q in ev]
    GOLD.write_text(json.dumps(gold, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {GOLD} from {LIVE.name}")
    return gold


def main():
    ev = json.loads(EVAL.read_text(encoding="utf-8"))
    res = pd.read_csv(RESULTS)
    gold = {g["chunk_id"]: g for g in ensure_gold_chunks(ev)}

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    q = pd.DataFrame([{
        "question_id": e["question_id"],
        "is_table": bool(e["gold_is_table"]),
        "chars": gold[e["ground_truth_chunk_id"]]["char_count"],
        "tokens": len(tok(gold[e["ground_truth_chunk_id"]]["text"],
                          add_special_tokens=True)["input_ids"]),
    } for e in ev])
    q["over_limit"] = q.tokens > EMBED_MAX_TOKENS
    df = res.merge(q, on="question_id")

    L = []
    w = L.append
    w("LENGTH-CONFOUND ANALYSIS - table vs prose Recall@5")
    w("=" * 74)
    w(f"inputs : {RESULTS.name} ({len(res)} rows), {EVAL.name} ({len(ev)} questions), "
      f"{GOLD.name} ({len(gold)} chunks)")
    w(f"embedder input limit : {EMBED_MAX_TOKENS} tokens ({TOKENIZER} default max_seq_length)")
    w("test : two-sided Fisher exact on hit counts")
    w("")

    w("1. PUBLISHED RESULT, REPRODUCED")
    w(f"   {'mode':<15}{'table R@5':>12}{'prose R@5':>12}{'gap':>8}{'p':>10}")
    for m in MODES:
        d = df[df["mode"] == m]
        t, p_ = d[d.is_table].recall_at_5, d[~d.is_table].recall_at_5
        pv = fisher_two_sided(int(t.sum()), len(t) - int(t.sum()),
                              int(p_.sum()), len(p_) - int(p_.sum()))
        w(f"   {LABEL[m]:<15}{t.mean():>7.3f} ({int(t.sum())}/{len(t)})"
          f"{p_.mean():>7.3f} ({int(p_.sum())}/{len(p_)}){t.mean()-p_.mean():>+8.3f}{pv:>10.4f}")
    w("")

    # n=40 per slice, so a median is the mean of two middle values and can end
    # in .5. Round half UP for display (Python's default rounds half to even,
    # which would show 497.5 as 498 but 94.5 as 94) and print the exact values.
    from decimal import Decimal, ROUND_HALF_UP
    def half_up(x):
        return int(Decimal(str(x)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

    w("2. GOLD CHUNK LENGTH BY SLICE")
    w(f"   {'slice':<8}{'n':>4}{'chars med':>11}{'chars max':>11}{'tok med':>9}{'tok max':>9}"
      f"{'> ' + str(EMBED_MAX_TOKENS) + ' tok':>11}")
    for lab, m in (("table", q.is_table), ("prose", ~q.is_table)):
        s = q[m]
        w(f"   {lab:<8}{len(s):>4}{half_up(s.chars.median()):>11}{s.chars.max():>11}"
          f"{half_up(s.tokens.median()):>9}{s.tokens.max():>9}{int(s.over_limit.sum()):>7}/{len(s)}")
    w("   medians shown rounded half-up to whole units; exact values (mean of the")
    w(f"   two middle values, n=40): chars table {q[q.is_table].chars.median():.1f}, "
      f"prose {q[~q.is_table].chars.median():.1f}; tokens table "
      f"{q[q.is_table].tokens.median():.1f}, prose {q[~q.is_table].tokens.median():.1f}.")
    w("   Prose chunks are capped at 512 characters by chunk_text; table chunks may")
    w("   run to 1,500 by chunk_table. The strata differ in length by construction.")
    w("")

    tab = df[df.is_table]
    n_short = int((~q[q.is_table].over_limit).sum())
    n_long = int(q[q.is_table].over_limit.sum())
    w(f"3. TABLE SLICE SPLIT AT {EMBED_MAX_TOKENS} TOKENS  (same generation prompt on both sides)")
    w(f"   {'mode':<15}{'<= limit':>16}{'> limit':>16}{'p long vs short':>18}")
    w(f"   {'':<15}{'(n=' + str(n_short) + ')':>16}{'(n=' + str(n_long) + ')':>16}")
    for m in MODES:
        d = tab[tab["mode"] == m]
        s, l = d[~d.over_limit].recall_at_5, d[d.over_limit].recall_at_5
        pv = fisher_two_sided(int(l.sum()), len(l) - int(l.sum()),
                              int(s.sum()), len(s) - int(s.sum()))
        w(f"   {LABEL[m]:<15}{s.mean():>9.3f} ({int(s.sum())}/{len(s)})"
          f"{l.mean():>9.3f} ({int(l.sum())}/{len(l)}){pv:>18.5f}")
    w("")

    w("4. SHORT TABLE CHUNKS vs PROSE  (is there a table deficit at matched length?)")
    w(f"   {'mode':<15}{'table<=limit':>14}{'prose':>10}{'p':>10}")
    for m in MODES:
        d = df[df["mode"] == m]
        s = d[d.is_table & ~d.over_limit].recall_at_5
        p_ = d[~d.is_table].recall_at_5
        pv = fisher_two_sided(int(s.sum()), len(s) - int(s.sum()),
                              int(p_.sum()), len(p_) - int(p_.sum()))
        w(f"   {LABEL[m]:<15}{s.mean():>14.3f}{p_.mean():>10.3f}{pv:>10.2f}")
    w("")

    w("5. READING")
    w("   Within the table slice, chunks under the embedder's input limit retrieve")
    w("   at rates not distinguishable from prose in any mode; chunks over it")
    w("   retrieve far worse in every mode, including BM25, which does not truncate.")
    w("   The published table/prose gap is carried by the long table chunks. This")
    w("   is a post-hoc split on n=40, and length is not the only property that")
    w("   differs between long and short table chunks, but it is sufficient to")
    w("   show that the data do not support attributing the gap to bi-encoder")
    w("   handling of numerals: short table chunks are full of numerals and show")
    w("   no dense deficit.")
    w("")
    w("6. PER-QUESTION DETAIL (table slice)")
    w(f"   {'qid':<6}{'chars':>6}{'tokens':>7}{'over':>5}" + "".join(f"{m[:6]:>8}" for m in MODES))
    piv = tab.pivot(index="question_id", columns="mode", values="recall_at_5")
    for qid, r in q[q.is_table].sort_values("tokens", ascending=False).set_index("question_id").iterrows():
        w(f"   {qid:<6}{r.chars:>6}{r.tokens:>7}{'Y' if r.over_limit else '':>5}"
          + "".join(f"{int(piv.loc[qid, m]):>8}" for m in MODES))

    txt = "\n".join(L)
    OUT.write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
