"""
Write 20 fresh-pool questions out for human review.

Stratified 10 table-page / 10 prose-page, seeded. Each entry carries everything
needed to judge it without opening the PDF: the question, the gold span, and the
surrounding page text so the span can be checked in context.

Three things are worth checking on each one:
  1. Does the question stand alone? (no pointer back to a page you cannot see)
  2. Does the gold span actually answer it?
  3. Is the span the RIGHT answer, not merely a plausible-looking string?

Writes: ablation/out/HANDCHECK_20.txt
"""

import json
import random
from pathlib import Path

from arms import list_pdfs, extract_flat
from judgments import _NON_ALNUM

OUT = Path("ablation/out")
CONTEXT = 420


def norm_map(text):
    """Normalized string plus a map from each normalized index back to its raw
    index, so a span that passed validation on normalized text can still be
    shown in its raw context even when it straddles line breaks."""
    out, idx = [], []
    for i, ch in enumerate(text):
        c = ch.lower()
        if not _NON_ALNUM.match(c):
            out.append(c)
            idx.append(i)
    return "".join(out), idx


def locate(page_text, span):
    """(raw_start, raw_end) of the span in page_text, or None."""
    pn, pmap = norm_map(page_text)
    sn, _ = norm_map(span)
    if not sn:
        return None
    j = pn.find(sn)
    if j < 0:
        return None
    return pmap[j], pmap[min(j + len(sn), len(pmap)) - 1] + 1


def main():
    pool = json.loads((OUT / "fresh_pool.json").read_text(encoding="utf-8"))
    if not pool:
        print("FAILED: fresh_pool.json is empty. Nothing to hand-check.")
        return

    tab = [q for q in pool if q["page_stratum"] == "table"]
    pro = [q for q in pool if q["page_stratum"] == "prose"]
    rng = random.Random(0)
    pick = (rng.sample(tab, min(10, len(tab))) + rng.sample(pro, min(10, len(pro))))
    pick.sort(key=lambda q: q["question_id"])

    flat = {}
    for p in list_pdfs():
        for pg in extract_flat(p):
            flat[(p.name, pg["page_number"])] = pg["text"]

    n_unloc = [0]
    L = []
    w = L.append
    w("=" * 78)
    w("HAND-CHECK: 20 QUESTIONS FROM THE FRESH POOL")
    w("=" * 78)
    w("")
    w("BIAS LABEL: this pool was generated from page.get_text('text'), which is")
    w("ARM B's extraction path. It is BIASED TOWARD ARM B. It is not neutral and")
    w("must not be described as unbiased. The bias runs against the expected")
    w("effect, so an Arm A advantage measured here is a lower bound.")
    w("")
    w(f"Sampled {len(pick)} of {len(pool)} accepted questions, seeded, stratified")
    w(f"({sum(1 for q in pick if q['page_stratum']=='table')} from table-bearing pages, "
      f"{sum(1 for q in pick if q['page_stratum']=='prose')} from prose-only pages).")
    w("")
    w("Check each one:")
    w("  1. Does the question stand alone, with no pointer to a page you cannot see?")
    w("  2. Does the gold span actually answer it?")
    w("  3. Is the span the right answer, not just a plausible-looking string?")
    w("")
    w("Mark any you reject and I will drop them and recount before any retrieval runs.")
    w("")

    for i, q in enumerate(pick, 1):
        page_text = flat.get((q["source_file"], q["page_number"]), "")
        loc = locate(page_text, q["gold_span"])
        if loc is None:
            ctx = page_text[:CONTEXT * 2]
            marker = "(span not locatable for display; it passed normalized validation)"
            n_unloc[0] += 1
        else:
            a, b = loc
            s_, e_ = max(0, a - CONTEXT), min(len(page_text), b + CONTEXT)
            ctx = (page_text[s_:a] + ">>>SPAN>>>" + page_text[a:b]
                   + "<<<SPAN<<<" + page_text[b:e_])
            marker = ""

        w("=" * 78)
        w(f"[{i:02d}]  {q['question_id']}   stratum={q['page_stratum']}   "
          f"span={q['gold_span_norm_len']} norm chars")
        w(f"      {q['source_file']}  page {q['page_number']}")
        w("-" * 78)
        w("QUESTION:")
        w(f"  {q['question']}")
        w("")
        w("GOLD SPAN (must answer the question, verbatim from the page):")
        for line in q["gold_span"].splitlines() or [q["gold_span"]]:
            w(f"  > {line}")
        if marker:
            w(f"  {marker}")
        w("")
        w("PAGE CONTEXT (raw extraction, as the generator saw it):")
        for line in ctx.splitlines():
            w(f"  | {line}")
        w("")
        w("  ACCEPT / REJECT: ______    notes: ______________________________")
        w("")

    w("=" * 78)
    (OUT / "HANDCHECK_20.txt").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT/'HANDCHECK_20.txt'} — {len(pick)} questions "
          f"({sum(1 for q in pick if q['page_stratum']=='table')} table / "
          f"{sum(1 for q in pick if q['page_stratum']=='prose')} prose)")


if __name__ == "__main__":
    main()
