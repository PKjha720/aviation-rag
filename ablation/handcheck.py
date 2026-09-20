"""
Write 20 fresh-pool questions out for human review.

The table half is NOT random: it is weighted toward items the row-pairing check
flagged (different_row first, then undetermined), because those are the ones
where a gold could be WRONG rather than merely weak. The prose half is a seeded
random sample.

Each table entry shows the find_tables() row-wise reconstruction next to the raw
column-wise page text, so the two views can be compared without opening the PDF.

Writes: ablation/out/HANDCHECK_20.txt
"""

import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from arms import list_pdfs, extract_flat
from judgments import _NON_ALNUM
from notes import ATTRIBUTION, BIAS, PROMPT_ASYMMETRY, STRATUM_RELABEL

OUT = Path("ablation/out")
CONTEXT = 420
N_TABLE = 10
N_PROSE = 10


def norm_map(text):
    """Normalized string plus a map back to raw indices, so a span validated on
    normalized text can still be shown in raw context when it straddles line
    breaks."""
    out, idx = [], []
    for i, ch in enumerate(text):
        c = ch.lower()
        if not _NON_ALNUM.match(c):
            out.append(c)
            idx.append(i)
    return "".join(out), idx


def locate(page_text, span):
    pn, pmap = norm_map(page_text)
    sn, _ = norm_map(span)
    if not sn:
        return None
    j = pn.find(sn)
    if j < 0:
        return None
    return pmap[j], pmap[min(j + len(sn), len(pmap)) - 1] + 1


def main():
    src = OUT / "fresh_pool_v2.json"
    if not src.exists():
        src = OUT / "fresh_pool.json"
    pool = json.loads(src.read_text(encoding="utf-8"))
    print(f"pool: {src.name} ({len(pool)} questions)")

    rp = {}
    rp_path = OUT / "row_pairing_check.csv"
    if rp_path.exists():
        for _, r in pd.read_csv(rp_path).iterrows():
            rp[r["question_id"]] = r.to_dict()
        print(f"row-pairing verdicts loaded: {len(rp)}")
    else:
        print("row-pairing check NOT found - table entries will show 'not run'")

    # Split on the GOLD, not the page. Older pool files lack gold_stratum;
    # fall back to page_stratum so the script still runs on them.
    strat = lambda q: q.get("gold_stratum", q["page_stratum"])
    tab = [q for q in pool if strat(q) == "table"]
    pro = [q for q in pool if strat(q) == "prose"]

    def priority(q):
        v = rp.get(q["question_id"], {})
        verdict = str(v.get("verdict"))
        in_tab = bool(v.get("gold_in_table_chunk", False))
        if verdict == "different_row":
            return 0                      # label/value may come from two rows
        if verdict == "undetermined" and in_tab:
            return 1                      # tabular content we could not place
        if verdict == "undetermined":
            return 2                      # only reachable on pool files lacking gold_stratum
        return 3                          # same_row
    tab.sort(key=lambda q: (priority(q), q["question_id"]))
    rng = random.Random(0)
    pick_t = tab[:N_TABLE]
    pick_p = rng.sample(pro, min(N_PROSE, len(pro)))
    pick = pick_t + sorted(pick_p, key=lambda q: q["question_id"])

    flat = {}
    for p in list_pdfs():
        for pg in extract_flat(p):
            flat[(p.name, pg["page_number"])] = pg["text"]

    n_unloc = 0
    L = []
    w = L.append
    w("=" * 78)
    w("HAND-CHECK: 20 QUESTIONS FROM THE FRESH POOL (v2)")
    w("=" * 78)
    w("")
    w(BIAS)
    w("")
    w(PROMPT_ASYMMETRY)
    w("")
    w(STRATUM_RELABEL)
    w("")
    w(ATTRIBUTION)
    w("")
    w("-" * 78)
    w("SELECTION: the table half is NOT random. It is weighted toward items the")
    w("row-pairing check flagged different_row or undetermined, because those are")
    w("the ones where the gold may be WRONG rather than merely weak. The prose")
    w("half is a seeded random sample. Do NOT read the flag rate in this file as")
    w("the pool's flag rate - row_pairing_check.csv has that.")
    w("")
    w("ROW-PAIRING VERDICTS ARE EVIDENCE, NOT GROUND TRUTH. find_tables() is Arm")
    w("A's own view of the page and can itself be wrong. Nothing was dropped on")
    w("the strength of a verdict.")
    w("")
    w("Check each one:")
    w("  1. Does the question stand alone, with no pointer to a page you cannot see?")
    w("  2. Does the gold span actually answer it?")
    w("  3. TABLE ITEMS: does the span pair a label with the value from the SAME")
    w("     row? Compare the raw text against the reconstruction shown above it.")
    w("")
    w("Mark any you reject and I will drop them and recount before retrieval runs.")
    w("-" * 78)
    w("")

    for i, q in enumerate(pick, 1):
        page_text = flat.get((q["source_file"], q["page_number"]), "")
        loc = locate(page_text, q["gold_span"])
        if loc is None:
            ctx = page_text[:CONTEXT * 2]
            marker = "(span not locatable for display; it passed normalized validation)"
            n_unloc += 1
        else:
            a, b = loc
            s_, e_ = max(0, a - CONTEXT), min(len(page_text), b + CONTEXT)
            ctx = (page_text[s_:a] + ">>>SPAN>>>" + page_text[a:b]
                   + "<<<SPAN<<<" + page_text[b:e_])
            marker = ""

        v = rp.get(q["question_id"], {})
        verdict = v.get("verdict", "not run" if strat(q) == "table" else "n/a")

        w("=" * 78)
        w(f"[{i:02d}]  {q['question_id']}   gold_stratum={strat(q)}   "
          f"page_stratum={q['page_stratum']}   span={q['gold_span_norm_len']} norm chars   "
          f"prompt={q.get('prompt_version', 'v1_generic')}")
        w(f"      {q['source_file']}  page {q['page_number']}")
        if strat(q) == "table":
            extra = ""
            if v:
                extra = (f"   (best-row token coverage {v.get('best_row_cov')}, "
                         f"two-row {v.get('pair_row_cov')})")
            w(f"      ROW-PAIRING: {str(verdict).upper()}{extra}")
            if v:
                w(f"      GOLD SITS IN: "
                  + ("a TABLE chunk" if v.get("gold_in_table_chunk")
                     else "a PROSE chunk (page has a table elsewhere)"))
        w("-" * 78)
        w("QUESTION:")
        w(f"  {q['question']}")
        w("")
        w("GOLD SPAN (must answer the question, verbatim from the page):")
        for line in (q["gold_span"].splitlines() or [q["gold_span"]]):
            w(f"  > {line}")
        if marker:
            w(f"  {marker}")
        w("")

        if strat(q) == "table" and v:
            w("find_tables() ROW-WISE RECONSTRUCTION (Arm A's view of this page):")
            br = str(v.get("best_row_text", "") or "").strip()
            pr_ = str(v.get("pair_row_text", "") or "").strip()
            w(f"  best-matching row : {br if br else '(none located)'}")
            if str(verdict) == "different_row" and pr_:
                w(f"  second row needed : {pr_}")
                w("  ^^ the span's content required TWO rows. Check whether the label")
                w("     and the value it pairs actually belong together.")
            elif str(verdict) == "undetermined":
                if v.get("gold_in_table_chunk"):
                    w("  ^^ the gold IS tabular content but could not be placed in any")
                    w("     reconstructed row. Worth your attention.")
                else:
                    w("  ^^ the gold is PROSE, so there is no row to match. This item")
                    w("     should be counted in the prose-gold stratum, not here; if it")
                    w("     appears, the pool file predates the gold_stratum relabel.")
            w("")

        w("RAW PAGE CONTEXT (column-wise extraction, as the generator saw it):")
        for line in ctx.splitlines():
            w(f"  | {line}")
        w("")
        w("  ACCEPT / REJECT: ______    notes: ______________________________")
        w("")

    w("=" * 78)
    (OUT / "HANDCHECK_20.txt").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT/'HANDCHECK_20.txt'} - {len(pick)} questions "
          f"({len(pick_t)} table / {len(pick_p)} prose)")
    if rp:
        shown = [str(rp.get(q["question_id"], {}).get("verdict")) for q in pick_t]
        print(f"  table half verdicts shown: {dict(Counter(shown))}")
    if n_unloc:
        print(f"  note: {n_unloc}/{len(pick)} spans not locatable for raw display")


if __name__ == "__main__":
    main()
