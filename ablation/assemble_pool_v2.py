"""
Assemble the v2 pool from whatever generations actually exist.

The v2 table sweep was cut short by Groq's daily token quota (200,000 TPD),
which was exhausted partway through. Pages attempted after that point returned
empty content and are NOT samples - they are failures, and they are counted as
such here rather than silently folded into a denominator.

Table stratum : validated from fresh_raw_v2.jsonl (v2 label+value prompt)
Prose stratum : carried over unchanged from fresh_pool.json (v1 generic prompt)

Writes: ablation/out/fresh_pool_v2.json
        ablation/out/fresh_pool_v2_rejects.csv
        ablation/out/POOL_STATUS.txt
"""

import json
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd

from judgments import normalize, coverage
from fresh_pool import DEICTIC, parse_json, SPAN_MIN_NORM, SPAN_CAP_NORM
from notes import ATTRIBUTION, BIAS, PROMPT_ASYMMETRY, STRATUM_DROP

OUT = Path("ablation/out")

# Rejected at the hand-check stage (HANDCHECK_20.txt, model-annotated at the
# author's instruction). Listed here so a re-run of this script cannot
# resurrect them. Reason recorded beside each id.
HANDCHECK_REJECTS = {
    # first pass (Claude, at the author's instruction)
    "T013": "layout/order question, not regulatory; span straddles two reconstructed rows",
    # second pass (independent blind adversarial model review; source PDFs consulted)
    "T003": "question asks for 'the limit' (singular); span holds three severity thresholds (Low/Medium/High Limit columns) with no way to pick",
    "T004": "'Element 3-6' is fabricated - page number fused with a column header; question also leaks 'safety reporting environment'",
    "T006": "'for a Normal approach' is not on the page (the only 'Normal' is normal acceleration); span holds three thresholds, question singular",
    "T010": "'under the specified certification criteria' is a dangling pointer; the defined term (complex motor-powered aircraft) is on the previous page and absent from the question",
    "P005": "'EU's capacity building and implementation support framework' is fabricated from a chapter title; stripped, the question is generic",
}

TARGET_TABLE = 45
TARGET_PROSE = 45
TOTAL_TABLE_PAGES = 199


def main():
    arm_a = json.loads((OUT / "arm_a_chunks.json").read_text(encoding="utf-8"))
    arm_b = json.loads((OUT / "arm_b_chunks.json").read_text(encoding="utf-8"))
    ap, bp = defaultdict(list), defaultdict(list)
    a_chunks_by_page = defaultdict(list)
    for c in arm_a:
        ap[(c["source_file"], c["page_number"])].append(normalize(c["text"]))
        a_chunks_by_page[(c["source_file"], c["page_number"])].append(c)
    for c in arm_b:
        bp[(c["source_file"], c["page_number"])].append(normalize(c["text"]))

    def gold_stratum(k, span_norm):
        """'table' if the gold span sits inside an Arm A chunk that ingest
        flagged is_table, else 'prose'. This is the analysis label; the page
        the item was sampled from is kept separately as page_stratum."""
        for c in a_chunks_by_page.get(k, []):
            if coverage(span_norm, normalize(c["text"])) >= 1.0:
                return "table" if c["is_table"] else "prose"
        return "prose"

    recs = [json.loads(l) for l in
            (OUT / "fresh_raw_v2.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    table_recs = [r for r in recs if r["stratum"] == "table"]
    n_empty = sum(1 for r in table_recs if not r.get("raw"))
    n_real = len(table_recs) - n_empty

    accepted, rejects = [], []
    got = 0
    for r in table_recs:
        k = (r["file"], r["page"])
        rec = {"stratum": "table", "file": k[0], "page": k[1]}
        if not r.get("raw"):
            rejects.append({**rec, "stage": "QUOTA_empty_response",
                            "detail": "Groq daily token quota exhausted"})
            continue
        o = parse_json(r["raw"])
        if o is None:
            rejects.append({**rec, "stage": "V1_unparseable", "detail": r["raw"][:120]})
            continue
        q, span = o["question"].strip(), o["answer_span"].strip()
        if "?" not in q or len(q) < 20:
            rejects.append({**rec, "stage": "V2_malformed_question", "detail": q[:120]})
            continue
        hit = next((d for d in DEICTIC if d in q.lower()), None)
        if hit:
            rejects.append({**rec, "stage": "V3_deictic", "detail": f"'{hit}': {q[:90]}"})
            continue
        sn = normalize(span)
        if not sn:
            rejects.append({**rec, "stage": "V4_empty_span", "detail": span[:120]})
            continue
        if len(sn) < SPAN_MIN_NORM:
            rejects.append({**rec, "stage": "V5_span_too_short", "detail": len(sn)})
            continue
        if len(sn) > SPAN_CAP_NORM:
            rejects.append({**rec, "stage": "V5_span_over_cap", "detail": len(sn)})
            continue
        if not any(coverage(sn, c) >= 1.0 for c in ap.get(k, [])):
            rejects.append({**rec, "stage": "V6_no_fit_armA", "detail": span[:120]})
            continue
        if not any(coverage(sn, c) >= 1.0 for c in bp.get(k, [])):
            rejects.append({**rec, "stage": "V7_no_fit_armB", "detail": span[:120]})
            continue
        hits = sum(1 for kk, ch in ap.items()
                   if any(coverage(sn, c) >= 1.0 for c in ch))
        if hits != 1:
            rejects.append({**rec, "stage": "V8_ambiguous_span",
                            "detail": f"matches {hits} pages"})
            continue
        got += 1
        accepted.append({
            "question_id": f"T{got:03d}", "question": q, "gold_span": span,
            "gold_span_norm_len": len(sn), "source_file": k[0], "page_number": k[1],
            "page_stratum": "table", "gold_stratum": gold_stratum(k, sn),
            "prompt_version": "v2_label_plus_value",
            "generated_from": "get_text('text') raw page - ARM B EXTRACTION PATH",
        })

    # Prose carried over from v1, unchanged.
    v1 = json.loads((OUT / "fresh_pool.json").read_text(encoding="utf-8"))
    prose = [q for q in v1 if q["page_stratum"] == "prose"]
    for i, q in enumerate(prose, 1):
        q = dict(q)
        q["question_id"] = f"P{i:03d}"
        q["prompt_version"] = "v1_generic"
        q["gold_stratum"] = gold_stratum((q["source_file"], q["page_number"]),
                                         normalize(q["gold_span"]))
        accepted.append(q)

    # DROP any item whose gold stratum disagrees with its page stratum. Moving
    # them across strata (an earlier revision did) mixes generator prompts
    # within a stratum. IDs are assigned before the drop so they stay stable.
    dropped = [a["question_id"] for a in accepted if a["page_stratum"] != a["gold_stratum"]]
    n_before = len(accepted)
    accepted = [a for a in accepted if a["page_stratum"] == a["gold_stratum"]]
    hc_dropped = [a["question_id"] for a in accepted if a["question_id"] in HANDCHECK_REJECTS]
    accepted = [a for a in accepted if a["question_id"] not in HANDCHECK_REJECTS]
    n_page_tab = sum(1 for a in accepted if a["page_stratum"] == "table")
    n_tab = sum(1 for a in accepted if a["gold_stratum"] == "table")
    n_pro = len(accepted) - n_tab
    assert all(a["page_stratum"] == a["gold_stratum"] for a in accepted)

    (OUT / "fresh_pool_v2.json").write_text(
        json.dumps(accepted, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rejects).to_csv(OUT / "fresh_pool_v2_rejects.csv", index=False)

    L = []
    w = L.append
    w("=" * 78)
    w("POOL v2 STATUS - INCOMPLETE")
    w("=" * 78)
    w("")
    w("RUN WAS CUT SHORT BY AN EXTERNAL QUOTA, NOT BY THE CORPUS.")
    w("  Groq free tier: 200,000 tokens per day for qwen/qwen3.8-27b.")
    w("  The quota was exhausted mid-sweep:")
    w("    429 ... on tokens per day (TPD): Limit 200000, Used 199856")
    w("")
    w(f"  table pages attempted     : {len(table_recs)} of {TOTAL_TABLE_PAGES}")
    w(f"    real responses          : {n_real}")
    w(f"    empty (post-quota)      : {n_empty}   <- failures, not samples")
    w(f"    pages never attempted   : {TOTAL_TABLE_PAGES - len(table_recs)}")
    w("")
    w("  POOL COUNTS (page_stratum == gold_stratum for every item):")
    w(f"    TABLE : {n_tab}   (target {TARGET_TABLE};  v2 label+value prompt)")
    w(f"    PROSE : {n_pro}   (target {TARGET_PROSE};  v1 generic prompt, carry-over)")
    w(f"    TOTAL : {len(accepted)}")
    w("")
    w(f"  DROPPED {len(dropped)} of {n_before} validated items whose gold sits in a PROSE chunk")
    w("  although they were sampled from a table-bearing page:")
    w(f"    {', '.join(dropped) if dropped else '(none)'}")
    w("  An earlier revision MOVED these into the prose stratum. That is RETRACTED:")
    w("  it mixed two generator prompts inside one stratum. Dropping keeps each")
    w("  stratum on a single prompt. IDs were assigned before the drop, so the")
    w("  remaining table IDs have gaps where these three were.")
    w("")
    w(f"  DROPPED {len(hc_dropped)} at the hand-check stage (HANDCHECK_20.txt, model-annotated at")
    w("  the author's instruction - not a human review):")
    for qid in hc_dropped:
        w(f"    {qid}: {HANDCHECK_REJECTS[qid]}")
    w("")
    w(f"  acceptance rate on real table-page responses (after the drop): {n_tab}/{n_real} = "
      f"{n_tab/max(n_real,1):.3f}")
    w(f"  projection if all {TOTAL_TABLE_PAGES} table pages were swept at that rate: "
      f"{n_tab/max(n_real,1)*TOTAL_TABLE_PAGES:.0f}")
    w("  That projection is an ARITHMETIC EXTRAPOLATION, not a measurement, and")
    w("  must not be reported as a result.")
    w("")
    w("REJECTS by stage:")
    if rejects:
        for stage, n in Counter(r["stage"] for r in rejects).most_common():
            w(f"  {stage:<26} {n:>4}")
    w("")
    w("=" * 78)
    w("")
    w(BIAS)
    w("")
    w(PROMPT_ASYMMETRY)
    w("")
    w(STRATUM_DROP)
    w("")
    w(ATTRIBUTION)

    txt = "\n".join(L)
    (OUT / "POOL_STATUS.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
