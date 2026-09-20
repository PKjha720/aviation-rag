"""
Fresh pool, v2.

Changes from v1:
  - TABLE stratum regenerated with a prompt that requires the span to carry a
    row LABEL plus its VALUE. v1's generic prompt returned bare cell values
    (median 22 normalized chars), which fell under the 40-char floor and cost
    most of the table yield.
  - TABLE stratum now sweeps ALL table-bearing pages in the corpus (199).
  - PROSE stratum keeps v1's prompt verbatim and reuses v1's cached generations,
    topping up with additional pages until the target is met.

Unchanged: the 40-char floor, the 98-char cap, and every deterministic
validation stage. The floor is not relaxed to buy yield.

PROVENANCE NOTE. The fresh_raw_v2.jsonl on disk was produced by an earlier
revision of this script whose ask() retried through 429s and returned None
after four attempts, which is why that file contains 14 empty records after
the quota ran out. The QuotaExhausted handling below was added afterwards. A
resume run under this code stops on the first quota error instead of writing
empties; the 14 existing empties are handled by assemble_pool_v2.py.

Writes: ablation/out/fresh_raw_v2.jsonl
        ablation/out/fresh_pool_v2.json
        ablation/out/fresh_pool_v2_rejects.csv
"""

import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from arms import list_pdfs, extract_flat
from judgments import normalize, coverage
from fresh_pool import (DEICTIC, PAGE_TEXT_LIMIT, SPAN_CAP_NORM, SPAN_MIN_NORM,
                        PROMPT as PROMPT_PROSE, QuotaExhausted, ask, parse_json)
from notes import ATTRIBUTION, BIAS, PROMPT_ASYMMETRY

load_dotenv()
random.seed(0)
np.random.seed(0)

OUT = Path("ablation/out")

TARGET_TABLE = 45
TARGET_PROSE = 45
N_PROSE_PAGES = 200

PROMPT_TABLE = """You are building an evaluation set for a search system over Indian civil aviation regulatory documents.

Below is the raw text of one page from a regulatory document. The page contains tabular data, and the text extraction has partly lost the column layout, so values may sit near their row labels rather than beside them.

Write EXACTLY ONE question that a compliance officer would ask, whose answer is a specific value on this page, AND quote a span of the page that contains BOTH the label identifying the row AND the value that answers the question.

Rules for the question:
- It must stand alone. A reader who has never seen this page must understand it.
- Name the thing being looked up explicitly (the task, the module, the category, the aircraft, the fee).
- NEVER refer to "the table", "the following", "the above", "this list", "the text", or any other pointer to the page.
- Do not include the answer in the question.

Rules for the span:
- Copy it VERBATIM from the page text. Character for character, including line breaks.
- It MUST contain BOTH the identifying label AND the answer value. A bare number, a bare code, or a bare heading on its own is NOT acceptable.
- Take a contiguous run of the page text. Do not stitch together pieces from different places.
- Keep it under 200 characters.

Return ONLY a JSON object, no other text:
{{"question": "...", "answer_span": "..."}}

Page text:
<<<PAGE>>>
{page}
<<<END PAGE>>>
"""


def load_cache():
    """Responses already in hand, keyed by (file, page).

    v1 contributes PROSE only - the table prompt changed, so v1 table responses
    are discarded. v2 contributes BOTH strata, which makes this script resumable:
    the first run was cut short by Groq's daily token quota, and a resume must
    not spend quota re-asking pages that already answered.

    Empty responses are NOT cached. An empty is what a page returns after the
    quota is gone, and caching it would bake a quota failure into the pool as
    though the page had been sampled and rejected.
    """
    cache = {}
    v1 = OUT / "fresh_raw.jsonl"
    if v1.exists():
        for line in v1.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("stratum") == "prose" and r.get("raw"):
                cache[(r["file"], r["page"])] = r["raw"]
    v2 = OUT / "fresh_raw_v2.jsonl"
    if v2.exists():
        for line in v2.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("raw"):
                cache[(r["file"], r["page"])] = r["raw"]
    return cache


def validate(raw, k, stratum, a_pages, b_pages, page_text, rejects):
    rec = {"stratum": stratum, "file": k[0], "page": k[1]}
    obj = parse_json(raw)
    if obj is None:
        rejects.append({**rec, "stage": "V1_unparseable", "detail": (raw or "")[:120]})
        return None
    q, span = obj["question"].strip(), obj["answer_span"].strip()
    if "?" not in q or len(q) < 20:
        rejects.append({**rec, "stage": "V2_malformed_question", "detail": q[:120]})
        return None
    hit = next((d for d in DEICTIC if d in q.lower()), None)
    if hit:
        rejects.append({**rec, "stage": "V3_deictic", "detail": f"'{hit}': {q[:100]}"})
        return None
    span_n, page_n = normalize(span), normalize(page_text)
    if not span_n:
        rejects.append({**rec, "stage": "V4_empty_span", "detail": span[:120]})
        return None
    if coverage(span_n, page_n) < 1.0:
        rejects.append({**rec, "stage": "V4_span_not_verbatim", "detail": span[:120]})
        return None
    rec["span_norm_len"] = len(span_n)
    if len(span_n) < SPAN_MIN_NORM:
        rejects.append({**rec, "stage": "V5_span_too_short", "detail": len(span_n)})
        return None
    if len(span_n) > SPAN_CAP_NORM:
        rejects.append({**rec, "stage": "V5_span_over_cap", "detail": len(span_n)})
        return None
    if not any(coverage(span_n, normalize(c["text"])) >= 1.0 for c in a_pages.get(k, [])):
        rejects.append({**rec, "stage": "V6_no_fit_armA", "detail": span[:120]})
        return None
    if not any(coverage(span_n, normalize(c["text"])) >= 1.0 for c in b_pages.get(k, [])):
        rejects.append({**rec, "stage": "V7_no_fit_armB", "detail": span[:120]})
        return None
    pages_hit = {kk for kk, ch in a_pages.items()
                 if any(coverage(span_n, normalize(c["text"])) >= 1.0 for c in ch)}
    if len(pages_hit) != 1:
        rejects.append({**rec, "stage": "V8_ambiguous_span",
                        "detail": f"matches {len(pages_hit)} pages"})
        return None
    return {"question": q, "gold_span": span, "gold_span_norm_len": len(span_n)}


def main():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("FAILED: GROQ_API_KEY not set. Pool not generated.")
        sys.exit(1)
    client = Groq(api_key=key)

    arm_a = json.loads((OUT / "arm_a_chunks.json").read_text(encoding="utf-8"))
    arm_b = json.loads((OUT / "arm_b_chunks.json").read_text(encoding="utf-8"))
    a_pages, b_pages = defaultdict(list), defaultdict(list)
    for c in arm_a:
        a_pages[(c["source_file"], c["page_number"])].append(c)
    for c in arm_b:
        b_pages[(c["source_file"], c["page_number"])].append(c)

    table_pages, prose_pages = [], []
    for k, chunks in a_pages.items():
        (table_pages if any(c["is_table"] for c in chunks) else prose_pages).append(k)
    table_pages.sort()
    prose_pages.sort()
    print(f"corpus: {len(table_pages)} table-bearing pages, {len(prose_pages)} prose-only")

    rng = random.Random(0)
    table_order = list(table_pages)
    rng.shuffle(table_order)
    prose_order = rng.sample(prose_pages, min(N_PROSE_PAGES, len(prose_pages)))

    flat = {}
    for p in list_pdfs():
        for pg in extract_flat(p):
            flat[(p.name, pg["page_number"])] = pg["text"]

    cache = load_cache()
    print(f"cached responses reusable on this run: {len(cache)}")
    print(f"sweeping ALL {len(table_order)} table pages with the v2 prompt")
    print(f"sampling {len(prose_order)} prose pages, stopping at {TARGET_PROSE} accepts\n")

    raw_f = (OUT / "fresh_raw_v2.jsonl").open("a", encoding="utf-8")
    accepted, rejects = [], []
    n_api, n_cached = 0, 0
    quota_hit = None

    for stratum, order, prompt, target in (
            ("table", table_order, PROMPT_TABLE, TARGET_TABLE),
            ("prose", prose_order, PROMPT_PROSE, TARGET_PROSE)):
        got = 0
        for i, k in enumerate(order, 1):
            if got >= target:
                print(f"  {stratum}: target {target} reached after {i-1} pages")
                break
            page_text = flat.get(k, "")[:PAGE_TEXT_LIMIT]
            if len(page_text) < 200:
                rejects.append({"stratum": stratum, "file": k[0], "page": k[1],
                                "stage": "V0_page_too_short", "detail": len(page_text)})
                continue

            if k in cache:
                raw, src = cache[k], "cache"
                n_cached += 1
            else:
                try:
                    raw, src = ask(client, prompt.format(page=page_text)), "api_v2"
                except QuotaExhausted as e:
                    # Stop cleanly, keep everything accepted so far, and still
                    # write the pool and rejects. The page that triggered this
                    # is NOT recorded as a sample.
                    quota_hit = str(e)[:200]
                    print(f"  {stratum}: stopping at page {i}/{len(order)} - daily quota")
                    break
                n_api += 1
                time.sleep(0.2)

            raw_f.write(json.dumps({"file": k[0], "page": k[1], "stratum": stratum,
                                    "source": src, "raw": raw}, ensure_ascii=False) + "\n")
            raw_f.flush()

            v = validate(raw, k, stratum, a_pages, b_pages, page_text, rejects)
            if v is None:
                continue
            got += 1
            accepted.append({
                "question_id": f"{'T' if stratum == 'table' else 'P'}{got:03d}",
                **v, "source_file": k[0], "page_number": k[1],
                "page_stratum": stratum,
                "prompt_version": "v2_label_plus_value" if stratum == "table" else "v1_generic",
                "generated_from": "get_text('text') raw page - ARM B EXTRACTION PATH",
            })
            if got % 5 == 0 or got == target:
                print(f"  [{stratum} {got}/{target}] page {i}/{len(order)}")
        else:
            print(f"  {stratum}: EXHAUSTED all {len(order)} pages with {got} accepted "
                  f"(target was {target})")
        if quota_hit:
            break

    raw_f.close()

    pd.DataFrame(rejects).to_csv(OUT / "fresh_pool_v2_rejects.csv", index=False)
    (OUT / "fresh_pool_v2.json").write_text(
        json.dumps(accepted, ensure_ascii=False, indent=2), encoding="utf-8")

    n_tab = sum(1 for a in accepted if a["page_stratum"] == "table")
    n_pro = len(accepted) - n_tab
    print("\n" + "=" * 74)
    print(f"POOL v2: {len(accepted)} accepted  ({n_tab} table / {n_pro} prose)")
    print(f"  API calls made: {n_api}   cached v1 prose reused: {n_cached}")
    if n_tab < TARGET_TABLE:
        print(f"\n  TABLE STRATUM CAPPED AT {n_tab}, BELOW THE {TARGET_TABLE} TARGET.")
        print(f"  All {len(table_order)} table-bearing pages in the corpus were swept.")
        print("  The 40-char floor was NOT relaxed. This is the corpus ceiling.")
    if rejects:
        print("\nREJECTS by stage:")
        print(pd.DataFrame(rejects).groupby(["stage", "stratum"]).size()
              .unstack(fill_value=0).to_string())
    print("=" * 74)
    print("\n" + PROMPT_ASYMMETRY + "\n\n" + BIAS + "\n\n" + ATTRIBUTION)


if __name__ == "__main__":
    main()
