"""
Fresh question pool for the structure ablation.

BIAS LABEL — READ THIS BEFORE QUOTING ANY NUMBER FROM THIS POOL.
    Questions are generated from page.get_text("text"), which IS ARM B's
    extraction path. This pool is therefore BIASED TOWARD ARM B. It is not
    neutral and must never be described as unbiased or bias-free. The bias is
    conservative: it works AGAINST the expected direction of the effect, so a
    surviving Arm A advantage is a lower bound, not an estimate. An Arm B
    advantage measured on this pool would be uninterpretable.

Gold is a SHORT SPAN (one sentence or one table row), not a chunk from either
arm, so neither arm's chunk boundaries define the target.

Every candidate must clear deterministic validation before it enters the pool.
Nothing is accepted on the generator's say-so.

Writes: ablation/out/fresh_raw.jsonl        every generator response, audit trail
        ablation/out/fresh_pool.json        the validated pool
        ablation/out/fresh_pool_rejects.csv why each candidate was dropped
"""

import json
import os
import random
import re
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

load_dotenv()
random.seed(0)
np.random.seed(0)

OUT = Path("ablation/out")
OUT.mkdir(parents=True, exist_ok=True)

GROQ_MODEL = "qwen/qwen3.8-27b"     # verified available; gpt-oss returns empty
N_PER_STRATUM = 60
PAGE_TEXT_LIMIT = 3000
SPAN_CAP_NORM = 98                  # Arm B p10 normalized length
SPAN_MIN_NORM = 40                  # below this a span stops locating anything

DEICTIC = [
    "table above", "table below", "the following", "the above", "this table",
    "as shown", "shown above", "shown below", "in the passage", "the passage",
    "the text above", "the document states", "the list above", "the list below",
    "the preceding", "herein", "said table", "this list", "this section states",
    "the excerpt", "the extract", "in the text", "based on the text",
    "according to the text", "the chart above", "this figure", "the figure above",
]

PROMPT = """You are building an evaluation set for a search system over Indian civil aviation regulatory documents.

Below is the raw text of one page from a regulatory document.

Write EXACTLY ONE question that a compliance officer would ask, whose answer is stated on this page, AND quote the shortest span of the page that contains the answer.

Rules for the question:
- It must stand alone. A reader who has never seen this page must understand it.
- Name the thing being looked up explicitly (the rule, the aircraft, the category, the fee).
- NEVER refer to "the table", "the following", "the above", "this list", "the text", or any other pointer to the page. The question must not reveal that a page was shown to you.
- Do not include the answer in the question.

Rules for the span:
- Copy it VERBATIM from the page text. Character for character.
- Keep it SHORT: one sentence, or one row's worth of values. Under 200 characters if you can.
- It must actually contain the answer.

Return ONLY a JSON object, no other text:
{{"question": "...", "answer_span": "..."}}

Page text:
<<<PAGE>>>
{page}
<<<END PAGE>>>
"""


def ask(client, prompt, retries=4):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400, temperature=0.0, seed=0,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            wait = 30 if "429" in str(e) else 3
            print(f"      retry {attempt + 1}: {str(e)[:80]} (waiting {wait}s)")
            time.sleep(wait)
    return None


def parse_json(raw):
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        o = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(o, dict) or "question" not in o or "answer_span" not in o:
        return None
    if not isinstance(o["question"], str) or not isinstance(o["answer_span"], str):
        return None
    return o


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

    # Stratify page selection on whether Arm A detected a real table there.
    # The generator never learns this; it only ever sees flat page text.
    table_pages, prose_pages = [], []
    for k, chunks in a_pages.items():
        (table_pages if any(c["is_table"] for c in chunks) else prose_pages).append(k)
    table_pages.sort()
    prose_pages.sort()
    print(f"pages: {len(table_pages)} table-bearing, {len(prose_pages)} prose-only")

    rng = random.Random(0)
    picked = ([("table", k) for k in rng.sample(table_pages, min(N_PER_STRATUM, len(table_pages)))]
              + [("prose", k) for k in rng.sample(prose_pages, min(N_PER_STRATUM, len(prose_pages)))])
    print(f"sampled {len(picked)} pages "
          f"({sum(1 for s, _ in picked if s == 'table')} table / "
          f"{sum(1 for s, _ in picked if s == 'prose')} prose)\n")

    flat = {}
    for p in list_pdfs():
        for pg in extract_flat(p):
            flat[(p.name, pg["page_number"])] = pg["text"]

    raw_f = (OUT / "fresh_raw.jsonl").open("w", encoding="utf-8")
    accepted, rejects = [], []
    qid = 0

    for i, (stratum, k) in enumerate(picked, 1):
        page_text = flat.get(k, "")[:PAGE_TEXT_LIMIT]
        rec = {"stratum": stratum, "file": k[0], "page": k[1]}

        if len(page_text) < 200:
            rejects.append({**rec, "stage": "V0_page_too_short", "detail": len(page_text)})
            continue

        raw = ask(client, PROMPT.format(page=page_text))
        raw_f.write(json.dumps({**rec, "raw": raw}, ensure_ascii=False) + "\n")
        raw_f.flush()

        obj = parse_json(raw)
        if obj is None:
            rejects.append({**rec, "stage": "V1_unparseable", "detail": (raw or "")[:120]})
            continue

        q, span = obj["question"].strip(), obj["answer_span"].strip()

        if "?" not in q or len(q) < 20:
            rejects.append({**rec, "stage": "V2_malformed_question", "detail": q[:120]})
            continue

        hit = next((d for d in DEICTIC if d in q.lower()), None)
        if hit:
            rejects.append({**rec, "stage": "V3_deictic", "detail": f"'{hit}' in: {q[:100]}"})
            continue

        span_n, page_n = normalize(span), normalize(page_text)
        if not span_n:
            rejects.append({**rec, "stage": "V4_empty_span", "detail": span[:120]})
            continue
        if coverage(span_n, page_n) < 1.0:
            rejects.append({**rec, "stage": "V4_span_not_verbatim", "detail": span[:120]})
            continue

        rec["span_norm_len"] = len(span_n)
        if len(span_n) < SPAN_MIN_NORM:
            rejects.append({**rec, "stage": "V5_span_too_short", "detail": len(span_n)})
            continue
        if len(span_n) > SPAN_CAP_NORM:
            rejects.append({**rec, "stage": "V5_span_over_cap", "detail": len(span_n)})
            continue

        a_ok = any(coverage(span_n, normalize(c["text"])) >= 1.0 for c in a_pages.get(k, []))
        b_ok = any(coverage(span_n, normalize(c["text"])) >= 1.0 for c in b_pages.get(k, []))
        if not a_ok:
            rejects.append({**rec, "stage": "V6_no_fit_armA", "detail": span[:120]})
            continue
        if not b_ok:
            rejects.append({**rec, "stage": "V7_no_fit_armB", "detail": span[:120]})
            continue

        # Discriminative: the span must locate ONE page, not many.
        pages_hit = {kk for kk, ch in a_pages.items()
                     if any(coverage(span_n, normalize(c["text"])) >= 1.0 for c in ch)}
        if len(pages_hit) != 1:
            rejects.append({**rec, "stage": "V8_ambiguous_span",
                            "detail": f"matches {len(pages_hit)} pages"})
            continue

        qid += 1
        accepted.append({
            "question_id": f"F{qid:03d}", "question": q, "gold_span": span,
            "gold_span_norm_len": len(span_n), "source_file": k[0],
            "page_number": k[1], "page_stratum": stratum,
            "generated_from": "get_text('text') raw page - ARM B EXTRACTION PATH",
        })
        print(f"  [{i}/{len(picked)}] {stratum} ACCEPT F{qid:03d} ({len(span_n)}c) {q[:62]}")
        time.sleep(0.25)

    raw_f.close()

    pd.DataFrame(rejects).to_csv(OUT / "fresh_pool_rejects.csv", index=False)
    (OUT / "fresh_pool.json").write_text(
        json.dumps(accepted, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"ACCEPTED {len(accepted)} / {len(picked)} sampled pages")
    print(f"  table-stratum: {sum(1 for a in accepted if a['page_stratum'] == 'table')}")
    print(f"  prose-stratum: {sum(1 for a in accepted if a['page_stratum'] == 'prose')}")
    if rejects:
        print("\nREJECTS by stage:")
        rc = pd.DataFrame(rejects).groupby(["stage", "stratum"]).size().unstack(fill_value=0)
        print(rc.to_string())
    print("=" * 74)


if __name__ == "__main__":
    main()
