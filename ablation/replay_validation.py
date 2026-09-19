"""
Replay the fresh-pool validation at different span-length bands.

Costs nothing: every generator response is already saved in fresh_raw.jsonl, so
the whole validation pipeline can be re-run offline across a grid of (min, cap)
settings. This turns "the cap was too tight" into a measured tradeoff curve
rather than a guess.

The two settings are NOT symmetric:
  - raising the CAP admits longer spans, which risks failing V7 (the span no
    longer fits inside any single Arm B chunk) — that failure is the thing the
    experiment is supposed to measure, so admitting it would be circular.
  - lowering the MIN admits shorter spans, which risks failing V8 (the span
    stops identifying one page) — that failure makes "relevant" ill-defined.

So the useful band is bounded by real failures at both ends, and this prints
where they bite.

Writes: ablation/out/validation_yield.csv
"""

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from judgments import normalize, coverage
from fresh_pool import DEICTIC, parse_json

OUT = Path("ablation/out")

MINS = [0, 20, 30, 40]
CAPS = [98, 130, 160, 200, 240]


def main():
    arm_a = json.loads((OUT / "arm_a_chunks.json").read_text(encoding="utf-8"))
    arm_b = json.loads((OUT / "arm_b_chunks.json").read_text(encoding="utf-8"))
    a_pages, b_pages = defaultdict(list), defaultdict(list)
    for c in arm_a:
        a_pages[(c["source_file"], c["page_number"])].append(normalize(c["text"]))
    for c in arm_b:
        b_pages[(c["source_file"], c["page_number"])].append(normalize(c["text"]))

    raws = [json.loads(l) for l in
            (OUT / "fresh_raw.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"replaying {len(raws)} saved generations\n")

    # Stage-independent validation, computed once per candidate.
    cands = []
    for r in raws:
        obj = parse_json(r.get("raw"))
        if obj is None:
            continue
        q, span = obj["question"].strip(), obj["answer_span"].strip()
        if "?" not in q or len(q) < 20:
            continue
        if any(d in q.lower() for d in DEICTIC):
            continue
        span_n = normalize(span)
        if not span_n:
            continue
        k = (r["file"], r["page"])
        # verbatim check against the page text is already implied by V4 in the
        # original run; re-derive it from Arm A page chunks as a proxy is NOT
        # valid, so carry only candidates whose span sits in some Arm A chunk.
        a_ok = any(coverage(span_n, c) >= 1.0 for c in a_pages.get(k, []))
        b_ok = any(coverage(span_n, c) >= 1.0 for c in b_pages.get(k, []))
        pages_hit = sum(1 for kk, ch in a_pages.items()
                        if any(coverage(span_n, c) >= 1.0 for c in ch))
        cands.append({"stratum": r["stratum"], "len": len(span_n),
                      "a_ok": a_ok, "b_ok": b_ok, "pages_hit": pages_hit})

    df = pd.DataFrame(cands)
    print(f"{len(df)} candidates passed parse/format/deictic checks\n")

    rows = []
    print(f"{'min':>5}{'cap':>6}{'total':>8}{'table':>7}{'prose':>7}"
          f"{'lost:len':>10}{'lost:V7':>9}{'lost:V8':>9}")
    print("-" * 61)
    for mn in MINS:
        for cap in CAPS:
            inband = df[(df.len >= mn) & (df.len <= cap)]
            ok = inband[inband.a_ok & inband.b_ok & (inband.pages_hit == 1)]
            lost_len = len(df) - len(inband)
            lost_v7 = int((inband.a_ok & ~inband.b_ok).sum())
            lost_v8 = int((inband.a_ok & inband.b_ok & (inband.pages_hit != 1)).sum())
            rows.append({"min": mn, "cap": cap, "total": len(ok),
                         "table": int((ok.stratum == "table").sum()),
                         "prose": int((ok.stratum == "prose").sum()),
                         "lost_len": lost_len, "lost_V7_no_fit_armB": lost_v7,
                         "lost_V8_ambiguous": lost_v8})
            print(f"{mn:>5}{cap:>6}{len(ok):>8}"
                  f"{int((ok.stratum=='table').sum()):>7}"
                  f"{int((ok.stratum=='prose').sum()):>7}"
                  f"{lost_len:>10}{lost_v7:>9}{lost_v8:>9}")
        print("-" * 61)

    pd.DataFrame(rows).to_csv(OUT / "validation_yield.csv", index=False)
    print(f"\nwrote {OUT/'validation_yield.csv'}")

    print("\nspan length distribution of all parsed candidates, by stratum:")
    for s in ("table", "prose"):
        x = df[df.stratum == s].len
        if len(x):
            print(f"  {s}: n={len(x)} min={x.min()} p25={x.quantile(.25):.0f} "
                  f"median={x.median():.0f} p75={x.quantile(.75):.0f} max={x.max()}")

    print("\nV7 (span does not fit any single Arm B chunk) by span length:")
    for lo, hi in ((0, 98), (99, 130), (131, 160), (161, 200), (201, 240), (241, 10**6)):
        sub = df[(df.len >= lo) & (df.len <= hi)]
        if len(sub):
            print(f"  {lo:>4}-{hi if hi < 10**6 else '+':<5} n={len(sub):>3}  "
                  f"fails V7: {int((~sub.b_ok).sum()):>3}  ({(~sub.b_ok).mean():.3f})")


if __name__ == "__main__":
    main()
