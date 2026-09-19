"""
Chunk-size parity sweep.

Finds the Arm B token window whose resulting chunk-length distribution matches
Arm A's within 10% on BOTH mean and median characters. If no window satisfies
that, this script says so and does not pick a "closest" one.

Characters are the matching unit: the arms differ in how they tokenize-by-
construction, so tokens are not a common scale, and character length is what
determines how much evidence a chunk can physically hold.

Flat extraction and tokenization are done once and cached; each candidate
window is then pure slicing over the cached offsets.

Writes: ablation/out/parity_sweep.csv
        ablation/out/length_distributions.txt
"""

import json
import random
import statistics as st
from pathlib import Path

import numpy as np

from arms import list_pdfs, extract_flat, _TOK, token_len

random.seed(0)
np.random.seed(0)

OUT = Path("ablation/out")
OUT.mkdir(parents=True, exist_ok=True)

TOLERANCE = 0.10
OVERLAP_RATIO = 64 / 512  # Arm A's overlap ratio, carried over


def cache_pages():
    """[(page_text, offsets)] for every page of every PDF, flat extraction."""
    cached = []
    for p in list_pdfs():
        for page in extract_flat(p):
            text = page["text"]
            enc = _TOK(text, return_offsets_mapping=True, add_special_tokens=False)
            cached.append((text, enc["offset_mapping"]))
    return cached


def window_lengths(cached, window: int, overlap: int) -> list[int]:
    """Char lengths Arm B would produce at this window, without materializing
    the chunk text."""
    stride = window - overlap
    lens = []
    for text, offsets in cached:
        if not offsets:
            continue
        if len(offsets) <= window:
            s = text.strip()
            if s:
                lens.append(len(s))
            continue
        for start in range(0, len(offsets), stride):
            sl = offsets[start:start + window]
            if not sl:
                break
            piece = text[sl[0][0]:sl[-1][1]].strip()
            if piece:
                lens.append(len(piece))
            if start + window >= len(offsets):
                break
    return lens


def describe(lens: list[int]) -> dict:
    a = np.array(lens)
    return {
        "n": len(a),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "max": int(a.max()),
        "min": int(a.min()),
        "std": float(a.std(ddof=1)),
    }


def main():
    arm_a = json.loads((OUT / "arm_a_chunks.json").read_text(encoding="utf-8"))
    a_lens = [c["char_count"] for c in arm_a]
    a_stats = describe(a_lens)

    print("Arm A (current pipeline, unchanged)")
    print(f"  n={a_stats['n']}  mean={a_stats['mean']:.1f}  "
          f"median={a_stats['median']:.1f}  std={a_stats['std']:.1f}")
    print(f"  target band (+/-{TOLERANCE:.0%}): "
          f"mean [{a_stats['mean']*(1-TOLERANCE):.1f}, {a_stats['mean']*(1+TOLERANCE):.1f}]  "
          f"median [{a_stats['median']*(1-TOLERANCE):.1f}, {a_stats['median']*(1+TOLERANCE):.1f}]")

    print("\nCaching flat extraction + tokenization once...")
    cached = cache_pages()
    print(f"  {len(cached)} pages cached")

    rows = []
    print(f"\n{'window':>7}{'overlap':>8}{'n':>8}{'mean':>9}{'median':>9}"
          f"{'mean_err':>10}{'med_err':>9}{'both<10%':>10}")
    print("-" * 70)
    for window in range(60, 221, 4):
        overlap = max(1, round(window * OVERLAP_RATIO))
        lens = window_lengths(cached, window, overlap)
        s = describe(lens)
        me = (s["mean"] - a_stats["mean"]) / a_stats["mean"]
        qe = (s["median"] - a_stats["median"]) / a_stats["median"]
        ok = abs(me) <= TOLERANCE and abs(qe) <= TOLERANCE
        rows.append({"window": window, "overlap": overlap, **s,
                     "mean_err": me, "median_err": qe, "within_tol": ok})
        print(f"{window:>7}{overlap:>8}{s['n']:>8}{s['mean']:>9.1f}"
              f"{s['median']:>9.1f}{me:>+10.1%}{qe:>+9.1%}{str(ok):>10}")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "parity_sweep.csv", index=False)

    ok_rows = df[df.within_tol]
    print("\n" + "=" * 70)
    if len(ok_rows) == 0:
        print("NO WINDOW SATISFIES THE 10% PARITY CONSTRAINT ON BOTH MEAN AND MEDIAN.")
        print("Reporting the closest candidates by combined absolute error; a")
        print("window has NOT been selected.")
        df["combined"] = df.mean_err.abs() + df.median_err.abs()
        print(df.nsmallest(5, "combined")[
            ["window", "overlap", "n", "mean", "median", "mean_err", "median_err"]
        ].to_string(index=False))
        chosen = None
    else:
        # Among qualifying windows, take the one with smallest combined error.
        ok = ok_rows.copy()
        ok["combined"] = ok.mean_err.abs() + ok.median_err.abs()
        best = ok.nsmallest(1, "combined").iloc[0]
        chosen = int(best.window)
        print(f"{len(ok_rows)} window(s) satisfy parity. "
              f"Selected window={chosen}, overlap={int(best.overlap)}")
        print(f"  Arm B: n={int(best.n)} mean={best['mean']:.1f} median={best['median']:.1f}")
        print(f"  error vs Arm A: mean {best.mean_err:+.1%}, median {best.median_err:+.1%}")

    # Full distribution report for the selected window (or nothing if none).
    with (OUT / "length_distributions.txt").open("w", encoding="utf-8") as f:
        f.write("CHUNK LENGTH DISTRIBUTIONS (characters)\n")
        f.write("=" * 70 + "\n\n")
        f.write("ARM A  (table-preserving + page boundaries, current pipeline)\n")
        for k, v in a_stats.items():
            f.write(f"  {k:>7}: {v:,.1f}\n" if isinstance(v, float) else f"  {k:>7}: {v:,}\n")
        a_tok = [token_len(c["text"]) for c in arm_a]
        ats = describe(a_tok)
        f.write("\n  same chunks measured in MiniLM tokens:\n")
        f.write(f"    mean={ats['mean']:.1f} median={ats['median']:.1f} "
                f"p90={ats['p90']:.1f} max={ats['max']}\n")

        if chosen is not None:
            overlap = max(1, round(chosen * OVERLAP_RATIO))
            b_lens = window_lengths(cached, chosen, overlap)
            bs = describe(b_lens)
            f.write(f"\n\nARM B  (flat extraction, {chosen}-token window, "
                    f"{overlap}-token overlap)\n")
            for k, v in bs.items():
                f.write(f"  {k:>7}: {v:,.1f}\n" if isinstance(v, float) else f"  {k:>7}: {v:,}\n")
            f.write("\n\nPARITY\n")
            f.write(f"  mean   : Arm A {a_stats['mean']:.1f} vs Arm B {bs['mean']:.1f}"
                    f"  ({(bs['mean']-a_stats['mean'])/a_stats['mean']:+.1%})\n")
            f.write(f"  median : Arm A {a_stats['median']:.1f} vs Arm B {bs['median']:.1f}"
                    f"  ({(bs['median']-a_stats['median'])/a_stats['median']:+.1%})\n")
            f.write(f"  count  : Arm A {a_stats['n']:,} vs Arm B {bs['n']:,}\n")
            json.dump({"window": chosen, "overlap": overlap},
                      (OUT / "arm_b_config.json").open("w"))
        else:
            f.write("\n\nARM B: no window satisfied the 10% parity constraint.\n")

    print(f"\nwrote {OUT/'parity_sweep.csv'} and {OUT/'length_distributions.txt'}")


if __name__ == "__main__":
    main()
