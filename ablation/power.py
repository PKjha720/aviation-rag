"""
Minimum detectable effect for the paired comparison this experiment will run.

Both arms answer the same queries, and each query scores 0/1 on Recall@k. The
natural test is McNemar's on the discordant pairs (queries where exactly one
arm succeeds). Its power depends on n and on the DISCORDANT RATE, which is not
known until retrieval has run. So this file does not state one number; it
states the minimum detectable difference at 80% power for a grid of plausible
discordant rates, and the reader picks the row once the real rate is known.

Method: exact. Under H0 the successes among d discordant pairs favour Arm A
with probability 0.5. The two-sided exact binomial rejection region at
alpha = 0.05 is enumerated. Under H1 the probability is p1 > 0.5; power is the
binomial mass of the rejection region under p1. The minimum detectable effect
is the smallest (b - c)/n = delta * (2*p1 - 1) whose power reaches 0.80, found
by scanning p1 upward. Stdlib only (math.comb); no scipy.

Everything printed here is a property of the test at a given n and discordant
rate. It is not a measurement of this data.

Writes: ablation/out/power_mde.csv
        ablation/out/POWER.txt
"""

from math import comb
from pathlib import Path

import pandas as pd

OUT = Path("ablation/out")

ALPHA = 0.05
POWER = 0.80
NS = [42, 80, 90]
DISCORDANT_RATES = [0.10, 0.20, 0.30, 0.40]


def pmf(d, p):
    return [comb(d, b) * p ** b * (1 - p) ** (d - b) for b in range(d + 1)]


def rejection_region(d, alpha=ALPHA):
    """b values with two-sided exact p <= alpha under Bin(d, 0.5)."""
    p0 = pmf(d, 0.5)
    cdf = [sum(p0[:b + 1]) for b in range(d + 1)]
    sf = [sum(p0[b:]) for b in range(d + 1)]
    return [b for b in range(d + 1) if min(1.0, 2 * min(cdf[b], sf[b])) <= alpha]


def power_at(d, p1, region):
    p = pmf(d, p1)
    return sum(p[b] for b in region)


def mde(n, delta, target=POWER):
    d = round(delta * n)
    if d < 1:
        return None
    region = rejection_region(d)
    if not region:
        return {"n": n, "discordant_rate": delta, "d": d, "note": "no rejection region at this d"}
    # scan p1 upward
    for i in range(500, 1000):
        p1 = i / 1000
        if power_at(d, p1, region) >= target:
            return {"n": n, "discordant_rate": delta, "d": d, "p1": p1,
                    "mde_recall_points": round(delta * (2 * p1 - 1), 3),
                    "power": round(power_at(d, p1, region), 3),
                    "min_b_to_reject": min(b for b in region if b > d / 2)}
    return {"n": n, "discordant_rate": delta, "d": d, "note": "not reachable at 80% power"}


def main():
    rows = [r for r in (mde(n, dl) for n in NS for dl in DISCORDANT_RATES) if r]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "power_mde.csv", index=False)

    L = []
    w = L.append
    w("MINIMUM DETECTABLE EFFECT - paired binary Recall, McNemar exact test")
    w("=" * 74)
    w(f"alpha = {ALPHA} two-sided, power = {POWER}. Not a measurement of this data:")
    w("the discordant rate is unknown until retrieval runs. Read the row that")
    w("matches the observed rate.")
    w("")
    w("  n    = number of paired queries")
    w("  disc = fraction of queries where exactly one arm succeeds")
    w("  d    = number of discordant pairs the test actually uses")
    w("  MDE  = smallest difference in Recall (Arm A minus Arm B, in points of")
    w("         proportion) detectable at 80% power")
    w("  b*   = smallest number of Arm-A-only wins out of d that rejects H0")
    w("")
    w(f"  {'n':>4}{'disc':>7}{'d':>5}{'MDE':>8}{'power':>8}{'b*':>5}")
    for _, r in df.iterrows():
        if "note" in r and isinstance(r.get("note"), str):
            w(f"  {int(r.n):>4}{r.discordant_rate:>7.2f}{int(r.d):>5}   {r.note}")
        else:
            w(f"  {int(r.n):>4}{r.discordant_rate:>7.2f}{int(r.d):>5}"
              f"{r.mde_recall_points:>8.3f}{r.power:>8.3f}{int(r.min_b_to_reject):>5}")
    w("")
    w("How to read it: at n=80 with a 20% discordant rate there are 16 usable")
    w("pairs, and the arms must disagree lopsidedly enough that the difference in")
    w("Recall is at least the MDE shown before an 80%-power test would call it.")
    w("Differences below the MDE are not detectable, whatever their sign.")
    w("")
    w("Discordant pairs and the test statistic must be REPORTED alongside any")
    w("delta, because a delta with few discordant pairs is a delta the test")
    w("cannot see.")
    txt = "\n".join(L)
    (OUT / "POWER.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"\nwrote {OUT/'power_mde.csv'}, {OUT/'POWER.txt'}")


if __name__ == "__main__":
    main()
