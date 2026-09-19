"""
Row-pairing check for the table stratum.

THE RISK THIS ADDRESSES.
    The v2 prompt asks for a span carrying a row label AND its value. The span
    is validated against get_text("text"), which reads tables column-wise. So a
    span can be verbatim-contiguous in the flat text, sit in the length band,
    fit both arms and locate exactly one page — and still pair a label from one
    row with a value from a DIFFERENT row. That gold would be wrong, not merely
    weak. No check based on flat-text contiguity can see it, because flat-text
    contiguity is the thing being validated.

WHAT THIS DOES.
    Reconstructs each page's tables row-wise with find_tables() and asks whether
    the span's content falls inside ONE reconstructed row.

WHAT THIS IS NOT.
    find_tables() is Arm A's own view of the page. Agreement with it is evidence,
    not ground truth, and a different_row verdict is not proof the gold is wrong
    (the reconstruction can itself be off). Nothing is auto-dropped here.

Writes: ablation/out/row_pairing_check.csv
"""

import json
import re
from pathlib import Path

import pandas as pd

try:
    import fitz
except ImportError:
    import pymupdf as fitz

from arms import list_pdfs
from ingest import _table_to_markdown, _is_real_table
from judgments import normalize, coverage

OUT = Path("ablation/out")

# Fraction of the span's tokens that must sit in a single reconstructed row for
# the pairing to count as consistent.
SAME_ROW_MIN = 0.80


def span_tokens(span: str) -> list[str]:
    toks = [normalize(w) for w in re.findall(r"\w+", span)]
    return [t for t in toks if len(t) >= 2]


def row_tokens(row) -> set[str]:
    out = set()
    for cell in row:
        if cell is None:
            continue
        for w in re.findall(r"\w+", str(cell)):
            n = normalize(w)
            if len(n) >= 2:
                out.add(n)
    return out


def check_page(pdf_path: Path, page_number: int, span: str) -> dict:
    """Verdict for one span against one page's reconstructed tables."""
    st = set(span_tokens(span))
    res = {"verdict": "undetermined", "n_tables": 0, "best_row_cov": 0.0,
           "best_row_idx": -1, "pair_row_cov": 0.0, "best_row_text": "",
           "pair_row_text": "", "table_is_real": ""}
    if not st:
        res["note"] = "no usable tokens in span"
        return res

    try:
        doc = fitz.open(str(pdf_path))
        page = doc[page_number - 1]
        found = page.find_tables()
        tables = list(found.tables)
    except Exception as e:
        res["note"] = f"find_tables failed: {str(e)[:60]}"
        return res

    res["n_tables"] = len(tables)
    if not tables:
        doc.close()
        res["note"] = "no tables found on page"
        return res

    best = (0.0, -1, None, None)
    for t in tables:
        try:
            rows = t.extract()
            md = _table_to_markdown(rows)
        except Exception:
            continue
        covs = []
        for i, row in enumerate(rows):
            rt = row_tokens(row)
            covs.append((len(st & rt) / len(st), i, row, _is_real_table(md) if md else False))
        if not covs:
            continue
        top = max(covs, key=lambda x: x[0])
        if top[0] > best[0]:
            best = top
            # best pair of two distinct rows, for the different_row case
            covs_sorted = sorted(covs, key=lambda x: x[0], reverse=True)
            if len(covs_sorted) >= 2:
                a, b = covs_sorted[0], covs_sorted[1]
                union = len(st & (row_tokens(a[2]) | row_tokens(b[2]))) / len(st)
                res["pair_row_cov"] = round(union, 4)
                res["pair_row_text"] = " | ".join(
                    "" if c is None else str(c).replace("\n", " ") for c in b[2])[:300]
    doc.close()

    cov, idx, row, is_real = best
    res["best_row_cov"] = round(cov, 4)
    res["best_row_idx"] = idx
    res["table_is_real"] = str(bool(is_real))
    if row is not None:
        res["best_row_text"] = " | ".join(
            "" if c is None else str(c).replace("\n", " ") for c in row)[:300]

    if cov >= SAME_ROW_MIN:
        res["verdict"] = "same_row"
    elif res["pair_row_cov"] >= SAME_ROW_MIN:
        res["verdict"] = "different_row"
    else:
        res["verdict"] = "undetermined"
        res["note"] = "span content not located in any reconstructed row"
    return res


def main():
    src = OUT / "fresh_pool_v2.json"
    if not src.exists():
        print("FAILED: fresh_pool_v2.json not found. Nothing to check.")
        return
    pool = json.loads(src.read_text(encoding="utf-8"))
    table_items = [q for q in pool if q["page_stratum"] == "table"]
    print(f"checking {len(table_items)} table-stratum items")

    # Does the gold span sit inside an Arm A chunk that ingest flagged as a
    # table? This separates "prose on a table-bearing page" (undetermined is
    # expected and harmless) from "tabular content we could not place"
    # (undetermined is worth your time).
    arm_a = json.loads((OUT / "arm_a_chunks.json").read_text(encoding="utf-8"))
    a_pages = {}
    for c in arm_a:
        a_pages.setdefault((c["source_file"], c["page_number"]), []).append(c)

    def gold_in_table_chunk(q):
        sn = normalize(q["gold_span"])
        for c in a_pages.get((q["source_file"], q["page_number"]), []):
            if coverage(sn, normalize(c["text"])) >= 1.0:
                return bool(c["is_table"])
        return False

    paths = {p.name: p for p in list_pdfs()}
    rows = []
    for i, q in enumerate(table_items, 1):
        p = paths.get(q["source_file"])
        if p is None:
            rows.append({**{k: q[k] for k in ("question_id", "source_file", "page_number")},
                         "verdict": "undetermined", "note": "pdf not found"})
            continue
        r = check_page(p, q["page_number"], q["gold_span"])
        rows.append({"question_id": q["question_id"], "source_file": q["source_file"],
                     "page_number": q["page_number"],
                     "gold_in_table_chunk": gold_in_table_chunk(q),
                     "gold_span": q["gold_span"][:160].replace("\n", " / "), **r})
        if i % 10 == 0:
            print(f"  {i}/{len(table_items)}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "row_pairing_check.csv", index=False)

    counts = df.verdict.value_counts().to_dict()
    print("\n" + "=" * 66)
    print("ROW-PAIRING CHECK — table stratum")
    print("=" * 66)
    for v in ("same_row", "different_row", "undetermined"):
        n = counts.get(v, 0)
        print(f"  {v:<16} {n:>4} / {len(df)}   ({n/max(len(df),1):.3f})")
    und = df[df.verdict == "undetermined"]
    if len(und):
        in_tab = int(und.gold_in_table_chunk.sum())
        print("")
        print(f"  of the {len(und)} undetermined:")
        print(f"    gold sits in a TABLE chunk  : {in_tab}  <- worth reviewing")
        print(f"    gold is PROSE on a table page: {len(und)-in_tab}  <- expected, benign")
    print("=" * 66)
    print("find_tables() is Arm A's own view of the page. These verdicts are")
    print("EVIDENCE, NOT GROUND TRUTH. Nothing has been dropped on this basis.")
    print(f"\nwrote {OUT/'row_pairing_check.csv'}")


if __name__ == "__main__":
    main()
