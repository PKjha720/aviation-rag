"""
Verify that the rebuilt Arm A reproduces the index currently on disk, and
write the result to a file.

This exists so ARM_A_VERIFICATION.txt has a script as its provenance. It was
first produced by an ad-hoc snippet, which the audit rightly flagged. It is
called at the end of build_arm_a.py and can also be run on its own against an
existing arm_a_chunks.json (fast; no PDF parsing).

Reads:  ablation/out/arm_a_chunks.json
        data/processed/chunks_metadata.json
        eval_results/eval_dataset_v2.json
Writes: ablation/out/ARM_A_VERIFICATION.txt
"""

import json
from pathlib import Path

OUT = Path("ablation/out")
LIVE = Path("data/processed/chunks_metadata.json")
EVAL = Path("eval_results/eval_dataset_v2.json")


def verify(write: bool = True) -> dict:
    new = json.loads((OUT / "arm_a_chunks.json").read_text(encoding="utf-8"))
    if not LIVE.exists():
        msg = "data/processed/chunks_metadata.json missing - cannot verify."
        if write:
            (OUT / "ARM_A_VERIFICATION.txt").write_text(msg + "\n", encoding="utf-8")
        print(msg)
        return {"verified": False}

    live = json.loads(LIVE.read_text(encoding="utf-8"))
    nb = {c["chunk_id"]: c for c in new}
    lb = {c["chunk_id"]: c for c in live}
    shared = set(nb) & set(lb)
    text_diff = [c for c in shared if nb[c]["text"] != lb[c]["text"]]
    only_live, only_new = set(lb) - set(nb), set(nb) - set(lb)
    live_tab = sum(1 for c in live if c.get("is_table"))
    new_tab = sum(1 for c in new if c["is_table"])
    exact = (len(live) == len(new) and not only_live and not only_new and not text_diff)

    ev = json.loads(EVAL.read_text(encoding="utf-8"))
    missing = [q["question_id"] for q in ev if q["ground_truth_chunk_id"] not in nb]

    L = [
        "ARM A REBUILD VERIFICATION",
        "=" * 60,
        "Produced by ablation/verify_arm_a.py.",
        "Arm A rebuilt from the PDFs using functions imported from ingest.py,",
        "compared against the index already on disk",
        "(data/processed/chunks_metadata.json).",
        "",
        f"  chunk count      : live={len(live)}  rebuilt={len(new)}  "
        f"{'MATCH' if len(live) == len(new) else 'MISMATCH'}",
        f"  ids only in live : {len(only_live)}",
        f"  ids only in new  : {len(only_new)}",
        f"  shared ids       : {len(shared)}",
        f"  shared, text differs: {len(text_diff)}",
        f"  is_table chunks  : live={live_tab}  rebuilt={new_tab}  "
        f"{'MATCH' if live_tab == new_tab else 'MISMATCH'}",
        "",
        f"  REPRODUCES EXACTLY: {exact}",
        "",
        f"  eval_dataset_v2 golds resolvable in rebuilt Arm A: {len(ev) - len(missing)}/{len(ev)}",
    ]
    if missing:
        L.append(f"    unresolvable: {missing}")
    if text_diff:
        L.append("")
        L.append("  first 3 differing chunk_ids (live vs rebuilt, 200 chars):")
        for cid in text_diff[:3]:
            L.append(f"   --- {cid}")
            L.append(f"     live : {lb[cid]['text'][:200]!r}")
            L.append(f"     new  : {nb[cid]['text'][:200]!r}")

    txt = "\n".join(L)
    if write:
        (OUT / "ARM_A_VERIFICATION.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return {"verified": exact, "n": len(new), "text_diff": len(text_diff),
            "golds_resolvable": len(ev) - len(missing)}


if __name__ == "__main__":
    verify()
