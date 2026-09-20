"""
Build Arm B and the re-grounded judgment anchors.

The anchors re-key each of the original 80 judgments on the gold chunk's TEXT
rather than its chunk_id, because chunk_id hashes position and is meaningless
across chunkings.

The reachability ceiling formerly computed here is now in diagnostics.py
(ceiling_contiguous.csv), which uses the contiguous coverage criterion.

Writes: ablation/out/arm_b_chunks.json
        ablation/out/anchors.json
"""

import json
import random
from pathlib import Path

import numpy as np

from arms import build_arm, as_dicts
from judgments import normalize

random.seed(0)
np.random.seed(0)

OUT = Path("ablation/out")
OUT.mkdir(parents=True, exist_ok=True)


def load_arm_b():
    cfg = json.loads((OUT / "arm_b_config.json").read_text())
    print(f"Building Arm B (window={cfg['window']}, overlap={cfg['overlap']})...")
    chunks = as_dicts(build_arm("B", cfg["window"], cfg["overlap"]))
    print(f"  Arm B built: {len(chunks)} chunks")
    (OUT / "arm_b_chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    return chunks, cfg


def main():
    arm_a = json.loads((OUT / "arm_a_chunks.json").read_text(encoding="utf-8"))
    arm_b, cfg = load_arm_b()

    ev = json.loads(Path("eval_results/eval_dataset_v2.json").read_text(encoding="utf-8"))
    a_by_id = {c["chunk_id"]: c for c in arm_a}

    # ── anchors: the chunking-invariant judgment ──────────────────────────
    anchors = []
    for q in ev:
        g = a_by_id.get(q["ground_truth_chunk_id"])
        if g is None:
            print(f"  !! {q['question_id']}: gold chunk id not in Arm A — skipped")
            continue
        anchors.append({
            "question_id": q["question_id"],
            "question": q["question"],
            "category": q["category"],
            "gold_is_table": q["gold_is_table"],
            "gold_chunk_id_armA": q["ground_truth_chunk_id"],
            "gold_text": g["text"],
            "gold_norm_len": len(normalize(g["text"])),
            "source_file": g["source_file"],
            "page_number": g["page_number"],
        })
    (OUT / "anchors.json").write_text(
        json.dumps(anchors, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT/'anchors.json'} — {len(anchors)} anchors "
          f"({sum(a['gold_is_table'] for a in anchors)} table / "
          f"{sum(not a['gold_is_table'] for a in anchors)} prose)")


    # The reachability ceiling that used to be computed here (ceiling.csv,
    # ceiling_report.txt) was written with the pre-contiguity-fix coverage()
    # and is superseded by diagnostics.py -> ceiling_contiguous.csv. It has
    # been removed rather than left to overwrite stale files silently.


if __name__ == "__main__":
    main()
