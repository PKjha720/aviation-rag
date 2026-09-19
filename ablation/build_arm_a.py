"""
Build Arm A from the PDFs and verify it reproduces the index currently on disk.

This is a gate. If Arm A does not reproduce data/processed/chunks_metadata.json
chunk-for-chunk, the existing relevance judgments do not describe the corpus we
are about to measure, and the experiment stops here.

Writes: ablation/out/arm_a_chunks.json
"""

import json
import random
from pathlib import Path

import numpy as np

from arms import build_arm, as_dicts

random.seed(0)
np.random.seed(0)

OUT = Path("ablation/out")
OUT.mkdir(parents=True, exist_ok=True)
LIVE = Path("data/processed/chunks_metadata.json")


def main():
    print("Building Arm A from PDFs (find_tables pass — this is the slow one)...")
    chunks = build_arm("A")
    print(f"  Arm A built: {len(chunks)} chunks")

    d = as_dicts(chunks)
    (OUT / "arm_a_chunks.json").write_text(
        json.dumps(d, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {OUT / 'arm_a_chunks.json'}")

    if not LIVE.exists():
        print("\n!! data/processed/chunks_metadata.json missing — cannot verify.")
        return

    live = json.loads(LIVE.read_text(encoding="utf-8"))
    print(f"\nVerifying against live index ({len(live)} chunks)...")

    same_count = len(live) == len(chunks)
    print(f"  chunk count      : live={len(live)} rebuilt={len(chunks)}  "
          f"{'MATCH' if same_count else 'MISMATCH'}")

    live_by_id = {c["chunk_id"]: c for c in live}
    new_by_id = {c["chunk_id"]: c for c in d}

    only_live = set(live_by_id) - set(new_by_id)
    only_new = set(new_by_id) - set(live_by_id)
    shared = set(live_by_id) & set(new_by_id)
    text_diff = [cid for cid in shared
                 if live_by_id[cid]["text"] != new_by_id[cid]["text"]]

    print(f"  ids only in live : {len(only_live)}")
    print(f"  ids only in new  : {len(only_new)}")
    print(f"  shared ids       : {len(shared)}")
    print(f"  shared but TEXT DIFFERS: {len(text_diff)}")

    live_tab = sum(1 for c in live if c.get("is_table"))
    new_tab = sum(1 for c in d if c["is_table"])
    print(f"  is_table         : live={live_tab} rebuilt={new_tab}  "
          f"{'MATCH' if live_tab == new_tab else 'MISMATCH'}")

    exact = same_count and not only_live and not only_new and not text_diff
    print(f"\n  ARM A REPRODUCES LIVE INDEX EXACTLY: {exact}")

    if text_diff:
        print("\n  first 3 differing chunk_ids (live vs rebuilt, 200 chars):")
        for cid in text_diff[:3]:
            print(f"   --- {cid}")
            print(f"     live : {live_by_id[cid]['text'][:200]!r}")
            print(f"     new  : {new_by_id[cid]['text'][:200]!r}")

    # Gold recoverability under the live judgments.
    ev = json.loads(Path("eval_results/eval_dataset_v2.json").read_text(encoding="utf-8"))
    missing = [q["question_id"] for q in ev if q["ground_truth_chunk_id"] not in new_by_id]
    print(f"\n  eval_dataset_v2 golds resolvable in rebuilt Arm A: "
          f"{len(ev) - len(missing)}/{len(ev)}")
    if missing:
        print(f"    unresolvable: {missing}")


if __name__ == "__main__":
    main()
