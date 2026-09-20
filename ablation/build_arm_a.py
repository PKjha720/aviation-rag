"""
Build Arm A from the PDFs and verify it reproduces the index currently on disk.

This is a gate. If Arm A does not reproduce data/processed/chunks_metadata.json
chunk-for-chunk, the existing relevance judgments do not describe the corpus we
are about to measure, and the experiment stops here.

Writes: ablation/out/arm_a_chunks.json
        ablation/out/ARM_A_VERIFICATION.txt   (via verify_arm_a.py)
"""

import json
from pathlib import Path

from arms import build_arm, as_dicts

OUT = Path("ablation/out")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("Building Arm A from PDFs (find_tables pass — this is the slow one)...")
    chunks = build_arm("A")
    print(f"  Arm A built: {len(chunks)} chunks")

    d = as_dicts(chunks)
    (OUT / "arm_a_chunks.json").write_text(
        json.dumps(d, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {OUT / 'arm_a_chunks.json'}")

    # Verification lives in verify_arm_a.py so the result has a script as its
    # provenance and can be re-run without re-parsing the PDFs.
    from verify_arm_a import verify
    verify(write=True)


if __name__ == "__main__":
    main()
