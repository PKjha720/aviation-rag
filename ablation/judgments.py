"""
Chunking-invariant relevance judgments.

The existing judgments key on chunk_id, and chunk_id is
md5(file::page::within_page_index) — an ADDRESS, not a content hash. Re-chunk
and the same id points at different text, so the stored judgments cannot be
carried across arms.

This module re-grounds each judgment on the gold chunk's TEXT, which survives
re-chunking, and scores a retrieved chunk by how much of that gold text it
actually covers.

Normalization strips case, whitespace and all punctuation, so Arm A's markdown
table rendering ("| 4.2 | 18 kg |") and Arm B's raw extracted line ("4.2 18 kg")
normalize to the same characters. Without this, Arm B could never match a table
gold, because the pipes and dashes exist only in Arm A's rendering — the
criterion itself would decide the experiment.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

THRESHOLDS = [0.40, 0.50, 0.60, 0.70, 0.80]

_NON_ALNUM = re.compile(r"[^a-z0-9]")


def normalize(text: str) -> str:
    """Lowercase, drop every non-alphanumeric character."""
    return _NON_ALNUM.sub("", text.lower())


def matched_blocks(gold_norm: str, chunk_norm: str):
    """Matching blocks between gold and chunk, as (gold_start, size) pairs.

    autojunk is disabled: on strings over 200 characters difflib's default
    heuristic treats frequent characters as junk, which silently suppresses
    matches on exactly the long regulatory text this is applied to.
    """
    sm = SequenceMatcher(None, gold_norm, chunk_norm, autojunk=False)
    return [(b.a, b.size) for b in sm.get_matching_blocks() if b.size > 0]


def coverage(gold_norm: str, chunk_norm: str) -> float:
    """Fraction of gold characters that appear, in order, in this chunk."""
    if not gold_norm:
        return 0.0
    return sum(size for _, size in matched_blocks(gold_norm, chunk_norm)) / len(gold_norm)


def covered_mask(gold_norm: str, chunk_norm: str) -> set[int]:
    """Which gold character positions this chunk covers. Used for union-over-k."""
    out: set[int] = set()
    for start, size in matched_blocks(gold_norm, chunk_norm):
        out.update(range(start, start + size))
    return out


def union_coverage(gold_norm: str, chunk_norms: list[str]) -> float:
    """Fraction of gold characters covered by ANY chunk in the list.

    This is the secondary criterion: it asks whether the evidence is present
    across the returned set, not whether any single chunk holds it.
    """
    if not gold_norm:
        return 0.0
    seen: set[int] = set()
    for cn in chunk_norms:
        seen |= covered_mask(gold_norm, cn)
    return len(seen) / len(gold_norm)


def best_coverage(gold_norm: str, chunk_norms: list[str]) -> tuple[float, int]:
    """(max per-chunk coverage, index of the chunk achieving it)."""
    best, best_i = 0.0, -1
    for i, cn in enumerate(chunk_norms):
        c = coverage(gold_norm, cn)
        if c > best:
            best, best_i = c, i
    return best, best_i
