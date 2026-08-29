"""Deterministic episode-level train/val/test split for the RF record corpus.

Each ``output_N.txt`` file is an independent synthetic episode with its own
emitter population (verified: emitter counts differ per file -- e.g. 78 vs.
7 vs. 21 -- while spanning a similar simulated duration). Episodes must
never be split *within* a file for a deinterleaving model: splitting must
hold out whole files, never mix records from the same episode across train
and val/test.

This does NOT split by time and does NOT merge files together -- each
episode is used whole, in exactly one split.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Sequence, Union

__all__ = ["assign_split", "split_files"]


def _stable_fraction(key: str) -> float:
    """Deterministic pseudo-random float in [0, 1) derived from a string key.

    Uses SHA-256 (not Python's salted ``hash()``) so the same file always
    lands in the same split across runs, processes, and machines.
    """
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def assign_split(
    source_id: str,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    salt: str = "sim_env-split-v1",
) -> str:
    """Return "train" | "val" | "test" for a given episode's source_id."""
    frac = _stable_fraction(f"{salt}:{source_id}")
    if frac < test_fraction:
        return "test"
    if frac < test_fraction + val_fraction:
        return "val"
    return "train"


def split_files(
    paths: Sequence[Union[str, Path]],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    salt: str = "sim_env-split-v1",
) -> Dict[str, List[Path]]:
    """Bucket a list of episode (RF record) files into train/val/test.

    Splitting is keyed on each file's stem (e.g. "output_7"), so re-running
    with the same file list and salt always reproduces the same split.
    """
    buckets: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}
    for p in paths:
        p = Path(p)
        split = assign_split(p.stem, val_fraction, test_fraction, salt)
        buckets[split].append(p)
    return buckets
