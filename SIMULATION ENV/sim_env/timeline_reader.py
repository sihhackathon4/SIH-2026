"""Streaming reader for the NDJSON event log (for an ML scheduler).

Consumes the timeline produced by :class:`TimelineWriter` line by line, in
time order, without loading it whole into memory. An ML scheduler can build
training windows from the fine-grained ``entry``/``exit`` events or from the
dense ``snapshot`` frames.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional, TextIO, Union


def iter_events(path: Union[str, Path, TextIO]) -> Iterator[dict]:
    """Yield each NDJSON event as a dict, in stream order.

    Handles a path or an open text stream (e.g. a pipe). Skips empty lines.
    """
    if hasattr(path, "read"):
        stream: TextIO = path  # type: ignore[assignment]
        for line in stream:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
    else:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def read_meta_only(path: Union[str, Path]) -> Optional[dict]:
    """Read just the ``meta`` record (the first event) for schema inspection."""
    try:
        obj = next(iter_events(path))
    except StopIteration:
        return None
    return obj if obj.get("event") == "meta" else None


def rebuild_frames(path: Union[str, Path]) -> Iterator[dict]:
    """Reconstruct the per-instant active scene by folding entry/exit events.

    Yields ``{time_us, active: {pulse_id: pulse}}`` after every event so a
    scheduler that prefers a state-based representation can replay the stream
    without using the periodic ``snapshot`` records.
    """
    active: dict[int, dict] = {}
    for ev in iter_events(path):
        etype = ev.get("event")
        if etype == "entry":
            p = ev["pulse"]
            active[p["pulse_id"]] = p
        elif etype == "exit":
            active.pop(ev.get("pulse_id"), None)
        yield {"time_us": ev.get("time_us"), "active": dict(active)}
