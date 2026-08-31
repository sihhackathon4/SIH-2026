"""Streaming / continuous-ingestion parsing of the RF record files.

The source files look like::

    dataset_names: data, labels, metadata
    data_shape: (29748, 5)
    ...
    records:
    record_1: data=[570614.875, 9227.7236328125, 0.01109, 61.937, -158.533], label=68
    record_2: ...

Only the lines matching ``record_N: data=[...], label=...`` are parsed. Each
file is read line by line (never loaded whole), yielding raw rows. A
``FileRecordSource`` streams several files and merges their records in time-of-
arrival order so the environment can keep ingesting data continuously without
holding the dataset in memory.
"""

from __future__ import annotations

import heapq
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Union

_RECORD_RE = re.compile(
    r"record_\d+:\s*data=\[(?P<data>[^\]]*)\]\s*,\s*label=(?P<label>\S+)\s*"
)

__all__ = ["FileRecordSource", "RecordSource", "parse_record_line"]


@dataclass
class PulseRecord:
    """A single parsed radiopulse record.

    ``data`` keeps the raw feature vector in the source order
    (ToA, Frequency, Pulse Width, Amplitude, AoA). Named accessors expose the
    physical meaning documented in the task.
    """

    toa_us: float
    frequency_mhz: float
    pulse_width_us: float
    amplitude_db: float
    aoa_deg: float
    emitter_id: int
    data: tuple = field(repr=False)
    source_id: str = ""

    def as_dict(self, fields: Sequence[str]) -> Dict[str, float]:
        values = (self.toa_us, self.frequency_mhz, self.pulse_width_us,
                  self.amplitude_db, self.aoa_deg)
        return dict(zip(fields, values))


def parse_record_line(
    line: str, source_id: str = "", allow_nonfinite: bool = True, on_nonfinite: str = "allow"
) -> Optional[PulseRecord]:
    """Parse a single ``record_N: ...`` line.

    Returns ``None`` when the line is a header / non-record line. Raises
    ``ValueError`` for malformed record lines so ingestion fails loudly on
    corrupt data rather than silently skipping pulses. ``source_id`` is
    stamped onto the resulting record unchanged (e.g. the originating file's
    name), so downstream consumers -- notably ML dataset splitting -- can
    trace every pulse back to its recording/session without re-parsing files.

    ``allow_nonfinite`` is kept for backward compatibility: ``False`` is
    equivalent to ``on_nonfinite="raise"``. New code should prefer
    ``on_nonfinite``, a policy over ``inf``/``nan`` in ``frequency_mhz`` /
    ``amplitude_db`` / ``aoa_deg`` (NOT ``pulse_width_us``, which is handled
    downstream in ``RadioEnvironment``):

    * ``"allow"`` (default) -- preserve the non-finite value unchanged.
    * ``"drop"`` -- return ``None`` for the offending record so the caller can
      skip it and keep building the stream.
    * ``"raise"`` -- raise ``ValueError`` so corruption is surfaced instead of
      silently polluting downstream statistics.

    A real, sizeable slice of the source corpus can contain literal ``inf`` /
    ``nan`` values in these fields, so the choice matters most when building an
    ML dataset, where a single bad feature would corrupt every statistic and
    gradient computed from it.
    """
    if allow_nonfinite is False and on_nonfinite == "allow":
        on_nonfinite = "raise"
    m = _RECORD_RE.search(line)
    if m is None:
        return None
    try:
        toks = m.group("data").split(",")
        if len(toks) != 5:
            raise ValueError(f"expected 5 feature values, got {len(toks)}")
        toa, freq, pw, amp, aoa = (float(t) for t in toks)
        label = int(m.group("label"))
        if on_nonfinite != "allow":
            import math

            for name, v in (("frequency_mhz", freq), ("amplitude_db", amp), ("aoa_deg", aoa)):
                if not math.isfinite(v):
                    if on_nonfinite == "raise":
                        raise ValueError(f"non-finite {name}={v!r}")
                    return None  # "drop"
    except ValueError as exc:
        raise ValueError(f"malformed record line: {line!r} ({exc})") from exc
    return PulseRecord(
        toa_us=toa,
        frequency_mhz=freq,
        pulse_width_us=pw,
        amplitude_db=amp,
        aoa_deg=aoa,
        emitter_id=label,
        data=(toa, freq, pw, amp, aoa),
        source_id=source_id,
    )


def _iter_records(
    path: Path, allow_nonfinite: bool = True, on_nonfinite: str = "allow"
) -> Iterator[PulseRecord]:
    """Yield every record from one file, in file order (assumed sorted by ToA)."""
    source_id = path.stem  # e.g. "output_0" -- stable, human-readable episode tag
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            rec = parse_record_line(
                line, source_id=source_id,
                allow_nonfinite=allow_nonfinite, on_nonfinite=on_nonfinite,
            )
            if rec is not None:
                yield rec


class RecordSource:
    """Iterable of :class:`PulseRecord` sorted by time-of-arrival."""

    def __init__(self, records: Iterable[PulseRecord]):
        self._records = iter(records)

    def __iter__(self) -> Iterator[PulseRecord]:
        return self._records


class _Resolver:
    """Base class for path/glob expansion hooks (dependency injection point)."""

    def resolve(self, path: Path) -> List[Path]:
        raise NotImplementedError


class FileRecordSource:
    """Streaming source over one or more record files merged in ToA order.

    Physical files are opened lazily and read incrementally, so memory use is
    bounded by the merge window rather than the dataset size -- this is what
    makes *continuous data ingestion* possible. Records are produced in
    nondecreasing ``toa_us`` order.
    """

    def __init__(
        self,
        paths: Sequence[Union[str, Path]],
        resolver: Optional[_Resolver] = None,
        allow_nonfinite: bool = True,
        on_nonfinite: str = "allow",
    ):
        self._paths = list(paths)
        self._resolver = resolver or _DefaultResolver()
        self._allow_nonfinite = allow_nonfinite
        self._on_nonfinite = on_nonfinite
        self._files: Iterable[Iterable[PulseRecord]] = []

    def _prepare(self) -> None:
        """Resolve every path/glob to concrete existing files, sorted by name."""
        resolved: List[Path] = []
        for p in self._paths:
            resolved.extend(self._resolver.resolve(Path(p)))
        resolved = sorted(
            {r for r in resolved if r.is_file()},
            key=lambda r: r.name,
        )
        self._files = [
            _iter_records(p, allow_nonfinite=self._allow_nonfinite,
                          on_nonfinite=self._on_nonfinite)
            for p in resolved
        ]

    def __iter__(self) -> Iterator[PulseRecord]:
        self._prepare()
        if not self._files:
            return

        # K-way merge over file iterators keyed on time-of-arrival. A monotonic
        # counter breaks ties so equal ToA records from different files keep a
        # stable, deterministic order and never compare the file iterators.
        heap: List[tuple] = []
        counter = 0
        for it in self._files:
            try:
                first = next(it)
            except StopIteration:
                continue
            heapq.heappush(heap, (first.toa_us, counter, first, it))
            counter += 1

        while heap:
            _, _, rec, it = heapq.heappop(heap)
            yield rec
            try:
                nxt = next(it)
            except StopIteration:
                continue
            heapq.heappush(heap, (nxt.toa_us, counter, nxt, it))
            counter += 1


class _DefaultResolver(_Resolver):
    """Expand glob patterns to concrete file paths (sorted, stable)."""

    def resolve(self, path: Path) -> List[Path]:
        import glob as _glob

        text = str(path)
        if _glob.has_magic(text):
            return [Path(p) for p in sorted(_glob.glob(text))]
        return [path]
