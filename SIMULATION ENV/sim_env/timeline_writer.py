"""NDJSON event-log writer for a Machine-Learning scheduler.

The environment emits a continuous stream of :class:`SimulationEvent` objects.
This writer serialises them as one JSON object per line (NDJSON) so an ML
downstream system can consume the timeline incrementally -- streaming line by
line, in order, on disk or over a pipe -- without buffering the whole log.

Every emitted object carries:

* ``event``    -- ``"entry"`` | ``"exit"`` | ``"snapshot"`` | ``"meta"``
* ``time_us``  -- simulated time (microseconds) of the event
* ``active_count`` -- number of pulses alive in the environment right then

``entry`` and ``exit`` events additionally embed the full 5-feature vector
(``toa_us``, ``frequency_mhz``, ``pulse_width_us``, ``amplitude_db``,
``aoa_deg``), the ``pulse_id``, the ground-truth ``emitter_id`` label, and the
pulse's ``exit_us``. A ``snapshot`` event embeds ``active_pulses`` -- every
pulse alive at that instant -- which is the form most useful for sequence /
state-based ML schedulers.

The first line is always a ``meta`` record describing the schema version, the
feature order/units, and the run configuration so any parser can self-describe.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional, TextIO, Union

from .config import FEATURE_FIELDS, FEATURE_UNITS, SimConfig
from .environment import SimulationEvent

SCHEMA_VERSION = 1


class TimelineWriter:
    """Streams :class:`SimulationEvent` objects to an NDJSON destination."""

    def __init__(
        self,
        destination: Optional[Union[str, Path, TextIO]] = None,
        config: Optional[SimConfig] = None,
        fields: tuple = FEATURE_FIELDS,
    ):
        self.config = config or SimConfig()
        self.fields = tuple(fields)
        self._owns_handle = False
        self._closed = False

        if destination is None:
            self._stream = sys.stdout
        elif hasattr(destination, "write"):
            self._stream = destination
        else:
            self._path = Path(destination)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = open(self._path, "w", encoding="utf-8")
            self._owns_handle = True

    # ------------------------------------------------------------------ meta

    def write_meta(self, record_count: Optional[int] = None,
                   time_min_us: Optional[float] = None,
                   time_max_us: Optional[float] = None) -> None:
        meta: dict[str, Any] = {
            "event": "meta",
            "schema_version": SCHEMA_VERSION,
            "feature_order": list(self.fields),
            "feature_units": {f: FEATURE_UNITS[f] for f in self.fields},
            "time_unit": "microseconds",
            "snapshot_interval_us": self.config.snapshot_interval_us,
            "min_pw_us": self.config.min_pw_us,
        }
        if record_count is not None:
            meta["record_count"] = record_count
        if time_min_us is not None:
            meta["time_min_us"] = time_min_us
        if time_max_us is not None:
            meta["time_max_us"] = time_max_us
        self._write_line(meta)

    # ----------------------------------------------------------------- events

    def on_event(self, event: SimulationEvent) -> None:
        self.write_event(event)

    def write_event(self, event: SimulationEvent) -> None:
        if event.event_type == "entry" and not self.config.emit_entries:
            return
        if event.event_type == "exit" and not self.config.emit_exits:
            return
        if event.event_type == "snapshot" and not self.config.emit_snapshots:
            return
        self._write_line(event.to_dict(self.fields))

    def _write_line(self, obj: dict) -> None:
        self._stream.write(json.dumps(obj, separators=(",", ":")) + "\n")

    # ---------------------------------------------------------------- lifecycle

    def close(self) -> None:
        if self._closed:
            return
        self._stream.flush()
        if self._owns_handle:
            self._stream.close()
        self._closed = True

    def __enter__(self) -> "TimelineWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
