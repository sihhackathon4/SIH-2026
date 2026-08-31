"""Event-driven radiowave simulation environment.

The environment models a real-world radiowave stream. Every radiopulse

* **enters** the environment at its Time-of-Arrival (``toa_us``),
* stays **active** for exactly its Pulse Width (``pulse_width_us``),
* then **disappears completely** at ``toa_us + pulse_width_us``.

The clock advances with a *sweep line* from one event to the next (pulse entry,
pulse exit, or an optional grid tick for periodic snapshots), so sub-microsecond
pulse widths are handled exactly and runtime scales with the number of events --
not with wall-clock time nor with the time span. Records are streamed
incrementally from a :class:`RecordSource`, so the environment can ingest data
continuously without loading the whole dataset into memory.

Basic usage (API)::

    cfg = SimConfig(inputs=["OUTPUT FILES/*.txt"])
    src = FileRecordSource(cfg.inputs)
    env = RadioEnvironment(src, cfg)
    while (ev := env.step()) is not None:
        print(ev.event_type, ev.time_us, ev.active_count)
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from .config import SimConfig
from .ingest import PulseRecord, RecordSource

__all__ = ["ActivePulse", "SimulationEvent", "RadioEnvironment"]


@dataclass
class ActivePulse:
    """A radiopulse currently present in the environment.

    A pulse is *active* over the half-open interval ``[toa_us, exit_us)`` and
    disappears completely at ``exit_us``.
    """

    pulse_id: int
    toa_us: float
    frequency_mhz: float
    pulse_width_us: float
    amplitude_db: float
    aoa_deg: float
    emitter_id: int
    exit_us: float = field(repr=False)
    source_id: str = ""

    @property
    def active_us(self) -> float:
        """How long the pulse is active for (>= 0)."""
        return max(self.exit_us - self.toa_us, 0.0)

    def feature_vector(self, fields: Sequence[str]) -> Dict[str, float]:
        values = (self.toa_us, self.frequency_mhz, self.pulse_width_us,
                  self.amplitude_db, self.aoa_deg)
        return dict(zip(fields, values))

    def summary_dict(self, fields: Sequence[str]) -> Dict[str, Any]:
        d = self.feature_vector(fields)
        d["pulse_id"] = self.pulse_id
        d["emitter_id"] = self.emitter_id
        d["exit_us"] = self.exit_us
        d["source_id"] = self.source_id
        return d


@dataclass
class SimulationEvent:
    """A single event emitted by the environment.

    ``event_type`` is one of:

    * ``"entry"``    -- a pulse entered the environment at ``time_us``.
    * ``"exit"``     -- a pulse left the environment at ``time_us``.
    * ``"snapshot"`` -- periodic ground-truth frame describing *all* currently
      active pulses at ``time_us``.

    For ``entry``/``exit``, ``pulse`` holds the entering/leaving pulse. For
    ``snapshot``, ``active_pulses`` lists every pulse active at ``time_us``.
    ``active_count`` is always the size of the active set after processing the
    event.
    """

    event_type: str
    time_us: float
    pulse_id: Optional[int] = None
    pulse: Optional[ActivePulse] = None
    active_pulses: List[ActivePulse] = field(default_factory=list)
    active_count: int = 0

    def to_dict(self, fields: Sequence[str]) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "event": self.event_type,
            "time_us": self.time_us,
            "active_count": self.active_count,
        }
        if self.pulse is not None:
            d["pulse"] = self.pulse.summary_dict(fields)
            d["pulse_id"] = self.pulse.pulse_id
        if self.event_type == "snapshot":
            d["active_pulses"] = [p.summary_dict(fields) for p in self.active_pulses]
        return d


class RadioEnvironment:
    """Sweep-line simulator of the radiowave stream.

    The clock advances from the first recorded ToA to the last pulse exit, one
    event at a time. Between events the active set is constant. Deriving
    subclasses can override :meth:`_on_event` to observe every event (or pass
    ``on_event`` to the constructor).
    """

    def __init__(
        self,
        source: RecordSource,
        config: Optional[SimConfig] = None,
        on_event: Optional[
            Union[Callable[[SimulationEvent], None],
                  Sequence[Callable[[SimulationEvent], None]]]
        ] = None,
    ):
        self.config = config or SimConfig()
        # Accept either a single callback or a list -- lets a caller attach
        # both a TimelineWriter (for the on-disk log) and a live model/dataset
        # consumer (e.g. an ML scheduler reacting to pulses in real time) to
        # the same run without one having to wrap the other.
        if on_event is None:
            self._callbacks: List[Callable[[SimulationEvent], None]] = []
        elif callable(on_event):
            self._callbacks = [on_event]
        else:
            self._callbacks = list(on_event)

        # Streaming arrival state.
        self._arrivals = iter(source)
        self._pending: Optional[PulseRecord] = None
        self._time_min: Optional[float] = None
        self._time_max = 0.0

        # Active-pulse bookkeeping.
        self._pulse_seq = 0
        self._exit_seq = 0
        self.active: Dict[int, ActivePulse] = {}
        self._exit_heap: List[tuple] = []  # (exit_us, seq, pulse_id)

        # Clock state.
        self.time_us: Optional[float] = None
        self.done = False
        self.total_entries = 0
        self.total_exits = 0
        self.total_snapshots = 0
        # Snapshot grid bookkeeping.
        self._snap_interval = (
            float(self.config.snapshot_interval_us)
            if self.config.snapshot_interval_us is not None
            else None
        )
        self._next_tick: Optional[float] = None

        self._prime()

        # Align the periodic snapshot grid to the first arrival so a frame is
        # emitted at the very start of the stream and every `interval` after.
        if self._snap_interval is not None and self._pending is not None:
            self._next_tick = self._pending.toa_us

    # ------------------------------------------------------------------ setup

    @property
    def time_min_us(self) -> Optional[float]:
        """Simulated time of the first recorded pulse (ToA), or None if empty."""
        return self._time_min

    @property
    def time_max_us(self) -> float:
        """Latest ToA seen among all ingested records."""
        return self._time_max

    def _prime(self) -> None:
        """Pull the next unprocessed arrival (or set ``_pending`` to None)."""
        try:
            self._pending = next(self._arrivals)
            if self._time_min is None:
                self._time_min = self._pending.toa_us
            self._time_max = max(self._time_max, self._pending.toa_us)
        except StopIteration:
            self._pending = None

    def add_callback(self, cb: Callable[[SimulationEvent], None]) -> None:
        """Attach another event consumer after construction (e.g. a model)."""
        self._callbacks.append(cb)

    def step(self) -> Optional[SimulationEvent]:
        """Advance the clock to the next event and process it.

        Returns the :class:`SimulationEvent` for the processed instant
        (entry / exit / snapshot) or ``None`` once the run is finished.
        """
        if self.done:
            return None

        next_arrival = self._pending.toa_us if self._pending is not None else math.inf
        next_exit = self._exit_heap[0][0] if self._exit_heap else math.inf

        # The simulation ends once every arrival has entered and every pulse has
        # left the environment -- further periodic snapshot ticks would only
        # emit empty frames forever, so they do not keep the clock alive.
        if math.isinf(next_arrival) and math.isinf(next_exit):
            self.done = True
            return None

        next_tick = self._next_tick if self._next_tick is not None else math.inf
        t = min(next_arrival, next_exit, next_tick)

        if math.isinf(t):
            self.done = True
            return None

        if self.time_us is None or t > self.time_us:
            self.time_us = t
        t = self.time_us

        emitted: Optional[SimulationEvent] = None

        # 1) All arrivals with ToA <= t enter the environment.
        while self._pending is not None and self._pending.toa_us <= t:
            self._add_pulse(self._pending)
            self._prime()

        # 2) All pulses whose exit time has passed disappear completely.
        while self._exit_heap and self._exit_heap[0][0] <= t:
            _, _, pulse_id = heapq.heappop(self._exit_heap)
            ap = self.active.pop(pulse_id, None)
            if ap is not None:
                self.total_exits += 1
                emitted = SimulationEvent(
                    event_type="exit",
                    time_us=ap.exit_us,
                    pulse_id=pulse_id,
                    pulse=ap,
                    active_count=len(self.active),
                )
                self._on_event(emitted)

        # 3) If this instant is (past) a grid tick, emit snapshot(s).
        if self._next_tick is not None and t >= self._next_tick:
            while self._next_tick is not None and t >= self._next_tick:
                tick = self._next_tick
                self.total_snapshots += 1
                emitted = SimulationEvent(
                    event_type="snapshot",
                    time_us=tick,
                    active_pulses=list(self.active.values()),
                    active_count=len(self.active),
                )
                self._on_event(emitted)
                self._next_tick = tick + self._snap_interval

        return emitted

    # ------------------------------------------------------------- internals

    def _add_pulse(self, rec: PulseRecord) -> None:
        # A pulse is active for exactly its pulse width: it enters at its ToA
        # and disappears at ToA + PW. Invalid (<=0 / non-finite) widths are
        # already rejected upstream by the validation layer / ingest, so no
        # record is ever turned into a fake zero-duration pulse here.
        exit_us = rec.toa_us + rec.pulse_width_us
        # Defense-in-depth: emit canonical AoA in [0, 360) even if a raw file
        # was fed directly to the source (the validation layer normally folds
        # signed angles already).
        aoa = rec.aoa_deg
        if aoa < self.config.min_aoa_deg or aoa >= self.config.max_aoa_deg:
            span = self.config.max_aoa_deg - self.config.min_aoa_deg
            aoa = ((aoa - self.config.min_aoa_deg) % span) + self.config.min_aoa_deg
        ap = ActivePulse(
            pulse_id=self._pulse_seq,
            toa_us=rec.toa_us,
            frequency_mhz=rec.frequency_mhz,
            pulse_width_us=rec.pulse_width_us,
            amplitude_db=rec.amplitude_db,
            aoa_deg=aoa,
            emitter_id=rec.emitter_id,
            exit_us=exit_us,
            source_id=rec.source_id,
        )
        self.active[self._pulse_seq] = ap
        heapq.heappush(self._exit_heap, (exit_us, self._exit_seq, self._pulse_seq))
        self._exit_seq += 1
        self._pulse_seq += 1
        self.total_entries += 1

        event = SimulationEvent(
            event_type="entry",
            time_us=rec.toa_us,
            pulse_id=ap.pulse_id,
            pulse=ap,
            active_count=len(self.active),
        )
        self._on_event(event)

    def _on_event(self, event: SimulationEvent) -> None:
        """Dispatch an event to every user-supplied callback, in order."""
        for cb in self._callbacks:
            cb(event)

    def add_callback(self, cb: Callable[[SimulationEvent], None]) -> None:
        """Attach another event consumer after construction (e.g. a model)."""
        self._callbacks.append(cb)

    # --------------------------------------------------------------- run-all

    def run(self) -> None:
        """Run the full sweep to completion, invoking ``on_event`` as it goes."""
        while not self.done:
            self.step()
