"""Configuration for the radiowave stream simulation environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

# ---------------------------------------------------------------------------
# Default physical constants / data semantics
# ---------------------------------------------------------------------------
TOA_US = "toa_us"            # Time of Arrival, microseconds
FREQ_MHZ = "frequency_mhz"   # Carrier frequency, megahertz
PW_US = "pulse_width_us"     # Pulse width, microseconds
AMP_DB = "amplitude_db"      # Amplitude, relative decibel scale
AOA_DEG = "aoa_deg"          # Angle of arrival, degrees
EMITTER_ID = "emitter_id"    # Label: which emitter is transmitting

FEATURE_FIELDS = (TOA_US, FREQ_MHZ, PW_US, AMP_DB, AOA_DEG)

# Human-readable physical units for each feature, keyed by field name.
FEATURE_UNITS = {
    TOA_US: "microseconds",   # Time of Arrival
    FREQ_MHZ: "megahertz",    # Carrier frequency
    PW_US: "microseconds",    # Pulse width
    AMP_DB: "relative db",    # Amplitude
    AOA_DEG: "degrees",       # Angle of arrival
}


@dataclass
class SimConfig:
    """Tunable parameters controlling an environment run.

    Attributes:
        inputs:
            Paths to the RF record files (the `output*.txt` files). Each may
            instead be a glob pattern. The files are streamed in a stable order
            (sorted by name) and their records are merged in arrival order so
            the environment can ingest data continuously without loading the
            whole dataset into memory.
        output_log:
            Where the NDJSON ML-scheduler event log is written. If ``None``
            nothing is written (useful for dry runs / pure API use).
        snapshot_interval_us:
            Emit a periodic "snapshot" event every ``snapshot_interval_us``
            microseconds of simulated time describing *all* currently-active
            pulses. Set to ``None`` to disable period snapshots. The value is
            clamped to an integer number of microseconds.
        metadata_path:
            Optional path where a small NDJSON header/metadata record is
            written alongside the event log (schema version, config, ranges).
            If ``None`` the header is written as the first line of the event
            log itself.
        min_pw_us:
            Records whose pulse width is below this value (or negative /
            non-finite) are treated as having an *instantaneous* presence at
            their ToA and expire immediately (effective PW clamped to 0). This
            guards against the occasional non-physical PW in the source data.
        pulse_cache_size:
            If a pulse id is reused across records (not present in this source
            data, where each record is a distinct pulse), this memory cap keeps
            deduplication bounded. Unused when ids are always fresh.
        emit_entries:
            Whether to write ``entry`` events to the log.
        emit_exits:
            Whether to write ``exit`` events to the log.
        emit_snapshots:
            Whether to write periodic ``snapshot`` events to the log.
        fields:
            Feature field names, in the exact 5-element data order of the
            source files. Kept as configuration so the ingestion layer is
            data-order independent.
    """

    inputs: Sequence[Union[str, Path]] = field(default_factory=list)
    output_log: Optional[Union[str, Path]] = None
    snapshot_interval_us: Optional[float] = 1_000_000.0
    metadata_path: Optional[Union[str, Path]] = None
    min_pw_us: float = 0.0
    pulse_cache_size: int = 1_000_000
    emit_entries: bool = True
    emit_exits: bool = True
    emit_snapshots: bool = True
    fields: tuple = (TOA_US, FREQ_MHZ, PW_US, AMP_DB, AOA_DEG)

    def __post_init__(self) -> None:
        if isinstance(self.inputs, (str, Path)):
            self.inputs = [self.inputs]
        if self.snapshot_interval_us is not None:
            self.snapshot_interval_us = max(0, self.snapshot_interval_us)
