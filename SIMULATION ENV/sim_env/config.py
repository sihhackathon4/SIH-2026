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
            Reserved for backward compatibility. Records whose pulse width is
            below this value are **rejected at ingestion** (never turned into
            zero-duration pulses). The validation layer removes ``PW <= 0``
            records upstream so this is normally already clean.
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
        nonfinite:
            How to treat records whose ``frequency_mhz`` / ``amplitude_db`` /
            ``aoa_deg`` values are ``inf`` / ``nan`` (a real occurrence in the
            legacy corpus). One of:

            * ``"drop"`` (default) -- skip the offending record entirely but
              keep building the stream/window set. Invalid non-finite data is
              **not** silently passed into the environment.
            * ``"raise"`` -- fail loudly on the first non-finite record.
            * ``"allow"`` -- pass non-finite values through untouched (legacy
              behaviour; not recommended now that the validation layer is the
              upstream gate).

            Non-finite ``pulse_width_us`` records are always rejected here,
            independent of this policy.
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
    nonfinite: str = "drop"

    # --- RF validation rules (upstream gate; authoritative in data_validation)
    min_frequency_mhz: float = 0.0
    max_frequency_mhz: Optional[float] = None
    min_aoa_deg: float = 0.0
    max_aoa_deg: float = 360.0
    normalize_signed_aoa: bool = True
    reject_duplicate_timestamps: bool = False
    min_duration_us: float = 0.0
    max_duration_us: Optional[float] = None

    def __post_init__(self) -> None:
        if isinstance(self.inputs, (str, Path)):
            self.inputs = [self.inputs]
        if self.snapshot_interval_us is not None:
            self.snapshot_interval_us = max(0, self.snapshot_interval_us)
        if self.nonfinite not in ("allow", "drop", "raise"):
            raise ValueError(
                f"unknown 'nonfinite' policy {self.nonfinite!r}; "
                "expected one of 'allow', 'drop', 'raise'"
            )


@dataclass
class FeatureStats:
    """Per-feature normalization statistics (mean/std), fit once on train data.

    A deinterleaving transformer needs normalized inputs -- the raw features
    span wildly different scales (frequency in the thousands of MHz, pulse
    width sometimes below 0.01 us). Fit this once on the *training* split
    only, then reuse the exact same stats for validation/test/inference --
    never refit on new data, or train/inference will silently skew.
    """

    mean: dict = field(default_factory=dict)
    std: dict = field(default_factory=dict)

    def normalize(self, feature: str, value: float) -> float:
        m = self.mean.get(feature, 0.0)
        s = self.std.get(feature, 1.0) or 1.0
        return (value - m) / s

    def to_dict(self) -> dict:
        return {"mean": dict(self.mean), "std": dict(self.std)}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureStats":
        return cls(mean=dict(d.get("mean", {})), std=dict(d.get("std", {})))

    @classmethod
    def fit(cls, records: Iterable, fields: Sequence[str] = FEATURE_FIELDS) -> "FeatureStats":
        """Compute mean/std per feature over an iterable of PulseRecord objects.

        Streams the iterable once (does not require it to fit in memory).
        Non-finite values are skipped per feature, so a stray ``inf``/``nan``
        can never propagate into ``nan`` statistics -- an ML pipeline should
        also drop such records at ingestion time (``nonfinite="drop"``), but
        this makes the fit itself robust regardless.
        """
        import math

        sums = {f: 0.0 for f in fields}
        sumsq = {f: 0.0 for f in fields}
        counts = {f: 0 for f in fields}
        any_seen = False
        for rec in records:
            values = dict(zip(FEATURE_FIELDS, rec.data))
            for f in fields:
                v = float(values[f])
                if not math.isfinite(v):
                    continue
                sums[f] += v
                sumsq[f] += v * v
                counts[f] += 1
                any_seen = True
        if not any_seen:
            raise ValueError("cannot fit FeatureStats on zero (finite) records")
        mean = {f: sums[f] / counts[f] for f in fields}
        std = {f: math.sqrt(max(sumsq[f] / counts[f] - mean[f] ** 2, 0.0)) or 1.0
               for f in fields}
        return cls(mean=mean, std=std)
