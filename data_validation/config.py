"""Configuration for the RF data validation layer.

This is the authoritative source for the *physical* rules applied *before*
dirty RF records ever reach the simulation environment. The semantic record
order is::

    (toa_us, frequency_mhz, pulse_width_us, amplitude_db, aoa_deg, emitter_id)

with ``emitter_id`` as the label.

Known source orderings (used by the transform/cleaner to map into the
semantic order):

* HDF5 (source):     ``[ToA, Frequency, PW, AoA, Amplitude]``
* output_*.txt:      ``[ToA, Frequency, PW, Amplitude, AoA]`` (+ ``label``)

Do NOT invent physical limits: only the bounds explicitly required by the
project are enforced here.
"""

from __future__ import annotations

from dataclasses import dataclass

# Semantic field order (as consumed by PulseRecord / NDJSON / ML datasets).
SEMANTIC_FIELDS = (
    "toa_us",
    "frequency_mhz",
    "pulse_width_us",
    "amplitude_db",
    "aoa_deg",
    "emitter_id",
)

# The canonical *data* vector order inside output_*.txt records (5 features,
# label separate) -- matches sim_env FEATURE_FIELDS.
TXT_DATA_FIELDS = (
    "toa_us",
    "frequency_mhz",
    "pulse_width_us",
    "amplitude_db",
    "aoa_deg",
)

# The source HDF5 column order, before the transform swaps AoA/Amplitude.
HDF5_SOURCE_FIELDS = ("toa_us", "frequency_mhz", "pulse_width_us", "aoa_deg", "amplitude_db")


@dataclass
class ValidationConfig:
    """Tunable rules for the RF record validator.

    Attributes:
        min_frequency_mhz:
            Records with ``frequency_mhz <= min_frequency_mhz`` are invalid.
            Set to ``None`` to disable the lower bound. Default ``0.0``
            (frequency must be strictly positive).
        max_frequency_mhz:
            Upper bound on frequency, or ``None`` for no upper bound. We do
            NOT invent a physical upper limit, so this defaults to ``None``.
        min_aoa_deg:
            Lower bound of the canonical AoA range. Default ``0.0``.
        max_aoa_deg:
            Upper bound (exclusive) of the canonical AoA range. Default
            ``360.0``; canonical AoA satisfies ``0 <= aoa < 360``.
        normalize_signed_aoa:
            When ``True``, signed angles outside ``[0, 360)`` are folded with
            ``aoa % 360.0`` (e.g. ``-10 -> 350``) instead of being rejected.
            Default ``True``.
        reject_duplicate_timestamps:
            When ``False`` (default) equal ToA values are *preserved* -- two or
            more emitters may legitimately transmit simultaneously. When
            ``True``, a record whose ToA equals the previous one is rejected.
        min_duration_us:
            Lower bound on an episode's total duration
            ``max(valid_toa) - min(valid_toa)``. A value ``<= 0`` by default
            means no minimum is enforced (reported, not fatal).
        max_duration_us:
            Upper bound on an episode's duration, or ``None`` for none.
            Default ``None``.
    """

    min_frequency_mhz: float = 0.0
    max_frequency_mhz: Optional[float] = None
    min_aoa_deg: float = 0.0
    max_aoa_deg: float = 360.0
    normalize_signed_aoa: bool = True
    reject_duplicate_timestamps: bool = False
    min_duration_us: float = 0.0
    max_duration_us: Optional[float] = None

    def __post_init__(self) -> None:
        if self.max_frequency_mhz is not None and self.min_frequency_mhz is not None:
            if self.max_frequency_mhz <= self.min_frequency_mhz:
                raise ValueError("max_frequency_mhz must exceed min_frequency_mhz")
        if self.max_aoa_deg is not None and self.max_aoa_deg <= self.min_aoa_deg:
            raise ValueError("max_aoa_deg must exceed min_aoa_deg")
        if self.max_duration_us is not None and self.min_duration_us is not None:
            if self.max_duration_us <= self.min_duration_us:
                raise ValueError("max_duration_us must exceed min_duration_us")
