"""Data models for the receiver component.

These are the structured objects a caller (eventually an ML scheduler)
receives from :class:`~.sieve_receiver.SieveReceiver`. They deliberately keep
*receiver-observable* RF measurements separate from *ground-truth* labels: a
real receiver knows frequency, time, amplitude, width and angle-of-arrival, but
it does NOT magically know the emitter's identity.

Units follow the repository convention:

* frequencies in ``frequency_mhz`` (MHz);
* times / widths in microseconds (``time_us``, ``pulse_width_us``);
* amplitude in relative decibels (``amplitude_db``, non-positive power-like);
* AoA in degrees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["DetectionObservation", "ReceiverObservation"]


@dataclass
class DetectionObservation:
    """One pulse the receiver detected during a single observation window.

    This is the *receiver-observable* representation of a detected pulse. It
    contains only measurements a real receiver could obtain.

    ``emitter_id`` is the ground-truth label attached for testing/validation
    only; it is carried separately and may be ``None``. It is NOT an RF
    measurement a receiver would natively know.
    """

    time_us: float = 0.0
    frequency_mhz: float = 0.0
    pulse_width_us: float = 0.0
    amplitude_db: float = 0.0
    aoa_deg: float = 0.0
    pulse_id: Optional[int] = None
    center_frequency_mhz: float = 0.0
    detected: bool = False
    # --- ground-truth label (for testing), not receiver-observable ----------
    emitter_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging / downstream scheduling input."""
        return {
            "detected": self.detected,
            "time_us": self.time_us,
            "frequency_mhz": self.frequency_mhz,
            "pulse_width_us": self.pulse_width_us,
            "amplitude_db": self.amplitude_db,
            "aoa_deg": self.aoa_deg,
            "pulse_id": self.pulse_id,
            "center_frequency_mhz": self.center_frequency_mhz,
            "emitter_id": self.emitter_id,  # ground-truth label, not a measurement
        }


@dataclass
class ReceiverObservation:
    """The deterministic snapshot of a receiver dwell / scan.

    This is the object a future ML scheduler will observe before choosing an
    action. It describes the receiver state and every detection made during the
    last dwell.
    """

    time_us: float = 0.0
    center_frequency_mhz: float = 0.0
    ibw_mhz: float = 0.0
    dwell_time_us: float = 0.0
    dwell_interval_us: List[float] = field(default_factory=list)
    window_mhz: List[float] = field(default_factory=list)
    detections: List[DetectionObservation] = field(default_factory=list)

    @property
    def detected(self) -> bool:
        return len(self.detections) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the observation (input format for the future scheduler)."""
        return {
            "time_us": self.time_us,
            "center_frequency_mhz": self.center_frequency_mhz,
            "ibw_mhz": self.ibw_mhz,
            "dwell_time_us": self.dwell_time_us,
            "dwell_interval_us": list(self.dwell_interval_us),
            "window_mhz": list(self.window_mhz),
            "detections": [d.to_dict() for d in self.detections],
        }
