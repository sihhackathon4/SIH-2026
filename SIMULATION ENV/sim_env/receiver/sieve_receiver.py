"""Deterministic sieve/scanning RF receiver.

The receiver models a limited-bandwidth RF receiver operating in the
repository's native units:

    frequency -> MHz
    time      -> microseconds
    amplitude -> relative dB
    AoA       -> degrees

The receiver has its own clock and scan state.  RF pulses become known to
the receiver only when the environment delivers their entry events.

No ML scheduler is implemented here.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .models import DetectionObservation, ReceiverObservation


__all__ = [
    "SieveReceiver",
    "ReceiverConfigError",
    "to_hz",
    "to_ghz",
    "MHZ_TO_HZ",
    "ACTION_TUNE",
    "ACTION_STEP_UP",
    "ACTION_STEP_DOWN",
    "ACTION_DWELL",
]


MHZ_TO_HZ = 1_000_000.0

ACTION_TUNE = "TUNE"
ACTION_STEP_UP = "STEP_UP"
ACTION_STEP_DOWN = "STEP_DOWN"
ACTION_DWELL = "DWELL"


def to_hz(frequency_mhz: float) -> float:
    """Convert MHz to Hz."""
    return float(frequency_mhz) * MHZ_TO_HZ


def to_ghz(frequency_mhz: float) -> float:
    """Convert MHz to GHz."""
    return float(frequency_mhz) / 1000.0


class ReceiverConfigError(ValueError):
    """Raised when the receiver configuration is physically invalid."""


class SieveReceiver:
    """Deterministic scanning receiver with a static interception policy.

    The receiver does not know future RF events.

    Environment entry events are added to the receiver's pulse buffer.
    Environment exit events remove pulses from the buffer.

    The receiver independently maintains:

        - current_time_us
        - dwell_start_us
        - center_frequency_mhz
        - frequency_step_mhz

    The static scan therefore continues independently of RF arrival timing.
    """

    def __init__(
        self,
        total_bandwidth: float = 18e3,
        ibw: float = 1e3,
        frequency_step: float = 500.0,
        dwell_time: float = 100.0,
        detection_threshold_db: float = -140.0,
        spectrum_threshold: float = 5.0,
    ) -> None:
        self._validate_config(
            total_bandwidth=total_bandwidth,
            ibw=ibw,
            frequency_step=frequency_step,
            dwell_time=dwell_time,
            detection_threshold_db=detection_threshold_db,
            spectrum_threshold=spectrum_threshold,
        )

        # Repository receiver units.
        self.total_bandwidth_mhz = float(total_bandwidth)
        self.ibw_mhz = float(ibw)
        self.frequency_step_mhz = float(frequency_step)

        self.dwell_time_us = float(dwell_time)

        # Real RF pulse sensitivity.
        self.detection_threshold_db = float(
            detection_threshold_db
        )

        # Synthetic-spectrum detection threshold.
        self.spectrum_threshold = float(
            spectrum_threshold
        )

        # Receiver state.
        self.center_frequency_mhz = 0.0
        self.current_time_us = 0.0

        # Start of the currently active dwell interval.
        self.dwell_start_us = 0.0

        # Detections belonging to the most recently completed/manual dwell.
        self.detections: List[DetectionObservation] = []

        self.last_observation: Optional[
            ReceiverObservation
        ] = None

        # Complete detection history for live integration/debugging.
        self.detection_history: List[
            DetectionObservation
        ] = []

        # Pulses that have ACTUALLY ARRIVED from the environment.
        #
        # Key:
        #     pulse_id where available
        #
        # Value:
        #     repository ActivePulse/PulseRecord/dict
        self._pulse_buffer: Dict[Any, Any] = {}

        # Deterministic scan state.
        self.scan_count = 0

        # Static scan direction used by the internal scan loop.
        self._scan_direction = 1

        self.reset()

    # ================================================================
    # Configuration
    # ================================================================

    @staticmethod
    def _validate_config(
        total_bandwidth: float,
        ibw: float,
        frequency_step: float,
        dwell_time: float,
        detection_threshold_db: float,
        spectrum_threshold: float,
    ) -> None:
        if (
            not math.isfinite(float(total_bandwidth))
            or float(total_bandwidth) <= 0
        ):
            raise ReceiverConfigError(
                "total_bandwidth must be finite and > 0, "
                f"got {total_bandwidth!r}"
            )

        if (
            not math.isfinite(float(ibw))
            or float(ibw) <= 0
        ):
            raise ReceiverConfigError(
                f"ibw must be finite and > 0, got {ibw!r}"
            )

        if float(ibw) > float(total_bandwidth):
            raise ReceiverConfigError(
                f"ibw ({ibw}) cannot exceed "
                f"total_bandwidth ({total_bandwidth})"
            )

        if (
            not math.isfinite(float(frequency_step))
            or float(frequency_step) <= 0
        ):
            raise ReceiverConfigError(
                "frequency_step must be finite and > 0, "
                f"got {frequency_step!r}"
            )

        if (
            not math.isfinite(float(dwell_time))
            or float(dwell_time) <= 0
        ):
            raise ReceiverConfigError(
                "dwell_time must be finite and > 0, "
                f"got {dwell_time!r}"
            )

        if not math.isfinite(float(detection_threshold_db)):
            raise ReceiverConfigError(
                "detection_threshold_db must be finite, "
                f"got {detection_threshold_db!r}"
            )

        if not math.isfinite(float(spectrum_threshold)):
            raise ReceiverConfigError(
                "spectrum_threshold must be finite, "
                f"got {spectrum_threshold!r}"
            )

    # ================================================================
    # Legal center-frequency range
    # ================================================================

    @property
    def legal_center_min_mhz(self) -> float:
        return self.ibw_mhz / 2.0

    @property
    def legal_center_max_mhz(self) -> float:
        return (
            self.total_bandwidth_mhz
            - self.ibw_mhz / 2.0
        )

    def _clip_center(
        self,
        frequency_mhz: float,
    ) -> float:
        return min(
            max(
                float(frequency_mhz),
                self.legal_center_min_mhz,
            ),
            self.legal_center_max_mhz,
        )

    # ================================================================
    # Reset
    # ================================================================

    def reset(self) -> None:
        """Reset receiver to a completely deterministic initial state."""
        self.center_frequency_mhz = (
            self.legal_center_min_mhz
        )

        self.current_time_us = 0.0
        self.dwell_start_us = 0.0

        self.detections = []
        self.last_observation = None
        self.detection_history = []

        self._pulse_buffer.clear()

        self.scan_count = 0
        self._scan_direction = 1

    # ================================================================
    # Tuning
    # ================================================================

    def tune(
        self,
        frequency_mhz: float,
    ) -> float:
        """Tune receiver center frequency in MHz.

        The requested value is clipped so the complete IBW remains inside
        the available spectrum.
        """
        if not math.isfinite(
            float(frequency_mhz)
        ):
            raise ValueError(
                "invalid non-finite center frequency "
                f"{frequency_mhz!r}"
            )

        self.center_frequency_mhz = (
            self._clip_center(frequency_mhz)
        )

        return self.center_frequency_mhz

    def step_up(self) -> float:
        """Move the center frequency upward by one configured step."""
        self.center_frequency_mhz = self._clip_center(
            self.center_frequency_mhz
            + self.frequency_step_mhz
        )

        self._scan_direction = 1

        return self.center_frequency_mhz

    def step_down(self) -> float:
        """Move the center frequency downward by one configured step."""
        self.center_frequency_mhz = self._clip_center(
            self.center_frequency_mhz
            - self.frequency_step_mhz
        )

        self._scan_direction = -1

        return self.center_frequency_mhz

    # ================================================================
    # Frequency window
    # ================================================================

    def get_frequency_window(
        self,
    ) -> Tuple[float, float]:
        """Return current IBW as (lower_mhz, upper_mhz)."""
        lower = (
            self.center_frequency_mhz
            - self.ibw_mhz / 2.0
        )

        upper = (
            self.center_frequency_mhz
            + self.ibw_mhz / 2.0
        )

        return lower, upper

    def frequency_in_window(
        self,
        frequency_mhz: float,
    ) -> bool:
        """Return True when frequency lies inside the current IBW."""
        if not math.isfinite(
            float(frequency_mhz)
        ):
            return False

        lower, upper = (
            self.get_frequency_window()
        )

        return (
            lower
            <= float(frequency_mhz)
            <= upper
        )

    # ================================================================
    # Synthetic spectrum mode
    # ================================================================

    def observe(
        self,
        spectrum,
        frequencies,
    ):
        """Observe a synthetic NumPy spectrum through the current IBW.

        Frequencies are expressed in MHz.
        """
        if frequencies is None or len(frequencies) == 0:
            raise ValueError(
                "frequencies must be a non-empty "
                "array/sequence"
            )

        if spectrum is None or len(spectrum) == 0:
            raise ValueError(
                "spectrum must be a non-empty "
                "array/sequence"
            )

        if len(frequencies) != len(spectrum):
            raise ValueError(
                "frequencies and spectrum must have "
                "equal length"
            )

        frequencies_array = np.asarray(
            frequencies
        )

        spectrum_array = np.asarray(
            spectrum
        )

        lower, upper = (
            self.get_frequency_window()
        )

        mask = (
            (frequencies_array >= lower)
            &
            (frequencies_array <= upper)
        )

        return (
            frequencies_array[mask],
            spectrum_array[mask],
        )

    def detect(
        self,
        observed_spectrum,
    ) -> bool:
        """Detect a signal in a synthetic normalized spectrum."""
        array = np.asarray(
            observed_spectrum
        )

        if array.size == 0:
            return False

        peak_power = np.max(array)

        return bool(
            peak_power
            >= self.spectrum_threshold
        )

    def perform_dwell(
        self,
        spectrum,
        frequencies,
    ):
        """Synthetic mode: observe current spectrum and advance one dwell."""
        observed_frequencies, observed_spectrum = (
            self.observe(
                spectrum,
                frequencies,
            )
        )

        self.dwell()

        return (
            observed_frequencies,
            observed_spectrum,
        )

    # ================================================================
    # Pulse extraction
    # ================================================================

    @staticmethod
    def _pulse_values(
        pulse,
    ) -> Tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        Optional[int],
    ]:
        """Extract standard RF fields from repository pulse forms."""
        if isinstance(pulse, dict):
            getter = pulse.get

            frequency = getter(
                "frequency_mhz"
            )

            toa = getter(
                "toa_us",
                getter("time_us"),
            )

            width = getter(
                "pulse_width_us"
            )

            exit_us = getter(
                "exit_us"
            )

            amplitude = getter(
                "amplitude_db"
            )

            aoa = getter(
                "aoa_deg"
            )

            pulse_id = getter(
                "pulse_id"
            )

        else:
            frequency = getattr(
                pulse,
                "frequency_mhz",
                None,
            )

            toa = getattr(
                pulse,
                "toa_us",
                getattr(
                    pulse,
                    "time_us",
                    None,
                ),
            )

            width = getattr(
                pulse,
                "pulse_width_us",
                None,
            )

            exit_us = getattr(
                pulse,
                "exit_us",
                None,
            )

            amplitude = getattr(
                pulse,
                "amplitude_db",
                None,
            )

            aoa = getattr(
                pulse,
                "aoa_deg",
                None,
            )

            pulse_id = getattr(
                pulse,
                "pulse_id",
                None,
            )

        # ------------------------------------------------------------
        # Validate required values.
        # ------------------------------------------------------------

        if frequency is None:
            raise ValueError(
                "pulse is missing frequency_mhz"
            )

        if toa is None:
            raise ValueError(
                "pulse is missing toa_us"
            )

        if width is None:
            raise ValueError(
                "pulse is missing pulse_width_us"
            )

        if exit_us is None:
            exit_us = (
                float(toa)
                + float(width)
            )

        if amplitude is None:
            raise ValueError(
                "pulse is missing amplitude_db"
            )

        if not math.isfinite(float(frequency)):
            raise ValueError(
                f"invalid pulse frequency {frequency!r}"
            )

        if not math.isfinite(float(toa)):
            raise ValueError(
                f"invalid pulse ToA {toa!r}"
            )

        if not math.isfinite(float(width)):
            raise ValueError(
                f"invalid pulse width {width!r}"
            )

        if float(width) <= 0:
            raise ValueError(
                f"pulse width must be > 0, got {width!r}"
            )

        if not math.isfinite(float(exit_us)):
            raise ValueError(
                f"invalid pulse exit time {exit_us!r}"
            )

        if float(exit_us) <= float(toa):
            raise ValueError(
                "pulse exit_us must be greater than "
                "toa_us"
            )

        if not math.isfinite(float(amplitude)):
            raise ValueError(
                f"invalid amplitude {amplitude!r}"
            )

        if aoa is None:
            aoa = 0.0

        if not math.isfinite(float(aoa)):
            raise ValueError(
                f"invalid AoA {aoa!r}"
            )

        return (
            float(frequency),
            float(toa),
            float(exit_us),
            float(width),
            float(amplitude),
            float(aoa) % 360.0,
            pulse_id,
        )

    # ================================================================
    # Pulse visibility
    # ================================================================

    def _is_time_visible(
        self,
        toa_us: float,
        exit_us: float,
        sample_time_us: Optional[float] = None,
    ) -> bool:
        """Instantaneous visibility under [ToA, exit_us)."""
        t = (
            self.current_time_us
            if sample_time_us is None
            else float(sample_time_us)
        )

        return (
            toa_us
            <= t
            < exit_us
        )

    def _pulse_overlaps_interval(
        self,
        toa_us: float,
        exit_us: float,
        start_us: float,
        end_us: float,
    ) -> bool:
        """Return True when [ToA, exit) overlaps [start, end)."""
        return (
            toa_us < end_us
            and exit_us > start_us
        )

    def _is_amplitude_visible(
        self,
        amplitude_db: float,
    ) -> bool:
        """Real RF detection uses a dB sensitivity floor."""
        return (
            amplitude_db
            >= self.detection_threshold_db
        )

    # ================================================================
    # Detection evaluation
    # ================================================================

    def _evaluate(
        self,
        frequency_mhz: float,
        toa_us: float,
        exit_us: float,
        width_us: float,
        amplitude_db: float,
        aoa_deg: float,
        pulse_id,
        sample_time_us: Optional[float] = None,
    ) -> DetectionObservation:
        """Evaluate one pulse at one receiver time."""
        sample_time = (
            self.current_time_us
            if sample_time_us is None
            else float(sample_time_us)
        )

        in_window = (
            math.isfinite(frequency_mhz)
            and self.frequency_in_window(
                frequency_mhz
            )
        )

        time_ok = (
            math.isfinite(toa_us)
            and math.isfinite(exit_us)
            and toa_us < exit_us
            and self._is_time_visible(
                toa_us,
                exit_us,
                sample_time,
            )
        )

        amplitude_ok = (
            math.isfinite(amplitude_db)
            and self._is_amplitude_visible(
                amplitude_db
            )
        )

        detected = (
            in_window
            and time_ok
            and amplitude_ok
        )

        return DetectionObservation(
            time_us=sample_time,
            frequency_mhz=frequency_mhz,
            pulse_width_us=width_us,
            amplitude_db=amplitude_db,
            aoa_deg=aoa_deg,
            pulse_id=pulse_id,
            center_frequency_mhz=(
                self.center_frequency_mhz
            ),
            detected=detected,
        )

    def _evaluate_interval(
        self,
        pulse,
        start_us: float,
        end_us: float,
    ) -> Optional[DetectionObservation]:
        """Evaluate whether a pulse is detectable during a dwell interval."""
        (
            frequency_mhz,
            toa_us,
            exit_us,
            width_us,
            amplitude_db,
            aoa_deg,
            pulse_id,
        ) = self._pulse_values(pulse)

        if not self.frequency_in_window(
            frequency_mhz
        ):
            return None

        if not self._pulse_overlaps_interval(
            toa_us,
            exit_us,
            start_us,
            end_us,
        ):
            return None

        if not self._is_amplitude_visible(
            amplitude_db
        ):
            return None

        # The receiver can only know about the pulse from its arrival.
        detection_time = max(
            start_us,
            toa_us,
        )

        # Never claim detection after the pulse exited.
        if detection_time >= exit_us:
            return None

        return DetectionObservation(
            time_us=detection_time,
            frequency_mhz=frequency_mhz,
            pulse_width_us=width_us,
            amplitude_db=amplitude_db,
            aoa_deg=aoa_deg,
            pulse_id=pulse_id,
            center_frequency_mhz=(
                self.center_frequency_mhz
            ),
            detected=True,
        )

    # ================================================================
    # Direct pulse API
    # ================================================================

    def could_detect(
        self,
        pulse,
    ) -> DetectionObservation:
        """Evaluate one pulse at the receiver's current instant."""
        (
            frequency_mhz,
            toa_us,
            exit_us,
            width_us,
            amplitude_db,
            aoa_deg,
            pulse_id,
        ) = self._pulse_values(pulse)

        return self._evaluate(
            frequency_mhz,
            toa_us,
            exit_us,
            width_us,
            amplitude_db,
            aoa_deg,
            pulse_id,
        )

    def process_pulse(
        self,
        pulse,
    ) -> Optional[DetectionObservation]:
        """Evaluate a pulse against the current receiver state."""
        detection = self.could_detect(
            pulse
        )

        if detection.detected:
            return detection

        return None

    # ================================================================
    # LIVE PULSE BUFFER
    # ================================================================

    def _pulse_key(
        self,
        pulse,
    ):
        """Return a stable buffer key."""
        if isinstance(pulse, dict):
            pulse_id = pulse.get(
                "pulse_id"
            )
        else:
            pulse_id = getattr(
                pulse,
                "pulse_id",
                None,
            )

        if pulse_id is not None:
            return pulse_id

        # Fallback for a pulse object that lacks a pulse ID.
        return id(pulse)

    def add_pulse(
        self,
        pulse,
    ):
        """Add a pulse that has actually arrived in the environment.

        This method does NOT advance time and does NOT retune the receiver.
        """
        self._pulse_values(
            pulse
        )

        key = self._pulse_key(
            pulse
        )

        self._pulse_buffer[key] = pulse

        return key

    def remove_pulse(
        self,
        pulse_id,
    ) -> bool:
        """Remove a pulse after its environment exit event."""
        if pulse_id in self._pulse_buffer:
            del self._pulse_buffer[
                pulse_id
            ]
            return True

        return False

    def buffered_pulses(
        self,
    ) -> List[Any]:
        """Return a copy of pulses currently known to the receiver."""
        return list(
            self._pulse_buffer.values()
        )

    def _current_pulses(
        self,
    ) -> List[Any]:
        """Return the currently buffered live pulse set."""
        return self.buffered_pulses()

    # ================================================================
    # Detection of current buffered state
    # ================================================================

    def _detect_buffered_interval(
        self,
        start_us: float,
        end_us: float,
    ) -> List[DetectionObservation]:
        """Detect all buffered pulses visible during an interval."""
        detections: List[
            DetectionObservation
        ] = []

        # Snapshot the values so callbacks/removals cannot mutate iteration.
        for pulse in list(
            self._pulse_buffer.values()
        ):
            detection = self._evaluate_interval(
                pulse,
                start_us,
                end_us,
            )

            if detection is not None:
                detections.append(
                    detection
                )

        # Deterministic ordering.
        detections.sort(
            key=lambda item: (
                item.time_us,
                item.frequency_mhz,
                -(
                    item.pulse_id
                    if item.pulse_id is not None
                    else -1
                ),
            )
        )

        return detections

    def _detect_buffered_at_current_time(
        self,
    ) -> List[DetectionObservation]:
        """Detect pulses visible at the current receiver instant."""
        detections: List[
            DetectionObservation
        ] = []

        for pulse in list(
            self._pulse_buffer.values()
        ):
            detection = self.process_pulse(
                pulse
            )

            if detection is not None:
                detections.append(
                    detection
                )

        detections.sort(
            key=lambda item: (
                item.time_us,
                item.frequency_mhz,
                -(
                    item.pulse_id
                    if item.pulse_id is not None
                    else -1
                ),
            )
        )

        return detections

    def _store_detections(
        self,
        detections: Sequence[DetectionObservation],
    ) -> None:
        self.detections = list(
            detections
        )

        self.detection_history.extend(
            detections
        )

    # ================================================================
    # Observation creation
    # ================================================================

    def _record(
        self,
        detections: Sequence[
            DetectionObservation
        ],
        observation_time_us: Optional[float] = None,
    ) -> ReceiverObservation:
        """Create the latest receiver observation."""
        self._store_detections(
            detections
        )

        time_us = (
            self.current_time_us
            if observation_time_us is None
            else float(observation_time_us)
        )

        observation = ReceiverObservation(
            time_us=time_us,
            center_frequency_mhz=(
                self.center_frequency_mhz
            ),
            ibw_mhz=self.ibw_mhz,
            dwell_time_us=self.dwell_time_us,
            window_mhz=list(
                self.get_frequency_window()
            ),
            detections=list(
                detections
            ),
        )

        self.last_observation = (
            observation
        )

        return observation

    def get_observation(
        self,
    ) -> ReceiverObservation:
        """Return the most recent receiver observation."""
        if self.last_observation is None:
            self._record([])

        return self.last_observation

    # ================================================================
    # Static scan frequency movement
    # ================================================================

    def _advance_scan_frequency(
        self,
    ) -> float:
        """Advance scan frequency for the internal static scanner.

        This uses a deterministic bouncing scan:

            min -> ... -> max -> ... -> min -> ...

        Public ``step_up``/``step_down`` retain their clipping behavior.
        """
        if (
            self.legal_center_max_mhz
            <= self.legal_center_min_mhz
        ):
            return self.center_frequency_mhz

        next_frequency = (
            self.center_frequency_mhz
            + (
                self._scan_direction
                * self.frequency_step_mhz
            )
        )

        if (
            self._scan_direction > 0
            and next_frequency
            >= self.legal_center_max_mhz
        ):
            self.center_frequency_mhz = (
                self.legal_center_max_mhz
            )
            self._scan_direction = -1
            return self.center_frequency_mhz

        if (
            self._scan_direction < 0
            and next_frequency
            <= self.legal_center_min_mhz
        ):
            self.center_frequency_mhz = (
                self.legal_center_min_mhz
            )
            self._scan_direction = 1
            return self.center_frequency_mhz

        self.center_frequency_mhz = (
            next_frequency
        )

        return self.center_frequency_mhz

    # ================================================================
    # Dwell
    # ================================================================

    def dwell(self) -> float:
        """Advance receiver clock by one dwell without changing frequency."""
        self.current_time_us += (
            self.dwell_time_us
        )

        self.dwell_start_us = (
            self.current_time_us
        )

        return self.current_time_us

    # ================================================================
    # Environment-time synchronization
    # ================================================================

    def advance_to(
        self,
        target_time_us: float,
    ) -> List[ReceiverObservation]:
        """Advance the receiver to a future environment time.

        This is the key live-simulation mechanism.

        The receiver DOES NOT jump its state to the incoming pulse.
        Instead, its clock advances monotonically through its normal
        dwell intervals. Completed dwell intervals are processed against
        only pulses that have already arrived in the receiver buffer.

        ``target_time_us`` must not be earlier than the current time.
        """
        target = float(
            target_time_us
        )

        if not math.isfinite(target):
            raise ValueError(
                f"target_time_us must be finite, got {target!r}"
            )

        if target < self.current_time_us:
            raise ValueError(
                "receiver time cannot move backwards: "
                f"{target} < {self.current_time_us}"
            )

        observations: List[
            ReceiverObservation
        ] = []

        while (
            target
            >= self.dwell_start_us
            + self.dwell_time_us
        ):
            dwell_end = (
                self.dwell_start_us
                + self.dwell_time_us
            )

            detections = (
                self._detect_buffered_interval(
                    self.dwell_start_us,
                    dwell_end,
                )
            )

            self.current_time_us = (
                dwell_end
            )

            observation = self._record(
                detections,
                observation_time_us=dwell_end,
            )

            observations.append(
                observation
            )

            self.dwell_start_us = (
                dwell_end
            )

            self._advance_scan_frequency()

            self.scan_count += 1

        self.current_time_us = target

        return observations

    # ================================================================
    # Environment event handling
    # ================================================================

    @staticmethod
    def _event_type_and_pulse(
        event,
    ):
        if isinstance(event, dict):
            return (
                event.get("event"),
                event.get("pulse"),
            )

        return (
            getattr(
                event,
                "event_type",
                getattr(
                    event,
                    "event",
                    None,
                ),
            ),
            getattr(
                event,
                "pulse",
                None,
            ),
        )

    @staticmethod
    def _event_time(
        event,
        pulse=None,
    ) -> Optional[float]:
        if isinstance(event, dict):
            value = event.get(
                "time_us"
            )

            if value is not None:
                return float(value)

        else:
            value = getattr(
                event,
                "time_us",
                None,
            )

            if value is not None:
                return float(value)

        if pulse is not None:
            if isinstance(pulse, dict):
                value = pulse.get(
                    "toa_us",
                    pulse.get(
                        "time_us"
                    ),
                )
            else:
                value = getattr(
                    pulse,
                    "toa_us",
                    getattr(
                        pulse,
                        "time_us",
                        None,
                    ),
                )

            if value is not None:
                return float(value)

        return None

    @staticmethod
    def _ground_truth_emitter_id(
        event,
        pulse,
    ) -> Optional[int]:
        """Read emitter ID as ground truth only.

        Actual NDJSON places emitter_id inside ``pulse``.
        """
        if isinstance(event, dict):
            value = event.get(
                "ground_truth_emitter_id"
            )

            if value is None:
                value = event.get(
                    "emitter_id"
                )

            if value is None and isinstance(
                pulse,
                dict,
            ):
                value = pulse.get(
                    "emitter_id"
                )

        else:
            value = getattr(
                event,
                "ground_truth_emitter_id",
                None,
            )

            if value is None:
                value = getattr(
                    event,
                    "emitter_id",
                    None,
                )

            if value is None and pulse is not None:
                value = getattr(
                    pulse,
                    "emitter_id",
                    None,
                )

        if value is None:
            return None

        return int(value)

    def handle_environment_event(
        self,
        event,
    ) -> Optional[DetectionObservation]:
        """Consume one actual environment event.

        Rules:

        ENTRY:
            1. advance receiver clock to event time;
            2. add the pulse;
            3. evaluate it at the current receiver state.

        EXIT:
            1. advance receiver clock to event time;
            2. remove the pulse.

        SNAPSHOT:
            ignored for pulse-buffer bookkeeping.

        The receiver never retunes based on the arriving pulse.
        """
        event_type, pulse = (
            self._event_type_and_pulse(
                event
            )
        )

        event_time = self._event_time(
            event,
            pulse,
        )

        if event_time is not None:
            self.advance_to(
                event_time
            )

        if event_type == "entry":
            if pulse is None:
                return None

            self.add_pulse(
                pulse
            )

            detection = (
                self.process_pulse(
                    pulse
                )
            )

            if detection is None:
                return None

            # Ground truth only.
            emitter_id = (
                self._ground_truth_emitter_id(
                    event,
                    pulse,
                )
            )

            if emitter_id is not None:
                detection.emitter_id = (
                    emitter_id
                )

            self._store_detections(
                [detection]
            )

            # Preserve a useful latest observation at the actual arrival time.
            self._record(
                [detection],
                observation_time_us=(
                    self.current_time_us
                ),
            )

            return detection

        if event_type == "exit":
            if pulse is not None:
                if isinstance(
                    pulse,
                    dict,
                ):
                    pulse_id = pulse.get(
                        "pulse_id"
                    )
                else:
                    pulse_id = getattr(
                        pulse,
                        "pulse_id",
                        None,
                    )

                if pulse_id is not None:
                    self.remove_pulse(
                        pulse_id
                    )
            else:
                pulse_id = (
                    event.get("pulse_id")
                    if isinstance(
                        event,
                        dict,
                    )
                    else getattr(
                        event,
                        "pulse_id",
                        None,
                    )
                )

                if pulse_id is not None:
                    self.remove_pulse(
                        pulse_id
                    )

            return None

        return None

    def process_event(
        self,
        event,
    ) -> Optional[DetectionObservation]:
        """Backward-compatible public event-processing API."""
        return self.handle_environment_event(
            event
        )

    # ================================================================
    # Static scan
    # ================================================================

    def scan_once(
        self,
        pulses=None,
    ) -> ReceiverObservation:
        """Perform one deterministic static scan dwell.

        1. Evaluate the current dwell interval.
        2. Record detections.
        3. Advance time by one dwell.
        4. Advance deterministic scan frequency.

        If ``pulses`` is omitted, only pulses currently known to the
        receiver are considered.
        """
        dwell_start = (
            self.current_time_us
        )

        dwell_end = (
            dwell_start
            + self.dwell_time_us
        )

        candidates = (
            self._current_pulses()
            if pulses is None
            else list(pulses)
        )

        detections: List[
            DetectionObservation
        ] = []

        for pulse in candidates:
            detection = (
                self._evaluate_interval(
                    pulse,
                    dwell_start,
                    dwell_end,
                )
            )

            if detection is not None:
                detections.append(
                    detection
                )

        detections.sort(
            key=lambda item: (
                item.time_us,
                item.frequency_mhz,
                -(
                    item.pulse_id
                    if item.pulse_id is not None
                    else -1
                ),
            )
        )

        self.current_time_us = (
            dwell_end
        )

        self.dwell_start_us = (
            dwell_end
        )

        observation = self._record(
            detections,
            observation_time_us=(
                dwell_start
            ),
        )

        self._advance_scan_frequency()

        self.scan_count += 1

        return observation

    def scan(
        self,
        n_pulses: Optional[int] = None,
    ) -> List[ReceiverObservation]:
        """Run a deterministic sequence of static scan dwells."""
        if n_pulses is None:
            raise ValueError(
                "scan() requires n_pulses"
            )

        n = int(
            n_pulses
        )

        if n < 0:
            raise ValueError(
                "n_pulses must be >= 0"
            )

        observations: List[
            ReceiverObservation
        ] = []

        for _ in range(n):
            observations.append(
                self.scan_once()
            )

        return observations

    # ================================================================
    # Future scheduler-compatible action API
    # ================================================================

    def apply_action(
        self,
        action: str,
        value: Optional[float] = None,
    ):
        """Apply a deterministic receiver action.

        No ML logic exists here.

        Supported actions:

            TUNE
            STEP_UP
            STEP_DOWN
            DWELL
        """
        normalized_action = str(
            action
        ).upper()

        if normalized_action == ACTION_TUNE:
            if value is None:
                raise ValueError(
                    "TUNE requires a frequency value in MHz"
                )

            return self.tune(
                float(value)
            )

        if normalized_action == ACTION_STEP_UP:
            return self.step_up()

        if normalized_action == ACTION_STEP_DOWN:
            return self.step_down()

        if normalized_action == ACTION_DWELL:
            dwell_start = (
                self.current_time_us
            )

            dwell_end = (
                dwell_start
                + self.dwell_time_us
            )

            detections = (
                self._detect_buffered_interval(
                    dwell_start,
                    dwell_end,
                )
            )

            observation = self._record(
                detections,
                observation_time_us=(
                    dwell_start
                ),
            )

            self.current_time_us = (
                dwell_end
            )

            self.dwell_start_us = (
                dwell_end
            )

            return observation

        raise ValueError(
            f"unknown receiver action {action!r}"
        )
