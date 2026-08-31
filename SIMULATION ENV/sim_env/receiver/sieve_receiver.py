"""Sieve / scanning RF receiver component.

A realistic, deterministic simulation of a limited-bandwidth (instantaneous
bandwidth, ``IBW``) scanning receiver. The receiver can only observe the RF
spectrum through its current frequency window during a dwell period, and it
reports the pulses it actually detects.

This replicates the role a real *receiver* plays in the system::

    RF Environment
          |
          v
    RF Pulse/Event Stream
          |
          v
    +------------------+
    |   SieveReceiver  |
    +------------------+
      |      |      |
    IBW   Time   Detection
      +------+------+
             |
             v
     Receiver Observation
             |
             v
     Future ML Scheduler (NOT implemented here)

The receiver's *tuning/interception policy is static and deterministic* for
now. There is no ML scheduler, no policy network, no RL, no reward, no
optimizer. The receiver stays fully usable without any ML code and exposes a
clean, minimal action surface (``tune`` / ``step_up`` / ``step_down`` /
``dwell`` / ``apply_action``) so that an ML scheduler can later replace the
static scanning policy without rewriting receiver internals.

Units
-----
The repository RF corpus and NDJSON stream use **megahertz** for frequency and
**microseconds** for time. The receiver therefore works internally in:

* ``*_mhz`` for every frequency (``total_bandwidth_mhz``, ``ibw_mhz``,
  ``frequency_step_mhz``, ``center_frequency_mhz``);
* ``*_us`` for every time (``current_time_us``, ``dwell_time_us``,
  ``toa_us``, ``exit_us``);
* ``amplitude_db`` in relative decibels (non-positive, power-like);
* ``aoa_deg`` in degrees.

Explicit conversion helpers (``to_hz``, ``to_ghz``) are provided so callers who
think in Hz/GHz never mix units silently. There is NO implicit MHz/Hz mixing.

Boundary rules (documented, deterministic)
------------------------------------------
* Valid center frequency range: ``[ibw/2, total_bandwidth - ibw/2]`` MHz; the
  receiver window never extends outside the available spectrum.
* Frequency visibility: a pulse is in-window when
  ``lower_mhz <= frequency_mhz <= upper_mhz`` (inclusive bounds).
* Time overlap: a pulse is time-visible when its ``[toa_us, exit_us)`` interval
  overlaps the current dwell observation instant ``current_time_us``, i.e.
  ``toa_us <= current_time_us < exit_us`` (half-open, matching the
  environment's pulse semantics: a pulse that exits exactly at
  ``current_time_us`` is no longer present).
* Detection: frequency-visible AND time-visible AND
  ``amplitude_db >= detection_threshold_db``.

Amplitude / detection-threshold semantics
-----------------------------------------
The RF dataset stores amplitude as non-positive, power-like dB values such as
``-121.8``, ``-114.2``, ``-100``. ``detection_threshold_db`` is therefore a
*sensitivity floor in dB*: a pulse is detected when its amplitude is at least
that floor (``amplitude_db >= detection_threshold_db``). To detect weak real
signals around ``-120`` dB you would set a threshold like ``-140`` dB; to detect
only strong signals you raise it (e.g. ``-90`` dB).

The synthetic-spectrum ``observe()``/``detect()`` path keeps its own
independent, normalized positive peak-power threshold (``spectrum_threshold``,
default ``5.0``), because synthetic spectra use arbitrary normalized power
units, not dB. The two thresholds are separate and each is tested separately.
No arbitrary conversion is invented between dB and the synthetic power scale.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .models import DetectionObservation, ReceiverObservation

__all__ = [
    "SieveReceiver",
    "ReceiverConfigError",
    "to_hz",
    "to_ghz",
    "MHZ_TO_HZ",
]

MHZ_TO_HZ = 1_000_000.0


def to_hz(frequency_mhz: float) -> float:
    """Convert MHz to Hz."""
    return frequency_mhz * MHZ_TO_HZ


def to_ghz(frequency_mhz: float) -> float:
    """Convert MHz to GHz."""
    return frequency_mhz / 1000.0


class ReceiverConfigError(ValueError):
    """Raised for physically impossible receiver configuration."""


# Action identifiers for :meth:`SieveReceiver.apply_action`.
ACTION_TUNE = "TUNE"
ACTION_STEP_UP = "STEP_UP"
ACTION_STEP_DOWN = "STEP_DOWN"
ACTION_DWELL = "DWELL"


class SieveReceiver:
    """Deterministic scanning (sieve) receiver with a static policy."""

    def __init__(
        self,
        total_bandwidth: float = 18e3,  # MHz (18 GHz)
        ibw: float = 1e3,               # MHz (1 GHz)
        frequency_step: float = 500.0,  # MHz (500 MHz)
        dwell_time: float = 100.0,      # microseconds
        detection_threshold_db: float = -140.0,
        spectrum_threshold: float = 5.0,
    ):
        # All frequencies are MHz; all times are microseconds.
        self._validate_config(
            total_bandwidth, ibw, frequency_step, dwell_time,
            detection_threshold_db, spectrum_threshold,
        )
        self.total_bandwidth_mhz = float(total_bandwidth)
        self.ibw_mhz = float(ibw)
        self.frequency_step_mhz = float(frequency_step)
        self.dwell_time_us = float(dwell_time)
        self.detection_threshold_db = float(detection_threshold_db)
        self.spectrum_threshold = float(spectrum_threshold)

        self.center_frequency_mhz = 0.0
        self.current_time_us = 0.0
        # Explicit dwell interval bookkeeping (half-open [start_us, end_us)).
        self.dwell_start_us = 0.0
        self.dwell_end_us = 0.0

        # Detections from the most recent dwell (cleared by reset()).
        self.detections: List[DetectionObservation] = []
        self.last_observation: Optional[ReceiverObservation] = None

        # Scan-state bookkeeping (deterministic).
        self.scan_count = 0
        self._scan_direction = 1  # +1 = stepping up; -1 = stepping down

        # Time-aware pulse buffer: pulses the receiver has *learned about* from
        # the environment up to the current time, keyed by pulse_id. This is the
        # receiver's honest knowledge of the RF world -- it never contains
        # pulses that have not yet been announced (no future/information leak),
        # and pulses are expelled once their exit time has passed.
        self._pulses: Dict[int, Dict[str, Any]] = {}

        self.reset()

    # ------------------------------------------------------------------ config

    @staticmethod
    def _validate_config(total_bandwidth, ibw, frequency_step, dwell_time,
                         detection_threshold_db, spectrum_threshold) -> None:
        if not math.isfinite(total_bandwidth) or total_bandwidth <= 0:
            raise ReceiverConfigError(
                f"total_bandwidth must be finite and > 0, got {total_bandwidth!r}")
        if not math.isfinite(ibw) or ibw <= 0:
            raise ReceiverConfigError(f"ibw must be finite and > 0, got {ibw!r}")
        if ibw > total_bandwidth:
            raise ReceiverConfigError(
                f"ibw ({ibw}) cannot exceed total_bandwidth ({total_bandwidth})")
        if not math.isfinite(frequency_step) or frequency_step <= 0:
            raise ReceiverConfigError(
                f"frequency_step must be finite and > 0, got {frequency_step!r}")
        if not math.isfinite(dwell_time) or dwell_time <= 0:
            raise ReceiverConfigError(
                f"dwell_time must be finite and > 0, got {dwell_time!r}")
        if not math.isfinite(detection_threshold_db):
            raise ReceiverConfigError(
                f"detection_threshold_db must be finite, got {detection_threshold_db!r}")

    # ------------------------------------------------------------- legal range

    @property
    def legal_center_min_mhz(self) -> float:
        return self.ibw_mhz / 2.0

    @property
    def legal_center_max_mhz(self) -> float:
        return self.total_bandwidth_mhz - self.ibw_mhz / 2.0

    def _clip_center(self, frequency_mhz: float) -> float:
        return min(max(frequency_mhz, self.legal_center_min_mhz),
                   self.legal_center_max_mhz)

    # ------------------------------------------------------------------- reset

    def reset(self) -> None:
        """Restore a deterministic initial state (no stale detections/scan state)."""
        self.center_frequency_mhz = self.legal_center_min_mhz
        self.current_time_us = 0.0
        self.dwell_start_us = 0.0
        self.dwell_end_us = 0.0
        self.detections = []
        self.last_observation = None
        self.scan_count = 0
        self._scan_direction = 1
        self._pulses.clear()

    # ------------------------------------------------------------------- tune

    def tune(self, frequency_mhz: float) -> float:
        """Set center frequency (MHz), clipped to the legal receiver range.

        ``frequency_mhz`` is the center frequency in **megahertz**. Returns the
        applied (possibly clipped) center frequency.
        """
        if not math.isfinite(frequency_mhz):
            raise ValueError(f"invalid (non-finite) center frequency {frequency_mhz!r}")
        self.center_frequency_mhz = self._clip_center(float(frequency_mhz))
        return self.center_frequency_mhz

    # ---------------------------------------------------------------- stepping

    def step_up(self) -> float:
        """Advance center frequency by ``frequency_step_mhz`` (clipped)."""
        self.center_frequency_mhz = self._clip_center(
            self.center_frequency_mhz + self.frequency_step_mhz)
        self._scan_direction = 1
        return self.center_frequency_mhz

    def step_down(self) -> float:
        """Retreat center frequency by ``frequency_step_mhz`` (clipped)."""
        self.center_frequency_mhz = self._clip_center(
            self.center_frequency_mhz - self.frequency_step_mhz)
        self._scan_direction = -1
        return self.center_frequency_mhz

    # ------------------------------------------------------------------- dwell

    def dwell(self) -> float:
        """Advance simulation time by one dwell period. Returns new current time."""
        self.current_time_us += self.dwell_time_us
        return self.current_time_us

    # ------------------------------------------------------------- frequency

    def get_frequency_window(self) -> Tuple[float, float]:
        """Return ``(lower_mhz, upper_mhz)`` of the current observation band."""
        lower = self.center_frequency_mhz - self.ibw_mhz / 2.0
        upper = self.center_frequency_mhz + self.ibw_mhz / 2.0
        return lower, upper

    def frequency_in_window(self, frequency_mhz: float) -> bool:
        """Is ``frequency_mhz`` inside the current IBW observation band?"""
        lower, upper = self.get_frequency_window()
        return lower <= frequency_mhz <= upper

    # ------------------------------------------------- synthetic spectrum mode

    def observe(self, spectrum, frequencies):
        """(Preserved prototype API) Observe a synthetic spectrum through the IBW.

        ``frequencies`` are expected in **MHz** (matching the repo's unit
        convention). Returns ``(observed_frequencies, observed_spectrum)``.
        """
        if frequencies is None or len(frequencies) == 0:
            raise ValueError("frequencies must be a non-empty array/sequence")
        if spectrum is None or len(spectrum) == 0:
            raise ValueError("spectrum must be a non-empty array/sequence")
        if len(frequencies) != len(spectrum):
            raise ValueError(
                f"frequencies ({len(frequencies)}) and spectrum ({len(spectrum)}) "
                "must have equal length")
        lower, upper = self.get_frequency_window()
        mask = [(f >= lower and f <= upper) for f in frequencies]
        import numpy as np

        observed_spectrum = np.asarray(spectrum)[mask]
        observed_frequencies = np.asarray(frequencies)[mask]
        return observed_frequencies, observed_spectrum

    def detect(self, observed_spectrum) -> bool:
        """(Preserved prototype API) Peak-power detection on a synthetic spectrum.

        Uses the independent ``spectrum_threshold`` (normalized power units).
        """
        import numpy as np

        arr = np.asarray(observed_spectrum)
        if arr.size == 0:
            return False
        peak_power = np.max(arr)
        return bool(peak_power >= self.spectrum_threshold)

    def perform_dwell(self, spectrum, frequencies):
        """(Preserved prototype API) observe + advance time.

        Returns ``(observed_frequencies, observed_spectrum)``.
        """
        observed_frequencies, observed_spectrum = self.observe(spectrum, frequencies)
        self.dwell()
        return observed_frequencies, observed_spectrum

    # ------------------------------------------------------ RF pulse/event mode

    @staticmethod
    def _norm_freq(frequency_mhz) -> float:
        if frequency_mhz is None or not math.isfinite(float(frequency_mhz)):
            raise ValueError(f"invalid pulse frequency {frequency_mhz!r}")
        return float(frequency_mhz)

    def _pulse_values(self, pulse) -> Tuple[float, float, float, float, float, float, Optional[int]]:
        """Extract (freq_mhz, toa_us, exit_us, width_us, amp_db, aoa_deg, pulse_id)
        from a repository pulse object without hard dependencies.

        Accepts dicts (from NDJSON) or objects with attributes (``ActivePulse`` /
        ``PulseRecord`` / a detection payload). Uses repository field names.
        """
        if isinstance(pulse, dict):
            get = pulse.get
            freq = get("frequency_mhz")
            toa = get("toa_us", get("time_us", None))
            width = get("pulse_width_us", None)
            exit_us = get("exit_us", None)
            amp = get("amplitude_db", None)
            aoa = get("aoa_deg", None)
            pid = get("pulse_id", None)
        else:
            freq = getattr(pulse, "frequency_mhz", None)
            toa = getattr(pulse, "toa_us", getattr(pulse, "time_us", None))
            width = getattr(pulse, "pulse_width_us", None)
            exit_us = getattr(pulse, "exit_us", None)
            amp = getattr(pulse, "amplitude_db", None)
            aoa = getattr(pulse, "aoa_deg", None)
            pid = getattr(pulse, "pulse_id", None)
        return (float(freq), float(toa), float(exit_us), float(width), float(amp),
                float(aoa) if aoa is not None else 0.0, pid)

    def _is_time_visible(self, toa_us, exit_us) -> bool:
        """Half-open overlap: ``toa_us <= current_time_us < exit_us``."""
        return toa_us <= self.current_time_us < exit_us

    def _is_amplitude_visible(self, amplitude_db) -> bool:
        """Pulse signal is above (>=) the configured dB sensitivity floor."""
        return amplitude_db >= self.detection_threshold_db

    def could_detect(self, pulse) -> DetectionObservation:
        """Answer: 'is this RF pulse visible to me right now?'

        Visibility requires BOTH frequency-in-IBW, time-overlap with the current
        receiver time, and amplitude above the dB threshold. Raises ``ValueError``
        if the pulse object is malformed / non-finite.
        """
        freq, toa, exit_us, width, amp, aoa, pid = self._pulse_values(pulse)
        return self._evaluate(freq, toa, exit_us, width, amp, aoa, pid)

    def _evaluate(self, freq_mhz, toa_us, exit_us, width_us, amp_db, aoa_deg,
                  pulse_id) -> DetectionObservation:
        in_window = math.isfinite(freq_mhz) and self.frequency_in_window(freq_mhz)
        time_ok = math.isfinite(toa_us) and math.isfinite(exit_us) and (
            toa_us <= exit_us) and self._is_time_visible(toa_us, exit_us)
        amp_ok = math.isfinite(amp_db) and self._is_amplitude_visible(amp_db)
        detected = in_window and time_ok and amp_ok
        return DetectionObservation(
            time_us=self.current_time_us,
            frequency_mhz=freq_mhz,
            pulse_width_us=width_us if math.isfinite(width_us) else 0.0,
            amplitude_db=amp_db,
            aoa_deg=aoa_deg if math.isfinite(aoa_deg) else 0.0,
            pulse_id=pulse_id,
            center_frequency_mhz=self.center_frequency_mhz,
            detected=detected,
        )

    def process_pulse(self, pulse) -> Optional[DetectionObservation]:
        """Evaluate one RF pulse against the receiver's *current* state.

        Returns a :class:`DetectionObservation` if the pulse is detected, else
        ``None``. Does not advance time or change frequency.
        """
        det = self.could_detect(pulse)
        return det if det.detected else None

    def process_event(self, event) -> Optional[DetectionObservation]:
        """Process a repository ``SimulationEvent`` (``entry`` type).

        ``event`` may be a :class:`~.environment.SimulationEvent`, a dict from
        ``timeline_reader.iter_events``, or any object exposing ``.pulse`` /
        ``["pulse"]``. Only ``entry`` events are evaluated (an ``exit`` event
        means the pulse is no longer present). Returns a detection or ``None``.

        This is the single-pulse *instant* evaluation: it checks the pulse
        against the current observation instant and does not advance the clock.
        Use :meth:`add_pulse` / the buffer with :meth:`scan_once` for a full
        time-aware dwell-aware scan.
        """
        if isinstance(event, dict):
            etype = event.get("event")
            pulse = event.get("pulse")
        else:
            etype = getattr(event, "event_type", getattr(event, "event", None))
            pulse = getattr(event, "pulse", None)
        if etype not in (None, "entry"):
            return None
        if pulse is None:
            return None
        det = self.could_detect(pulse)
        if not det.detected:
            return None
        # Ground-truth emitter id may be nested inside ``pulse`` (the actual
        # NDJSON structure per ActivePulse.summary_dict) or at the event root.
        # It is a label for evaluation only -- never required for detection.
        gt = (pulse.get("emitter_id")
              if isinstance(pulse, dict) and pulse.get("emitter_id") is not None
              else (event.get("emitter_id")
                    if isinstance(event, dict) else None))
        if gt is None:
            gt = (getattr(pulse, "emitter_id", None)
                  if not isinstance(pulse, dict) and hasattr(pulse, "emitter_id")
                  else None)
        if gt is not None:
            det.emitter_id = int(gt)
        return det

    # -------------------------------------------------- time-aware pulse buffer

    def add_pulse(self, pulse) -> Optional[Dict[str, Any]]:
        """Buffer a pulse the receiver has just learned about (its ``entry``).

        The pulse is stored with its full field set plus ``toa_us`` and
        ``exit_us`` so that later dwells can apply *interval* overlap. This is
        called as soon as the environment announces the pulse -- the receiver
        never learns a pulse before its entry event, so there is no future
        information leakage. Returns the normalized buffer entry, or ``None``
        for a malformed pulse (defensively ignored, never repaired).
        """
        try:
            values = self._pulse_values(pulse)
        except Exception:
            return None
        freq, toa, exit_us, width, amp, aoa, pid = values
        if not (math.isfinite(toa) and math.isfinite(exit_us)
                and math.isfinite(freq)):
            return None
        key = int(pid) if pid is not None else len(self._pulses)
        entry = {
            "frequency_mhz": freq,
            "toa_us": toa,
            "exit_us": exit_us,
            "pulse_width_us": width if math.isfinite(width) else 0.0,
            "amplitude_db": amp,
            "aoa_deg": aoa if math.isfinite(aoa) else 0.0,
            "pulse_id": pid,
            "emitter_id": (pulse.get("emitter_id")
                           if isinstance(pulse, dict) else
                           getattr(pulse, "emitter_id", None)),
        }
        self._pulses[key] = entry
        return entry

    def remove_pulse(self, pulse_id: Optional[int]) -> None:
        """Forget a pulse (e.g. on its ``exit`` event)."""
        if pulse_id is None:
            return
        self._pulses.pop(int(pulse_id), None)

    def advance(self, t_us: float) -> float:
        """Advance the receiver clock to ``t_us``, ending stale pulses.

        Pulses whose ``exit_us <= t_us`` are removed from the buffer because
        they no longer exist at (or after) the new time. Pulses that have not
        yet entered are simply absent from the buffer (they appear only when
        their ``entry`` event is announced via :meth:`add_pulse`). Returns the
        new current time.
        """
        if not math.isfinite(t_us):
            raise ValueError(f"invalid time {t_us!r}")
        self.current_time_us = float(t_us)
        stale = [k for k, p in self._pulses.items()
                 if p["exit_us"] <= self.current_time_us]
        for k in stale:
            del self._pulses[k]
        return self.current_time_us

    @staticmethod
    def _overlaps(pulse: Dict[str, Any], t0: float, t1: float) -> bool:
        """Interval overlap: ``[toa, exit)`` vs ``[t0, t1)``.

        A pulse is observable during a dwell iff its active interval overlaps
        the dwell interval: ``toa < t1 and exit > t0`` (strict on both sides so
        a pulse that has already exited at ``t0`` or has not yet started at
        ``t1`` is not counted).
        """
        return pulse["toa_us"] < t1 and pulse["exit_us"] > t0

    def _current_pulses(self) -> List[Any]:
        """Live pulse set for buffered scanning (the time-aware buffer)."""
        return list(self._pulses.values())

    def _evaluate_overlap(self, entry: Dict[str, Any],
                          t0: float, t1: float) -> Optional[DetectionObservation]:
        """Evaluate a *buffered* pulse against the dwell interval ``[t0, t1)``.

        Frequency must be in the current window (inclusive edges) AND the
        pulse's active interval must overlap the dwell interval AND its
        amplitude must clear the dB sensitivity floor. Returns a detection or
        ``None``.
        """
        freq = entry["frequency_mhz"]
        if not self.frequency_in_window(freq):
            return None
        if not self._overlaps(entry, t0, t1):
            return None
        amp = entry["amplitude_db"]
        if not (math.isfinite(amp) and amp >= self.detection_threshold_db):
            return None
        det = DetectionObservation(
            time_us=t0,
            frequency_mhz=freq,
            pulse_width_us=entry["pulse_width_us"],
            amplitude_db=amp,
            aoa_deg=entry["aoa_deg"],
            pulse_id=entry["pulse_id"],
            center_frequency_mhz=self.center_frequency_mhz,
            detected=True,
        )
        det.emitter_id = entry.get("emitter_id")  # ground-truth passthrough
        return det

    # -------------------------------------------------------------- scan logic

    def _record(self, detections: Sequence[DetectionObservation]) -> None:
        self.detections = list(detections)
        self.last_observation = ReceiverObservation(
            time_us=self.current_time_us,
            center_frequency_mhz=self.center_frequency_mhz,
            ibw_mhz=self.ibw_mhz,
            dwell_time_us=self.dwell_time_us,
            dwell_interval_us=[self.dwell_start_us, self.dwell_end_us],
            window_mhz=list(self.get_frequency_window()),
            detections=list(detections),
        )

    def scan_once(self, pulses=None) -> ReceiverObservation:
        """One static scan step (deterministic ordering):

        1. observe the current window over the dwell interval ``[t, t + dwell)``,
        2. evaluate every buffered pulse for *interval* overlap + detection,
        3. record detections,
        4. advance simulation time by ``dwell_time_us``,
        5. move center frequency by ``frequency_step_mhz``.

        ``pulses`` optionally overrides the pulse set (defaults to the
        receiver's time-aware buffer). Returns the :class:`ReceiverObservation`
        for this dwell.
        """
        t0 = self.current_time_us
        t1 = t0 + self.dwell_time_us
        self.dwell_start_us = t0
        self.dwell_end_us = t1
        if pulses is None:
            candidates = self._current_pulses()
        else:
            candidates = pulses
        detections = []
        for p in candidates:
            det = self._candidate_detection(p, t0, t1)
            if det is not None:
                detections.append(det)
        self._record(detections)
        self.current_time_us = t1
        # Any buffered pulse that had already exited by the dwell end is gone.
        self._prune(t1)
        self.step_up()
        self.scan_count += 1
        return self.last_observation

    def _candidate_detection(self, p, t0: float, t1: float) -> Optional[DetectionObservation]:
        """Evaluate one candidate (dict buffer entry OR raw pulse object) against
        a dwell interval ``[t0, t1)``. Buffer dict entries (with an ``exit_us``
        key) use *interval* overlap; raw pulse objects use the instantaneous
        evaluation at the dwell start ``t0``."""
        if isinstance(p, dict) and "exit_us" in p:
            return self._evaluate_overlap(p, t0, t1)
        det = self.could_detect(p)
        return det if (det is not None and det.detected) else None

    def _prune(self, t_us: float) -> None:
        stale = [k for k, p in self._pulses.items() if p["exit_us"] <= t_us]
        for k in stale:
            del self._pulses[k]

    def _current_pulses(self) -> List[Any]:
        """Live pulse set for buffered scanning (the time-aware buffer)."""
        return list(self._pulses.values())

    def scan(self, n_pulses: Optional[int] = None) -> List[ReceiverObservation]:
        """Run ``n_pulses`` scan steps (or until time/frequency bounds stabilize).

        ``n_pulses`` is the number of dwells to perform (deterministic).
        Returns the list of observations, one per dwell.
        """
        if n_pulses is None:
            raise ValueError("scan() requires n_pulses (an integer dwell count)")
        n = int(n_pulses)
        if n < 0:
            raise ValueError("n_pulses must be >= 0")
        observations = []
        for _ in range(n):
            observations.append(self.scan_once())
        return observations

    # --------------------------------------------------------------- observation

    def get_observation(self) -> ReceiverObservation:
        """Return the most recent :class:`ReceiverObservation` (deterministic)."""
        if self.last_observation is None:
            self._record([])
        return self.last_observation

    # -------------------------------------------------------------- action API

    def apply_action(self, action: str, value: Optional[float] = None):
        """Apply a deterministic receiver action (no ML logic).

        Supported actions:
            * ``"TUNE"``     -- tune to ``value`` MHz (clipped). ``value`` required.
            * ``"STEP_UP"``  -- step center frequency up.
            * ``"STEP_DOWN"``-- step center frequency down.
            * ``"DWELL"``    -- advance time by one dwell, return an observation
                                over the current buffered pulses.

        Returns the observation (for DWELL) or the resulting center frequency.
        """
        a = str(action).upper()
        if a == ACTION_TUNE:
            if value is None:
                raise ValueError("TUNE requires a frequency value (MHz)")
            return self.tune(float(value))
        if a == ACTION_STEP_UP:
            return self.step_up()
        if a == ACTION_STEP_DOWN:
            return self.step_down()
        if a == ACTION_DWELL:
            return self.scan_once()
        raise ValueError(f"unknown action {action!r}")
