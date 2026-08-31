"""Audit-focused receiver integration tests.

Covers the five integration audit items with explicit assertions:

1. RadioEnvironment -> SieveReceiver *live* connection (no manual per-pulse
   retune / time-setting inside an event loop).
2. Time-aware pulse buffer: pulse-before/begins/ends/after-dwell handling via
   interval overlap.
3. Independent static scanning: the receiver keeps its own clock/frequency and
   never retunes to an arriving pulse.
4. Nested emitter-ID / ground-truth handling from the real NDJSON structure;
   emitter ID is never required for detection.
5. Real validated RF data, end-to-end, through the repository pipeline.

Run from the repo root:
    python -m pytest tests/test_receiver_audit.py -q
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_ENV_DIR = REPO_ROOT / "SIMULATION ENV"
OUTPUT_DIR = REPO_ROOT / "OUTPUT FILES"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SIM_ENV_DIR))

from sim_env import (  # noqa: E402
    SieveReceiver,
    RadioEnvironment,
    SimConfig,
    FileRecordSource,
    RadioReceiverBridge,
    attach_receiver,
    ReceiverObservation,
)
from sim_env.environment import SimulationEvent, ActivePulse  # noqa: E402

VALIDATED_FILE = OUTPUT_DIR / "output_134.txt"


def _rx(**over) -> SieveReceiver:
    kw = {
        "total_bandwidth": 18000.0,
        "ibw": 1000.0,
        "frequency_step": 500.0,
        "dwell_time": 100.0,
        "detection_threshold_db": -140.0,
    }
    kw.update(over)
    return SieveReceiver(**kw)


def _entry_event(toa, freq, pw, amp=-120.0, pid=0, emitter=None):
    """Build a real environment ``SimulationEvent`` entry (as the env emits)."""
    ap = ActivePulse(
        pulse_id=pid, toa_us=toa, frequency_mhz=freq, pulse_width_us=pw,
        amplitude_db=amp, aoa_deg=90.0,
        emitter_id=emitter if emitter is not None else 0,
        exit_us=toa + pw, source_id="t",
    )
    return SimulationEvent(event_type="entry", time_us=toa, pulse_id=pid, pulse=ap,
                           active_count=1)


# ---------------------------------------------------------------------------
# Item 1 & 5 -- LIVE RadioEnvironment -> SieveReceiver via the bridge.
# ---------------------------------------------------------------------------
@unittest.skipUnless(VALIDATED_FILE.exists(), "validated output file not present")
class TestLiveEnvironmentToReceiver(unittest.TestCase):
    def test_bridge_consumes_live_events_without_manual_retune(self):
        cfg = SimConfig(inputs=[VALIDATED_FILE], snapshot_interval_us=None)
        source = FileRecordSource([VALIDATED_FILE], on_nonfinite="drop")
        r = _rx()
        r.tune(3200.0)              # one-time configuration, not per-pulse retune
        env = RadioEnvironment(source, cfg)
        bridge = attach_receiver(env, r)
        self.assertIsInstance(bridge, RadioReceiverBridge)
        self.assertIn(bridge.on_event, env._callbacks)  # first-class consumer

        # Drive the environment step-by-step; the bridge feeds entries/exits
        # into the buffer with NO manual receiver.tune()/time-setting per pulse.
        # (entry steps return None from step(), so loop on done rather than None.)
        max_buffer = 0
        while not env.done:
            env.step()
            max_buffer = max(max_buffer, len(r._pulses))
        # The buffer was populated during the run (live flow) and empties once
        # every pulse exits (no stale leakage) -> honest end state.
        self.assertGreater(max_buffer, 0)
        self.assertEqual(len(r._pulses), 0)

    def test_live_stream_detects_eligible_real_pulse(self):
        cfg = SimConfig(inputs=[VALIDATED_FILE], snapshot_interval_us=None)
        source = FileRecordSource([VALIDATED_FILE], on_nonfinite="drop")
        r = _rx()
        r.tune(3200.0)
        env = RadioEnvironment(source, cfg)
        attach_receiver(env, r)

        # Stop as soon as the environment has announced a live entry pulse.
        while not env.done:
            env.step()
            if len(r._pulses) > 0:
                break
        self.assertGreater(len(r._pulses), 0)

        # The receiver then dwells on its OWN window over the pulse's active
        # interval (a single advance, not a per-pulse retune). Every buffered,
        # in-band, overlapping real pulse is reported with structured fields.
        r.advance(min(p["toa_us"] for p in r._pulses.values()) - 1.0)
        dets = r.scan_once().detections
        self.assertIsInstance(dets, list)
        self.assertGreaterEqual(len(dets), 1)
        for d in dets:
            self.assertTrue(d.detected)
            self.assertIn("frequency_mhz", d.to_dict())
            self.assertIn("pulse_id", d.to_dict())


# ---------------------------------------------------------------------------
# Item 2 -- real time-aware pulse buffer / interval overlap timing.
# ---------------------------------------------------------------------------
class TestPulseTiming(unittest.TestCase):
    def _scan_at(self, rx, t0, toa, exit_us, freq, amp=-120.0):
        """Advance to dwell start, add pulse (as it becomes known), scan once."""
        rx.advance(t0)
        rx.tune(rx.center_frequency_mhz)     # keep current center
        rx.add_pulse({"frequency_mhz": freq, "toa_us": toa, "exit_us": exit_us,
                      "pulse_width_us": exit_us - toa, "amplitude_db": amp,
                      "aoa_deg": 90.0, "pulse_id": 0, "emitter_id": 7})
        obs = rx.scan_once()
        return obs.detections

    def test_pulse_completely_before_dwell_not_detected(self):
        r = _rx()
        r.tune(3000.0)               # window [2500, 3500]
        # dwell [200,300), pulse active [50,60): already expired before dwell.
        dets = self._scan_at(r, 200.0, 50.0, 60.0, 3000.0)
        self.assertEqual(len(dets), 0)

    def test_pulse_begins_before_dwell_and_stays_active_detected(self):
        r = _rx()
        r.tune(3000.0)
        # dwell [200,300), pulse active [150,250): overlaps throughout [200,250).
        dets = self._scan_at(r, 200.0, 150.0, 250.0, 3000.0)
        self.assertEqual(len(dets), 1)

    def test_pulse_begins_during_dwell_detected(self):
        r = _rx()
        r.tune(3000.0)
        # dwell [200,300), pulse active [220,230): starts during dwell.
        dets = self._scan_at(r, 200.0, 220.0, 230.0, 3000.0)
        self.assertEqual(len(dets), 1)

    def test_pulse_ends_during_dwell_detected(self):
        r = _rx()
        r.tune(3000.0)
        # dwell [200,300), pulse active [180,240): ends during dwell.
        dets = self._scan_at(r, 200.0, 180.0, 240.0, 3000.0)
        self.assertEqual(len(dets), 1)

    def test_pulse_completely_after_dwell_not_detected(self):
        r = _rx()
        r.tune(3000.0)
        # dwell [200,300), pulse active [350,355): starts after dwell ends.
        dets = self._scan_at(r, 200.0, 350.0, 355.0, 3000.0)
        self.assertEqual(len(dets), 0)

    def test_pulse_outside_frequency_window_not_detected(self):
        r = _rx()
        r.tune(3000.0)               # window [2500, 3500]
        dets = self._scan_at(r, 200.0, 210.0, 220.0, 5000.0)  # 5 GHz -> outside
        self.assertEqual(len(dets), 0)


# ---------------------------------------------------------------------------
# Item 3 -- independent static scanning (no per-pulse retune).
# ---------------------------------------------------------------------------
class TestIndependentStaticScan(unittest.TestCase):
    def test_scan_progresses_and_detects_only_on_overlap(self):
        r = _rx()
        # receiver starts at legal center 500, window [0,1000], at t=0.
        # Pulses arrive independently (announced at their ToA => buffered only
        # once their entry time has passed -> no future leakage).
        arrivals = [
            (50.0, 300.0, 55.0),     # P1: 300 MHz, active [50,55)
            (120.0, 1200.0, 125.0),  # P2: 1200 MHz, active [120,125)
            (175.0, 6000.0, 178.0),  # P3: 6000 MHz (never in any window)
            (280.0, 1300.0, 290.0),  # P4: 1300 MHz, active [280,290)
        ]
        # Buffer announcements up front (as the env would announce each entry).
        for toa, freq, exit_us in arrivals:
            r.add_pulse({"frequency_mhz": freq, "toa_us": toa, "exit_us": exit_us,
                         "pulse_width_us": exit_us - toa, "amplitude_db": -120.0,
                         "aoa_deg": 90.0, "pulse_id": int(toa), "emitter_id": 1})

        centers_seen = []
        for step in range(4):
            obs = r.scan_once()
            centers_seen.append(r.center_frequency_mhz)

        # Receiver scanned deterministically with its own clock, never retuning
        # to a pulse's frequency.
        self.assertEqual(centers_seen, [1000.0, 1500.0, 2000.0, 2500.0])

        # Dwell windows visited:
        # [0,100) c500 w[0,1000]   -> P1 (300) detected
        # [100,200) c1000 w[500,1500] -> P2 (1200) detected; P3 (6000) not
        # [200,300) c1500 w[1000,2000] -> P4 (1300) detected
        # [300,400) c2000 -> P3 (6000) outside; P2 ended; none
        def det_freqs(obs):
            return sorted(d.frequency_mhz for d in obs.detections)

        r2 = _rx()
        r2.add_pulse({"frequency_mhz": 300.0, "toa_us": 50.0, "exit_us": 55.0,
                      "pulse_width_us": 5.0, "amplitude_db": -120.0,
                      "aoa_deg": 90.0, "pulse_id": 1, "emitter_id": 1})
        o0 = r2.scan_once()                       # [0,100) window [0,1000]
        self.assertEqual(det_freqs(o0), [300.0])
        r2.add_pulse({"frequency_mhz": 1200.0, "toa_us": 120.0, "exit_us": 125.0,
                      "pulse_width_us": 5.0, "amplitude_db": -120.0,
                      "aoa_deg": 90.0, "pulse_id": 2, "emitter_id": 2})
        r2.add_pulse({"frequency_mhz": 6000.0, "toa_us": 175.0, "exit_us": 178.0,
                      "pulse_width_us": 3.0, "amplitude_db": -118.0,
                      "aoa_deg": 90.0, "pulse_id": 3, "emitter_id": 3})
        o1 = r2.scan_once()                       # [100,200) window [500,1500]
        self.assertEqual(det_freqs(o1), [1200.0])  # 6000 out-of-band ignored
        r2.add_pulse({"frequency_mhz": 1300.0, "toa_us": 280.0, "exit_us": 290.0,
                      "pulse_width_us": 10.0, "amplitude_db": -122.0,
                      "aoa_deg": 90.0, "pulse_id": 4, "emitter_id": 4})
        o2 = r2.scan_once()                       # [200,300) window [1000,2000]
        self.assertEqual(det_freqs(o2), [1300.0])


# ---------------------------------------------------------------------------
# Item 4 -- nested emitter-id / ground truth; not required for detection.
# ---------------------------------------------------------------------------
class TestEmitterGroundTruth(unittest.TestCase):
    def test_detection_does_not_require_emitter_id(self):
        r = _rx()
        r.tune(3000.0)
        r.advance(200.0)
        # A pulse with NO emitter id at all.
        r.add_pulse({"frequency_mhz": 3000.0, "toa_us": 210.0, "exit_us": 220.0,
                     "pulse_width_us": 10.0, "amplitude_db": -120.0,
                     "aoa_deg": 90.0, "pulse_id": 0})
        dets = r.scan_once().detections
        self.assertEqual(len(dets), 1)         # detected without emitter id

    def test_nested_ndjson_emitter_id_recovered_for_evaluation(self):
        # Real NDJSON nests emitter_id inside the "pulse" object
        # (ActivePulse.summary_dict -> pulse dict).
        r = _rx()
        r.tune(3000.0)
        r.advance(210.0)                   # sample point inside the pulse [210,220)
        ndjson_entry = {
            "event": "entry",
            "time_us": 210.0,
            "pulse": {
                "toa_us": 210.0, "frequency_mhz": 3000.0,
                "pulse_width_us": 10.0, "amplitude_db": -120.0,
                "aoa_deg": 90.0, "emitter_id": 42,   # nested ground truth
                "exit_us": 220.0, "pulse_id": 9,
            },
        }
        det = r.process_event(ndjson_entry)
        self.assertIsNotNone(det)
        self.assertTrue(det.detected)
        self.assertEqual(det.emitter_id, 42)   # ground truth preserved separately

    def test_root_emitter_id_also_recovered(self):
        r = _rx()
        r.tune(3000.0)
        r.advance(210.0)
        ev = _entry_event(210.0, 3000.0, 10.0, pid=3, emitter=7)
        det = r.process_event(ev)
        self.assertIsNotNone(det)
        self.assertEqual(det.emitter_id, 7)

    def test_emitter_id_carried_through_buffer_scan(self):
        r = _rx()
        r.tune(3000.0)
        r.advance(200.0)
        r.add_pulse({"frequency_mhz": 3000.0, "toa_us": 210.0, "exit_us": 220.0,
                     "pulse_width_us": 10.0, "amplitude_db": -120.0,
                     "aoa_deg": 90.0, "pulse_id": 5, "emitter_id": 99})
        det = r.scan_once().detections[0]
        self.assertEqual(det.emitter_id, 99)


# ---------------------------------------------------------------------------
# No future-information leakage.
# ---------------------------------------------------------------------------
class TestNoFutureLeakage(unittest.TestCase):
    def test_receiver_does_not_learn_future_pulses(self):
        r = _rx()
        r.tune(3000.0)
        r.advance(200.0)             # dwell starts at 200
        # A pulse that has NOT yet arrived (starts at 350, after the dwell).
        r.add_pulse({"frequency_mhz": 3000.0, "toa_us": 350.0, "exit_us": 355.0,
                     "pulse_width_us": 5.0, "amplitude_db": -120.0,
                     "aoa_deg": 90.0, "pulse_id": 8, "emitter_id": 8})
        dets = r.scan_once().detections   # dwell [200,300)
        self.assertEqual(len(dets), 0)     # not observable during this dwell

    def test_static_scanner_does_not_retune_to_pulse(self):
        r = _rx()
        # Pulse far from initial center; scanner must NOT snap to its frequency.
        r.add_pulse({"frequency_mhz": 12000.0, "toa_us": 0.0, "exit_us": 10.0,
                     "pulse_width_us": 10.0, "amplitude_db": -120.0,
                     "aoa_deg": 90.0, "pulse_id": 1, "emitter_id": 1})
        c_before = r.center_frequency_mhz
        r.scan_once()
        self.assertEqual(r.center_frequency_mhz, c_before + r.frequency_step_mhz)


# ---------------------------------------------------------------------------
# Pulse buffer state: entry adds, exit removes, reset clears, buffer correct.
# ---------------------------------------------------------------------------
class TestPulseBufferState(unittest.TestCase):
    def test_entry_adds_and_exit_removes_pulse(self):
        r = _rx()
        r.add_pulse({"frequency_mhz": 3000.0, "toa_us": 100.0, "exit_us": 200.0,
                     "pulse_width_us": 100.0, "amplitude_db": -120.0,
                     "aoa_deg": 90.0, "pulse_id": 1, "emitter_id": 5})
        self.assertIn(1, r._pulses)                 # entry -> buffer
        r.remove_pulse(1)                            # exit -> removed
        self.assertNotIn(1, r._pulses)

    def test_active_buffer_reflects_lifetime(self):
        r = _rx()
        r.add_pulse({"frequency_mhz": 3000.0, "toa_us": 100.0, "exit_us": 200.0,
                     "pulse_width_us": 100.0, "amplitude_db": -120.0,
                     "aoa_deg": 90.0, "pulse_id": 1, "emitter_id": 5})
        r.add_pulse({"frequency_mhz": 3100.0, "toa_us": 100.0, "exit_us": 400.0,
                     "pulse_width_us": 300.0, "amplitude_db": -110.0,
                     "aoa_deg": 90.0, "pulse_id": 2, "emitter_id": 6})
        # Both active from toa 100; after advancing past pulse 1's exit it is gone.
        r.advance(150.0)
        self.assertEqual(set(r._pulses.keys()), {1, 2})
        r.advance(250.0)                             # pulse 1 exit (200) passed
        self.assertEqual(set(r._pulses.keys()), {2})

    def test_reset_clears_pulse_state(self):
        r = _rx()
        self.assertEqual(len(r._pulses), 0)
        r.add_pulse({"frequency_mhz": 3000.0, "toa_us": 100.0, "exit_us": 200.0,
                     "pulse_width_us": 100.0, "amplitude_db": -120.0,
                     "aoa_deg": 90.0, "pulse_id": 3, "emitter_id": 1})
        self.assertEqual(len(r._pulses), 1)
        r.reset()
        self.assertEqual(len(r._pulses), 0)          # no stale pulses remain
        self.assertEqual(r.dwell_start_us, 0.0)
        self.assertEqual(r.dwell_end_us, 0.0)

    def test_equal_toa_pulses_remain_separate(self):
        r = _rx()
        r.tune(3200.0)               # window [2700, 3700]
        r.advance(100.0)
        r.add_pulse({"frequency_mhz": 3200.0, "toa_us": 100.0, "exit_us": 130.0,
                     "pulse_width_us": 30.0, "amplitude_db": -120.0,
                     "aoa_deg": 90.0, "pulse_id": 1, "emitter_id": 1})
        r.add_pulse({"frequency_mhz": 3300.0, "toa_us": 100.0, "exit_us": 130.0,
                     "pulse_width_us": 30.0, "amplitude_db": -115.0,
                     "aoa_deg": 90.0, "pulse_id": 2, "emitter_id": 2})
        dets = r.scan_once().detections              # dwell [100, 200)
        freqs = sorted(d.frequency_mhz for d in dets)
        self.assertEqual(freqs, [3200.0, 3300.0])    # two separate observations

    def test_observation_exposes_dwell_interval(self):
        r = _rx()
        r.advance(200.0)
        obs = r.scan_once()
        self.assertEqual(obs.dwell_interval_us, [200.0, 300.0])
        self.assertIn("dwell_interval_us", obs.to_dict())
        self.assertEqual(r.dwell_start_us, 200.0)
        self.assertEqual(r.dwell_end_us, 300.0)


# ---------------------------------------------------------------------------
# CRITICAL end-to-end: validated data -> env -> event callback -> adapter ->
# receiver buffer -> static scan -> observation. Detection depends only on what
# the receiver can physically observe (no per-pulse retune, no clock jump to a
# future pulse ToA). Frequencies are MHz, times are microseconds.
# ---------------------------------------------------------------------------
@unittest.skipUnless(VALIDATED_FILE.exists(), "validated output file not present")
class TestCriticalEndToEnd(unittest.TestCase):
    def _write_controlled_file(self):
        import tempfile
        lines = [
            "record_1: data=[100.0, 3200.0, 30.0, -120.0, 90.0], label=1",
            "record_2: data=[100.0, 3300.0, 30.0, -115.0, 90.0], label=2",
            "record_3: data=[120.0, 3500.0, 30.0, -118.0, 90.0], label=3",
            "record_4: data=[175.0, 9000.0, 30.0, -116.0, 90.0], label=4",
            "record_5: data=[280.0, 3400.0, 30.0, -122.0, 90.0], label=5",
        ]
        d = Path(tempfile.mkdtemp())
        f = d / "controlled.txt"
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return f

    def test_environment_to_receiver_full_chain(self):
        f = self._write_controlled_file()
        # Broad IBW centred on the band, tuned ONCE (not per pulse); the step
        # drags the window far away after the first dwell so only the pulses
        # observable in that first dwell can ever be seen.
        rx = SieveReceiver(total_bandwidth=18000.0, ibw=4000.0,
                           frequency_step=100000.0, dwell_time=100.0,
                           detection_threshold_db=-140.0)
        rx.tune(3200.0)                 # one-time configuration
        env = RadioEnvironment(
            FileRecordSource([str(f)], on_nonfinite="drop"),
            SimConfig(inputs=[str(f)], snapshot_interval_us=None))
        attach_receiver(env, rx)

        observations = []
        max_buffer = 0
        while not env.done:
            env.step()                   # env drives the simulated clock
            max_buffer = max(max_buffer, len(rx._pulses))
            if len(rx._pulses) > 0:      # receiver dwells on its own schedule
                observations.append(rx.scan_once())

        # Live event path populated then drained the buffer (no stale state).
        self.assertGreater(max_buffer, 0)
        self.assertEqual(len(rx._pulses), 0)

        # Only the first dwell overlaps the two simultaneous in-band pulses:
        # P1/P2 (3200 & 3300, toa 100, active [100,130)) are BOTH detected and
        # kept separate. P3 (3500) arrives just after the receiver's first
        # dwell moved on; P4 (9000) is out of band; P5 (3400) falls in a window
        # the receiver has already stepped away from.
        det_freqs = []
        for obs in observations:
            det_freqs.extend(sorted(d.frequency_mhz for d in obs.detections))
        self.assertEqual(det_freqs, [3200.0, 3300.0])

        # Structured, deterministic ReceiverObservation with explicit dwell
        # interval and frequency window is produced.
        self.assertTrue(observations)
        first = observations[0]
        self.assertIsInstance(first, ReceiverObservation)
        self.assertEqual(len(first.dwell_interval_us), 2)
        self.assertEqual(len(first.window_mhz), 2)
        # Ground-truth emitter ids are carried separately on detections.
        self.assertEqual(sorted(d.emitter_id for d in first.detections),
                         [1, 2])


if __name__ == "__main__":
    unittest.main()
