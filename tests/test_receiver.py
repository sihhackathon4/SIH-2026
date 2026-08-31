"""Unit tests for the SieveReceiver receiver component.

Uses the standard-library unittest. Run from the repo root:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_ENV_DIR = REPO_ROOT / "SIMULATION ENV"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SIM_ENV_DIR))

import numpy as np  # noqa: E402

from sim_env import SieveReceiver, ReceiverConfigError, to_hz, to_ghz  # noqa: E402
from sim_env.receiver.sieve_receiver import ACTION_TUNE  # noqa: E402

# Repository-realistic defaults: 18 GHz band, 1 GHz IBW, 500 MHz step.
# (detection_threshold_db is overridden per-test below; default is -140 dB.)
RX_KW = {
    "total_bandwidth": 18e3,   # MHz (18 GHz)
    "ibw": 1e3,                # MHz (1 GHz)
    "frequency_step": 500.0,   # MHz (500 MHz)
    "dwell_time": 100.0,       # us
}


def pulse(freq_mhz, toa_us, width_us, amp_db=-100.0, aoa_deg=45.0, pulse_id=None):
    """Build a repository-style pulse dict (like an NDJSON entry's pulse)."""
    return {
        "frequency_mhz": float(freq_mhz),
        "toa_us": float(toa_us),
        "pulse_width_us": float(width_us),
        "exit_us": float(toa_us) + float(width_us),
        "amplitude_db": float(amp_db),
        "aoa_deg": float(aoa_deg),
        "pulse_id": pulse_id,
    }


class TestInitialization(unittest.TestCase):
    def test_center_frequency_valid(self):
        r = SieveReceiver(**RX_KW)
        self.assertEqual(r.center_frequency_mhz, r.legal_center_min_mhz)
        self.assertEqual(r.center_frequency_mhz, 500.0)

    def test_current_time_zero(self):
        r = SieveReceiver(**RX_KW)
        self.assertEqual(r.current_time_us, 0.0)

    def test_ibw_bounds(self):
        r = SieveReceiver(**RX_KW)
        self.assertEqual(r.legal_center_min_mhz, 500.0)
        self.assertEqual(r.legal_center_max_mhz, 17500.0)


class TestConfigValidation(unittest.TestCase):
    def test_negative_bandwidth(self):
        with self.assertRaises(ReceiverConfigError):
            SieveReceiver(total_bandwidth=-1, ibw=1e3, frequency_step=500, dwell_time=100)

    def test_zero_ibw(self):
        with self.assertRaises(ReceiverConfigError):
            SieveReceiver(total_bandwidth=18e3, ibw=0, frequency_step=500, dwell_time=100)

    def test_ibw_exceeds_bandwidth(self):
        with self.assertRaises(ReceiverConfigError):
            SieveReceiver(total_bandwidth=1e3, ibw=2e3, frequency_step=500, dwell_time=100)

    def test_negative_step(self):
        with self.assertRaises(ReceiverConfigError):
            SieveReceiver(total_bandwidth=18e3, ibw=1e3, frequency_step=-500, dwell_time=100)

    def test_negative_dwell(self):
        with self.assertRaises(ReceiverConfigError):
            SieveReceiver(total_bandwidth=18e3, ibw=1e3, frequency_step=500, dwell_time=-1)

    def test_nan_config_rejected(self):
        with self.assertRaises(ReceiverConfigError):
            SieveReceiver(total_bandwidth=float("nan"), ibw=1e3, frequency_step=500,
                          dwell_time=100)


class TestReset(unittest.TestCase):
    def test_reset_deterministic_and_clears_state(self):
        r = SieveReceiver(**RX_KW)
        r.tune(8000)
        r.dwell()  # t = 100
        r.scan_once(pulses=[pulse(8000, toa_us=100, width_us=50)])  # active at t=100
        self.assertEqual(len(r.detections), 1)
        r.reset()
        self.assertEqual(r.center_frequency_mhz, 500.0)
        self.assertEqual(r.current_time_us, 0.0)
        self.assertEqual(r.detections, [])
        self.assertIsNone(r.last_observation)


class TestTuning(unittest.TestCase):
    def test_valid_tune(self):
        r = SieveReceiver(**RX_KW)
        self.assertEqual(r.tune(8000), 8000.0)
        self.assertEqual(r.center_frequency_mhz, 8000.0)

    def test_tune_below_lower_clipped(self):
        r = SieveReceiver(**RX_KW)
        r.tune(0)
        self.assertEqual(r.center_frequency_mhz, r.legal_center_min_mhz)

    def test_tune_above_upper_clipped(self):
        r = SieveReceiver(**RX_KW)
        r.tune(1e9)
        self.assertEqual(r.center_frequency_mhz, r.legal_center_max_mhz)

    def test_tune_nan_rejected(self):
        r = SieveReceiver(**RX_KW)
        with self.assertRaises(ValueError):
            r.tune(float("nan"))


class TestFrequencyWindow(unittest.TestCase):
    def test_exact_ibw(self):
        r = SieveReceiver(**RX_KW)
        r.tune(8000)
        lower, upper = r.get_frequency_window()
        self.assertAlmostEqual(upper - lower, 1000.0)  # exactly 1 GHz IBW
        self.assertEqual(lower, 7500.0)
        self.assertEqual(upper, 8500.0)


class TestStep(unittest.TestCase):
    def test_step_up(self):
        r = SieveReceiver(**RX_KW)
        r.tune(8000)
        self.assertEqual(r.step_up(), 8500.0)
        self.assertEqual(r.step_up(), 9000.0)

    def test_step_down(self):
        r = SieveReceiver(**RX_KW)
        r.tune(8000)
        self.assertEqual(r.step_down(), 7500.0)

    def test_step_up_clips_at_max(self):
        r = SieveReceiver(**RX_KW)
        r.tune(r.legal_center_max_mhz)
        self.assertEqual(r.step_up(), r.legal_center_max_mhz)

    def test_step_down_clips_at_min(self):
        r = SieveReceiver(**RX_KW)
        r.tune(r.legal_center_min_mhz)
        self.assertEqual(r.step_down(), r.legal_center_min_mhz)


class TestDwell(unittest.TestCase):
    def test_time_advances(self):
        r = SieveReceiver(**RX_KW)
        self.assertEqual(r.current_time_us, 0.0)
        r.dwell()
        self.assertEqual(r.current_time_us, 100.0)
        r.dwell()
        self.assertEqual(r.current_time_us, 200.0)


class TestUnitConsistency(unittest.TestCase):
    def test_mhz_to_hz_and_ghz(self):
        self.assertEqual(to_hz(3199.19), 3199190000.0)
        self.assertEqual(to_ghz(3199.19), 3.19919)

    def test_receiver_uses_mhz_not_hz(self):
        # Tuning to 8 GHz in MHz = 8000, window around it is 7500..8500 MHz.
        r = SieveReceiver(**RX_KW)
        r.tune(8000)
        lower, upper = r.get_frequency_window()
        # If units were wrongly Hz (deltas ~1e9 apart), this would fail.
        self.assertTrue(7000 <= lower <= 8000 <= upper <= 9000)


class TestFrequencyVisibility(unittest.TestCase):
    def test_pulse_inside_ibw_candidate(self):
        r = SieveReceiver(**RX_KW)
        r.tune(8000)
        _ = r.dwell()  # current_time_us == 100
        det = r.process_pulse(pulse(7990, toa_us=100, width_us=50))
        self.assertIsNotNone(det)
        self.assertTrue(det.detected)

    def test_pulse_outside_ibw_not_visible(self):
        r = SieveReceiver(**RX_KW)
        r.tune(8000)
        _ = r.dwell()
        self.assertIsNone(r.process_pulse(pulse(12000, toa_us=100, width_us=50)))

    def test_pulse_on_window_boundary_visible(self):
        r = SieveReceiver(**RX_KW)
        r.tune(8000)
        _ = r.dwell()
        self.assertIsNotNone(r.process_pulse(pulse(7500, toa_us=100, width_us=50)))
        self.assertIsNotNone(r.process_pulse(pulse(8500, toa_us=100, width_us=50)))


class TestPulseTiming(unittest.TestCase):
    def _rx_at(self, t_us):
        r = SieveReceiver(**RX_KW)
        r.tune(8000)
        while r.current_time_us < t_us:
            r.dwell()
        return r

    def test_pulse_before_dwell_not_detected(self):
        # Pulse fully before receiver arrives (toa+width < current time).
        r = self._rx_at(200)
        self.assertIsNone(r.process_pulse(pulse(8000, toa_us=50, width_us=25)))

    def test_pulse_active_during_dwell_detected(self):
        r = self._rx_at(100)
        det = r.process_pulse(pulse(8000, toa_us=100, width_us=200))  # 100..300
        self.assertIsNotNone(det)

    def test_pulse_begins_during_dwell_detected_at_its_toa(self):
        r = self._rx_at(150)
        det = r.process_pulse(pulse(8000, toa_us=150, width_us=100))  # begins at 150
        self.assertIsNotNone(det)

    def test_pulse_ends_during_dwell_not_detected_after_end(self):
        # Pulse active 100..140; receiver samples at 150 -> not present.
        r = self._rx_at(150)
        self.assertIsNone(r.process_pulse(pulse(8000, toa_us=100, width_us=40)))

    def test_pulse_ending_exactly_at_sample_not_detected_half_open(self):
        r = self._rx_at(100)
        # exit_us = 100 exactly -> not visible under half-open [toa, exit)
        self.assertIsNone(r.process_pulse(pulse(8000, toa_us=50, width_us=50)))


class TestDetectionThreshold(unittest.TestCase):
    def test_amplitude_below_threshold_not_detected(self):
        r = SieveReceiver(**RX_KW, detection_threshold_db=-90.0)
        r.tune(8000)
        _ = r.dwell()
        # -100 < -90 -> below sensitivity floor -> not detected
        self.assertIsNone(r.process_pulse(pulse(8000, toa_us=100, width_us=50, amp_db=-100.0)))

    def test_amplitude_at_threshold_detected(self):
        r = SieveReceiver(**RX_KW, detection_threshold_db=-90.0)
        r.tune(8000)
        _ = r.dwell()
        det = r.process_pulse(pulse(8000, toa_us=100, width_us=50, amp_db=-90.0))
        self.assertIsNotNone(det)

    def test_amplitude_above_threshold_detected(self):
        r = SieveReceiver(**RX_KW, detection_threshold_db=-90.0)
        r.tune(8000)
        _ = r.dwell()
        det = r.process_pulse(pulse(8000, toa_us=100, width_us=50, amp_db=-80.0))
        self.assertIsNotNone(det)


class TestMultipleSimultaneousPulses(unittest.TestCase):
    def test_equal_toa_kept_separate(self):
        r = SieveReceiver(**RX_KW)
        r.tune(8000)
        _ = r.dwell()
        a = pulse(7900, toa_us=100, width_us=50, pulse_id=1)
        b = pulse(8100, toa_us=100, width_us=50, pulse_id=2)  # same ToA, different freq/id
        det_a = r.process_pulse(a)
        det_b = r.process_pulse(b)
        self.assertIsNotNone(det_a)
        self.assertIsNotNone(det_b)
        self.assertNotEqual(det_a.pulse_id, det_b.pulse_id)
        self.assertNotEqual(det_a.frequency_mhz, det_b.frequency_mhz)


class TestSyntheticSpectrum(unittest.TestCase):
    def test_observe_limits_to_ibw(self):
        r = SieveReceiver(**RX_KW)
        r.tune(8000)
        freqs = np.linspace(7000, 9000, 5)  # 7500..8500 only in window
        spectrum = np.ones(5)
        of, os_ = r.observe(spectrum, freqs)
        self.assertTrue(np.all(of >= 7500 - 1e-6))
        self.assertTrue(np.all(of <= 8500 + 1e-6))

    def test_detect_peak_above_threshold(self):
        r = SieveReceiver(spectrum_threshold=5.0, **RX_KW)
        self.assertTrue(r.detect([0.0, 10.0, 3.0]))
        self.assertFalse(r.detect([0.0, 1.0, 3.0]))
        self.assertFalse(r.detect([]))

    def test_perform_dwell_advances_time(self):
        r = SieveReceiver(**RX_KW)
        r.tune(8000)
        freqs = np.linspace(7500, 8500, 3)
        r.perform_dwell(np.ones(3), freqs)
        self.assertEqual(r.current_time_us, 100.0)


class TestScan(unittest.TestCase):
    def test_scan_once_deterministic_order(self):
        r = SieveReceiver(**RX_KW)
        r.tune(8000)
        r.dwell()  # t = 100
        obs = r.scan_once(pulses=[pulse(7900, toa_us=100, width_us=50)])
        # ordering: observe(window 7500-8500 at t=100) -> detect -> dwell (t=200)
        #        -> step up (center 8500)
        self.assertEqual(obs.time_us, 100.0)
        self.assertEqual(obs.center_frequency_mhz, 8000.0)
        self.assertEqual(len(obs.detections), 1)
        self.assertEqual(r.current_time_us, 200.0)
        self.assertEqual(r.center_frequency_mhz, 8500.0)

    def test_scan_steps(self):
        r = SieveReceiver(**RX_KW)
        r.tune(8000)
        obs = r.scan(2, )
        self.assertEqual(len(obs), 2)
        # after 2 dwells time = 200, center stepped up twice to 9000
        self.assertEqual(r.current_time_us, 200.0)
        self.assertEqual(r.center_frequency_mhz, 9000.0)


class TestActionAPI(unittest.TestCase):
    def test_tune_action(self):
        r = SieveReceiver(**RX_KW)
        r.apply_action("TUNE", 8000)
        self.assertEqual(r.center_frequency_mhz, 8000.0)

    def test_step_actions(self):
        r = SieveReceiver(**RX_KW)
        r.tune(8000)
        r.apply_action("STEP_UP")
        self.assertEqual(r.center_frequency_mhz, 8500.0)
        r.apply_action("STEP_DOWN")
        self.assertEqual(r.center_frequency_mhz, 8000.0)

    def test_unknown_action(self):
        r = SieveReceiver(**RX_KW)
        with self.assertRaises(ValueError):
            r.apply_action("FLY")


class TestSyntheticDemonstration(unittest.TestCase):
    """Mirrors the classic 'Test_Environment.py' demonstration: an 18 GHz
    spectrum with 3 artificial signals, receiver tuned to 8 GHz with a 1 GHz
    IBW, but with hard assertions (not just a plot)."""

    def _make_environment(self):
        rng = np.random.default_rng(42)
        n = 18001
        freqs = np.linspace(0, 18e3, n)  # 0..18 GHz in MHz
        spectrum = 0.1 * rng.standard_normal(n)
        # 3 deterministic artificial signals within the 8 GHz observation band.
        for center in (7900.0, 8000.0, 8100.0):
            idx = int(np.argmin(np.abs(freqs - center)))
            spectrum[idx] = 12.0
        return rng, freqs, spectrum

    def test_18ghz_demo_tuning_window_detection(self):
        rng, freqs, spectrum = self._make_environment()
        r = SieveReceiver(**RX_KW)          # 18 GHz total, 1 GHz IBW
        r.tune(8000.0)                       # tune to 8 GHz
        lower, upper = r.get_frequency_window()
        self.assertEqual(lower, 7500.0)
        self.assertEqual(upper, 8500.0)      # exactly 1 GHz IBW

        observed_freqs, observed_spec = r.observe(spectrum, freqs)
        # Only in-band frequencies survive the IBW window.
        self.assertTrue(np.all(observed_freqs >= 7500.0 - 1e-6))
        self.assertTrue(np.all(observed_freqs <= 8500.0 + 1e-6))
        self.assertTrue(r.detect(observed_spec))   # signals are above threshold

    def test_18ghz_demo_no_detection_outside_ibw(self):
        rng, freqs, spectrum = self._make_environment()
        r = SieveReceiver(**RX_KW)
        r.tune(5000.0)                        # 5 GHz, away from the 8 GHz signals
        observed_freqs, observed_spec = r.observe(spectrum, freqs)
        # Tuned window (4500..5500 MHz) contains no strong signal.
        self.assertFalse(r.detect(observed_spec))

    def test_18ghz_demo_dwell_timing(self):
        rng, freqs, spectrum = self._make_environment()
        r = SieveReceiver(**RX_KW)             # dwell_time=100 us from RX_KW
        r.tune(8000.0)
        self.assertEqual(r.current_time_us, 0.0)
        r.perform_dwell(spectrum, freqs)
        self.assertEqual(r.current_time_us, 100.0)


class TestDefensiveInput(unittest.TestCase):
    def _rx(self, t_us=100):
        r = SieveReceiver(**RX_KW)
        r.tune(8000.0)
        r.current_time_us = t_us
        return r

    def test_nan_frequency_not_detected(self):
        r = self._rx()
        self.assertIsNone(r.process_pulse(pulse(float("nan"), 100, 50)))

    def test_inf_frequency_not_detected(self):
        r = self._rx()
        self.assertIsNone(r.process_pulse(pulse(float("inf"), 100, 50)))

    def test_nan_width_not_detected(self):
        r = self._rx()
        self.assertIsNone(r.process_pulse(pulse(8000, 100, float("nan"))))

    def test_nonpositive_width_not_detected(self):
        r = self._rx()
        self.assertIsNone(r.process_pulse(pulse(8000, 100, 0.0)))
        self.assertIsNone(r.process_pulse(pulse(8000, 100, -5.0)))

    def test_nan_toa_not_detected(self):
        r = self._rx()
        self.assertIsNone(r.process_pulse(pulse(8000, float("nan"), 50)))

    def test_negative_toa_not_detected(self):
        r = self._rx()
        # pulse ended long before receiver time 100 -> not visible
        self.assertIsNone(r.process_pulse(pulse(8000, -50.0, 10.0)))


if __name__ == "__main__":
    unittest.main()
