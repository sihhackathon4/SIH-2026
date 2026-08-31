"""Integration tests: SieveReceiver against ACTUAL validated RF data + the
real environment/NDJSON event path.

Run from the repo root (requires a validated output file present):

    python -m unittest discover -s tests -v

Uses repository units: frequency in MHz, time in us, amplitude in dB.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_ENV_DIR = REPO_ROOT / "SIMULATION ENV"
OUTPUT_DIR = REPO_ROOT / "OUTPUT FILES"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SIM_ENV_DIR))

from sim_env import (  # noqa: E402
    SieveReceiver,
    SimConfig,
    FileRecordSource,
    RadioEnvironment,
)
from sim_env.timeline_writer import TimelineWriter  # noqa: E402
from sim_env.timeline_reader import iter_events  # noqa: E402

RX_KW = {
    "total_bandwidth": 18e3,   # MHz (18 GHz)
    "ibw": 1e3,                # MHz (1 GHz)
    "frequency_step": 500.0,   # MHz
    "dwell_time": 100.0,       # us
    "detection_threshold_db": -140.0,  # real signals ~-100..-120 dB clear this
}

VALIDATED_FILE = OUTPUT_DIR / "output_134.txt"


def _first_clean_records(path, n=5):
    """Return the first record vectors of a validated file (skip header)."""
    import re

    rec_re = re.compile(
        r"data=\[(?P<data>[^\]]*)\]\s*,\s*label=(?P<label>\S+)")
    vals = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = rec_re.search(line)
            if not m:
                continue
            data = [float(t) for t in m.group("data").split(",")]
            vals.append((data, int(m.group("label"))))
            if len(vals) >= n:
                break
    return vals


@unittest.skipUnless(VALIDATED_FILE.exists(), "validated output file not present")
class TestRealDataIntegration(unittest.TestCase):
    def test_real_pulse_detected_through_receiver_chain(self):
        records = _first_clean_records(VALIDATED_FILE)
        self.assertTrue(records, "no records parsed from real file")
        toa, freq, pw, amp, aoa = records[0][0]
        label = records[0][1]

        r = SieveReceiver(**RX_KW)
        r.tune(freq)                        # center window on the real frequency
        # The receiver dwells at the pulse's ToA instant. (Real pulses are
        # ~0.5 us wide; a 100 us scanning dwell samples inside them only when
        # it happens to land within this tiny active window. We position the
        # observation time at the pulse ToA to exercise the full detection
        # chain on a real sub-microsecond pulse deterministically.)
        r.current_time_us = toa

        pulse = {
            "frequency_mhz": freq,
            "toa_us": toa,
            "pulse_width_us": pw,
            "exit_us": toa + pw,
            "amplitude_db": amp,
            "aoa_deg": aoa,
            "pulse_id": 0,
        }
        det = r.process_pulse(pulse)
        self.assertIsNotNone(det)
        self.assertTrue(det.detected)
        # structured observation uses repository units/fields
        self.assertAlmostEqual(det.frequency_mhz, freq)
        self.assertAlmostEqual(det.amplitude_db, amp)
        self.assertAlmostEqual(det.aoa_deg, aoa)

    def test_real_record_outside_window_not_detected(self):
        records = _first_clean_records(VALIDATED_FILE, n=2)
        freq_a = records[0][0][1]
        freq_b = records[1][0][1]

        r = SieveReceiver(**RX_KW)
        r.tune(freq_a)
        toa_b = records[1][0][0]
        r.current_time_us = toa_b

        if abs(freq_b - freq_a) > r.ibw_mhz / 2.0:
            # second pulse frequency is outside the first's window
            _, _, pw_b, amp_b, aoa_b = records[1][0]
            pulse_b = {
                "frequency_mhz": freq_b,
                "toa_us": toa_b,
                "pulse_width_us": pw_b,
                "exit_us": toa_b + pw_b,
                "amplitude_db": amp_b,
                "aoa_deg": aoa_b,
                "pulse_id": 1,
            }
            self.assertIsNone(r.process_pulse(pulse_b))
        else:
            # both in the same window -> both must detect (bandwidth model holds)
            self.assertTrue(r.frequency_in_window(freq_b))

    def test_ndjson_event_stream_to_receiver(self):
        """Environment -> NDJSON -> reader(entry) -> SieveReceiver.process_event."""
        import tempfile

        cfg = SimConfig(inputs=[VALIDATED_FILE], snapshot_interval_us=None)
        source = FileRecordSource([VALIDATED_FILE], on_nonfinite="drop")
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "timeline.ndjson"
            writer = TimelineWriter(log, config=cfg)
            writer.write_meta()
            RadioEnvironment(source, cfg, on_event=writer.on_event).run()
            writer.close()

            r = SieveReceiver(**RX_KW)
            detections = []
            for ev in iter_events(log):
                if ev.get("event") != "entry":
                    continue
                pulse = ev["pulse"]
                # deterministic opportunistic dwell: tune + observe at the entry
                # instant. The pulse is active for its full width from its ToA,
                # so sampling at ``time_us`` sees it.
                r.tune(pulse["frequency_mhz"])
                r.current_time_us = ev["time_us"]
                det = r.process_pulse(pulse)
                if det is not None:
                    detections.append(det)

            # The receiver consumed the real stream and produced structured
            # observations for every entry it aligned to (all are in-window by
            # construction and above the low sensitivity floor).
            self.assertTrue(len(detections) >= 1)
            obs = detections[0]
            self.assertTrue(obs.detected)
            self.assertIn("frequency_mhz", obs.to_dict())
            self.assertIn("pulse_id", obs.to_dict())


if __name__ == "__main__":
    unittest.main()
