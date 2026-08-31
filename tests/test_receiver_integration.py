"""True RadioEnvironment -> SieveReceiver integration tests.

These tests deliberately avoid:

    receiver.tune(pulse_frequency)
    receiver.current_time_us = pulse_toa

The receiver must operate using its own static scan state and only learn
about pulses when the RadioEnvironment emits entry/exit events.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_ENV_DIR = REPO_ROOT / "SIMULATION ENV"

sys.path.insert(
    0,
    str(REPO_ROOT),
)

sys.path.insert(
    0,
    str(SIM_ENV_DIR),
)

from sim_env import (  # noqa: E402
    FileRecordSource,
    PulseRecord,
    RadioEnvironment,
    SimConfig,
    SieveReceiver,
)

from sim_env.receiver import (  # noqa: E402
    RadioReceiverBridge,
    attach_receiver,
)


def make_record(
    *,
    toa_us: float,
    frequency_mhz: float,
    pulse_width_us: float,
    amplitude_db: float = -100.0,
    aoa_deg: float = 45.0,
    emitter_id: int = 1,
    source_id: str = "integration",
) -> PulseRecord:
    """Create a repository-native PulseRecord."""
    return PulseRecord(
        toa_us=float(toa_us),
        frequency_mhz=float(
            frequency_mhz
        ),
        pulse_width_us=float(
            pulse_width_us
        ),
        amplitude_db=float(
            amplitude_db
        ),
        aoa_deg=float(
            aoa_deg
        ),
        emitter_id=int(
            emitter_id
        ),
        data=(
            float(toa_us),
            float(frequency_mhz),
            float(pulse_width_us),
            float(amplitude_db),
            float(aoa_deg),
        ),
        source_id=source_id,
    )


class TestReceiverEnvironmentIntegration(
    unittest.TestCase
):
    def test_environment_feeds_receiver_without_retuning(
        self,
    ):
        """Environment events reach the receiver through the real callback."""
        records = [
            # Initial receiver state:
            #
            # center = 500 MHz
            # window = 0..1000 MHz
            #
            # Therefore this pulse should be detected WITHOUT tuning.
            make_record(
                toa_us=10.0,
                frequency_mhz=750.0,
                pulse_width_us=50.0,
                emitter_id=1,
            ),

            # After the first 100 us dwell, the static scanner moves to:
            #
            # center = 1000 MHz
            # window = 500..1500 MHz
            #
            # This pulse arrives at 120 us and should be detected.
            make_record(
                toa_us=120.0,
                frequency_mhz=1400.0,
                pulse_width_us=50.0,
                emitter_id=2,
            ),

            # Same time, outside the current 500..1500 MHz window.
            # This must NOT be detected.
            make_record(
                toa_us=120.0,
                frequency_mhz=3000.0,
                pulse_width_us=50.0,
                emitter_id=3,
            ),

            # Same time as the 1400 MHz pulse and also inside the window.
            # Must remain a separate simultaneous detection.
            make_record(
                toa_us=120.0,
                frequency_mhz=1300.0,
                pulse_width_us=50.0,
                emitter_id=4,
            ),
        ]

        source = __import__(
            "sim_env"
        ).RecordSource(records)

        config = SimConfig(
            inputs=[],
            snapshot_interval_us=None,
        )

        environment = RadioEnvironment(
            source,
            config,
        )

        receiver = SieveReceiver(
            total_bandwidth=18e3,
            ibw=1e3,
            frequency_step=500.0,
            dwell_time=100.0,
            detection_threshold_db=-140.0,
        )

        bridge = attach_receiver(
            environment,
            receiver,
        )

        self.assertIsInstance(
            bridge,
            RadioReceiverBridge,
        )

        # The receiver begins at 500 MHz.
        self.assertEqual(
            receiver.center_frequency_mhz,
            500.0,
        )

        environment.run()

        # --------------------------------------------------------------
        # Receiver must have advanced its own clock.
        # --------------------------------------------------------------

        self.assertGreaterEqual(
            receiver.current_time_us,
            120.0,
        )

        # --------------------------------------------------------------
        # Receiver must NOT be sitting on 750/1400 MHz because of
        # pulse-driven retuning.
        #
        # At 120 us it has followed the static scan:
        #
        #   500 MHz -> 1000 MHz
        # --------------------------------------------------------------

        self.assertEqual(
            receiver.center_frequency_mhz,
            1000.0,
        )

        # --------------------------------------------------------------
        # The 750 MHz pulse should have been detected.
        # --------------------------------------------------------------

        detected_ids = {
            detection.pulse_id
            for detection
            in receiver.detection_history
        }

        self.assertIn(
            0,
            detected_ids,
        )

        # --------------------------------------------------------------
        # Pulse at 1400 MHz and pulse at 1300 MHz should both have
        # been detected at the same environment ToA.
        # --------------------------------------------------------------

        detections_at_120 = [
            detection
            for detection
            in receiver.detection_history
            if abs(
                detection.time_us - 120.0
            ) < 1e-9
        ]

        frequencies_at_120 = {
            round(
                detection.frequency_mhz,
                6,
            )
            for detection
            in detections_at_120
        }

        self.assertIn(
            1400.0,
            frequencies_at_120,
        )

        self.assertIn(
            1300.0,
            frequencies_at_120,
        )

        # 3000 MHz must not be detected.
        self.assertNotIn(
            3000.0,
            frequencies_at_120,
        )

    def test_receiver_does_not_need_emitter_id_to_detect(
        self,
    ):
        records = [
            make_record(
                toa_us=10.0,
                frequency_mhz=750.0,
                pulse_width_us=50.0,
                emitter_id=99,
            )
        ]

        # Remove the emitter label from the RF observation by passing
        # a pulse dictionary without emitter_id.
        receiver = SieveReceiver(
            detection_threshold_db=-140.0
        )

        receiver.current_time_us = 10.0

        pulse = {
            "frequency_mhz": 750.0,
            "toa_us": 10.0,
            "pulse_width_us": 50.0,
            "exit_us": 60.0,
            "amplitude_db": -100.0,
            "aoa_deg": 45.0,
            "pulse_id": 42,
        }

        detection = receiver.process_pulse(
            pulse
        )

        self.assertIsNotNone(
            detection
        )

        self.assertTrue(
            detection.detected
        )

        # Emitter identity is not required for physical detection.
        self.assertIsNone(
            detection.emitter_id
        )

    def test_nested_ndjson_emitter_id_is_ground_truth_only(
        self,
    ):
        receiver = SieveReceiver(
            detection_threshold_db=-140.0
        )

        event = {
            "event": "entry",
            "time_us": 10.0,
            "pulse": {
                "frequency_mhz": 750.0,
                "toa_us": 10.0,
                "pulse_width_us": 50.0,
                "exit_us": 60.0,
                "amplitude_db": -100.0,
                "aoa_deg": 45.0,
                "pulse_id": 7,
                "emitter_id": 123,
            },
        }

        detection = receiver.process_event(
            event
        )

        self.assertIsNotNone(
            detection
        )

        self.assertTrue(
            detection.detected
        )

        # Ground-truth label is recovered from nested pulse data.
        self.assertEqual(
            detection.emitter_id,
            123,
        )

    def test_entry_adds_and_exit_removes_pulse(
        self,
    ):
        receiver = SieveReceiver(
            detection_threshold_db=-140.0
        )

        entry_event = {
            "event": "entry",
            "time_us": 10.0,
            "pulse": {
                "frequency_mhz": 750.0,
                "toa_us": 10.0,
                "pulse_width_us": 50.0,
                "exit_us": 60.0,
                "amplitude_db": -100.0,
                "aoa_deg": 45.0,
                "pulse_id": 11,
                "emitter_id": 2,
            },
        }

        receiver.process_event(
            entry_event
        )

        self.assertEqual(
            len(receiver.buffered_pulses()),
            1,
        )

        exit_event = {
            "event": "exit",
            "time_us": 60.0,
            "pulse_id": 11,
            "pulse": entry_event["pulse"],
        }

        receiver.process_event(
            exit_event
        )

        self.assertEqual(
            len(receiver.buffered_pulses()),
            0,
        )

    def test_pulse_outside_frequency_window_is_not_detected(
        self,
    ):
        receiver = SieveReceiver(
            detection_threshold_db=-140.0
        )

        receiver.tune(
            8000.0
        )

        receiver.current_time_us = 100.0
        receiver.dwell_start_us = 100.0

        pulse = {
            "frequency_mhz": 12000.0,
            "toa_us": 100.0,
            "pulse_width_us": 50.0,
            "exit_us": 150.0,
            "amplitude_db": -80.0,
            "aoa_deg": 45.0,
            "pulse_id": 1,
        }

        self.assertIsNone(
            receiver.process_pulse(
                pulse
            )
        )

    def test_pulse_outside_time_window_is_not_detected(
        self,
    ):
        receiver = SieveReceiver(
            detection_threshold_db=-140.0
        )

        receiver.tune(
            8000.0
        )

        receiver.current_time_us = 200.0
        receiver.dwell_start_us = 200.0

        pulse = {
            "frequency_mhz": 8000.0,
            "toa_us": 50.0,
            "pulse_width_us": 25.0,
            "exit_us": 75.0,
            "amplitude_db": -80.0,
            "aoa_deg": 45.0,
            "pulse_id": 1,
        }

        self.assertIsNone(
            receiver.process_pulse(
                pulse
            )
        )

    def test_reset_clears_live_pulse_buffer(
        self,
    ):
        receiver = SieveReceiver()

        receiver.add_pulse(
            {
                "frequency_mhz": 750.0,
                "toa_us": 10.0,
                "pulse_width_us": 50.0,
                "exit_us": 60.0,
                "amplitude_db": -100.0,
                "aoa_deg": 45.0,
                "pulse_id": 1,
            }
        )

        self.assertEqual(
            len(receiver.buffered_pulses()),
            1,
        )

        receiver.reset()

        self.assertEqual(
            len(receiver.buffered_pulses()),
            0,
        )

        self.assertEqual(
            receiver.current_time_us,
            0.0,
        )

        self.assertEqual(
            receiver.center_frequency_mhz,
            receiver.legal_center_min_mhz,
        )


if __name__ == "__main__":
    unittest.main()
