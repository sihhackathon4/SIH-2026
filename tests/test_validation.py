"""Tests for the RF data validation pipeline + its integration with sim_env.

Runs with the standard-library ``unittest`` (no third-party test framework):

    python -m unittest discover -s tests -v

Run from the repo root so both packages are importable.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_ENV_DIR = REPO_ROOT / "SIMULATION ENV"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SIM_ENV_DIR))

from data_validation import (  # noqa: E402
    ValidationConfig,
    RecordValidator,
    validate_parsed_record,
    normalize_aoa,
    clean_output_file,
    parse_output_line,
)
from data_validation.validator import (  # noqa: E402
    BAD_WIDTH,
    NON_FINITE,
    INVALID_EMITTER,
    NEGATIVE_TOA,
    DEcreasing_TOA,
    DUPLICATE_TOA,
    PW_NOT_POSITIVE,
    FREQ_NOT_POSITIVE,
    AOA_NOT_FINITE,
    AOA_OUT_OF_RANGE,
    AMP_NOT_FINITE,
    MISSING_FIELD,
)
from sim_env.ingest import parse_record_line  # noqa: E402
from sim_env.timeline_writer import SCHEMA_VERSION, TimelineWriter  # noqa: E402


VALID = (100.0, 1000.0, 2.0, -20.0, 45.0)


class TestBasicValid(unittest.TestCase):
    def test_valid_record_passes(self):
        rec = validate_parsed_record(VALID, 1)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["toa_us"], 100.0)
        self.assertEqual(rec["frequency_mhz"], 1000.0)
        self.assertEqual(rec["pulse_width_us"], 2.0)
        self.assertEqual(rec["amplitude_db"], -20.0)
        self.assertEqual(rec["aoa_deg"], 45.0)
        self.assertEqual(rec["emitter_id"], 1)

    def test_valid_record_summary(self):
        v = RecordValidator()
        self.assertIsNotNone(v.validate(VALID, 1, 1))
        self.assertEqual(v.summary.valid_records, 1)
        self.assertEqual(v.summary.invalid_records, 0)
        self.assertEqual(v.summary.total_records, 1)


class TestNumericRejections(unittest.TestCase):
    def test_nan(self):
        self.assertIsNone(validate_parsed_record((float("nan"), 1000, 2, -20, 45), 1))

    def test_inf(self):
        self.assertIsNone(validate_parsed_record((100, float("inf"), 2, -20, 45), 1))

    def test_nan_frequency(self):
        self.assertIsNone(validate_parsed_record((100, float("nan"), 2, -20, 45), 1))

    def test_nan_aoa(self):
        self.assertIsNone(validate_parsed_record((100, 1000, 2, -20, float("nan")), 1))

    def test_infinite_aoa(self):
        self.assertIsNone(validate_parsed_record((100, 1000, 2, -20, float("inf")), 1))

    def test_nan_amplitude(self):
        self.assertIsNone(validate_parsed_record((100, 1000, 2, float("nan"), 45), 1))

    def test_zero_pw_rejected(self):
        rec = validate_parsed_record((100, 1000, 0.0, -20, 45), 1)
        self.assertIsNone(rec)

    def test_negative_pw_rejected(self):
        self.assertIsNone(validate_parsed_record((100, 1000, -1.0, -20, 45), 1))

    def test_zero_frequency_rejected(self):
        self.assertIsNone(validate_parsed_record((100, 0.0, 2, -20, 45), 1))

    def test_negative_frequency_rejected(self):
        self.assertIsNone(validate_parsed_record((100, -50.0, 2, -20, 45), 1))

    def test_issue_codes_recorded(self):
        v = RecordValidator()
        v.validate((100, -5.0, 2, -20, 45), 1, 1)
        self.assertIn(FREQ_NOT_POSITIVE, v.summary.issue_counts)


class TestToA(unittest.TestCase):
    def test_negative_toa_rejected(self):
        self.assertIsNone(validate_parsed_record((-1.0, 1000, 2, -20, 45), 1))

    def test_decreasing_toa_rejected(self):
        v = RecordValidator()
        self.assertIsNotNone(v.validate((100.0, 1000, 2, -20, 45), 1, 1))
        self.assertIsNone(v.validate((50.0, 1000, 2, -20, 45), 1, 2))
        self.assertIn(DEcreasing_TOA, v.summary.issue_counts)

    def test_equal_toa_preserved_by_default(self):
        v = RecordValidator()
        self.assertIsNotNone(v.validate((100.0, 1000, 2, -20, 45), 1, 1))
        self.assertIsNotNone(v.validate((100.0, 2000, 2, -30, 10), 2, 2))
        self.assertEqual(v.summary.valid_records, 2)
        # recorded as a non-fatal duplicate issue
        dups = [i for i in v.summary.issues if i.code == DUPLICATE_TOA]
        self.assertEqual(len(dups), 1)
        self.assertFalse(dups[0].fatal)

    def test_duplicate_toa_rejected_in_strict_mode(self):
        cfg = ValidationConfig(reject_duplicate_timestamps=True)
        v = RecordValidator(cfg)
        self.assertIsNotNone(v.validate((100.0, 1000, 2, -20, 45), 1, 1))
        self.assertIsNone(v.validate((100.0, 2000, 2, -30, 10), 2, 2))


class TestAoA(unittest.TestCase):
    def test_signed_aoa_normalized(self):
        self.assertEqual(normalize_aoa(-10), 350.0)
        self.assertEqual(normalize_aoa(-20), 340.0)
        self.assertEqual(normalize_aoa(-45), 315.0)
        self.assertEqual(normalize_aoa(-90), 270.0)

    def test_aoa_360_to_0(self):
        self.assertEqual(normalize_aoa(360), 0.0)

    def test_signed_aoa_survives_validation(self):
        rec = validate_parsed_record((100, 1000, 2, -20, -20), 1)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["aoa_deg"], 340.0)

    def test_aoa_360_recorded_as_0(self):
        rec = validate_parsed_record((100, 1000, 2, -20, 360), 1)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["aoa_deg"], 0.0)

    def test_aoa_out_of_canonical_range_when_not_normalized(self):
        cfg = ValidationConfig(normalize_signed_aoa=False)
        self.assertIsNone(validate_parsed_record((100, 1000, 2, -20, 400), 1, config=cfg))


class TestEmitter(unittest.TestCase):
    def test_invalid_emitter_rejected(self):
        self.assertIsNone(validate_parsed_record(VALID, "abc", 1))
        self.assertIsNone(validate_parsed_record(VALID, -1, 1))
        self.assertIsNone(validate_parsed_record(VALID, float("nan"), 1))

    def test_valid_emitter(self):
        self.assertIsNotNone(validate_parsed_record(VALID, 0, 1))
        self.assertIsNotNone(validate_parsed_record(VALID, 78, 1))


class TestSchemaWidth(unittest.TestCase):
    def test_wrong_width_rejected(self):
        self.assertIsNone(validate_parsed_record((100, 1000, 2, -20), 1))
        self.assertIsNone(validate_parsed_record((100, 1000, 2, -20, 45, 99), 1))

    def test_missing_field_detected(self):
        v = RecordValidator()
        self.assertIsNone(v.validate(None, 1, 1))
        self.assertIn(BAD_WIDTH, v.summary.issue_counts)


class TestParserIntegration(unittest.TestCase):
    def test_parse_record_line_valid(self):
        rec = parse_record_line("record_1: data=[100.0, 1000.0, 2.0, -20.0, 45.0], label=1",
                                source_id="s", on_nonfinite="drop")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.pulse_width_us, 2.0)
        self.assertEqual(rec.emitter_id, 1)

    def test_parse_record_line_rejects_bad_pw_by_default(self):
        rec = parse_record_line("record_1: data=[100.0, 1000.0, 0.0, -20.0, 45.0], label=1",
                                source_id="s", on_nonfinite="drop")
        self.assertIsNone(rec)

    def test_parse_record_line_rejects_nonfinite_by_default(self):
        rec = parse_record_line("record_1: data=[100.0, nan, 2.0, -20.0, 45.0], label=1",
                                source_id="s", on_nonfinite="drop")
        self.assertIsNone(rec)


class TestCleanOutput(unittest.TestCase):
    HEADER = ("dataset_names: data, labels, metadata\n"
              "data_shape: (5, 5)\n"
              "data_dtype: float32\n"
              "labels_shape: (5, 1)\n"
              "labels_dtype: int8\n"
              "record_count: 5\n")

    def _write(self, path, lines):
        path.write_text(self.HEADER + "records:\n" + "\n".join(lines) + "\n",
                        encoding="utf-8")

    def test_renumber_contiguous(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "output_99.txt"
            self._write(src, [
                "record_1: data=[100.0, 1000.0, 2.0, -20.0, 45.0], label=1",
                "record_2: data=[101.0, 1000.0, 0.0, -20.0, 45.0], label=1",  # bad PW
                "record_3: data=[102.0, 1000.0, 2.0, -20.0, 45.0], label=2",
                "record_4: data=[103.0, 1000.0, 2.0, -20.0, 45.0], label=2",
                "record_8: data=[104.0, 1000.0, 2.0, -20.0, -10.0], label=3",  # signed aoa
            ])
            out = Path(d) / "output_99.clean.txt"
            report = clean_output_file(src, output_path=out,
                                       validation_report_dir=Path(d) / "reports")
            self.assertEqual(report["valid_records"], 4)
            self.assertEqual(report["invalid_records"], 1)
            lines = out.read_text(encoding="utf-8").splitlines()
            self.assertIn("record_count: 4", lines)
            self.assertTrue(any(l == "record_1: data=[100.0, 1000.0, 2.0, -20.0, 45.0], label=1"
                                for l in lines))
            self.assertTrue(any(l == "record_2: data=[102.0, 1000.0, 2.0, -20.0, 45.0], label=2"
                                for l in lines))
            self.assertTrue(any(l == "record_3: data=[103.0, 1000.0, 2.0, -20.0, 45.0], label=2"
                                for l in lines))
            self.assertTrue(any(l == "record_4: data=[104.0, 1000.0, 2.0, -20.0, 350.0], label=3"
                                for l in lines))

    def test_header_counts_match(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "output_7.txt"
            self._write(src, [f"record_{i}: data=[{100+i}.0, 1000.0, 2.0, -20.0, 45.0], label=1"
                              for i in range(1, 6)])
            out = Path(d) / "output_7.clean.txt"
            clean_output_file(src, output_path=out, validation_report_dir=Path(d) / "r")
            lines = out.read_text(encoding="utf-8").splitlines()
            hdr = {l.split(":")[0]: l for l in lines if l.startswith(("data_shape", "labels_shape", "record_count"))}
            import re
            n = int(re.search(r"\d+", hdr["record_count"]).group())
            ds = int(re.search(r"\((\d+),", hdr["data_shape"]).group(1))
            ls = int(re.search(r"\((\d+),", hdr["labels_shape"]).group(1))
            self.assertEqual(ds, ls)
            self.assertEqual(ds, n)

    def test_empty_cleaned_file_does_not_crash(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "output_3.txt"
            self._write(src, [
                "record_1: data=[100.0, 1000.0, 0.0, -20.0, 45.0], label=1",  # bad pw
                "record_2: data=[101.0, 1000.0, 2.0, nan, 45.0], label=1",    # nan amp
            ])
            out = Path(d) / "output_3.clean.txt"
            report = clean_output_file(src, output_path=out,
                                       validation_report_dir=Path(d) / "r")
            self.assertEqual(report["valid_records"], 0)
            self.assertTrue(out.exists())


class TestNDJSONIntegration(unittest.TestCase):
    def _run_writer(self, tmp):
        from sim_env.config import SimConfig
        from sim_env.ingest import FileRecordSource
        from sim_env.environment import RadioEnvironment

        src_txt = tmp / "output_0.txt"
        src_txt.write_text(
            "dataset_names: data, labels\n"
            "data_shape: (2, 5)\n"
            "labels_shape: (2, 1)\n"
            "record_count: 2\n"
            "records:\n"
            "record_1: data=[100.0, 1000.0, 2.0, -20.0, -10.0], label=1\n"
            "record_2: data=[100.0, 2000.0, 3.0, -30.0, 350.0], label=2\n",
            encoding="utf-8",
        )
        out_ndjson = tmp / "timeline.ndjson"
        cfg = SimConfig(inputs=[src_txt], snapshot_interval_us=None,
                        output_log=out_ndjson)
        source = FileRecordSource([src_txt], on_nonfinite="drop")
        writer = TimelineWriter(out_ndjson, config=cfg)
        writer.write_meta(record_count=2)
        env = RadioEnvironment(source, cfg, on_event=writer.on_event)
        env.run()
        writer.close()
        return out_ndjson

    def test_meta_first_and_data_validated(self):
        with tempfile.TemporaryDirectory() as d:
            out = self._run_writer(Path(d))
            with open(out, encoding="utf-8") as fh:
                first = json.loads(fh.readline())
            self.assertEqual(first["event"], "meta")
            self.assertTrue(first["data_validated"])
            self.assertEqual(first["schema_version"], SCHEMA_VERSION)
            self.assertEqual(first["aoa_range"], [0.0, 360.0])
            self.assertEqual(first["label"], "emitter_id")

    def test_equal_timestamps_separate_and_aoa_canonical(self):
        with tempfile.TemporaryDirectory() as d:
            out = self._run_writer(Path(d))
            from sim_env.timeline_reader import iter_events
            entries = [ev for ev in iter_events(out) if ev["event"] == "entry"]
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0]["time_us"], 100.0)
            self.assertEqual(entries[1]["time_us"], 100.0)  # distinct, not merged
            # aoA normalized for the signed first record
            self.assertEqual(entries[0]["pulse"]["aoa_deg"], 350.0)
            self.assertEqual(entries[1]["pulse"]["aoa_deg"], 350.0)

    def test_no_invalid_records_generate_events(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "output_5.txt"
            src.write_text(
                "records:\n"
                "record_1: data=[100.0, 1000.0, 2.0, -20.0, 45.0], label=1\n"
                "record_2: data=[100.0, 1000.0, 0.0, -20.0, 45.0], label=1\n"
                "record_3: data=[100.0, 1000.0, 2.0, nan, 45.0], label=1\n",
                encoding="utf-8",
            )
            out = Path(d) / "t.ndjson"
            from sim_env.config import SimConfig
            from sim_env.environment import RadioEnvironment
            from sim_env.ingest import FileRecordSource
            cfg = SimConfig(inputs=[src], snapshot_interval_us=None)
            source = FileRecordSource([src], on_nonfinite="drop")
            writer = TimelineWriter(out, config=cfg)
            writer.write_meta()
            RadioEnvironment(source, cfg, on_event=writer.on_event).run()
            writer.close()
            from sim_env.timeline_reader import iter_events
            entries = [ev for ev in iter_events(out) if ev["event"] == "entry"]
            self.assertEqual(len(entries), 1)  # only the valid record


if __name__ == "__main__":
    unittest.main()
