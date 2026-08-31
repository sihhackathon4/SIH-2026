"""CLI entry point for the data_validation package.

Examples
--------
Validate (without rewriting) one or more output files, writing reports::

    python -m data_validation --validate "OUTPUT FILES/output_0.txt"

Clean (validate + rewrite + report) files into a clean dir::

    python -m data_validation --clean "OUTPUT FILES/*.txt" \
        --output-dir "VALIDATED OUTPUT FILES" \
        --report-dir "validation_reports"

Options mirror the simulator's own so the two CLIs agree on the physical rules.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .clean_output import clean_output_file, clean_output_files
from .config import ValidationConfig
from .validator import RecordValidator


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="data_validation",
        description=(
            "Validate RF output_*.txt records against the semantic RF schema "
            "before they reach the simulation environment."
        ),
    )
    p.add_argument("paths", nargs="*", help="Files or globs to validate/clean.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--validate", action="store_true",
                      help="Validate only (report to --report-dir); do not rewrite.")
    mode.add_argument("--clean", action="store_true",
                      help="Validate and rewrite cleaned files (default).")
    p.add_argument("--output-dir", default=None,
                   help="Directory for cleaned files (default: alongside source).")
    p.add_argument("--report-dir", default="validation_reports",
                   help="Directory for validation reports.")
    p.add_argument("--min-frequency-mhz", type=float, default=0.0)
    p.add_argument("--max-frequency-mhz", type=float, default=None)
    p.add_argument("--min-aoa-deg", type=float, default=0.0)
    p.add_argument("--max-aoa-deg", type=float, default=360.0)
    p.add_argument("--no-normalize-aoa", action="store_true",
                   help="Do not fold signed AoA into [0, 360).")
    p.add_argument("--reject-duplicate-timestamps", action="store_true")
    p.add_argument("--min-duration-us", type=float, default=0.0)
    p.add_argument("--max-duration-us", type=float, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _expand(path_args) -> list:
    import glob

    expanded = []
    for a in path_args:
        if glob.has_magic(a):
            expanded.extend(sorted(glob.glob(a)))
        else:
            expanded.append(a)
    return sorted({Path(x) for x in expanded if Path(x).is_file()})


def main(argv=None) -> int:
    parser = _build_parser()
    ns = parser.parse_args(argv)

    if not ns.paths:
        parser.print_usage()
        return 2

    cfg = ValidationConfig(
        min_frequency_mhz=ns.min_frequency_mhz,
        max_frequency_mhz=ns.max_frequency_mhz,
        min_aoa_deg=ns.min_aoa_deg,
        max_aoa_deg=ns.max_aoa_deg,
        normalize_signed_aoa=not ns.no_normalize_aoa,
        reject_duplicate_timestamps=ns.reject_duplicate_timestamps,
        min_duration_us=ns.min_duration_us,
        max_duration_us=ns.max_duration_us,
    )

    files = _expand(ns.paths)
    reports = []
    for f in files:
        if ns.validate:
            # validate-only path: reuse cleaner's parser but don't rewrite
            validator = RecordValidator(cfg)
            count_valid = _validate_only(f, validator)
            report = validator.summary.to_report()
            report["output_file"] = f.name
            rep_dir = Path(ns.report_dir)
            rep_dir.mkdir(parents=True, exist_ok=True)
            import json

            (rep_dir / f"{f.stem}.validation.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8")
            reports.append(report)
            if ns.verbose:
                print(f"[validate] {f.name}: valid={count_valid} "
                      f"invalid={validator.summary.invalid_records}")
        else:
            report = clean_output_file(
                f, output_path=(Path(ns.output_dir) / f.name
                                if ns.output_dir else None),
                validation_report_dir=ns.report_dir, config=cfg)
            reports.append(report)
            if ns.verbose:
                print(f"[clean] {f.name}: valid={report['valid_records']} "
                      f"invalid={report['invalid_records']}")

    total_v = sum(r["valid_records"] for r in reports)
    total_i = sum(r["invalid_records"] for r in reports)
    print(f"Done: {len(files)} file(s), {total_v} valid, {total_i} invalid.")
    return 0


def _validate_only(path: Path, validator: RecordValidator) -> int:
    """Reuse clean_output parsing for the validate-only path without writing."""
    from .clean_output import parse_output_line

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parsed = parse_output_line(line)
            if parsed is None:
                continue
            data_toks, label_str = parsed
            rec_no = validator.summary.total_records + 1
            data = []
            ok = True
            for t in data_toks:
                try:
                    data.append(float(t.strip()))
                except ValueError:
                    ok = False
                    break
            try:
                label = int(label_str)
            except ValueError:
                label = label_str
            validator.validate(data if ok and len(data) == 5 else None, label, rec_no)
    validator.enforce_duration()
    return validator.summary.valid_records


if __name__ == "__main__":
    raise SystemExit(main())
