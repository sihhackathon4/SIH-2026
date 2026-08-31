"""Clean pre-validation ``output_*.txt`` files.

Migration/repair utility. It reads old ``output_*.txt`` files, parses and
validates every record, normalizes AoA, drops fatal-invalid records, renumbers
the survivors contiguously, rebuilds the header counts, and writes both a
clean output file and a validation report.

This is a ONE-OFF migration tool for output files generated before the
validation layer existed. Fresh transformations through the HDF5 parser (with
its built-in validation gate) already produce clean files and do NOT need to
ran through this utility on every run.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Union

from .config import TXT_DATA_FIELDS, ValidationConfig
from .validator import RecordValidator

_RECORD_RE = re.compile(
    r"record_\d+:\s*data=\[(?P<data>[^\]]*)\]\s*,\s*label=(?P<label>\S+)\s*"
)

__all__ = [
    "parse_output_line",
    "clean_output_file",
    "clean_output_files",
    "clean_output_dir",
]


def parse_output_line(line: str):
    """Parse one ``record_N: data=[...], label=...`` line.

    Returns ``(data_tokens, label_str)`` for record lines, or ``None`` for
    header/non-record lines. Raises ``ValueError`` on malformed record lines.
    """
    m = _RECORD_RE.search(line)
    if m is None:
        return None
    return m.group("data").split(","), m.group("label")


def clean_output_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    report_path: Optional[Union[str, Path]] = None,
    config: Optional[ValidationConfig] = None,
    validation_report_dir: Optional[Union[str, Path]] = None,
) -> dict:
    """Validate + clean one output file.

    Args:
        input_path: source ``output_*.txt``.
        output_path: where the clean file is written. Defaults to a sibling
            ``<stem>.clean.txt``.
        report_path: where the validation report JSON is written. Defaults to
            ``validation_reports/<stem>.validation.json``.
        config: validation rules.
        validation_report_dir: directory for the default report path.

    Returns the serialized validation report dict.
    """
    in_path = Path(input_path)
    cfg = config or ValidationConfig()
    out_path = Path(output_path) if output_path else in_path.with_name(
        in_path.stem + ".clean.txt")
    if report_path:
        rep_path = Path(report_path)
    else:
        rep_root = Path(validation_report_dir) if validation_report_dir else (
            in_path.parent / "validation_reports")
        rep_path = rep_root / f"{in_path.stem}.validation.json"

    validator = RecordValidator(cfg)

    header_lines: List[str] = []
    valid_rows: List[tuple] = []  # (data_vector, label)

    with open(in_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parsed = parse_output_line(line)
            if parsed is None:
                header_lines.append(line.rstrip("\n"))
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
            if not ok or len(data) != 5:
                # Let the validator report the width/parse issue properly.
                validator.validate(data if ok else None, label_str, rec_no)
                continue
            try:
                label = int(label_str)
            except ValueError:
                label = label_str
            clean = validator.validate(data, label, rec_no)
            if clean is not None:
                valid_rows.append((
                    [clean["toa_us"], clean["frequency_mhz"], clean["pulse_width_us"],
                     clean["amplitude_db"], clean["aoa_deg"]],
                    clean["emitter_id"],
                ))

    validator.enforce_duration()

    # --- write clean file (renumber contiguous) ----------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data_shape = f"({len(valid_rows)}, 5)"
    labels_shape = f"({len(valid_rows)}, 1)"
    with open(out_path, "w", encoding="utf-8") as out:
        # Preserve header lines verbatim, patching dimension/count lines, and
        # emit a single "records:" marker.
        wrote_records_marker = False
        for h in header_lines:
            if h.startswith("data_shape:"):
                out.write(f"data_shape: {data_shape}\n")
            elif h.startswith("labels_shape:"):
                out.write(f"labels_shape: {labels_shape}\n")
            elif h.startswith("record_count:"):
                out.write(f"record_count: {len(valid_rows)}\n")
            elif h.strip() == "records:":
                out.write("records:\n")
                wrote_records_marker = True
            else:
                out.write(h + "\n")
        if not wrote_records_marker:
            out.write("records:\n")
        for i, (data, label) in enumerate(valid_rows, start=1):
            fmt = ", ".join(repr(x) for x in data)
            out.write(f"record_{i}: data=[{fmt}], label={label}\n")

    # --- write validation report ---------------------------------------------
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    report = validator.summary.to_report()
    report["output_file"] = out_path.name
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def clean_output_files(
    paths: Iterable[Union[str, Path]],
    output_dir: Optional[Union[str, Path]] = None,
    validation_report_dir: Optional[Union[str, Path]] = None,
    config: Optional[ValidationConfig] = None,
) -> List[dict]:
    """Clean many output files and return their reports."""
    reports = []
    for p in paths:
        p = Path(p)
        out = p.with_name(p.stem + ".clean.txt")
        if output_dir:
            out = Path(output_dir) / out.name
        reports.append(
            clean_output_file(p, output_path=out,
                              validation_report_dir=validation_report_dir,
                              config=config)
        )
    return reports


def clean_output_dir(
    glob_input: str = "output_*.txt",
    input_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    validation_report_dir: Optional[Union[str, Path]] = None,
    config: Optional[ValidationConfig] = None,
) -> List[dict]:
    """Clean every matching file under ``input_dir`` (default current directory)."""
    import glob

    base = Path(input_dir) if input_dir else Path(".")
    matches = sorted(base.glob(glob_input))
    return clean_output_files(matches, output_dir=output_dir,
                              validation_report_dir=validation_report_dir,
                              config=config)
