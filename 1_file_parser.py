"""Convert every RF record .h5 file into a matching output_<id>.txt (+ sidecar
metadata JSON), for consumption by sim_env/ingest.py.

Changes vs. the original 1_file_parser.py:

1. BATCHED, not hand-edited: loops over every .h5 file found under
   `input_dir` instead of one hardcoded `file_path` / `output_path` pair.
   The output filename is *derived* from the input filename (test_142.h5 ->
   output_142.txt), so there's never an ambiguous mapping between a source
   .h5 file and its parsed .txt, and nothing gets silently skipped.

2. CRITICAL FIX -- AOA/Amplitude column swap: metadata/feature_names in the
   source .h5 files gives the true column order as
   [UTCTime, RF, PulseWidth, AOA, PA]. Every downstream consumer in
   sim_env (config.py's FEATURE_FIELDS, ingest.py's parse_record_line,
   README.md) assumes [toa, freq, pulse_width, amplitude_db, aoa_deg] --
   i.e. columns 3 and 4 (0-indexed) reversed relative to the source. This
   was verified numerically: column 3 is always in [-180, 180] (an angle --
   AOA), column 4 is always negative / power-like (amplitude), across every
   file checked. We swap columns 3 and 4 here, at the single point where
   .h5 becomes .txt, so every other sim_env module keeps working completely
   unchanged.

3. Captures the `metadata` group instead of dropping it: feature names,
   the transmitters lookup table (decoded with ast.literal_eval -- these
   strings are Python repr(), NOT valid JSON: single-quoted, with a raw
   tuple for start_position_km, so json.loads() will raise), the group's
   own attrs (collection_time_s, date_created, description, num_pulses,
   type), and an explicit flag if metadata/receiver/* is empty (it's empty
   in the source data itself -- not something a parser can recover).

   IMPORTANT: `label` is the row INDEX into that file's own local
   `transmitters` table, not a globally stable emitter id -- verified
   different files have completely different transmitter counts/contents
   (e.g. 88 vs. 50 vs. 21 entries). So label=68 in one file means nothing
   in another file. This metadata is written to a *separate* sidecar file
   rather than inline in the record lines, so it doesn't touch the
   `record_N: data=[...], label=...` line format that ingest.py's regex
   depends on.

Run:
    python 1_file_parser.py
(edit INPUT_DIR / OUTPUT_DIR below if your layout differs)
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import h5py

# The RF validation gate: rejects dirty records (NaN/Inf, PW<=0, ToA<0,
# decreasing ToA, freq<=0, invalid emitter) and normalizes signed AoA BEFORE
# they are written to output_*.txt. It lives at the repo root.
try:
    from data_validation import RecordValidator, ValidationConfig
    _VALIDATION_AVAILABLE = True
except ImportError as exec:
   raise RuntimeError(
        "data_validation is required. "
        "Refusing to generate unvalidated output files."
    ) from exc

# ---------------------------------------------------------------------------
# Configuration -- adjust these two paths to your layout.
# ---------------------------------------------------------------------------
INPUT_DIR = Path(__file__).resolve().parent  # folder containing the .h5 files
OUTPUT_DIR = INPUT_DIR.parent / "OUTPUT FILES"  # where output_<id>.txt goes

REQUIRED_DATASETS = {"data", "labels"}


def decode_transmitters(raw: "h5py.Dataset") -> list:
    """Decode the transmitters array. Entries are Python repr() strings
    (single-quoted dict literals with a raw tuple inside), NOT JSON --
    ast.literal_eval is required; json.loads will raise on these.
    """
    decoded = []
    for item in raw[:]:
        s = item.decode("utf-8", errors="replace") if isinstance(item, bytes) else item
        try:
            decoded.append(ast.literal_eval(s))
        except (ValueError, SyntaxError):
            decoded.append({"_raw_undecodable": s})
    return decoded


def extract_metadata(h5file: "h5py.File") -> dict:
    """Pull everything under metadata/ into a plain-JSON-serializable dict."""
    meta_out: dict = {}
    if "metadata" not in h5file:
        return meta_out

    meta_group = h5file["metadata"]

    # Group-level attrs (collection_time_s, date_created, description, ...)
    meta_out["attrs"] = {k: (v.item() if hasattr(v, "item") else v)
                          for k, v in meta_group.attrs.items()}

    # Feature names -> lets a human/consumer confirm the post-swap column order
    if "feature_names" in meta_group:
        names = [n.decode("utf-8") if isinstance(n, bytes) else n
                  for n in meta_group["feature_names"][:]]
        meta_out["feature_names_source_order"] = names
        # Document the corrected order actually written to the .txt records:
        meta_out["feature_names_written_order"] = (
            names[:3] + [names[4], names[3]] if len(names) >= 5 else names
        )

    # Transmitters lookup table (label = row index into THIS list, this file only)
    if "transmitters" in meta_group:
        meta_out["transmitters"] = decode_transmitters(meta_group["transmitters"])
        meta_out["transmitters_note"] = (
            "label in records: is the row INDEX into this list, local to this "
            "file only -- NOT a global/cross-file emitter id. Different files "
            "have different transmitter counts and contents."
        )

    # Receiver config -- flag rather than silently omit if empty at the source
    if "receiver" in meta_group:
        receiver = meta_group["receiver"]
        empty_fields = [k for k in receiver if len(receiver[k].attrs) == 0
                         and getattr(receiver[k], "shape", None) is None]
        meta_out["receiver_status"] = (
            "empty at source (no attrs/data present) -- not a parsing gap, "
            "the generator never populated these fields"
            if empty_fields == list(receiver.keys())
            else "partially populated -- inspect manually"
        )

    return meta_out


def parse_one_file(h5_path: Path, output_dir: Path) -> None:
    output_path = output_dir / f"output_{h5_path.stem.split('_', 1)[-1]}.txt"
    meta_path = output_dir / f"output_{h5_path.stem.split('_', 1)[-1]}.meta.json"

    with h5py.File(h5_path, "r") as file:
        missing = REQUIRED_DATASETS.difference(file.keys())
        if missing:
            raise KeyError(f"{h5_path.name}: missing datasets {sorted(missing)}")

        data = file["data"][:]
        labels = file["labels"][:]

        if len(data) != len(labels):
            raise ValueError(
                f"{h5_path.name}: data/label length mismatch: "
                f"{len(data)} data rows, {len(labels)} labels"
            )

        # --- CRITICAL FIX: swap columns 3 and 4 (0-indexed) -----------------
        # Source order: [UTCTime, RF, PulseWidth, AOA, PA]
        # Written order (matches sim_env's FEATURE_FIELDS assumption):
        #                [toa,     freq, pulse_width, amplitude_db, aoa_deg]
        if data.shape[1] >= 5:
            data = data.copy()
            data[:, [3, 4]] = data[:, [4, 3]]

        # --- RF VALIDATION GATE (primary boundary) ---------------------------
        # Map the swapped rows into the semantic 5-feature vector in written
        # order [toa, freq, pw, amp, aoa], validate each, normalize AoA, and
        # discard invalid records. Only validated rows are written.
        validator = RecordValidator(ValidationConfig()) if _VALIDATION_AVAILABLE else None
        clean_rows: list = []       # list of (data_vector[5], label)
        original_indices: list = []  # source row index (1-based) of each kept row

        if validator is not None:
            for row_number, (row, label) in enumerate(list(zip(data, labels.ravel())), start=1):
                vals = [float(x) for x in row]
                try:
                    emitter = int(label)
                except (TypeError, ValueError):
                    emitter = label
                clean = validator.validate(vals, emitter, row_number)
                if clean is not None:
                    clean_rows.append((
                        [clean["toa_us"], clean["frequency_mhz"],
                         clean["pulse_width_us"], clean["amplitude_db"],
                         clean["aoa_deg"]],
                        clean["emitter_id"],
                    ))
                    original_indices.append(row_number)
        else:
            # Degenerate fallback (validation not importable): keep raw rows.
            for row, label in zip(data, labels.ravel()):
                clean_rows.append(([float(x) for x in row], int(label)))

        output_lines = []
        output_lines.append("dataset_names: " + ", ".join(file.keys()))
        output_lines.append(f"data_shape: ({len(clean_rows)}, 5)")
        output_lines.append(f"data_dtype: {data.dtype}")
        output_lines.append(f"labels_shape: ({len(clean_rows)}, 1)")
        output_lines.append(f"labels_dtype: {labels.dtype}")
        output_lines.append(f"record_count: {len(clean_rows)}")
        output_lines.append("records:")

        # Renumber contiguously -- never leave holes (record_1, record_2, ...).
        for new_number, ((row_values, label), orig_idx) in enumerate(
            zip(clean_rows, original_indices or [None] * len(clean_rows)), start=1
        ):
            output_lines.append(
                f"record_{new_number}: data={row_values}, label={label}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

        # Validation report for this episode, kept separate from sim input.
        if validator is not None:
            report_path = output_dir / f"{output_path.stem}.validation.json"
            report_path.write_text(
                json.dumps(validator.summary.to_report(), indent=2),
                encoding="utf-8",
            )

        # Sidecar metadata -- kept separate so the record-line format ingest.py
        # parses via regex never changes.
        metadata = extract_metadata(file)
        meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    print(f"[ok] {h5_path.name} -> {output_path.name} (+ {meta_path.name})")


def main() -> None:
    h5_files = sorted(INPUT_DIR.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found under: {INPUT_DIR}")

    errors = []
    for h5_path in h5_files:
        try:
            parse_one_file(h5_path, OUTPUT_DIR)
        except Exception as exc:  # noqa: BLE001 -- report and continue the batch
            errors.append((h5_path.name, str(exc)))
            print(f"[FAIL] {h5_path.name}: {exc}")

    print(f"\nDone: {len(h5_files) - len(errors)}/{len(h5_files)} converted.")
    if errors:
        print("Failures:")
        for name, err in errors:
            print(f"  - {name}: {err}")


if __name__ == "__main__":
    main()
