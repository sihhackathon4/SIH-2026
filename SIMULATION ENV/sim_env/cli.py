"""Command-line entry point for the radiowave stream simulator.

Examples
--------
Run the full sweep over all OUTPUT FILES and write an NDJSON timeline::

    python -m sim_env.cli \
        --input "OUTPUT FILES/*.txt" \
        --output "OUTPUT FILES/stream_timeline.ndjson" \
        --snapshot-interval-us 5000

API-use example (no file output, just drive the environment)::

    python -c "from sim_env import *; ..."
"""

from __future__ import annotations

import argparse
import sys
import time as _time
from pathlib import Path

from .config import SimConfig
from .environment import RadioEnvironment
from .ingest import FileRecordSource
from .timeline_writer import TimelineWriter


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sim_env",
        description=(
            "Simulate a real-world radiowave stream from RF record files. "
            "Each radiopulse enters at its ToA, stays active for its Pulse "
            "Width, then disappears completely. Emits a continuous NDJSON "
            "timeline consumable by an ML scheduler."
        ),
    )
    p.add_argument(
        "-i", "--input", action="append", default=[],
        help=(
            "Path (or glob) to an RF record file. Repeatable. "
            "Default: 'OUTPUT FILES/output*.txt'."
        ),
    )
    p.add_argument(
        "-o", "--output", default=None,
        help="NDJSON event-log output path. Default: stdout.",
    )
    p.add_argument(
        "-s", "--snapshot-interval-us", type=float, default=1_000_000.0,
        help="Periodic snapshot interval in microseconds (None disables). "
             "Default: 1000000.0",
    )
    p.add_argument(
        "--no-entries", action="store_true", help="Do not emit 'entry' events."
    )
    p.add_argument(
        "--no-exits", action="store_true", help="Do not emit 'exit' events."
    )
    p.add_argument(
        "--no-snapshots", action="store_true", help="Do not emit 'snapshot' events."
    )
    p.add_argument(
        "--min-pw-us", type=float, default=0.0,
        help="Pulse widths below this (or negative/non-finite) are treated as "
             "instantaneous. Default: 0.0",
    )
    p.add_argument(
        "--nonfinite", choices=["allow", "drop", "raise"], default="drop",
        help="How to treat inf/nan in frequency/amplitude/aoa (and PW<=0). "
             "'drop' skips such records (default); 'raise' fails loudly; "
             "'allow' keeps them (legacy). For ML dataset building prefer "
             "--drop.",
    )
    p.add_argument(
        "--min-frequency-mhz", type=float, default=0.0,
        help="Records with frequency <= this are rejected. Default: 0.0 "
             "(strictly positive frequency).",
    )
    p.add_argument(
        "--max-frequency-mhz", type=float, default=None,
        help="Records with frequency > this are rejected. Default: None (no "
             "invented upper limit).",
    )
    p.add_argument(
        "--min-aoa-deg", type=float, default=0.0,
        help="Lower bound of canonical AoA range. Default: 0.0.",
    )
    p.add_argument(
        "--max-aoa-deg", type=float, default=360.0,
        help="Upper bound (exclusive) of canonical AoA range. Default: 360.0.",
    )
    p.add_argument(
        "--no-normalize-aoa", action="store_true",
        help="Do not fold signed AoA into [0, 360).",
    )
    p.add_argument(
        "--reject-duplicate-timestamps", action="store_true",
        help="Reject records whose ToA equals the previous record's ToA. "
             "Default: keep equal timestamps (separate observations).",
    )
    p.add_argument(
        "--min-duration-us", type=float, default=0.0,
        help="Minimum episode duration to enforce (reported). Default: 0.0.",
    )
    p.add_argument(
        "--max-duration-us", type=float, default=None,
        help="Maximum episode duration to enforce (reported). Default: None.",
    )
    p.add_argument(
        "--count-records", action="store_true",
        help="Pre-scan the inputs to count total records (extra pass).",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Print progress/summary info."
    )
    return p


def _emit_selector(config: SimConfig, ns: argparse.Namespace) -> None:
    config.emit_entries = not ns.no_entries
    config.emit_exits = not ns.no_exits
    config.emit_snapshots = not ns.no_snapshots


def main(argv: Optional[list] = None) -> int:
    parser = _build_parser()
    ns = parser.parse_args(argv)

    inputs = ns.input or ["OUTPUT FILES/output*.txt"]
    config = SimConfig(
        inputs=inputs,
        output_log=ns.output,
        snapshot_interval_us=ns.snapshot_interval_us,
        min_pw_us=ns.min_pw_us,
        nonfinite=ns.nonfinite,
        min_frequency_mhz=ns.min_frequency_mhz,
        max_frequency_mhz=ns.max_frequency_mhz,
        min_aoa_deg=ns.min_aoa_deg,
        max_aoa_deg=ns.max_aoa_deg,
        normalize_signed_aoa=not ns.no_normalize_aoa,
        reject_duplicate_timestamps=ns.reject_duplicate_timestamps,
        min_duration_us=ns.min_duration_us,
        max_duration_us=ns.max_duration_us,
        emit_entries=not ns.no_entries,
        emit_exits=not ns.no_exits,
        emit_snapshots=not ns.no_snapshots,
    )
    _emit_selector(config, ns)

    source = FileRecordSource(config.inputs, on_nonfinite=config.nonfinite)

    record_count = None
    if ns.count_records:
        record_count = sum(1 for _ in source)
        source = FileRecordSource(config.inputs, on_nonfinite=config.nonfinite)

    writer = TimelineWriter(ns.output, config=config)
    writer.write_meta(record_count=record_count)

    start = _time.perf_counter()
    env = RadioEnvironment(source, config, on_event=writer.on_event)
    env.run()
    elapsed = _time.perf_counter() - start

    writer.close()

    if ns.verbose:
        print(
            f"[sim_env] entries={env.total_entries} "
            f"exits={env.total_exits} snapshots={env.total_snapshots} "
            f"time_span_us={env.time_max_us - env.time_min_us:.3f} "
            f"elapsed_s={elapsed:.2f}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
