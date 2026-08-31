# RadioWave Stream Simulation Environment

Simulates a **real-world radiowave stream** from the RF emitter record files in
`OUTPUT FILES/`. Every radiopulse

1. **enters** the environment at its Time-of-Arrival (`toa_us`),
2. stays **active** for exactly its Pulse Width (`pulse_width_us`),
3. then **disappears completely** at `toa_us + pulse_width_us`.

The result is a continuous, machine-readable event timeline that a downstream
**ML scheduler** can consume to learn / predict emitter activity.

---

## Data semantics (per record)

A source line looks like:

```
record_1: data=[570614.875, 9227.7236328125, 0.01109, 61.937, -158.533], label=68
```

| Position | Field           | Unit            | Meaning                          |
|----------|-----------------|-----------------|----------------------------------|
| 1        | `toa_us`        | microseconds    | Time of Arrival                 |
| 2        | `frequency_mhz` | megahertz       | Carrier frequency               |
| 3        | `pulse_width_us`| microseconds    | Pulse width (active duration)   |
| 4        | `amplitude_db`  | relative dB     | Amplitude                       |
| 5        | `aoa_deg`       | degrees         | Angle of arrival                |
| label    | `emitter_id`    | —               | Ground-truth emitter identity   |

---

## Design

### Continuous ingestion (memory-safe streaming)
`FileRecordSource` opens every file lazily and reads it **line by line**, then
**k-way merges** the file streams in ascending Time-of-Arrival order. It never
loads the full dataset into memory, so it can ingest data continuously even
though the corpus totals ~5.7 M records (~10.6 s of simulated time).

### Event-driven sweep line
`RadioEnvironment` advances its clock to the *next* event (pulse entry, pulse
exit, or a periodic snapshot tick) rather than stepping through wall-clock time.
A min-heap of `(exit_us, seq, pulse_id)` plus a streaming arrival pointer keep
the active set exact:

* at `t`: every arrival with `toa_us <= t` enters,
* every pulse whose `exit_us <= t` leaves **completely**,
* optionally, periodic **snapshot** frames describe the whole active scene.

Sub-microsecond pulse widths are therefore handled exactly, and runtime scales
with the number of events — not with the 10.6 s time span.

---

## Output format (NDJSON event log)

One self-describing JSON object per line (the first line is always `meta`):

```json
{"event":"meta","schema_version":2,"data_validated":true,"feature_order":["toa_us","frequency_mhz","pulse_width_us","amplitude_db","aoa_deg"],"label":"emitter_id","aoa_range":[0.0,360.0],...}

{"event":"entry","time_us":2163460.0,"active_count":1,
 "pulse":{"toa_us":2163460.0,"frequency_mhz":9658.665,"pulse_width_us":0.0114,"amplitude_db":28.24,"aoa_deg":152.13,"pulse_id":0,"emitter_id":2,"exit_us":2163460.0114}}

{"event":"snapshot","time_us":2163460.0,"active_count":2,
 "active_pulses":[ ...every pulse alive at that instant... ]}

{"event":"exit","time_us":2163460.0114,"active_count":0,"pulse":{...},"pulse_id":0}
```

Event types:

| `event`     | Description                                                               |
|-------------|---------------------------------------------------------------------------|
| `meta`      | Schema version (2), `data_validated`, feature order/units, `aoa_range`, config. |
| `entry`     | A pulse **entered** at `time_us`; embeds the full feature vector + `emitter_id`. |
| `exit`      | A pulse **left entirely** at `time_us`; embeds the same fields.            |
| `snapshot`  | Periodic ground-truth frame: **all** pulses active at `time_us`.           |

This is intentionally streaming-friendly: an ML scheduler can read the log line
by line / over a pipe, in time order, without buffering, and can train on either
the fine-grained `entry`/`exit` events or the dense `snapshot` frames.

---

## Validated RF data pipeline

Dirty records are removed **before** the simulation environment (the validation
boundary is upstream, not the NDJSON writer/reader).

```
HDF5 → transform → DATA VALIDATION → output_*.txt → FileRecordSource
      → RadioEnvironment → TimelineWriter → NDJSON → ML scheduler / dataset
```

Validation rejects (never silently repairs):

* missing fields / wrong record width / unparsable numerics;
* NaN, +Inf, -Inf in any field (incl. emitter id);
* `ToA < 0` and decreasing ToA (input must be monotonically non-decreasing);
* `PW <= 0` (never turned into a zero-duration pulse);
* `Frequency <= 0` (no invented maximum by default);
* invalid emitter id (must float-parse to an integer `>= 0`).

It **normalizes** signed AoA into `[0, 360)` (`-10 -> 350`, `360 -> 0`) and
**preserves** equal timestamps by default (legitimate simultaneous emitters) —
duplicate ToAs are detected and reported, and only made fatal when
`reject_duplicate_timestamps=True`. Episode duration is reported and can be
bounded via `min/max_duration_us`.

The `data_validation/` package at the repo root provides:

| Module              | Responsibility                                         |
|---------------------|--------------------------------------------------------|
| `config.py`         | `ValidationConfig` — the physical rules                |
| `validator.py`      | `RecordValidator`, `validate_parsed_record`, `normalize_aoa` — reusable gate |
| `clean_output.py`   | Migrate old `output_*.txt` (validate → drop → renumber → rebuild counts → report) |
| `__main__.py`       | CLI (`--validate`, `--clean`, report dirs)            |

Cleaner (one-off migration only — fresh transformations are already validated):

```bash
python -m data_validation --clean \
  --output-dir "VALIDATED OUTPUT FILES" --report-dir "validation_reports" \
  "OUTPUT FILES/output*.txt"
```

Validate-only (writes `validation_reports/*.validation.json` without rewriting):

```bash
python -m data_validation --validate "OUTPUT FILES/output*.txt"
```


## Usage

Requires Python 3.9+ with only the standard library (plus `numpy` optionally
for downstream analysis).

```bash
# Run the full sweep over all OUTPUT FILES -> NDJSON timeline
python run.py \
  --input "OUTPUT FILES/*.txt" \
  --output "OUTPUT FILES/stream_timeline.ndjson" \
  --snapshot-interval-us 5000 \
  --verbose
```

### API use (no file output)

```python
from sim_env import SimConfig, FileRecordSource, RadioEnvironment

cfg = SimConfig(inputs=["OUTPUT FILES/*.txt"], snapshot_interval_us=5000)
src = FileRecordSource(cfg.inputs)
env = RadioEnvironment(src, cfg, on_event=lambda ev: print(ev.event_type, ev.time_us))
env.run()
print(env.total_entries, env.total_exits, env.total_snapshots)
```

### CLI options

| Flag                        | Meaning                                                        |
|-----------------------------|----------------------------------------------------------------|
| `-i, --input`               | Record file path/glob (repeatable; default `OUTPUT FILES/*.txt`) |
| `-o, --output`              | NDJSON output path (default: stdout)                          |
| `-s, --snapshot-interval-us`| Snapshot period in microseconds (`None` disables)             |
| `--no-entries/--no-exits/--no-snapshots` | Suppress that event type                      |
| `--min-pw-us`               | Reserved (invalid PW records are rejected, not clamped)       |
| `--nonfinite`               | `allow`/`drop`/`raise` for `inf`/`nan`/`PW<=0` (default `drop`) |
| `--min/max-frequency-mhz`   | Frequency bounds (defaults: strictly positive, no max)        |
| `--min/max-aoa-deg`         | Canonical AoA range (default `0..360`)                         |
| `--no-normalize-aoa`        | Do not fold signed AoA into `[0, 360)`                         |
| `--reject-duplicate-timestamps` | Reject equal shared ToA (default: preserve + report)        |
| `--min/max-duration-us`     | Episode duration bounds (reported)                            |
| `-v, --verbose`             | Print a run summary to stderr                                 |

---

## Building an ML deinterleaving dataset

Beyond the raw event stream, the package ships a deinterleaving-focused data
layer. Each RF record file is treated as an **independent episode** (own emitter
population), so episodes are never spliced together and never split across
train/val/test.

### Deterministic episode split
```python
from sim_env import split_files
import glob
paths = sorted(glob.glob("OUTPUT FILES/output*.txt"))
splits = split_files(paths, val_fraction=0.15, test_fraction=0.15)
# -> {"train": [Path...], "val": [...], "test": [...]}
```
Split is keyed on each file's stem (e.g. `output_7`) via SHA-256, so it is
reproducible across runs and machines.

### Fixed-length pulse-sequence windows
```python
from sim_env import iter_episode_windows, FeatureStats, FileRecordSource

stats = FeatureStats.fit(FileRecordSource(splits["train"]))  # fit on train ONLY

for split in ("train", "val", "test"):
    windows = list(iter_episode_windows(splits[split], window_len=64,
                                        stride=64, nonfinite="drop"))
    # windows[i].features : [64][5], .emitter_ids : [64], .source_id : episode
```
Windows are slices of interleaved pulse-descriptor-word (PDW) vectors; each
row's `emitter_id` is the grouping target for metric / triplet-loss deinterleave
training. `normalized_features(stats)` applies the train-fitted z-score stats.

### Non-finite / invalid records
The legacy corpus can contain `inf`/`nan` in
`frequency_mhz`/`amplitude_db`/`aoa_deg` and occasional `PW <= 0`. The
`nonfinite` policy (also `--nonfinite`) selects the behaviour, defaulting to
**`drop`** so invalid data is not silently passed on:

| policy   | behaviour                                                              |
|----------|------------------------------------------------------------------------|
| `drop`   | skip the offending record, keep building (default everywhere)         |
| `raise`  | fail loudly on the first bad record                                   |
| `allow`  | keep the values (legacy; not recommended)                             |

`FeatureStats.fit` also skips non-finite values per feature so a stray bad value
can never turn its statistics into `nan`. Invalid `PW <= 0` records are always
rejected (never turned into zero-duration pulses).

---

## Files

| File                    | Responsibility                                    |
|-------------------------|---------------------------------------------------|
| `sim_env/ingest.py`     | Streaming / continuous-ingestion record parser    |
| `sim_env/environment.py`| Event-driven sweep + active-pulse tracking        |
| `sim_env/timeline_writer.py` | NDJSON event-log writer + self-describing meta |
| `sim_env/timeline_reader.py` | Stream the NDJSON log back for an ML scheduler |
| `sim_env/config.py`     | Tuning parameters, field/unit definitions, `FeatureStats` |
| `sim_env/splits.py`     | Deterministic episode-level train/val/test split |
| `sim_env/dataset.py`    | Windowed pulse-sequence dataset + live collector |
| `sim_env/cli.py`        | Command-line entry point                          |
| `run.py`                | Convenience launcher                              |

---

## Notes

* Invalid records are rejected at the validation layer upstream. A record with
  `PW <= 0`, `NaN`/`Inf`, `ToA < 0`, decreasing ToA, or `Frequency <= 0` never
  enters the environment and never produces a zero-duration `entry`+`exit` pair.
* Signed AoA is normalized to `[0, 360)` both at the validation gate and, as a
  defense-in-depth, in the receiver environment.
* `sim_env/timeline_reader.py` provides `iter_events()`, a streaming reader an
  ML scheduler can use today to consume the log line by line.
