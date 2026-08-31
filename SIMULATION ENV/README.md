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
{"event":"meta","schema_version":1,"data_validated":true,"feature_order":["toa_us","frequency_mhz","pulse_width_us","amplitude_db","aoa_deg"],"label":"emitter_id","aoa_range":[0.0,360.0],...}

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

## Receiver Component (`sim_env/receiver/`)

The environment describes the **RF world** (the full event timeline). The
**`SieveReceiver`** describes what a *limited-bandwidth* receiver can actually
intercept. It sits between the environment and a future ML scheduler:

```
VALIDATED RF DATA
      |
      v
RadioEnvironment
      |
   RF events (existing NDJSON schema v2, unchanged)
      |
      v
+----------------+
| SieveReceiver  |   IBW  *  TIME  *  DETECTION
+----------------+
      |
      v
Receiver Observation   (future ML scheduler's observation space)
      |
      v
FUTURE ML SCHEDULER -- NOT IMPLEMENTED in this phase
```

### Architectural note
The **ML scheduler is NOT implemented in this phase.** The receiver uses a
**static, deterministic interception/scanning policy**. It is designed so a
future scheduler can replace *only* the static control policy
(via `get_observation()` / `apply_action()`) without rewriting the receiver.

- `RadioEnvironment` = RF world / event timeline (unchanged).
- `SieveReceiver` = receiver limitations and observations.
- `ReceiverObservation` = what the receiver can observe (the scheduler's future
  observation space) — not every RF event is exposed, only what is observable.

### Units
All receiver frequencies are **MHz internally** (matching the repository's
`frequency_mhz`); all times are **microseconds**. Helpers `to_hz()` / `to_ghz()`
provide explicit conversion only where required — the receiver never silently
treats `3199.19 MHz` as `3199.19 Hz`.

- `total_bandwidth_mhz` — full available spectrum (e.g. `18e3` MHz = 18 GHz)
- `ibw_mhz` — instantaneous bandwidth of the observation window
- `frequency_step_mhz` — step size for scanning
- `dwell_time_us` — observation interval
- `center_frequency_mhz` — current tuned center
- `current_time_us` — receiver simulation clock

### Configuration validation
Invalid configuration is rejected at construction with a clear `ValueError`
(`ReceiverConfigError`), never silently repaired:

```
total_bandwidth > 0        ibw > 0            ibw <= total_bandwidth
frequency_step > 0         dwell_time > 0     finite, sane thresholds
```

### Frequency window / ICC rule
`get_frequency_window()` returns `(center - ibw/2, center + ibw/2)`. The window
is **inclusive at both edges** (`lower <= f <= upper`) consistently across
synthetic-spectrum mode, pulse/event mode, and tests.

- Legal center range is `[ibw/2, total_bandwidth - ibw/2]`; tuning/stepping is
  clamped to this range so the window never exceeds the total spectrum.
- `tune(f)` clips (does not raise) and raises on non-finite input.
- `step_up()` / `step_down()` move by `frequency_step` with legal-band clipping
  and set the scan direction.

### Time model / dwell
`dwell()` advances `current_time_us` by `dwell_time_us`. `perform_dwell(...)`
observes the window and then advances time. Time matters for visibility: a pulse
that ended before observation is **not** detected; a pulse active during the
dwell is; pulses beginning or ending during a dwell are handled via the exact
half-open overlap rule `toa_us <= current_time_us < exit_us`.

### Detection model
The repository's amplitude convention is **relative dB** and *non-positive*
(e.g. `-121.8`). The original prototype's `peak >= detection_threshold` (with a
positive `5.0`) is therefore **incompatible** with real data. The receiver uses
two separate, documented mechanisms:

1. **`detection_threshold_db` (sensitivity floor, dB)** — used for real RF
   pulse data. A pulse's amplitude is observed when
   `amplitude_db >= detection_threshold_db`. Real weak signals (≈ `-100` to
   `-120` dB) clear a `-140` dB floor.
2. **`spectrum_threshold` (positive normalized power)** — used *only* for the
   synthetic NumPy-spectrum path, preserving the prototype's
   `peak_power >= threshold` behavior for unit tests.

No physical dB↔power conversion is fabricated; the model is deliberately simple
and deterministic.

### Visibility & detection
A pulse is detected only when **all** of the following hold (see §14):

```
frequency inside current window (inclusive edges)
AND current_time inside [toa, toa + pw)   (half-open time overlap)
AND amplitude_db >= detection_threshold_db
```

`process_pulse(pulse)` / `process_event(event)` accept the repository's actual
pulse representation (a dict from NDJSON, or an `ActivePulse` / `PulseRecord`
object). They return a structured `DetectionObservation` on detection, else
`None`, and never advance time or change frequency. Multiple simultaneous pulses
with equal ToA remain **separate** observations — never merged.

### Time-aware pulse buffer & live environment connection
The receiver keeps a **time-aware pulse buffer** (`_pulses`) of every pulse it
has *learned about* from the environment up to the current time:

- `add_pulse(pulse)` buffers a pulse the instant its `entry` event is announced.
- `remove_pulse(id)` / `advance(t)` clear pulses once their `exit` time has
  passed (and only then).
- A buffered pulse is observable during a dwell `[t0, t0 + dwell)` via **interval
  overlap**: `toa < t0 + dwell` **and** `exit > t0`. This correctly distinguishes
  pulses that begin/end during a dwell from those entirely before or after it.
- The dwell interval is tracked explicitly on the receiver
  (`dwell_start_us`, `dwell_end_us`) and exposed on every `ReceiverObservation`
  as `dwell_interval_us = [start, end)`, so the receiver can always answer
  *"which buffered pulses overlap my current dwell?"*.

A first-class bridge connects the receiver to the live environment stream:

```python
from sim_env import (SieveReceiver, RadioEnvironment, SimConfig,
                     FileRecordSource, attach_receiver)

rx = SieveReceiver(total_bandwidth=18e3, ibw=1e3)
rx.tune(3200.0)                     # one-time configuration, NOT per-pulse retune
env = RadioEnvironment(FileRecordSource(["OUTPUT FILES/output_134.txt"]), SimConfig())
attach_receiver(env, rx)            # env's on_event -> receiver buffer (no manual loop)
env.run()
obs = rx.scan_once()                # receiver dwells on its own window/clock
```

The bridge registers a callback on the environment's `on_event` list so live
`entry`/`exit` events propagate into the receiver buffer **without** any manual
`receiver.tune(pulse_frequency)` / `receiver.current_time = pulse_time` inside
the event loop. The receiver keeps its own deterministic clock and scan position
— it never retunes to an arriving pulse and never learns a pulse before its entry
event (no future-information leakage).

### Defensive input handling
The receiver defensively rejects impossible input rather than silently repairing
it: `NaN`/`Inf` frequency, `NaN`/`Inf` PW, `PW <= 0`, `NaN` ToA, and negative ToA
produce **no** detection (and never a fabricated zero-duration pulse).

### Structured output

`DetectionObservation` (per-pulse, receiver-observable fields; `pulse_id` and
`emitter_id` are passthrough ground-truth, **not** receiver measurements):
```json
{"detected": true, "time_us": 2068786.625, "frequency_mhz": 3199.19,
 "pulse_width_us": 0.49, "amplitude_db": -121.81, "aoa_deg": 84.26,
 "pulse_id": 0, "center_frequency_mhz": 3199.19, "emitter_id": null}
```

`get_observation()` returns the scheduler-facing observation space:
```json
{"time_us": 100.0, "center_frequency_mhz": 8000.0, "ibw_mhz": 1000.0,
 "dwell_time_us": 100.0, "dwell_interval_us": [100.0, 200.0],
 "window_mhz": [7500.0, 8500.0], "detections": [...]}
```

### Static scanning API
`scan_once(...)` runs one deterministic cycle —
`observe → detect → record → dwell → step` — with no hidden state; `scan(n)`
repeats it. `reset()` clears all state (time, center, detections, scan
position) so no state leaks between runs. The scan sequence is deterministic.

### Future ML scheduler interface
The receiver already exposes the operations a future scheduler will drive, but
is fully usable without any scheduler classes:
```
observation = receiver.get_observation()     # scheduler reads this
action = scheduler.choose_action(observation)  # NOT implemented yet
receiver.apply_action(action)                  # TUNE / STEP_UP / STEP_DOWN / DWELL
```
`apply_action` accepts `"TUNE"` (also `"STEP_UP"`, `"STEP_DOWN"`, `"DWELL"`)
or a set of target state, raising `ValueError` on unknown actions. Because the
control loop is a thin, exposed seam, the static policy can later be swapped for
an ML policy without touching the receiver internals.

### Receiver files
| File                                   | Responsibility                                    |
|----------------------------------------|---------------------------------------------------|
| `sim_env/receiver/models.py`           | `DetectionObservation`, `ReceiverObservation`     |
| `sim_env/receiver/sieve_receiver.py`   | `SieveReceiver`, `ReceiverConfigError`, `to_hz`/`to_ghz`, time-aware buffer |
| `sim_env/receiver/adapter.py`          | `RadioReceiverBridge`, `attach_receiver` (live env → receiver) |
| `sim_env/receiver/__init__.py`         | Public receiver API re-exports                    |

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
| `sim_env/receiver/*`    | Receiver component (see Receiver section above)   |

---

## Notes

* Invalid records are rejected at the validation layer upstream. A record with
  `PW <= 0`, `NaN`/`Inf`, `ToA < 0`, decreasing ToA, or `Frequency <= 0` never
  enters the environment and never produces a zero-duration `entry`+`exit` pair.
* Signed AoA is normalized to `[0, 360)` both at the validation gate and, as a
  defense-in-depth, in the receiver environment.
* `sim_env/timeline_reader.py` provides `iter_events()`, a streaming reader an
  ML scheduler can use today to consume the log line by line.
