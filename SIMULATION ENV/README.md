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
{"event":"meta","schema_version":1,"feature_order":["toa_us","frequency_mhz","pulse_width_us","amplitude_db","aoa_deg"],"time_unit":"microseconds",...}

{"event":"entry","time_us":2163460.0,"active_count":1,
 "pulse":{"toa_us":2163460.0,"frequency_mhz":9658.665,"pulse_width_us":0.0114,"amplitude_db":28.24,"aoa_deg":-152.13,"pulse_id":0,"emitter_id":2,"exit_us":2163460.0114}}

{"event":"snapshot","time_us":2163460.0,"active_count":2,
 "active_pulses":[ ...every pulse alive at that instant... ]}

{"event":"exit","time_us":2163460.0114,"active_count":0,"pulse":{...},"pulse_id":0}
```

Event types:

| `event`     | Description                                                               |
|-------------|---------------------------------------------------------------------------|
| `meta`      | Schema version, feature order/units, config.                              |
| `entry`     | A pulse **entered** at `time_us`; embeds the full feature vector + `emitter_id`. |
| `exit`      | A pulse **left entirely** at `time_us`; embeds the same fields.            |
| `snapshot`  | Periodic ground-truth frame: **all** pulses active at `time_us`.           |

This is intentionally streaming-friendly: an ML scheduler can read the log line
by line / over a pipe, in time order, without buffering, and can train on either
the fine-grained `entry`/`exit` events or the dense `snapshot` frames.

---

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
| `--min-pw-us`               | Treat pulse widths below this as instantaneous                |
| `-v, --verbose`             | Print a run summary to stderr                                 |

---

## Files

| File                    | Responsibility                                    |
|-------------------------|---------------------------------------------------|
| `sim_env/ingest.py`     | Streaming / continuous-ingestion record parser    |
| `sim_env/environment.py`| Event-driven sweep + active-pulse tracking        |
| `sim_env/timeline_writer.py` | NDJSON event-log writer + self-describing meta |
| `sim_env/timeline_reader.py` | Stream the NDJSON log back for an ML scheduler |
| `sim_env/config.py`     | Tuning parameters and field/unit definitions      |
| `sim_env/cli.py`        | Command-line entry point                          |
| `run.py`                | Convenience launcher                              |

---

## Notes

* Pulse widths that are negative or below `--min-pw-us` are treated as
  **instantaneous** (enter at ToA and exit immediately), matching the few
  non-physical values present in the source data.
* `sim_env/timeline_reader.py` provides `iter_events()`, a streaming reader an
  ML scheduler can use today to consume the log.
