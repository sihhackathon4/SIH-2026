"""RadioWave Stream Simulation Environment.

Reads RF emitter records from the OUTPUT FILES and simulates a real-world
radiowave stream in which every radiopulse *enters* the environment at its
Time-of-Arrival and stays *active* for exactly its Pulse Width before it
*disappears completely*.

The environment is event-driven (a sweep line over entry/exit events), streams
records from disk in an incremental / continuous-ingestion fashion, and emits a
Machine-Learning-scheduler-friendly NDJSON event log.
"""

from .config import SimConfig
from .ingest import RecordSource, FileRecordSource, parse_record_line, PulseRecord
from .environment import RadioEnvironment, ActivePulse, SimulationEvent
from .timeline_writer import TimelineWriter
from .timeline_reader import iter_events, read_meta_only, rebuild_frames

__all__ = [
    "SimConfig",
    "RecordSource",
    "FileRecordSource",
    "parse_record_line",
    "PulseRecord",
    "RadioEnvironment",
    "ActivePulse",
    "SimulationEvent",
    "TimelineWriter",
    "iter_events",
    "read_meta_only",
    "rebuild_frames",
]

__version__ = "0.1.0"
