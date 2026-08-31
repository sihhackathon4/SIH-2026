"""RF Data Validation Pipeline.

Validates and cleans RF records *before* they reach the simulation
environment / ML pipeline, enforcing the canonical semantic schema::

    (toa_us, frequency_mhz, pulse_width_us, amplitude_db, aoa_deg, emitter_id)

Public API:
    - ``ValidationConfig`` -- the physical rules (bounds, AoA normalization,
      duplicate-timestamp policy, duration bounds).
    - ``RecordValidator`` -- streaming stateful validator (per-episode); returns
      clean records or ``None``, accumulating a ``ValidationSummary``.
    - ``validate_parsed_record`` -- single-shot validation.
    - ``clean_output_file`` / ``clean_output_files`` / ``clean_output_dir`` --
      migrate pre-validation ``output_*.txt`` files (validate, drop, renumber,
      rebuild counts, write reports).
    - ``normalize_aoa`` -- fold signed angles into ``[0, 360)``.
"""

from .config import (
    SEMANTIC_FIELDS,
    TXT_DATA_FIELDS,
    HDF5_SOURCE_FIELDS,
    ValidationConfig,
)
from .validator import (
    RecordValidator,
    ValidationIssue,
    ValidationSummary,
    validate_parsed_record,
    normalize_aoa,
)
from .clean_output import (
    parse_output_line,
    clean_output_file,
    clean_output_files,
    clean_output_dir,
)

__all__ = [
    "SEMANTIC_FIELDS",
    "TXT_DATA_FIELDS",
    "HDF5_SOURCE_FIELDS",
    "ValidationConfig",
    "RecordValidator",
    "ValidationIssue",
    "ValidationSummary",
    "validate_parsed_record",
    "normalize_aoa",
    "parse_output_line",
    "clean_output_file",
    "clean_output_files",
    "clean_output_dir",
]

__version__ = "0.1.0"
