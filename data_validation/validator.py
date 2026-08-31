"""Reusable RF record validation engine.

This is the primary validation gate. It is used by:

1. the HDF5 -> TXT transform (rejects dirty records before they are written);
2. ``clean_output.py`` (migration of pre-validation ``output_*.txt`` files);
3. regression tests.

The validator decides record validity, normalizes AoA, tracks monotonic
timestamp state, and reports every issue without raising -- so a caller can
drop invalid records and keep streaming valid ones. It never *repairs* values
into different ones; it only normalizes AoA (a representational fold, not a
repair of dirty data) and reports the rest as issues.

Validation rules (see ValidationConfig for the tunable bounds):

* emitter id must be a finite integer >= 0;
* record width must be exactly 5 data features + 1 label;
* every numeric field must parse and be finite (reject NaN / +/-Inf);
* toa_us must be >= 0 and monotonically non-decreasing;
* frequency_mhz must be > min_frequency_mhz (default strictly positive);
* pulse_width_us must be > 0 (no zero/negative/instantaneous pulses);
* amplitude_db must be finite (no invented bounds);
* aoa_deg must be finite and, after normalization, in [0, 360).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .config import SEMANTIC_FIELDS, ValidationConfig

__all__ = [
    "ValidationIssue",
    "ValidationSummary",
    "RecordValidator",
    "validate_parsed_record",
    "normalize_aoa",
]

# Issue codes (stable, machine-checkable).
MISSING_FIELD = "missing_field"
BAD_WIDTH = "bad_width"
NON_FINITE = "non_finite"
PARSE_ERROR = "parse_error"
INVALID_EMITTER = "invalid_emitter"
NON_FINITE_EMITTER = "non_finite_emitter"
NEGATIVE_TOA = "negative_toa"
DEcreasing_TOA = "decreasing_toa"
DUPLICATE_TOA = "duplicate_toa"
PW_NOT_POSITIVE = "pw_not_positive"
FREQ_NOT_POSITIVE = "freq_not_positive"
FREQ_TOO_HIGH = "freq_too_high"
AOA_NOT_FINITE = "aoa_not_finite"
AOA_OUT_OF_RANGE = "aoa_out_of_range"
AMP_NOT_FINITE = "amp_not_finite"


@dataclass
class ValidationIssue:
    """A single validation finding for one record.

    Attributes:
        code: stable machine-readable issue code.
        message: human-readable description.
        record_number: 1-based index of the offending record within its file.
        fatal: whether the record must be dropped. Non-fatal issues are
            reported but the record is still retained (e.g. an AoA that needs
            normalization, or a detected-but-preserved duplicate timestamp).
    """

    code: str
    message: str
    record_number: Optional[int] = None
    fatal: bool = True


@dataclass
class ValidationSummary:
    """Per-episode validation result."""

    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    duration_us: float = 0.0
    first_valid_toa_us: Optional[float] = None
    last_valid_toa_us: Optional[float] = None
    issue_counts: dict = field(default_factory=dict)
    issues: List[ValidationIssue] = field(default_factory=list)

    def to_report(self) -> dict:
        """Serialize to the ``output_N.validation.json`` report structure."""
        return {
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
            "duration_us": self.duration_us,
            "first_valid_toa_us": self.first_valid_toa_us,
            "last_valid_toa_us": self.last_valid_toa_us,
            "issue_counts": dict(self.issue_counts),
            "issues": [
                {
                    "code": i.code,
                    "message": i.message,
                    "record_number": i.record_number,
                    "fatal": i.fatal,
                }
                for i in self.issues
            ],
        }


def normalize_aoa(aoa_deg: float, lo: float = 0.0, hi: float = 360.0) -> float:
    """Fold a signed angle into ``[lo, hi)`` using ``x % (hi - lo) + lo``.

    Examples: ``-10 -> 350``, ``-20 -> 340``, ``360 -> 0``.
    """
    import math as _m

    span = hi - lo
    if not _m.isfinite(aoa_deg):
        return aoa_deg
    return ((aoa_deg - lo) % span) + lo


class RecordValidator:
    """Stateful validator that streams records and tracks ordering state.

    Instantiate one per episode (file) so decreasing/duplicate-timestamp
    detection sees the previous record *within that episode*.
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.summary = ValidationSummary()
        self._prev_toa: Optional[float] = None
        self._any_valid = False

    def validate(
        self,
        values: Sequence[float],
        label,
        record_number: Optional[int] = None,
    ) -> Optional[dict]:
        """Validate one parsed record.

        Args:
            values: the 5-element data vector ``[toa, freq, pw, amp, aoa]``
                (which may contain non-finite / non-numeric entries).
            label: the raw emitter-id label (will be coerced / checked).
            record_number: 1-based record index for reporting.

        Returns:
            The record as an ordered dict keyed by SEMANTIC_FIELDS when the
            record is valid (with AoA normalized), else ``None`` to signal the
            record must be dropped. Issues are accumulated on ``self.summary``
            either way.
        """
        self.summary.total_records += 1
        issues: List[ValidationIssue] = []
        cfg = self.config

        # --- schema / width -------------------------------------------------
        if values is None or len(values) != 5:
            issues.append(
                ValidationIssue(BAD_WIDTH,
                                f"expected 5 data features, got "
                                f"{0 if values is None else len(values)}",
                                record_number, fatal=True)
            )
            return self._finish_invalid(issues)

        toa_raw, freq_raw, pw_raw, amp_raw, aoa_raw = values

        # --- parse + non-finite checks --------------------------------------
        parsed: List[Optional[float]] = []
        for name, raw in (("toa_us", toa_raw), ("frequency_mhz", freq_raw),
                          ("pulse_width_us", pw_raw), ("amplitude_db", amp_raw),
                          ("aoa_deg", aoa_raw)):
            try:
                v = float(raw)
            except (TypeError, ValueError):
                issues.append(ValidationIssue(
                    PARSE_ERROR, f"could not parse {name}={raw!r}",
                    record_number, fatal=True))
                parsed.append(None)
                continue
            if not math.isfinite(v):
                issues.append(ValidationIssue(
                    NON_FINITE, f"non-finite {name}={v!r}",
                    record_number, fatal=True))
                parsed.append(None)
                continue
            parsed.append(v)

        if any(p is None for p in parsed):
            return self._finish_invalid(issues)

        toa, freq, pw, amp, aoa = parsed

        # --- emitter id ------------------------------------------------------
        try:
            emitter = int(label)
        except (TypeError, ValueError):
            issues.append(ValidationIssue(
                INVALID_EMITTER, f"invalid emitter id {label!r}",
                record_number, fatal=True))
            return self._finish_invalid(issues)
        if not math.isfinite(emitter):
            issues.append(ValidationIssue(
                NON_FINITE_EMITTER, f"non-finite emitter id {label!r}",
                record_number, fatal=True))
            return self._finish_invalid(issues)
        if emitter < 0:
            issues.append(ValidationIssue(
                INVALID_EMITTER, f"emitter id {emitter} < 0",
                record_number, fatal=True))
            return self._finish_invalid(issues)

        # --- ToA -------------------------------------------------------------
        if toa < 0:
            issues.append(ValidationIssue(
                NEGATIVE_TOA, f"toa {toa} < 0", record_number, fatal=True))
            return self._finish_invalid(issues)
        if self._prev_toa is not None:
            if toa < self._prev_toa:
                issues.append(ValidationIssue(
                    DEcreasing_TOA,
                    f"toa {toa} < previous {self._prev_toa} (not non-decreasing)",
                    record_number, fatal=True))
                return self._finish_invalid(issues)
            if toa == self._prev_toa:
                if cfg.reject_duplicate_timestamps:
                    issues.append(ValidationIssue(
                        DUPLICATE_TOA,
                        f"duplicate toa {toa} and reject_duplicate_timestamps=True",
                        record_number, fatal=True))
                    return self._finish_invalid(issues)
                issues.append(ValidationIssue(
                    DUPLICATE_TOA, f"duplicate toa {toa} preserved",
                    record_number, fatal=False))

        # --- PW ---------------------------------------------------------------
        if pw <= 0:
            issues.append(ValidationIssue(
                PW_NOT_POSITIVE, f"pulse_width {pw} <= 0", record_number, fatal=True))
            return self._finish_invalid(issues)

        # --- frequency ---------------------------------------------------------
        if cfg.min_frequency_mhz is not None and freq <= cfg.min_frequency_mhz:
            issues.append(ValidationIssue(
                FREQ_NOT_POSITIVE,
                f"frequency {freq} <= min_frequency_mhz {cfg.min_frequency_mhz}",
                record_number, fatal=True))
            return self._finish_invalid(issues)
        if cfg.max_frequency_mhz is not None and freq > cfg.max_frequency_mhz:
            issues.append(ValidationIssue(
                FREQ_TOO_HIGH,
                f"frequency {freq} > max_frequency_mhz {cfg.max_frequency_mhz}",
                record_number, fatal=True))
            return self._finish_invalid(issues)

        # --- amplitude (finite only, no invented bounds) -----------------------
        if not math.isfinite(amp):
            issues.append(ValidationIssue(
                AMP_NOT_FINITE, f"amplitude {amp} not finite", record_number, fatal=True))
            return self._finish_invalid(issues)

        # --- AoA (normalize, then range-check) ----------------------------------
        aoa_norm = aoa
        if not math.isfinite(aoa):
            issues.append(ValidationIssue(
                AOA_NOT_FINITE, f"aoa {aoa} not finite", record_number, fatal=True))
            return self._finish_invalid(issues)
        if cfg.normalize_signed_aoa:
            aoa_norm = normalize_aoa(aoa, cfg.min_aoa_deg, cfg.max_aoa_deg)
        if not (cfg.min_aoa_deg <= aoa_norm < cfg.max_aoa_deg):
            issues.append(ValidationIssue(
                AOA_OUT_OF_RANGE,
                f"aoa {aoa_norm} outside [{cfg.min_aoa_deg}, {cfg.max_aoa_deg})",
                record_number, fatal=True))
            return self._finish_invalid(issues)

        # --- accept -------------------------------------------------------------
        self._prev_toa = toa
        self.summary.valid_records += 1
        self.summary.issues.extend(issues)
        self._count_issues(issues)
        if self.summary.first_valid_toa_us is None:
            self.summary.first_valid_toa_us = toa
        self.summary.last_valid_toa_us = toa
        self.summary.duration_us = (
            (self.summary.last_valid_toa_us - (self.summary.first_valid_toa_us or 0.0))
            if self.summary.first_valid_toa_us is not None else 0.0
        )
        return {
            "toa_us": toa,
            "frequency_mhz": freq,
            "pulse_width_us": pw,
            "amplitude_db": amp,
            "aoa_deg": aoa_norm,
            "emitter_id": emitter,
        }

    # ------------------------------------------------------------------ helpers

    def _finish_invalid(self, issues: List[ValidationIssue]) -> None:
        self.summary.invalid_records += 1
        self.summary.issues.extend(issues)
        self._count_issues(issues)
        return None

    def _count_issues(self, issues: Sequence[ValidationIssue]) -> None:
        for i in issues:
            if i.fatal:
                self.summary.issue_counts[i.code] = (
                    self.summary.issue_counts.get(i.code, 0) + 1
                )

    def enforce_duration(self) -> None:
        """After streaming, mark the episode invalid if its duration is outside
        the configured bounds (reported only -- records are not re-dropped)."""
        cfg = self.config
        if self.summary.first_valid_toa_us is None:
            return
        dur = self.summary.duration_us
        if cfg.min_duration_us is not None and dur < cfg.min_duration_us:
            self.summary.issues.append(
                ValidationIssue("duration_too_short",
                                f"duration {dur} < min_duration_us {cfg.min_duration_us}",
                                fatal=False))
        if cfg.max_duration_us is not None and dur > cfg.max_duration_us:
            self.summary.issues.append(
                ValidationIssue("duration_too_long",
                                f"duration {dur} > max_duration_us {cfg.max_duration_us}",
                                fatal=False))


def validate_parsed_record(
    values: Sequence[float],
    label,
    record_number: Optional[int] = None,
    config: Optional[ValidationConfig] = None,
) -> Optional[dict]:
    """Single-shot validation (no ordering state) of one parsed record.

    Prefer :class:`RecordValidator` for episodes so decreasing/duplicate ToA
    detection is correct, but this is convenient for one-off checks/tests.
    """
    v = RecordValidator(config)
    return v.validate(values, label, record_number)
