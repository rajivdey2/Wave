"""Shared data contracts — the single source of truth (ARCHITECTURE.md §2).

Every function returning signal metadata returns these types, never ad-hoc dicts.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum


PIPELINE_VERSION = "0.1.0-day1"


class ConfidenceState(str, Enum):
    DETECTED = "detected"      # direct measurement, tight error bounds
    ESTIMATED = "estimated"    # standard algorithm, known error bounds
    INFERRED = "inferred"      # ML output above threshold
    UNKNOWN = "unknown"        # below SNR/quality floor — refused to guess


@dataclass
class FieldValue:
    value: float | str | None
    state: ConfidenceState
    confidence: float | None  # 0-1, only for ESTIMATED/INFERRED
    evidence: str             # always present

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "state": self.state.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }

    @staticmethod
    def unknown(evidence: str) -> "FieldValue":
        return FieldValue(value=None, state=ConfidenceState.UNKNOWN,
                          confidence=None, evidence=evidence)

    @staticmethod
    def detected(value: float | str, evidence: str) -> "FieldValue":
        return FieldValue(value=value, state=ConfidenceState.DETECTED,
                          confidence=1.0, evidence=evidence)


@dataclass
class SignalDetection:
    id: str
    t_start_s: float
    t_end_s: float
    f_low_hz: float
    f_high_hz: float
    center_freq_hz: FieldValue
    bandwidth_hz: FieldValue
    snr_db: FieldValue
    symbol_rate_hz: FieldValue
    modulation: FieldValue
    modulation_evidence_method: str  # "classical" | "ml" | "both"
    demod_available: bool = False
    fec_identification: FieldValue | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("center_freq_hz", "bandwidth_hz", "snr_db",
                  "symbol_rate_hz", "modulation"):
            fv: FieldValue = getattr(self, k)
            d[k] = fv.to_dict()
        d["fec_identification"] = (
            self.fec_identification.to_dict()
            if self.fec_identification else None
        )
        return d


@dataclass
class CaptureReport:
    source_file: str
    format: str                      # "iq_cf32" | "iq_int16" | "wav"
    sample_rate_hz: float
    duration_s: float
    noise_floor_db: float
    dynamic_range_db: float
    detections: list[SignalDetection] = field(default_factory=list)
    pipeline_version: str = PIPELINE_VERSION

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "format": self.format,
            "sample_rate_hz": self.sample_rate_hz,
            "duration_s": self.duration_s,
            "noise_floor_db": self.noise_floor_db,
            "dynamic_range_db": self.dynamic_range_db,
            "detections": [d.to_dict() for d in self.detections],
            "pipeline_version": self.pipeline_version,
        }
