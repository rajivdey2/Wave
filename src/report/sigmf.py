"""SigMF annotation writer (ARCHITECTURE.md §3.2)."""
from __future__ import annotations

import json
from pathlib import Path

from src.types import CaptureReport


def to_sigmf(report: CaptureReport) -> dict:
    anns = []
    for d in report.detections:
        anns.append({
            "core:sample_start": int(d.t_start_s * report.sample_rate_hz),
            "core:sample_count": int(max(1, (d.t_end_s - d.t_start_s) * report.sample_rate_hz)),
            "core:freq_lower_edge": d.f_low_hz,
            "core:freq_upper_edge": d.f_high_hz,
            "ntro_sih:modulation": d.modulation.value,
            "ntro_sih:confidence_state": d.modulation.state.value,
            "ntro_sih:evidence": d.modulation.evidence,
        })
    return {
        "global": {
            "core:datatype": "cf32_le" if report.format == "iq_cf32" else report.format,
            "core:sample_rate": report.sample_rate_hz,
            "core:version": "1.0.0",
            "ntro_sih:pipeline_version": report.pipeline_version,
        },
        "captures": [{"core:sample_start": 0, "core:frequency": 0}],
        "annotations": anns,
    }


def write_sigmf(report: CaptureReport, out_path: str | Path) -> Path:
    p = Path(out_path)
    p.write_text(json.dumps(to_sigmf(report), indent=2))
    return p
