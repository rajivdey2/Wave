"""End-to-end pipeline: ingestion → detection → extraction → classify → report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.ingestion import load_capture, quality_metrics
from src.detection import detect_regions, snr_of_region
from src.extraction import estimate_center_bw, estimate_snr, estimate_symbol_rate
from src.classify import classify_classical
from src.demod import demod_psk
from src.fec import identify_fec
from src.types import CaptureReport, FieldValue, SignalDetection


def analyze_file(path: str | Path, sample_rate_hz: float | None = None,
                 dtype_hint: str | None = None) -> CaptureReport:
    cap = load_capture(path, sample_rate_hz=sample_rate_hz, dtype_hint=dtype_hint)
    floor, dyn = quality_metrics(cap.samples)
    regions = detect_regions(cap.samples, cap.sample_rate_hz)

    detections: list[SignalDetection] = []
    for i, r in enumerate(regions):
        snr = snr_of_region(cap.samples, cap.sample_rate_hz, r)
        cf, bw = estimate_center_bw(cap.samples, cap.sample_rate_hz, r)
        snr_fv = estimate_snr(cap.samples, cap.sample_rate_hz, r, snr)
        # baseband slice for classifier: mix region center to DC, then
        # isolate the region (kills other signals in multi-signal captures)
        t = np.arange(len(cap.samples)) / cap.sample_rate_hz
        cf_hz = float(cf.value) if isinstance(cf.value, (int, float)) else 0.0
        bb = cap.samples * np.exp(-1j * 2 * np.pi * cf_hz * t)
        bb_mixed = bb  # unfiltered copy: Rs estimation needs the true
        # spectrum (isolation edges create false cepstral periodicity)
        from src.classify.classical import isolate_region
        # Isolation passband = region expanded (min ±1.5 kHz): regions are
        # CFAR-tight and may cut sidelobes — a tight brickwall rings and
        # corrupts statistics. Other signals are much farther away.
        elo, ehi = r.f_low_hz - cf_hz, r.f_high_hz - cf_hz
        pad = max(1500.0, 0.25 * (ehi - elo))
        bb = isolate_region(bb, cap.sample_rate_hz, elo - pad, ehi + pad)
        mod_fv, method = classify_classical(
            bb, snr, sample_rate_hz=cap.sample_rate_hz,
            bandwidth_hz=(float(bw.value) if isinstance(bw.value, (int, float)) else None))
        # ML fusion: confirm / rescue / override (never below SNR floor,
        # never without the model file — see src/classify/fuse.py).
        from src.classify.fuse import fuse
        mod_fv, method = fuse(
            mod_fv, method, bb, cap.sample_rate_hz,
            (float(bw.value) if isinstance(bw.value, (int, float)) else None),
            snr)
        label = str(mod_fv.value or "UNKNOWN")
        # Symbol rate is only meaningful for clocked digital modes.
        # bb_mixed is already at DC (cf 0 = identity mix inside).
        if label in ("PSK", "BPSK", "QPSK", "QAM", "FSK"):
            sr_fv = estimate_symbol_rate(bb_mixed, cap.sample_rate_hz,
                                         0.0, snr, label=label)
        else:
            sr_fv = FieldValue.unknown(
                f"symbol rate not applicable to {label}")

        demod_available = False
        fec_fv: FieldValue | None = None
        sr_val = sr_fv.value if isinstance(sr_fv.value, (int, float)) else None
        if label in ("PSK", "BPSK", "QPSK"):
            try:
                from src.demod import demod_psk, pick_psk_mode
                submode = pick_psk_mode(bb)  # BPSK vs QPSK slicer
                bits = demod_psk(bb, submode, sr_val, cap.sample_rate_hz)
                fec_fv = identify_fec(bits)
                if fec_fv.value is None:
                    fec_fv = FieldValue.unknown(
                        f"{submode} demod ok ({len(bits)} bits), no FEC structure")
                else:
                    fec_fv = FieldValue(
                        value=fec_fv.value, state=fec_fv.state,
                        confidence=fec_fv.confidence,
                        evidence=f"{submode} ({len(bits)} bits): {fec_fv.evidence}")
                demod_available = True
            except Exception as e:
                fec_fv = FieldValue.unknown(f"demod failed: {e}")
        else:
            demod_available = False
            fec_fv = None

        detections.append(SignalDetection(
            id=f"sig-{i:02d}", t_start_s=r.t_start_s, t_end_s=r.t_end_s,
            f_low_hz=r.f_low_hz, f_high_hz=r.f_high_hz,
            center_freq_hz=cf, bandwidth_hz=bw, snr_db=snr_fv,
            symbol_rate_hz=sr_fv, modulation=mod_fv,
            modulation_evidence_method=method,
            demod_available=demod_available, fec_identification=fec_fv))

    # Pure-noise edge case: no regions → emit single UNKNOWN detection? No —
    # return empty detections; UNKNOWN semantics live per-field. Keep empty.
    return CaptureReport(source_file=cap.source_file, format=cap.format,
                         sample_rate_hz=cap.sample_rate_hz,
                         duration_s=cap.duration_s,
                         noise_floor_db=floor, dynamic_range_db=dyn,
                         detections=detections)


def write_json_report(report: CaptureReport, out_path: str | Path) -> Path:
    p = Path(out_path)
    p.write_text(json.dumps(report.to_dict(), indent=2))
    return p
