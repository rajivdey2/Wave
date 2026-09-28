"""Annotated spectrogram renderer — the lead demo artifact (§3.3)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.ingestion import load_capture
from src.types import CaptureReport, ConfidenceState

COLORS = {
    ConfidenceState.DETECTED: ("lime", "-"),
    ConfidenceState.ESTIMATED: ("cyan", "-"),
    ConfidenceState.INFERRED: ("yellow", "--"),
    ConfidenceState.UNKNOWN: ("red", ":"),
}


def render_spectrogram(report: CaptureReport, out_path: str | Path,
                       iq: np.ndarray | None = None) -> Path:
    if iq is None:
        cap = load_capture(report.source_file,
                           sample_rate_hz=report.sample_rate_hz)
        iq = cap.samples
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.specgram(iq, NFFT=1024, Fs=report.sample_rate_hz,
                noverlap=768, cmap="viridis")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    for d in report.detections:
        color, ls = COLORS.get(d.modulation.state, ("white", "-"))
        w, h = d.t_end_s - d.t_start_s, d.f_high_hz - d.f_low_hz
        # specgram centers frequency at 0 when complex; shift box accordingly
        rect = patches.Rectangle((d.t_start_s, d.f_low_hz), max(w, 1e-3),
                                 max(h, 1.0), linewidth=1.5,
                                 edgecolor=color, facecolor="none",
                                 linestyle=ls)
        ax.add_patch(rect)
        conf = f"{d.modulation.confidence*100:.0f}%" if d.modulation.confidence else "?"
        ax.text(d.t_start_s, d.f_high_hz,
                f"{d.modulation.value} ({d.modulation.state.value},{conf})",
                color=color, fontsize=8,
                bbox=dict(facecolor="black", alpha=0.6, pad=1))
    ax.set_title(f"{Path(report.source_file).name} — {len(report.detections)} detection(s)")
    fig.tight_layout()
    p = Path(out_path)
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p
