"""FEC scheme identification on recovered bitstreams (lightweight)."""
from __future__ import annotations

import numpy as np

from src.types import ConfidenceState, FieldValue


def identify_fec(bits: np.ndarray) -> FieldValue:
    """Heuristic: balance/bias + autocorrelation repetition → block vs convolutional vs none."""
    if bits is None or len(bits) < 200:
        return FieldValue.unknown("bitstream too short for FEC identification")
    b = np.asarray(bits, dtype=float)
    balance = float(b.mean())  # ~0.5 for scrambled/coded
    # repetition check (simple block-code hint): autocorr peaks at short lags
    n = min(len(b), 2048)
    bc = b[:n] - b[:n].mean()
    ac = np.correlate(bc, bc, mode="full")[n - 1:]
    ac = ac / (ac[0] + 1e-18)
    peak = float(np.max(ac[8:64])) if len(ac) > 64 else 0.0
    if peak > 0.25:
        return FieldValue(value="BLOCK?", state=ConfidenceState.INFERRED,
                          confidence=0.55,
                          evidence=f"autocorr repetition peak {peak:.2f} at short lag — possible block code/interleaver")
    if 0.45 < balance < 0.55:
        return FieldValue(value="UNCODED/SCR?", state=ConfidenceState.INFERRED,
                          confidence=0.5,
                          evidence=f"balanced bits ({balance:.2f}), no repetition — uncoded or scrambled")
    return FieldValue(value="UNKNOWN", state=ConfidenceState.UNKNOWN,
                      confidence=None,
                      evidence=f"biased bits ({balance:.2f}) — insufficient structure")
