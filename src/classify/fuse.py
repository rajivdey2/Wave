"""Classical × ML fusion with honest bookkeeping.

- Classical stays primary (interpretable, evidence-rich).
- ML confirms (method "both"), rescues classical-UNKNOWN (method "ml"),
  or overrides a low-confidence classical vote (method "ml").
- ML never fires below the SNR floor, on low proba (<0.5), or when the
  model file is absent (offline-safe classical-only fallback).
"""
from __future__ import annotations

import numpy as np

from src.types import ConfidenceState, FieldValue

_ML = None
_ML_MISSING = False

RESCUE_PROBA = 0.60
OVERRIDE_PROBA = 0.75


def _get_model():
    global _ML, _ML_MISSING
    if _ML is None and not _ML_MISSING:
        try:
            from src.classify.ml import load_model
            _ML = load_model()
        except Exception:
            _ML = None
        _ML_MISSING = _ML is None
    return _ML


def fuse(mod_fv: FieldValue, method: str, bb: np.ndarray,
         sample_rate_hz: float, bandwidth_hz: float | None,
         snr_db: float) -> tuple[FieldValue, str]:
    clf = _get_model()
    if clf is None or snr_db < 0.0:
        return mod_fv, method
    from src.classify.ml import extract_feature_vector, predict_proba
    try:
        vec = extract_feature_vector(bb, sample_rate_hz, bandwidth_hz, snr_db)
    except Exception:
        return mod_fv, method
    probs = predict_proba(clf, vec)
    top = max(probs, key=probs.get)
    p = float(probs[top])
    if p < 0.5:
        return mod_fv, method

    label = str(mod_fv.value or "UNKNOWN")
    conf = mod_fv.confidence or 0.0

    if label == "UNKNOWN":
        if p >= RESCUE_PROBA:
            return FieldValue(value=top, state=ConfidenceState.INFERRED,
                              confidence=round(p, 2),
                              evidence=mod_fv.evidence + f" | ML vote {top} ({p:.2f})"), "ml"
        return mod_fv, method
    if top == label:
        return FieldValue(value=label, state=mod_fv.state,
                          confidence=round(min(0.95, max(conf, p) + 0.05), 2),
                          evidence=mod_fv.evidence + f" + ML agrees ({p:.2f})"), "both"
    if p >= OVERRIDE_PROBA and conf < OVERRIDE_PROBA:
        return FieldValue(value=top, state=ConfidenceState.INFERRED,
                          confidence=round(p, 2),
                          evidence=f"classical={label} ({conf:.2f}) overruled by ML={top} ({p:.2f})"), "ml"
    return mod_fv, method
