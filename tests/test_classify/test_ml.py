"""ML layer tests: trains a small forest, checks held-out + fusion behavior."""
import numpy as np


def _tiny_forest():
    from src.classify.ml import build_training_matrix, train
    X, y = build_training_matrix(duration_s=0.25, seeds_per=1,
                                 snrs=(5, 15), seed=7)
    clf = train(n_estimators=25, random_state=1)
    clf.fit(X, y)
    return clf


def test_ml_held_out_accuracy():
    from src.classify.ml import build_training_matrix, predict_proba
    clf = _tiny_forest()
    Xte, yte = build_training_matrix(duration_s=0.25, seeds_per=1,
                                     snrs=(5, 15), seed=999)
    preds = [max(predict_proba(clf, v), key=predict_proba(clf, v).get)
             for v in Xte]
    acc = sum(p == t for p, t in zip(preds, yte)) / len(yte)
    assert acc >= 0.75, f"tiny-forest held-out accuracy {acc:.2f} < 0.75"


def test_fusion_agreement_marks_both():
    # Full-strength model should agree with classical on a clean BPSK file.
    from src.ingestion import load_capture
    from src.classify.ml import extract_feature_vector
    from src.classify.fuse import fuse
    from src.types import ConfidenceState, FieldValue
    cap = load_capture("data/synthetic/BPSK_10dB_cf32.iq",
                       sample_rate_hz=100_000.0)
    t = np.arange(len(cap.samples)) / 100_000.0
    bb = cap.samples * np.exp(-1j * 2 * np.pi * 10_000.0 * t)
    fv = FieldValue(value="PSK", state=ConfidenceState.INFERRED,
                    confidence=0.7, evidence="test")
    out, method = fuse(fv, "classical", bb, 100_000.0, 2400.0, 10.0)
    assert out.value == "PSK", f"fusion flipped clean BPSK to {out.value}"
    assert method in ("classical", "both"), method


def test_fusion_missing_model_falls_back():
    import importlib
    import src.classify.fuse  # noqa: F401  (ensures sys.modules entry)
    import sys
    fusemod = sys.modules["src.classify.fuse"]
    from src.types import ConfidenceState, FieldValue
    fv = FieldValue(value="FM", state=ConfidenceState.INFERRED,
                    confidence=0.7, evidence="test")
    old = fusemod._ML, fusemod._ML_MISSING
    fusemod._ML, fusemod._ML_MISSING = None, True  # simulate absent .pkl
    try:
        out, method = fusemod.fuse(fv, "classical", np.zeros(1024, complex),
                                   100_000.0, 6000.0, 10.0)
    finally:
        fusemod._ML, fusemod._ML_MISSING = old
    assert out.value == "FM" and method == "classical"
