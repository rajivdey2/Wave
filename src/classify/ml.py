"""ML modulation classifier over engineered DSP features.

Classical discriminators (classical.py) are interpretable but threshold-fragile
near decision boundaries. This layer trains a RandomForest on the SAME
features (+ a few extras) so the model learns the boundaries from data.
CPU-only, offline, no torch — a .pkl ships in models/.

Fusion policy lives in src/report/builder.py; this module only trains,
saves, loads, and predicts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL_PATH = Path("models/ml_classifier.pkl")
FEATURE_NAMES = [
    "sa", "sf", "papr", "p2", "p4", "log_disc", "cells", "n_levels",
    "log_cas", "log_eratio", "comb", "bw_norm", "spec_flat", "phase_ent",
    "snr_db",
]
# Generator mode → family label (shared vocabulary with classical.py)
FAMILY = {"AM": "AM", "FM": "FM", "SSB": "SSB", "CW": "CW", "FSK2": "FSK",
          "BPSK": "PSK", "QPSK": "PSK", "QAM16": "QAM", "OFDM": "OFDM",
          "CHIRP": "CHIRP"}


def _phase_entropy(seg: np.ndarray) -> float:
    ph, _ = np.histogram(np.angle(seg + 1e-12), bins=16,
                         range=(-np.pi, np.pi))
    p = ph / (ph.sum() + 1e-18) + 1e-12
    return float(-np.sum(p * np.log(p)) / np.log(16))


def _spectral_flatness(seg: np.ndarray) -> float:
    n = min(len(seg), 1 << 14)
    S = np.abs(np.fft.fft(seg[:n])) ** 2 + 1e-18
    return float(np.exp(np.mean(np.log(S))) / np.mean(S))


def extract_feature_vector(bb: np.ndarray, sample_rate_hz: float,
                           bandwidth_hz: float | None,
                           snr_db: float) -> np.ndarray:
    """Fixed-order feature vector (see FEATURE_NAMES). bb must already be
    mixed to center + region-isolated, exactly as in the production path."""
    from src.classify.classical import (
        _amp_levels, _carrier_asymmetry, _constellation_cells,
        _disc_tone_ratio, _envelope_comb_count, _nonlinear_coherence)
    seg = np.asarray(bb).ravel()[:min(len(bb), 1 << 15)]
    amp = np.abs(seg)
    amp_n = amp / (np.mean(amp) + 1e-12)
    sa = float(np.std(amp_n))
    inst = np.diff(np.unwrap(np.angle(seg + 1e-12)))
    sf = float(np.std(inst))
    papr = float(np.max(amp ** 2) / (np.mean(amp ** 2) + 1e-18))
    env = amp - amp.mean()
    n = len(env)
    espec = np.abs(np.fft.rfft(env)) ** 2
    efreq = np.fft.rfftfreq(n, 1 / sample_rate_hz)
    m = (efreq >= 20) & (efreq <= 1000)
    eratio = float(espec[m].max() / (np.median(espec[m]) + 1e-18)) if m.any() else 1.0
    return np.array([
        sa, sf, papr,
        _nonlinear_coherence(seg, 2), _nonlinear_coherence(seg, 4),
        np.log10(_disc_tone_ratio(seg, sample_rate_hz) + 1),
        _constellation_cells(seg), _amp_levels(amp_n),
        np.log10(_carrier_asymmetry(seg, sample_rate_hz) + 1e-9),
        np.log10(eratio),
        _envelope_comb_count(seg, sample_rate_hz),
        (bandwidth_hz / sample_rate_hz) if bandwidth_hz else 0.0,
        _spectral_flatness(seg), _phase_entropy(seg), snr_db,
    ], dtype=float)


def build_training_matrix(sample_rate_hz: float = 100_000.0, duration_s: float = 0.5,
                          snrs: tuple = (-2, 0, 5, 10, 15, 20),
                          seeds_per: int = 3, seed: int = 0,
                          jitter: bool = True):
    """Generate labeled feature rows using TRUE center/band (clean supervision;
    production substitutes estimates, which match at ≥5 dB).

    jitter=True mimics production estimation error (residual center offset,
    band-scale error, SNR error) so the model stays accurate on estimated —
    not just perfect — features. This closed the 0 dB BPSK/QPSK gap.
    """
    from src.synthesis.generator import MODES, generate
    from src.classify.classical import isolate_region
    from src.synthesis.impair import apply_cfo_fade
    rng = np.random.default_rng(seed + 77)
    X, y = [], []
    k = 0
    # (pulse, impaired-channel) variants: clean rect rows + RRC/CFO/fading rows
    variants = [("rect", False)] * seeds_per + [("rrc", True)] * max(2, seeds_per // 2)
    for mode in MODES:
        for snr in snrs:
            for pulse, impaired in variants:
                use_rrc = pulse == "rrc" and mode in ("BPSK", "QPSK", "QAM16")
                iq, bw, _ = generate(mode, sample_rate_hz, duration_s,
                                     10_000.0 if mode != "OFDM" else 0.0,
                                     float(snr), seed=seed + k,
                                     pulse="rrc" if use_rrc else "rect")
                k += 1
                cf = 10_000.0 if mode != "OFDM" else 0.0
                cfo = float(rng.uniform(-300, 300)) if impaired else 0.0
                if impaired:
                    iq = apply_cfo_fade(iq, sample_rate_hz, cfo, 0.3, rng)
                t = np.arange(len(iq)) / sample_rate_hz
                if jitter:
                    cf_e = cf + cfo + float(rng.uniform(-300, 300))
                    bw_e = bw * float(rng.uniform(0.7, 1.4))
                    snr_e = float(snr) + float(rng.uniform(-1, 1))
                else:
                    cf_e, bw_e, snr_e = cf + cfo, bw, float(snr)
                # Mix (possibly mis-estimated) center to DC, isolate band
                bb = iq * np.exp(-1j * 2 * np.pi * cf_e * t)
                pad = max(1500.0, 0.25 * bw_e)
                bb = isolate_region(bb, sample_rate_hz, -bw_e / 2 - pad, bw_e / 2 + pad)
                X.append(extract_feature_vector(bb, sample_rate_hz, bw_e, snr_e))
                y.append(FAMILY[mode])
    return np.array(X), np.array(y)


def train(n_estimators: int = 150, **kw):
    from sklearn.ensemble import RandomForestClassifier
    kw.setdefault("random_state", 0)
    kw.setdefault("n_jobs", -1)
    clf = RandomForestClassifier(n_estimators=n_estimators, **kw)
    return clf


def save_model(clf, path: str | Path = MODEL_PATH) -> Path:
    import joblib
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "features": FEATURE_NAMES}, p)
    return p


def load_model(path: str | Path = MODEL_PATH):
    try:
        import joblib
    except ImportError:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return joblib.load(p)["model"]
    except Exception:
        return None


def predict_proba(clf, vec: np.ndarray) -> dict[str, float]:
    proba = clf.predict_proba(vec.reshape(1, -1))[0]
    return {str(c): float(p) for c, p in zip(clf.classes_, proba)}
