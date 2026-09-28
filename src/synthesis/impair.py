"""Recorded-RF impairment models (RadioML-style channel).

Our base generator uses rect pulses, zero offset, flat fading — clean.
Recorded captures (RadioML 2016.10a and real receivers) have RRC-shaped
pulses, carrier/sample-rate offsets, and slow fading. Training the ML layer
on impaired rows closes that domain gap WITHOUT needing recorded RF at
train time; RadioML then serves as an honest held-out check.
"""
from __future__ import annotations

import numpy as np


def rrc_taps(sps: int, beta: float = 0.35, span_symbols: int = 8) -> np.ndarray:
    """Root-raised-cosine FIR taps (Tranter/Danish formula, singularities fixed)."""
    n = span_symbols * sps
    t = (np.arange(n + 1) - n / 2) / sps  # time in symbols
    h = np.zeros_like(t)
    for i, tt in enumerate(t):
        if abs(tt) < 1e-10:
            h[i] = 1.0 + beta * (4 / np.pi - 1)
        elif abs(abs(tt) - 1 / (4 * beta)) < 1e-10:
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            num = (np.sin(np.pi * tt * (1 - beta))
                   + 4 * beta * tt * np.cos(np.pi * tt * (1 + beta)))
            den = np.pi * tt * (1 - (4 * beta * tt) ** 2)
            h[i] = num / den
    h /= np.sqrt(np.sum(h ** 2) / sps)  # unit gain per symbol
    return h


def rrc_shape(symbols: np.ndarray, sps: int, beta: float = 0.35) -> np.ndarray:
    """Pulse-shape a symbol stream (upsample → RRC → Matched-ish output)."""
    up = np.zeros(len(symbols) * sps, dtype=complex)
    up[::sps] = symbols
    h = rrc_taps(sps, beta)
    y = np.convolve(up, h, mode="same")
    y /= np.sqrt(np.mean(np.abs(y) ** 2) + 1e-18)
    return y


def apply_cfo_fade(iq: np.ndarray, sample_rate_hz: float, cfo_hz: float,
                   fade_depth: float, rng: np.random.Generator) -> np.ndarray:
    """Carrier offset + slow amplitude random-walk fading (mean gain = 1)."""
    n = len(iq)
    t = np.arange(n) / sample_rate_hz
    out = iq * np.exp(1j * 2 * np.pi * cfo_hz * t)
    if fade_depth > 0:
        walk = np.cumsum(rng.standard_normal(n)) * 0.002
        walk = walk - walk.mean()
        gain = 1.0 + fade_depth * walk / (np.std(walk) + 1e-9) * 0.3
        out = out * gain
    return out.astype(np.complex64)
