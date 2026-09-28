"""BPSK/QPSK bit recovery (scoped to 2 modes done well).

Chain per mode: FFT carrier wipe (2nd/4th power) → timing search → hard
slicer. Phase ambiguity (×2 / ×4) is intrinsic to blind recovery and is
resolved in tests by minimum-BER rotation; production notes it in evidence
(real deployments use differential coding or a known preamble).
"""
from __future__ import annotations

import numpy as np


def wipe_carrier(bb: np.ndarray, order: int) -> np.ndarray:
    """Remove residual carrier via FFT peak of bb^order (mean-phasor fails
    with any frequency offset; the FFT wipe tolerates ±fs/4)."""
    n = len(bb)
    sp = bb ** order
    F = np.abs(np.fft.fft(sp)) ** 2
    k = int(np.argmax(F))
    f = (k / n) if k <= n // 2 else ((k - n) / n)  # cycles/sample
    t = np.arange(n)
    return bb * np.exp(-1j * 2 * np.pi * (f / order) * t)


def best_offset(x: np.ndarray, sps: int) -> int:
    """Symbol-center offset maximizing sum|x|^4 (Godard-lite timing)."""
    best, score = 0, -1.0
    for off in range(sps):
        s = float(np.sum(np.abs(x[off::sps]) ** 4))
        if s > score:
            best, score = off, s
    return best


def demod_bpsk(bb: np.ndarray, sps: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (symbols, bits). Bits have π ambiguity (see min_ber)."""
    c = wipe_carrier(bb, 2)
    syms = c[best_offset(c, sps)::sps]
    bits = (np.real(syms) > 0).astype(np.int8)
    return syms, bits


def demod_qpsk(bb: np.ndarray, sps: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (symbols, bits[I0,Q0,I1,Q1...]). 4-fold phase ambiguity."""
    c = wipe_carrier(bb, 4)
    syms = c[best_offset(c, sps)::sps]
    bits = np.empty(2 * len(syms), dtype=np.int8)
    bits[0::2] = (np.real(syms) > 0).astype(np.int8)
    bits[1::2] = (np.imag(syms) > 0).astype(np.int8)
    return syms, bits


def demod_psk(bb: np.ndarray, mode: str, symbol_rate_hz: float | None,
              sample_rate_hz: float) -> np.ndarray:
    """Blind PSK demod entry point (kept for pipeline use)."""
    if symbol_rate_hz is None or symbol_rate_hz <= 0:
        symbol_rate_hz = 2000.0
    sps = max(1, int(round(sample_rate_hz / symbol_rate_hz)))
    if mode.upper().startswith("Q"):
        return demod_qpsk(bb, sps)[1]
    return demod_bpsk(bb, sps)[1]


def pick_psk_mode(bb: np.ndarray) -> str:
    """BPSK vs QPSK slicer choice via wiped coherence (p2 wins → BPSK)."""
    from src.classify.classical import _nonlinear_coherence
    seg = np.asarray(bb).ravel()[:min(len(bb), 1 << 15)]
    p2 = _nonlinear_coherence(seg, 2)
    p4 = _nonlinear_coherence(seg, 4)
    return "BPSK" if p2 >= p4 else "QPSK"


def min_ber(rx_bits: np.ndarray, tx_bits: np.ndarray,
            rotations: int) -> tuple[float, int]:
    """Minimum BER over carrier-phase rotations (test/validation helper).

    BPSK: {identity, inversion}. QPSK: applied at symbol level by the caller
    via rotate+reslice — here implemented as bit transforms for speed:
    rot k of (i,q) pairs = [(q,!i)... ] — instead we brute-force by rotating
    the RX SYMBOLS; this helper only handles BPSK inversion. Use
    min_ber_qpsk for QPSK.
    """
    n = min(len(rx_bits), len(tx_bits))
    rx, tx = rx_bits[:n], tx_bits[:n]
    ber0 = float(np.mean(rx != tx))
    ber1 = float(np.mean((1 - rx) != tx))
    return (ber0, 0) if ber0 <= ber1 else (ber1, 1)


def min_ber_qpsk(rx_syms: np.ndarray, tx_bits: np.ndarray) -> tuple[float, int]:
    """Min BER over the 4 QPSK phase rotations (symbols rotated, resliced)."""
    n_sym = min(len(rx_syms), len(tx_bits) // 2)
    tx = tx_bits[:2 * n_sym]
    best = (1.0, 0)
    for k in range(4):
        rot = rx_syms[:n_sym] * np.exp(1j * k * np.pi / 2)
        bits = np.empty(2 * n_sym, dtype=np.int8)
        bits[0::2] = (np.real(rot) > 0).astype(np.int8)
        bits[1::2] = (np.imag(rot) > 0).astype(np.int8)
        ber = float(np.mean(bits != tx))
        if ber < best[0]:
            best = (ber, k)
    return best
