"""Synthetic signal generator — labeled captures + ground-truth manifest.

Covers PS-required modes: AM, FM, SSB, CW, FSK, PSK variants, QAM variants,
OFDM, chirp. Every file gets a manifest row (ARCHITECTURE.md §5).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

MODES = ["AM", "FM", "SSB", "CW", "FSK2", "BPSK", "QPSK", "QAM16", "OFDM", "CHIRP"]
SNR_DB_SWEEP = [-5, 0, 5, 10, 15, 20]


@dataclass
class GroundTruth:
    file: str
    modulation: str
    center_freq_hz: float
    bandwidth_hz: float
    symbol_rate_hz: float | None
    snr_db: float
    sample_rate_hz: float


def _awgn(sig: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    p_sig = float(np.mean(np.abs(sig) ** 2)) + 1e-18
    p_noise = p_sig / (10 ** (snr_db / 10))
    noise = rng.standard_normal(len(sig)) + 1j * rng.standard_normal(len(sig))
    noise *= np.sqrt(p_noise / 2)
    return (sig + noise).astype(np.complex64)


def _bits(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(0, 2, n).astype(np.float64) * 2 - 1


def generate(mode: str, sample_rate_hz: float, duration_s: float,
             center_freq_hz: float, snr_db: float, seed: int = 0,
             return_bits: bool = False, pulse: str = "rect") -> np.ndarray:
    """Returns (iq, bw, rs), or (iq, bw, rs, tx_bits) with return_bits=True.

    tx_bits ({0,1} int8) is the transmitted hard-bit stream for BPSK
    (1 bit/symbol) and QPSK ([I0,Q0,I1,Q1...]) — ground truth for BER checks.
    None for other modes.

    pulse="rrc" applies RRC shaping (beta 0.35) to BPSK/QPSK/QAM16 —
    recorded-RF-like; default "rect" preserves legacy behavior exactly.
    """
    rng = np.random.default_rng(seed)
    n = int(sample_rate_hz * duration_s)
    t = np.arange(n) / sample_rate_hz
    msg_rate = 100.0  # Hz message tone for analog modes
    tx_bits = None

    if mode == "AM":
        m = 0.7 * np.cos(2 * np.pi * msg_rate * t)
        base = (1 + m) * np.exp(1j * 2 * np.pi * 1_000 * t)  # complex, single-sided
        bw, rs = 2 * msg_rate * 1.2, None
    elif mode == "FM":
        freq_dev = 3_000.0  # Hz — clearly wideband FM
        phase = 2 * np.pi * 1_000 * t + (freq_dev / msg_rate) * np.sin(2 * np.pi * msg_rate * t)
        base = np.exp(1j * phase)  # complex, single-sided
        bw, rs = 2 * (freq_dev + msg_rate), None
    elif mode == "SSB":
        # USB: carrier + single upper sideband (complex envelope keeps it
        # truly single-sided; a real-valued message would create DSB/AM).
        env_usb = 0.5 + 0.5 * np.exp(1j * 2 * np.pi * msg_rate * t)
        base = env_usb * np.exp(1j * 2 * np.pi * 1_000 * t)
        bw, rs = msg_rate * 1.5 + 50.0, None
    elif mode == "CW":
        keying = ((np.sin(2 * np.pi * 5 * t) > 0)).astype(float)  # 5 Hz on/off
        base = keying * np.exp(1j * 2 * np.pi * 1_000 * t)
        bw, rs = 60.0, None
    elif mode == "FSK2":
        rs = 1_000.0
        sps = max(1, int(sample_rate_hz / rs))
        n_sym = n // sps
        syms = rng.integers(0, 2, n_sym) * 2 - 1
        freq_dev = 800.0
        phase = np.cumsum(np.repeat(syms, sps)[:n] * freq_dev / sample_rate_hz * 2 * np.pi)
        base = np.exp(1j * phase)
        bw = 2 * freq_dev + rs
    elif mode == "BPSK":
        rs = 2_000.0
        sps = max(1, int(sample_rate_hz / rs))
        n_sym = n // sps
        syms = _bits(rng, n_sym)
        if pulse == "rrc":
            from src.synthesis.impair import rrc_shape
            base = np.resize(rrc_shape(syms, sps), n)
        else:
            base = np.repeat(syms, sps)[:n].astype(complex)
        bw = rs * 1.2
        tx_bits = ((syms + 1) // 2).astype(np.int8)
    elif mode == "QPSK":
        rs = 2_000.0
        sps = max(1, int(sample_rate_hz / rs))
        n_sym = n // sps
        b = _bits(rng, 2 * n_sym)
        syms_q = ((b[0::2] + 1j * b[1::2]) / np.sqrt(2))
        if pulse == "rrc":
            from src.synthesis.impair import rrc_shape
            base = np.resize(rrc_shape(syms_q, sps), n)
        else:
            base = np.repeat(syms_q, sps)[:n]
        bw = rs * 1.2
        tx_bits = ((b + 1) // 2).astype(np.int8)  # [I0,Q0,I1,Q1...]
    elif mode == "QAM16":
        rs = 2_000.0
        sps = max(1, int(sample_rate_hz / rs))
        n_sym = n // sps
        lvl = np.array([-3, -1, 1, 3.0])
        i = lvl[rng.integers(0, 4, n_sym)]
        q = lvl[rng.integers(0, 4, n_sym)]
        syms_16 = ((i + 1j * q) / np.sqrt(10))
        if pulse == "rrc":
            from src.synthesis.impair import rrc_shape
            base = np.resize(rrc_shape(syms_16, sps), n)
        else:
            base = np.repeat(syms_16, sps)[:n]
        bw = rs * 1.2
    elif mode == "OFDM":
        n_sc, cp = 64, 16
        sym_len = n_sc + cp
        n_sym = max(1, n // sym_len)
        scaff = np.zeros((n_sym, n_sc), complex)
        # Symmetric subcarriers around DC (incl. DC bin 0), guards at edges
        used = list(range(0, 25)) + list(range(39, 64))
        qb = (_bits(rng, n_sym * len(used) * 2).reshape(n_sym, len(used), 2))
        scaff[:, used] = (qb[:, :, 0] + 1j * qb[:, :, 1]) / np.sqrt(2)
        time_syms = np.fft.ifft(scaff, axis=1)
        with_cp = np.concatenate([time_syms[:, -cp:], time_syms], axis=1)
        base = with_cp.reshape(-1)[:n]
        if len(base) < n:
            base = np.tile(base, int(np.ceil(n / len(base))))[:n]
        rs, bw = None, sample_rate_hz * len(used) / n_sc
    elif mode == "CHIRP":
        k = 20_000.0 / duration_s  # 20 kHz sweep, centered on base_cf=DC
        base = np.exp(1j * np.pi * k * (t - duration_s / 2) ** 2)
        bw, rs = 20_000.0, None
    else:
        raise ValueError(f"unknown mode {mode}")

    # Shift baseband to requested center. Analog modes sit on a 1 kHz
    # carrier/tone; digital modes + chirp + OFDM are centered at DC.
    base_cf = 1_000.0 if mode in ("AM", "FM", "SSB", "CW") else 0.0
    shift = center_freq_hz - base_cf
    base = np.asarray(base, dtype=complex) * np.exp(1j * 2 * np.pi * shift * t)
    noisy = _awgn(np.asarray(base, dtype=complex), snr_db, rng)
    if return_bits:
        return noisy.astype(np.complex64), float(bw), rs, tx_bits
    return noisy.astype(np.complex64), float(bw), rs


def write_iq_cf32(path: Path, iq: np.ndarray) -> None:
    interleaved = np.empty(2 * len(iq), dtype=np.float32)
    interleaved[0::2] = np.real(iq).astype(np.float32)
    interleaved[1::2] = np.imag(iq).astype(np.float32)
    path.write_bytes(interleaved.tobytes())


def build_dataset(out_dir: str | Path, sample_rate_hz: float = 100_000.0,
                  duration_s: float = 1.0,
                  snrs: list[float] | None = None,
                  modes: list[str] | None = None,
                  seed: int = 42) -> list[GroundTruth]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    snrs = snrs if snrs is not None else SNR_DB_SWEEP
    modes = modes if modes is not None else MODES
    manifest: list[GroundTruth] = []
    k = 0
    for mode in modes:
        for snr in snrs:
            cf = 10_000.0 if mode != "OFDM" else 0.0
            iq, bw, rs = generate(mode, sample_rate_hz, duration_s, cf, float(snr),
                                  seed=seed + k)
            fname = f"{mode}_{int(snr)}dB_cf32.iq"
            write_iq_cf32(out / fname, iq)
            manifest.append(GroundTruth(
                file=fname, modulation=mode, center_freq_hz=float(cf),
                bandwidth_hz=float(bw), symbol_rate_hz=rs,
                snr_db=float(snr), sample_rate_hz=float(sample_rate_hz)))
            k += 1
    # Multi-signal file: BPSK + CW + FM at distinct offsets, 10 dB
    rng = np.random.default_rng(seed + 9999)
    n = int(sample_rate_hz * duration_s)
    mix = np.zeros(n, dtype=complex)
    for mode, cf in (("BPSK", -20_000.0), ("CW", 0.0), ("FM", 20_000.0)):
        iq, _, _ = generate(mode, sample_rate_hz, duration_s, cf, 60.0, seed=seed + k)
        k += 1
        mix += iq / 3.0
    mix = _awgn(mix, 10.0, rng)
    write_iq_cf32(out / "MULTI_10dB_cf32.iq", mix.astype(np.complex64))
    manifest.append(GroundTruth(
        file="MULTI_10dB_cf32.iq", modulation="MULTI:BPSK+CW+FM",
        center_freq_hz=0.0, bandwidth_hz=sample_rate_hz,
        symbol_rate_hz=None, snr_db=10.0, sample_rate_hz=float(sample_rate_hz)))
    # Pure noise file for UNKNOWN-floor check
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    write_iq_cf32(out / "NOISE_cf32.iq", noise.astype(np.complex64))
    manifest.append(GroundTruth(
        file="NOISE_cf32.iq", modulation="NOISE",
        center_freq_hz=0.0, bandwidth_hz=0.0,
        symbol_rate_hz=None, snr_db=-99.0, sample_rate_hz=float(sample_rate_hz)))

    with open(out / "manifest.json", "w") as f:
        json.dump([asdict(m) for m in manifest], f, indent=2)
    return manifest
