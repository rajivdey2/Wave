"""Per-region parameter estimation. All outputs are FieldValue-typed."""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

from src.detection import Region
from src.types import ConfidenceState, FieldValue

SNR_FLOOR_DB = 0.0  # below this → UNKNOWN


def _mix_to_baseband(iq: np.ndarray, sample_rate_hz: float, cf_hz: float) -> np.ndarray:
    t = np.arange(len(iq)) / sample_rate_hz
    return (iq * np.exp(-1j * 2 * np.pi * cf_hz * t)).astype(np.complex64)


def estimate_center_bw(iq: np.ndarray, sample_rate_hz: float,
                       r: Region) -> tuple[FieldValue, FieldValue]:
    nperseg = min(8192, max(1024, len(iq) // 8))
    freqs, psd = welch(iq, fs=sample_rate_hz, nperseg=nperseg,
                       return_onesided=False, scaling="spectrum",
                       detrend=False)
    order = np.argsort(freqs)
    freqs, psd = freqs[order], psd[order]
    inband = (freqs >= r.f_low_hz) & (freqs <= r.f_high_hz)
    if not inband.any():
        return (FieldValue.unknown("no PSD bins inside region"),
                FieldValue.unknown("no PSD bins inside region"))
    f_in, p_in = freqs[inband], psd[inband]
    # Center = power centroid over the main lobe (bins within 6 dB of peak).
    # Peak-bin alone jitters on flat-topped digital lobes; full-region
    # centroid biases on asymmetrically truncated sidelobes. Main-lobe mask
    # is robust for both — and degrades gracefully to region centroid for
    # flat spectra (chirp/OFDM).
    peak = float(p_in.max())
    lobe = p_in >= peak / 4
    if int(lobe.sum()) >= 3:
        w = p_in[lobe]
        centroid = float(np.sum(f_in[lobe] * w) / (np.sum(w) + 1e-18))
    else:
        centroid = float(f_in[int(np.argmax(p_in))])
    above_half = f_in[p_in >= peak / 2]
    bw3 = float(above_half.max() - above_half.min()) if len(above_half) > 1 else float(
        r.f_high_hz - r.f_low_hz)
    above_1p = f_in[p_in >= peak / 100]
    bw20 = float(above_1p.max() - above_1p.min()) if len(above_1p) > 1 else bw3 * 2
    cf = FieldValue(value=centroid, state=ConfidenceState.DETECTED,
                    confidence=1.0,
                    evidence=f"main-lobe centroid over {int(lobe.sum())} bins, peak {10*np.log10(peak+1e-18):.1f}dB")
    bw = FieldValue(value=bw3, state=ConfidenceState.ESTIMATED,
                    confidence=0.85,
                    evidence=f"-3dB width {bw3:.0f}Hz, -20dB width {bw20:.0f}Hz from Welch PSD")
    return cf, bw


def estimate_snr(iq: np.ndarray, sample_rate_hz: float, r: Region,
                 snr_db: float) -> FieldValue:
    if snr_db < SNR_FLOOR_DB:
        return FieldValue(value=round(snr_db, 1), state=ConfidenceState.UNKNOWN,
                          confidence=None,
                          evidence=f"SNR {snr_db:.1f}dB below {SNR_FLOOR_DB}dB floor — refused to trust")
    conf = float(min(0.99, 0.5 + snr_db / 40.0))
    return FieldValue(value=round(snr_db, 1), state=ConfidenceState.ESTIMATED,
                      confidence=conf,
                      evidence=f"in-band vs out-of-band power ratio = {snr_db:.1f}dB")


def _cepstral_rs(seg: np.ndarray, sample_rate_hz: float,
                 lo_hz: float = 200.0, hi_hz: float = 4000.0,
                 span_hz: float = 10000.0) -> tuple[float | None, float]:
    """Symbol rate from cepstral peak (periodicity of spectral nulls).

    Rect-pulse spectra have nulls spaced Rs apart; the IFFT of the log
    spectrum (cepstrum) peaks at quefrency 1/Rs. Robust to shallow individual
    nulls and noise ripple that defeat first-null walks. Returns (Rs, ratio).
    """
    from scipy.signal import welch as _welch
    n = min(len(seg), 1 << 15)
    freqs, psd = _welch(seg[:n], fs=sample_rate_hz,
                        nperseg=min(4096, n), return_onesided=False,
                        scaling="spectrum", detrend=False)
    order = np.argsort(freqs)
    freqs, psd = freqs[order], psd[order]
    m = np.abs(freqs) <= span_hz
    L = np.log(psd[m] + 1e-18)
    L = L - L.mean()
    C = np.abs(np.fft.ifft(L)) ** 2
    df = float(freqs[1] - freqs[0])
    N = len(C)
    q = np.arange(N) / (N * df)  # quefrency, seconds
    m2 = (q >= 1 / hi_hz) & (q <= 1 / lo_hz)
    if not m2.any():
        return None, 0.0
    idx = np.arange(N)[m2]
    k = int(np.argmax(C[m2]))
    ratio = float(C[m2][k] / (np.median(C[m2]) + 1e-18))
    return float(1 / q[idx[k]]), ratio


def estimate_symbol_rate(iq: np.ndarray, sample_rate_hz: float,
                         cf_hz: float, snr_db: float,
                         label: str = "") -> FieldValue:
    """Symbol rate via PSD first-null spacing (rect-pulse linear modes) with
    envelope-spectrum fallback (FSK/AM/QAM with envelope tone)."""
    if snr_db < SNR_FLOOR_DB:
        return FieldValue.unknown("SNR below floor — symbol rate not attempted")
    bb = _mix_to_baseband(iq, sample_rate_hz, cf_hz)
    if label == "FSK":
        # NRZ discriminator spectrum has nulls at multiples of Rs.
        inst = np.diff(np.unwrap(np.angle(bb + 1e-12)))
        k = np.ones(8) / 8
        d = np.convolve(inst, k, mode="same")
        n = min(len(d), 1 << 15)
        freqs, psd = welch(d[:n], fs=sample_rate_hz, nperseg=min(4096, n),
                           return_onesided=True, scaling="spectrum",
                           detrend=False)
        peak = float(psd.max() + 1e-18)
        bw_bin = float(freqs[1] - freqs[0])
        w = max(1, int(150 / bw_bin))
        ker = np.ones(2 * w + 1) / (2 * w + 1)
        sm = np.convolve(psd, ker, mode="same")
        null_f: float | None = None
        for j in range(1, len(sm) - 1):
            f = float(freqs[j])
            if f < 200.0:
                continue
            if f > 5000.0:
                break
            if sm[j] < sm[j - 1] and sm[j] <= sm[j + 1] and sm[j] < 0.3 * peak:
                null_f = f
                break
        if null_f is not None:
            return FieldValue(value=round(null_f, 1),
                              state=ConfidenceState.ESTIMATED, confidence=0.65,
                              evidence=f"first discriminator-spectrum null at {null_f:.0f}Hz — FSK symbol rate")
        return FieldValue.unknown("no discriminator null — symbol rate unclear")
    rs_cep, ratio = _cepstral_rs(bb, sample_rate_hz)
    if rs_cep is not None and ratio > 10.0:
        return FieldValue(value=round(rs_cep, 1),
                          state=ConfidenceState.ESTIMATED,
                          confidence=float(min(0.9, 0.5 + np.log10(ratio) / 4)),
                          evidence=f"cepstral peak at 1/{rs_cep:.0f}Hz (ratio {ratio:.0f}) — rect-pulse symbol rate")
    return FieldValue.unknown(
        "no cepstral symbol-rate peak — likely analog/unmodulated")
