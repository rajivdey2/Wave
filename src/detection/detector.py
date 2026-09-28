"""Detection: Welch PSD + CFAR thresholding + STFT time segmentation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch, stft


@dataclass
class Region:
    f_low_hz: float
    f_high_hz: float
    t_start_s: float
    t_end_s: float
    peak_db: float


def _group(freqs: np.ndarray, psd_db: np.ndarray, thresh: float,
           guard_bins: int, med: float) -> list[tuple[float, float, float]]:
    """Threshold → merged [(f_lo, f_hi, peak_db)] groups with sliver rejection."""
    bin_w = float(freqs[1] - freqs[0])
    guard_bins = max(guard_bins, int(1500 / bin_w))
    mask = psd_db > thresh
    groups: list[list[int]] = []
    cur: list[int] = []
    gap = 0
    for i, m in enumerate(mask):
        if m:
            if cur and gap > guard_bins:
                groups.append(cur)
                cur = []
            cur.append(i)
            gap = 0
        else:
            gap += 1
            if gap > guard_bins and cur:
                groups.append(cur)
                cur = []
    if cur:
        groups.append(cur)
    out: list[tuple[float, float, float]] = []
    for g in groups:
        f_lo, f_hi = float(freqs[g[0]]), float(freqs[g[-1]])
        peak = float(psd_db[g].max())
        prom = peak - med
        wide = (f_hi - f_lo) > 10_000.0
        if (f_hi - f_lo) < 300.0 and prom < 20.0:
            continue  # narrow + weak = noise spike (CW survives: prom huge)
        if prom < (3.0 if wide else 8.0):
            continue  # flat wideband (OFDM) has low per-bin prominence
        out.append((f_lo, f_hi, peak))
    return out


def detect_regions(iq: np.ndarray, sample_rate_hz: float,
                   guard_bins: int = 4, train_bins: int = 16,
                   pfa_factor: float = 6.0,
                   min_bandwidth_hz: float = 200.0,
                   stft_nperseg: int = 1024) -> list[Region]:
    """CFAR-style energy detection over Welch PSD, time-gated by STFT burst check."""
    nperseg = min(8192, max(1024, len(iq) // 8))
    # NOTE: detrend=False is load-bearing. The default per-segment constant
    # detrending destroys keyed-carrier line structure in multi-signal
    # captures (verified: CW band share 3.8% vs true 33% with detrending).
    # Global DC is already removed at ingest, so detrending buys nothing.
    freqs, psd = welch(iq, fs=sample_rate_hz, nperseg=nperseg,
                       return_onesided=False, scaling="spectrum",
                       detrend=False)
    order = np.argsort(freqs)
    freqs, psd = freqs[order], psd[order]
    psd_db = 10 * np.log10(psd + 1e-18)

    med = float(np.median(psd_db))
    mad = float(np.median(np.abs(psd_db - med)) + 1e-9)
    cfar = med + pfa_factor * 1.4826 * mad
    groups = _group(freqs, psd_db, cfar, guard_bins, med)
    if not groups:
        # Fallback for flat wideband signals (OFDM) that lift the median and
        # defeat the CFAR threshold: retry at (90th pct − 3 dB).
        groups = _group(freqs, psd_db,
                        float(np.percentile(psd_db, 90)) - 3.0,
                        guard_bins, med)

    # Time gating via STFT: region must be active for >=20% of time frames
    f_stft, t_stft, Z = stft(iq, fs=sample_rate_hz, nperseg=stft_nperseg,
                             noverlap=stft_nperseg // 2,
                             return_onesided=False)
    f_stft = np.fft.fftshift(np.fft.fftfreq(stft_nperseg, 1 / sample_rate_hz))
    Z = np.fft.fftshift(Z, axes=0)
    mag = np.abs(Z)
    mag_med = float(np.median(mag))
    mag_thresh = mag_med + 4.0 * float(np.median(np.abs(mag - mag_med)))

    regions: list[Region] = []
    dur = len(iq) / sample_rate_hz
    for (f_lo, f_hi, peak) in groups:
        if f_hi - f_lo < min_bandwidth_hz:
            f_c = (f_lo + f_hi) / 2
            f_lo, f_hi = f_c - min_bandwidth_hz / 2, f_c + min_bandwidth_hz / 2
        rows = np.where((f_stft >= f_lo) & (f_stft <= f_hi))[0]
        if len(rows):
            active = (mag[rows, :].max(axis=0) > mag_thresh)
            frac = float(active.mean())
            on_idx = np.where(active)[0]
            if len(on_idx) and frac >= 0.2:
                t0, t1 = float(t_stft[on_idx[0]]), float(t_stft[on_idx[-1]])
            else:
                t0, t1, frac = 0.0, dur, float(frac)
                if frac < 0.05:
                    continue  # likely noise spike
        else:
            t0, t1 = 0.0, dur
        regions.append(Region(f_low_hz=f_lo, f_high_hz=f_hi,
                              t_start_s=t0, t_end_s=t1, peak_db=peak))
    regions.sort(key=lambda r: r.peak_db, reverse=True)
    return regions[:8]


def snr_of_region(iq: np.ndarray, sample_rate_hz: float, r: Region) -> float:
    """Excess in-band power over noise, referred to full-band SNR.

    Welch `spectrum` scaling: bins sum to total power, so
    (mean_in/median_out − 1) × N_in/N estimates the wideband SNR.
    """
    from scipy.signal import welch as _welch
    nperseg = min(8192, max(1024, len(iq) // 8))
    freqs, psd = _welch(iq, fs=sample_rate_hz, nperseg=nperseg,
                        return_onesided=False, scaling="spectrum",
                        detrend=False)
    order = np.argsort(freqs)
    freqs, psd = freqs[order], psd[order]
    inband = (freqs >= r.f_low_hz) & (freqs <= r.f_high_hz)
    n_in = int(inband.sum())
    if n_in == 0 or (~inband).sum() == 0:
        return -99.0
    # Full-region power minus noise expectation: dilution-free, since
    # noise-only bins inside the region subtract out exactly.
    p_in = float(np.sum(psd[inband]) + 1e-18)
    # Noise reference = 25th percentile outside the region — the median can
    # be contaminated by OTHER signals in multi-signal captures.
    p_out = float(np.percentile(psd[~inband], 25) + 1e-18)
    n_total = len(freqs)
    sig_power = max(p_in - p_out * n_in, 1e-12)
    snr_true = sig_power / (p_out * n_total)
    return float(10 * np.log10(snr_true + 1e-12))
