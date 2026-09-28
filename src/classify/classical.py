"""Classical modulation discriminators — interpretable, fast, evidence-rich.

Decision order (each step rules out one family with its signature feature):
  UNKNOWN → CHIRP → CW → SSB → OFDM → AM → QAM-levels → PSK-loop/FM/FSK/CW
"""
from __future__ import annotations

import numpy as np

from src.types import ConfidenceState, FieldValue

UNKNOWN_SNR_FLOOR = 0.0


def _infer(label: str, conf: float, evidence: str) -> tuple[FieldValue, str]:
    return FieldValue(value=label, state=ConfidenceState.INFERRED,
                      confidence=round(float(conf), 2), evidence=evidence), "classical"


def _unknown(reason: str) -> tuple[FieldValue, str]:
    return FieldValue(value="UNKNOWN", state=ConfidenceState.UNKNOWN,
                      confidence=None, evidence=reason), "classical"


def isolate_region(bb: np.ndarray, sample_rate_hz: float,
                   f_lo_hz: float, f_hi_hz: float) -> np.ndarray:
    """Brickwall (tapered) bandpass around a detected region.

    Mandatory for multi-signal captures: without it, every region's
    baseband contains all other signals in the capture and the statistical
    features (envelope variance, constellation, discriminator) are polluted.
    """
    n = len(bb)
    F = np.fft.fft(bb)
    freqs = np.fft.fftfreq(n, 1 / sample_rate_hz)
    # Raised-cosine rolloff (300 Hz) evaluated in FREQUENCY space. (A previous
    # version tapered in FFT-index space, which wraps around DC and zeroed
    # carriers — index-space tapering of a baseband passband is just wrong.)
    roll = 300.0
    H = np.zeros(n)
    inner = (freqs >= f_lo_hz + roll) & (freqs <= f_hi_hz - roll)
    H[inner] = 1.0
    lo_edge = (freqs >= f_lo_hz) & (freqs < f_lo_hz + roll)
    H[lo_edge] = 0.5 - 0.5 * np.cos(np.pi * (freqs[lo_edge] - f_lo_hz) / roll)
    hi_edge = (freqs > f_hi_hz - roll) & (freqs <= f_hi_hz)
    H[hi_edge] = 0.5 + 0.5 * np.cos(np.pi * (freqs[hi_edge] - (f_hi_hz - roll)) / roll)
    return np.fft.ifft(F * H)


def _carrier_asymmetry(seg: np.ndarray, sample_rate_hz: float) -> float:
    """One-sidedness around the strongest tone (carrier).

    SSB puts all energy on ONE side of its carrier; AM mirrors on both; a
    lone CW carrier has neither. Compares ±(50..500 Hz) side windows so the
    carrier itself never leaks into the measurement. Returns max/min ratio.
    """
    n = min(len(seg), 1 << 15)
    N = 4 * n
    spec = np.abs(np.fft.fft(seg[:n], N)) ** 2
    freqs = np.fft.fftfreq(N, 1 / sample_rate_hz)
    order = np.argsort(freqs)
    spec, freqs = spec[order], freqs[order]
    fc = float(freqs[int(np.argmax(spec))])
    up = (freqs >= fc + 50) & (freqs <= fc + 500)
    lo = (freqs >= fc - 500) & (freqs <= fc - 50)
    if not up.any() or not lo.any():
        return 1.0
    # Peak (not median): sideband energy is a narrow tone that median would
    # erase; noise spikes set the floor in the empty window instead.
    u = float(spec[up].max())
    l = float(spec[lo].max())
    return max(u, l) / (min(u, l) + 1e-18)


def _amp_levels(amp_n: np.ndarray) -> int:
    """Distinct amplitude levels via smoothed histogram peaks (PSK≈1, QAM≥3)."""
    hist, _ = np.histogram(amp_n, bins=60, range=(0, 2.5))
    hist = hist.astype(float)
    sm = np.convolve(hist, np.ones(3) / 3, mode="same")
    sm /= sm.max() + 1e-18
    peaks = 0
    for i in range(1, len(sm) - 1):
        if sm[i] > 0.20 and sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1]:
            peaks += 1
    return peaks


def _nonlinear_coherence(seg: np.ndarray, power: int) -> float:
    """|E[s^p]| with residual-carrier wipe via FFT peak (squaring loop generalized).

    BPSK: power=2 → ≈1. QPSK: power=4 → ≈1. Other modes → ≈0.
    Without the wipe, even a few Hz of center error averages the tone out.
    """
    n = len(seg)
    sp = seg ** power
    F = np.abs(np.fft.fft(sp)) ** 2
    k = int(np.argmax(F))
    f_res = (k / n) if k <= n // 2 else ((k - n) / n)  # cycles/sample
    t = np.arange(n)
    wiped = sp * np.exp(-1j * 2 * np.pi * f_res * t)
    denom = float(np.mean(np.abs(seg) ** power) + 1e-18)
    return float(np.abs(np.mean(wiped)) / denom)


def _envelope_comb_count(seg: np.ndarray, sample_rate_hz: float) -> int:
    """Envelope-spectrum peaks in 10..200 Hz: keyed-CW shows a COMB (5 Hz
    keying → lines at 5, 15, 25, ...); AM shows 1 message tone. Count ≥4 →
    keyed carrier, not AM."""
    env = np.abs(seg) - np.abs(seg).mean()
    n = len(env)
    espec = np.abs(np.fft.rfft(env)) ** 2
    efreq = np.fft.rfftfreq(n, 1 / sample_rate_hz)
    m = (efreq >= 3) & (efreq <= 200)
    if not m.any():
        return 0
    # Significance gate: without a dominant structure, relative peak counts
    # just enumerate noise bumps (this stole BPSK at 6 dB, QAM16 at 10 dB).
    if float(espec[m].max() / (np.median(espec[m]) + 1e-18)) < 30.0:
        return 0
    s = espec[m] / (espec[m].max() + 1e-18)
    count = 0
    for i in range(1, len(s) - 1):
        if s[i] > 0.15 and s[i] >= s[i - 1] and s[i] >= s[i + 1]:
            count += 1
    return count


def _disc_tone_ratio(seg: np.ndarray, sample_rate_hz: float) -> float:
    """Peakiness of the discriminator-output spectrum.

    Sine-FM's discriminator is a pure message tone (ratio ≫ 1000); 2-FSK's is
    a random NRZ square wave — broadband with nulls (ratio ≪ 1000).
    """
    inst = np.diff(np.unwrap(np.angle(seg + 1e-12)))
    k = np.ones(8) / 8
    d = np.convolve(inst, k, mode="same")
    n = len(d)
    S = np.abs(np.fft.rfft(d)) ** 2
    f = np.fft.rfftfreq(n, 1 / sample_rate_hz)
    m = (f >= 50) & (f <= 2000)
    if not m.any():
        return 0.0
    return float(S[m].max() / (np.median(S[m]) + 1e-18))


def _constellation_cells(seg: np.ndarray) -> int:
    """Occupied cells on a 10×10 grid after 4th-power carrier wipe.

    BPSK→~2, QPSK→~4, QAM16→≥9 at ≥10 dB. Timing-free (uses all samples
    decimated ÷4; transition samples spread thin, clusters dominate).
    """
    n = len(seg)
    sp = seg ** 4
    F = np.abs(np.fft.fft(sp)) ** 2
    k = int(np.argmax(F))
    f4 = (k / n) if k <= n // 2 else ((k - n) / n)
    t = np.arange(n)
    wiped = seg * np.exp(-1j * 2 * np.pi * (f4 / 4) * t)
    sub = wiped[::4]
    # Robust span (98th pct): max|·| is set by noise spikes/ringing and would
    # cram the clusters into the center cells.
    span = float(np.percentile(np.abs(sub), 98) + 1e-12)
    gx = np.clip(sub.real / span, -1, 1)
    gy = np.clip(sub.imag / span, -1, 1)
    H, _, _ = np.histogram2d(gx, gy, bins=10,
                             range=[[-1, 1], [-1, 1]])
    return int(np.sum(H >= max(2, 0.003 * len(sub))))


def classify_classical(bb: np.ndarray, snr_db: float,
                       sample_rate_hz: float = 100_000.0,
                       bandwidth_hz: float | None = None) -> tuple[FieldValue, str]:
    if snr_db < UNKNOWN_SNR_FLOOR:
        return _unknown(f"SNR {snr_db:.1f}dB below floor — refused to guess")

    bb = np.asarray(bb).ravel()
    seg = bb[:min(len(bb), 1 << 15)]
    amp = np.abs(seg)
    amp_n = amp / (np.mean(amp) + 1e-12)
    sa = float(np.std(amp_n))
    phase = np.unwrap(np.angle(seg + 1e-12))
    inst = np.diff(phase)
    sf = float(np.std(inst))
    papr = float(np.max(amp ** 2) / (np.mean(amp ** 2) + 1e-18))

    # 1. CHIRP: linear frequency ramp. Phase noise swamps per-sample inst.
    # freq at 10 dB, so smooth heavily first (ramp survives, noise averages out).
    inst_full = np.diff(np.unwrap(np.angle(bb + 1e-12)))
    w = 256
    ker = np.ones(w) / w
    sm = np.convolve(inst_full, ker, mode="same")[::4]
    x = np.arange(len(sm))
    if len(x) > 200:
        A = np.vstack([x, np.ones_like(x)]).T
        coef, res, *_ = np.linalg.lstsq(A, sm, rcond=None)
        ss_tot = float(np.sum((sm - sm.mean()) ** 2) + 1e-18)
        r2 = float(1 - (res[0] if len(res) else 0) / ss_tot)
        # slope threshold in rad/sample² scaled back up by decimation
        if r2 > 0.5 and abs(float(coef[0])) * 4 > 1e-7:
            return _infer("CHIRP", min(0.95, 0.6 + r2 * 0.3),
                          f"linear freq ramp R²={r2:.2f} — chirp signature")

    # 2. SSB: all energy on one side of the carrier (checked before the
    # narrowband CW rule — SSB carrier+sideband is narrow too).
    cas = _carrier_asymmetry(seg, sample_rate_hz)
    if cas > 5.0 and (bandwidth_hz or 1e9) < 3000:
        return _infer("SSB", 0.7,
                      f"energy {cas:.1f}x one-sided around carrier — SSB")

    # 3. CW: nothing else is this narrowband (keyed CW carries an envelope
    # tone like AM, so bandwidth — not envelope — is the discriminator).
    if bandwidth_hz is not None and bandwidth_hz < 150:
        return _infer("CW", 0.85,
                      f"bandwidth {bandwidth_hz:.0f}Hz <150Hz — carrier")
    if bandwidth_hz is None and sf < 0.05 and sa < 0.35:
        return _infer("CW", 0.8,
                      f"near-constant envelope (σA={sa:.3f}), stable freq — carrier")

    # 4. OFDM: high PAPR + wideband
    if papr > 6.0 and (bandwidth_hz or 0) > 10_000:
        return _infer("OFDM", 0.75,
                      f"PAPR {papr:.1f}>6 over {bandwidth_hz:.0f}Hz — multicarrier")

    # 5. AM: envelope carries a low-rate tone + mirrored sidebands.
    # Keyed-CW check first: a COMB of envelope lines (5 Hz keying) is a
    # keyed carrier, not AM's single message tone.
    if _envelope_comb_count(seg, sample_rate_hz) >= 4:
        return _infer("CW", 0.7,
                      "envelope-line comb — keyed carrier (CW)")
    env = amp - amp.mean()
    n = len(env)
    espec = np.abs(np.fft.rfft(env)) ** 2
    efreq = np.fft.rfftfreq(n, 1 / sample_rate_hz)
    m = (efreq >= 20) & (efreq <= 1000)
    if m.any():
        eratio = float(espec[m].max() / (np.median(espec[m]) + 1e-18))
        if eratio > 15 and cas < 5.0 and sa > 0.25:
            f0 = float(efreq[m][int(np.argmax(espec[m]))])
            # Persistence: a real message tone repeats in both halves;
            # a noise bump (e.g. BPSK at 6 dB) doesn't.
            h = len(env) // 2
            ok = True
            for part in (env[:h], env[h:]):
                S = np.abs(np.fft.rfft(part)) ** 2
                f = np.fft.rfftfreq(len(part), 1 / sample_rate_hz)
                mm = (f >= 20) & (f <= 1000)
                kk = int(np.argmin(np.abs(f[mm] - f0)))
                if float(S[mm][kk] / (np.median(S[mm]) + 1e-18)) < 8.0:
                    ok = False
            if ok:
                return _infer("AM", 0.75,
                              f"envelope tone at {f0:.0f}Hz (ratio {eratio:.0f}), mirrored sidebands — AM")

    # 6. QAM: ≥3 distinct amplitude rings (before constant-envelope branch,
    # since noise ripple can push QAM's σA below naive thresholds).
    # SNR-gated: at low SNR, noise alone fakes extra rings (BPSK at 6 dB
    # shows 4 phantom levels) — QAM needs clean rings to earn this shortcut.
    n_levels = _amp_levels(amp_n)
    if snr_db >= 8.0 and n_levels >= 3 and sa > 0.2:
        return _infer("QAM", 0.7,
                      f"{n_levels} amplitude levels — QAM")

    # 7. Constant-envelope family via carrier-wiped nonlinear coherence.
    # p2: BPSK ≈ 0.6–0.9, everything else ≤ 0.16. p4: QPSK ≈ 0.55–0.9,
    # QAM16 ≈ 0.35–0.6 (overlaps QPSK — constellation breaks the tie).
    p2 = _nonlinear_coherence(seg, 2)
    p4 = _nonlinear_coherence(seg, 4)
    noise_ripple = float(10 ** (-snr_db / 20))
    if sa < 0.25 + noise_ripple:
        if p2 > 0.45:
            # FM's Bessel-line structure can fake p2 up to ~0.4; a true BPSK
            # collapses to few clusters while FM's ring fills the grid.
            cells = _constellation_cells(seg)
            if cells < 7:
                return _infer("PSK", 0.8,
                              f"squaring-loop coherence {p2:.2f}>0.45, {cells} clusters — BPSK-like")
            if p4 > 0.62:
                # Smeared but coherent (multi-signal/noise) → still PSK.
                # FM's p4 stays ≤ ~0.2, so this gate is FM-safe.
                return _infer("PSK", 0.7,
                              f"squaring-loop {p2:.2f}, 4th-power {p4:.2f} — PSK")
            # else: ring constellation + incoherent → fall to FM/FSK disc test
        elif p4 > 0.62:
            return _infer("PSK", 0.8,
                          f"4th-power coherence {p4:.2f}>0.62 — QPSK-like")
        if p4 > 0.30:
            cells = _constellation_cells(seg)
            if cells >= 7:
                return _infer("QAM", 0.7,
                              f"{cells} constellation cells — QAM")
            return _infer("PSK", 0.7,
                          f"{cells} constellation cells, 4th-power {p4:.2f} — PSK")
        if _disc_tone_ratio(seg, sample_rate_hz) > 1000.0:
            return _infer("FM", 0.75,
                          f"constant envelope, discriminator is a pure tone — FM")
        return _infer("FSK", 0.7,
                      f"constant envelope, spread discriminator — FSK")
    # 8. Non-constant envelope that escaped the levels test: cells decide.
    cells = _constellation_cells(seg)
    if cells >= 7:
        return _infer("QAM", 0.65,
                      f"{cells} constellation cells — QAM")
    if cells <= 4 and p2 > 0.25:
        return _infer("PSK", 0.6,
                      f"{cells} constellation cells — PSK")
    if n_levels <= 2:
        return _infer("PSK", 0.6,
                      f"{n_levels} amplitude level(s), phase-modulated — PSK")
    return _infer("QAM", 0.6,
                  f"{n_levels} amplitude levels — QAM")
