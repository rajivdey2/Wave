"""Demod BER checks against generator bit-level ground truth."""
import numpy as np


def _baseband(mode: str, snr_db: float, seed: int, cf_err_hz: float = 0.0):
    from src.synthesis.generator import generate
    iq, bw, rs, tx = generate(mode, 100_000.0, 0.5, 10_000.0, snr_db,
                              seed=seed, return_bits=True)
    assert tx is not None
    t = np.arange(len(iq)) / 100_000.0
    bb = iq * np.exp(-1j * 2 * np.pi * (10_000.0 + cf_err_hz) * t)
    return bb, rs, tx


def test_bpsk_ber_zero_at_high_snr():
    from src.demod import demod_bpsk, min_ber
    bb, rs, tx = _baseband("BPSK", 20.0, seed=11)
    sps = int(round(100_000.0 / rs))
    _, rx = demod_bpsk(bb, sps)
    ber, rot = min_ber(rx, tx, 2)
    assert ber == 0.0, f"BPSK@20dB BER {ber} (rot {rot})"


def test_qpsk_ber_zero_at_high_snr():
    from src.demod import demod_qpsk, min_ber_qpsk
    bb, rs, tx = _baseband("QPSK", 20.0, seed=12)
    sps = int(round(100_000.0 / rs))
    syms, _ = demod_qpsk(bb, sps)
    ber, rot = min_ber_qpsk(syms, tx)
    assert ber == 0.0, f"QPSK@20dB BER {ber} (rot {rot})"


def test_qpsk_survives_carrier_error():
    """30 Hz residual center error (typical estimator miss) must not break demod."""
    from src.demod import demod_qpsk, min_ber_qpsk
    bb, rs, tx = _baseband("QPSK", 10.0, seed=13, cf_err_hz=30.0)
    sps = int(round(100_000.0 / rs))
    syms, _ = demod_qpsk(bb, sps)
    ber, _ = min_ber_qpsk(syms, tx)
    # 1 bit in 2000 at 10 dB is theory-normal noise; without the wipe this is ~0.5
    assert ber < 0.01, f"QPSK@10dB+30Hz BER {ber}"


def test_qpsk_ber_degrades_gracefully():
    from src.demod import demod_qpsk, min_ber_qpsk
    bb, rs, tx = _baseband("QPSK", 5.0, seed=14)
    sps = int(round(100_000.0 / rs))
    syms, _ = demod_qpsk(bb, sps)
    ber, _ = min_ber_qpsk(syms, tx)
    print(f"\nQPSK@5dB BER {ber:.4f}")
    assert ber < 0.05, f"QPSK@5dB BER {ber} too high"
