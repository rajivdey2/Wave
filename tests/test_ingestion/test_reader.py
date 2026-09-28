def test_ingestion_roundtrip(tmp_path):
    import numpy as np
    from src.synthesis import generate, write_iq_cf32
    from src.ingestion import load_capture

    iq, _, _ = generate("BPSK", 100_000.0, 0.2, 10_000.0, 20.0, seed=1)
    p = tmp_path / "t.iq"
    write_iq_cf32(p, iq)
    cap = load_capture(p, sample_rate_hz=100_000.0)
    assert cap.samples.dtype == np.complex64
    assert abs(cap.sample_rate_hz - 100_000.0) < 1e-6


def test_format_failure():
    import pytest
    from src.ingestion import load_capture, FormatError
    with pytest.raises(FormatError):
        load_capture("nonexistent.iq", sample_rate_hz=100_000.0)
