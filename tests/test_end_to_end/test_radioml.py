"""Recorded-RF validation: skips cleanly when the dataset is absent."""
from pathlib import Path

PKL = Path("data/radioml/RML2016.10a_dict.pkl")


def test_radioml_recorded_rf():
    import pytest
    if not PKL.exists():
        pytest.skip("RadioML pkl absent — run scripts/download_radioml.py first")
    import sys
    sys.path.insert(0, "scripts")
    from eval_radioml import load_captures, FAMILY, FS_HZ
    from src.report import analyze_file
    import tempfile, os
    from src.synthesis.generator import write_iq_cf32

    caps = load_captures(PKL, snrs=[10], per_capture=64)
    assert caps, "no captures built from RadioML pkl"
    ok, n = 0, 0
    for fam, snr, iq in caps:
        with tempfile.NamedTemporaryFile(suffix=".iq", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            write_iq_cf32(Path(tmp_path), iq)
            rep = analyze_file(tmp_path, sample_rate_hz=FS_HZ)
        finally:
            os.unlink(tmp_path)
        # Structural contract holds on recorded RF too
        for d in rep.detections:
            assert d.modulation.evidence, "evidence must always be present"
        pred = rep.detections[0].modulation.value if rep.detections else "NONE"
        ok += (pred == fam)
        n += 1
    acc = ok / n
    print(f"\nRadioML@10dB recorded-RF family accuracy: {acc:.2f} ({ok}/{n})")
    assert acc >= 0.30, f"recorded-RF accuracy {acc:.2f} below 0.30 floor"
