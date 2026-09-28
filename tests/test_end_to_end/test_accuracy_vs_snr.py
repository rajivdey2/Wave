import json
from pathlib import Path


def test_accuracy_vs_snr():
    """Accuracy-vs-SNR table from synthetic manifest (ARCHITECTURE.md §5/§6)."""
    from src.report import analyze_file

    synth = Path("data/synthetic")
    manifest_path = synth / "manifest.json"
    if not manifest_path.exists():
        import pytest
        pytest.skip("synthetic dataset not generated yet")
    manifest = json.loads(manifest_path.read_text())
    rows = []
    for m in manifest:
        if m["modulation"] in ("NOISE",) or m["modulation"].startswith("MULTI"):
            continue
        rep = analyze_file(synth / m["file"],
                           sample_rate_hz=m["sample_rate_hz"])
        pred = rep.detections[0].modulation.value if rep.detections else "NONE"
        # coarse family match: FSK2→FSK, BPSK/QPSK→PSK, QAM16→QAM
        true = m["modulation"]
        fam = {"FSK2": "FSK", "BPSK": "PSK", "QPSK": "PSK",
               "QAM16": "QAM"}.get(true, true)
        rows.append((m["snr_db"], true, pred, pred == fam or pred == true))
    assert rows, "no manifest rows evaluated"
    # At high SNR (≥10dB) require ≥50% family accuracy — honest floor for Day-1
    hi = [r for r in rows if r[0] >= 10]
    acc = sum(1 for r in hi if r[3]) / len(hi)
    print(f"\naccuracy@>=10dB: {acc:.2f} over {len(hi)} files")
    for snr, true, pred, ok in sorted(rows):
        print(f"  {snr:>5}dB {true:>6} -> {pred:<7} {'OK' if ok else 'MISS'}")
    assert acc >= 0.5, f"high-SNR accuracy {acc:.2f} below 0.5 floor"


def test_unknown_on_noise():
    from src.report import analyze_file
    p = Path("data/synthetic/NOISE_cf32.iq")
    if not p.exists():
        import pytest
        pytest.skip("synthetic dataset not generated yet")
    rep = analyze_file(p, sample_rate_hz=100_000.0)
    # Either zero detections or all modulations UNKNOWN
    assert all(d.modulation.state.value == "unknown" for d in rep.detections), \
        "noise must not produce confident labels"


def test_multi_signal_segmentation():
    from src.report import analyze_file
    p = Path("data/synthetic/MULTI_10dB_cf32.iq")
    if not p.exists():
        import pytest
        pytest.skip("synthetic dataset not generated yet")
    rep = analyze_file(p, sample_rate_hz=100_000.0)
    assert len(rep.detections) >= 2, \
        f"multi-signal file yielded {len(rep.detections)} detections, need ≥2"
