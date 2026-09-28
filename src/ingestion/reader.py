"""Ingestion: parse .iq/.cfile/.wav into normalized complex64 arrays."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

VALID_SUFFIXES = {".iq", ".cfile", ".wav", ".bin"}


class FormatError(ValueError):
    """Explicit format failure — caller maps to API 422, never a garbage report."""


@dataclass
class Capture:
    samples: np.ndarray          # complex64, DC-removed, RMS-normalized optional
    sample_rate_hz: float
    format: str                  # "iq_cf32" | "iq_int16" | "wav"
    source_file: str
    duration_s: float


def _read_wav(path: Path) -> tuple[np.ndarray, float]:
    from scipy.io import wavfile
    try:
        sr, data = wavfile.read(str(path))
    except Exception as e:
        raise FormatError(f"unrecognized WAV file: {e}") from e
    data = np.asarray(data)
    if data.size == 0:
        raise FormatError("empty WAV file")
    if data.ndim == 2:
        # stereo/mono: I=ch0, Q=ch1 if 2ch else real-only
        if data.shape[1] >= 2:
            iq = data[:, 0].astype(np.float64) + 1j * data[:, 1].astype(np.float64)
        else:
            iq = data[:, 0].astype(np.float64) + 0j
    else:
        iq = data.astype(np.float64) + 0j
    # scale int16/int32 to [-1, 1]
    if np.issubdtype(data.dtype, np.integer):
        iq = iq / float(np.iinfo(data.dtype).max)
    return iq.astype(np.complex64), float(sr)


def _read_raw_iq(path: Path, sample_rate_hz: float | None,
                 dtype_hint: str | None) -> tuple[np.ndarray, str]:
    raw = path.read_bytes()
    if len(raw) < 16:
        raise FormatError("file too small to be an IQ capture")
    n = len(raw)
    # Auto-detect: try cf32 (8 bytes/sample) vs int16 (4 bytes/sample).
    # Heuristic: if dtype_hint given, honor it; else prefer cf32 when the
    # byte length is not divisible into a sane int16 magnitude distribution.
    # Simplest robust rule: explicit hint wins, else try cf32 first and
    # validate with a finite/RMS sanity check.
    candidates: list[str] = []
    if dtype_hint in ("cf32", "float32"):
        candidates = ["cf32"]
    elif dtype_hint in ("int16", "iq_int16"):
        candidates = ["int16"]
    else:
        candidates = ["cf32", "int16"] if n % 8 == 0 else ["int16"]

    last_err: Exception | None = None
    for cand in candidates:
        try:
            if cand == "cf32":
                if n % 8 != 0:
                    raise FormatError("byte length not divisible by 8 for cf32")
                arr = np.frombuffer(raw, dtype="<f4").astype(np.float64)
                iq = arr[0::2] + 1j * arr[1::2]
            else:
                if n % 4 != 0:
                    raise FormatError("byte length not divisible by 4 for int16")
                arr = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
                iq = arr[0::2] + 1j * arr[1::2]
            if iq.size == 0 or not np.all(np.isfinite(iq)):
                raise FormatError(f"parsed {cand} produced no finite samples")
            return iq.astype(np.complex64), (
                "iq_cf32" if cand == "cf32" else "iq_int16")
        except FormatError as e:
            last_err = e
    raise FormatError(f"unrecognized format: {last_err}")


def load_capture(path: str | Path, sample_rate_hz: float | None = None,
                 dtype_hint: str | None = None,
                 normalize: bool = True) -> Capture:
    """Load any supported capture. Fails loudly (FormatError) on unknown input."""
    p = Path(path)
    if not p.exists():
        raise FormatError(f"file not found: {path}")
    suffix = p.suffix.lower()
    if suffix not in VALID_SUFFIXES:
        raise FormatError(
            f"unrecognized format '{suffix}': expected one of {sorted(VALID_SUFFIXES)}")
    if suffix == ".wav":
        iq, sr = _read_wav(p)
        fmt = "wav"
    else:
        if sample_rate_hz is None:
            raise FormatError(
                "sample_rate_required: raw .iq/.cfile has no header; "
                "pass sample_rate_hz explicitly")
        iq, fmt = _read_raw_iq(p, sample_rate_hz, dtype_hint)
        sr = float(sample_rate_hz)

    # DC offset removal (a top source of phantom "accuracy" bugs)
    iq = iq - np.mean(iq)
    if normalize:
        rms = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
        if rms > 1e-12:
            iq = (iq / rms).astype(np.complex64)

    return Capture(samples=iq, sample_rate_hz=float(sr), format=fmt,
                   source_file=str(p), duration_s=len(iq) / float(sr))


def quality_metrics(iq: np.ndarray) -> tuple[float, float]:
    """Return (noise_floor_db, dynamic_range_db) from power spectrum percentiles."""
    from scipy.signal import welch
    _, psd = welch(iq, nperseg=min(4096, len(iq)), return_onesided=False,
                   detrend=False)
    psd_db = 10 * np.log10(psd + 1e-18)
    floor = float(np.percentile(psd_db, 10))
    peak = float(np.percentile(psd_db, 99.9))
    return floor, float(peak - floor)
