# SIH26147 — Automated .IQ/.WAV Signal Analysis & Parameter Extraction

**Live demo:** https://wave-kqmh.onrender.com (landing page at `/`, console at `/app`)

DSP + classical classification pipeline for NTRO-style radio captures.
Ingests raw IQ / WAV, detects every signal present, extracts parameters
(center freq, bandwidth, SNR, symbol rate), classifies modulation with
**evidence + confidence on every field**, and reports JSON + SigMF + annotated
spectrogram. Runs fully offline, CPU-only.

## Quickstart

```bash
pip install -r requirements.txt
python scripts/generate_synthetic.py          # labeled test captures + manifest
pytest tests/ -q                              # accuracy-vs-SNR, noise, multi
uvicorn src.api.app:app --port 8000           # POST /analyze
```

Analyze one file:

```bash
PYTHONPATH=. python -c "
from src.report import analyze_file, write_json_report, write_sigmf, render_spectrogram
r = analyze_file('data/synthetic/BPSK_10dB_cf32.iq', sample_rate_hz=100000.0)
write_json_report(r, 'report.json'); write_sigmf(r, 'report.sigmf-meta')
render_spectrogram(r, 'spectrogram.png')"
```

## Measured performance (synthetic, `pytest tests/test_end_to_end -s`)

| SNR | accuracy (family) |
|-----|-------------------|
| +20/+15/+10 dB | **100%** (30/30) |
| +5 dB | **100%** (10/10) |
| 0 dB | 80% (8/10) |
| −5 dB | explicit `UNKNOWN` refusal (no confident wrong guesses) |

Multi-signal capture (BPSK + CW + FM in one file): 3/3 segmented correctly.
Pure noise: zero confident labels.

## Web Interface (Landing Page)

Open `index.html` in a browser to access the interactive landing page with:
- **3D Wave Visualization** — Real-time animated 3D wave model representing RF signals
- **Launch App Button** — Opens the FastAPI analysis interface
- **Live Analysis Results** — View signal detections with confidence states and evidence

## Deploy on Render

1. Push this repo to GitHub (`venv/` is git-ignored — never commit it).
2. In Render: **New → Blueprint** → select the repo.
   `render.yaml` wires everything automatically (build, start, health check).
   Manual alternative: New Web Service with build `pip install -r requirements.txt`,
   start `uvicorn src.api.app:app --host 0.0.0.0 --port $PORT`, health check `/health`.
3. Open the service URL: `/` serves the landing page, `/app` serves the console.
   No extra config or env vars needed — demos ship in `data/synthetic/`.

## Layout

See `docs/ARCHITECTURE.md` (technical contract) and `docs/CLAUDE.md` (strategy/build order).

## Known limitations (honest)

- Keyed-CW below ~3 dB per-signal SNR in multi-signal mixes can read as PSK
  (squaring coherence of on/off keying ≈ 1.0); isolated CW is fine to 0 dB.
- QPSK/QAM16 split degrades gracefully below ~5 dB (constellation smear).
- Symbol rate reported only for PSK/QAM/FSK; N/A for analog/OFDM/chirp by design.

## Architecture Overview

```
.IQ/.WAV Input → Ingestion → Detection → Extraction → Classification → Report
                      ↓           ↓           ↓             ↓           ↓
                 Normalize   CFAR/STFT  Params      Classical+ML  JSON+SigMF
                                                    ↓             ↓
                                               Confidence    Spectrogram
                                               Evidence      (Annotated)
```

## Output Formats

1. **JSON Report** — Structured `CaptureReport` with all detections, parameters, confidence states, and evidence strings
2. **SigMF Annotation** (`.sigmf-meta`) — Standards-compliant annotations for interoperability
3. **Annotated Spectrogram** — PNG with bounding boxes color-coded by confidence state
