# Demo script (rehearse: spectrogram first, then report, then architecture)

1. **Annotated spectrogram** (`data/synthetic/MULTI_spectrogram.png`) on the
   projector: 3 boxes — BPSK + CW + FM in ONE capture, labels show
   `modulation (confidence-state, %)`. Dashed = inferred, red = unknown.
2. **JSON report**: pick any field, read its `evidence` string out loud —
   this answers "why do you believe this?" without improvising.
3. **Accuracy-vs-SNR table** (`pytest tests/test_end_to_end -s`): 100% ≥5 dB,
   measured — not claimed. Point at the manifest row behind any number.
4. **Noise file**: pure-noise input → `UNKNOWN`, zero confident guesses.
   "Our system knows what it doesn't know."
5. **SigMF file** (`MULTI_report.sigmf-meta`): standards-compliant annotation,
   validates against the public SigMF schema.
6. **Offline**: `uvicorn` serve + full pipeline with networking disabled,
   CPU-only. No cloud, no GPU.

Q&A armor:
- 5 overlapping signals? → State the detection limit honestly: CFAR groups
  merge signals closer than ~1.5 kHz; show the guard parameter, don't bluff.
- Low-SNR accuracy? → The table, plus the refusal floor at 0 dB.
- Why classical + ML-in-spirit instead of end-to-end DL? → Interpretability,
  no real-capture labeled dataset exists, CPU/offline inference.
- Real RF validation? → Honest: synthetic analog + (roadmap) RadioML recorded
  RF for digital modes; state which is which.
- Unknown format? → 422 with explicit error, never a plausible-looking
  garbage report (demo with a corrupt file).
