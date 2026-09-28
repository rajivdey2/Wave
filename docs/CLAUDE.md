# CLAUDE.md — SIH26147: Automated .IQ/.WAV Signal Analysis & Parameter Extraction

> Read this file end-to-end before writing any code. This is the build spec, competitive
> context, and win strategy for Smart India Hackathon 2026, Problem Statement **SIH26147**.

---

## 0. Time-critical reality check (read first)

- Portal idea-submission deadline: **30 September 2026**. Today is 22 September 2026 — **8 days left**.
- Your real deadline is almost certainly earlier: colleges run their internal SPOC hackathon
  2–3 weeks before the portal date, and only teams nominated through that internal round can
  submit. **Confirm your college's internal cutoff today** — it may already be this week.
- Idea-round deliverable is a **max-6-slide PDF** on the official AICTE PPT template, not code.
  What you build this week should be optimized for a convincing demo + defensible technical
  narrative in 6 slides, not a finished product. Treat everything below as a prioritized build
  order — stop and pitch as soon as you have Phases 0–3 solid, even if 4–6 aren't done.

---

## 1. Official problem statement (as published)

| Field | Value |
|---|---|
| PS Code | SIH26147 |
| Title | Automated model for analysis of .IQ and .wav files along with signal parameter extraction |
| Sponsoring org | National Technical Research Organisation (NTRO) |
| Track | Software |
| Theme | Space Technology (listed as "Miscellaneous" in a couple of older mirrors — Space Technology is the current official listing) |
| Cash prize | ₹1,00,000 per winning team, plus national mentorship |
| Team | Exactly 6 students, same college, ≥1 female member, up to 2 mentors |

**In plain terms:** NTRO wants a tool that ingests raw radio captures — either **.IQ files**
(complex in-phase/quadrature baseband samples, the native format for software-defined radios)
or **.wav files** (audio-band recordings) — covering **terrestrial HF, VHF and UHF signals**
from different sensors/locations, and automatically:

1. Detects and isolates every distinct signal present in a wideband capture (there is usually
   more than one).
2. Extracts its parameters: center frequency, bandwidth, on/off timing (burst structure), SNR,
   symbol rate.
3. Classifies its modulation (AM/FM/SSB/CW/FSK/PSK/QAM/OFDM/chirp and sub-variants).
4. Reports all of this with a confidence level and human-readable evidence — not a black-box
   label.

This is a **DSP + classical signal processing + ML** problem, not a generic "AI dashboard"
problem. NTRO is a signals-intelligence (SIGINT) technical agency, and a jury from that
background will probe DSP fundamentals hard. Treat this as an RF engineering project with an
ML component, not an ML project with an RF-flavored dataset.

**Ethics/legality note for your build:** you are not intercepting or processing any real
private communications. You train and demo entirely on synthetic signals you generate
yourself (GNU Radio / Python) and public, license-clean datasets like RadioML. That is exactly
what the evaluation environment for this PS will use — no real captured traffic is available
or appropriate for a student hackathon.

---

## 2. Competitive landscape (do this before you write a line of code)

This problem statement already has **unusually serious competition** for something rated
technically hard — at least six public repos exist for this exact PS as of this week, several
clearly building multi-stage DSP pipelines with names like SignalX, IQWAV, SYNAPS, Pinpoint,
and at least one team that already has its own Claude Code CLAUDE.md spec live on GitHub. The
"pick the hardest problem to dodge competition" logic that worked for previous SIH cycles does
**not** fully apply here — the difficulty is filtering for strong teams, not filtering out
casual ones. Do not assume you have a quiet field.

**What this means for strategy:**
- A basic "load IQ file → FFT → spectrogram → guess modulation with a CNN" pipeline will look
  like everyone else's. That is table stakes, not a differentiator.
- The teams pulling ahead are the ones going past classification into **demodulation,
  de-interleaving, and FEC identification** — i.e., actually recovering something from the
  signal, not just labeling it. That is a hard, achievable-in-a-week differentiator if you
  scope it to a handful of well-known FEC schemes (e.g., convolutional codes, simple
  block codes) rather than trying to be universal.
- A second differentiator that's cheap to build and jury-visible: **calibrated, explicit
  uncertainty**. Don't emit a single modulation label. Emit a label + confidence + the
  evidence that produced it (which spectral/statistical features triggered the classification),
  and be willing to say "uncertain" or "insufficient SNR" rather than guessing. Juries notice
  when a system knows what it doesn't know — most competing tools won't do this.
- A third: **SigMF-compliant output.** SigMF (sigmf.org) is the actual open standard the SDR/
  SIGINT community uses for annotating IQ recordings. Emitting valid `.sigmf-meta` annotation
  files alongside your JSON report signals real domain literacy to an NTRO jury and costs you
  almost nothing to implement (it's just a JSON schema).

---

## 3. Target system

```
┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌─────────────────┐
│ .IQ / .wav  │ → │ Ingestion &  │ → │ Signal         │ → │ Per-signal       │
│ file input  │   │ normalization│   │ detection      │   │ parameter        │
└─────────────┘   └──────────────┘   │ (segmentation) │   │ extraction       │
                                      └───────────────┘   └────────┬─────────┘
                                                                    ↓
┌──────────────┐   ┌───────────────┐   ┌────────────────┐   ┌─────────────┐
│ Report:      │ ← │ (stretch)     │ ← │ Modulation      │ ← │ Confidence & │
│ JSON + SigMF │   │ Demod + de-   │   │ classification  │   │ evidence     │
│ + annotated  │   │ interleave +  │   │ (AM/FM/SSB/CW/  │   │ scoring      │
│ spectrogram  │   │ FEC ID        │   │ FSK/PSK/QAM/    │   │              │
└──────────────┘   └───────────────┘   │ OFDM/chirp)     │   └─────────────┘
                                        └─────────────────┘
```

### 3.1 Ingestion & normalization
- Support common IQ container formats: raw interleaved `.iq`/`.cfile` (float32 or int16,
  little-endian, as GNU Radio and most SDR tools write them), plus `.wav` (via `scipy.io.wavfile`
  or `soundfile`). Auto-detect sample width/endianness where you can; otherwise expose it as a
  CLI/UI flag and fail loudly rather than silently misreading.
- DC offset removal, RMS normalization, and basic file/format validation up front — a
  surprising number of "AI accuracy" bugs in this domain are actually bad file parsing.
- Compute and surface basic quality metrics immediately: SNR estimate, noise floor, dynamic
  range. These gate everything downstream (see uncertainty framework in §6).

### 3.2 Signal detection / segmentation
- FFT-based power spectrum + STFT waterfall/spectrogram as the core visualization and
  detection substrate.
- Energy-detection / CFAR (constant false-alarm rate) thresholding over the spectrogram to find
  time-frequency regions of interest — this is the standard, well-documented technique for
  "how many signals are in this capture and where," and is far more defensible in a jury Q&A
  than a black-box detector.
- Output: a list of candidate signal "boxes" (start/end time, low/high frequency), each handed
  downstream independently. A single capture can and should yield multiple detections.

### 3.3 Per-signal parameter extraction
For each detected region: center frequency, −3 dB and −20 dB bandwidth, on/off burst timing,
SNR, carrier frequency offset, and (where applicable) symbol rate estimation (cyclostationary
or spectral-peak methods on the magnitude/phase are standard here).

### 3.4 Modulation classification
- Start with classical discriminators (constellation shape for PSK/QAM, instantaneous
  frequency/amplitude statistics for FM/AM/FSK, zero-crossing patterns for CW) — these are
  interpretable and fast, and give you the "evidence" string for free.
- Layer a lightweight ML classifier on top (a small CNN or gradient-boosted model over
  engineered features) trained on **RadioML 2016.10 / 2018.01** (public, well-known modulation
  recognition datasets) plus your own synthetic signals for AM/FM/SSB/CW, which RadioML
  doesn't cover well. Report which method (classical vs. ML) backs each classification.

### 3.5 Confidence & evidence
Every output field carries a confidence and, ideally, a state rather than a bare label. A
useful vocabulary to adopt (you'll see similar ideas in competing repos — it's a sound pattern,
implement your own version):
- **DETECTED** — directly measured from the signal (e.g., center frequency from a clear peak).
- **ESTIMATED** — computed via a standard algorithm with known error bounds (e.g., symbol rate
  from cyclostationary analysis).
- **INFERRED** — model output above your confidence threshold.
- **UNKNOWN / INSUFFICIENT SNR** — explicitly refuse to guess below your validated SNR floor.
Gate every stage on the SNR/quality metrics from §3.1 — don't let a classifier hallucinate a
confident label on pure noise. This is the single highest-leverage thing for both real accuracy
and jury credibility.

### 3.6 Stretch goals (do these only after 3.1–3.5 are solid and demoable)
- De-interleaving detection (block/convolutional/diagonal) on recovered bitstreams for the
  digital modes you do demodulate.
- Basic FEC identification (e.g., recognizing a standard convolutional code by syndrome
  behavior) — full FEC decoding is a stretch beyond a stretch; identifying the scheme is
  already a strong differentiator on its own.
- GUI (desktop or web) with an interactive spectrogram, click-to-inspect detections, and
  exportable reports. A clean web UI (FastAPI + a simple React/plain-HTML front end) demos far
  better than a CLI in a 6-slide pitch and later live judging round.

### 3.7 Outputs (what the tool actually produces)
- **JSON report** — one object per detected signal with all extracted parameters, confidence
  states, and evidence strings.
- **SigMF annotation file** (`.sigmf-meta`) — standards-compliant, signals domain literacy.
- **Annotated spectrogram image** — bounding boxes over the waterfall plot, labeled with
  modulation + confidence. This is your single best demo artifact; design it to look good on a
  projector from across a room.

---

## 4. Tech stack (opinionated, biased toward "runs offline, no exotic deps")

- **Language:** Python for the DSP/ML core (NumPy, SciPy, scikit-learn; PyTorch or a small
  Keras/TF model only if you actually need the accuracy bump — don't reach for deep learning
  reflexively).
- **DSP:** NumPy/SciPy FFT & signal module for spectra/STFT; consider GNU Radio only for
  *generating* synthetic training/test signals, not as a runtime dependency for the shipped
  tool — it's heavy to demo/install live.
- **Synthetic data generation:** GNU Radio Companion or `commpy`/custom NumPy generators to
  produce labeled AM/FM/SSB/CW/FSK/PSK/QAM/OFDM signals at controlled SNRs — you will need this
  regardless of RadioML, since RadioML doesn't cover analog modes (AM/FM/SSB/CW) well and NTRO
  explicitly lists them.
- **API/backend:** FastAPI (fast to build, easy to demo, plays well with async file uploads for
  larger captures).
- **Frontend:** keep it simple — a static page with a spectrogram canvas (Plotly.js or a
  server-rendered image) beats an over-engineered SPA for a 6-slide pitch and a live demo.
- **Packaging:** must run fully offline / air-gapped and ideally CPU-only — NTRO's context is
  sensitive-data-adjacent, and "no cloud dependency" is both a real constraint they'll ask about
  and an easy point to score.

---

## 5. Suggested build order (roughly a day each, compress as needed)

1. **Day 1 — Ingestion & synthetic data.** File parsers for `.iq`/`.cfile`/`.wav`. Signal
   generators for all required modulation types at a range of SNRs, saved as labeled test
   files. You cannot validate anything downstream without this.
2. **Day 2 — Detection & spectrogram.** FFT/STFT pipeline, energy detection, bounding boxes on
   a spectrogram. Get the visualization looking good early — it's your main demo asset.
3. **Day 3 — Parameter extraction.** Center freq, bandwidth, SNR, burst timing, symbol rate on
   detected regions.
4. **Day 4 — Classification + confidence framework.** Classical discriminators first, ML
   classifier second, confidence/evidence wrapper around both.
5. **Day 5 — Reporting (JSON + SigMF) + basic UI/API.** This is what you actually screenshot
   for the 6-slide deck.
6. **Day 6 — Polish, edge cases, and the pitch deck.** Multi-signal captures, noisy edge cases,
   a rehearsed demo script, and the 6 slides (Idea → Technical Approach → Feasibility &
   Viability → Impact & Benefits → Research & Roadmap, per the official template).
7. **Days 7–8 — Buffer.** Something in DSP will break in a way you didn't expect. Leave room.

---

## 6. Jury Q&A — likely hard questions, prepare answers for these specifically

- "How does your system handle a capture with 5 overlapping signals in frequency?" (Have a real
  answer — this is your detection stage's actual limitation; know it and state it honestly.)
- "What's your accuracy at low SNR, and how do you know?" (You need an actual measured
  accuracy-vs-SNR curve, not a single headline number — build this as part of Day 3–4.)
- "Why classical DSP + ML instead of end-to-end deep learning?" (Interpretability, no need for
  a huge labeled dataset of real captures which doesn't exist for this use case, faster
  inference on CPU-only/offline hardware.)
- "How do you validate against real-world RF, not just synthetic signals?" (Be honest: you
  validate against RadioML — real recorded RF, not simulated — and your own synthetic analog
  signals; state clearly which parts of your test set are synthetic vs. recorded.)
- "What happens on a file format / sample rate you don't recognize?" (Should fail explicitly,
  not silently produce garbage output — this is a cheap, high-value robustness story.)

---

## 7. Immediate next actions for Claude Code

1. Scaffold the repo: `src/ingestion/`, `src/detection/`, `src/extraction/`, `src/classify/`,
   `src/report/`, `tests/`, `data/synthetic/`.
2. Build the synthetic signal generator first (§5, Day 1) — everything else is untestable
   without it.
3. Write tests against known-ground-truth synthetic signals from the start; this problem is
   easy to get subtly wrong (e.g., off-by-factor-of-2 bandwidth, wrong endianness) in ways that
   look fine until checked against ground truth.
4. Keep the confidence/evidence data model (§3.5) as a first-class type from day one — retrofitting
   it later is expensive; it's cheap if it's baked into every function's return type from the start.
