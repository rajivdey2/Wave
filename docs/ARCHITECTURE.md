# ARCHITECTURE.md — SIH26147 Technical Contracts & Win-Condition Checklist

> Companion to CLAUDE.md. That file has strategy and build order; this one is the strict
> technical contract Claude Code should conform to in every session, plus the specific
> features that separate a "worked, technically" submission from a winning one. If a session
> touches code, it should be consistent with this file — don't silently redefine a schema
> that's already specified here.

---

## 1. Repo layout (authoritative)

```
src/
  ingestion/      # file parsers: .iq/.cfile, .wav → normalized complex64 numpy array
  detection/      # FFT/STFT, energy detection, region-of-interest segmentation
  extraction/      # per-region parameter estimation (freq, bandwidth, SNR, symbol rate...)
  classify/        # classical discriminators + ML classifier, confidence wrapper
  demod/           # (Phase 2) demodulation for classified digital modes
  fec/             # (Phase 2) de-interleaving + FEC scheme identification
  report/          # JSON report builder, SigMF annotation writer, spectrogram renderer
  api/             # FastAPI app, request/response models
  types.py         # shared dataclasses/enums — the contract in §2, nowhere else
data/
  synthetic/       # generated labeled signals + ground-truth manifest (see §5)
  radioml/         # cached RadioML samples (not committed — download script only)
tests/
  test_ingestion/
  test_detection/
  test_extraction/
  test_classify/
  test_end_to_end/  # full pipeline against synthetic ground truth, asserts accuracy metrics
docs/
  CLAUDE.md
  ARCHITECTURE.md   # this file
  demo_script.md
```

**Rule:** any function returning signal metadata returns the types in §2, not ad-hoc dicts.
Grep for `dict(` return statements near the DSP core in review — that's the tell that a session
skipped the contract.

---

## 2. Core data contracts

Define once in `src/types.py`, import everywhere else. Illustrative shape — Claude Code should
implement as real dataclasses/pydantic models, not copy this verbatim as strings:

```python
class ConfidenceState(Enum):
    DETECTED = "detected"          # direct measurement, tight error bounds
    ESTIMATED = "estimated"        # standard algorithm, known error bounds
    INFERRED = "inferred"          # ML output above threshold
    UNKNOWN = "unknown"            # below SNR/quality floor — refused to guess

@dataclass
class FieldValue:
    value: float | str | None
    state: ConfidenceState
    confidence: float | None       # 0-1, only meaningful for ESTIMATED/INFERRED
    evidence: str                  # short human-readable justification, always present

@dataclass
class SignalDetection:
    id: str
    t_start_s: float
    t_end_s: float
    f_low_hz: float
    f_high_hz: float
    center_freq_hz: FieldValue
    bandwidth_hz: FieldValue
    snr_db: FieldValue
    symbol_rate_hz: FieldValue
    modulation: FieldValue          # value is the label, e.g. "PSK" / "QAM16" / "UNKNOWN"
    modulation_evidence_method: str # "classical" | "ml" | "both"
    demod_available: bool
    fec_identification: FieldValue | None

@dataclass
class CaptureReport:
    source_file: str
    format: str                     # "iq_cf32" | "iq_int16" | "wav"
    sample_rate_hz: float
    duration_s: float
    noise_floor_db: float
    dynamic_range_db: float
    detections: list[SignalDetection]
    pipeline_version: str
```

**Why this matters more than it looks:** every FieldValue forces the pipeline to say *how*
confident it is and *why*, not just what. This is the single biggest jury-visible
differentiator from CLAUDE.md §3.5/§6, and it only works if it's structurally impossible to
bypass — hence baking it into the type, not adding it as an afterthought field.

---

## 3. Output schemas (exact, versioned)

### 3.1 JSON report
`CaptureReport` serialized directly (dataclasses → dict → JSON, pydantic recommended for
FastAPI integration). Include `pipeline_version` in every report — you will change the
pipeline mid-week and need to know which run produced which report when you're debugging a
regression the night before submission.

### 3.2 SigMF annotation (`.sigmf-meta`)
Follow the SigMF spec (sigmf.org) for the `annotations` array: each `SignalDetection` maps to
one annotation object with `core:sample_start`, `core:sample_count` (derive from t_start/t_end
× sample_rate), `core:freq_lower_edge`, `core:freq_upper_edge`, and put your
label/confidence/evidence under a custom namespace, e.g. `ntro_sih:modulation`,
`ntro_sih:confidence_state`, `ntro_sih:evidence`. Validate every emitted file against the
public SigMF JSON schema before demo day — a malformed "standards-compliant" file is worse
than not claiming compliance at all if a jury member checks it.

### 3.3 Spectrogram render
PNG or interactive HTML (Plotly), waterfall with bounding boxes per detection, label =
`modulation (confidence_state, confidence%)`. Color-code by confidence state (e.g., solid box
for DETECTED/ESTIMATED, dashed for INFERRED, red/hatched for UNKNOWN) — this single visual
convention does a lot of work in a live demo without you having to narrate every box.

---

## 4. API contract (if you build the FastAPI layer)

```
POST /analyze
  multipart/form-data: file (.iq/.cfile/.wav), optional sample_rate_hz if not inferable
  → 200: CaptureReport (JSON)
  → 422: {"error": "unrecognized format" | "sample_rate_required" | ...}  — explicit, not a 500

GET /analyze/{id}/spectrogram.png
GET /analyze/{id}/report.sigmf-meta
GET /health
  → {"status": "ok", "pipeline_version": "..."}
```

Keep it this small. A sprawling API surface is wasted effort for a hackathon judged on a demo
and a report — three or four routes is enough to look production-shaped without eating build time.

---

## 5. Synthetic data & ground truth (non-negotiable, build this first)

- `data/synthetic/manifest.json`: one row per generated file — file path, true modulation,
  true center freq, true bandwidth, true symbol rate (where applicable), injected SNR.
- Generate across a real SNR sweep (e.g. −5, 0, 5, 10, 15, 20 dB) for every modulation type
  required by the PS (AM, FM, SSB, CW, FSK, PSK variants, QAM variants, OFDM, chirp).
- `tests/test_end_to_end/test_accuracy_vs_snr.py` must produce an actual accuracy-vs-SNR table/
  plot from this manifest — this is the artifact that answers the jury's SNR question from
  CLAUDE.md §6 with data instead of a claim.
- Never hand-wave "it works on our test files" without this manifest backing it. It's the
  difference between a demo and an evaluated system.

---

## 6. Must-have differentiators — build checklist

Each item: what it is, minimum viable version, and the acceptance check that proves it's real
(not just present in the pitch deck).

- [ ] **Confidence/evidence on every field** (§2). *Check:* pick any report field, jury can ask
      "why do you believe this," and the `evidence` string answers it without you improvising.
- [ ] **Explicit UNKNOWN below SNR floor.** *Check:* feed a pure-noise or very-low-SNR file;
      pipeline must emit `ConfidenceState.UNKNOWN`, not a confident wrong guess.
- [ ] **Accuracy-vs-SNR curve from real measurement** (§5). *Check:* the number in your slide
      traces back to a test file and a manifest row, not a remembered figure.
- [ ] **SigMF-compliant annotation output**, schema-validated. *Check:* run the file through a
      public SigMF validator before demo day, not just "it looks right."
- [ ] **Multi-signal detection in one capture.** *Check:* a single synthetic file with 2–3
      simultaneous signals at different center frequencies is correctly segmented into separate
      detections, not merged into one.
- [ ] **At least one digital mode taken past classification into demodulation** (e.g., BPSK/
      QPSK bit recovery) plus a basic FEC scheme identification on the recovered bits. Scope to
      one or two modes done well over five done shallowly.
- [ ] **Explicit offline/CPU-only operation.** *Check:* run the full pipeline with networking
      disabled and no GPU available; it must still complete. This directly answers an NTRO
      deployment concern and costs you nothing if you didn't reach for a cloud API anyway.
- [ ] **Format-failure is explicit, not silent.** *Check:* feed a corrupt or unrecognized file;
      pipeline returns a clear error (§4, 422), never a garbage report that looks plausible.
- [ ] **Annotated spectrogram as the lead demo artifact** (§3.3), rehearsed as the first thing
      shown, before any code or architecture slide.

Everything above is achievable inside the Day 1–6 plan in CLAUDE.md if you don't let scope
creep into the demod/FEC item — cap it at one or two modes and move on.

---

## 7. Engineering conventions

- Every module in §1 gets its own test directory; a PR/commit that touches `detection/` without
  a corresponding test change should be treated as incomplete.
- Log DSP intermediate values (raw FFT bins, threshold crossings) at debug level during
  development — this domain fails silently and subtly (§7 of CLAUDE.md), and you will need
  these logs at 2am the night before submission.
- Pin `pipeline_version` (§3.1) to a git short-hash or semantic tag, bumped on any change to
  detection/extraction/classification logic, so every report and every demo screenshot is
  traceable to the exact code that produced it.
- When starting a new Claude Code session mid-build, re-read CLAUDE.md §5 (build order) and
  this file's §6 checklist before writing code — don't re-derive architecture from scratch each
  session; consistency across sessions is itself a differentiator under time pressure.
