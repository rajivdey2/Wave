"""FastAPI layer (ARCHITECTURE.md §4). Upload + demo routes, static frontend.

Uptime hardening (fixes 502 on /health):
- /health does ZERO heavy imports (no numpy/scipy/matplotlib/sklearn at
  module import time) so Render's health check passes even during cold boot.
- All DSP imports are lazy (inside request handlers) + pre-warmed in a
  background thread after startup, so the worker binds $PORT immediately.
- CPU-bound pipeline runs via asyncio.to_thread so /health stays responsive
  while /analyze is crunching (previously the event loop blocked -> 502).
"""
from __future__ import annotations

import asyncio
import logging
import os

# --- must run BEFORE numpy/scipy/matplotlib are ever imported ---
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import base64
import json
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# src.types is stdlib-only (dataclasses + enum) — safe to import eagerly.
# Everything else (numpy/scipy/matplotlib/sklearn) is imported lazily.
from src.types import PIPELINE_VERSION

ROOT = Path(__file__).resolve().parents[2]  # repo root (src/api/ -> root)
SYNTH_DIR = ROOT / "data" / "synthetic"

log = logging.getLogger("wavesight")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="SIH26147 Signal Analyzer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
_STORE: dict[str, object] = {}


# ---- health: must stay dependency-free (no numpy/scipy/mpl/sklearn) ----
@app.get("/health", include_in_schema=False)
@app.head("/health", include_in_schema=False)
def health():
    return {"status": "ok", "pipeline_version": PIPELINE_VERSION}


@app.get("/ready", include_in_schema=False)
def ready():
    # Liveness vs readiness split: ready warms only on demand, never blocks.
    return {"status": "ok", "pipeline_version": PIPELINE_VERSION}


@app.on_event("startup")
async def _preload_heavy_deps() -> None:
    """Import numpy/scipy/mpl in background AFTER the worker is listening.

    Render marks the service live as soon as /health responds. If we import
    everything at module top-level, binding is delayed 5-15s on free-tier
    CPU and the proxy returns 502. Background preload keeps first-/analyze
    fast without delaying health.
    """

    def _warm() -> None:
        try:
            import matplotlib  # noqa: F401

            matplotlib.use("Agg")
            import numpy  # noqa: F401
            import scipy.signal  # noqa: F401
            from src.report import analyze_file  # noqa: F401
            log.info("heavy deps pre-warmed")
        except Exception as e:  # never fail startup because of preload
            log.warning("preload failed (lazy load on first request): %s", e)

    asyncio.create_task(asyncio.to_thread(_warm))


def _is_format_error(exc: BaseException) -> bool:
    # Avoid importing src.ingestion at module load; match duck-typed.
    return exc.__class__.__name__ == "FormatError"


def _store_report(report):
    rid = uuid.uuid4().hex[:12]
    _STORE[rid] = report
    d = report.to_dict()
    d["id"] = rid
    return d


def _analyze_sync(tmp_path: str, sample_rate_hz: float | None, orig_name: str):
    from src.report import analyze_file

    report = analyze_file(tmp_path, sample_rate_hz=sample_rate_hz)
    try:
        report.source_file = orig_name or report.source_file
    except Exception:
        pass
    return report


def _demo_sync(path_str: str, sr: float):
    from src.report import analyze_file, to_sigmf

    report = analyze_file(path_str, sample_rate_hz=sr)
    d = _store_report(report)
    d["sigmf"] = to_sigmf(report)
    return d


def _render_png_sync(report, out_path: str) -> str:
    from src.report import render_spectrogram

    render_spectrogram(report, out_path)
    return out_path


@app.post("/analyze")
async def analyze(file: UploadFile = File(...),
                  sample_rate_hz: float | None = Query(default=None)):
    suffix = Path(file.filename or "upload.iq").suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        # to_thread: DSP is CPU-bound/blocking; keep event loop free for /health
        report = await asyncio.to_thread(
            _analyze_sync, tmp_path, sample_rate_hz, file.filename or "")
    except Exception as e:
        if _is_format_error(e):
            raise HTTPException(status_code=422, detail=str(e))
        raise
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    return JSONResponse(_store_report(report))


# Back-compat alias used by the web app
@app.post("/api/analyze")
async def api_analyze(file: UploadFile = File(...),
                      sample_rate_hz: float | None = Query(default=None)):
    return await analyze(file, sample_rate_hz)


@app.get("/api/demos")
def list_demos():
    """Pre-embedded test cases: every synthetic capture + manifest row."""
    manifest_path = SYNTH_DIR / "manifest.json"
    manifest = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = []
    by_file = {m.get("file"): m for m in manifest if isinstance(m, dict)}
    files = sorted(SYNTH_DIR.glob("*.iq")) if SYNTH_DIR.exists() else []
    demos = []
    for p in files:
        m = by_file.get(p.name, {})
        demos.append({
            "file": p.name,
            "modulation": m.get("modulation", "UNKNOWN"),
            "snr_db": m.get("snr_db"),
            "center_freq_hz": m.get("center_freq_hz"),
            "bandwidth_hz": m.get("bandwidth_hz"),
            "symbol_rate_hz": m.get("symbol_rate_hz"),
            "sample_rate_hz": m.get("sample_rate_hz", 100000.0),
            "size_bytes": p.stat().st_size,
        })
    return {"demos": demos, "count": len(demos)}


@app.get("/api/demo/{filename}")
async def analyze_demo(filename: str):
    """Run the full pipeline live on a pre-embedded demo capture."""
    safe = Path(filename).name
    path = SYNTH_DIR / safe
    if not path.exists() or path.suffix.lower() not in (".iq", ".cfile", ".wav", ".bin"):
        raise HTTPException(status_code=404, detail="unknown demo file")
    # sample rate: manifest value or 100 kHz default for synthetic set
    sr: float | None = 100000.0
    try:
        manifest = json.loads((SYNTH_DIR / "manifest.json").read_text())
        for m in manifest:
            if m.get("file") == safe and m.get("sample_rate_hz"):
                sr = float(m["sample_rate_hz"])
                break
    except Exception:
        pass
    try:
        d = await asyncio.to_thread(_demo_sync, str(path), float(sr or 100000.0))
    except Exception as e:
        if _is_format_error(e):
            raise HTTPException(status_code=422, detail=str(e))
        raise
    return JSONResponse(d)


@app.get("/api/report/{rid}/spectrogram")
async def api_spectrogram_png(rid: str):
    report = _STORE.get(rid)
    if report is None:
        raise HTTPException(status_code=404, detail="unknown id")
    fd, out = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    await asyncio.to_thread(_render_png_sync, report, out)
    return FileResponse(out, media_type="image/png", filename="spectrogram.png")


@app.get("/api/report/{rid}/spectrogram-b64")
async def api_spectrogram_b64(rid: str):
    """Inline base64 spectrogram so the app can render without a second fetch."""
    report = _STORE.get(rid)
    if report is None:
        raise HTTPException(status_code=404, detail="unknown id")
    fd, out = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    await asyncio.to_thread(_render_png_sync, report, out)
    data = base64.b64encode(Path(out).read_bytes()).decode("ascii")
    try:
        os.unlink(out)
    except Exception:
        pass
    return {"image_b64": data, "media_type": "image/png"}


@app.get("/analyze/{rid}/report.sigmf-meta")
def sigmf_meta(rid: str):
    from src.report import to_sigmf

    report = _STORE.get(rid)
    if report is None:
        raise HTTPException(status_code=404, detail="unknown id")
    with tempfile.NamedTemporaryFile(delete=False,
                                     suffix=".sigmf-meta", mode="w") as tmp:
        tmp.write(json.dumps(to_sigmf(report), indent=2))
        path = tmp.name
    return FileResponse(path, media_type="application/json",
                        filename="report.sigmf-meta")


@app.get("/analyze/{rid}/spectrogram.png")
async def spectrogram(rid: str):
    report = _STORE.get(rid)
    if report is None:
        raise HTTPException(status_code=404, detail="unknown id")
    fd, out = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    await asyncio.to_thread(_render_png_sync, report, out)
    return FileResponse(out, media_type="image/png")


# ---- static frontend (index.html landing + app.html console) ----
for _static_dir in (ROOT / "assets", ROOT / "web" / "assets"):
    if _static_dir.exists():
        try:
            app.mount("/assets", StaticFiles(directory=str(_static_dir)), name="assets")
        except Exception:
            pass
        break


def _serve_html(name: str):
    for cand in (ROOT / name, ROOT / "web" / name):
        if cand.exists():
            return HTMLResponse(cand.read_text(encoding="utf-8"))
    return HTMLResponse(f"<h1>{name} not built yet</h1>", status_code=404)


@app.get("/", include_in_schema=False)
def root():
    return _serve_html("index.html")


@app.get("/index.html", include_in_schema=False)
def root_alias():
    return _serve_html("index.html")


@app.get("/app", include_in_schema=False)
def app_page():
    return _serve_html("app.html")


@app.get("/app.html", include_in_schema=False)
def app_page_alias():
    return _serve_html("app.html")
