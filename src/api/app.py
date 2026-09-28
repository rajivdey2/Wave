"""FastAPI layer (ARCHITECTURE.md §4). Upload + demo routes, static frontend."""
from __future__ import annotations

import base64
import io
import json
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.ingestion import FormatError
from src.report import analyze_file, to_sigmf, render_spectrogram
from src.types import PIPELINE_VERSION

ROOT = Path(__file__).resolve().parents[2]  # repo root (src/api/ -> root)
SYNTH_DIR = ROOT / "data" / "synthetic"

app = FastAPI(title="SIH26147 Signal Analyzer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
_STORE: dict[str, object] = {}


@app.get("/health")
def health():
    return {"status": "ok", "pipeline_version": PIPELINE_VERSION}


def _store_report(report):
    rid = uuid.uuid4().hex[:12]
    _STORE[rid] = report
    d = report.to_dict()
    d["id"] = rid
    return d


@app.post("/analyze")
async def analyze(file: UploadFile = File(...),
                  sample_rate_hz: float | None = Query(default=None)):
    suffix = Path(file.filename or "upload.iq").suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        report = analyze_file(tmp_path, sample_rate_hz=sample_rate_hz)
    except FormatError as e:
        raise HTTPException(status_code=422, detail=str(e))
    # relabel source with the original upload name for clarity
    try:
        report.source_file = file.filename or report.source_file
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
def analyze_demo(filename: str):
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
        report = analyze_file(str(path), sample_rate_hz=sr)
    except FormatError as e:
        raise HTTPException(status_code=422, detail=str(e))
    d = _store_report(report)
    d["sigmf"] = to_sigmf(report)
    return JSONResponse(d)


@app.get("/api/report/{rid}/spectrogram")
def api_spectrogram_png(rid: str):
    report = _STORE.get(rid)
    if report is None:
        raise HTTPException(status_code=404, detail="unknown id")
    out = tempfile.mktemp(suffix=".png")
    render_spectrogram(report, out)
    return FileResponse(out, media_type="image/png", filename="spectrogram.png")


@app.get("/api/report/{rid}/spectrogram-b64")
def api_spectrogram_b64(rid: str):
    """Inline base64 spectrogram so the app can render without a second fetch."""
    import matplotlib
    matplotlib.use("Agg")
    report = _STORE.get(rid)
    if report is None:
        raise HTTPException(status_code=404, detail="unknown id")
    out = tempfile.mktemp(suffix=".png")
    render_spectrogram(report, out)
    data = base64.b64encode(Path(out).read_bytes()).decode("ascii")
    return {"image_b64": data, "media_type": "image/png"}


@app.get("/analyze/{rid}/report.sigmf-meta")
def sigmf_meta(rid: str):
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
def spectrogram(rid: str):
    report = _STORE.get(rid)
    if report is None:
        raise HTTPException(status_code=404, detail="unknown id")
    out = tempfile.mktemp(suffix=".png")
    render_spectrogram(report, out)
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
