"""Fetch the RadioML 2016.10a recorded-RF dataset (for validation only).

The canonical file `RML2016.10a_dict.pkl` (~500 MB) is distributed via the
DeepSig dataset page behind a request form. This script tries, in order:

  1. --from-file /path/to/RML2016.10a_dict.pkl  (you downloaded it manually)
  2. --url <explicit mirror you trust>
  3. Built-in community mirrors (best effort — they can rot):
     - Zenodo RML2016.10a re-serialization (protocol 4, ~213 MB .tar.bz2)
     - opendata.deepsig.io official tarball (may be retired)

Archives (.tar.bz2) are extracted automatically; the .pkl/.dat inside is
validated and placed at data/radioml/.
"""
from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path
from urllib.request import urlretrieve

DEST = Path("data/radioml/RML2016.10a_dict.pkl")
MIRRORS = [
    "https://zenodo.org/records/18397070/files/RML2016.10a.tar.bz2?download=1",
    "http://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2",
]


def _show_report(path: Path) -> None:
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    if isinstance(d, dict):
        keys = list(d.keys())[:3]
        print(f"keys sample: {keys}")
        k0 = list(d.keys())[0]
        print(f"entry shape: {d[k0].shape}, n_entries: {len(d)}")
    else:
        print(f"top-level type: {type(d)}")
    print(f"size: {path.stat().st_size/1e6:.0f} MB")


def _extract_pkl(archive: Path, dest: Path) -> Path:
    import tarfile
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tf:
        members = [m for m in tf.getmembers()
                   if m.isfile() and m.name.endswith((".pkl", ".dat"))]
        if not members:
            raise SystemExit(f"no .pkl/.dat inside {archive}")
        inner = max(members, key=lambda m: m.size)
        tf.extract(inner, dest.parent)
        got = dest.parent / inner.name
    if got.resolve() != dest.resolve():
        shutil.move(str(got), str(dest))
    return dest


def _fetch(url: str, dest: Path) -> Path:
    import tempfile
    dest.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".tar.bz2" if ".tar" in url else ".pkl"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
    print(f"downloading {url} ...")
    urlretrieve(url, tmp_path)
    if suffix != ".pkl":
        try:
            return _extract_pkl(Path(tmp_path), dest)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    shutil.move(tmp_path, dest)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-file", default=None)
    ap.add_argument("--url", default=None)
    ap.add_argument("--dest", default=str(DEST))
    args = ap.parse_args()

    dest = Path(args.dest)
    if args.from_file:
        src = Path(args.from_file)
        if not src.exists():
            raise SystemExit(f"not found: {src}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dest.resolve():
            shutil.copy(src, dest)
    elif args.url:
        _fetch(args.url, dest)
    else:
        for url in MIRRORS:
            try:
                _fetch(url, dest)
                break
            except Exception as e:
                print(f"mirror failed ({url}): {e}")
        else:
            raise SystemExit(
                "All mirrors failed. Download RML2016.10a_dict.pkl from "
                "https://www.deepsig.ai/datasets (request form) or a Kaggle "
                'mirror, then rerun with --from-file <path>.')
    _show_report(dest)
    print(f"ready: {dest}")


if __name__ == "__main__":
    main()
