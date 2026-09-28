"""Generate the synthetic dataset + manifest (ARCHITECTURE.md §5)."""
import argparse
from src.synthesis import build_dataset

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--sr", type=float, default=100_000.0)
    ap.add_argument("--dur", type=float, default=1.0)
    args = ap.parse_args()
    manifest = build_dataset(args.out, sample_rate_hz=args.sr, duration_s=args.dur)
    print(f"wrote {len(manifest)} files to {args.out}")
