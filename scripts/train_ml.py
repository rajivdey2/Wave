"""Train the ML modulation classifier and ship models/ml_classifier.pkl."""
import argparse
import time

from src.classify.ml import build_training_matrix, save_model, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--seeds-per", type=int, default=3)
    ap.add_argument("--out", default="models/ml_classifier.pkl")
    args = ap.parse_args()

    t0 = time.time()
    X, y = build_training_matrix(seeds_per=args.seeds_per)
    print(f"training matrix: {X.shape}, classes: {sorted(set(y))}")
    clf = train(n_estimators=args.trees)
    clf.fit(X, y)
    print(f"train accuracy: {clf.score(X, y):.3f} ({time.time()-t0:.1f}s)")
    p = save_model(clf, args.out)
    print(f"saved {p} ({p.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
