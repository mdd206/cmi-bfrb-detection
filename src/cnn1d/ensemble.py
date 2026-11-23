import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from metric import competition_score


def report(name, y_true, probs, gestures):
    score, binary, macro = competition_score(y_true, [gestures[i] for i in probs.argmax(axis=1)])
    print(f"{name}: score {score:.4f} (binary {binary:.4f}, macro {macro:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Average the probabilities of several train.py runs")
    parser.add_argument("runs", nargs="+", help="run prefixes, e.g. outputs/cnn1d/final_s42_20261123_140000")
    args = parser.parse_args()

    folds = pd.read_csv(ROOT / "folds/folds.csv")
    dev = folds[folds["split"] == "train"].reset_index(drop=True)
    test = folds[folds["split"] == "test"].reset_index(drop=True)
    gestures = sorted(dev["gesture"].unique())  # same class order as train.py

    oof = [np.load(f"{run}_oof_probs.npy") for run in args.runs]
    for run, probs in zip(args.runs, oof):
        report(f"OOF {Path(run).name}", dev["gesture"], probs, gestures)
    report(f"OOF ensemble of {len(oof)}", dev["gesture"], np.mean(oof, axis=0), gestures)

    if all(Path(f"{run}_test_probs.npy").exists() for run in args.runs):
        tests = [np.load(f"{run}_test_probs.npy") for run in args.runs]
        for run, probs in zip(args.runs, tests):
            report(f"Holdout {Path(run).name}", test["gesture"], probs, gestures)
        report(f"Holdout ensemble of {len(tests)}", test["gesture"], np.mean(tests, axis=0), gestures)


if __name__ == "__main__":
    main()
