import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from metric import BFRB_GESTURES


def main():
    parser = argparse.ArgumentParser(description="Plot confusion matrix from an _oof.csv file")
    parser.add_argument("oof_csv")
    args = parser.parse_args()

    oof = pd.read_csv(args.oof_csv)
    others = sorted(set(oof["gesture"]) - set(BFRB_GESTURES))
    labels = BFRB_GESTURES + others  # BFRB first, then non-BFRB

    # each row sums to 1: share of a true gesture predicted as each gesture
    fig, ax = plt.subplots(figsize=(13, 11))
    ConfusionMatrixDisplay.from_predictions(
        oof["gesture"], oof["pred"], labels=labels, normalize="true",
        values_format=".2f", cmap="Blues", colorbar=False, ax=ax)
    ax.axhline(len(BFRB_GESTURES) - 0.5, color="red")
    ax.axvline(len(BFRB_GESTURES) - 0.5, color="red")
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_title(Path(args.oof_csv).stem)
    fig.tight_layout()

    out = Path(args.oof_csv).with_suffix(".png")
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
