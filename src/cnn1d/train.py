import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from metric import competition_score
from prepare import BASE_FEATURES, load_data, make_padded, preprocess

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log = logging.getLogger("cnn1d")


class CNN1D(nn.Module):
    def __init__(self, n_inputs, n_classes, channels, kernel_size, bn=False, dropout=0.0,
                 masked_pool=False, convs=1):
        super().__init__()
        self.masked_pool = masked_pool
        layers = []
        in_ch = n_inputs
        for out_ch in channels:
            for _ in range(convs):
                layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2))
                if bn:
                    layers.append(nn.BatchNorm1d(out_ch))
                layers.append(nn.ReLU())
                in_ch = out_ch
            layers += [nn.MaxPool1d(2), nn.Dropout(dropout)]
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(in_ch, n_classes)

    def forward(self, x):
        # padding steps are exactly 0 in every channel after normalize()
        mask = (x != 0).any(dim=1, keepdim=True).float()
        x = self.conv(x)
        if self.masked_pool:
            # average only over real steps, ignoring the padding
            mask = nn.functional.adaptive_max_pool1d(mask, x.shape[2])
            x = (x * mask).sum(dim=2) / mask.sum(dim=2)
        else:
            x = x.mean(dim=2)  # global average pooling
        return self.fc(x)


def normalize(X_train, *others):
    """Scale with training-fold statistics, set padding to 0, return (n, channels, time)."""
    mean = np.nanmean(X_train, axis=(0, 1))
    std = np.nanstd(X_train, axis=(0, 1)) + 1e-6
    out = []
    for X in (X_train, *others):
        X = np.nan_to_num((X - mean) / std)
        out.append(torch.from_numpy(X.transpose(0, 2, 1)).to(DEVICE))
    return out


def augment(x, strength):
    """Scale each channel of each sample by a random factor near 1 and add small noise.
    Padding steps stay 0."""
    mask = (x != 0).any(dim=1, keepdim=True)
    scale = 1 + strength * torch.randn(x.shape[0], x.shape[1], 1, device=x.device)
    noise = 0.5 * strength * torch.randn_like(x)
    return (x * scale + noise) * mask


def time_stretch(x, max_stretch):
    """Speed the whole batch up or down by a random factor, keep the end of the sequences."""
    rate = 1 + max_stretch * (2 * np.random.rand() - 1)
    n = x.shape[2]
    x = nn.functional.interpolate(x, size=round(n * rate), mode="linear", align_corners=False)
    if x.shape[2] >= n:
        return x[:, :, -n:]
    return nn.functional.pad(x, (n - x.shape[2], 0))  # pad at the start like make_padded


def train_one_epoch(model, X, y, optimizer, loss_fn, batch_size, mixup, aug, stretch):
    model.train()
    perm = torch.randperm(len(y), device=DEVICE)
    for i in range(0, len(y), batch_size):
        idx = perm[i:i + batch_size]
        x = augment(X[idx], aug) if aug > 0 else X[idx]
        if stretch > 0:
            x = time_stretch(x, stretch)
        optimizer.zero_grad()
        if mixup > 0:
            # blend each sample with another random sample of the batch
            lam = np.random.beta(mixup, mixup)
            other = torch.randperm(len(idx), device=DEVICE)
            logits = model(lam * x + (1 - lam) * x[other])
            loss = lam * loss_fn(logits, y[idx]) + (1 - lam) * loss_fn(logits, y[idx][other])
        else:
            loss = loss_fn(model(x), y[idx])
        loss.backward()
        optimizer.step()


@torch.no_grad()
def predict_logits(model, X, batch_size=512):
    model.eval()
    return torch.cat([model(X[i:i + batch_size]) for i in range(0, len(X), batch_size)])


def evaluate(model, X, y, loss_fn, gestures):
    """Return loss, (score, binary_f1, macro_f1) and class probabilities."""
    logits = predict_logits(model, X)
    loss = loss_fn(logits, y).item()
    probs = logits.softmax(dim=1).cpu().numpy()
    true = y.cpu().numpy()
    scores = competition_score([gestures[i] for i in true], [gestures[i] for i in probs.argmax(axis=1)])
    return loss, scores, probs


def train_fold(k, X_train, y_train, X_val, y_val, gestures, cfg):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    y_train = torch.from_numpy(y_train).to(DEVICE)
    y_val = torch.from_numpy(y_val).to(DEVICE)

    model = CNN1D(X_train.shape[1], len(gestures), cfg["channels"], cfg["kernel_size"],
                  cfg["bn"], cfg["dropout"], cfg["masked_pool"], cfg["convs"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"]) if cfg["cosine"] else None
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    history = []
    best_score, best_state, best_probs, wait = -1.0, None, None, 0
    for epoch in range(1, cfg["epochs"] + 1):
        train_one_epoch(model, X_train, y_train, optimizer, loss_fn, cfg["batch_size"],
                        cfg["mixup"], cfg["aug"], cfg["stretch"])
        if scheduler:
            scheduler.step()
        train_loss, (train_score, _, _), _ = evaluate(model, X_train, y_train, loss_fn, gestures)
        val_loss, (val_score, val_binary, val_macro), val_probs = evaluate(
            model, X_val, y_val, loss_fn, gestures)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_score": train_score,
                        "val_loss": val_loss, "val_score": val_score,
                        "val_binary_f1": val_binary, "val_macro_f1": val_macro})
        log.info(f"Fold {k} | epoch {epoch:3d} | train loss {train_loss:.4f} score {train_score:.4f}"
                 f" | val loss {val_loss:.4f} score {val_score:.4f}"
                 f" (binary {val_binary:.4f}, macro {val_macro:.4f})")

        if val_score > best_score:
            best_score, best_probs, wait = val_score, val_probs, 0
            best_state = {name: v.clone() for name, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break

    model.load_state_dict(best_state)
    return model, history, best_probs


def run_cv(df, folds, cfg):
    dev = folds[folds["split"] == "train"].reset_index(drop=True)
    gestures = sorted(dev["gesture"].unique())
    labels = dev["gesture"].map({g: i for i, g in enumerate(gestures)}).to_numpy()
    fold_ids = dev["fold"].to_numpy()

    X_all = make_padded(df, cfg["features"], dev["sequence_id"], cfg["max_len"])
    # holdout subjects: never used for training or model selection
    test = folds[folds["split"] == "test"].reset_index(drop=True)
    X_test_all = make_padded(df, cfg["features"], test["sequence_id"], cfg["max_len"])
    test_probs = np.zeros((len(test), len(gestures)))

    oof_probs = np.zeros((len(dev), len(gestures)))
    rows, histories = [], []
    for k in sorted(np.unique(fold_ids)):
        start = time.time()
        tr, va = fold_ids != k, fold_ids == k
        X_train, X_val, X_test = normalize(X_all[tr], X_all[va], X_test_all)
        model, history, val_probs = train_fold(k, X_train, labels[tr], X_val, labels[va], gestures, cfg)
        oof_probs[va] = val_probs
        if cfg["holdout"]:
            # holdout prediction = average of the 5 fold models' probabilities
            test_probs += predict_logits(model, X_test).softmax(dim=1).cpu().numpy() / 5
        histories += [{"fold": k, **h} for h in history]

        best = max(history, key=lambda h: h["val_score"])
        rows.append({"fold": k, **best})
        log.info(f"Fold {k} best: epoch {best['epoch']}/{len(history)}"
                 f" | train loss {best['train_loss']:.4f} score {best['train_score']:.4f}"
                 f" | val loss {best['val_loss']:.4f} score {best['val_score']:.4f}"
                 f" (binary {best['val_binary_f1']:.4f}, macro {best['val_macro_f1']:.4f})"
                 f" | {time.time() - start:.0f}s")

    summary = pd.DataFrame(rows)
    oof = oof_probs.argmax(axis=1)
    oof_score = competition_score([gestures[i] for i in labels], [gestures[i] for i in oof])
    log.info(f"CV val score: {summary['val_score'].mean():.4f} +/- {summary['val_score'].std():.4f}"
             f" | train score: {summary['train_score'].mean():.4f}")
    log.info(f"OOF: score {oof_score[0]:.4f} (binary {oof_score[1]:.4f}, macro {oof_score[2]:.4f})")
    if cfg["holdout"]:
        test_score = competition_score(test["gesture"], [gestures[i] for i in test_probs.argmax(axis=1)])
        log.info(f"Holdout: score {test_score[0]:.4f} (binary {test_score[1]:.4f}, macro {test_score[2]:.4f})")
    oof_df = pd.DataFrame({"sequence_id": dev["sequence_id"], "fold": fold_ids,
                           "gesture": dev["gesture"], "pred": [gestures[i] for i in oof]})
    return summary, pd.DataFrame(histories), oof_df, oof_probs, test_probs


def main():
    parser = argparse.ArgumentParser(description="Train 1D CNN with 5-fold CV")
    parser.add_argument("--features", nargs="+", default=BASE_FEATURES)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--quat_sign", action="store_true")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--bn", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--masked_pool", action="store_true")
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--aug", type=float, default=0.0)
    parser.add_argument("--stretch", type=float, default=0.0)
    parser.add_argument("--convs", type=int, default=1, help="conv layers per block")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--cosine", action="store_true")
    # only for the final, chosen config: score the untouched test subjects once
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="cnn1d")
    cfg = vars(parser.parse_args())

    out_dir = ROOT / "outputs" / "cnn1d"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"{cfg['name']}_{time.strftime('%Y%m%d_%H%M%S')}"
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(f"{prefix}.log", encoding="utf-8")])
    log.info(f"{cfg} | device: {DEVICE}")

    train, demo, folds = load_data(ROOT)
    df = preprocess(train, demo, cfg["quat_sign"])
    summary, history, oof, oof_probs, test_probs = run_cv(df, folds, cfg)

    cfg_cols = {key: str(v) for key, v in cfg.items()}
    summary.assign(**cfg_cols).to_csv(f"{prefix}_summary.csv", index=False)
    history.assign(**cfg_cols).to_csv(f"{prefix}_history.csv", index=False)
    oof.to_csv(f"{prefix}_oof.csv", index=False)
    # probabilities, for averaging several runs in ensemble.py
    np.save(f"{prefix}_oof_probs.npy", oof_probs)
    if cfg["holdout"]:
        np.save(f"{prefix}_test_probs.npy", test_probs)
    log.info(f"Saved {prefix}.log, _summary.csv, _history.csv, _oof.csv")


if __name__ == "__main__":
    main()
