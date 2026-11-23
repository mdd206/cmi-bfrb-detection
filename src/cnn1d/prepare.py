import numpy as np
import pandas as pd

from features_imu import LIN_ACC_COLS, add_imu_features
from preprocess import ACC_COLS, ROT_COLS, clean_imu

JERK_COLS = ["jerk_x", "jerk_y", "jerk_z"]
BASE_FEATURES = LIN_ACC_COLS + ROT_COLS + ["rot_missing"]


def load_data(root):
    train = pd.read_csv(root / "data/raw/train.csv",
                        usecols=["sequence_id", "sequence_counter", "subject"] + ACC_COLS + ROT_COLS)
    demo = pd.read_csv(root / "data/raw/train_demographics.csv")
    folds = pd.read_csv(root / "folds/folds.csv")
    return train, demo, folds


def preprocess(df, demo, quat_sign=False):
    """Clean IMU data and add per-step IMU features (rows must be sorted within each sequence)."""
    df = pd.concat([df[["sequence_id", "sequence_counter", "subject"]], clean_imu(df, demo)], axis=1)
    if quat_sign:
        df = fix_quat_sign(df)
    df = pd.concat([df, add_imu_features(df)], axis=1)
    return add_lin_acc_features(df)


def make_padded(df, cols, sequence_ids, max_len):
    """Return (n_sequences, max_len, n_cols): keep the last max_len steps, NaN-pad at the start."""
    groups = {sid: g[cols].to_numpy(dtype=np.float32) for sid, g in df.groupby("sequence_id")}
    out = np.full((len(sequence_ids), max_len, len(cols)), np.nan, dtype=np.float32)
    for i, sid in enumerate(sequence_ids):
        seq = groups[sid][-max_len:]
        out[i, max_len - len(seq):] = seq
    return out


def fix_quat_sign(df):
    """Flip q -> -q where needed so quaternions change smoothly inside each sequence."""
    df = df.copy()
    q = df[ROT_COLS].to_numpy()
    q_prev = df.groupby("sequence_id")[ROT_COLS].shift().to_numpy()
    dot = np.sum(q * q_prev, axis=1)
    step_sign = np.where(dot < 0, -1, 1)
    sign = pd.Series(step_sign, index=df.index).groupby(df["sequence_id"]).cumprod()
    df[ROT_COLS] = q * sign.to_numpy()[:, None]
    return df


def add_lin_acc_features(df):
    """Add the size of linear acceleration and its change between steps (jerk)."""
    df = df.copy()
    df["lin_acc_mag"] = np.linalg.norm(df[LIN_ACC_COLS].to_numpy(), axis=1)
    # first step of each sequence has no previous step, so it gets 0
    df[JERK_COLS] = df.groupby("sequence_id")[LIN_ACC_COLS].diff().fillna(0.0).to_numpy()
    return df
