from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["sequence_id", "sequence_counter"]
# gesture and sequence_type only exist in train
INFO_COLS = ["subject", "gesture", "sequence_type"]

ACC_COLS = ["acc_x", "acc_y", "acc_z"]
ROT_COLS = ["rot_w", "rot_x", "rot_y", "rot_z"]
IMU_COLS = ACC_COLS + ROT_COLS

THM_COLS = [f"thm_{i}" for i in range(1, 6)]
TOF_SENSOR_COLS = {
    f"tof_{i}": [f"tof_{i}_v{p}" for p in range(64)] for i in range(1, 6)
}
TOF_COLS = [col for i in range(1, 6) for col in TOF_SENSOR_COLS[f"tof_{i}"]]

# Physical ceiling (mm) for "no object detected" (-1), just above max valid reading (249 mm)
NO_OBJECT_DIST = 255.0
# Minimum normal wrist/ambient temperature (°C); below this indicates sensor offset or dead channel
NORMAL_THM_FLOOR = 20.0


def clean_imu(df, demo):
    """Fill missing IMU values, normalize quaternions and mirror left-handed subjects.

    Output: acc_*, rot_* (cleaned) and rot_missing, same index as df.
    """
    work = df[["sequence_id"] + IMU_COLS].copy()
    work["rot_missing"] = work[ROT_COLS].isna().any(axis=1).astype(np.int8)

    # Impute strictly within each sequence (separate groupby calls so bfill never crosses sequences)
    work[IMU_COLS] = work.groupby("sequence_id", sort=False)[IMU_COLS].ffill()
    work[IMU_COLS] = work.groupby("sequence_id", sort=False)[IMU_COLS].bfill()
    work[ACC_COLS] = work[ACC_COLS].fillna(0.0)

    # Fill sequences missing orientation entirely with the identity quaternion (1, 0, 0, 0)
    still_missing = work[ROT_COLS].isna().any(axis=1)
    work.loc[still_missing, "rot_w"] = 1.0
    work.loc[still_missing, ["rot_x", "rot_y", "rot_z"]] = 0.0

    # Normalize to unit quaternions; fall back to identity if norm is degenerate
    q = work[ROT_COLS].to_numpy(dtype=np.float64)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    valid = norm[:, 0] > 1e-8
    q[valid] = q[valid] / norm[valid]
    q[~valid] = [1.0, 0.0, 0.0, 0.0]
    work[ROT_COLS] = q

    # Mirror across the YZ plane (x -> -x) for left-handed subjects (handedness: 0 = left)
    left_subjects = demo.loc[demo["handedness"] == 0, "subject"]
    left_mask = df["subject"].isin(left_subjects).to_numpy()
    work.loc[left_mask, "acc_x"] = -work.loc[left_mask, "acc_x"]
    # Reflection M = diag(-1, 1, 1) conjugates rotation matrix R -> M R M,
    # mapping unit quaternion (w, x, y, z) to (w, x, -y, -z)
    rot_mask = left_mask & (work["rot_missing"].to_numpy() == 0)
    work.loc[rot_mask, ["rot_y", "rot_z"]] = -work.loc[rot_mask, ["rot_y", "rot_z"]]

    return work[IMU_COLS + ["rot_missing"]]


def clean_thm(df):
    """Flag broken thermal channels, fill gaps and align absolute temperatures.

    Output: thm_* (cleaned) and thm_*_broken, same index as df.
    """
    out = df[["sequence_id", "sequence_counter"] + THM_COLS].copy()
    seq_grp = out.groupby("sequence_id", sort=False)

    # Detect dead flatline sensors per sequence (e.g. SUBJ_028998 thm_3 == 0.0 throughout)
    # Active offset sensors (e.g. SUBJ_036405 thm_1 with std > 0.2) are kept to preserve delta_thm
    seq_std = seq_grp[THM_COLS].transform("std").fillna(0.0)
    seq_mean = seq_grp[THM_COLS].transform("mean")
    dead_flatline = (seq_mean < NORMAL_THM_FLOOR) & (seq_std < 1e-4)

    for c in THM_COLS:
        out[f"{c}_broken"] = (out[c].isna() | dead_flatline[c]).astype(np.int8)
        out.loc[dead_flatline[c], c] = np.nan

    # Interpolate partial missing values strictly within each sequence
    out[THM_COLS] = out.groupby("sequence_id", sort=False)[THM_COLS].ffill()
    out[THM_COLS] = out.groupby("sequence_id", sort=False)[THM_COLS].bfill()

    # Temperature change from the start of the sequence (needed to rebuild broken channels)
    seq_base = thm_baseline(out)
    delta_df = out[THM_COLS] - seq_base

    # If a channel is broken for the entire sequence, use the row mean change of working channels
    row_mean_delta = delta_df.mean(axis=1).fillna(0.0)
    for c in THM_COLS:
        delta_df[c] = delta_df[c].fillna(row_mean_delta)

    # Align absolute temperature for broken or offset-shifted (< 20°C) channels using healthy sensors
    healthy_base = seq_base.where(seq_base >= NORMAL_THM_FLOOR).mean(axis=1).fillna(27.0)
    for c in THM_COLS:
        needs_align = out[c].isna() | (seq_base[c] < NORMAL_THM_FLOOR)
        out.loc[needs_align, c] = healthy_base[needs_align] + delta_df.loc[needs_align, c]

    return out[THM_COLS + [f"{c}_broken" for c in THM_COLS]]


def thm_baseline(df):
    """Start-of-sequence temperature of each channel (mean of the first 5 steps)."""
    is_start = df["sequence_counter"] < 5
    start_df = df[THM_COLS].where(is_start)
    seq_base = start_df.groupby(df["sequence_id"], sort=False).transform("mean")
    return seq_base.fillna(df.groupby("sequence_id", sort=False)[THM_COLS].transform("first"))


def clean_tof(df):
    """Flag broken ToF sensors, map "no object" (-1) to a far distance and fill gaps.

    Output: tof_*_v* pixels (cleaned) and tof_*_broken, same index as df.
    """
    arr = df[TOF_COLS].to_numpy(dtype=np.float32)

    # Sensor completely missing (all 64 pixels NaN)
    broken = pd.DataFrame(
        {f"tof_{i}_broken": np.isnan(arr[:, (i - 1) * 64 : i * 64]).all(axis=1).astype(np.int8)
         for i in range(1, 6)},
        index=df.index,
    )

    # Map -1 to 255.0 mm, impute broken steps per sequence, fallback to 255.0 mm
    pixels = pd.DataFrame(np.where(arr == -1.0, NO_OBJECT_DIST, arr), index=df.index, columns=TOF_COLS)
    pixels = pixels.groupby(df["sequence_id"], sort=False).ffill()
    pixels = pixels.groupby(df["sequence_id"], sort=False).bfill()
    pixels = pixels.fillna(NO_OBJECT_DIST)

    return pd.concat([pixels, broken], axis=1)


def preprocess(df, demo):
    """Clean raw sensor data (train or test). Every step works on one sequence at a time.

    Input:
        df   : raw sensor data, one row per time step
        demo : demographics table, one row per subject
    Output:
        DataFrame with the same rows as df, sorted by sequence_id and sequence_counter
        (original index kept): keys, info columns (only those present in df),
        cleaned IMU, thermal and ToF columns.
    """
    df = df.sort_values(KEYS, kind="mergesort")
    info = [c for c in INFO_COLS if c in df.columns]
    data = pd.concat([df[KEYS + info], clean_imu(df, demo), clean_thm(df), clean_tof(df)], axis=1)
    assert len(data) == len(df), "Number of rows changed after cleaning"
    return data


if __name__ == "__main__":
    from features_imu import add_imu_features
    from features_tof import add_tof_thm_features

    train = pd.read_csv("data/raw/train.csv")
    demo = pd.read_csv("data/raw/train_demographics.csv")

    clean = preprocess(train, demo)
    data = pd.concat([clean, add_imu_features(clean), add_tof_thm_features(clean)], axis=1)
    data = data.sort_index()  # back to the original row order

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    data.to_parquet("data/processed/train.parquet", index=False)

    print("Rows:", len(data))
    print("Columns:", data.shape[1])
    print(data.head())
