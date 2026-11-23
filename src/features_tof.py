import numpy as np
import pandas as pd

from preprocess import NO_OBJECT_DIST, THM_COLS, TOF_SENSOR_COLS, thm_baseline


def add_thm_features(df, feat):
    """Add temperature change from the start of the sequence and cross-sensor summaries."""
    # Relative temperature change cancels subject baseline shifts and calibration offsets
    delta_df = df[THM_COLS] - thm_baseline(df)
    for c in THM_COLS:
        feat[f"{c}_delta"] = delta_df[c].to_numpy(dtype=np.float32)

    # Cross-sensor thermal summaries and spatial contrast (upper/side thm_1..3 vs lower thm_4..5)
    delta_arr = delta_df.to_numpy(dtype=np.float32)
    feat["thm_delta_mean"] = delta_arr.mean(axis=1)
    feat["thm_delta_std"] = delta_arr.std(axis=1)
    feat["thm_delta_max"] = delta_arr.max(axis=1)
    feat["thm_delta_min"] = delta_arr.min(axis=1)
    feat["thm_123_vs_45_delta"] = delta_arr[:, :3].mean(axis=1) - delta_arr[:, 3:].mean(axis=1)
    feat["thm_delta_diff"] = (
        feat["thm_delta_mean"].groupby(df["sequence_id"], sort=False).diff().fillna(0.0)
    )


def add_tof_features(df, feat):
    """Summarize each 8x8 ToF grid, then compare the 5 sensors."""
    n_rows = len(df)
    stat_names = ["active", "mean", "std", "min", "max", "lr_diff", "tb_diff"]
    tof_stats = {s: np.empty((n_rows, 5), dtype=np.float32) for s in stat_names}

    for idx, i in enumerate(range(1, 6)):
        arr = df[TOF_SENSOR_COLS[f"tof_{i}"]].to_numpy(dtype=np.float32)
        # Pixels at NO_OBJECT_DIST came from -1 ("no object") in the raw data
        valid_mask = arr < NO_OBJECT_DIST
        n_valid = valid_mask.sum(axis=1)
        no_obj = n_valid == 0

        # Distance statistics over valid pixels
        safe_count = np.maximum(n_valid, 1).astype(np.float32)
        valid_vals = np.where(valid_mask, arr, 0.0)
        mean_val = valid_vals.sum(axis=1) / safe_count
        sq_diff = np.where(valid_mask, (arr - mean_val[:, None]) ** 2, 0.0)
        std_val = np.sqrt(sq_diff.sum(axis=1) / safe_count)
        min_val = np.where(valid_mask, arr, np.inf).min(axis=1)
        max_val = np.where(valid_mask, arr, -np.inf).max(axis=1)

        # When no object is detected, set distance to physical ceiling (255.0 mm)
        mean_val[no_obj] = NO_OBJECT_DIST
        min_val[no_obj] = NO_OBJECT_DIST
        max_val[no_obj] = NO_OBJECT_DIST
        std_val[no_obj] = 0.0

        # Steps where the sensor was broken (all NaN in the raw data) get no statistics here
        broken = df[f"tof_{i}_broken"].to_numpy() == 1
        active = n_valid.astype(np.float32) / 64.0  # Active pixel fraction (0..1)
        for s_arr in (active, mean_val, std_val, min_val, max_val):
            s_arr[broken] = np.nan

        # 4x4 Quadrant directional asymmetries on the 8x8 grid
        grid_valid = valid_mask.reshape(n_rows, 8, 8).astype(np.float32)
        q1 = grid_valid[:, :4, :4].mean(axis=(1, 2))  # Top-Left
        q2 = grid_valid[:, :4, 4:].mean(axis=(1, 2))  # Top-Right
        q3 = grid_valid[:, 4:, :4].mean(axis=(1, 2))  # Bottom-Left
        q4 = grid_valid[:, 4:, 4:].mean(axis=(1, 2))  # Bottom-Right

        lr_diff = 0.5 * (q2 + q4) - 0.5 * (q1 + q3)
        tb_diff = 0.5 * (q1 + q2) - 0.5 * (q3 + q4)
        lr_diff[broken] = np.nan
        tb_diff[broken] = np.nan

        tof_stats["active"][:, idx] = active
        tof_stats["mean"][:, idx] = mean_val
        tof_stats["std"][:, idx] = std_val
        tof_stats["min"][:, idx] = min_val
        tof_stats["max"][:, idx] = max_val
        tof_stats["lr_diff"][:, idx] = lr_diff
        tof_stats["tb_diff"][:, idx] = tb_diff

    # Impute per-sensor ToF summary features within each sequence, then cross-sensor fallback
    defaults = {
        "active": 0.0,
        "mean": NO_OBJECT_DIST,
        "std": 0.0,
        "min": NO_OBJECT_DIST,
        "max": NO_OBJECT_DIST,
        "lr_diff": 0.0,
        "tb_diff": 0.0,
    }
    for s_name in stat_names:
        s_cols = [f"tof_{i}_{s_name}" for i in range(1, 6)]
        s_df = pd.DataFrame(tof_stats[s_name], index=df.index, columns=s_cols)
        s_df = s_df.groupby(df["sequence_id"], sort=False).ffill()
        s_df = s_df.groupby(df["sequence_id"], sort=False).bfill()
        row_avg = s_df.mean(axis=1).fillna(defaults[s_name])
        for c in s_cols:
            s_df[c] = s_df[c].fillna(row_avg)
            feat[c] = s_df[c].to_numpy(dtype=np.float32)
        tof_stats[s_name] = s_df.to_numpy(dtype=np.float32)

    # Cross-sensor ToF aggregations (separates BFRB face/neck from knee/leg touches)
    act_mat = tof_stats["active"]
    mean_mat = tof_stats["mean"]
    min_mat = tof_stats["min"]

    feat["tof_active_mean"] = act_mat.mean(axis=1)
    feat["tof_active_std"] = act_mat.std(axis=1)
    feat["tof_mean_dist_all"] = mean_mat.mean(axis=1)
    feat["tof_min_dist_all"] = min_mat.min(axis=1)
    feat["tof_5_vs_1234_active"] = act_mat[:, 4] - act_mat[:, :4].mean(axis=1)
    feat["tof_5_vs_3_mean"] = mean_mat[:, 4] - mean_mat[:, 2]
    feat["tof_min_dist_diff"] = (
        feat["tof_min_dist_all"].groupby(df["sequence_id"], sort=False).diff().fillna(0.0)
    )


def add_tof_thm_features(df):
    """Per-step thermal and ToF features from the output of preprocess
    (rows sorted within each sequence).

    Output: DataFrame of new feature columns, same index as df.
    """
    feat = pd.DataFrame(index=df.index)
    add_thm_features(df, feat)
    add_tof_features(df, feat)
    return feat
