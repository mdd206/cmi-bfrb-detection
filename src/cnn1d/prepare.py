import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ACC_COLS = ["acc_x", "acc_y", "acc_z"]
ROT_COLS = ["rot_w", "rot_x", "rot_y", "rot_z"]
LIN_ACC_COLS = ["lin_acc_x", "lin_acc_y", "lin_acc_z"]
GRAV_COLS = ["grav_x", "grav_y", "grav_z"]
ANGVEL_COLS = ["angvel_x", "angvel_y", "angvel_z", "angvel_mag"]
JERK_COLS = ["jerk_x", "jerk_y", "jerk_z"]
MIRROR_COLS = ["acc_x", "rot_y", "rot_z"]
IDENTITY_QUAT = {"rot_w": 1.0, "rot_x": 0.0, "rot_y": 0.0, "rot_z": 0.0}
GRAVITY = [0.0, 0.0, 9.81]
BASE_FEATURES = LIN_ACC_COLS + ROT_COLS + ["rot_missing"]


def load_data(root):
    train = pd.read_csv(root / "data/raw/train.csv",
                        usecols=["sequence_id", "sequence_counter", "subject"] + ACC_COLS + ROT_COLS)
    demo = pd.read_csv(root / "data/raw/train_demographics.csv")
    folds = pd.read_csv(root / "folds/folds.csv")
    return train, demo, folds


def preprocess(df, demo, quat_sign=False):
    df = mirror_left_handed(df, demo)
    df = fill_missing_rot(df)
    if quat_sign:
        df = fix_quat_sign(df)
    df = remove_gravity(df)
    df = add_gravity_direction(df)
    df = add_angular_velocity(df)
    df = add_lin_acc_features(df)
    return df


def make_padded(df, cols, sequence_ids, max_len):
    """Return (n_sequences, max_len, n_cols): keep the last max_len steps, NaN-pad at the start."""
    groups = {sid: g[cols].to_numpy(dtype=np.float32) for sid, g in df.groupby("sequence_id")}
    out = np.full((len(sequence_ids), max_len, len(cols)), np.nan, dtype=np.float32)
    for i, sid in enumerate(sequence_ids):
        seq = groups[sid][-max_len:]
        out[i, max_len - len(seq):] = seq
    return out


def fill_missing_rot(df):
    """Add rot_missing flag, fill rot gaps inside each sequence, use identity if all missing."""
    df = df.copy()
    df["rot_missing"] = df[ROT_COLS].isna().any(axis=1).astype("int8")
    df[ROT_COLS] = df.groupby("sequence_id")[ROT_COLS].ffill()
    df[ROT_COLS] = df.groupby("sequence_id")[ROT_COLS].bfill()
    df[ROT_COLS] = df[ROT_COLS].fillna(IDENTITY_QUAT)
    return df


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


def remove_gravity(df):
    """Add linear acceleration (acc minus gravity in the sensor frame); 0 where rot is missing."""
    df = df.copy()
    # scipy expects quaternions as (x, y, z, w)
    quat = df[["rot_x", "rot_y", "rot_z", "rot_w"]].to_numpy()
    gravity = Rotation.from_quat(quat).apply(GRAVITY, inverse=True)
    lin_acc = df[ACC_COLS].to_numpy() - gravity
    lin_acc[df["rot_missing"].to_numpy() == 1] = 0.0
    df[LIN_ACC_COLS] = lin_acc
    return df


def add_gravity_direction(df):
    """Add the unit gravity direction seen from the watch (wrist tilt); 0 where rot is missing."""
    df = df.copy()
    quat = df[["rot_x", "rot_y", "rot_z", "rot_w"]].to_numpy()
    grav = Rotation.from_quat(quat).apply([0.0, 0.0, 1.0], inverse=True)
    grav[df["rot_missing"].to_numpy() == 1] = 0.0
    df[GRAV_COLS] = grav
    return df


def add_angular_velocity(df):
    """Add rotation between consecutive steps in the watch frame (rad/step) and its size."""
    df = df.copy()
    xyzw = ["rot_x", "rot_y", "rot_z", "rot_w"]
    quat = df[xyzw].to_numpy()
    # first step of each sequence has no previous step, so it gets 0
    prev = df.groupby("sequence_id")[xyzw].shift().fillna(df[xyzw]).to_numpy()
    angvel = (Rotation.from_quat(prev).inv() * Rotation.from_quat(quat)).as_rotvec()
    angvel[df["rot_missing"].to_numpy() == 1] = 0.0
    df[ANGVEL_COLS[:3]] = angvel
    df["angvel_mag"] = np.linalg.norm(angvel, axis=1)
    return df


def add_lin_acc_features(df):
    """Add the size of linear acceleration and its change between steps (jerk)."""
    df = df.copy()
    df["lin_acc_mag"] = np.linalg.norm(df[LIN_ACC_COLS].to_numpy(), axis=1)
    # first step of each sequence has no previous step, so it gets 0
    df[JERK_COLS] = df.groupby("sequence_id")[LIN_ACC_COLS].diff().fillna(0.0).to_numpy()
    return df


def mirror_left_handed(df, demo):
    """Mirror left-handed subjects across the YZ plane so they look right-handed."""
    df = df.copy()
    left_subjects = demo.loc[demo["handedness"] == 0, "subject"]
    left = df["subject"].isin(left_subjects)
    df.loc[left, MIRROR_COLS] *= -1
    return df
