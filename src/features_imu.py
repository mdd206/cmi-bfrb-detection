import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from preprocess import ACC_COLS, ROT_COLS

LIN_ACC_COLS = ["lin_acc_x", "lin_acc_y", "lin_acc_z"]
GRAV_COLS = ["grav_x", "grav_y", "grav_z"]
ANGVEL_COLS = ["angvel_x", "angvel_y", "angvel_z", "angvel_mag"]
GRAVITY = [0.0, 0.0, 9.81]
# scipy expects quaternions as (x, y, z, w)
XYZW = ["rot_x", "rot_y", "rot_z", "rot_w"]

FEATURE_COLS = LIN_ACC_COLS + GRAV_COLS + ANGVEL_COLS + [
    "acc_mag",
    "acc_pitch",
    "acc_roll",
    "acc_jerk",
    "acc_mag_roll_mean_5",
    "acc_mag_roll_std_5",
    "acc_dyn_mag",
    "rot_roll",
    "rot_pitch",
    "rot_yaw",
    "rot_ang_vel",
]


def add_lin_acc(df, feat):
    """Add linear acceleration (acc minus gravity in the sensor frame); 0 where rot is missing."""
    gravity = Rotation.from_quat(df[XYZW].to_numpy()).apply(GRAVITY, inverse=True)
    lin_acc = df[ACC_COLS].to_numpy() - gravity
    lin_acc[df["rot_missing"].to_numpy() == 1] = 0.0
    feat[LIN_ACC_COLS] = lin_acc


def add_gravity_direction(df, feat):
    """Add the unit gravity direction seen from the watch (wrist tilt); 0 where rot is missing."""
    grav = Rotation.from_quat(df[XYZW].to_numpy()).apply([0.0, 0.0, 1.0], inverse=True)
    grav[df["rot_missing"].to_numpy() == 1] = 0.0
    feat[GRAV_COLS] = grav


def add_angular_velocity(df, feat):
    """Add rotation between consecutive steps in the watch frame (rad/step) and its size."""
    quat = df[XYZW].to_numpy()
    # first step of each sequence has no previous step, so it gets 0
    prev = df.groupby("sequence_id", sort=False)[XYZW].shift().fillna(df[XYZW]).to_numpy()
    angvel = (Rotation.from_quat(prev).inv() * Rotation.from_quat(quat)).as_rotvec()
    angvel[df["rot_missing"].to_numpy() == 1] = 0.0
    feat[ANGVEL_COLS[:3]] = angvel
    feat["angvel_mag"] = np.linalg.norm(angvel, axis=1)


def add_acc_features(df, feat):
    """Add magnitude, wrist tilt angles, jerk, and local rolling acceleration features."""
    ax = df["acc_x"].to_numpy(dtype=np.float64)
    ay = df["acc_y"].to_numpy(dtype=np.float64)
    az = df["acc_z"].to_numpy(dtype=np.float64)

    feat["acc_mag"] = np.sqrt(ax * ax + ay * ay + az * az)
    # Wrist tilt angles capture gravity direction (key for posture separation)
    feat["acc_pitch"] = np.arctan2(ay, np.sqrt(ax * ax + az * az))
    feat["acc_roll"] = np.arctan2(-ax, az)

    seq_group = df.groupby("sequence_id", sort=False)

    # Step-to-step acceleration change (jerk) within each sequence
    d_acc = seq_group[ACC_COLS].diff().fillna(0.0).to_numpy(dtype=np.float64)
    feat["acc_jerk"] = np.linalg.norm(d_acc, axis=1)

    # Short centered rolling window captures local motion intensity and tremor amplitude
    roll_5 = feat["acc_mag"].groupby(df["sequence_id"], sort=False).rolling(
        window=5, min_periods=1, center=True
    )
    feat["acc_mag_roll_mean_5"] = roll_5.mean().reset_index(level=0, drop=True)
    feat["acc_mag_roll_std_5"] = roll_5.std(ddof=0).reset_index(level=0, drop=True).fillna(0.0)

    # Subtract wider rolling mean (gravity baseline) to isolate dynamic hand movement
    grav_est = (
        seq_group[ACC_COLS]
        .rolling(window=11, min_periods=1, center=True)
        .mean()
        .reset_index(level=0, drop=True)
        .loc[df.index]
        .to_numpy(dtype=np.float64)
    )
    dyn_acc = df[ACC_COLS].to_numpy(dtype=np.float64) - grav_est
    feat["acc_dyn_mag"] = np.linalg.norm(dyn_acc, axis=1)


def add_rot_features(df, feat):
    """Add Euler angles and step-to-step angular velocity from unit quaternions."""
    w = df["rot_w"].to_numpy(dtype=np.float64)
    x = df["rot_x"].to_numpy(dtype=np.float64)
    y = df["rot_y"].to_numpy(dtype=np.float64)
    z = df["rot_z"].to_numpy(dtype=np.float64)

    # Intrinsic ZYX / extrinsic XYZ Euler angles (radians)
    feat["rot_roll"] = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    feat["rot_pitch"] = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    feat["rot_yaw"] = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    # Angular displacement between consecutive steps using |q_t . q_{t-1}|
    # (absolute dot product avoids pi jumps caused by q == -q sign ambiguity)
    q_curr = df[ROT_COLS].to_numpy(dtype=np.float64)
    q_prev = df.groupby("sequence_id", sort=False)[ROT_COLS].shift(1)
    q_prev = q_prev.fillna(df[ROT_COLS]).to_numpy(dtype=np.float64)
    dot = np.clip(np.abs(np.sum(q_curr * q_prev, axis=1)), 0.0, 1.0)
    feat["rot_ang_vel"] = 2.0 * np.arccos(dot)


def add_imu_features(df):
    """Per-step IMU features from the output of preprocess (rows sorted within each sequence).

    Output: DataFrame with FEATURE_COLS, same index as df.
    """
    feat = pd.DataFrame(index=df.index)
    add_lin_acc(df, feat)
    add_gravity_direction(df, feat)
    add_angular_velocity(df, feat)
    add_acc_features(df, feat)
    add_rot_features(df, feat)
    return feat[FEATURE_COLS]
