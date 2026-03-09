"""Animal3D loading and body-frame extraction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from .config import ANIMAL3D_NUM_JOINTS, LEG_JOINT_IDXS


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def load_sequence(directory: str | Path) -> np.ndarray:
    """Load all frame files named XXXX_3D.npz into (N, 26, 3)."""
    directory = Path(directory)
    files = sorted(directory.glob("*_3D.npz"), key=lambda p: int(p.stem.split("_")[0]))
    if not files:
        raise FileNotFoundError(f"No '*_3D.npz' files found in {directory}")

    poses = []
    for file in files:
        arr = np.load(file)["pose3d"]
        if arr.shape != (ANIMAL3D_NUM_JOINTS, 3):
            raise ValueError(f"{file} has shape {arr.shape}, expected ({ANIMAL3D_NUM_JOINTS}, 3)")
        poses.append(arr.astype(np.float64))
    return np.stack(poses, axis=0)


def compute_body_frame(pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute body frame rotation R (body->world) and translation t."""
    neck = pose[18]
    tail_base = pose[7]
    left_shoulder = pose[12]
    right_shoulder = pose[13]

    forward = _normalize(neck - tail_base)
    lateral_raw = left_shoulder - right_shoulder
    lateral = lateral_raw - np.dot(lateral_raw, forward) * forward
    lateral = _normalize(lateral)
    up = _normalize(np.cross(forward, lateral))

    # Re-orthogonalize for numerical stability.
    lateral = _normalize(np.cross(up, forward))
    R = np.column_stack((forward, lateral, up))
    t = neck.copy()
    return R, t


def compute_dog_leg_lengths(sequence: np.ndarray) -> Dict[str, float]:
    """Average shoulder->thigh->knee->paw chain length per leg across frames."""
    lengths: Dict[str, float] = {}
    for leg, idxs in LEG_JOINT_IDXS.items():
        shoulder = sequence[:, idxs["shoulder"], :]
        thigh = sequence[:, idxs["thigh"], :]
        knee = sequence[:, idxs["knee"], :]
        paw = sequence[:, idxs["paw"], :]
        l1 = np.linalg.norm(shoulder - thigh, axis=1)
        l2 = np.linalg.norm(thigh - knee, axis=1)
        l3 = np.linalg.norm(knee - paw, axis=1)
        lengths[leg] = float(np.mean(l1 + l2 + l3))
    return lengths


def rotation_matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (qx, qy, qz, qw)."""
    return Rotation.from_matrix(R).as_quat()

