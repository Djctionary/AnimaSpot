"""End-to-end Animal3D -> Spot retargeting pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.signal import savgol_filter

from .config import (
    HIP_ATTACHMENTS,
    HIP_X_OFFSET,
    JOINT_LIMITS,
    LEG_JOINT_IDXS,
    LEG_ORDER,
    LEG_SIDE,
    L_LOWER,
    L_UPPER,
    RetargetConfig,
)
from .ik_solver import forward_kinematics, leg_keypoints, solve_leg_ik
from .skeleton import compute_body_frame, compute_dog_leg_lengths, load_sequence, rotation_matrix_to_quat_xyzw

LOGGER = logging.getLogger(__name__)


def _safe_savgol(data: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    n = data.shape[0]
    if n < 3:
        return data.copy()
    window = min(window, n if n % 2 == 1 else n - 1)
    if window < 3:
        return data.copy()
    polyorder = min(polyorder, window - 1)
    return savgol_filter(data, window_length=window, polyorder=polyorder, axis=0, mode="interp")


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n[n < 1e-12] = 1.0
    return q / n


def _enforce_quat_continuity(quats: np.ndarray) -> np.ndarray:
    out = quats.copy()
    for i in range(1, out.shape[0]):
        if np.dot(out[i - 1], out[i]) < 0.0:
            out[i] *= -1.0
    return out


def _slerp_pair(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1


def smooth_quaternions_slerp(quats: np.ndarray, alpha: float) -> np.ndarray:
    """Two-pass SLERP smoothing (forward then backward)."""
    quats = _enforce_quat_continuity(_normalize_quat(quats))
    out = quats.copy()
    for i in range(1, out.shape[0]):
        out[i] = _slerp_pair(out[i - 1], out[i], alpha)
    for i in range(out.shape[0] - 2, -1, -1):
        out[i] = _slerp_pair(out[i + 1], out[i], alpha)
    return _normalize_quat(_enforce_quat_continuity(out))


def compute_leg_scale_factors(sequence: np.ndarray) -> Dict[str, float]:
    dog_leg_lengths = compute_dog_leg_lengths(sequence)
    spot_leg_length = L_UPPER + L_LOWER
    scales = {}
    for leg, dog_len in dog_leg_lengths.items():
        if dog_len < 1e-6:
            LOGGER.warning("Dog leg length near zero for %s. Falling back to scale=1.0.", leg)
            scales[leg] = 1.0
        else:
            scales[leg] = spot_leg_length / dog_len
    return scales


def compute_paw_targets_body_frame(pose: np.ndarray, R: np.ndarray, t: np.ndarray) -> Dict[str, np.ndarray]:
    targets = {}
    for leg in LEG_ORDER:
        paw_idx = LEG_JOINT_IDXS[leg]["paw"]
        paw_world = pose[paw_idx]
        paw_body = R.T @ (paw_world - t)
        targets[leg] = paw_body - HIP_ATTACHMENTS[leg]
    return targets


def solve_frame_ik(pose: np.ndarray, scales: Dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    R, t = compute_body_frame(pose)
    quat = rotation_matrix_to_quat_xyzw(R)
    paw_targets = compute_paw_targets_body_frame(pose, R, t)

    frame_angles = np.zeros((12,), dtype=np.float64)
    for i, leg in enumerate(LEG_ORDER):
        target = paw_targets[leg] * scales[leg]
        # Reachability diagnostics.
        s = 1.0 if LEG_SIDE[leg] == "left" else -1.0
        y_out = s * target[1]
        yz_norm = np.hypot(y_out, target[2])
        if yz_norm < HIP_X_OFFSET - 1e-6:
            LOGGER.warning("Leg %s target inside hip-offset radius; clamping via IK.", leg)
        z_sag = -np.sqrt(max(yz_norm * yz_norm - HIP_X_OFFSET * HIP_X_OFFSET, 0.0))
        reach = np.hypot(target[0], z_sag)
        if reach > L_UPPER + L_LOWER + 1e-6:
            LOGGER.warning("Leg %s target out of reach (%.3f m); clamping via IK.", leg, reach)

        hx, hy, kn = solve_leg_ik(
            target_pos=target,
            hip_offset=HIP_X_OFFSET,
            L_upper=L_UPPER,
            L_lower=L_LOWER,
            joint_limits=JOINT_LIMITS,
            side=LEG_SIDE[leg],
        )
        frame_angles[3 * i : 3 * i + 3] = [hx, hy, kn]
    return frame_angles, quat


def validate_fk_rmse(joint_angles: np.ndarray, paw_targets_scaled: np.ndarray) -> float:
    """Compute FK-vs-target RMSE for diagnostics."""
    all_errors = []
    for f in range(joint_angles.shape[0]):
        for i, leg in enumerate(LEG_ORDER):
            hx, hy, kn = joint_angles[f, 3 * i : 3 * i + 3]
            pred = forward_kinematics(hx, hy, kn, HIP_X_OFFSET, L_UPPER, L_LOWER, LEG_SIDE[leg])
            err = np.linalg.norm(pred - paw_targets_scaled[f, i])
            all_errors.append(err)
    return float(np.sqrt(np.mean(np.square(all_errors))))


def validate_link_length_invariance(joint_angles: np.ndarray, tol: float = 1e-4) -> float:
    """
    Validate rigid-link consistency across all frames.

    Returns max absolute length deviation from nominal links.
    """
    max_dev = 0.0
    for f in range(joint_angles.shape[0]):
        for i, leg in enumerate(LEG_ORDER):
            hx, hy, kn = joint_angles[f, 3 * i : 3 * i + 3]
            knee, paw = leg_keypoints(hx, hy, kn, HIP_X_OFFSET, L_UPPER, L_LOWER, LEG_SIDE[leg])
            s = 1.0 if LEG_SIDE[leg] == "left" else -1.0
            hx_raw = s * hx
            c = np.cos(hx_raw)
            ss = np.sin(hx_raw)
            hy_joint = np.array([0.0, s * (HIP_X_OFFSET * c), HIP_X_OFFSET * ss], dtype=np.float64)
            upper = np.linalg.norm(knee - hy_joint)
            lower = np.linalg.norm(paw - knee)
            max_dev = max(max_dev, abs(upper - L_UPPER), abs(lower - L_LOWER))
    if max_dev > tol:
        LOGGER.warning("Link-length invariance violation: max deviation %.6f m", max_dev)
    else:
        LOGGER.info("Link-length invariance check passed (max deviation %.6f m)", max_dev)
    return float(max_dev)


def retarget_sequence(input_dir: str | Path, config: RetargetConfig) -> Dict[str, np.ndarray]:
    """Run full retargeting and return arrays for export."""
    sequence = load_sequence(input_dir)
    n_frames = sequence.shape[0]
    scales = compute_leg_scale_factors(sequence)

    joint_angles = np.zeros((n_frames, 12), dtype=np.float64)
    root_quat = np.zeros((n_frames, 4), dtype=np.float64)
    root_pos = np.tile(np.array(config.root_position, dtype=np.float64), (n_frames, 1))
    paw_targets_scaled = np.zeros((n_frames, 4, 3), dtype=np.float64)

    for i in range(n_frames):
        pose = sequence[i]
        angles, quat = solve_frame_ik(pose, scales)
        joint_angles[i] = angles
        root_quat[i] = quat

        # Cache scaled targets for FK error diagnostics.
        R, t = compute_body_frame(pose)
        targets = compute_paw_targets_body_frame(pose, R, t)
        for j, leg in enumerate(LEG_ORDER):
            paw_targets_scaled[i, j] = targets[leg] * scales[leg]

    joint_angles = _safe_savgol(joint_angles, config.smooth_window, config.smooth_polyorder)
    root_quat = smooth_quaternions_slerp(root_quat, config.quat_slerp_alpha)

    # Clamp post-smoothing to enforce physical limits.
    for j, key in enumerate(["hx", "hy", "kn"] * 4):
        lo, hi = JOINT_LIMITS[key]
        joint_angles[:, j] = np.clip(joint_angles[:, j], lo, hi)

    rmse = validate_fk_rmse(joint_angles, paw_targets_scaled)
    LOGGER.info("FK target RMSE: %.4f m", rmse)
    if rmse > 0.02:
        LOGGER.warning("FK RMSE %.4f m exceeds 2cm criterion.", rmse)
    validate_link_length_invariance(joint_angles, tol=1e-4)

    return {
        "joint_angles": joint_angles,
        "root_quat": _normalize_quat(root_quat),
        "root_pos": root_pos,
        "fps": np.array(config.fps, dtype=np.int32),
    }

