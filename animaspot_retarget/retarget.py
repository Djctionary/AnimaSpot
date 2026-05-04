"""End-to-end Animal3D -> Spot retargeting pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from .config import (
    HIP_ATTACHMENTS,
    HIP_X_OFFSET,
    JOINT_LIMITS,
    LEG_ORDER,
    LEG_SIDE,
    LEG_TARGET_JOINT_IDXS,
    L_LOWER,
    L_UPPER,
    RetargetConfig,
)
from .ik_solver import forward_kinematics, leg_keypoints, solve_leg_ik
from .skeleton import compute_body_frame, compute_dog_leg_lengths, load_sequence, rotation_matrix_to_quat_xyzw

LOGGER = logging.getLogger(__name__)


METHOD_ANALYTICAL_IK = "analytical_ik"
METHOD_TRAJECTORY_IK = "trajectory_ik"
RETARGET_METHODS = (METHOD_ANALYTICAL_IK, METHOD_TRAJECTORY_IK)


@dataclass
class RetargetContext:
    """Shared stage data consumed by retarget methods and artifact writers."""

    input_dir: Path
    sequence: np.ndarray
    scales: Dict[str, float]
    body_axes: np.ndarray
    body_origins: np.ndarray
    root_quat_raw: np.ndarray
    scaled_targets: np.ndarray
    fps: int


@dataclass
class RetargetRun:
    """Complete output of one retarget method execution."""

    method_name: str
    context: RetargetContext
    raw_joint_angles: np.ndarray
    smoothed_joint_angles: np.ndarray
    root_pos_stage6: np.ndarray
    root_quat_stage6: np.ndarray
    result: Dict[str, np.ndarray]


def _apply_joint_test_overrides(joint_angles: np.ndarray, config: RetargetConfig) -> np.ndarray:
    """Apply optional test-only joint overrides in-place."""
    if config.fix_hx_zero:
        joint_angles[..., [0, 3, 6, 9]] = 0.0
    return joint_angles



# ---------------------------------------------------------------------------
# 1-Euro filter  (Casiez et al., CHI 2012)
# ---------------------------------------------------------------------------

def _one_euro_pass(data: np.ndarray, freq: float, min_cutoff: float, beta: float, d_cutoff: float) -> np.ndarray:
    """Single causal pass of the 1-Euro filter on an (N, D) array."""
    n = data.shape[0]
    out = np.empty_like(data)
    out[0] = data[0]
    dx = np.zeros(data.shape[1], dtype=np.float64)

    te = 1.0 / freq
    tau_d = 1.0 / (2.0 * np.pi * d_cutoff)
    alpha_d = 1.0 / (1.0 + tau_d / te)

    for i in range(1, n):
        raw_dx = (data[i] - out[i - 1]) * freq
        dx = alpha_d * raw_dx + (1.0 - alpha_d) * dx
        cutoff = min_cutoff + beta * np.abs(dx)
        tau = 1.0 / (2.0 * np.pi * cutoff)
        alpha = 1.0 / (1.0 + tau / te)
        out[i] = alpha * data[i] + (1.0 - alpha) * out[i - 1]
    return out


def one_euro_filter(
    data: np.ndarray,
    freq: float,
    min_cutoff: float = 1.7,
    beta: float = 0.01,
    d_cutoff: float = 1.0,
) -> np.ndarray:
    """Zero-phase 1-Euro filter (forward-backward) for an (N, D) array.

    Forward-backward eliminates the phase lag inherent in causal filtering,
    which is critical for offline motion-capture post-processing.
    """
    if data.shape[0] < 3:
        return data.copy()
    forward = _one_euro_pass(data, freq, min_cutoff, beta, d_cutoff)
    backward = _one_euro_pass(forward[::-1], freq, min_cutoff, beta, d_cutoff)
    return backward[::-1].copy()


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

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


def _smooth_quaternions(
    quats: np.ndarray,
    freq: float,
    min_cutoff: float,
    beta: float,
    d_cutoff: float,
) -> np.ndarray:
    """Smooth an (N, 4) quaternion sequence via 1-Euro on components."""
    quats = _enforce_quat_continuity(_normalize_quat(quats))
    smoothed = one_euro_filter(quats, freq, min_cutoff, beta, d_cutoff)
    return _normalize_quat(smoothed)


# ---------------------------------------------------------------------------
# Rotation helper
# ---------------------------------------------------------------------------

def _rotation_between(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Minimum-angle rotation matrix that maps *v_from* onto *v_to* (Rodrigues)."""
    a = v_from / np.linalg.norm(v_from)
    b = v_to / np.linalg.norm(v_to)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    if s < 1e-8:
        if c > 0.0:
            return np.eye(3, dtype=np.float64)
        # 180-degree rotation: choose any stable axis orthogonal to *a*.
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = axis - np.dot(axis, a) * a
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


# ---------------------------------------------------------------------------
# IK pipeline helpers
# ---------------------------------------------------------------------------

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
        target_idx = LEG_TARGET_JOINT_IDXS[leg]
        target_world = pose[target_idx]
        target_body = R.T @ (target_world - t)
        targets[leg] = target_body - HIP_ATTACHMENTS[leg]
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


# ---------------------------------------------------------------------------
# Paw body-frame positions (shared by ground contact + validation)
# ---------------------------------------------------------------------------

def _compute_paw_body_positions(joint_angles_frame: np.ndarray) -> np.ndarray:
    """Return (4, 3) paw positions in body frame for one frame."""
    paw_body = np.zeros((4, 3), dtype=np.float64)
    for i, leg in enumerate(LEG_ORDER):
        hx, hy, kn = joint_angles_frame[3 * i : 3 * i + 3]
        paw_local = forward_kinematics(hx, hy, kn, HIP_X_OFFSET, L_UPPER, L_LOWER, LEG_SIDE[leg])
        paw_body[i] = HIP_ATTACHMENTS[leg] + paw_local
    return paw_body


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def prepare_retarget_context(input_dir: str | Path, config: RetargetConfig) -> RetargetContext:
    """Run shared preprocessing stages used by every retarget method."""
    input_path = Path(input_dir)
    sequence = load_sequence(input_path)
    n_frames = sequence.shape[0]
    scales = compute_leg_scale_factors(sequence)

    body_axes = np.zeros((n_frames, 3, 3), dtype=np.float64)
    body_origins = np.zeros((n_frames, 3), dtype=np.float64)
    root_quat_raw = np.zeros((n_frames, 4), dtype=np.float64)
    scaled_targets = np.zeros((n_frames, len(LEG_ORDER), 3), dtype=np.float64)

    for frame_idx, pose in enumerate(sequence):
        R, t = compute_body_frame(pose)
        body_axes[frame_idx] = R
        body_origins[frame_idx] = t
        root_quat_raw[frame_idx] = rotation_matrix_to_quat_xyzw(R)
        targets = compute_paw_targets_body_frame(pose, R, t)
        for leg_idx, leg in enumerate(LEG_ORDER):
            scaled_targets[frame_idx, leg_idx] = targets[leg] * scales[leg]

    return RetargetContext(
        input_dir=input_path,
        sequence=sequence,
        scales=scales,
        body_axes=body_axes,
        body_origins=body_origins,
        root_quat_raw=root_quat_raw,
        scaled_targets=scaled_targets,
        fps=config.fps,
    )


def _solve_analytical_ik(context: RetargetContext, config: RetargetConfig) -> np.ndarray:
    """Solve the AnalyticalIK retarget stage frame-by-frame."""
    joint_angles = np.zeros((context.sequence.shape[0], 12), dtype=np.float64)
    for frame_idx, scaled_targets in enumerate(context.scaled_targets):
        frame_angles = np.zeros((12,), dtype=np.float64)
        for leg_idx, leg in enumerate(LEG_ORDER):
            target = scaled_targets[leg_idx]
            s = 1.0 if LEG_SIDE[leg] == "left" else -1.0
            y_out = s * target[1]
            yz_norm = np.hypot(y_out, target[2])
            if yz_norm < HIP_X_OFFSET - 1e-6:
                LOGGER.warning("Leg %s target inside hip-offset radius; clamping via IK.", leg)
            z_sag = -np.sqrt(max(yz_norm * yz_norm - HIP_X_OFFSET * HIP_X_OFFSET, 0.0))
            reach = np.hypot(target[0], z_sag)
            if reach > L_UPPER + L_LOWER + 1e-6:
                LOGGER.warning("Leg %s target out of reach (%.3f m); clamping via IK.", leg, reach)

            frame_angles[3 * leg_idx : 3 * leg_idx + 3] = solve_leg_ik(
                target_pos=target,
                hip_offset=HIP_X_OFFSET,
                L_upper=L_UPPER,
                L_lower=L_LOWER,
                joint_limits=JOINT_LIMITS,
                side=LEG_SIDE[leg],
            )
        joint_angles[frame_idx] = frame_angles
    return _apply_joint_test_overrides(joint_angles, config)


def _smooth_and_clamp_joint_angles(joint_angles: np.ndarray, config: RetargetConfig) -> np.ndarray:
    """Apply the shared AnalyticalIK smoothing and hard joint limits."""
    freq = float(config.fps)
    smoothed = one_euro_filter(
        joint_angles,
        freq,
        min_cutoff=config.one_euro_min_cutoff,
        beta=config.one_euro_beta,
        d_cutoff=config.one_euro_d_cutoff,
    )

    for joint_idx, key in enumerate(["hx", "hy", "kn"] * 4):
        lo, hi = JOINT_LIMITS[key]
        smoothed[:, joint_idx] = np.clip(smoothed[:, joint_idx], lo, hi)
    return _apply_joint_test_overrides(smoothed, config)


def run_retarget_pipeline(
    input_dir: str | Path,
    config: RetargetConfig,
    method: str = METHOD_ANALYTICAL_IK,
) -> RetargetRun:
    """Run shared stages plus the selected retarget method."""
    if method not in RETARGET_METHODS:
        valid = ", ".join(RETARGET_METHODS)
        raise ValueError(f"Unknown retarget method '{method}'. Expected one of: {valid}")

    context = prepare_retarget_context(input_dir, config)

    freq = float(config.fps)
    root_quat_stage6 = _smooth_quaternions(
        context.root_quat_raw,
        freq,
        min_cutoff=config.one_euro_min_cutoff,
        beta=config.one_euro_beta,
        d_cutoff=config.one_euro_d_cutoff,
    )
    if config.root_quaternion is not None:
        manual_root_quat = _normalize_quat(np.asarray(config.root_quaternion, dtype=np.float64))
        if manual_root_quat.shape != (4,):
            raise ValueError(
                f"Expected config.root_quaternion shape (4,), got {manual_root_quat.shape}"
            )
        root_quat_stage6 = np.tile(manual_root_quat, (context.sequence.shape[0], 1))
    root_pos_stage6 = np.tile(np.array(config.root_position, dtype=np.float64), (context.sequence.shape[0], 1))

    if method == METHOD_ANALYTICAL_IK:
        raw_joint_angles = _solve_analytical_ik(context, config)
        smoothed_joint_angles = _smooth_and_clamp_joint_angles(raw_joint_angles, config)
    else:
        from .trajectory_ik import solve_trajectory_ik

        analytical_init = _solve_analytical_ik(context, config)
        raw_joint_angles = solve_trajectory_ik(
            context,
            config,
            q_init=analytical_init,
            root_quat=root_quat_stage6,
            root_pos=root_pos_stage6,
        )
        smoothed_joint_angles = raw_joint_angles.copy()

    if config.ground_contact:
        LOGGER.warning(
            "RetargetConfig.ground_contact is deprecated and ignored inside the retarget pipeline. "
            "Use the independent global pose postprocess instead."
        )

    # --- Validation ---
    rmse = validate_fk_rmse(smoothed_joint_angles, context.scaled_targets)
    LOGGER.info("FK target RMSE: %.4f m", rmse)
    if rmse > 0.02:
        LOGGER.warning("FK RMSE %.4f m exceeds 2cm criterion.", rmse)
    validate_link_length_invariance(smoothed_joint_angles, tol=1e-4)

    result = {
        "joint_angles": smoothed_joint_angles,
        "root_quat": _normalize_quat(root_quat_stage6),
        "root_pos": root_pos_stage6.copy(),
        "fps": np.array(config.fps, dtype=np.int32),
    }

    if config.postprocess_global_pose and method == METHOD_ANALYTICAL_IK:
        from .postprocess import apply_global_pose_postprocess

        result = apply_global_pose_postprocess(result, config)

    return RetargetRun(
        method_name=method,
        context=context,
        raw_joint_angles=raw_joint_angles,
        smoothed_joint_angles=smoothed_joint_angles,
        root_pos_stage6=root_pos_stage6,
        root_quat_stage6=_normalize_quat(root_quat_stage6),
        result=result,
    )


def retarget_sequence(
    input_dir: str | Path,
    config: RetargetConfig,
    method: str = METHOD_ANALYTICAL_IK,
) -> Dict[str, np.ndarray]:
    """Run retargeting and return arrays for export."""
    return run_retarget_pipeline(input_dir, config, method=method).result
