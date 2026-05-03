"""Trajectory-level soft-constrained IK for Spot retargeting."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from .config import HIP_ATTACHMENTS, HIP_X_OFFSET, JOINT_LIMITS, LEG_ORDER, LEG_SIDE, L_LOWER, L_UPPER, RetargetConfig
from .ik_solver import forward_kinematics

if TYPE_CHECKING:
    from .retarget import RetargetContext

LOGGER = logging.getLogger(__name__)


def _joint_bounds(n_frames: int) -> list[tuple[float, float]]:
    per_frame = [JOINT_LIMITS[key] for key in ["hx", "hy", "kn"] * len(LEG_ORDER)]
    return per_frame * n_frames


def _paw_body_positions(q: np.ndarray) -> np.ndarray:
    """Return FK paw positions relative to each hip attachment, shape (T, 4, 3)."""
    paw = np.zeros((q.shape[0], len(LEG_ORDER), 3), dtype=np.float64)
    for frame_idx in range(q.shape[0]):
        for leg_idx, leg in enumerate(LEG_ORDER):
            hx, hy, kn = q[frame_idx, 3 * leg_idx : 3 * leg_idx + 3]
            paw[frame_idx, leg_idx] = forward_kinematics(
                hx,
                hy,
                kn,
                HIP_X_OFFSET,
                L_UPPER,
                L_LOWER,
                LEG_SIDE[leg],
            )
    return paw


def _ground_penetration_cost(
    paw_body: np.ndarray,
    root_quat: np.ndarray,
    root_pos: np.ndarray,
    ground_level: float,
) -> float:
    rotations = Rotation.from_quat(root_quat).as_matrix()
    hip_offsets = np.stack([HIP_ATTACHMENTS[leg] for leg in LEG_ORDER], axis=0)
    penalty = 0.0
    for frame_idx in range(paw_body.shape[0]):
        paw_world = (rotations[frame_idx] @ (hip_offsets + paw_body[frame_idx]).T).T + root_pos[frame_idx]
        penetration = np.maximum(ground_level - paw_world[:, 2], 0.0)
        penalty += float(np.sum(penetration * penetration))
    return penalty


def solve_trajectory_ik(
    context: "RetargetContext",
    config: RetargetConfig,
    q_init: np.ndarray,
    root_quat: np.ndarray,
    root_pos: np.ndarray,
) -> np.ndarray:
    """Optimize a full joint-angle sequence with four soft objective terms."""
    q_init = np.asarray(q_init, dtype=np.float64)
    if q_init.ndim != 2 or q_init.shape[1] != 12:
        raise ValueError(f"Expected q_init shape (T, 12), got {q_init.shape}")

    n_frames = q_init.shape[0]
    targets = np.asarray(context.scaled_targets, dtype=np.float64)
    stable_joint_indices = tuple(
        idx for idx in config.trajectory_stable_joint_indices
        if 0 <= idx < q_init.shape[1]
    )
    q_ref = q_init.copy()

    def objective(flat_q: np.ndarray) -> float:
        q = flat_q.reshape(n_frames, 12)
        paw_body = _paw_body_positions(q)

        track = float(np.sum((paw_body - targets) ** 2))

        if n_frames > 1:
            dq = q[1:] - q[:-1]
            velocity = float(np.sum(dq * dq))
        else:
            velocity = 0.0
        if n_frames > 2:
            ddq = q[2:] - 2.0 * q[1:-1] + q[:-2]
            acceleration = float(np.sum(ddq * ddq))
        else:
            acceleration = 0.0
        smooth = (
            config.trajectory_smooth_velocity_weight * velocity
            + config.trajectory_smooth_acceleration_weight * acceleration
        )

        ground = _ground_penetration_cost(
            paw_body,
            root_quat,
            root_pos,
            config.trajectory_ground_level,
        )

        if stable_joint_indices:
            stable_delta = q[:, stable_joint_indices] - q_ref[:, stable_joint_indices]
            stable = float(np.sum(stable_delta * stable_delta))
        else:
            stable = 0.0

        return (
            config.trajectory_w_track * track
            + config.trajectory_w_smooth * smooth
            + config.trajectory_w_ground * ground
            + config.trajectory_w_stable * stable
        )

    initial_cost = objective(q_init.reshape(-1))
    progress = tqdm(
        total=config.trajectory_maxiter,
        desc="TrajectoryIK",
        unit="iter",
        dynamic_ncols=True,
    )

    def callback(flat_q: np.ndarray) -> None:
        cost = objective(flat_q)
        progress.update(1)
        progress.set_postfix(cost=f"{cost:.6g}")

    try:
        result = minimize(
            objective,
            q_init.reshape(-1),
            method="L-BFGS-B",
            bounds=_joint_bounds(n_frames),
            callback=callback,
            options={
                "maxiter": config.trajectory_maxiter,
                "ftol": config.trajectory_ftol,
            },
        )
    finally:
        progress.close()

    if not result.success:
        LOGGER.warning("TrajectoryIK did not fully converge: %s", result.message)
    LOGGER.info(
        "TrajectoryIK objective %.6f -> %.6f in %d iteration(s).",
        initial_cost,
        float(result.fun),
        int(result.nit),
    )
    return result.x.reshape(n_frames, 12)
