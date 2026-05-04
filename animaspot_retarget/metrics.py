"""Evaluation metrics for retargeted Spot motion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from .config import HIP_ATTACHMENTS, HIP_X_OFFSET, LEG_JOINT_IDXS, LEG_ORDER, LEG_SIDE, L_LOWER, L_UPPER
from .ik_solver import leg_keypoints

if TYPE_CHECKING:
    from .config import RetargetConfig
    from .retarget import RetargetRun


@dataclass(frozen=True)
class RetargetMetrics:
    """Scalar metrics computed from one completed retarget run."""

    scale_aligned_mpjpe: float
    joint_jump_rate: float
    ground_penetration_rate: float
    joint_jump_threshold: float
    ground_level: float


def _source_scaled_leg_landmarks(run: "RetargetRun") -> np.ndarray:
    """Return source leg landmarks scaled into the Spot body frame, shape (T, 12, 3)."""
    context = run.context
    landmarks = np.zeros((context.sequence.shape[0], len(LEG_ORDER) * 3, 3), dtype=np.float64)

    for frame_idx, pose in enumerate(context.sequence):
        R = context.body_axes[frame_idx]
        t = context.body_origins[frame_idx]
        point_idx = 0
        for leg in LEG_ORDER:
            hip = HIP_ATTACHMENTS[leg]
            idxs = LEG_JOINT_IDXS[leg]
            scale = context.scales[leg]
            for joint_name in ("thigh", "knee", "paw"):
                body_pos = R.T @ (pose[idxs[joint_name]] - t)
                landmarks[frame_idx, point_idx] = hip + (body_pos - hip) * scale
                point_idx += 1
    return landmarks


def _robot_leg_landmarks_body(joint_angles: np.ndarray) -> np.ndarray:
    """Return Spot leg articulation landmarks in body frame, shape (T, 12, 3)."""
    joint_angles = np.asarray(joint_angles, dtype=np.float64)
    landmarks = np.zeros((joint_angles.shape[0], len(LEG_ORDER) * 3, 3), dtype=np.float64)

    for frame_idx in range(joint_angles.shape[0]):
        point_idx = 0
        for leg_idx, leg in enumerate(LEG_ORDER):
            hip = HIP_ATTACHMENTS[leg]
            hx, hy, kn = joint_angles[frame_idx, 3 * leg_idx : 3 * leg_idx + 3]
            side_sign = 1.0 if LEG_SIDE[leg] == "left" else -1.0
            hx_raw = side_sign * hx
            abduction_joint = np.array(
                [
                    0.0,
                    side_sign * (HIP_X_OFFSET * np.cos(hx_raw)),
                    HIP_X_OFFSET * np.sin(hx_raw),
                ],
                dtype=np.float64,
            )
            knee_rel, paw_rel = leg_keypoints(hx, hy, kn, HIP_X_OFFSET, L_UPPER, L_LOWER, LEG_SIDE[leg])
            landmarks[frame_idx, point_idx] = hip + abduction_joint
            landmarks[frame_idx, point_idx + 1] = hip + knee_rel
            landmarks[frame_idx, point_idx + 2] = hip + paw_rel
            point_idx += 3
    return landmarks


def _paw_positions_world(joint_angles: np.ndarray, root_pos: np.ndarray, root_quat: np.ndarray) -> np.ndarray:
    """Return paw world coordinates, shape (T, 4, 3)."""
    robot_landmarks_body = _robot_leg_landmarks_body(joint_angles).reshape(joint_angles.shape[0], len(LEG_ORDER), 3, 3)
    paw_body = robot_landmarks_body[:, :, 2, :]
    rotations = Rotation.from_quat(root_quat).as_matrix()
    paw_world = np.empty_like(paw_body)
    for frame_idx in range(joint_angles.shape[0]):
        paw_world[frame_idx] = (rotations[frame_idx] @ paw_body[frame_idx].T).T + root_pos[frame_idx]
    return paw_world


def compute_retarget_metrics(run: "RetargetRun", config: "RetargetConfig") -> RetargetMetrics:
    """Compute the default retarget-evaluation metrics for one run."""
    source_landmarks = _source_scaled_leg_landmarks(run)
    robot_landmarks = _robot_leg_landmarks_body(run.result["joint_angles"])
    mpjpe = float(np.mean(np.linalg.norm(source_landmarks - robot_landmarks, axis=-1)))

    joint_angles = np.asarray(run.result["joint_angles"], dtype=np.float64)
    if joint_angles.shape[0] > 1:
        jump_mask = np.max(np.abs(np.diff(joint_angles, axis=0)), axis=1) > config.metrics_joint_jump_threshold
        joint_jump_rate = float(np.mean(jump_mask))
    else:
        joint_jump_rate = 0.0

    paw_world = _paw_positions_world(
        joint_angles,
        np.asarray(run.result["root_pos"], dtype=np.float64),
        np.asarray(run.result["root_quat"], dtype=np.float64),
    )
    penetration_mask = np.min(paw_world[:, :, 2], axis=1) < config.metrics_ground_level
    ground_penetration_rate = float(np.mean(penetration_mask))

    return RetargetMetrics(
        scale_aligned_mpjpe=mpjpe,
        joint_jump_rate=joint_jump_rate,
        ground_penetration_rate=ground_penetration_rate,
        joint_jump_threshold=float(config.metrics_joint_jump_threshold),
        ground_level=float(config.metrics_ground_level),
    )
