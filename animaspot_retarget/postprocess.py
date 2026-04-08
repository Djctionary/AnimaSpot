"""Independent post-processing for exported global pose."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from scipy.spatial.transform import Rotation

from .config import RetargetConfig
from .retarget import _compute_paw_body_positions, _enforce_quat_continuity, _normalize_quat, _rotation_between

LOGGER = logging.getLogger(__name__)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return vector / norm


def _compute_paw_world_positions(
    joint_angles_frame: np.ndarray,
    root_quat_frame: np.ndarray,
    root_pos_frame: np.ndarray,
) -> np.ndarray:
    paw_body = _compute_paw_body_positions(joint_angles_frame)
    rotation = Rotation.from_quat(root_quat_frame).as_matrix()
    return (rotation @ paw_body.T).T + root_pos_frame[None, :]


def _fit_support_normal(paw_world: np.ndarray) -> np.ndarray:
    centered = paw_world - paw_world.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return _normalize_vector(vh[-1])


def _orient_support_normal(normal: np.ndarray, paw_world: np.ndarray, root_pos: np.ndarray) -> np.ndarray:
    torso_direction = root_pos - paw_world.mean(axis=0)
    if np.linalg.norm(torso_direction) < 1e-8:
        torso_direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if float(np.dot(normal, torso_direction)) < 0.0:
        normal = -normal
    return _normalize_vector(normal)


def apply_global_pose_postprocess(retarget_result: Dict[str, np.ndarray], config: RetargetConfig) -> Dict[str, np.ndarray]:
    """Apply a per-frame rigid ground-contact alignment without changing joint angles."""
    joint_angles = np.asarray(retarget_result["joint_angles"], dtype=np.float64)
    root_quat = np.asarray(retarget_result["root_quat"], dtype=np.float64)
    root_pos = np.asarray(retarget_result["root_pos"], dtype=np.float64)

    if joint_angles.ndim != 2 or root_quat.ndim != 2 or root_pos.ndim != 2:
        raise ValueError("Expected 2D arrays for joint_angles, root_quat, and root_pos")
    if joint_angles.shape[0] == 0:
        return dict(retarget_result)

    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    root_rot_mats = Rotation.from_quat(root_quat).as_matrix()
    aligned_rot_mats = np.empty_like(root_rot_mats)
    aligned_root_pos = root_pos.copy()
    normals_before = []
    normals_after = []
    plane_height_offsets = []
    body_flip_count = 0

    local_forward_flip = Rotation.from_euler("x", np.pi).as_matrix()
    for frame_idx in range(joint_angles.shape[0]):
        paw_world = _compute_paw_world_positions(joint_angles[frame_idx], root_quat[frame_idx], aligned_root_pos[frame_idx])
        support_normal = _orient_support_normal(
            _fit_support_normal(paw_world),
            paw_world,
            aligned_root_pos[frame_idx],
        )
        normals_before.append(support_normal)

        rotation_align = _rotation_between(support_normal, up)
        frame_rot = rotation_align @ root_rot_mats[frame_idx]

        # Fallback: if the torso is still upside down, flip around the local
        # forward axis and re-level from that corrected frame.
        if float(np.dot(frame_rot[:, 2], up)) < 0.0:
            body_flip_count += 1
            frame_rot = frame_rot @ local_forward_flip
            paw_world_flipped = (frame_rot @ _compute_paw_body_positions(joint_angles[frame_idx]).T).T + aligned_root_pos[frame_idx]
            support_normal = _orient_support_normal(
                _fit_support_normal(paw_world_flipped),
                paw_world_flipped,
                aligned_root_pos[frame_idx],
            )
            rotation_align = _rotation_between(support_normal, up)
            frame_rot = rotation_align @ frame_rot

        paw_world_aligned = (frame_rot @ _compute_paw_body_positions(joint_angles[frame_idx]).T).T + aligned_root_pos[frame_idx]
        plane_height = float(np.mean(paw_world_aligned[:, 2]))
        delta_z = config.ground_clearance - plane_height
        aligned_root_pos[frame_idx, 2] += delta_z
        plane_height_offsets.append(delta_z)

        paw_world_grounded = (frame_rot @ _compute_paw_body_positions(joint_angles[frame_idx]).T).T + aligned_root_pos[frame_idx]
        normals_after.append(_orient_support_normal(_fit_support_normal(paw_world_grounded), paw_world_grounded, aligned_root_pos[frame_idx]))
        aligned_rot_mats[frame_idx] = frame_rot

    aligned_root_quat = Rotation.from_matrix(aligned_rot_mats).as_quat()
    aligned_root_quat = _normalize_quat(_enforce_quat_continuity(aligned_root_quat))
    final_first_frame_paw_world = _compute_paw_world_positions(
        joint_angles[0],
        aligned_root_quat[0],
        aligned_root_pos[0],
    )
    final_first_frame_min_z = float(np.min(final_first_frame_paw_world[:, 2]))
    final_first_frame_mean_z = float(np.mean(final_first_frame_paw_world[:, 2]))
    mean_normal_before = _normalize_vector(np.mean(np.stack(normals_before, axis=0), axis=0))
    mean_normal_after = _normalize_vector(np.mean(np.stack(normals_after, axis=0), axis=0))

    LOGGER.info(
        "Applied per-frame rigid ground contact: mean support normal %s -> %s",
        np.array2string(mean_normal_before, precision=4),
        np.array2string(mean_normal_after, precision=4),
    )
    LOGGER.info(
        "Applied %d body-up flip correction(s) across %d frame(s).",
        body_flip_count,
        joint_angles.shape[0],
    )
    LOGGER.info(
        "Applied per-frame root height offsets: mean=%.4f m, std=%.4f m. "
        "First-frame paw mean z=%.4f m, min paw z=%.4f m (target clearance=%.4f m)",
        float(np.mean(plane_height_offsets)),
        float(np.std(plane_height_offsets)),
        final_first_frame_mean_z,
        final_first_frame_min_z,
        config.ground_clearance,
    )

    result = dict(retarget_result)
    result["root_quat"] = aligned_root_quat
    result["root_pos"] = aligned_root_pos
    return result
