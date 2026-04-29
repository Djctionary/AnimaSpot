"""Save per-stage retargeting debug geometry for Viser inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from scipy.spatial.transform import Rotation

from .config import (
    BONE_I,
    BONE_J,
    BODY_HALF_LENGTH,
    BODY_HALF_WIDTH,
    HIP_ATTACHMENTS,
    HIP_X_OFFSET,
    JOINT_LIMITS,
    LEG_JOINT_IDXS,
    LEG_ORDER,
    LEG_SIDE,
    LEG_TARGET_JOINT_IDXS,
    LEG_TARGET_JOINT_NAME,
    L_LOWER,
    L_UPPER,
    RetargetConfig,
    SPOT_JOINT_NAMES,
)
from .ik_solver import forward_kinematics, leg_keypoints, solve_leg_ik
from .postprocess import apply_global_pose_postprocess
from .retarget import (
    _apply_joint_test_overrides,
    _normalize_quat,
    _smooth_quaternions,
    compute_leg_scale_factors,
    compute_paw_targets_body_frame,
    one_euro_filter,
)
from .skeleton import compute_body_frame, load_sequence, rotation_matrix_to_quat_xyzw


SPOT_STAGE_POINT_NAMES = np.array(
    [
        "fl_mount",
        "fr_mount",
        "hl_mount",
        "hr_mount",
        "fl_hy",
        "fl_knee",
        "fl_paw",
        "fr_hy",
        "fr_knee",
        "fr_paw",
        "hl_hy",
        "hl_knee",
        "hl_paw",
        "hr_hy",
        "hr_knee",
        "hr_paw",
    ],
    dtype="<U16",
)

SPOT_STAGE_EDGES = np.array(
    [
        [0, 1],
        [1, 3],
        [3, 2],
        [2, 0],
        [0, 4],
        [4, 5],
        [5, 6],
        [1, 7],
        [7, 8],
        [8, 9],
        [2, 10],
        [10, 11],
        [11, 12],
        [3, 13],
        [13, 14],
        [14, 15],
    ],
    dtype=np.int32,
)

_LEG_POINT_BASE = {"fl": 4, "fr": 7, "hl": 10, "hr": 13}


def default_debug_path(output_npz: str | Path) -> Path:
    """Infer the debug-stage NPZ path from the normal retarget NPZ path."""
    output_npz = Path(output_npz)
    stem = output_npz.stem
    if stem.endswith("_spot"):
        stem = stem[: -len("_spot")]
    return output_npz.with_name(f"{stem}_debug_stages.npz")


def _frame_indices(input_dir: Path) -> np.ndarray:
    files = sorted(input_dir.glob("*_3D.npz"), key=lambda p: int(p.stem.split("_")[0]))
    return np.array([int(path.stem.split("_")[0]) for path in files], dtype=np.int32)


def _scales_to_array(scales: Dict[str, float]) -> np.ndarray:
    return np.array([scales[leg] for leg in LEG_ORDER], dtype=np.float64)


def _body_point(pose: np.ndarray, joint_idx: int, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return R.T @ (pose[joint_idx] - t)


def _diagnostic_leg_skeleton(
    pose: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    scales: Dict[str, float] | None = None,
) -> np.ndarray:
    """Build a connected Spot-body skeleton from transformed source leg joints."""
    points = np.zeros((len(SPOT_STAGE_POINT_NAMES), 3), dtype=np.float64)
    for hip_idx, leg in enumerate(LEG_ORDER):
        points[hip_idx] = HIP_ATTACHMENTS[leg]
        scale = 1.0 if scales is None else scales[leg]
        base = _LEG_POINT_BASE[leg]
        hip = HIP_ATTACHMENTS[leg]
        idxs = LEG_JOINT_IDXS[leg]
        target_joint_name = LEG_TARGET_JOINT_NAME[leg]
        for offset, joint_name in enumerate(("thigh", "knee", target_joint_name)):
            body_pos = _body_point(pose, idxs[joint_name], R, t)
            points[base + offset] = hip + (body_pos - hip) * scale
    return points


def _spot_skeleton_from_angles(joint_angles_frame: np.ndarray) -> np.ndarray:
    """Build a connected Spot FK skeleton in body coordinates."""
    points = np.zeros((len(SPOT_STAGE_POINT_NAMES), 3), dtype=np.float64)
    for hip_idx, leg in enumerate(LEG_ORDER):
        hip = HIP_ATTACHMENTS[leg]
        points[hip_idx] = hip

        hx, hy, kn = joint_angles_frame[3 * hip_idx : 3 * hip_idx + 3]
        side_sign = 1.0 if LEG_SIDE[leg] == "left" else -1.0
        hx_raw = side_sign * hx
        c = np.cos(hx_raw)
        s = np.sin(hx_raw)
        abduction_joint = np.array(
            [0.0, side_sign * (HIP_X_OFFSET * c), HIP_X_OFFSET * s],
            dtype=np.float64,
        )
        knee_rel, paw_rel = leg_keypoints(hx, hy, kn, HIP_X_OFFSET, L_UPPER, L_LOWER, LEG_SIDE[leg])

        base = _LEG_POINT_BASE[leg]
        points[base] = hip + abduction_joint
        points[base + 1] = hip + knee_rel
        points[base + 2] = hip + paw_rel
    return points


def _spot_skeleton_sequence_from_angles(joint_angles: np.ndarray) -> np.ndarray:
    return np.stack([_spot_skeleton_from_angles(frame) for frame in joint_angles], axis=0)


def _world_skeleton_sequence(
    body_skeleton: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
) -> np.ndarray:
    rotations = Rotation.from_quat(root_quat).as_matrix()
    world = np.empty_like(body_skeleton)
    for frame_idx in range(body_skeleton.shape[0]):
        world[frame_idx] = (rotations[frame_idx] @ body_skeleton[frame_idx].T).T + root_pos[frame_idx]
    return world


def _paw_points_from_stage_skeleton(stage_skeleton: np.ndarray) -> np.ndarray:
    return stage_skeleton[:, [6, 9, 12, 15], :]


def _target_errors(joint_angles: np.ndarray, scaled_targets: np.ndarray) -> np.ndarray:
    errors = np.zeros((joint_angles.shape[0], len(LEG_ORDER)), dtype=np.float64)
    for frame_idx in range(joint_angles.shape[0]):
        for leg_idx, leg in enumerate(LEG_ORDER):
            hx, hy, kn = joint_angles[frame_idx, 3 * leg_idx : 3 * leg_idx + 3]
            pred = forward_kinematics(hx, hy, kn, HIP_X_OFFSET, L_UPPER, L_LOWER, LEG_SIDE[leg])
            errors[frame_idx, leg_idx] = np.linalg.norm(pred - scaled_targets[frame_idx, leg_idx])
    return errors


def compute_debug_stages(input_dir: str | Path, config: RetargetConfig) -> dict[str, np.ndarray]:
    """Compute stage geometry using the same helpers as the retargeting pipeline."""
    input_dir = Path(input_dir)
    sequence = load_sequence(input_dir)
    n_frames = sequence.shape[0]
    scales = compute_leg_scale_factors(sequence)
    freq = float(config.fps)

    body_origins = np.zeros((n_frames, 3), dtype=np.float64)
    body_axes = np.zeros((n_frames, 3, 3), dtype=np.float64)
    stage3_body_skeleton = np.zeros((n_frames, len(SPOT_STAGE_POINT_NAMES), 3), dtype=np.float64)
    stage4_scaled_skeleton = np.zeros_like(stage3_body_skeleton)
    stage4_scaled_targets = np.zeros((n_frames, len(LEG_ORDER), 3), dtype=np.float64)
    raw_joint_angles = np.zeros((n_frames, 12), dtype=np.float64)
    raw_root_quat = np.zeros((n_frames, 4), dtype=np.float64)
    root_pos_stage6 = np.tile(np.array(config.root_position, dtype=np.float64), (n_frames, 1))

    for frame_idx, pose in enumerate(sequence):
        R, t = compute_body_frame(pose)
        body_axes[frame_idx] = R
        body_origins[frame_idx] = t
        raw_root_quat[frame_idx] = rotation_matrix_to_quat_xyzw(R)
        stage3_body_skeleton[frame_idx] = _diagnostic_leg_skeleton(pose, R, t)
        stage4_scaled_skeleton[frame_idx] = _diagnostic_leg_skeleton(pose, R, t, scales=scales)

        paw_targets = compute_paw_targets_body_frame(pose, R, t)
        frame_angles = np.zeros((12,), dtype=np.float64)
        for leg_idx, leg in enumerate(LEG_ORDER):
            target = paw_targets[leg] * scales[leg]
            stage4_scaled_targets[frame_idx, leg_idx] = target
            frame_angles[3 * leg_idx : 3 * leg_idx + 3] = solve_leg_ik(
                target_pos=target,
                hip_offset=HIP_X_OFFSET,
                L_upper=L_UPPER,
                L_lower=L_LOWER,
                joint_limits=JOINT_LIMITS,
                side=LEG_SIDE[leg],
            )
        _apply_joint_test_overrides(frame_angles, config)
        raw_joint_angles[frame_idx] = frame_angles

    stage5_ik_skeleton = _spot_skeleton_sequence_from_angles(raw_joint_angles)

    smoothed_joint_angles = one_euro_filter(
        raw_joint_angles,
        freq,
        min_cutoff=config.one_euro_min_cutoff,
        beta=config.one_euro_beta,
        d_cutoff=config.one_euro_d_cutoff,
    )
    for joint_idx, key in enumerate(["hx", "hy", "kn"] * 4):
        lo, hi = JOINT_LIMITS[key]
        smoothed_joint_angles[:, joint_idx] = np.clip(smoothed_joint_angles[:, joint_idx], lo, hi)
    _apply_joint_test_overrides(smoothed_joint_angles, config)
    root_quat_stage6 = _smooth_quaternions(
        raw_root_quat,
        freq,
        min_cutoff=config.one_euro_min_cutoff,
        beta=config.one_euro_beta,
        d_cutoff=config.one_euro_d_cutoff,
    )
    stage6_smoothed_skeleton = _spot_skeleton_sequence_from_angles(smoothed_joint_angles)

    stage7_result = {
        "joint_angles": smoothed_joint_angles,
        "root_quat": _normalize_quat(root_quat_stage6),
        "root_pos": root_pos_stage6.copy(),
        "fps": np.array(config.fps, dtype=np.int32),
    }
    if config.postprocess_global_pose:
        stage7_result = apply_global_pose_postprocess(stage7_result, config)

    stage7_postprocessed_skeleton = _world_skeleton_sequence(
        stage6_smoothed_skeleton,
        stage7_result["root_pos"],
        stage7_result["root_quat"],
    )

    mesh_dir = input_dir.parent / "meshes"
    return {
        "schema_version": np.array(1, dtype=np.int32),
        "input_dir": np.array(str(input_dir)),
        "mesh_dir": np.array(str(mesh_dir) if mesh_dir.exists() else ""),
        "frame_indices": _frame_indices(input_dir),
        "fps": np.array(config.fps, dtype=np.int32),
        "postprocess_global_pose": np.array(bool(config.postprocess_global_pose)),
        "animal_bone_i": BONE_I.astype(np.int32),
        "animal_bone_j": BONE_J.astype(np.int32),
        "spot_stage_edges": SPOT_STAGE_EDGES,
        "spot_stage_point_names": SPOT_STAGE_POINT_NAMES,
        "leg_order": np.array(LEG_ORDER, dtype="<U2"),
        "leg_target_joint_names": np.array([LEG_TARGET_JOINT_NAME[leg] for leg in LEG_ORDER], dtype="<U8"),
        "leg_target_joint_indices": np.array([LEG_TARGET_JOINT_IDXS[leg] for leg in LEG_ORDER], dtype=np.int32),
        "spot_joint_names": np.array(SPOT_JOINT_NAMES, dtype="<U8"),
        "body_half_length": np.array(BODY_HALF_LENGTH, dtype=np.float64),
        "body_half_width": np.array(BODY_HALF_WIDTH, dtype=np.float64),
        "hip_x_offset": np.array(HIP_X_OFFSET, dtype=np.float64),
        "scale_factors": _scales_to_array(scales),
        "stage1_animal3d": sequence,
        "stage2_body_origins": body_origins,
        "stage2_body_axes": body_axes,
        "stage3_body_skeleton": stage3_body_skeleton,
        "stage4_scaled_skeleton": stage4_scaled_skeleton,
        "stage4_scaled_targets": stage4_scaled_targets,
        "stage5_ik_skeleton": stage5_ik_skeleton,
        "stage6_smoothed_skeleton": stage6_smoothed_skeleton,
        "stage7_postprocessed_skeleton": stage7_postprocessed_skeleton,
        "raw_joint_angles": raw_joint_angles,
        "smoothed_joint_angles": smoothed_joint_angles,
        "root_pos_stage6": root_pos_stage6,
        "root_quat_stage6": _normalize_quat(root_quat_stage6),
        "root_pos_stage7": stage7_result["root_pos"],
        "root_quat_stage7": stage7_result["root_quat"],
        "stage5_target_errors": _target_errors(raw_joint_angles, stage4_scaled_targets),
        "stage6_target_errors": _target_errors(smoothed_joint_angles, stage4_scaled_targets),
        "stage3_paw_points": _paw_points_from_stage_skeleton(stage3_body_skeleton),
        "stage4_paw_points": _paw_points_from_stage_skeleton(stage4_scaled_skeleton),
        "stage5_paw_points": _paw_points_from_stage_skeleton(stage5_ik_skeleton),
        "stage6_paw_points": _paw_points_from_stage_skeleton(stage6_smoothed_skeleton),
    }


def save_debug_stages(input_dir: str | Path, output_path: str | Path, config: RetargetConfig) -> Path:
    """Write the debug-stage NPZ and return its path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **compute_debug_stages(input_dir, config))
    return output_path
