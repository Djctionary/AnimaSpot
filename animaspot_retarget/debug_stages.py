"""Saved stage artifacts for AnimaSpot retargeting visualization.

This module keeps the historical debug-stage API as a compatibility layer, but
the artifact data now comes from a completed retarget run instead of recomputing
the IK pipeline.
"""

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
from .ik_solver import forward_kinematics, leg_keypoints
from .retarget import (
    METHOD_ANALYTICAL_IK,
    METHOD_TRAJECTORY_IK,
    RetargetRun,
    run_retarget_pipeline,
)


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


def default_stage_artifact_path(output_npz: str | Path) -> Path:
    """Infer the saved stage artifact NPZ path from the normal retarget NPZ path."""
    output_npz = Path(output_npz)
    stem = output_npz.stem
    if stem.endswith("_spot"):
        stem = stem[: -len("_spot")]
    return output_npz.with_name(f"{stem}_stages.npz")


def default_debug_path(output_npz: str | Path) -> Path:
    """Backward-compatible alias for older callers."""
    return default_stage_artifact_path(output_npz)


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
        for offset, joint_name in enumerate(("thigh", "knee", LEG_TARGET_JOINT_NAME[leg])):
            body_pos = _body_point(pose, idxs[joint_name], R, t)
            points[base + offset] = hip + (body_pos - hip) * scale
    return points


def _diagnostic_leg_skeleton_sequence(run: RetargetRun, scaled: bool) -> np.ndarray:
    context = run.context
    scales = context.scales if scaled else None
    return np.stack(
        [
            _diagnostic_leg_skeleton(
                pose,
                context.body_axes[frame_idx],
                context.body_origins[frame_idx],
                scales=scales,
            )
            for frame_idx, pose in enumerate(context.sequence)
        ],
        axis=0,
    )


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


def build_stage_artifacts(run: RetargetRun, config: RetargetConfig) -> dict[str, np.ndarray]:
    """Build a saved visualization artifact from a completed retarget run."""
    context = run.context
    stage3_body_skeleton = _diagnostic_leg_skeleton_sequence(run, scaled=False)
    stage4_scaled_skeleton = _diagnostic_leg_skeleton_sequence(run, scaled=True)
    stage5_method_skeleton = _spot_skeleton_sequence_from_angles(run.raw_joint_angles)
    stage6_smoothed_skeleton = _spot_skeleton_sequence_from_angles(run.smoothed_joint_angles)
    stage7_postprocessed_skeleton = _world_skeleton_sequence(
        stage6_smoothed_skeleton,
        run.result["root_pos"],
        run.result["root_quat"],
    )

    mesh_dir = context.input_dir.parent / "meshes"
    if run.method_name == METHOD_TRAJECTORY_IK:
        stage_names = np.array(
            [
                "RecoveredPose",
                "BodyTransformed",
                "LegScaled",
                "Retargeted_TrajectoryIK",
            ],
            dtype="<U32",
        )
    else:
        stage_names = np.array(
            [
                "RecoveredPose",
                "BodyTransformed",
                "LegScaled",
                "Retargeted_AnalyticalIK",
                "Smoothed_AnalyticalIK",
                "Ground_AnalyticalIK",
            ],
            dtype="<U32",
        )

    return {
        "schema_version": np.array(2, dtype=np.int32),
        "method_name": np.array(run.method_name),
        "stage_names": stage_names,
        "input_dir": np.array(str(context.input_dir)),
        "mesh_dir": np.array(str(mesh_dir) if mesh_dir.exists() else ""),
        "frame_indices": _frame_indices(context.input_dir),
        "fps": np.array(config.fps, dtype=np.int32),
        "postprocess_global_pose": np.array(bool(config.postprocess_global_pose and run.method_name == METHOD_ANALYTICAL_IK)),
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
        "scale_factors": _scales_to_array(context.scales),
        "stage1_animal3d": context.sequence,
        "stage2_body_origins": context.body_origins,
        "stage2_body_axes": context.body_axes,
        "stage3_body_skeleton": stage3_body_skeleton,
        "stage4_scaled_skeleton": stage4_scaled_skeleton,
        "stage4_scaled_targets": context.scaled_targets,
        "stage5_ik_skeleton": stage5_method_skeleton,
        "stage6_smoothed_skeleton": stage6_smoothed_skeleton,
        "stage7_postprocessed_skeleton": stage7_postprocessed_skeleton,
        "raw_joint_angles": run.raw_joint_angles,
        "smoothed_joint_angles": run.smoothed_joint_angles,
        "root_pos_stage6": run.root_pos_stage6,
        "root_quat_stage6": run.root_quat_stage6,
        "root_pos_stage7": run.result["root_pos"],
        "root_quat_stage7": run.result["root_quat"],
        "stage5_target_errors": _target_errors(run.raw_joint_angles, context.scaled_targets),
        "stage6_target_errors": _target_errors(run.smoothed_joint_angles, context.scaled_targets),
        "stage3_paw_points": _paw_points_from_stage_skeleton(stage3_body_skeleton),
        "stage4_paw_points": _paw_points_from_stage_skeleton(stage4_scaled_skeleton),
        "stage5_paw_points": _paw_points_from_stage_skeleton(stage5_method_skeleton),
        "stage6_paw_points": _paw_points_from_stage_skeleton(stage6_smoothed_skeleton),
    }


def save_stage_artifacts(run: RetargetRun, output_path: str | Path, config: RetargetConfig) -> Path:
    """Write the stage artifact NPZ and return its path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **build_stage_artifacts(run, config))
    return output_path


def compute_debug_stages(
    input_dir: str | Path,
    config: RetargetConfig,
    method: str = METHOD_ANALYTICAL_IK,
) -> dict[str, np.ndarray]:
    """Compatibility wrapper that computes a run before building artifacts."""
    return build_stage_artifacts(run_retarget_pipeline(input_dir, config, method=method), config)


def save_debug_stages(
    input_dir: str | Path,
    output_path: str | Path,
    config: RetargetConfig,
    method: str = METHOD_ANALYTICAL_IK,
) -> Path:
    """Compatibility wrapper for older callers."""
    run = run_retarget_pipeline(input_dir, config, method=method)
    return save_stage_artifacts(run, output_path, config)
