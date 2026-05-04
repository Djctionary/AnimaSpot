"""Configuration constants for the AnimaSpot retargeting pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
from xml.etree import ElementTree as ET

import numpy as np


# Animal3D skeleton definition
ANIMAL3D_NUM_JOINTS = 26
ANIMAL3D_ROOT_IDX = 18

ANIMAL3D_JOINTS: Dict[str, int] = {
    "left_eye": 0,
    "right_eye": 1,
    "mouth_mid": 2,
    "left_front_paw": 3,
    "right_front_paw": 4,
    "left_back_paw": 5,
    "right_back_paw": 6,
    "tail_base": 7,
    "left_front_thigh": 8,
    "right_front_thigh": 9,
    "left_back_thigh": 10,
    "right_back_thigh": 11,
    "left_shoulder": 12,
    "right_shoulder": 13,
    "left_front_knee": 14,
    "right_front_knee": 15,
    "left_back_knee": 16,
    "right_back_knee": 17,
    "neck": 18,
    "tail_end": 19,
    "left_ear": 20,
    "right_ear": 21,
    "left_mouth": 22,
    "right_mouth": 23,
    "nose": 24,
    "tail_mid": 25,
}

BONE_I = np.array(
    [24, 24, 1, 0, 24, 2, 2, 24, 18, 18, 12, 13, 8, 9, 14, 15, 18, 7, 7, 10, 11, 16, 17, 7, 25]
)
BONE_J = np.array(
    [0, 1, 21, 20, 2, 22, 23, 18, 12, 13, 8, 9, 14, 15, 3, 4, 7, 10, 11, 16, 17, 5, 6, 25, 19]
)


def _load_spot_urdf_geometry() -> dict[str, float]:
    """Load key Spot dimensions from the checked-in URDF when available."""
    fallback = {
        "body_half_length": 0.29785,
        "body_half_width": 0.055,
        "hip_x_offset": 0.110945,
    }

    urdf_path = Path(__file__).resolve().parent.parent / "urdf" / "isaacsim_spot" / "spot.urdf"
    if not urdf_path.exists():
        return fallback

    try:
        root = ET.parse(urdf_path).getroot()
    except ET.ParseError:
        return fallback

    joints: dict[str, np.ndarray] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        origin = joint.find("origin")
        if name is None or origin is None:
            continue
        xyz_text = origin.get("xyz")
        if xyz_text is None:
            continue
        try:
            joints[name] = np.fromstring(xyz_text, sep=" ", dtype=np.float64)
        except ValueError:
            continue

    fl_hx = joints.get("fl_hx")
    hl_hx = joints.get("hl_hx")
    fl_hy = joints.get("fl_hy")
    if fl_hx is None or hl_hx is None or fl_hy is None:
        return fallback

    return {
        "body_half_length": 0.5 * abs(float(fl_hx[0] - hl_hx[0])),
        "body_half_width": abs(float(fl_hx[1])),
        "hip_x_offset": abs(float(fl_hy[1])),
    }


_SPOT_URDF_GEOMETRY = _load_spot_urdf_geometry()


# Spot kinematics (meters / radians)
L_UPPER = 0.3405
L_LOWER = 0.3405
HIP_X_OFFSET = _SPOT_URDF_GEOMETRY["hip_x_offset"]
BODY_HALF_LENGTH = _SPOT_URDF_GEOMETRY["body_half_length"]
BODY_HALF_WIDTH = _SPOT_URDF_GEOMETRY["body_half_width"]
BODY_LENGTH_OFFSET = 0.08

JOINT_LIMITS = {
    "hx": (-0.7854, 0.7854),
    "hy": (-0.8988, 2.2951),
    "kn": (-2.7929, -0.2577),
}

SPOT_JOINT_NAMES = [
    "fl_hx",
    "fl_hy",
    "fl_kn",
    "fr_hx",
    "fr_hy",
    "fr_kn",
    "hl_hx",
    "hl_hy",
    "hl_kn",
    "hr_hx",
    "hr_hy",
    "hr_kn",
]

LEG_ORDER = ("fl", "fr", "hl", "hr")
LEG_SIDE = {
    "fl": "left",
    "fr": "right",
    "hl": "left",
    "hr": "right",
}

# HIP_ATTACHMENTS = {
#     "fl": np.array([BODY_HALF_LENGTH, BODY_HALF_WIDTH, 0.0], dtype=np.float64),
#     "fr": np.array([BODY_HALF_LENGTH, -BODY_HALF_WIDTH, 0.0], dtype=np.float64),
#     "hl": np.array([-BODY_HALF_LENGTH, BODY_HALF_WIDTH, 0.0], dtype=np.float64),
#     "hr": np.array([-BODY_HALF_LENGTH, -BODY_HALF_WIDTH, 0.0], dtype=np.float64),
# }

HIP_ATTACHMENTS = {
    "fl": np.array([-BODY_LENGTH_OFFSET, BODY_HALF_WIDTH, 0.0], dtype=np.float64),
    "fr": np.array([-BODY_LENGTH_OFFSET, -BODY_HALF_WIDTH, 0.0], dtype=np.float64),
    "hl": np.array([-BODY_LENGTH_OFFSET - 2 * BODY_HALF_LENGTH, BODY_HALF_WIDTH, 0.0], dtype=np.float64),
    "hr": np.array([-BODY_LENGTH_OFFSET - 2 * BODY_HALF_LENGTH, -BODY_HALF_WIDTH, 0.0], dtype=np.float64),
}


LEG_JOINT_IDXS = {
    "fl": {"shoulder": 12, "thigh": 8, "knee": 14, "paw": 3},
    "fr": {"shoulder": 13, "thigh": 9, "knee": 15, "paw": 4},
    "hl": {"shoulder": 7, "thigh": 10, "knee": 16, "paw": 5},
    "hr": {"shoulder": 7, "thigh": 11, "knee": 17, "paw": 6},
}

# End-effector joint used for retarget target mapping and IK.
LEG_TARGET_JOINT_NAME = {
    "fl": "paw",
    "fr": "paw",
    "hl": "paw",
    "hr": "paw",
}

LEG_TARGET_JOINT_IDXS = {
    leg: LEG_JOINT_IDXS[leg][joint_name]
    for leg, joint_name in LEG_TARGET_JOINT_NAME.items()
}


@dataclass
class RetargetConfig:
    """Runtime pipeline parameters."""

    fps: int = 24
    root_position: Tuple[float, float, float] = (0.0, 0.0, 0.55)
    root_quaternion: Tuple[float, float, float, float] | None = (0.0, 0.0, 0.0, 1.0)
    # 1-Euro filter (Casiez et al. 2012) — adaptive low-pass.
    one_euro_min_cutoff: float = 1.7
    one_euro_beta: float = 0.01
    one_euro_d_cutoff: float = 1.0
    # Legacy flag kept for backward compatibility. Global pose correction now
    # happens in an independent postprocess stage after retargeting.
    ground_contact: bool = False
    ground_clearance: float = 0.035
    postprocess_global_pose: bool = True
    postprocess_align_window: int = 5
    fix_hx_zero: bool = False
    # TrajectoryIK objective weights. Smoothness is one conceptual term with
    # velocity and acceleration subcomponents.
    trajectory_w_track: float = 1.0
    trajectory_w_smooth: float = 0.05
    trajectory_smooth_velocity_weight: float = 0.2
    trajectory_smooth_acceleration_weight: float = 1.0
    trajectory_w_ground: float = 5.0
    trajectory_w_stable: float = 0.02
    trajectory_ground_level: float = 0.0
    trajectory_maxiter: int = 80
    trajectory_ftol: float = 1e-6
    trajectory_stable_joint_indices: Tuple[int, ...] = (0, 3, 6, 9)
    metrics_joint_jump_threshold: float = 0.5
    metrics_ground_level: float = 0.0
