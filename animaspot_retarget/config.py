"""Configuration constants for the AnimaSpot retargeting pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

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


# Spot kinematics (meters / radians)
L_UPPER = 0.3405
L_LOWER = 0.3405
HIP_X_OFFSET = 0.0547
BODY_HALF_LENGTH = 0.1945
BODY_HALF_WIDTH = 0.055

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

HIP_ATTACHMENTS = {
    "fl": np.array([BODY_HALF_LENGTH, BODY_HALF_WIDTH, 0.0], dtype=np.float64),
    "fr": np.array([BODY_HALF_LENGTH, -BODY_HALF_WIDTH, 0.0], dtype=np.float64),
    "hl": np.array([-BODY_HALF_LENGTH, BODY_HALF_WIDTH, 0.0], dtype=np.float64),
    "hr": np.array([-BODY_HALF_LENGTH, -BODY_HALF_WIDTH, 0.0], dtype=np.float64),
}

LEG_JOINT_IDXS = {
    "fl": {"shoulder": 12, "thigh": 8, "knee": 14, "paw": 3},
    "fr": {"shoulder": 13, "thigh": 9, "knee": 15, "paw": 4},
    "hl": {"shoulder": 7, "thigh": 10, "knee": 16, "paw": 5},
    "hr": {"shoulder": 7, "thigh": 11, "knee": 17, "paw": 6},
}


@dataclass
class RetargetConfig:
    """Runtime pipeline parameters."""

    fps: int = 24
    root_position: Tuple[float, float, float] = (0.0, 0.0, 0.5)
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
