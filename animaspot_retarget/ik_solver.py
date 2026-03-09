"""Analytical IK / FK for Spot's 3-DOF leg."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _side_sign(side: str) -> float:
    if side not in {"left", "right"}:
        raise ValueError(f"Invalid side={side}, expected 'left' or 'right'")
    return 1.0 if side == "left" else -1.0


def _rot_x(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def solve_leg_ik(
    target_pos: np.ndarray,
    hip_offset: float,
    L_upper: float,
    L_lower: float,
    joint_limits: Dict[str, Tuple[float, float]],
    side: str,
) -> Tuple[float, float, float]:
    """
    Solve 3-DOF leg IK for target position relative to hip attachment.

    Returns (hx, hy, kn), all in radians.
    """
    px, py, pz = np.asarray(target_pos, dtype=np.float64)
    s = _side_sign(side)

    # Convert to outward-lateral frame so left/right share one geometry.
    y_out = s * py

    # 1) Solve abduction from yz geometry with fixed lateral offset.
    yz_norm = np.hypot(y_out, pz)
    yz_norm = max(yz_norm, hip_offset + 1e-6)
    z_sag_nom = -np.sqrt(max(yz_norm * yz_norm - hip_offset * hip_offset, 0.0))

    a = np.array([hip_offset, z_sag_nom], dtype=np.float64)
    b = np.array([y_out, pz], dtype=np.float64)
    dot = float(np.dot(a, b))
    cross = float(a[0] * b[1] - a[1] * b[0])
    hx_raw = np.arctan2(cross, dot)

    # Remove abduction to recover sagittal target.
    c = np.cos(hx_raw)
    ss = np.sin(hx_raw)
    x_sag = px
    z_sag = -y_out * ss + pz * c

    # 2) Sagittal 2-link IK (Spot knee convention is negative bend).
    r2 = x_sag * x_sag + z_sag * z_sag
    cos_kn = (r2 - L_upper * L_upper - L_lower * L_lower) / (2.0 * L_upper * L_lower)
    cos_kn = _clamp(cos_kn, -1.0, 1.0)
    kn = -np.arccos(cos_kn)

    # 3) Hip flexion/extension.
    hy = np.arctan2(x_sag, -z_sag) - np.arctan2(L_lower * np.sin(kn), L_upper + L_lower * np.cos(kn))
    hx = s * hx_raw

    # 4) Joint-limit clamp.
    hx = _clamp(hx, *joint_limits["hx"])
    hy = _clamp(hy, *joint_limits["hy"])
    kn = _clamp(kn, *joint_limits["kn"])
    return float(hx), float(hy), float(kn)


def forward_kinematics(
    hx: float,
    hy: float,
    kn: float,
    hip_offset: float,
    L_upper: float,
    L_lower: float,
    side: str,
) -> np.ndarray:
    """Compute paw position from (hx, hy, kn), relative to hip attachment."""
    s = _side_sign(side)
    hx_raw = s * hx

    p_sag = np.array(
        [
            L_upper * np.sin(hy) + L_lower * np.sin(hy + kn),
            hip_offset,
            -(L_upper * np.cos(hy) + L_lower * np.cos(hy + kn)),
        ],
        dtype=np.float64,
    )
    p_out = _rot_x(hx_raw) @ p_sag
    y_out = p_out[1]
    pz = p_out[2]
    py = s * y_out
    return np.array([p_out[0], py, pz], dtype=np.float64)


def leg_keypoints(
    hx: float,
    hy: float,
    kn: float,
    hip_offset: float,
    L_upper: float,
    L_lower: float,
    side: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return knee and paw positions relative to hip attachment."""
    s = _side_sign(side)
    hx_raw = s * hx

    knee_sag = np.array([L_upper * np.sin(hy), hip_offset, -L_upper * np.cos(hy)], dtype=np.float64)
    knee_out = _rot_x(hx_raw) @ knee_sag
    knee = np.array([knee_out[0], s * knee_out[1], knee_out[2]], dtype=np.float64)

    paw = forward_kinematics(hx, hy, kn, hip_offset, L_upper, L_lower, side)
    return knee, paw

