"""Unit tests for Spot leg IK/FK."""

from __future__ import annotations

import unittest

import numpy as np

from animaspot_retarget.config import HIP_X_OFFSET, JOINT_LIMITS, L_LOWER, L_UPPER
from animaspot_retarget.ik_solver import forward_kinematics, leg_keypoints, solve_leg_ik


class TestIK(unittest.TestCase):
    def _roundtrip(self, side: str, hx: float, hy: float, kn: float) -> None:
        target = forward_kinematics(hx, hy, kn, HIP_X_OFFSET, L_UPPER, L_LOWER, side)
        sx, sy, sk = solve_leg_ik(target, HIP_X_OFFSET, L_UPPER, L_LOWER, JOINT_LIMITS, side)
        recon = forward_kinematics(sx, sy, sk, HIP_X_OFFSET, L_UPPER, L_LOWER, side)
        self.assertLess(np.linalg.norm(target - recon), 1e-3)

    def test_nominal_standing_like_pose_left(self) -> None:
        self._roundtrip("left", hx=0.0, hy=0.7, kn=-1.4)

    def test_nominal_standing_like_pose_right(self) -> None:
        self._roundtrip("right", hx=0.0, hy=0.7, kn=-1.4)

    def test_limits_enforced(self) -> None:
        target = np.array([2.0, 2.0, -0.05], dtype=np.float64)
        hx, hy, kn = solve_leg_ik(target, HIP_X_OFFSET, L_UPPER, L_LOWER, JOINT_LIMITS, "left")
        self.assertGreaterEqual(hx, JOINT_LIMITS["hx"][0])
        self.assertLessEqual(hx, JOINT_LIMITS["hx"][1])
        self.assertGreaterEqual(hy, JOINT_LIMITS["hy"][0])
        self.assertLessEqual(hy, JOINT_LIMITS["hy"][1])
        self.assertGreaterEqual(kn, JOINT_LIMITS["kn"][0])
        self.assertLessEqual(kn, JOINT_LIMITS["kn"][1])

    def test_link_lengths_are_rigid(self) -> None:
        hx, hy, kn = 0.2, 0.9, -1.5
        knee, paw = leg_keypoints(hx, hy, kn, HIP_X_OFFSET, L_UPPER, L_LOWER, "right")
        s = -1.0  # right leg
        hx_raw = s * hx
        hy_joint = np.array(
            [0.0, s * (HIP_X_OFFSET * np.cos(hx_raw)), HIP_X_OFFSET * np.sin(hx_raw)],
            dtype=np.float64,
        )
        self.assertAlmostEqual(float(np.linalg.norm(knee - hy_joint)), L_UPPER, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(paw - knee)), L_LOWER, places=6)


if __name__ == "__main__":
    unittest.main()

