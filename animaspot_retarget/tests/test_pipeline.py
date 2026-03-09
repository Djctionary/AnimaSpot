"""Integration test for full retargeting pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from animaspot_retarget.config import JOINT_LIMITS, RetargetConfig
from animaspot_retarget.export import to_csv, to_numpy
from animaspot_retarget.retarget import retarget_sequence


def _synthetic_pose(frame_idx: int, n_frames: int) -> np.ndarray:
    pose = np.zeros((26, 3), dtype=np.float64)
    # Torso anchors for stable body frame.
    pose[7] = np.array([-0.2, 0.0, 0.0])   # tail_base
    pose[18] = np.array([0.2, 0.0, 0.0])   # neck
    pose[12] = np.array([0.15, 0.08, 0.0])  # left_shoulder
    pose[13] = np.array([0.15, -0.08, 0.0])  # right_shoulder

    # Front and hind limbs (simple play-bow style motion).
    t = frame_idx / max(1, (n_frames - 1))
    front_drop = -0.45 - 0.03 * np.sin(2.0 * np.pi * t)
    hind_drop = -0.35 - 0.02 * np.sin(2.0 * np.pi * t + 0.5)
    forward_shift = 0.05 * np.sin(2.0 * np.pi * t)

    pose[8] = np.array([0.16, 0.08, -0.10])
    pose[14] = np.array([0.18, 0.09, -0.25])
    pose[3] = np.array([0.23 + forward_shift, 0.10, front_drop])

    pose[9] = np.array([0.16, -0.08, -0.10])
    pose[15] = np.array([0.18, -0.09, -0.25])
    pose[4] = np.array([0.23 + forward_shift, -0.10, front_drop])

    pose[10] = np.array([-0.12, 0.08, -0.08])
    pose[16] = np.array([-0.14, 0.09, -0.20])
    pose[5] = np.array([-0.17, 0.11, hind_drop])

    pose[11] = np.array([-0.12, -0.08, -0.08])
    pose[17] = np.array([-0.14, -0.09, -0.20])
    pose[6] = np.array([-0.17, -0.11, hind_drop])

    # Minor head/tail points.
    pose[24] = np.array([0.25, 0.0, 0.02])
    pose[25] = np.array([-0.25, 0.0, 0.0])
    pose[19] = np.array([-0.32, 0.0, 0.0])
    return pose


class TestPipeline(unittest.TestCase):
    def test_end_to_end_export_and_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pose_dir = Path(tmp_dir) / "pose3D"
            pose_dir.mkdir(parents=True, exist_ok=True)

            n_frames = 12
            for i in range(n_frames):
                pose = _synthetic_pose(i, n_frames)
                np.savez(pose_dir / f"{i:04d}_3D.npz", pose3d=pose)

            cfg = RetargetConfig(smooth_window=7, smooth_polyorder=3)
            result = retarget_sequence(pose_dir, cfg)

            csv_path = Path(tmp_dir) / "play_bow.csv"
            npz_path = Path(tmp_dir) / "play_bow.npz"
            to_csv(result, csv_path)
            to_numpy(result, npz_path)

            csv_data = np.loadtxt(csv_path, delimiter=",")
            self.assertEqual(csv_data.shape, (n_frames, 19))

            quat = result["root_quat"]
            quat_norm = np.linalg.norm(quat, axis=1)
            self.assertTrue(np.all(np.abs(quat_norm - 1.0) < 1e-6))

            ja = result["joint_angles"]
            for leg_i in range(4):
                hx = ja[:, 3 * leg_i + 0]
                hy = ja[:, 3 * leg_i + 1]
                kn = ja[:, 3 * leg_i + 2]
                self.assertTrue(np.all((hx >= JOINT_LIMITS["hx"][0]) & (hx <= JOINT_LIMITS["hx"][1])))
                self.assertTrue(np.all((hy >= JOINT_LIMITS["hy"][0]) & (hy <= JOINT_LIMITS["hy"][1])))
                self.assertTrue(np.all((kn >= JOINT_LIMITS["kn"][0]) & (kn <= JOINT_LIMITS["kn"][1])))

            # Smoothness check: no large frame-to-frame jumps.
            jumps = np.abs(np.diff(ja, axis=0))
            self.assertLess(float(np.max(jumps)), 0.3 + 1e-6)


if __name__ == "__main__":
    unittest.main()

