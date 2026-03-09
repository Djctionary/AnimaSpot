"""Export retargeting results to CSV and NPZ."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def to_csv(retarget_result: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    root_pos = retarget_result["root_pos"]
    root_quat = retarget_result["root_quat"]
    joint_angles = retarget_result["joint_angles"]

    rows = np.hstack([root_pos, root_quat, joint_angles]).astype(np.float64)
    if rows.shape[1] != 19:
        raise ValueError(f"Expected 19 columns, got {rows.shape[1]}")
    np.savetxt(output_path, rows, delimiter=",", fmt="%.8f")


def to_numpy(retarget_result: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    np.savez(
        output_path,
        root_pos=retarget_result["root_pos"],
        root_quat=retarget_result["root_quat"],
        joint_angles=retarget_result["joint_angles"],
        fps=retarget_result["fps"],
    )

