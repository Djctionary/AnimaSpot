#!/usr/bin/env python3
"""Visualize Spot retarget CSV motion in MuJoCo.

Usage:
    python3 visualize_spot_csv_mujoco.py \
        --model /path/to/spot_scene.xml \
        --csv /path/to/play_bow.csv \
        --fps 24
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


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

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "urdf/isaacsim_spot/spot_scene.xml"


def _normalize_joint_key(name: str) -> str:
    return name.strip().lower().replace(".", "_").replace("-", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Spot generalized-coordinate CSV in MuJoCo.")
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to Spot model file (.xml recommended, .urdf still supported).",
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to motion CSV (N x 19, no header).")
    parser.add_argument("--fps", type=float, default=24.0, help="Playback FPS (default: 24).")
    parser.add_argument("--repeat", action="store_true", help="Loop playback when reaching the last frame.")
    parser.add_argument("--start_frame", type=int, default=0, help="Initial frame index.")
    parser.add_argument(
        "--root-offset",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Add a constant XYZ offset to the root position from the CSV.",
    )
    return parser.parse_args()


def load_motion_csv(csv_path: Path) -> np.ndarray:
    motion = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if motion.ndim == 1:
        motion = motion[None, :]
    if motion.shape[1] != 19:
        raise ValueError(f"Expected 19 columns in CSV, got {motion.shape[1]} from {csv_path}")
    return motion


def build_joint_qpos_map(model: mujoco.MjModel) -> dict[str, int]:
    qpos_map: dict[str, int] = {}

    # Build normalized lookup from model joint names.
    model_joint_lookup: dict[str, tuple[str, int]] = {}
    for jid in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if jname is None:
            continue
        model_joint_lookup[_normalize_joint_key(jname)] = (jname, int(model.jnt_qposadr[jid]))

    missing = []
    for expected in SPOT_JOINT_NAMES:
        # Try several aliases for robustness across URDF exporters.
        aliases = [
            expected,
            expected.replace("_", "."),
            expected.replace("_", ""),
            expected.replace("_", "-"),
        ]
        found = None
        for alias in aliases:
            key = _normalize_joint_key(alias)
            if key in model_joint_lookup:
                found = model_joint_lookup[key]
                break
        if found is None:
            missing.append(expected)
            continue
        real_name, qpos_adr = found
        qpos_map[expected] = qpos_adr
        print(f"[info] joint map: csv '{expected}' -> model '{real_name}' (qpos[{qpos_adr}])")

    if missing:
        missing_text = ", ".join(missing)
        available = sorted(model_joint_lookup.keys())
        raise ValueError(
            f"Missing required joints in model: {missing_text}\n"
            f"Available normalized joint names: {available}"
        )
    return qpos_map


def _set_root_pose_if_freejoint(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    row: np.ndarray,
    root_offset: np.ndarray,
) -> None:
    free_joint_ids = [jid for jid in range(model.njnt) if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE]
    if not free_joint_ids:
        return

    # Use the first free joint as root.
    root_id = free_joint_ids[0]
    root_qpos_adr = int(model.jnt_qposadr[root_id])

    x, y, z = row[0:3] + root_offset
    qx, qy, qz, qw = row[3:7]
    # CSV is (qx, qy, qz, qw). MuJoCo qpos freejoint uses (qw, qx, qy, qz).
    data.qpos[root_qpos_adr : root_qpos_adr + 3] = np.array([x, y, z], dtype=np.float64)
    data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = np.array([qw, qx, qy, qz], dtype=np.float64)


def _prepare_urdf_with_visuals(model_path: Path) -> Path:
    """
    Prepare a temporary URDF that keeps visual meshes during MuJoCo import.

    MuJoCo URDF import can discard visuals by default, which leaves only collision
    shapes (e.g., a box body and foot spheres). This helper forces:
      <mujoco><compiler discardvisual="false"/></mujoco>
    """
    if model_path.suffix.lower() != ".urdf":
        return model_path

    root = ET.fromstring(model_path.read_text(encoding="utf-8"))
    mujoco_elem = root.find("mujoco")
    if mujoco_elem is None:
        mujoco_elem = ET.SubElement(root, "mujoco")
    compiler_elem = mujoco_elem.find("compiler")
    if compiler_elem is None:
        compiler_elem = ET.SubElement(mujoco_elem, "compiler")

    # Keep visual meshes and preserve path prefixes like "meshes/...".
    compiler_elem.set("discardvisual", "false")
    compiler_elem.set("strippath", "false")
    compiler_elem.set("meshdir", ".")
    text = ET.tostring(root, encoding="unicode")

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".urdf",
        prefix="spot_visual_",
        dir=str(model_path.parent),
        delete=False,
        encoding="utf-8",
    )
    with tmp:
        tmp.write(text)
    return Path(tmp.name)


def set_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    row: np.ndarray,
    joint_qpos_map: dict[str, int],
    root_offset: np.ndarray,
) -> None:
    _set_root_pose_if_freejoint(model, data, row, root_offset)
    joint_angles = row[7:19]
    for i, joint_name in enumerate(SPOT_JOINT_NAMES):
        qpos_adr = joint_qpos_map[joint_name]
        data.qpos[qpos_adr] = joint_angles[i]
    mujoco.mj_forward(model, data)


def load_model(model_path: Path) -> mujoco.MjModel:
    original_cwd = Path.cwd()
    os.chdir(str(model_path.parent))
    try:
        load_path = _prepare_urdf_with_visuals(model_path)
        return mujoco.MjModel.from_xml_path(str(load_path.name))
    finally:
        os.chdir(str(original_cwd))


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).resolve()
    csv_path = Path(args.csv).resolve()
    root_offset = np.asarray(args.root_offset, dtype=np.float64)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    motion = load_motion_csv(csv_path)
    n_frames = motion.shape[0]
    frame_idx = int(np.clip(args.start_frame, 0, n_frames - 1))

    model = load_model(model_path)
    data = mujoco.MjData(model)
    joint_qpos_map = build_joint_qpos_map(model)

    dt = 1.0 / max(args.fps, 1e-6)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            tick = time.perf_counter()
            set_frame(model, data, motion[frame_idx], joint_qpos_map, root_offset)
            viewer.sync()

            frame_idx += 1
            if frame_idx >= n_frames:
                if args.repeat:
                    frame_idx = 0
                else:
                    break

            elapsed = time.perf_counter() - tick
            if elapsed < dt:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    main()

