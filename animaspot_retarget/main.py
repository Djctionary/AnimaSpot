"""CLI entry point for AnimaSpot retargeting."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:
    from .config import RetargetConfig, SPOT_JOINT_NAMES
    from .export import to_csv, to_numpy
    from .retarget import retarget_sequence
    from .skeleton import load_sequence
    from .visualize import animate_sequence, plot_frame, plot_joint_trajectories
except ImportError:  # Support direct script execution: python main.py ...
    from animaspot_retarget.config import RetargetConfig, SPOT_JOINT_NAMES
    from animaspot_retarget.export import to_csv, to_numpy
    from animaspot_retarget.retarget import retarget_sequence
    from animaspot_retarget.skeleton import load_sequence
    from animaspot_retarget.visualize import animate_sequence, plot_frame, plot_joint_trajectories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retarget Animal3D pose sequence to Spot generalized coordinates.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing XXXX_3D.npz files.")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path.")
    parser.add_argument("--behavior", type=str, default="behavior", help="Behavior name (for logging only).")
    parser.add_argument("--ground_contact", action="store_true", help="Adjust body height so paws touch the ground.")
    parser.add_argument("--visualize", action="store_true", help="Show frame and trajectory plots.")
    parser.add_argument("--animate", action="store_true", help="Play full 3D animation over all frames.")
    parser.add_argument("--log_level", type=str, default="INFO", help="Python logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    cfg = RetargetConfig(ground_contact=args.ground_contact)
    result = retarget_sequence(args.input_dir, cfg)

    out_csv = Path(args.output)
    to_csv(result, out_csv)
    print(f"Wrote CSV: {out_csv}")
    print("Reminder for downstream preprocessing: use --input_fps 24")

    if args.visualize or args.animate:
        sequence = load_sequence(args.input_dir)
    else:
        sequence = None

    if args.visualize:
        mid = len(sequence) // 2
        plot_frame(sequence[mid], result["joint_angles"][mid], mid)
        plot_joint_trajectories(result["joint_angles"], SPOT_JOINT_NAMES)
    if args.animate:
        animate_sequence(sequence, result["joint_angles"], fps=cfg.fps, repeat=True)


if __name__ == "__main__":
    main()
