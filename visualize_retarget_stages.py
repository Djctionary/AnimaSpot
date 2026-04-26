#!/usr/bin/env python3
"""Open a Viser viewer for a saved AnimaSpot retarget debug-stage NPZ."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize saved AnimaSpot retargeting stages with Viser.")
    parser.add_argument("--debug_npz", type=str, required=True, help="Path to *_debug_stages.npz.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--apply_cam_t",
        action="store_true",
        help="Apply AniMer mesh cam_t translations when source mesh data is available.",
    )
    parser.add_argument(
        "--rotate_x_deg",
        type=float,
        default=-90.0,
        help="Display-only X-axis rotation for Stage 1/2 in degrees. Does not modify saved data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from animaspot_retarget.stage_viewer import run_viewer

    run_viewer(
        debug_npz=args.debug_npz,
        host=args.host,
        port=args.port,
        apply_cam_t=args.apply_cam_t,
        rotate_x_deg=args.rotate_x_deg,
    )


if __name__ == "__main__":
    main()
