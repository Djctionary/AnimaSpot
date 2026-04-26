"""CLI entry point for AnimaSpot retargeting."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:
    from .config import RetargetConfig
    from .debug_stages import default_debug_path, save_debug_stages
    from .export import to_csv, to_numpy
    from .postprocess import apply_global_pose_postprocess
    from .retarget import retarget_sequence
    from .skeleton import load_sequence
    from .visualize import animate_sequence
except ImportError:  # Support direct script execution: python main.py ...
    from animaspot_retarget.config import RetargetConfig
    from animaspot_retarget.debug_stages import default_debug_path, save_debug_stages
    from animaspot_retarget.export import to_csv, to_numpy
    from animaspot_retarget.postprocess import apply_global_pose_postprocess
    from animaspot_retarget.retarget import retarget_sequence
    from animaspot_retarget.skeleton import load_sequence
    from animaspot_retarget.visualize import animate_sequence


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retarget Animal3D pose sequence to Spot generalized coordinates.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing XXXX_3D.npz files, usually a pipeline_data/intermediate/.../pose3D folder.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output CSV path. Defaults to pipeline_data/final/<source>/<behavior>/<behavior>_spot.csv.",
    )
    parser.add_argument(
        "--output_npz",
        type=str,
        default="",
        help="Optional output NPZ path. Defaults to pipeline_data/final/<source>/<behavior>/<behavior>_spot.npz.",
    )
    parser.add_argument(
        "--behavior",
        type=str,
        default="",
        help="Optional behavior name for logging. Defaults to the inferred input folder name.",
    )
    parser.add_argument(
        "--postprocess_global_pose",
        dest="postprocess_global_pose",
        action="store_true",
        default=True,
        help="Apply per-frame rigid ground-contact correction before export (default: enabled).",
    )
    parser.add_argument(
        "--no_postprocess_global_pose",
        dest="postprocess_global_pose",
        action="store_false",
        help="Disable the sequence-wide global pose correction stage.",
    )
    parser.add_argument(
        "--align_window",
        type=int,
        default=5,
        help="Deprecated compatibility option. The current ground-contact postprocess runs per frame.",
    )
    parser.add_argument(
        "--ground_clearance",
        type=float,
        default=0.035,
        help="Target ground height for the per-frame support plane after global pose postprocessing.",
    )
    parser.add_argument(
        "--ground_contact",
        action="store_true",
        help="Deprecated alias kept for compatibility. Global pose postprocessing is now the preferred behavior.",
    )
    parser.add_argument("--visualize", action="store_true", help="Save debug stages and open the Viser stage viewer.")
    parser.add_argument("--visualize_host", type=str, default="127.0.0.1", help="Host for the Viser stage viewer.")
    parser.add_argument("--visualize_port", type=int, default=8080, help="Port for the Viser stage viewer.")
    parser.add_argument(
        "--visualize_apply_cam_t",
        action="store_true",
        help="Apply AniMer mesh cam_t translations in the Viser stage viewer.",
    )
    parser.add_argument(
        "--visualize_rotate_x_deg",
        type=float,
        default=-90.0,
        help="Display-only X-axis rotation for Stage 1/2 in the Viser viewer. Does not modify saved data.",
    )
    parser.add_argument("--animate", action="store_true", help="Play full 3D animation over all frames.")
    parser.add_argument("--log_level", type=str, default="INFO", help="Python logging level.")
    return parser.parse_args()


def infer_behavior_name(input_dir: Path) -> str:
    """Infer the behavior name from a pose3D directory or a direct frame directory."""
    return input_dir.parent.name if input_dir.name == "pose3D" else input_dir.name


def infer_source_name(input_dir: Path) -> str:
    """Infer the upstream recovery pipeline from pipeline_data/intermediate/<source>/..."""
    parts = input_dir.resolve().parts
    try:
        pipeline_idx = parts.index("pipeline_data")
    except ValueError:
        return "retarget"

    if len(parts) > pipeline_idx + 2 and parts[pipeline_idx + 1] == "intermediate":
        return parts[pipeline_idx + 2]
    return "retarget"


def resolve_output_paths(input_dir: Path, output: str, output_npz: str) -> tuple[Path, Path]:
    behavior_name = infer_behavior_name(input_dir)
    source_name = infer_source_name(input_dir)
    default_dir = REPO_ROOT / "pipeline_data" / "final" / source_name / behavior_name
    out_csv = Path(output) if output else default_dir / f"{behavior_name}_spot.csv"
    out_npz = Path(output_npz) if output_npz else default_dir / f"{behavior_name}_spot.npz"
    return out_csv, out_npz


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    input_dir = Path(args.input_dir)
    behavior_name = args.behavior or infer_behavior_name(input_dir)
    out_csv, out_npz = resolve_output_paths(input_dir, args.output, args.output_npz)

    logging.info("Retargeting behavior '%s' from %s", behavior_name, input_dir)
    cfg = RetargetConfig(
        ground_contact=args.ground_contact,
        ground_clearance=args.ground_clearance,
        postprocess_global_pose=args.postprocess_global_pose,
        postprocess_align_window=args.align_window,
    )
    if args.ground_contact:
        logging.warning(
            "--ground_contact is deprecated. Global pose postprocessing is now a separate stage "
            "and is enabled by default."
        )
    result = retarget_sequence(input_dir, cfg)
    if cfg.postprocess_global_pose:
        result = apply_global_pose_postprocess(result, cfg)

    to_csv(result, out_csv)
    print(f"Wrote CSV: {out_csv}")
    to_numpy(result, out_npz)
    print(f"Wrote NPZ: {out_npz}")
    print("Reminder for downstream preprocessing: use --input_fps 24")

    if args.visualize:
        debug_path = default_debug_path(out_npz)
        save_debug_stages(input_dir, debug_path, cfg)
        print(f"Wrote debug stages: {debug_path}")
        try:
            from .stage_viewer import run_viewer
        except ImportError:  # Support direct script execution: python main.py ...
            from animaspot_retarget.stage_viewer import run_viewer

        run_viewer(
            debug_npz=debug_path,
            host=args.visualize_host,
            port=args.visualize_port,
            apply_cam_t=args.visualize_apply_cam_t,
            rotate_x_deg=args.visualize_rotate_x_deg,
        )

    if args.animate:
        sequence = load_sequence(input_dir)
        animate_sequence(sequence, result["joint_angles"], fps=cfg.fps, repeat=True)


if __name__ == "__main__":
    main()
