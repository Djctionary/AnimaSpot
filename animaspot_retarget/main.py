"""CLI entry point for AnimaSpot retargeting."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:
    from .config import SPOT_JOINT_NAMES, RetargetConfig
    from .debug_stages import default_stage_artifact_path, save_stage_artifacts
    from .export import to_csv, to_numpy
    from .metrics import compute_retarget_metrics
    from .retarget import METHOD_ANALYTICAL_IK, RETARGET_METHODS, run_retarget_pipeline
    from .skeleton import load_sequence
    from .visualize import animate_sequence
except ImportError:  # Support direct script execution: python main.py ...
    from animaspot_retarget.config import SPOT_JOINT_NAMES, RetargetConfig
    from animaspot_retarget.debug_stages import default_stage_artifact_path, save_stage_artifacts
    from animaspot_retarget.export import to_csv, to_numpy
    from animaspot_retarget.metrics import compute_retarget_metrics
    from animaspot_retarget.retarget import METHOD_ANALYTICAL_IK, RETARGET_METHODS, run_retarget_pipeline
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
        help=(
            "Optional output CSV path. Defaults to "
            "pipeline_data/final/<source>/<behavior>/<method>/<behavior>_spot.csv."
        ),
    )
    parser.add_argument(
        "--output_npz",
        type=str,
        default="",
        help=(
            "Optional output NPZ path. Defaults to "
            "pipeline_data/final/<source>/<behavior>/<method>/<behavior>_spot.npz."
        ),
    )
    parser.add_argument(
        "--behavior",
        type=str,
        default="",
        help="Optional behavior name for logging. Defaults to the inferred input folder name.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=METHOD_ANALYTICAL_IK,
        choices=RETARGET_METHODS,
        help="Retarget method to run.",
    )
    parser.add_argument("--trajectory_w_track", type=float, default=1.0, help="TrajectoryIK tracking weight.")
    parser.add_argument("--trajectory_w_smooth", type=float, default=0.05, help="TrajectoryIK smoothness weight.")
    parser.add_argument("--trajectory_w_ground", type=float, default=5.0, help="TrajectoryIK ground penetration weight.")
    parser.add_argument("--trajectory_w_stable", type=float, default=0.02, help="TrajectoryIK zero-position joint stability weight.")
    parser.add_argument("--trajectory_maxiter", type=int, default=80, help="TrajectoryIK L-BFGS-B max iterations.")
    parser.add_argument("--trajectory_maxfun", type=int, default=50000, help="TrajectoryIK L-BFGS-B max function evaluations.")
    parser.add_argument(
        "--trajectory_stable_joints",
        type=str,
        default="hx",
        help=(
            "Comma-separated joint names or indices stabilized by TrajectoryIK. "
            "Use hx/hy/kn groups or an empty string to disable."
        ),
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
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open the Viser stage viewer after the default stage artifacts are saved.",
    )
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
        help="Display-only X-axis rotation for recovered/body-transform stages. Does not modify saved data.",
    )
    parser.add_argument("--animate", action="store_true", help="Play full 3D animation over all frames.")
    parser.add_argument(
        "--fix_hx_zero",
        action="store_true",
        help="Force all hx joints to remain 0.0 radians for quick testing.",
    )
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


def resolve_output_paths(input_dir: Path, output: str, output_npz: str, method: str) -> tuple[Path, Path]:
    behavior_name = infer_behavior_name(input_dir)
    source_name = infer_source_name(input_dir)
    default_dir = REPO_ROOT / "pipeline_data" / "final" / source_name / behavior_name / method
    out_csv = Path(output) if output else default_dir / f"{behavior_name}_spot.csv"
    out_npz = Path(output_npz) if output_npz else default_dir / f"{behavior_name}_spot.npz"
    return out_csv, out_npz


def parse_stable_joint_indices(text: str) -> tuple[int, ...]:
    """Parse comma-separated Spot joint names, indices, or hx/hy/kn groups."""
    text = text.strip()
    if not text:
        return ()

    groups = {
        "hx": (0, 3, 6, 9),
        "hy": (1, 4, 7, 10),
        "kn": (2, 5, 8, 11),
    }
    indices: list[int] = []
    for item in (part.strip() for part in text.split(",")):
        if not item:
            continue
        if item in groups:
            indices.extend(groups[item])
        elif item in SPOT_JOINT_NAMES:
            indices.append(SPOT_JOINT_NAMES.index(item))
        else:
            try:
                idx = int(item)
            except ValueError as exc:
                raise ValueError(f"Unknown stable joint '{item}'") from exc
            if idx < 0 or idx >= len(SPOT_JOINT_NAMES):
                raise ValueError(f"Stable joint index {idx} is out of range 0-{len(SPOT_JOINT_NAMES) - 1}")
            indices.append(idx)
    return tuple(dict.fromkeys(indices))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    input_dir = Path(args.input_dir)
    behavior_name = args.behavior or infer_behavior_name(input_dir)
    out_csv, out_npz = resolve_output_paths(input_dir, args.output, args.output_npz, args.method)

    logging.info("Retargeting behavior '%s' from %s with method '%s'", behavior_name, input_dir, args.method)
    cfg = RetargetConfig(
        ground_contact=args.ground_contact,
        ground_clearance=args.ground_clearance,
        postprocess_global_pose=args.postprocess_global_pose,
        postprocess_align_window=args.align_window,
        fix_hx_zero=args.fix_hx_zero,
        trajectory_w_track=args.trajectory_w_track,
        trajectory_w_smooth=args.trajectory_w_smooth,
        trajectory_w_ground=args.trajectory_w_ground,
        trajectory_w_stable=args.trajectory_w_stable,
        trajectory_maxiter=args.trajectory_maxiter,
        trajectory_maxfun=args.trajectory_maxfun,
        trajectory_stable_joint_indices=parse_stable_joint_indices(args.trajectory_stable_joints),
    )
    if args.ground_contact:
        logging.warning(
            "--ground_contact is deprecated. Global pose postprocessing is now a separate stage "
            "and is enabled by default."
        )
    run = run_retarget_pipeline(input_dir, cfg, method=args.method)
    result = run.result
    metrics = compute_retarget_metrics(run, cfg)

    to_csv(result, out_csv)
    print(f"Wrote CSV: {out_csv}")
    to_numpy(result, out_npz)
    print(f"Wrote NPZ: {out_npz}")
    stage_path = default_stage_artifact_path(out_npz)
    save_stage_artifacts(run, stage_path, cfg)
    print(f"Wrote stage artifacts: {stage_path}")
    print(
        "Metrics: "
        f"Scale-aligned MPJPE={metrics.scale_aligned_mpjpe:.4f} m, "
        f"Joint Jump Rate={metrics.joint_jump_rate:.4%} "
        f"(threshold={metrics.joint_jump_threshold:.3f} rad), "
        f"Ground Penetration Rate={metrics.ground_penetration_rate:.4%} "
        f"(ground={metrics.ground_level:.3f} m)"
    )
    print("Reminder for downstream preprocessing: use --input_fps 24")

    if args.visualize:
        try:
            from .stage_viewer import run_viewer
        except ImportError:  # Support direct script execution: python main.py ...
            from animaspot_retarget.stage_viewer import run_viewer

        run_viewer(
            stage_npz=stage_path,
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
