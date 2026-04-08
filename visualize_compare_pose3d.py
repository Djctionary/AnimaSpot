"""Compare FMPose3D and AniMer pose3D outputs side by side."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

try:
    from scipy.signal import savgol_filter
except ImportError:  # Optional dependency for smoothing previews.
    savgol_filter = None


SKELETON_I = np.array([24, 24, 1, 0, 24, 2, 2, 24, 18, 18, 12, 13, 8, 9, 14, 15, 18, 7, 7, 10, 11, 16, 17, 7, 25])
SKELETON_J = np.array([0, 1, 21, 20, 2, 22, 23, 18, 12, 13, 8, 9, 14, 15, 3, 4, 7, 10, 11, 16, 17, 5, 6, 25, 19])
FMPOSE_COLOR = (0 / 255, 176 / 255, 240 / 255)
ANIMER_COLOR = (0 / 255, 170 / 255, 90 / 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two pose3D folders frame by frame.")
    parser.add_argument("--fmpose_dir", type=str, required=True, help="Directory containing FMPose3D *_3D.npz files.")
    parser.add_argument("--animer_dir", type=str, required=True, help="Directory containing AniMer *_3D.npz files.")
    parser.add_argument("--gif", type=str, default="", help="Optional GIF output path.")
    parser.add_argument("--mp4", type=str, default="", help="Optional MP4 output path.")
    parser.add_argument("--fps", type=int, default=15, help="FPS for GIF/MP4 export.")
    parser.add_argument("--root_joint", type=int, default=7, help="Joint index used for root-centering.")
    parser.add_argument("--smooth_window", type=int, default=11, help="Savitzky-Golay window size when --smooth is used.")
    parser.add_argument("--smooth", action="store_true", help="Optionally smooth both sequences for preview/export.")
    return parser.parse_args()


def load_poses(pose_dir: str | Path) -> list[np.ndarray]:
    pose_dir = Path(pose_dir)
    files = sorted(pose_dir.glob("*_3D.npz"), key=lambda path: int(path.stem.split("_")[0]))
    if not files:
        raise FileNotFoundError(f"No '*_3D.npz' files found in {pose_dir}")

    poses: list[np.ndarray] = []
    for file in files:
        data = np.load(file)
        key = "pose3d" if "pose3d" in data else list(data.keys())[0]
        poses.append(np.asarray(data[key], dtype=np.float64).reshape(26, 3))
    print(f"Loaded {len(poses)} poses from {pose_dir}")
    return poses


def process_motion(
    poses: list[np.ndarray],
    root_joint: int,
    smooth: bool,
    smooth_window: int,
    label: str,
) -> np.ndarray:
    motion = np.stack(poses, axis=0)
    motion = motion - motion[:, root_joint : root_joint + 1, :]
    print(f"{label}: root-centered on joint {root_joint}")

    if not smooth:
        return motion

    if savgol_filter is None:
        raise ImportError("scipy is required for --smooth")
    if motion.shape[0] < smooth_window:
        print(f"{label}: skipping smoothing, need >= {smooth_window} frames, got {motion.shape[0]}")
        return motion
    if smooth_window % 2 == 0:
        raise ValueError("--smooth_window must be odd")

    motion = savgol_filter(motion, window_length=smooth_window, polyorder=3, axis=0)
    print(f"{label}: applied Savitzky-Golay smoothing (window={smooth_window})")
    return motion


def trim_to_common_length(fmpose: np.ndarray, animer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame_count = min(len(fmpose), len(animer))
    if len(fmpose) != len(animer):
        print(
            f"Frame count mismatch: FMPose3D={len(fmpose)}, AniMer={len(animer)}. "
            f"Using the first {frame_count} frames for comparison."
        )
    return fmpose[:frame_count], animer[:frame_count]


def compute_limits(*motions: np.ndarray) -> tuple[np.ndarray, float]:
    all_poses = np.concatenate(motions, axis=0)
    lo = all_poses.min(axis=(0, 1))
    hi = all_poses.max(axis=(0, 1))
    center = (lo + hi) / 2.0
    half_range = max((hi - lo).max() / 2.0 * 1.1, 1e-6)
    return center, float(half_range)


def style_axis(ax, limits: tuple[np.ndarray, float]) -> None:
    center, half_range = limits
    ax.set_xlim3d([center[0] - half_range, center[0] + half_range])
    ax.set_ylim3d([center[1] - half_range, center[1] + half_range])
    ax.set_zlim3d([center[2] - half_range, center[2] + half_range])
    ax.set_box_aspect((1, 1, 1))

    clear = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(clear)
    ax.yaxis.set_pane_color(clear)
    ax.zaxis.set_pane_color(clear)

    ticks = np.linspace(-half_range, half_range, 5)
    ax.set_xticks(center[0] + ticks)
    ax.set_yticks(center[1] + ticks)
    ax.set_zticks(center[2] + ticks)
    ax.set_xlabel("X", labelpad=8)
    ax.set_ylabel("Y", labelpad=8)
    ax.set_zlabel("Z", labelpad=8)
    ax.tick_params(axis="both", which="major", labelsize=8, pad=2)
    ax.grid(True, alpha=0.3)


def draw_pose(ax, pose: np.ndarray, color: tuple[float, float, float], title: str, limits: tuple[np.ndarray, float]) -> None:
    for idx in range(len(SKELETON_I)):
        x = [pose[SKELETON_I[idx], 0], pose[SKELETON_J[idx], 0]]
        y = [pose[SKELETON_I[idx], 1], pose[SKELETON_J[idx], 1]]
        z = [pose[SKELETON_I[idx], 2], pose[SKELETON_J[idx], 2]]
        ax.plot(x, y, z, lw=2.5, color=color)

    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], s=15, c=[color], zorder=5)
    style_axis(ax, limits)
    ax.set_title(title, fontsize=13)


def draw_frame(
    ax_left,
    ax_right,
    fmpose: np.ndarray,
    animer: np.ndarray,
    frame_idx: int,
    frame_count: int,
    limits: tuple[np.ndarray, float],
) -> None:
    ax_left.cla()
    ax_right.cla()
    draw_pose(ax_left, fmpose[frame_idx], FMPOSE_COLOR, "FMPose3D", limits)
    draw_pose(ax_right, animer[frame_idx], ANIMER_COLOR, "AniMer", limits)
    ax_left.text2D(0.02, 0.95, f"Frame {frame_idx}/{frame_count - 1}", transform=ax_left.transAxes)


def interactive_viewer(fmpose: np.ndarray, animer: np.ndarray, limits: tuple[np.ndarray, float]) -> None:
    matplotlib.use("TkAgg")
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[15, 1])
    ax_left = fig.add_subplot(gs[0, 0], projection="3d")
    ax_right = fig.add_subplot(gs[0, 1], projection="3d")
    slider_ax = fig.add_subplot(gs[1, :])
    slider = Slider(slider_ax, "Frame", 0, len(fmpose) - 1, valinit=0, valstep=1)

    def update(_value: float) -> None:
        frame_idx = int(slider.val)
        draw_frame(ax_left, ax_right, fmpose, animer, frame_idx, len(fmpose), limits)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)
    plt.tight_layout()
    plt.show()


def render_frame_image(
    fmpose: np.ndarray,
    animer: np.ndarray,
    frame_idx: int,
    limits: tuple[np.ndarray, float],
) -> np.ndarray:
    fig = plt.figure(figsize=(10, 5))
    ax_left = fig.add_subplot(1, 2, 1, projection="3d")
    ax_right = fig.add_subplot(1, 2, 2, projection="3d")
    draw_frame(ax_left, ax_right, fmpose, animer, frame_idx, len(fmpose), limits)
    fig.tight_layout()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    plt.close(fig)
    return frame


def export_gif(fmpose: np.ndarray, animer: np.ndarray, limits: tuple[np.ndarray, float], output_path: str, fps: int) -> None:
    import imageio

    matplotlib.use("Agg")
    frames = []
    for idx in range(len(fmpose)):
        frames.append(render_frame_image(fmpose, animer, idx, limits))
    imageio.mimsave(output_path, frames, duration=1000 / fps, loop=0)
    print(f"Saved GIF to {output_path}")


def export_mp4(fmpose: np.ndarray, animer: np.ndarray, limits: tuple[np.ndarray, float], output_path: str, fps: int) -> None:
    import cv2

    matplotlib.use("Agg")
    writer = None
    for idx in range(len(fmpose)):
        frame = render_frame_image(fmpose, animer, idx, limits)
        if writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if writer is not None:
        writer.release()
    print(f"Saved MP4 to {output_path}")


def main() -> None:
    args = parse_args()
    fmpose = process_motion(load_poses(args.fmpose_dir), args.root_joint, args.smooth, args.smooth_window, "FMPose3D")
    animer = process_motion(load_poses(args.animer_dir), args.root_joint, args.smooth, args.smooth_window, "AniMer")
    fmpose, animer = trim_to_common_length(fmpose, animer)
    limits = compute_limits(fmpose, animer)

    if args.gif:
        export_gif(fmpose, animer, limits, args.gif, fps=args.fps)
    elif args.mp4:
        export_mp4(fmpose, animer, limits, args.mp4, fps=args.fps)
    else:
        interactive_viewer(fmpose, animer, limits)


if __name__ == "__main__":
    main()
