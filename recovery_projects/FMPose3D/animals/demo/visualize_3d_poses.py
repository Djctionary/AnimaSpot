"""
Visualize 3D animal poses from saved .npz files.

Usage:
    python visualize_3d_poses.py                          # interactive slider
    python visualize_3d_poses.py --gif output.gif         # export animated GIF
    python visualize_3d_poses.py --mp4 output.mp4         # export MP4 video
    python visualize_3d_poses.py --pose_dir <path>        # custom directory
"""
import argparse
import glob
import os

import numpy as np
from scipy.signal import savgol_filter
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec

SKELETON_I = np.array([24, 24, 1, 0, 24, 2, 2, 24, 18, 18, 12, 13, 8, 9, 14, 15, 18, 7, 7, 10, 11, 16, 17, 7, 25])
SKELETON_J = np.array([0, 1, 21, 20, 2, 22, 23, 18, 12, 13, 8, 9, 14, 15, 3, 4, 7, 10, 11, 16, 17, 5, 6, 25, 19])


def compute_global_limits(poses):
    """Compute fixed axis limits across all frames so the view stays stable."""
    all_poses = np.stack(poses, axis=0)  # (T, 26, 3)
    lo = all_poses.min(axis=(0, 1))      # (3,)
    hi = all_poses.max(axis=(0, 1))      # (3,)
    center = (lo + hi) / 2
    half_range = (hi - lo).max() / 2 * 1.1  # uniform cube with 10% padding
    return center, half_range


def draw_pose(ax, vals, limits, color=(0/255, 176/255, 240/255), linewidth=2.5):
    if vals.ndim == 1:
        vals = vals.reshape(26, 3)

    for k in range(len(SKELETON_I)):
        x = [vals[SKELETON_I[k], 0], vals[SKELETON_J[k], 0]]
        y = [vals[SKELETON_I[k], 1], vals[SKELETON_J[k], 1]]
        z = [vals[SKELETON_I[k], 2], vals[SKELETON_J[k], 2]]
        ax.plot(x, y, z, lw=linewidth, color=color)

    ax.scatter(vals[:, 0], vals[:, 1], vals[:, 2], s=15, c='red', zorder=5)

    center, half_range = limits
    ax.set_xlim3d([center[0] - half_range, center[0] + half_range])
    ax.set_ylim3d([center[1] - half_range, center[1] + half_range])
    ax.set_zlim3d([center[2] - half_range, center[2] + half_range])
    ax.set_aspect('auto')

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)
    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)


def load_poses(pose_dir):
    files = sorted(glob.glob(os.path.join(pose_dir, '*_3D.npz')))
    if not files:
        raise FileNotFoundError(f"No *_3D.npz files found in {pose_dir}")
    poses = []
    for f in files:
        data = np.load(f)
        key = 'pose3d' if 'pose3d' in data else list(data.keys())[0]
        poses.append(data[key])
    print(f"Loaded {len(poses)} poses from {pose_dir}")
    return poses


def process_motion(poses, root_joint=7, smooth_window=11, smooth=True):
    """Root-center all frames and optionally apply temporal smoothing.

    Args:
        poses: list of (26, 3) arrays
        root_joint: joint index to use as origin (default 7 = withers)
        smooth_window: Savitzky-Golay window length (must be odd)
        smooth: whether to apply temporal smoothing

    Returns:
        list of processed (26, 3) arrays
    """
    motion = np.stack(poses, axis=0)  # (T, 26, 3)

    # Root centering: subtract root joint position per frame
    root_pos = motion[:, root_joint:root_joint+1, :]  # (T, 1, 3)
    motion = motion - root_pos
    print(f"Root-centered on joint {root_joint}, {len(poses)} frames")

    if smooth and motion.shape[0] >= smooth_window:
        motion = savgol_filter(motion, window_length=smooth_window, polyorder=3, axis=0)
        print(f"Applied Savitzky-Golay smoothing (window={smooth_window}, polyorder=3)")
    elif smooth:
        print(f"Skipping smoothing: need >= {smooth_window} frames, got {motion.shape[0]}")

    return [motion[i] for i in range(motion.shape[0])]


def interactive_viewer(poses, limits):
    matplotlib.use('TkAgg')
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[15, 1])
    ax = fig.add_subplot(gs[0], projection='3d')

    ax_slider = fig.add_subplot(gs[1])
    slider = Slider(ax_slider, 'Frame', 0, len(poses) - 1, valinit=0, valstep=1)

    def update(val):
        idx = int(slider.val)
        ax.cla()
        draw_pose(ax, poses[idx], limits)
        ax.set_title(f'Frame {idx}/{len(poses)-1}', fontsize=14)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)
    plt.tight_layout()
    plt.show()


def export_gif(poses, limits, output_path, fps=15):
    import imageio
    matplotlib.use('Agg')
    frames = []
    print(f"Rendering {len(poses)} frames...")
    for i, pose in enumerate(poses):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        draw_pose(ax, pose, limits)
        ax.set_title(f'Frame {i}', fontsize=12)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        frames.append(buf)
        plt.close(fig)
        if (i + 1) % 50 == 0:
            print(f"  rendered {i+1}/{len(poses)}")
    imageio.mimsave(output_path, frames, duration=1000 / fps, loop=0)
    print(f"Saved GIF to {output_path}")


def export_mp4(poses, limits, output_path, fps=30):
    import cv2
    matplotlib.use('Agg')
    writer = None
    print(f"Rendering {len(poses)} frames...")
    for i, pose in enumerate(poses):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        draw_pose(ax, pose, limits)
        ax.set_title(f'Frame {i}', fontsize=12)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        writer.write(cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))
        plt.close(fig)
        if (i + 1) % 50 == 0:
            print(f"  rendered {i+1}/{len(poses)}")
    if writer:
        writer.release()
    print(f"Saved MP4 to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize 3D animal poses')
    parser.add_argument('--pose_dir', type=str,
                        default=r'C:\MENU\Jiacheng_Dong\FMPose3D\animals\demo\predictions\dog_settledown\pose3D',
                        help='Directory containing *_3D.npz files')
    parser.add_argument('--gif', type=str, default=None, help='Export as GIF to this path')
    parser.add_argument('--mp4', type=str, default=None, help='Export as MP4 to this path')
    parser.add_argument('--fps', type=int, default=15, help='FPS for GIF/MP4 export')
    parser.add_argument('--root_joint', type=int, default=7,
                        help='Joint index for root centering (default 7 = withers)')
    parser.add_argument('--smooth_window', type=int, default=11,
                        help='Savitzky-Golay smoothing window size (odd number)')
    parser.add_argument('--no_smooth', action='store_true',
                        help='Disable temporal smoothing (only root centering)')
    args = parser.parse_args()

    poses = load_poses(args.pose_dir)
    poses = process_motion(
        poses,
        root_joint=args.root_joint,
        smooth_window=args.smooth_window,
        smooth=not args.no_smooth,
    )
    limits = compute_global_limits(poses)

    if args.gif:
        export_gif(poses, limits, args.gif, fps=args.fps)
    elif args.mp4:
        export_mp4(poses, limits, args.mp4, fps=args.fps)
    else:
        interactive_viewer(poses, limits)
