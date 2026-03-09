"""Visualization helpers for retarget debugging."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .config import BONE_I, BONE_J, HIP_ATTACHMENTS, LEG_ORDER, LEG_SIDE, L_LOWER, L_UPPER, HIP_X_OFFSET
from .ik_solver import leg_keypoints


def _spot_points_from_angles(spot_angles: np.ndarray) -> dict:
    points = {}
    for i, leg in enumerate(LEG_ORDER):
        hx, hy, kn = spot_angles[3 * i : 3 * i + 3]
        hip = HIP_ATTACHMENTS[leg]
        knee_rel, paw_rel = leg_keypoints(
            hx=hx,
            hy=hy,
            kn=kn,
            hip_offset=HIP_X_OFFSET,
            L_upper=L_UPPER,
            L_lower=L_LOWER,
            side=LEG_SIDE[leg],
        )
        points[leg] = {
            "hip": hip,
            "knee": hip + knee_rel,
            "paw": hip + paw_rel,
        }
    return points


def plot_frame(dog_pose: np.ndarray, spot_angles: np.ndarray, frame_idx: int) -> None:
    """Overlay dog skeleton and reconstructed Spot leg keypoints."""
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    for i, j in zip(BONE_I, BONE_J):
        p1, p2 = dog_pose[i], dog_pose[j]
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="tab:blue", lw=1.5)
    ax1.scatter(dog_pose[:, 0], dog_pose[:, 1], dog_pose[:, 2], s=10, color="tab:blue")
    ax1.set_title(f"Dog Pose (frame {frame_idx})")

    spot = _spot_points_from_angles(spot_angles)
    for leg in LEG_ORDER:
        hip = spot[leg]["hip"]
        knee = spot[leg]["knee"]
        paw = spot[leg]["paw"]
        ax2.plot([hip[0], knee[0]], [hip[1], knee[1]], [hip[2], knee[2]], color="tab:orange", lw=2.0)
        ax2.plot([knee[0], paw[0]], [knee[1], paw[1]], [knee[2], paw[2]], color="tab:red", lw=2.0)
        ax2.scatter([hip[0], knee[0], paw[0]], [hip[1], knee[1], paw[1]], [hip[2], knee[2], paw[2]], s=20)
    ax2.set_title("Spot FK Reconstruction")

    for ax in (ax1, ax2):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    plt.show()


def _compute_axis_bounds(dog_sequence: np.ndarray, spot_joint_sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    all_points = [dog_sequence.reshape(-1, 3)]
    for f in range(spot_joint_sequence.shape[0]):
        spot = _spot_points_from_angles(spot_joint_sequence[f])
        leg_pts = []
        for leg in LEG_ORDER:
            leg_pts.extend([spot[leg]["hip"], spot[leg]["knee"], spot[leg]["paw"]])
        all_points.append(np.asarray(leg_pts))
    stacked = np.vstack(all_points)
    mins = np.min(stacked, axis=0)
    maxs = np.max(stacked, axis=0)
    span = np.maximum(maxs - mins, 1e-3)
    margin = 0.1 * span
    return mins - margin, maxs + margin


def _style_axis(ax: plt.Axes, title: str, mins: np.ndarray, maxs: np.ndarray) -> None:
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect((1, 1, 1))


def animate_sequence(
    dog_sequence: np.ndarray,
    spot_joint_sequence: np.ndarray,
    fps: int = 24,
    repeat: bool = True,
) -> FuncAnimation:
    """Animate dog skeleton and Spot FK reconstruction for all frames."""
    if dog_sequence.shape[0] != spot_joint_sequence.shape[0]:
        raise ValueError(
            f"Frame mismatch: dog_sequence has {dog_sequence.shape[0]}, "
            f"spot_joint_sequence has {spot_joint_sequence.shape[0]}"
        )

    n_frames = dog_sequence.shape[0]
    mins, maxs = _compute_axis_bounds(dog_sequence, spot_joint_sequence)
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    def _draw(frame_idx: int) -> None:
        dog_pose = dog_sequence[frame_idx]
        spot_angles = spot_joint_sequence[frame_idx]

        ax1.cla()
        ax2.cla()

        for i, j in zip(BONE_I, BONE_J):
            p1, p2 = dog_pose[i], dog_pose[j]
            ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="tab:blue", lw=1.5)
        ax1.scatter(dog_pose[:, 0], dog_pose[:, 1], dog_pose[:, 2], s=10, color="tab:blue")

        spot = _spot_points_from_angles(spot_angles)
        for leg in LEG_ORDER:
            hip = spot[leg]["hip"]
            knee = spot[leg]["knee"]
            paw = spot[leg]["paw"]
            ax2.plot([hip[0], knee[0]], [hip[1], knee[1]], [hip[2], knee[2]], color="tab:orange", lw=2.0)
            ax2.plot([knee[0], paw[0]], [knee[1], paw[1]], [knee[2], paw[2]], color="tab:red", lw=2.0)
            ax2.scatter(
                [hip[0], knee[0], paw[0]],
                [hip[1], knee[1], paw[1]],
                [hip[2], knee[2], paw[2]],
                s=20,
                color="tab:red",
            )

        _style_axis(ax1, f"Dog Pose (frame {frame_idx})", mins, maxs)
        _style_axis(ax2, "Spot FK Reconstruction", mins, maxs)

    animation = FuncAnimation(
        fig,
        lambda idx: _draw(int(idx)),
        frames=n_frames,
        interval=1000.0 / float(max(1, fps)),
        repeat=repeat,
        blit=False,
    )
    plt.tight_layout()
    plt.show()
    return animation


def plot_joint_trajectories(joint_angles: np.ndarray, joint_names: list[str]) -> None:
    """Plot all 12 joint trajectories over time."""
    n = joint_angles.shape[0]
    t = np.arange(n)
    fig, axes = plt.subplots(4, 3, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.plot(t, joint_angles[:, i], lw=1.5)
        ax.set_title(joint_names[i])
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("frame")
    plt.tight_layout()
    plt.show()

