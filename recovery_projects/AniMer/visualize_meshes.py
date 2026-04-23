"""Interactive viser viewer for AniMer per-frame SMAL meshes.

Expects the layout produced by demo_video.py:

    <out_folder>/meshes/faces.npy
    <out_folder>/meshes/0000.npz   # vertices, cam_t
    <out_folder>/meshes/0001.npz
    ...

For skeleton mode, also loads (if present):
    <out_folder>/pose3D/0000_3D.npz  # pose3d (26, 3)

Serves a Three.js viewer on 127.0.0.1:<port>; tunnel with
    ssh -L <port>:127.0.0.1:<port> <host>
and open http://127.0.0.1:<port> locally.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import viser


# Animal3D skeleton topology — same BONE_I/BONE_J used by animaspot_retarget/config.py.
# Each pair is one bone: (i, j) means joint i connects to joint j.
_BONE_I = [24, 24,  1,  0, 24,  2,  2, 24, 18, 18, 12, 13,  8,  9, 14, 15, 18,  7,  7, 10, 11, 16, 17,  7, 25]
_BONE_J = [ 0,  1, 21, 20,  2, 22, 23, 18, 12, 13,  8,  9, 14, 15,  3,  4,  7, 10, 11, 16, 17,  5,  6, 25, 19]
_BONES  = list(zip(_BONE_I, _BONE_J))

# Per-bone colors grouped by body region.
_BONE_COLORS: list[tuple[int, int, int]] = (
    [(200, 120, 255)] * 7   # head / face (0-6)
    + [(100, 180, 255)] * 2 # neck→shoulders (7-8)
    + [(80, 220, 100)]  * 6 # front legs (9-14)
    + [(100, 180, 255)] * 1 # neck→tail_base (15)
    + [(255, 230, 50)]  * 6 # back legs (16-21)
    + [(180, 180, 180)] * 3 # tail (22-24)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="viser viewer for AniMer SMAL meshes")
    parser.add_argument(
        "--mesh_dir",
        type=str,
        required=True,
        help="Path to the meshes/ folder produced by demo_video.py",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--apply_cam_t",
        action="store_true",
        help="Translate vertices by cam_t (camera-space placement). Off by default so the mesh sits at the world origin.",
    )
    return parser.parse_args()


def load_frames(mesh_dir: Path, apply_cam_t: bool):
    faces_path = mesh_dir / "faces.npy"
    if not faces_path.exists():
        raise FileNotFoundError(f"Missing {faces_path}; rerun demo_video.py to regenerate.")
    faces = np.load(faces_path).astype(np.int32)

    frame_files = sorted(p for p in mesh_dir.glob("*.npz"))
    if not frame_files:
        raise FileNotFoundError(f"No frame .npz files under {mesh_dir}")

    indices: list[int] = []
    vertices_per_frame: list[np.ndarray] = []
    for path in frame_files:
        try:
            idx = int(path.stem)
        except ValueError:
            continue
        data = np.load(path)
        verts = data["vertices"].astype(np.float32)
        if apply_cam_t and "cam_t" in data.files:
            verts = verts + data["cam_t"].astype(np.float32)[None, :]
        indices.append(idx)
        vertices_per_frame.append(verts)

    order = np.argsort(indices)
    indices = [indices[i] for i in order]
    vertices_per_frame = [vertices_per_frame[i] for i in order]
    return faces, indices, vertices_per_frame


def load_pose3d(mesh_dir: Path, frame_indices: list[int]) -> dict[int, np.ndarray]:
    """Load pose3d keypoints from the pose3D/ sibling folder, keyed by frame index."""
    pose3d_dir = mesh_dir.parent / "pose3D"
    result: dict[int, np.ndarray] = {}
    if not pose3d_dir.exists():
        return result
    for idx in frame_indices:
        path = pose3d_dir / f"{idx:04d}_3D.npz"
        if path.exists():
            result[idx] = np.load(path)["pose3d"].astype(np.float32)
    return result


def _bone_segments(joints: np.ndarray) -> np.ndarray:
    """Return (N_bones, 2, 3) array of line segment endpoints."""
    return np.array([[joints[i], joints[j]] for i, j in _BONES], dtype=np.float32)


def main() -> None:
    args = parse_args()
    mesh_dir = Path(args.mesh_dir).expanduser().resolve()
    faces, frame_indices, frame_vertices = load_frames(mesh_dir, args.apply_cam_t)
    pose3d_map = load_pose3d(mesh_dir, frame_indices)

    has_skeleton = len(pose3d_map) > 0
    print(f"Loaded {len(frame_vertices)} frames from {mesh_dir}")
    print(f"Mesh: {frame_vertices[0].shape[0]} vertices, {faces.shape[0]} faces")
    if has_skeleton:
        print(f"Skeleton: pose3d found for {len(pose3d_map)}/{len(frame_indices)} frames")
    else:
        print("Skeleton: no pose3D/ data found — skeleton mode unavailable")

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.world_axes.visible = True

    server.scene.add_mesh_simple(
        name="/ground",
        vertices=np.array(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        color=(220, 220, 220),
        opacity=0.4,
    )

    mesh_handle = server.scene.add_mesh_simple(
        name="/smal",
        vertices=frame_vertices[0],
        faces=faces,
        color=(166, 189, 219),
    )

    # Skeleton handles — only created when pose3D data exists.
    skel_joints_handle = None
    skel_bones_handle = None
    label_handles: list = []
    if has_skeleton:
        joints0 = pose3d_map.get(frame_indices[0])
        if joints0 is not None:
            skel_joints_handle = server.scene.add_point_cloud(
                name="/skel/joints",
                points=joints0,
                colors=np.array([255, 80, 80], dtype=np.uint8),
                point_size=0.015,
                point_shape="circle",
                visible=False,
            )
            bone_colors = np.array(
                [[c, c] for c in _BONE_COLORS], dtype=np.uint8
            )  # (N_bones, 2, 3) — one color per endpoint pair
            skel_bones_handle = server.scene.add_line_segments(
                name="/skel/bones",
                points=_bone_segments(joints0),
                colors=bone_colors,
                line_width=3.0,
                visible=False,
            )
            for idx in range(len(joints0)):
                label_handles.append(
                    server.scene.add_label(
                        name=f"/skel/label/{idx}",
                        text=str(idx),
                        position=tuple(joints0[idx].tolist()),
                        font_size_mode="screen",
                        font_screen_scale=0.6,
                        visible=False,
                    )
                )

    # GUI -----------------------------------------------------------------------
    view_options = ["Mesh", "Skeleton", "Both"] if has_skeleton else ["Mesh"]

    with server.gui.add_folder("View"):
        view_dropdown = server.gui.add_dropdown(
            label="Mode", options=view_options, initial_value="Mesh"
        )
        show_indices = server.gui.add_checkbox(
            label="Joint indices", initial_value=False,
            disabled=not has_skeleton,
        )

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            label="Frame index",
            min=0,
            max=len(frame_vertices) - 1,
            step=1,
            initial_value=0,
        )
        play_button = server.gui.add_button("Play")
        pause_button = server.gui.add_button("Pause")
        fps_slider = server.gui.add_slider(
            label="FPS", min=1, max=60, step=1, initial_value=24
        )
        info_text = server.gui.add_text(
            label="Source frame", initial_value=str(frame_indices[0]), disabled=True
        )

    state = {"playing": False}

    # ---------------------------------------------------------------------------

    def _apply_view(mode: str) -> None:
        show_mesh = mode in ("Mesh", "Both")
        show_skel = mode in ("Skeleton", "Both")
        mesh_handle.visible = show_mesh
        if skel_joints_handle is not None:
            skel_joints_handle.visible = show_skel
        if skel_bones_handle is not None:
            skel_bones_handle.visible = show_skel
        # labels follow skeleton visibility AND the checkbox
        _apply_labels(show_skel)

    def _apply_labels(skel_visible: bool) -> None:
        want = skel_visible and show_indices.value
        for lh in label_handles:
            lh.visible = want

    def _update_frame(i: int) -> None:
        mesh_handle.vertices = frame_vertices[i]
        info_text.value = str(frame_indices[i])
        joints = pose3d_map.get(frame_indices[i])
        if joints is not None:
            if skel_joints_handle is not None:
                skel_joints_handle.points = joints
            if skel_bones_handle is not None:
                skel_bones_handle.points = _bone_segments(joints)
            for idx, lh in enumerate(label_handles):
                lh.position = tuple(joints[idx].tolist())

    @frame_slider.on_update
    def _(_event) -> None:
        _update_frame(int(frame_slider.value))

    @view_dropdown.on_update
    def _(_event) -> None:
        _apply_view(view_dropdown.value)

    @show_indices.on_update
    def _(_event) -> None:
        show_skel = view_dropdown.value in ("Skeleton", "Both")
        _apply_labels(show_skel)

    @play_button.on_click
    def _(_event) -> None:
        state["playing"] = True

    @pause_button.on_click
    def _(_event) -> None:
        state["playing"] = False

    print(f"Viser viewer running at http://{args.host}:{args.port}")
    print("SSH tunnel: ssh -L <port>:127.0.0.1:<port> <host>")

    try:
        while True:
            if state["playing"]:
                next_i = (int(frame_slider.value) + 1) % len(frame_vertices)
                frame_slider.value = next_i
            time.sleep(1.0 / max(1, int(fps_slider.value)))
    except KeyboardInterrupt:
        print("Shutting down viewer.")


if __name__ == "__main__":
    main()
