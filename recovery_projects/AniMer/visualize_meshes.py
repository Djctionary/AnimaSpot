"""Interactive viser viewer for AniMer per-frame SMAL meshes.

Expects the layout produced by demo_video.py:

    <out_folder>/meshes/faces.npy
    <out_folder>/meshes/0000.npz   # vertices, cam_t
    <out_folder>/meshes/0001.npz
    ...

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


def main() -> None:
    args = parse_args()
    mesh_dir = Path(args.mesh_dir).expanduser().resolve()
    faces, frame_indices, frame_vertices = load_frames(mesh_dir, args.apply_cam_t)

    print(f"Loaded {len(frame_vertices)} frames from {mesh_dir}")
    print(f"Mesh: {frame_vertices[0].shape[0]} vertices, {faces.shape[0]} faces")

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

    @frame_slider.on_update
    def _(_event) -> None:
        i = int(frame_slider.value)
        mesh_handle.vertices = frame_vertices[i]
        info_text.value = str(frame_indices[i])

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
