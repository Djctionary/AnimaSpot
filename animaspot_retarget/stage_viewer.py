"""Viser viewer for saved AnimaSpot retargeting stage artifacts."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import viser
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The retarget stage viewer requires the `viser` package. "
        "Install it in this Python environment or run from the environment used by "
        "recovery_projects/AniMer/visualize_meshes.py."
    ) from exc


STAGE_OPTIONS = [
    "RecoveredPose",
    "BodyTransformed",
    "LegScaled",
    "Retargeted_AnalyticalIK",
    "Smoothed_AnalyticalIK",
    "Ground_AnalyticalIK",
    "Retargeted_TrajectoryIK",
    "World_TrajectoryIK",
]

_STAGE_ARRAYS = {
    "BodyTransformed": "stage3_body_skeleton",
    "LegScaled": "stage4_scaled_skeleton",
    "Retargeted_AnalyticalIK": "stage5_ik_skeleton",
    "Retargeted_TrajectoryIK": "stage5_ik_skeleton",
    "Smoothed_AnalyticalIK": "stage6_smoothed_skeleton",
    "Ground_AnalyticalIK": "stage7_postprocessed_skeleton",
    "World_TrajectoryIK": "stage7_postprocessed_skeleton",
}

_PREVIOUS_STAGE_ARRAYS = {
    "LegScaled": "stage3_body_skeleton",
    "Retargeted_AnalyticalIK": "stage4_scaled_skeleton",
    "Retargeted_TrajectoryIK": "stage4_scaled_skeleton",
    "Smoothed_AnalyticalIK": "stage5_ik_skeleton",
    "Ground_AnalyticalIK": "stage6_smoothed_skeleton",
    "World_TrajectoryIK": "stage5_ik_skeleton",
}


def _as_string(value: np.ndarray) -> str:
    if value.shape == ():
        return str(value.item())
    return str(value)


def _load_stage_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _stage_options(debug: dict[str, np.ndarray]) -> list[str]:
    if "stage_names" not in debug:
        return STAGE_OPTIONS
    return [str(stage) for stage in debug["stage_names"]]


def _segment_points(points: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.asarray([[points[i], points[j]] for i, j in edges], dtype=np.float32)


def _repeated_segment_colors(color: tuple[int, int, int], count: int) -> np.ndarray:
    return np.array([[color, color] for _ in range(count)], dtype=np.uint8)


def _animal_bone_colors(count: int) -> np.ndarray:
    colors: list[tuple[int, int, int]] = (
        [(200, 120, 255)] * 7
        + [(100, 180, 255)] * 2
        + [(80, 220, 100)] * 6
        + [(100, 180, 255)] * 1
        + [(255, 230, 50)] * 6
        + [(180, 180, 180)] * 3
    )
    if len(colors) < count:
        colors.extend([(220, 220, 220)] * (count - len(colors)))
    return np.array([[color, color] for color in colors[:count]], dtype=np.uint8)


def _rotation_x(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _transform_points(points: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return (rotation @ points.T).T


def _transform_segments(segments: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    flat = segments.reshape(-1, 3)
    return _transform_points(flat, rotation).reshape(segments.shape)


def _ground_grid(size: float = 1.5, step: float = 0.25) -> np.ndarray:
    coords = np.arange(-size, size + step * 0.5, step, dtype=np.float32)
    segments = []
    for value in coords:
        segments.append([[float(value), -size, 0.0], [float(value), size, 0.0]])
        segments.append([[-size, float(value), 0.0], [size, float(value), 0.0]])
    return np.array(segments, dtype=np.float32)


def _body_axis_segments(origin: np.ndarray, axes: np.ndarray, length: float) -> np.ndarray:
    return np.array(
        [
            [origin, origin + axes[:, 0] * length],
            [origin, origin + axes[:, 1] * length],
            [origin, origin + axes[:, 2] * length],
        ],
        dtype=np.float32,
    )


def _load_mesh_frames(mesh_dir: Path, apply_cam_t: bool) -> tuple[np.ndarray, dict[int, np.ndarray]] | None:
    faces_path = mesh_dir / "faces.npy"
    if not faces_path.exists():
        return None

    faces = np.load(faces_path).astype(np.int32)
    frames: dict[int, np.ndarray] = {}
    for path in sorted(mesh_dir.glob("*.npz")):
        try:
            frame_idx = int(path.stem)
        except ValueError:
            continue
        data = np.load(path)
        if "vertices" not in data:
            continue
        vertices = data["vertices"].astype(np.float32)
        if apply_cam_t and "cam_t" in data.files:
            vertices = vertices + data["cam_t"].astype(np.float32)[None, :]
        frames[frame_idx] = vertices

    if not frames:
        return None
    return faces, frames


def _format_diagnostics(debug: dict[str, np.ndarray], frame_idx: int, stage: str) -> str:
    frame_indices = debug["frame_indices"]
    lines = [f"source frame: {int(frame_indices[frame_idx])}", f"stage: {stage}"]

    if "scale_factors" in debug:
        leg_order = [str(item) for item in debug["leg_order"]]
        scales = debug["scale_factors"]
        lines.append(
            "scales: "
            + ", ".join(f"{leg}={float(scale):.3f}" for leg, scale in zip(leg_order, scales))
        )
    if "leg_target_joint_names" in debug:
        leg_order = [str(item) for item in debug["leg_order"]]
        target_names = [str(item) for item in debug["leg_target_joint_names"]]
        lines.append(
            "targets: "
            + ", ".join(f"{leg}={target_name}" for leg, target_name in zip(leg_order, target_names))
        )

    if stage in {"Retargeted_AnalyticalIK", "Retargeted_TrajectoryIK"} and "stage5_target_errors" in debug:
        errors = debug["stage5_target_errors"][frame_idx]
        lines.append("IK target error: " + ", ".join(f"{float(err):.3f}m" for err in errors))
        if "hip_x_offset" in debug and "body_half_width" in debug:
            lines.append(
                "retargeted torso points: 4 mounts + 4 HY joints "
                f"(body_half_width={float(debug['body_half_width']):.3f}m, "
                f"hip_x_offset={float(debug['hip_x_offset']):.3f}m)"
            )
    if stage == "Smoothed_AnalyticalIK" and "stage6_target_errors" in debug:
        errors = debug["stage6_target_errors"][frame_idx]
        lines.append("smoothed target error: " + ", ".join(f"{float(err):.3f}m" for err in errors))
    if stage == "Ground_AnalyticalIK":
        postprocess_enabled = bool(debug["postprocess_global_pose"].item())
        lines.append(f"postprocess enabled: {postprocess_enabled}")

    return "\n".join(lines)


def _set_labels_visible(handles: list[Any], visible: bool) -> None:
    for handle in handles:
        handle.visible = visible


def run_viewer(
    stage_npz: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    apply_cam_t: bool = False,
    rotate_x_deg: float = -90.0,
    debug_npz: str | Path | None = None,
) -> None:
    """Run the interactive Viser stage viewer."""
    if stage_npz is None:
        stage_npz = debug_npz
    if stage_npz is None:
        raise ValueError("Expected a stage artifact NPZ path.")
    stage_path = Path(stage_npz).expanduser().resolve()
    debug = _load_stage_npz(stage_path)
    stage_options = _stage_options(debug)
    frame_indices = debug["frame_indices"]
    n_frames = int(debug["stage1_animal3d"].shape[0])

    mesh_dir_text = _as_string(debug.get("mesh_dir", np.array("")))
    mesh_data = _load_mesh_frames(Path(mesh_dir_text), apply_cam_t) if mesh_dir_text else None
    has_mesh = mesh_data is not None

    animal_edges = np.column_stack([debug["animal_bone_i"], debug["animal_bone_j"]]).astype(np.int32)
    spot_edges = debug["spot_stage_edges"].astype(np.int32)
    source_rotation = _rotation_x(rotate_x_deg)

    server = viser.ViserServer(host=host, port=port)
    # The default world axes can look like a square billboard near the origin in
    # some viewer states. Use explicit stage axes/grid instead.
    server.scene.world_axes.visible = False
    ground_handle = server.scene.add_line_segments(
        name="/ground_grid",
        points=_ground_grid(),
        colors=_repeated_segment_colors((120, 120, 120), _ground_grid().shape[0]),
        line_width=1.0,
    )

    mesh_handle = None
    if mesh_data is not None:
        faces, mesh_frames = mesh_data
        first_mesh_frame = mesh_frames.get(int(frame_indices[0]), next(iter(mesh_frames.values())))
        mesh_handle = server.scene.add_mesh_simple(
            name="/source_mesh",
            vertices=_transform_points(first_mesh_frame, source_rotation).astype(np.float32),
            faces=faces,
            color=(166, 189, 219),
            opacity=0.45,
        )
    else:
        mesh_frames = {}

    animal0 = debug["stage1_animal3d"][0]
    animal0_display = _transform_points(animal0, source_rotation)
    animal_joints_handle = server.scene.add_point_cloud(
        name="/animal/joints",
        points=animal0_display.astype(np.float32),
        colors=np.array([255, 80, 80], dtype=np.uint8),
        point_size=0.015,
        point_shape="circle",
    )
    animal_bones_handle = server.scene.add_line_segments(
        name="/animal/bones",
        points=_transform_segments(_segment_points(animal0, animal_edges), source_rotation),
        colors=_animal_bone_colors(len(animal_edges)),
        line_width=3.0,
    )
    # Avoid Viser label billboards here: in some browser/camera states they can
    # appear as white planes that occlude the skeleton.
    animal_label_handles: list[Any] = []

    spot0 = debug["stage3_body_skeleton"][0]
    spot_joints_handle = server.scene.add_point_cloud(
        name="/spot_stage/joints",
        points=spot0.astype(np.float32),
        colors=np.array([255, 180, 40], dtype=np.uint8),
        point_size=0.018,
        point_shape="circle",
        visible=False,
    )
    spot_bones_handle = server.scene.add_line_segments(
        name="/spot_stage/bones",
        points=_segment_points(spot0, spot_edges),
        colors=_repeated_segment_colors((255, 180, 40), len(spot_edges)),
        line_width=4.0,
        visible=False,
    )
    prev_bones_handle = server.scene.add_line_segments(
        name="/spot_stage/previous",
        points=_segment_points(spot0, spot_edges),
        colors=_repeated_segment_colors((140, 140, 140), len(spot_edges)),
        line_width=2.0,
        visible=False,
    )
    spot_label_handles: list[Any] = []

    axis0 = _body_axis_segments(debug["stage2_body_origins"][0], debug["stage2_body_axes"][0], length=0.2)
    axes_handle = server.scene.add_line_segments(
        name="/body_frame/axes",
        points=_transform_segments(axis0, source_rotation),
        colors=np.array(
            [
                [(255, 80, 80), (255, 80, 80)],
                [(80, 220, 100), (80, 220, 100)],
                [(80, 140, 255), (80, 140, 255)],
            ],
            dtype=np.uint8,
        ),
        line_width=5.0,
        visible=False,
    )

    with server.gui.add_folder("Stage"):
        stage_dropdown = server.gui.add_dropdown(
            label="Output",
            options=stage_options,
            initial_value=stage_options[0],
        )
        show_mesh = server.gui.add_checkbox(label="Mesh", initial_value=False, disabled=not has_mesh)
        show_ground = server.gui.add_checkbox(label="Sparse ground", initial_value=True)
        show_animal = server.gui.add_checkbox(label="Animal3D skeleton", initial_value=True)
        show_spot = server.gui.add_checkbox(label="Spot/stage skeleton", initial_value=True)
        show_previous = server.gui.add_checkbox(label="Previous-stage overlay", initial_value=False)
        show_axes = server.gui.add_checkbox(label="Body axes", initial_value=True)
        show_labels = server.gui.add_checkbox(label="Labels", initial_value=False, disabled=True)

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            label="Frame index",
            min=0,
            max=n_frames - 1,
            step=1,
            initial_value=0,
        )
        play_button = server.gui.add_button("Play")
        pause_button = server.gui.add_button("Pause")
        fps_slider = server.gui.add_slider(
            label="FPS",
            min=1,
            max=60,
            step=1,
            initial_value=int(debug["fps"].item()) if "fps" in debug else 24,
        )
        diagnostics = server.gui.add_text(
            label="Diagnostics",
            initial_value=_format_diagnostics(debug, 0, stage_options[0]),
            disabled=True,
        )

    state = {"playing": False}

    def _current_stage_points(stage: str, frame_idx: int) -> np.ndarray | None:
        key = _STAGE_ARRAYS.get(stage)
        if key is None:
            return None
        return debug[key][frame_idx]

    def _update_frame(frame_idx: int) -> None:
        stage = stage_dropdown.value
        source_frame = int(frame_indices[frame_idx])

        if mesh_handle is not None:
            vertices = mesh_frames.get(source_frame)
            if vertices is not None:
                mesh_handle.vertices = _transform_points(vertices, source_rotation).astype(np.float32)

        animal = debug["stage1_animal3d"][frame_idx]
        animal_display = _transform_points(animal, source_rotation)
        animal_joints_handle.points = animal_display.astype(np.float32)
        animal_bones_handle.points = _transform_segments(_segment_points(animal, animal_edges), source_rotation)
        axes_handle.points = _transform_segments(
            _body_axis_segments(
                debug["stage2_body_origins"][frame_idx],
                debug["stage2_body_axes"][frame_idx],
                length=0.2,
            ),
            source_rotation,
        )

        stage_points = _current_stage_points(stage, frame_idx)
        if stage_points is not None:
            spot_joints_handle.points = stage_points.astype(np.float32)
            spot_bones_handle.points = _segment_points(stage_points, spot_edges)

        prev_key = _PREVIOUS_STAGE_ARRAYS.get(stage)
        if prev_key is not None:
            prev_bones_handle.points = _segment_points(debug[prev_key][frame_idx], spot_edges)

        diagnostics.value = _format_diagnostics(debug, frame_idx, stage)
        _apply_visibility()

    def _apply_visibility() -> None:
        stage = stage_dropdown.value
        source_stage = stage in {"RecoveredPose", "BodyTransformed"}
        spot_stage = stage in _STAGE_ARRAYS

        if mesh_handle is not None:
            mesh_handle.visible = bool(show_mesh.value and source_stage)
        ground_handle.visible = bool(show_ground.value)
        animal_visible = bool(show_animal.value and source_stage)
        animal_joints_handle.visible = animal_visible
        animal_bones_handle.visible = animal_visible
        axes_handle.visible = bool(show_axes.value and stage == "BodyTransformed")

        spot_visible = bool(show_spot.value and spot_stage)
        spot_joints_handle.visible = spot_visible
        spot_bones_handle.visible = spot_visible
        prev_bones_handle.visible = bool(
            show_previous.value and spot_stage and stage in _PREVIOUS_STAGE_ARRAYS
        )

        _set_labels_visible(animal_label_handles, bool(show_labels.value and animal_visible))
        _set_labels_visible(spot_label_handles, bool(show_labels.value and spot_visible))

    @frame_slider.on_update
    def _(_event: Any) -> None:
        _update_frame(int(frame_slider.value))

    @stage_dropdown.on_update
    def _(_event: Any) -> None:
        _update_frame(int(frame_slider.value))

    @show_mesh.on_update
    def _(_event: Any) -> None:
        _apply_visibility()

    @show_ground.on_update
    def _(_event: Any) -> None:
        _apply_visibility()

    @show_animal.on_update
    def _(_event: Any) -> None:
        _apply_visibility()

    @show_spot.on_update
    def _(_event: Any) -> None:
        _apply_visibility()

    @show_previous.on_update
    def _(_event: Any) -> None:
        _apply_visibility()

    @show_axes.on_update
    def _(_event: Any) -> None:
        _apply_visibility()

    @show_labels.on_update
    def _(_event: Any) -> None:
        _apply_visibility()

    @play_button.on_click
    def _(_event: Any) -> None:
        state["playing"] = True

    @pause_button.on_click
    def _(_event: Any) -> None:
        state["playing"] = False

    _update_frame(0)
    print(f"Loaded stage artifacts: {stage_path}")
    print(f"Display rotation: X={rotate_x_deg:g} degrees")
    print(f"Viser stage viewer running at http://{host}:{port}")
    print(f"SSH tunnel: ssh -L {port}:127.0.0.1:{port} <host>")

    try:
        while True:
            if state["playing"]:
                frame_slider.value = (int(frame_slider.value) + 1) % n_frames
            time.sleep(1.0 / max(1, int(fps_slider.value)))
    except KeyboardInterrupt:
        print("Shutting down viewer.")
