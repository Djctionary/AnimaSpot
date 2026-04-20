from pathlib import Path
import argparse
import os

import cv2
import detectron2
import detectron2.config
import detectron2.engine
import numpy as np
import torch
import torch.utils.data
from detectron2 import model_zoo
from tqdm import tqdm

from amr.datasets.vitdet_dataset import ViTDetDataset
from amr.models import load_amr
from amr.utils import recursive_to
from amr.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
ANIMAL_CLASS_IDS = {15, 16, 17, 18, 19, 21, 22}
DEFAULT_VIDEO_PATH = Path(__file__).resolve().parents[2] / "pipeline_data/input/videos/AI_PlayBow.mp4"


def parse_args():
    parser = argparse.ArgumentParser(description="AniMer video demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="data/AniMer/checkpoints/checkpoint.ckpt",
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=str(DEFAULT_VIDEO_PATH),
        help="Path to the input video",
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        default="demo_video_out",
        help="Output folder to save rendered videos",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for AniMer inference",
    )
    parser.add_argument(
        "--detector_score_thresh",
        type=float,
        default=0.5,
        help="Detectron2 score threshold",
    )
    parser.add_argument(
        "--animal_score_thresh",
        type=float,
        default=0.7,
        help="Animal detection score threshold",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=-1,
        help="Optional frame limit for debugging",
    )
    return parser.parse_args()


def build_detector(score_thresh: float):
    cfg = detectron2.config.get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.WEIGHTS = (
        "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/"
        "faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl"
    )
    return detectron2.engine.DefaultPredictor(cfg)


def get_animal_boxes(detector, frame_rgb: np.ndarray, score_thresh: float) -> np.ndarray:
    det_out = detector(frame_rgb)
    instances = det_out["instances"].to("cpu")
    valid_idx = [
        idx
        for idx, (cls_id, score) in enumerate(zip(instances.pred_classes, instances.scores))
        if int(cls_id) in ANIMAL_CLASS_IDS and float(score) > score_thresh
    ]
    if not valid_idx:
        return np.zeros((0, 4), dtype=np.float32)
    return instances.pred_boxes.tensor[valid_idx].numpy()


def predict_frame(
    frame_rgb: np.ndarray,
    detector,
    model,
    model_cfg,
    renderer,
    device,
    batch_size: int,
    animal_score_thresh: float,
):
    boxes = get_animal_boxes(detector, frame_rgb, animal_score_thresh)
    frame_h, frame_w = frame_rgb.shape[:2]

    if len(boxes) == 0:
        blank_frame = np.full((frame_h, frame_w, 3), 255, dtype=np.uint8)
        return frame_rgb, blank_frame, 0, None

    dataset = ViTDetDataset(model_cfg, frame_rgb, boxes)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_vertices = []
    all_cam_t = []
    all_keypoints_3d = []
    full_focal_length = None

    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        pred_cam = out["pred_cam"]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        pred_cam_t_full = cam_crop_to_full(
            pred_cam,
            box_center,
            box_size,
            img_size,
            scaled_focal_length,
        ).detach().cpu().numpy()

        full_focal_length = float(scaled_focal_length.detach().cpu().item())
        all_vertices.extend(out["pred_vertices"].detach().cpu().numpy())
        all_cam_t.extend(pred_cam_t_full)
        all_keypoints_3d.extend(out["pred_keypoints_3d"].detach().cpu().numpy())

    overlay_rgba = renderer.render_rgba_multiple(
        all_vertices,
        all_cam_t,
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(0, 0, 0),
        render_res=[frame_w, frame_h],
        focal_length=full_focal_length,
    )
    overlay_rgb = alpha_blend(frame_rgb, overlay_rgba)

    mesh_rgb = renderer.render_rgba_multiple(
        all_vertices,
        all_cam_t,
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
        render_res=[frame_w, frame_h],
        focal_length=full_focal_length,
    )
    mesh_rgb = (mesh_rgb[:, :, :3] * 255).astype(np.uint8)

    primary_pose3d = all_keypoints_3d[0] if all_keypoints_3d else None
    primary_vertices = all_vertices[0] if all_vertices else None
    primary_cam_t = all_cam_t[0] if all_cam_t else None
    return (
        overlay_rgb,
        mesh_rgb,
        len(all_vertices),
        primary_pose3d,
        primary_vertices,
        primary_cam_t,
    )


def alpha_blend(frame_rgb: np.ndarray, rgba: np.ndarray) -> np.ndarray:
    alpha = rgba[:, :, 3:4]
    blended = rgba[:, :, :3] * alpha + frame_rgb.astype(np.float32) / 255.0 * (1.0 - alpha)
    return (blended * 255).astype(np.uint8)


def main():
    args = parse_args()

    model, model_cfg = load_amr(args.checkpoint)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    renderer = Renderer(model_cfg, faces=model.smal.faces)
    detector = build_detector(args.detector_score_thresh)

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(args.out_folder, exist_ok=True)
    overlay_path = Path(args.out_folder) / f"{video_path.stem}_smal.mp4"
    mesh_path = Path(args.out_folder) / f"{video_path.stem}_3d.mp4"
    pose3d_dir = Path(args.out_folder) / "pose3D"
    pose3d_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = Path(args.out_folder) / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    smal_faces = model.smal.faces
    if isinstance(smal_faces, torch.Tensor):
        smal_faces = smal_faces.detach().cpu().numpy()
    np.save(mesh_dir / "faces.npy", np.asarray(smal_faces, dtype=np.int32))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay_writer = cv2.VideoWriter(str(overlay_path), fourcc, fps, (width, height))
    mesh_writer = cv2.VideoWriter(str(mesh_path), fourcc, fps, (width, height))

    frame_idx = 0
    pbar = tqdm(
        total=total_frames if total_frames > 0 else None,
        desc="Processing video",
        unit="frame",
        dynamic_ncols=True,
    )

    try:
        while True:
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break

            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            (
                overlay_rgb,
                mesh_rgb,
                num_animals,
                primary_pose3d,
                primary_vertices,
                primary_cam_t,
            ) = predict_frame(
                frame_rgb,
                detector,
                model,
                model_cfg,
                renderer,
                device,
                args.batch_size,
                args.animal_score_thresh,
            )

            if num_animals == 0:
                mesh_rgb = np.full_like(frame_rgb, 255)
            else:
                np.savez_compressed(
                    pose3d_dir / f"{frame_idx:04d}_3D.npz",
                    pose3d=primary_pose3d,
                )
                np.savez_compressed(
                    mesh_dir / f"{frame_idx:04d}.npz",
                    vertices=primary_vertices.astype(np.float32),
                    cam_t=primary_cam_t.astype(np.float32),
                )

            overlay_writer.write(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
            mesh_writer.write(cv2.cvtColor(mesh_rgb, cv2.COLOR_RGB2BGR))

            frame_idx += 1
            pbar.set_postfix(frame=frame_idx, animals=num_animals)
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        overlay_writer.release()
        mesh_writer.release()

    print(f"Saved overlay video to {overlay_path}")
    print(f"Saved raw 3D video to {mesh_path}")
    print(f"Saved raw 3D keypoints to {pose3d_dir}")
    print(f"Saved per-frame SMAL meshes to {mesh_dir}")


if __name__ == "__main__":
    main()
