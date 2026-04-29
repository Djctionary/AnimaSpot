# AnimaSpot Workflow

This repository contains two recovery projects and one retargeting project:

- `recovery_projects/FMPose3D/`: recovers Animal3D-format 3D joint positions from video.
- `recovery_projects/AniMer/`: recovers animal pose and shape for mesh-based visualization.
- `animaspot_retarget/`: converts recovered 3D joints into Spot motion for CSV export and MuJoCo playback.

## Environment Setup

The workflow below was validated in a single Conda environment named `AnimaSpot` with Python 3.10.

```bash
conda create -n AnimaSpot python=3.10 -y

conda run -n AnimaSpot python -m pip install \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu124

conda run -n AnimaSpot python -m pip install numpy<2.0 scipy matplotlib mujoco

conda run -n AnimaSpot python -m pip install -e ./recovery_projects/FMPose3D

conda run -n AnimaSpot python -m pip install --no-build-isolation \
  "git+https://github.com/mattloper/chumpy.git" \
  "git+https://github.com/facebookresearch/detectron2.git" \
  "git+https://github.com/facebookresearch/pytorch3d.git"

conda run -n AnimaSpot python -m pip install --no-build-isolation mmcv==1.3.9

conda run -n AnimaSpot python -m pip install \
  gdown pyrender pytorch-lightning smplx==0.1.28 xtcocotools open3d \
  gradio==5.1.0 pydantic==2.10.6 hydra-submitit-launcher hydra-colorlog \
  pyrootutils rich

conda run -n AnimaSpot python -m pip install --no-deps -e ./recovery_projects/AniMer
```

## Recommended Layout

Use `pipeline_data/` as the unified workflow root for inputs, intermediate artifacts, and final outputs:

```text
AnimaSpot/
├── animaspot_retarget/
├── pipeline_data/
│   ├── input/
│   │   └── videos/
│   │       ├── AI_Greeting.mp4
│   │       ├── AI_PlayBow.mp4
│   │       └── AI_SettleDown.mp4
│   ├── intermediate/
│   │   ├── animer/
│   │   └── fmpose3d/
│   │       ├── AI_Greeting/
│   │       │   ├── input_2D/
│   │       │   ├── pose2D/
│   │       │   ├── pose2D_on_image/
│   │       │   └── pose3D/
│   │       ├── AI_PlayBow/
│   │       └── AI_SettleDown/
│   └── final/
│       ├── AI_Greeting/
│       ├── AI_PlayBow/
│       └── AI_SettleDown/
├── recovery_projects/
│   ├── AniMer/
│   └── FMPose3D/
├── urdf/
├── visualize_spot_csv_mujoco.py
└── README.md
```

## What Goes Where

- Input MP4 videos:
  `pipeline_data/input/videos/AI_Greeting.mp4`
  `pipeline_data/input/videos/AI_PlayBow.mp4`
  `pipeline_data/input/videos/AI_SettleDown.mp4`
- Intermediate recovery output from `FMPose3D`:
  `pipeline_data/intermediate/fmpose3d/<video_name>/pose3D/*_3D.npz`
- Intermediate recovery output from `AniMer`:
  `pipeline_data/intermediate/animer/<video_name>/`
- Final Spot motion output:
  `pipeline_data/final/<video_name>/<video_name>_spot.csv`
- Optional final retarget archive:
  `pipeline_data/final/<video_name>/<video_name>_spot.npz`

The `*_3D.npz` files from `FMPose3D` are the unsmoothed recovered 3D positions currently consumed by `animaspot_retarget`.

## Step 1: Recover 3D Pose

Step 1 has two choices:

- Option A: `FMPose3D`
- Option B: `AniMer`

### Step 1A: Recover 3D Pose with FMPose3D

The default `vis_animals.sh` now writes outputs to:

- `pipeline_data/intermediate/fmpose3d/<video_name>/`

It also reads input videos from:

- `pipeline_data/input/videos/<video_name>.mp4`

Run it from `recovery_projects/FMPose3D/animals/demo/`, or call the Python entry point directly:

```bash
cd recovery_projects/FMPose3D/animals/demo

python vis_animals.py \
  --type video \
  --path ../../../../pipeline_data/input/videos/AI_PlayBow.mp4 \
  --output_root ../../../../pipeline_data/intermediate/fmpose3d \
  --saved_model_path ../pre_trained_models/fmpose3d_animals/fmpose3d_animals_pretrained_weights.pth \
  --model_type fmpose3d_animals \
  --sample_steps 3 \
  --batch_size 1 \
  --layers 5 \
  --dataset animal3d \
  --gpu 0 \
  --sh_file vis_animals.sh \
  --hypothesis_num 10 \
  --aggregation rpea \
  --topk 5 \
  --rpea_alpha 50.0 \
  --bone_norm True
```

The same pattern applies to:

- `pipeline_data/input/videos/AI_Greeting.mp4`
- `pipeline_data/input/videos/AI_SettleDown.mp4`

### Step 1B: Recover Pose/Shape with AniMer

AniMer is an alternative recovery step. Save its outputs under:

- `pipeline_data/intermediate/animer/<video_name>/`

```bash
cd recovery_projects/AniMer

python demo.py \
  --checkpoint data/AniMer/checkpoints/checkpoint.ckpt \
  --img_folder example_data \
  --out_folder ../../pipeline_data/intermediate/animer/demo_images

python demo_video.py \
  --video_path ../../pipeline_data/input/videos/AI_PlayBow.mp4 \
  --checkpoint data/AniMer/checkpoints/checkpoint.ckpt
```

Step 2 now consumes the shared `pose3D/*_3D.npz` output format produced by both `FMPose3D` and `AniMer`.

## Step 1.5: Visualize AniMer Meshes (Optional)

After running AniMer, you can preview the recovered meshes in a browser before retargeting.
Run from the `AnimaSpot/` root:

```bash
python recovery_projects/AniMer/visualize_meshes.py \
  --mesh_dir pipeline_data/intermediate/animer/AI_PlayBow/meshes \
  --port 8080
```

Open `http://localhost:8080` in a browser to scrub through the mesh frames interactively.

## Step 2: Retarget to Spot

Use either intermediate `pose3D` folder as the retargeting input:

- `pipeline_data/intermediate/fmpose3d/<video_name>/pose3D`
- `pipeline_data/intermediate/animer/<video_name>/pose3D`

If you omit `--output` and `--output_npz`, the retargeting CLI now saves automatically under:

- `pipeline_data/final/fmpose3d/<video_name>/<video_name>_spot.csv`
- `pipeline_data/final/fmpose3d/<video_name>/<video_name>_spot.npz`
- `pipeline_data/final/animer/<video_name>/<video_name>_spot.csv`
- `pipeline_data/final/animer/<video_name>/<video_name>_spot.npz`

```bash
python3 -m animaspot_retarget.main \
  --input_dir ./pipeline_data/intermediate/fmpose3d/AI_PlayBow/pose3D \
  --behavior AI_PlayBow \
  --visualize
```

Or retarget an `AniMer` result:

```bash
python3 -m animaspot_retarget.main \
  --input_dir ./pipeline_data/intermediate/animer/AI_PlayBow/pose3D \
  --behavior AI_PlayBow \
  --visualize
```

By default, retarget export now applies an independent global pose postprocess that:

- rotates each frame so the four-foot support plane is as horizontal as possible
- keeps the retargeted joint angles unchanged
- shifts each frame vertically so the support plane stays close to ground level

If you want the raw retarget root pose without this correction, add `--no_postprocess_global_pose`.

## Step 3: Visualize in MuJoCo

```bash
python3 visualize_spot_csv_mujoco.py \
  --model urdf/isaacsim_spot/spot_scene.xml \
  --csv pipeline_data/final/animer/AI_PlayBow/AI_PlayBow_spot.csv \
  --fps 24 \
  --repeat
```

## Notes

- `pipeline_data/input/videos/` is now the only input-video location used by the integrated workflow.
- `vis_animals.py` now supports `--output_root` so recovery outputs no longer have to stay under `animals/demo/predictions/`.
- Saved `FMPose3D` intermediate `pose3D` results are post-processed before export (multi-hypothesis aggregation, bone-length normalization, limb regularization), while saved `AniMer` `pose3D` results are written without extra temporal smoothing in `demo_video.py`.
- `animaspot_retarget.main` now infers default CSV/NPZ save paths from the intermediate input folder, applies a sequence-wide global pose correction by default, and creates parent directories automatically.
- Spot MuJoCo assets live under `urdf/isaacsim_spot/`.
