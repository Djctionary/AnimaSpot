# AnimaSpot Workflow

This repository contains one active recovery project and one retargeting project:

- `recovery_projects/AniMer/`: recovers animal pose and shape and exports Animal3D-format `pose3D`.
- `animaspot_retarget/`: converts recovered 3D joints into Spot motion for CSV export, NPZ export, and MuJoCo playback.

`recovery_projects/FMPose3D/` is kept in the repo for reference only and is now deprecated in the integrated workflow.

## Environment Setup

The workflow below was validated in a Conda environment named `AnimaSpot` with Python 3.10.

```bash
conda create -n AnimaSpot python=3.10 -y

conda run -n AnimaSpot python -m pip install \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu124

conda run -n AnimaSpot python -m pip install "numpy<2.0" scipy matplotlib mujoco tqdm

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
│   │   │   └── AI_PlayBow/
│   │   │       ├── meshes/
│   │   │       └── pose3D/
│   │   └── fmpose3d/
│   └── final/
│       └── animer/
│           └── AI_PlayBow/
│               ├── analytical_ik/
│               │   ├── AI_PlayBow_spot.csv
│               │   ├── AI_PlayBow_spot.npz
│               │   └── AI_PlayBow_stages.npz
│               └── trajectory_ik/
│                   ├── AI_PlayBow_spot.csv
│                   ├── AI_PlayBow_spot.npz
│                   └── AI_PlayBow_stages.npz
├── recovery_projects/
│   ├── AniMer/
│   └── FMPose3D/
├── urdf/
├── visualize_spot_csv_mujoco.py
└── README.md
```

## Paths

- Input videos: `pipeline_data/input/videos/<video_name>.mp4`
- AniMer intermediate output: `pipeline_data/intermediate/animer/<video_name>/`
- Retarget input: `pipeline_data/intermediate/animer/<video_name>/pose3D/*_3D.npz`
- Final CSV: `pipeline_data/final/<source>/<video_name>/<method>/<video_name>_spot.csv`
- Final NPZ: `pipeline_data/final/<source>/<video_name>/<method>/<video_name>_spot.npz`
- Stage artifact: `pipeline_data/final/<source>/<video_name>/<method>/<video_name>_stages.npz`

## Step 1: Recover 3D Pose with AniMer

Run AniMer from `recovery_projects/AniMer/`:

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

Expected output location:

```text
pipeline_data/intermediate/animer/<video_name>/
```

Retargeting consumes the shared `pose3D/*_3D.npz` output format.

## Step 1.5: Visualize AniMer Meshes

Run from the repository root:

```bash
python recovery_projects/AniMer/visualize_meshes.py \
  --mesh_dir pipeline_data/intermediate/animer/AI_PlayBow/meshes \
  --port 8080
```

Open `http://localhost:8080` in a browser to scrub through the mesh frames.

## Step 2: Retarget to Spot

Retarget input:

```text
pipeline_data/intermediate/animer/<video_name>/pose3D
```

Supported methods:

- `analytical_ik`: closed-form per-frame IK baseline, followed by smoothing and optional global pose grounding.
- `trajectory_ik`: full-sequence soft-constrained optimization with end-effector tracking, temporal smoothness, ground-penetration penalty, and joint-stability penalty.

If you omit `--output` and `--output_npz`, outputs are written to:

```text
pipeline_data/final/animer/<video_name>/<method>/<video_name>_spot.csv
pipeline_data/final/animer/<video_name>/<method>/<video_name>_spot.npz
pipeline_data/final/animer/<video_name>/<method>/<video_name>_stages.npz
```

### Step 2A: Run AnalyticalIK

```bash
python3 -m animaspot_retarget.main \
  --input_dir ./pipeline_data/intermediate/animer/AI_PlayBow/pose3D \
  --behavior AI_PlayBow \
  --method analytical_ik \
  --visualize
```

### Step 2B: Run TrajectoryIK

```bash
python3 -m animaspot_retarget.main \
  --input_dir ./pipeline_data/intermediate/animer/AI_PlayBow/pose3D \
  --behavior AI_PlayBow \
  --method trajectory_ik \
  --visualize
```

### Step 2C: Useful TrajectoryIK Options

```bash
python3 -m animaspot_retarget.main \
  --input_dir ./pipeline_data/intermediate/animer/AI_PlayBow/pose3D \
  --behavior AI_PlayBow \
  --method trajectory_ik \
  --trajectory_w_track 1.0 \
  --trajectory_w_smooth 0.05 \
  --trajectory_w_ground 5.0 \
  --trajectory_w_stable 0.02 \
  --trajectory_maxiter 80 \
  --trajectory_stable_joints hx
```

### Step 2D: Open a Saved Stage Artifact

```bash
python3 visualize_retarget_stages.py \
  --stage_npz pipeline_data/final/animer/AI_PlayBow/analytical_ik/AI_PlayBow_stages.npz \
  --port 8080
```

## Step 3: Visualize in MuJoCo

```bash
python3 visualize_spot_csv_mujoco.py \
  --model urdf/isaacsim_spot/spot_scene.xml \
  --csv pipeline_data/final/animer/AI_PlayBow/analytical_ik/AI_PlayBow_spot.csv \
  --fps 24 \
  --repeat
```

## Retarget Notes

### Root Pose Behavior

- `TrajectoryIK` does not optimize root pose. It optimizes only the 12 Spot joint angles.
- `root_position` is configured in `animaspot_retarget/config.py`.
- `root_quaternion` can now also be set manually in `animaspot_retarget/config.py`.
- When `root_quaternion` is provided, the retarget pipeline uses that quaternion for the entire exported sequence instead of the source-derived root quaternion sequence.
- For `trajectory_ik`, the same final `root_pos/root_quat` is used by ground-penetration evaluation, export, and `World_TrajectoryIK` visualization.

### Global Pose Postprocess

- `AnalyticalIK` applies an independent global pose postprocess by default.
- That postprocess rotates each frame so the four-foot support plane is as horizontal as possible.
- It keeps the retargeted joint angles unchanged.
- It shifts each frame vertically so the support plane stays close to ground level.
- Disable it with `--no_postprocess_global_pose` if you want the raw retarget root pose.

### TrajectoryIK Objective

`TrajectoryIK` optimizes the full sequence with `scipy.optimize.minimize(method="L-BFGS-B")`.
The decision variable is the full joint trajectory `q` with shape `(T, 12)`.

```text
E(q) =
  w_track  * E_track(q)
+ w_smooth * E_smooth(q)
+ w_ground * E_ground(q)
+ w_stable * E_stable(q)
```

```text
E_smooth(q) = alpha_vel * sum(||q[t+1] - q[t]||^2)
            + alpha_acc * sum(||q[t+2] - 2q[t+1] + q[t]||^2)
```

- `--trajectory_stable_joints` accepts joint groups `hx`, `hy`, `kn`, explicit Spot joint names such as `fl_hx`, or numeric joint indices.
- The optimizer displays a `tqdm` progress bar over L-BFGS-B iterations.

### Saved Stage Layout

Shared preprocessing stages:

```text
RecoveredPose
  -> BodyTransformed
  -> LegScaled
```

AnalyticalIK stages:

```text
Retargeted_AnalyticalIK
  -> Smoothed_AnalyticalIK
  -> Ground_AnalyticalIK
```

TrajectoryIK stages:

```text
Retargeted_TrajectoryIK
  -> World_TrajectoryIK
```

The Viser stage viewer reads the saved `*_stages.npz` artifact only. It does not rerun IK, smoothing, grounding, or any other retarget computation.

### Default Metrics

Every `python3 -m animaspot_retarget.main ...` retarget run now computes and prints:

- `Scale-aligned MPJPE`
- `Joint Jump Rate`
- `Ground Penetration Rate`

Metric definitions:

- `Scale-aligned MPJPE`: compares retarget input and retarget output in root-relative/body-relative coordinates using 12 leg landmarks after per-leg morphology scaling.
- `Joint Jump Rate`: measures how often any Spot joint changes more than the configured threshold between adjacent frames.
- `Ground Penetration Rate`: measures how often any paw falls below the configured ground plane in the final world-frame trajectory.

Related config defaults live in `animaspot_retarget/config.py`:

- `metrics_joint_jump_threshold = 0.5`
- `metrics_ground_level = 0.0`

## Deprecation Notes

- `FMPose3D` is deprecated in the integrated workflow and no longer has recommended command examples in this README.
- `pipeline_data/input/videos/` is the only input-video location used by the documented workflow.
- `animaspot_retarget.main` infers default CSV, NPZ, and stage-artifact save paths from the intermediate input folder and selected method.
- `--visualize` opens the Viser viewer from the saved stage artifact and does not trigger a second retarget pass.
- `analytical_ik` and `trajectory_ik` write to separate method folders.
- Spot MuJoCo assets live under `urdf/isaacsim_spot/`.
