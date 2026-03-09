# AnimaSpot Retargeting Pipeline — Coding Agent Prompt

## Project Context

You are building the retargeting module for AnimaSpot, a research project that transfers dog social behaviours (play bow, excited greeting, relaxed settle) from video-extracted 3D poses to a Boston Dynamics Spot robot. The retargeting module sits between the pose extraction stage (FMPose3D) and the RL training stage (BeyondMimic motion tracking in Isaac Lab via the whole_body_tracking framework).

The goal is to convert per-frame 26-joint Animal3D skeleton data into per-frame 12-DOF Spot joint angle trajectories, packaged as a generalized-coordinate CSV file compatible with the BeyondMimic / whole_body_tracking pipeline.

### Downstream Pipeline

After this retargeting module produces the CSV, the downstream workflow is:

1. A modified `csv_to_npz.py` (adapted from whole_body_tracking for Spot's kinematic tree) computes forward kinematics to produce body positions, velocities, and accelerations.
2. The resulting `.npz` file is used as the reference motion for BeyondMimic tracking policy training in Isaac Lab.

This retargeting module is responsible ONLY for producing the CSV of generalized coordinates. It does NOT perform the FK computation or produce the `.npz` file.

---

## Input Specification

Source: FMPose3D output, one `.npz` file per frame.

- Directory structure: `pose3D/XXXX_3D.npz` (e.g., `0000_3D.npz` through `0120_3D.npz`)
- Key inside each file: `'pose3d'`
- Shape: `(26, 3)` — 26 joints, each `(x, y, z)` in `float64`
- Coordinate space: root-relative 3D coordinates (joint 18, neck, is approximately at origin)
- Source frame rate: **24 fps**

### Animal3D 26-Joint Skeleton

```
Index  Joint Name
0      left_eye
1      right_eye
2      mouth_mid
3      left_front_paw
4      right_front_paw
5      left_back_paw
6      right_back_paw
7      tail_base
8      left_front_thigh
9      right_front_thigh
10     left_back_thigh
11     right_back_thigh
12     left_shoulder
13     right_shoulder
14     left_front_knee
15     right_front_knee
16     left_back_knee
17     right_back_knee
18     neck (root/center)
19     tail_end
20     left_ear
21     right_ear
22     left_mouth
23     right_mouth
24     nose
25     tail_mid
```

Bone connections (parent → child):

```python
BONE_I = np.array([24, 24, 1, 0, 24, 2, 2, 24, 18, 18, 12, 13, 8, 9, 14, 15, 18, 7, 7, 10, 11, 16, 17, 7, 25])
BONE_J = np.array([0, 1, 21, 20, 2, 22, 23, 18, 12, 13, 8, 9, 14, 15, 3, 4, 7, 10, 11, 16, 17, 5, 6, 25, 19])
```

---

## Output Specification

A single CSV file per behaviour clip, in Unitree generalized-coordinate convention, compatible with the BeyondMimic `csv_to_npz.py` preprocessing script.

### CSV Format

Each row is one frame. No header row. Values are comma-separated floats.

**Column layout (19 values per row):**

```
x, y, z, qx, qy, qz, qw, fl_hx, fl_hy, fl_kn, fr_hx, fr_hy, fr_kn, hl_hx, hl_hy, hl_kn, hr_hx, hr_hy, hr_kn
```

| Columns | Content | Units |
|---------|---------|-------|
| 0-2 | Root position (x, y, z) | meters |
| 3-6 | Root orientation quaternion (qx, qy, qz, qw) | unitless |
| 7-9 | Front-left leg: hx, hy, kn | radians |
| 10-12 | Front-right leg: hx, hy, kn | radians |
| 13-15 | Hind-left leg: hx, hy, kn | radians |
| 16-18 | Hind-right leg: hx, hy, kn | radians |

**Critical: The quaternion convention is (qx, qy, qz, qw), NOT (qw, qx, qy, qz).** This follows the Unitree/whole_body_tracking convention, not the DeepMimic convention.

### Root Position

Since the FMPose3D input is root-relative with no global translation, set root position to a fixed standing position `(0, 0, 0.5)` for all frames (Spot nominal standing height approximately 0.5m). These are stationary social behaviours, so a constant root position is acceptable. Root trajectory refinement can be done later if needed.

### Frame Rate

Output the CSV at the native input frame rate (24 fps). Do NOT resample to a higher frame rate. The downstream `csv_to_npz.py` script accepts an `--input_fps` argument and handles frame rate internally. When running the downstream script, specify `--input_fps 24`.

### Additional NumPy Output

Also output a `.npz` file with the same data in structured array form for convenience and debugging:

```python
np.savez(output_path,
    root_pos=root_pos,       # (N_frames, 3) — x, y, z
    root_quat=root_quat,     # (N_frames, 4) — qx, qy, qz, qw
    joint_angles=joint_angles, # (N_frames, 12) — fl_hx..hr_kn
    fps=24
)
```

This `.npz` is NOT the same as the BeyondMimic training `.npz` (which contains FK-derived body positions). It is a debug/intermediate file only.

---

## Spot Kinematic Parameters

### Leg Structure

Each leg has 3 revolute joints:
- **HX** (hip abduction/adduction): rotation about the x-axis (longitudinal body axis)
- **HY** (hip flexion/extension): rotation about the y-axis (lateral body axis)
- **KN** (knee flexion/extension): rotation about the y-axis

### Joint Limits (radians)

| Joint | Lower | Upper |
|-------|-------|-------|
| HX    | -0.7854 | +0.7854 |
| HY    | -0.8988 | +2.2951 |
| KN    | -2.7929 | -0.2577 |

Note: HY has a 50-degree bias from vertical. KN range is negative (backward-bending knee convention). The zero configuration has legs pointing straight down.

### Link Lengths (meters, from Clearpath Spot URDF)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `L_upper` | 0.3405 | Upper leg (hip to knee) |
| `L_lower` | 0.3405 | Lower leg (knee to foot) |
| `hip_x_offset` | 0.0547 | Lateral offset from body center to HX axis |
| `body_half_length` | 0.1945 | Longitudinal offset, body center to front/rear hip |
| `body_half_width` | 0.055 | Lateral offset, body center to hip |

### Hip Attachment Points (relative to body center)

```
FL_hip = (+body_half_length, +body_half_width, 0)
FR_hip = (+body_half_length, -body_half_width, 0)
HL_hip = (-body_half_length, +body_half_width, 0)
HR_hip = (-body_half_length, -body_half_width, 0)
```

Convention: x = forward, y = left, z = up.

### Joint Ordering in Isaac Lab

The Spot URDF in Isaac Lab uses this joint ordering:

```
Index  Joint Name
0      fl_hx    (front-left hip abduction)
1      fl_hy    (front-left hip flexion)
2      fl_kn    (front-left knee)
3      fr_hx    (front-right hip abduction)
4      fr_hy    (front-right hip flexion)
5      fr_kn    (front-right knee)
6      hl_hx    (hind-left hip abduction)
7      hl_hy    (hind-left hip flexion)
8      hl_kn    (hind-left knee)
9      hr_hx    (hind-right hip abduction)
10     hr_hy    (hind-right hip flexion)
11     hr_kn    (hind-right knee)
```

The CSV joint columns (columns 7-18) MUST follow this exact ordering.

---

## Algorithm: Analytical Inverse Kinematics

Implement a geometric closed-form IK solver for each 3-DOF leg. This is the standard approach for simple 3-DOF leg chains (used by Peng et al. 2020, Miller et al. 2025, and most quadruped IK implementations).

### Per-Frame Pipeline

For each frame:

#### Step 1: Extract Body Frame from Torso Joints

Use joints `{12 (left_shoulder), 13 (right_shoulder), 7 (tail_base), 18 (neck)}` to define the body plane.

- Forward direction: `neck (18) - tail_base (7)`, normalized
- Lateral direction: `left_shoulder (12) - right_shoulder (13)`, normalized
- Up direction: cross product of forward and lateral
- Orthogonalize via Gram-Schmidt
- Construct rotation matrix R_body → convert to quaternion in **(qx, qy, qz, qw)** order for CSV output

#### Step 2: Compute Paw Targets in Body Frame

For each leg, identify the corresponding paw joint and transform it into the body-centered coordinate frame:

| Leg | Shoulder | Thigh | Knee | Paw |
|-----|----------|-------|------|-----|
| FL  | 12 | 8  | 14 | 3  |
| FR  | 13 | 9  | 15 | 4  |
| HL  | 7* | 10 | 16 | 5  |
| HR  | 7* | 11 | 17 | 6  |

*Hind legs use tail_base (7) as the hip reference region, combined with back_thigh joints.

Compute paw position relative to the corresponding hip attachment point on Spot's body.

#### Step 3: Scale for Morphology Difference

The dog skeleton and Spot have different limb proportions. Compute a per-leg scaling factor:

```
scale = Spot_leg_length / Dog_leg_length
```

Where `Dog_leg_length` is measured from the Animal3D skeleton (shoulder-to-thigh + thigh-to-knee + knee-to-paw bone lengths, averaged across frames for stability). Apply this scale to the paw target vector.

#### Step 4: Solve 3-DOF Analytical IK

For each leg, given the scaled paw target vector `p = (px, py, pz)` relative to the hip attachment:

1. **HX (abduction):** `hx = atan2(py, -pz)` (project onto the y-z plane to find abduction angle, accounting for hip_x_offset)
2. **Effective reach:** After removing abduction, compute the planar distance in the sagittal plane:
   - `d_yz = sqrt(py^2 + pz^2) - hip_x_offset`
   - `r = sqrt(px^2 + d_yz^2)` (total reach in the leg's sagittal plane)
3. **KN (knee):** cosine rule:
   - `cos_kn = (r^2 - L_upper^2 - L_lower^2) / (2 * L_upper * L_lower)`
   - Clamp to [-1, 1], then `kn = acos(cos_kn)` or `kn = -acos(cos_kn)` depending on knee convention (Spot uses backward-bending, so the knee angle is negative)
4. **HY (hip flexion):**
   - `alpha = atan2(px, -d_yz)`
   - `beta = acos((L_upper^2 + r^2 - L_lower^2) / (2 * L_upper * r))`
   - `hy = alpha + beta` or `alpha - beta` depending on configuration

Clamp all angles to joint limits. Handle unreachable targets (r > L_upper + L_lower) by clamping to maximum extension.

**Important:** Verify the sign conventions against Spot's URDF. The zero configuration has legs pointing straight down. Test with a known pose (e.g., nominal standing: HX=0, HY approximately 0.7, KN approximately -1.4) before processing real data.

#### Step 5: Temporal Smoothing

Apply a Savitzky-Golay filter (window=7, polyorder=3) to the 12 joint angle trajectories AFTER IK solving. This removes high-frequency jitter from the pose estimation without distorting the overall motion shape.

Also smooth the root quaternion trajectory using SLERP-based smoothing to avoid orientation jitter between frames.

---

## File Structure

Create the following project structure:

```
animaspot_retarget/
├── config.py              # All constants: Spot kinematics, joint limits, joint mapping, file paths
├── skeleton.py            # Animal3D skeleton loader, bone length computation, body frame extraction
├── ik_solver.py           # Analytical 3-DOF IK solver for one Spot leg
├── retarget.py            # Full retargeting pipeline: load → body frame → scale → IK → smooth
├── export.py              # Export to Unitree-convention CSV and debug NumPy .npz
├── visualize.py           # Matplotlib 3D visualization: overlay dog skeleton and retargeted Spot pose
├── main.py                # CLI entry point: takes input directory, outputs CSV reference motion file
├── requirements.txt       # numpy, scipy, matplotlib
└── tests/
    ├── test_ik.py         # Unit tests: known paw positions → expected joint angles
    └── test_pipeline.py   # Integration test: load sample data → export → verify format
```

### File Responsibilities

**config.py**: Define all Spot kinematic constants (link lengths, hip offsets, joint limits, joint ordering), Animal3D joint index mapping, and default pipeline parameters (smoothing window, output fps). Include the Spot joint order list explicitly:

```python
SPOT_JOINT_NAMES = [
    "fl_hx", "fl_hy", "fl_kn",
    "fr_hx", "fr_hy", "fr_kn",
    "hl_hx", "hl_hy", "hl_kn",
    "hr_hx", "hr_hy", "hr_kn",
]
```

**skeleton.py**:
- `load_sequence(directory) -> np.ndarray` shape `(N_frames, 26, 3)`: Load all `XXXX_3D.npz` files in order.
- `compute_body_frame(pose) -> (R, t)`: From a single `(26, 3)` pose, extract body rotation matrix and translation using torso joints.
- `compute_dog_leg_lengths(sequence) -> dict`: Measure average bone lengths for each leg chain across the sequence.
- `rotation_matrix_to_quat_xyzw(R) -> np.ndarray`: Convert a 3x3 rotation matrix to quaternion in **(qx, qy, qz, qw)** order.

**ik_solver.py**:
- `solve_leg_ik(target_pos, hip_offset, L_upper, L_lower, joint_limits, side) -> (hx, hy, kn)`: Pure geometric IK for a single leg. `side` indicates left/right for sign convention.
- Include a `forward_kinematics(hx, hy, kn, ...) -> pos` function for validation (FK should reconstruct the target position within tolerance).

**retarget.py**:
- `retarget_sequence(input_dir, config) -> dict`: Full pipeline returning `{'joint_angles': (N, 12), 'root_quat': (N, 4), 'root_pos': (N, 3)}`.
- Each sub-step (body frame, mapping, scaling, IK, smoothing) should be a separate function called in sequence.

**export.py**:
- `to_csv(retarget_result, output_path)`: Write the CSV file in Unitree generalized-coordinate convention. Each row: `x, y, z, qx, qy, qz, qw, fl_hx, fl_hy, fl_kn, ..., hr_kn`. No header row.
- `to_numpy(retarget_result, output_path)`: Write debug `.npz` with keys `root_pos`, `root_quat`, `joint_angles`, `fps`.

**visualize.py**:
- `plot_frame(dog_pose, spot_angles, frame_idx)`: Side-by-side or overlaid 3D plot showing the original Animal3D skeleton and the reconstructed Spot pose via FK. This is critical for debugging.
- `plot_joint_trajectories(joint_angles, joint_names)`: Plot all 12 joint angles over time to inspect smoothness.

**main.py**:
- CLI: `python main.py --input_dir ./pose3D --output ./play_bow.csv --behavior play_bow --visualize`
- Should also accept `--smooth_window 7` for configuration.
- Should output both `.csv` and `.npz` (debug) files by default.

---

## Implementation Notes

1. **Sign conventions are critical.** Spot's URDF uses specific axis conventions. Left and right legs have mirrored HX signs. Test the IK solver first with nominal standing pose before running on real data.

2. **Quaternion convention: (qx, qy, qz, qw).** This is the Unitree/whole_body_tracking convention. scipy's `Rotation.as_quat()` returns (x, y, z, w) by default, which matches. Do NOT use (w, x, y, z) order.

3. **Do not over-engineer.** This is a research pipeline, not production code. Prioritize correctness and debuggability over abstraction. Each function should be independently testable.

4. **The visualization is not optional.** The primary way to verify correctness is to visually compare the dog skeleton pose and the Spot FK reconstruction. Without this, bugs in sign convention or joint mapping will go undetected.

5. **Degenerate cases:** When the paw target is inside the hip offset radius or beyond maximum reach, clamp gracefully and log a warning. Do not crash.

6. **Dependencies:** numpy, scipy, matplotlib only. No ROS, no URDF parser, no physics engine. Pure geometry.

7. **Verify with a sanity check:** Before processing real FMPose3D data, create a synthetic test: manually define a play-bow pose in Animal3D joint coordinates (front paws forward and low, rear up), run through the full pipeline, and verify the output joint angles produce a recognizable play-bow on Spot via FK visualization.

8. **No resampling needed.** Output at 24 fps (native FMPose3D rate). The downstream `csv_to_npz.py` script handles frame rate via its `--input_fps 24` argument.

---

## Validation Criteria

The retargeting is considered working if:

1. FK reconstruction from output joint angles produces paw positions that match the scaled input paw positions within 2cm RMSE.
2. All output joint angles are within Spot's joint limits at every frame.
3. Joint angle trajectories are smooth (no frame-to-frame jumps > 0.3 rad after smoothing).
4. The output CSV has exactly 19 comma-separated float values per row, with no header row.
5. The quaternion in columns 3-6 is in (qx, qy, qz, qw) order, and `sqrt(qx^2 + qy^2 + qz^2 + qw^2) ≈ 1.0` for every frame.
6. Visual inspection of the FK-reconstructed Spot pose shows recognizable behaviour shape (e.g., front body lowered for play bow).

---

## Downstream Integration Notes (for reference only, not part of this module)

After the CSV is produced, the user will:

1. Adapt `csv_to_npz.py` from `HybridRobotics/whole_body_tracking` to use Spot's URDF and kinematic tree instead of G1's.
2. Run: `python csv_to_npz.py --input_file play_bow.csv --input_fps 24 --output_name play_bow --headless`
3. This produces a `.npz` with FK-derived body positions, velocities, and accelerations.
4. The `.npz` is used to train a BeyondMimic tracking policy in Isaac Lab (IsaacSim 4.5, IsaacLab 2.1.0).

The whole_body_tracking codebase requires a Spot robot config (URDF/USD, joint stiffness/damping, action scales) and remapping of observation/reward body indices from G1 to Spot. These adaptations are separate from the retargeting module.
