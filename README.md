# AnimaSpot Retargeting Quick Notes

This repository now includes:

- `animaspot_retarget/`: Animal3D (26-joint) to Spot (12-DOF) retargeting pipeline.
- `visualize_spot_csv_mujoco.py`: MuJoCo playback script for Spot CSV motion.
- CSV output format compatible with downstream `csv_to_npz.py` workflow.

## What Was Implemented

- Analytical 3-DOF leg IK for Spot (`hx`, `hy`, `kn`) with joint-limit clamping.
- Root orientation extraction from Animal3D torso frame, exported as quaternion `(qx, qy, qz, qw)`.
- Joint smoothing (Savitzky-Golay) and quaternion smoothing.
- CSV export (`19` values per frame) + debug NPZ export.
- Matplotlib visualization:
  - single-frame overlay
  - full sequence animation (`FuncAnimation`)
- MuJoCo visualization script with Spot joint-name mapping and CSV playback.

## Command: Retarget + Animate (Matplotlib)

```bash
python3 -m animaspot_retarget.main --input_dir ./pose3D --output ./play_bow.csv --behavior play_bow --animate
```

## Command: Visualize CSV in MuJoCo

```bash
python3 visualize_spot_csv_mujoco.py --model /home/vergil/MENU/Projects/AnimaSpot/urdf/isaacsim_spot/spot.urdf --csv /home/vergil/MENU/Projects/AnimaSpot/play_bow.csv --fps 24 --repeat
```

## URDF Source Notes

- Primary URDF source used for MuJoCo visualization:
  - `urdf/isaacsim_spot/spot.urdf`
- Meshes are visible when using the generated visual URDF variant:
  - `urdf/isaacsim_spot/spot_visual_f2l6_1mq.urdf`
- Confirmed key block (as seen at lines 329-332):

```xml
<mujoco>
  <compiler discardvisual="false"/>
</mujoco>
```

