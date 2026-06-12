# Monocular Vision based 3D Hand Tracking and Pose Estimation pipeline for applications in robot learning and fine dexterous manipulation

---


## Overview

The pipeline combines a strong 3D-hand baseline (InterNet, ECCV 2020) with three extensions that make it usable for robot teleoperation from a single RGB webcam:

| Stage | Module | Role |
|---|---|---|
| **Baseline** | InterNet (ResNet-50 + 3D heatmap head) | 21 keypoints per hand, handedness, root depth |
| **EXT 1** | Depth Anything V2 + depth fusion | Per-pixel metric depth, scale-aligned to the palm |
| **EXT 2** | 1 Euro filter | Per-joint adaptive low-pass smoothing |
| **EXT 3** | Gesture abstraction | Digit distances + palm normals → `GRASP` / `OPEN PALM` / `PINCH` |

On top of that, the pipeline writes a full **session report PNG** on exit (gesture dwell, depth timelines, pinch history, FPS).

---

## Architecture

```
Webcam (RGB, 30 fps)
        │
        ├──► Depth Anything V2  ──► DepthFusion  ──► metric depth map (m)
        │                                                     │
        ▼                                                     │
MMPose preprocessing  →  ResNet-50 backbone  →  InterNet head │
(256×256 tensor)          (2048-d features)     3D heatmap    │
                                                root depth Z  │
        │                                       handedness    │
        ▼                                                     │
Hand3DHeatmap codec  →  21 × (x, y, z) per hand  ◄────────────┘
        │                     (camera space)
        ▼
1 Euro filter (per joint, per axis)  →  jitter-free keypoints
        │
        ▼
Gesture abstraction  →  { GRASP, OPEN PALM, PINCH }  + meta (pinch_norm,
                                                            extension,
                                                            palm normal)
```

---

## File Map

| File | Role |
|---|---|
| `src/camera.py` | Threaded OpenCV capture |
| `src/pose_backends.py` | MMPose InterNet backend (baseline) |
| `src/depth_estimator.py` | Depth Anything V2 loader + DepthFusion (EXT 1) |
| `src/one_euro_filter.py` | 1 Euro adaptive low-pass filter (EXT 2) |
| `src/gesture_abstraction.py` | Gesture classifier (EXT 3) |
| `src/pipeline.py` | End-to-end runtime + live GUI + session report |
| `scripts/run_robot_learning_gui.py` | Entry point / CLI |

---

## Setup

### 1. Install dependencies

```bash
pip install torch torchvision mmpose opencv-python matplotlib timm
```

Tested on Python 3.10 with CUDA 11.8 and on CPU-only machines.

### 2. Depth Anything V2 checkpoint

Clone the DA2 repo inside `checkpoints/depth_anything_v2/` and drop a checkpoint next to it:

```bash
mkdir -p checkpoints/depth_anything_v2
cd checkpoints/depth_anything_v2
git clone https://github.com/DepthAnything/Depth-Anything-V2
```

Then download one of the checkpoints from the Hugging Face mirrors below and place it at `checkpoints/depth_anything_v2/depth_anything_v2_<encoder>.pth`.

| Encoder | File | Size | Source |
|---|---|---|---|
| `vits` | `depth_anything_v2_vits.pth` | ~100 MB | [Depth-Anything-V2-Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small) |
| `vitb` | `depth_anything_v2_vitb.pth` | ~400 MB | [Depth-Anything-V2-Base](https://huggingface.co/depth-anything/Depth-Anything-V2-Base) |
| `vitl` | `depth_anything_v2_vitl.pth` | ~1.3 GB | [Depth-Anything-V2-Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large) |

Relative vs metric: any file without `metric` / `indoor` / `outdoor` in the name produces disparity and requires the 3-second palm calibration described below. Metric checkpoints (e.g. `depth_anything_v2_metric_indoor_vitl.pth`) output absolute metres directly; calibration is then optional.

### 3. InterNet checkpoint

Place MMPose's InterNet weights at `checkpoints/res50.pth`, with the matching config at
`configs/hand_3d_keypoint/internet/interhand3d/internet_res50_4xb16-20e_interhand3d-256x256.py`.

---

## Run

```bash
# Default: CPU, vitl encoder
python scripts/run_robot_learning_gui.py

# Faster: small encoder on CPU
python scripts/run_robot_learning_gui.py --da2-encoder vits

# GPU: large encoder with full precision
python scripts/run_robot_learning_gui.py \
    --device cuda:0 \
    --da2-encoder vitl \
    --depth-model checkpoints/depth_anything_v2
```

### Keyboard shortcuts

| Key | Action |
|---|---|
| `c` | Start 3-second depth calibration (hold open palm at 40 cm) |
| `r` | Reset depth calibration |
| `ESC` | Quit and save session report |

### Depth calibration

1. Press `c`.
2. Hold your open palm **exactly 40 cm** from the camera.
3. Wait for the 3-second countdown.
4. Depth readings switch from `m*` (uncalibrated) to `m` (calibrated).

---

## Egocentric-IK: Vision-Based Hand Shadowing (vbhs) + SO-101 integration

This branch adds a second pipeline path that retargets the **tracked right hand**
to a [LeRobot SO-101](https://huggingface.co/docs/lerobot) follower arm, using a
vendored copy of [Vision-Based-Hand-Shadowing](https://github.com/chichonnade/Vision-Based-Hand-Shadowing-for-Robotic-Manipulation-via-Inverse-Kinematics)
(`vbhs/`) for camera-space deprojection, robot-space transforms, and PyBullet
inverse kinematics.

```
RealSense colour + metric depth
        │
        ▼
HandState (InterHand2.6M 21 kp)  ──► vbhs_bridge.hand_state_to_pose3d  ──► camera-space HandPose3D
                                                                                  │
                                                                                  ▼
                                          SingleArmIKRetargeter (vbhs PyBullet IK, SO-101 URDF)
                                                                                  │
                                                                                  ▼
                                          [j1..j5, gripper] radians  ──► SO101FollowerDriver
                                                                                  │
                                                                                  ▼
                                                            physical SO-101 follower arm
```

### Additional file map

| File | Role |
|---|---|
| `vbhs/` | Vendored Vision-Based-Hand-Shadowing source, robot URDF + meshes |
| `src/vbhs_bridge.py` | Maps `HandState` (InterHand2.6M) → vbhs `HandPose3D` (MediaPipe-named, camera-space) |
| `src/ik_retargeter.py` | `SingleArmIKRetargeter` — vbhs PyBullet IK for the SO-101 right arm |
| `src/robot_driver.py` | `SO101FollowerDriver` — sends IK joint angles to a physical SO-101 follower via LeRobot |

### Linux setup

#### 1. System dependencies

```bash
sudo apt update
sudo apt install -y python3-dev build-essential libusb-1.0-0-dev pkg-config \
                     cmake git
```

#### 2. RealSense SDK (`pyrealsense2`)

```bash
pip install pyrealsense2
```

If the prebuilt wheel doesn't work for your distro/kernel, build
[librealsense](https://github.com/IntelRealSense/librealsense) from source instead.
Either way, install the udev rules so the camera doesn't need root:

```bash
git clone https://github.com/IntelRealSense/librealsense.git /tmp/librealsense
sudo cp /tmp/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Reconnect the camera, then verify with `rs-enumerate-devices` (from `librealsense`) or:

```bash
python -c "import pyrealsense2 as rs; print(rs.context().query_devices())"
```

#### 3. Python dependencies (IK)

```bash
pip install -r requirements.txt
```

This installs `pybullet` and `scipy` for the vendored `vbhs` IK solver, in
addition to the base pipeline requirements.

#### 4. LeRobot + SO-101 follower arm (optional, for `--enable-robot`)

```bash
pip install "lerobot[feetech]>=0.4.1"
```

Connect the SO-101 follower arm's USB-serial adapter, then:

```bash
# Add yourself to the dialout group so you can access /dev/ttyACM* without root
sudo usermod -aG dialout $USER
# Log out/in (or `newgrp dialout`) for the group change to take effect

# Find the arm's serial port
lerobot-find-port
# -> typically /dev/ttyACM0
```

Run LeRobot's one-time calibration for the arm (records each joint's min/max
raw positions — required before `--enable-robot` will track correctly):

```bash
lerobot-calibrate --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=right_follower
```

Follow the prompts to move the arm through its full range of motion for each
joint, then let it settle to rest.

### Running

```bash
# IK overlay only (no physical arm) — shows solved SO-101 joint angles
python scripts/run_robot_learning_gui.py \
    --camera-backend realsense \
    --enable-ik

# Full pipeline: IK + physical SO-101 right arm
python scripts/run_robot_learning_gui.py \
    --camera-backend realsense \
    --enable-ik \
    --enable-robot \
    --robot-port /dev/ttyACM0 \
    --robot-id right_follower
```

`--enable-ik` requires `--camera-backend realsense` (real depth + intrinsics
are needed for the camera-space deprojection). If `--robot-port` can't be
opened, the pipeline logs a warning and continues running with the robot
disabled — the IK overlay still works.

### Keyboard shortcuts (Egocentric-IK additions)

| Key | Action |
|---|---|
| `g` | Arm the physical robot — starts sending IK joint targets (only with `--enable-robot`) |
| `s` | Return the robot to standby and stop sending targets |

The robot **starts in standby** on launch and stays there until `g` is
pressed, and is returned to standby on exit (`ESC`). The HUD shows
`[ARMED]`/`[STANDBY]` when `--enable-robot` is active.

### Safety notes

- Start with the arm in a clear area with nothing it can collide with.
- The default servo gains/limits in `src/robot_driver.py`
  (`acceleration_limit=20`, `velocity_limit=500`) are deliberately
  conservative; raise them only after verifying the tracked motion looks
  correct in the IK overlay.
- Keep a hand near `s` (standby) or `ESC` while testing.

---

## Dataset

Trained on **InterHand2.6M** (ECCV 2020) — the first large-scale real-captured dataset with accurate 3D interacting hand poses: 2.6 M labelled frames, 80–140 cameras per capture, 21 keypoints per hand. The exact 21-joint layout from that dataset is used throughout this project.

---

## Applications

Teleoperated robotic manipulation, imitation learning from human demonstrations, telesurgery, VR / AR input, and accessible gesture control.

---

## Acknowledgements

- Moon et al., *InterHand2.6M* (ECCV 2020)
- Yang et al., *Depth Anything V2* (NeurIPS 2024)
- Casiez et al., *1 Euro Filter* (CHI 2012)
- MMPose, OpenMMLab
- LLM Assist : Understanding of concepts, code syntax, code structuring, integration strategy, report structuring, correcting grammatical issues, README.md
