# bin_picking

A ROS2-based bin-picking system for a Kassow 7-DOF robot arm with an Intel
RealSense depth camera.  The pipeline detects semiconductor chips lying flat in
a bin, estimates their 3-D poses, checks kinematic reachability, and picks them
one-by-one with a suction gripper, sorting them by class into class-specific
drop locations.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Hardware](#2-hardware)
3. [Directory Structure](#3-directory-structure)
4. [Installation](#4-installation)
5. [Calibration](#5-calibration)
6. [Running the Pipeline](#6-running-the-pipeline)
7. [Configuration Reference](#7-configuration-reference)
8. [Package Modules](#8-package-modules)

---

## 1. System Overview

The pick cycle runs as follows:

```
move_home
    └─► capture (11-frame temporal-median depth + 1 colour frame)
            └─► preprocess (guided-filter depth → unproject → BGRXYZ 6-channel image)
                    └─► YOLO inference (6-channel segmentation model)
                            └─► filter detections (confidence, IoU-dedup, percentile)
                                    └─► for each detection (closest Z first):
                                            verify area (SVD plane fit + per-pixel area)
                                            clamp approach angle
                                            check reachability (IK + capsule collision)
                                            ──► move_to_grasp → suction on → deliver_chip
                                        fallback: ICP circle fit for occluded chips
move_home
```

`bin_picking.pipeline.run` executes this cycle once and exits.  
`bin_picking.pipeline.loop` repeats it until interrupted with `Ctrl+C`.

For a detailed description of each stage see [PIPELINE.md](PIPELINE.md).  
For the module breakdown see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 2. Hardware

| Component | Details |
|---|---|
| Robot | Kassow KR1205 (7 DOF) |
| Camera | Intel RealSense D430 (aligned depth + colour) |
| Gripper | Suction cup on digital output 6 |
| Lighting | Ring light on analog output, brightness-controlled by `control/pid.py` |

The camera is mounted **eye-in-hand** (moves with the robot).  The robot must
return to the home/viewing pose before every capture so the camera looks down at
the bin from a known, calibrated position.

---

## 3. Directory Structure

```
bin_picking/
├── bin_picking/
│   ├── camera/
│   │   ├── camera_interface.py   # Abstract Camera base class
│   │   └── camera.py             # RealSenseCamera implementation
│   ├── common/
│   │   ├── calibrate_camera.py   # Standalone camera intrinsics calibration
│   │   ├── calibration_errors.py # InvalidRobotTransform exception
│   │   ├── ChArucoBoard_Creator.py  # Board generation utility
│   │   ├── eye_in_hand_calibration.py  # Hand-eye calibration + robot transform helpers
│   │   ├── helper.py             # Path helpers, HDF5 loaders
│   │   └── image_processing.py  # Guided-filter depth smoothing, point cloud generation
│   ├── control/
│   │   └── pid.py                # PID controller + brightness control loop
│   ├── pipeline/
│   │   ├── config.py             # Constants, PickResult, load_cfg()
│   │   ├── grasp.py              # find_grasp() — core pick logic
│   │   ├── inference.py          # YOLO model loading and inference
│   │   ├── motion.py             # Robot motion and camera capture
│   │   ├── occlusion.py          # RANSAC/ICP circle fit for occluded chips
│   │   ├── pose.py               # Grip transform from surface normal + centre
│   │   ├── postprocess.py        # Detection filtering (confidence, IoU, percentile)
│   │   ├── preprocess.py         # BGRXYZ image construction, point cloud unprojection
│   │   ├── reachability.py       # IK solve + capsule collision checking
│   │   ├── run.py                # Single-shot entry point
│   │   ├── loop.py               # Continuous pick loop entry point
│   │   ├── verify.py             # Mask area verification (SVD plane + per-pixel area)
│   │   └── visualize.py          # Non-blocking Matplotlib/OpenCV visualisation
│   ├── robot/
│   │   └── node.py               # ROS2 node wrapping the Kassow controller services
│   └── test/
│       ├── calibrate_bin.py      # Interactive: touch 4 corners → rewrite bin_cfg.json
│       ├── calibrate_tcp.py      # Pivot TCP calibration → update TCP_OFFSET in config.py
│       ├── hover_bin.py          # Move to bin corners/centre for visual verification
│       ├── camera_test.py        # Camera capture test
│       ├── light_control.py      # Ring-light PID tuning
│       ├── pickup_test.py        # Single pick without vision
│       ├── tune_pid.py           # PID parameter tuning with plots
│       └── torque_test.py        # Robot torque reading test
├── data/                         # Runtime data (not in git, created by calibration)
│   ├── cfg/                      # JSON config files (see §7)
│   ├── calibration/              # intrinsics.hdf5, hand_eye.hdf5
│   ├── urdf/                     # kr1205.urdf (for IK chain)
│   └── weights/                  # best.pt (trained YOLO weights)
├── plots/                        # Auto-saved pick and rejection visualisations
├── ARCHITECTURE.md
├── PIPELINE.md
└── README.md  ← this file
```

---

## 4. Installation

### Prerequisites

- ROS2 (tested on Humble)
- Python ≥ 3.8
- A patched `ultralytics` fork installed in `/home/fabian/venv` that supports
  6-channel input without normalisation (see [PIPELINE.md](PIPELINE.md) §3)

### Python dependencies

```bash
pip install pyrealsense2 open3d scipy ikpy h5py opencv-python
```

### Build

```bash
cd ~/praktikum
colcon build --packages-select bin_picking
source install/setup.bash
```

---

## 5. Calibration

Calibrations are stored under `data/calibration/` and must be run before the
first pick.  All calibration tools require the robot to be homed and ROS2 to be
running.

### 5.1 Camera intrinsics

Run the interactive calibration with a ChArUco board:

```bash
ros2 run bin_picking eye_in_hand_calibration --cam
```

Or reuse the existing `data/buf.hdf5` capture buffer (skip robot motion):

```bash
ros2 run bin_picking eye_in_hand_calibration --skip --cam
```

Writes: `data/calibration/intrinsics.hdf5`

### 5.2 Hand-eye calibration

Moves the robot through 22 pre-defined poses while a ChArUco board is visible
and runs OpenCV `calibrateHandEye`:

```bash
ros2 run bin_picking eye_in_hand_calibration
```

Writes: `data/calibration/hand_eye.hdf5`

### 5.3 TCP (tool-centre point)

Touch the same physical point from ≥5 different orientations:

```bash
python -m bin_picking.test.calibrate_tcp
```

Updates `TCP_OFFSET` in `bin_picking/pipeline/config.py` in-place.

### 5.4 Bin position

Touch the 4 inner corners of the bin interactively:

```bash
python -m bin_picking.test.calibrate_bin
```

Overwrites `data/cfg/bin_cfg.json`.

---

## 6. Running the Pipeline

### Single pick

```bash
ros2 run bin_picking run
```

### Continuous loop

```bash
ros2 run bin_picking loop
```

Both entry points require ROS2 to be running and the Kassow controller to be
connected and homed.

### Verification / test scripts

```bash
# Verify bin corners are correct
python -m bin_picking.test.hover_bin

# Manual pick without vision
python -m bin_picking.test.pickup_test

# Camera capture sanity check
python -m bin_picking.test.camera_test
```

---

## 7. Configuration Reference

All config files live under `data/cfg/`.

| File | Key fields | Description |
|---|---|---|
| `filter_cfg.json` | `confidence`, `iou_threshold`, `percentile`, `area_tolerance`, `pick` | Detection filtering thresholds. `pick` optionally restricts picking to one class name. |
| `real_areas.json` | `{"Chip_rot": <mm²>, "Chip_green": <mm²>}` | Reference surface areas per class (π·r²). Used for area acceptance and ICP fallback radius. |
| `bin_cfg.json` | `world_pos`, `size` | Bin centre in world frame (mm) and dimensions [W, D, H] (mm). Updated by `calibrate_bin.py`. |
| `home_pose.json` | `joint_config` | 7 joint angles (degrees) for the camera viewing pose. |
| `motion_cfg.json` | `drop_pose`, `class_drop_poses` | Drop position per chip class (world frame, mm). |
| `processing_cfg.json` | `guided_radius`, `guided_eps`, `sharp_lambda`, `residual_clamp` | Guided-filter depth smoothing parameters. |
| `high_density.json` | *(RealSense preset)* | Advanced depth preset for close-range, high-density measurement. |

---

## 8. Package Modules

### `camera/`

| Module | Purpose |
|---|---|
| `camera_interface.py` | Abstract `Camera` base class (`get_color`, `get_depth`). |
| `camera.py` | `RealSenseCamera` — pyrealsense2 wrapper with optional depth-to-colour alignment, advanced preset loading, and shared-memory streaming mode. |

### `common/`

| Module | Purpose |
|---|---|
| `image_processing.py` | `ImageProcessing` — guided-filter depth smoothing; temporal median filtering; Open3D point cloud generation. |
| `eye_in_hand_calibration.py` | Hand-eye calibration workflow; `get_robot_transform()` used at runtime to read the current TCP pose. |
| `helper.py` | `get_project_dir()`, HDF5 intrinsics/hand-eye loaders, JSON helpers. |
| `calibration_errors.py` | `InvalidRobotTransform` exception raised when the robot reports a zero or None pose. |

### `control/`

| Module | Purpose |
|---|---|
| `pid.py` | `PID` dataclass with anti-windup; `ControlLoop` — image-brightness feedback controller for the ring light. |

### `pipeline/`

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full per-module breakdown.

### `robot/`

| Module | Purpose |
|---|---|
| `node.py` | `RobotNode` — ROS2 node wrapping all Kassow controller services (linear move, joint move, digital/analog I/O, pose queries). Must be spun in a background thread. |
