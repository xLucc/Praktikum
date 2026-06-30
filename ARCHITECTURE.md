# Bin Picking Pipeline — Architecture

## Module Overview

The pipeline is split into focused modules under `bin_picking/pipeline/`:

| Module | Contents |
|---|---|
| `config.py` | Constants, `PickResult` dataclass, `load_cfg()` |
| `motion.py` | All robot motion and camera capture |
| `visualize.py` | Matplotlib/OpenCV pick visualization |
| `occlusion.py` | 3D ICP circle fit for occluded chips |
| `grasp.py` | Grasp search logic (`find_grasp` and helpers) |
| `run.py` | Single-shot entry point (`main()`) |
| `loop.py` | Continuous pick loop entry point (`main()`) |

---

## Module Details

### `config.py`
Global constants and configuration loading.

- `TCP_OFFSET` — tool-tip offset from the flange (TFC) frame in TFC-local coordinates (mm). Pivot-calibrated; updated by `test/calibrate_tcp.py`.
- `TCP_TUBE_END_MM` — suction tube tip distance from TFC along tool Z (mm).
- `MAX_APPROACH_ANGLE_DEG` — maximum allowed deviation of the approach axis from camera +Z. Catches degenerate plane fits before any motion is computed.
- `PickResult` — dataclass holding `transform`, `class_name`, `confidence`, `area`.
- `load_cfg()` — loads all JSON config files and the IK chain into a single dict.

---

### `motion.py`
All robot motion and camera capture. No grasp logic.

- `setup_robot()` — initialises ROS2, spins node in background thread, sets TCP offset and jogging frame.
- `move_home(robot, home_pose)` — joint-space PTP to home/viewing pose, blocks until arrival.
- `move_to_grasp(robot, hand_eye, T_camera_target, bin_cfg)` — two-step Cartesian approach: hover above bin rim, then drop to grasp Z.
- `retreat_from_grasp(robot, bin_cfg, drop_pos)` — three-step retreat: lift until suction tip clears bin rim → rotate to world -Z → translate to drop position.
- `deliver_chip(robot, class_name, bin_cfg, motion_cfg)` — calls `retreat_from_grasp` to the class-specific drop location, then releases digital output 6.
- `capture(cam, processor)` — 11-frame temporal-median depth + 1 color frame; clips depth to 100–1000 mm range.
- `_poll_until_close(get_current, target, tol, robot, timeout, poll_interval)` — polls a pose getter until within `tol` of target or timeout; aborts on robot halt/collision signal.

---

### `visualize.py`
Spawns a detached subprocess for non-blocking Matplotlib visualization.

- `start_visualization(color, points, detections, rejected, result)` — kills any prior viz process, spawns a new one.
- `stop_visualization()` — terminates the running viz process.
- `visualize_pick(...)` — runs in the subprocess: draws mask overlays on the color image (green=winner, orange=candidate, gray×=rejected) and a 3D scatter of point cloud with grasp arrow. Saves to `plots/pick_<timestamp>.png`.

---

### `occlusion.py`
Pure NumPy, no ROS dependencies. Used when a chip is only partially visible.

- `_ransac_circle_3d(bnd_proj, normal, p0, radius_mm)` — RANSAC circle fit with known radius on a planar 3D boundary. Samples 2 points, solves for both candidate centers via perpendicular bisector, counts inliers within `inlier_thresh_mm`. Returns best center + inlier mask. Robust to straight cut-edge points from occlusion boundaries.
- `fit_circle_3d_icp(points, mask, normal, radius_mm)` — full pipeline:
  1. Extract contour pixels → 3D positions → project onto chip plane.
  2. RANSAC init (discards straight cut-edge points that appear on partially occluded chips).
  3. ICP refinement on inliers: `c_new = mean(p_i − r · (p_i−c)/|p_i−c|)`, constrained to the chip plane.
  Returns the 3D center on the chip plane in camera frame (mm).

---

### `grasp.py`
Core pick logic. Consumes all other pipeline modules.

- `find_grasp(cam, processor, intrinsics, cfg, robot, hand_eye)` — main entry point:
  1. Capture → preprocess → inference → filter detections.
  2. Sort by median Z (closest chip first).
  3. For each detection: verify area, clamp approach angle, check reachability.
  4. On reachability failure: search alternative approaches via `_find_reachable_approach`.
  5. After main loop: ICP fallback for any under-area (occluded) candidates.
  Returns the first valid `PickResult`, or `None`.

- `_apply_approach_clamp(transform)` — clamps the approach axis to `MAX_APPROACH_ANGLE_DEG` from camera +Z. Returns `None` if the axis is degenerate (catches merged-chip plane fits).

- `_find_reachable_approach(T_world, T_cam_world, cfg)` — searches for a reachable approach when the default orientation is blocked. Tries straight-down (world −Z) and original elevation, each at 8 azimuths × 45°. Azimuth is a free DOF for round suction-cup chips. Returns the first reachable transform in camera frame, or `None`.

- `_save_reachability_sample(transform, reachable, bin_cfg)` — appends every evaluated transform with its reachability result to `/tmp/reachability_samples.npy` for offline analysis.

---

### `run.py`
Single-shot entry point. Calls `find_grasp` once, executes one pick cycle, then shuts down.

```
move_home → find_grasp → move_to_grasp → suction on → deliver_chip → move_home
```

### `loop.py`
Continuous entry point. Same pick cycle as `run.py`, repeated until `Ctrl+C`.

---

## Test Utilities

Located under `bin_picking/test/`:

| File | Purpose |
|---|---|
| `calibrate_bin.py` | Interactive: touch 4 bin corners, recomputes and overwrites `bin_cfg.json`. |
| `calibrate_tcp.py` | Pivot TCP calibration (≥5 poses). Updates `TCP_OFFSET` in `config.py` in-place. |
| `hover_bin.py` | Moves to all 4 bin corners + center bottom for bin position verification. |

---

## Data Files

All under `data/`:

| Path | Contents |
|---|---|
| `cfg/filter_cfg.json` | Detection confidence threshold, area tolerance, IoU threshold, percentile filter. |
| `cfg/real_areas.json` | Reference chip areas in mm² (π × r²). Used for area acceptance and ICP radius. |
| `cfg/bin_cfg.json` | Bin world position and dimensions. Updated by `calibrate_bin.py`. |
| `cfg/home_pose.json` | Joint-space home/viewing pose. |
| `cfg/motion_cfg.json` | Drop positions per class. |
| `cfg/high_density.json` | RealSense advanced depth preset. |
| `calibration/intrinsics.hdf5` | Camera intrinsics (fx, fy, cx, cy). |
| `calibration/hand_eye.hdf5` | Hand-eye calibration matrix T_cam2tfp (camera → TFC frame). |
