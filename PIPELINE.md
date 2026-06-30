# Bin-Picking Pipeline — Capture → Pose

Non-ROS camera path (pyrealsense2 wrapper directly, no `RobotNode`/ROS2 involved
in capture or vision). Single-shot per pick cycle. Designed for realtime use —
no unnecessary recomputation or offline-tool overhead in the runtime path.

Model reference: `model_6ch_api.md` (6-channel BGRXYZ YOLO26n-seg).

## 1. Capture

- `RealSenseCamera` from `bin_picking/camera/camera.py` (not the `nn/camera.py`
  duplicate on the drive — that one lacks the shared-memory streaming, irrelevant
  here since we're single-shot, but the `bin_picking` version is the canonical one).
- Depth: `cam.get_depth(num_frames=9)`, temporally median-filtered via
  `ImageProcessing.median_filtering_over_time`.
  - Kept at 9 frames deliberately. The RealSense's raw depth output is noisy
    enough that this temporal averaging is load-bearing, not just a nice-to-have.
    It is very likely the single largest latency cost in the whole pipeline.
    If a cycle-time budget ever forces a tradeoff, the lever to pull is the
    guided-filter parameters or this frame count — not removing temporal
    averaging outright, given the sensor's known noise floor.
- Color: `cam.get_color(num_frames=1)`, single frame.
- Output: one aligned `(depth, color)` pair.

## 2. Preprocess

Builds the model's 6-channel `BGRXYZ` input.

1. **Guided filter on depth** — `ImageProcessing.apply_guided_filter(depth_m, color)`
   (`bin_picking/common/image_processing.py`), Sobel-edge color-guided smoothing.
   Already vectorized OpenCV, no python loops.
2. **Unproject to X, Y, Z** — standard pinhole projection, vectorized over the
   whole image:
   - `X = (u - cx) * Z / fx`, `Y = (v - cy) * Z / fy`, `Z = depth`
   - The `(u, v)` meshgrid / undistort ray map depends only on resolution and
     intrinsics, **built once at startup and cached**, never recomputed per frame.
3. **Normalize to [0, 255]**:
   - `Z`: `log1p` transform, then min-max scale.
   - `X, Y`: plain min-max scale (no log — can be negative pre-scale).
4. **Stack**: `[B, G, R, X, Y, Z]` → single `(H, W, 6)` array, plain BGR order
   (no RGB conversion — see stage 3).

Explicitly dropped from the old (pre-model-change) pipeline: the
gradient-magnitude channel and its Open3D point-cloud round-trip
(`compute_gradient`'s `voxel_down_sample`/`estimate_normals`). Not part of the
new 6-channel spec, and was exactly the kind of overhead to avoid in the
realtime path.

Known risk: per-frame min-max normalization means a single outlier pixel
(specular hit, sensor dropout) sets that frame's scale. Mitigated in practice —
the pick surface was deliberately roughened to reduce specular reflection, so
this is a smaller concern than it would be for a flat/shiny object.

## 3. YOLO inference

- Model loaded **once at startup**: `YOLO('runs/segment/train-2/weights/best.pt')`
  (YOLO26n-seg, end2end/NMS-free head, nc=2: `Chip_rot`, `Chip_green`).
- Call: `model(img6, imgsz=1280, rect=True, retina_masks=True)`.
- **Feed the array as `[0, 255]`, as-is — no `/255` division, no BGR→RGB swap.**
  Verified two independent ways:
  - Reading `engine/predictor.py::preprocess()` in the patched ultralytics fork
    (`/home/fabian/venv`): it special-cases `shape[-1] == 3` (BGR→RGB swap) and
    `shape[-1] == 5` (partial swap, old 5-channel model), but has **no branch
    for `shape[-1] == 6`** — a 6-channel array passes through completely
    unmodified, which happens to match the doc's required `BGR` (not RGB) order.
    There is also no `/255` anywhere on this code path (the only `/255` in the
    whole package is in the training dataloader's tensor-input fallback, not
    used here).
  - Empirically verified by hooking the model's `forward()` and inspecting the
    actual input tensor: `min=0.0, max=255.0`.
- This reliance is on an **omission**, not a documented feature of the fork. It
  is considered low-risk here specifically because: the venv is a frozen,
  hand-patched library with no CI/CD and no auto-upgrade path — nobody is going
  to silently "complete" the channel-swap logic underneath this project. Still
  worth a one-line code comment at the call site stating this explicitly, so a
  future edit to that file doesn't break it by accident.

## 4. Filter masks

Per detection: `mask (H,W) bool`, `conf`, `cls_id`, `cls_name`.

- Confidence threshold + percentile filtering (`filter_masks_by_confidence` /
  `filter_masks_by_percentile` pattern from `nn/yolo_output_eval.py`).
  **Thresholds are placeholders** — training is still in progress
  (epoch 220: mAP50-95 = 0.732, mAP50 = 0.935). Final threshold tuning happens
  against the converged checkpoint, not the current one — thresholds picked
  against an under-trained model would need re-tuning anyway.
- **Mask-IoU dedup** — net new requirement vs. the old pipeline, since the
  end2end/NMS-free head doesn't deduplicate overlapping detections itself.
  Written from scratch (greedy, by confidence) — nothing existed for this on
  the drive.

## 5. Verify mask

Gap in the old pipeline: `check_area()` in `nn/yolo_output_eval.py` was a
`pass` stub waiting on `nn/psr_test.py`'s area-estimation/optimization work.

Since the model changed, the gradient-based depth-discontinuity splitting
(`gradient_magnitude_3d` / `split_by_gradient` / per-segment loop) is dropped
entirely — there is no `grad_threshold` hyperparameter anymore. Each mask is
treated as a single segment.

1. Reuse the **X, Y, Z already computed in stage 2** — no recomputation.
2. Stamp the mask onto X/Y/Z: `points = stack([X,Y,Z], axis=-1)[mask]` → `(N,3)`.
3. Single PCA plane fit with iterative inlier rejection (`_svd_plane` +
   `_plane_inlier_mask`, 2 passes, from `surface_area_from_pca` in
   `nn/psr_test.py`) → surface normal, center, refined inlier points.
4. Robust area estimate via the percentile-radius method (handles both
   filled-disk and ring shapes).
5. Compare against `real_areas.json[cls_name]` within a tolerance →
   accept/reject. **Both the real areas and the tolerance are placeholders**,
   pending training completion.

Known risk: the old gradient-split step specifically protected against a mask
spanning two physically different surfaces (e.g. partial occlusion by another
chip or the bin wall at a different depth). The single-plane 3-sigma inlier
rejection gives partial protection (one outlier population around one
dominant plane) but won't cleanly separate two comparably-sized clusters at
different depths. Whether this matters depends on how clean the new model's
per-instance masks turn out to be once fully trained — not yet established.

The PCA fit here (normal + center) is **reused directly in stage 6** — computed
once per accepted mask, not recomputed for the pose step.

## 6. Calculate gripper pose

`calculate_grip_transformation(surface_normal, center)`
(`nn/yolo_output_eval.py`), using stage 5's PCA output directly:

1. Arbitrary support vector `[1,0,0]`, cross with `surface_normal` → `y_axis`,
   cross `y_axis` with `surface_normal` → `x_axis`.
2. Flip `surface_normal` if it doesn't face the camera
   (`normal · (center - origin) < 0`).
3. Re-check right-handedness via `det([x_axis, y_axis, normal])`, flip `y_axis`
   if needed.
4. Assemble 4x4 `T`: rotation columns `[x_axis, y_axis, normal]`,
   translation = `center`.

Confirmed assumptions:
- Approach is straight down, along the surface normal. Any standoff/offset
  motion (approach point above the surface, then descend) is handled by the
  robot motion layer, outside this pose-calculation step.
- The arbitrary in-plane (`x_axis`/`y_axis`) orientation is fine because the
  gripper is rotationally symmetric — only the approach axis (the normal)
  actually matters.

## Outstanding items before production use

1. **Confidence/IoU thresholds (stage 4)** and **real areas + tolerance
   (stage 5)** are placeholders. Must be tuned against the fully converged
   model checkpoint, not the current epoch-220 one.
2. **Mask quality re-check**: once training converges, verify empirically that
   dropping the gradient-discontinuity split (stage 5) doesn't let through
   masks spanning two depth-discontinuous surfaces. If the new segmentation
   model's masks are clean per-instance, this is moot; if not, may need a
   lightweight discontinuity check reintroduced.
3. **Fork fragility note**: the 6-channel inference path depends on
   `engine/predictor.py` in `/home/fabian/venv`'s ultralytics fork *not* having
   a `shape[-1]==6` branch and *not* dividing by 255. Low risk given no
   CI/CD and a frozen, hand-patched venv, but should be documented with a code
   comment at the call site so a future edit to that file doesn't silently
   break inference.
