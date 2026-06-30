from __future__ import annotations

import numpy as np
import open3d as o3d

from bin_picking.common.image_processing import ImageProcessing

# Keyed on (intrinsics, resolution) so the per-pixel ray directions are computed
# once and reused across every shot -- never recomputed per frame.
_RAY_CACHE: dict[tuple, np.ndarray] = {}


def _get_rays(intrinsics: dict, h: int, w: int) -> np.ndarray:
    '''Normalized (x', y') view ray per pixel, cached per (intrinsics, resolution).'''
    key = (intrinsics['fx'], intrinsics['fy'], intrinsics['ppx'], intrinsics['ppy'], h, w)
    cached = _RAY_CACHE.get(key)

    if cached is not None:
        return cached

    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    rays = np.empty((h, w, 2), dtype=np.float32)
    rays[..., 0] = (u - intrinsics['ppx']) / intrinsics['fx']
    rays[..., 1] = (v - intrinsics['ppy']) / intrinsics['fy']
    _RAY_CACHE[key] = rays

    return rays


def unproject(depth: np.ndarray, intrinsics: dict) -> np.ndarray:
    '''Pinhole back-projection to camera-frame X/Y/Z, same unit as `depth`. HxWx3, NaN where invalid.'''
    h, w = depth.shape
    rays = _get_rays(intrinsics, h, w)
    depth = depth * 1000 # Cast back to mm.

    points = np.empty((h, w, 3), dtype=np.float32)
    points[..., 0] = rays[..., 0] * depth
    points[..., 1] = rays[..., 1] * depth
    points[..., 2] = depth
    invalid = ~np.isfinite(depth) | (depth <= 0)
    points[invalid] = np.nan

    return points


def _normalize_minmax(arr: np.ndarray) -> np.ndarray:
    valid = np.isfinite(arr)

    if not valid.any():
        return np.zeros(arr.shape, dtype=np.float32)

    lo, hi = arr[valid].min(), arr[valid].max()
    span = max(hi - lo, 1e-6)
    out = (arr - lo) / span * 255.0

    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def _normalize_z(z: np.ndarray) -> np.ndarray:
    # Invalid pixels (NaN/inf) become 0 before log1p, matching depth.h5's true
    # zero-pixels in training -- so lo=log1p(0)=0 here too, instead of being
    # excluded from min/max and skewing lo/span against the trained model.
    z_clean = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z_log = np.log1p(np.clip(z_clean, a_min=0.0, a_max=None))
    return _normalize_minmax(z_log)


def build_bgrxyz(color: np.ndarray, points: np.ndarray) -> np.ndarray:
    '''
    Stacks [B, G, R, X, Y, Z] into a single (H, W, 6) array, each channel scaled to [0, 255].

    Channel order matches the model's expected BGR (not RGB) -- see model_6ch_api.md
    and PIPELINE.md stage 3 for why no RGB conversion happens anywhere in this pipeline.
    '''
    img6 = np.empty((*color.shape[:2], 6), dtype=np.float32)
    img6[..., 0:3] = color.astype(np.float32)
    img6[..., 3] = _normalize_minmax(points[..., 0])
    img6[..., 4] = _normalize_minmax(points[..., 1])
    img6[..., 5] = _normalize_z(points[..., 2])

    return img6


def preprocess(color: np.ndarray, depth_m: np.ndarray, intrinsics: dict,
                processor: ImageProcessing) -> tuple[np.ndarray, np.ndarray]:
    '''
    Args:
        color:      HxWx3 BGR uint8.
        depth_m:    HxW float32 depth in meters (already temporally filtered) -- meters
                    because ImageProcessing.apply_guided_filter's tuned cfg (guided_eps,
                    residual_clamp) is calibrated in meters; converted to millimeters
                    below, aftedepth_mr filtering, since that's what the model and the rest of
                    this codebase's robot-frame convention (see get_poses()) expect.
        intrinsics: dict with fx, fy, ppx, ppy.
        processor:  ImageProcessing instance (guided filter).

    Returns:
        (img6, points): the model's [0,255]-range 6-channel input, and the raw
        (unnormalized, millimeters) HxWx3 point cloud -- reused downstream by mask
        verification and pose calculation instead of being recomputed.
    '''
    depth_smooth_m = processor.apply_guided_filter(depth_m, color)
    points = unproject(depth_smooth_m, intrinsics)
    img6 = build_bgrxyz(color, points)
    return img6, points
