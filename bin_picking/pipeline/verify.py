from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.spatial import Delaunay, QhullError

logger = logging.getLogger(__name__)


def stamp_mask(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    '''Pulls the masked pixels out of an HxWx3 point cloud, dropping invalid (NaN) points.'''
    pts = points[mask]
    return pts[~np.isnan(pts).any(axis=1)]


def _svd_plane(points: np.ndarray):
    center = points.mean(axis=0)
    centered = points - center
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    return center, vt[2], s, vt


def _plane_inlier_mask(points: np.ndarray, center: np.ndarray, normal: np.ndarray,
                        s: np.ndarray, sigma_scale: float) -> np.ndarray:
    dist = np.abs((points - center) @ normal)
    sigma_normal = float(s[2]) / np.sqrt(max(points.shape[0] - 1, 1))
    return dist < sigma_scale * sigma_normal


def fit_plane(points: np.ndarray, sigma_scale: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    '''
    Two-pass SVD plane fit with inlier rejection.
    Returns (normal, center) in the same frame as `points`.
    '''
    if points.shape[0] < 6:
        raise ValueError(f'Too few valid points to fit a plane: {points.shape[0]}')

    center0, normal0, s0, _ = _svd_plane(points)
    mask0 = _plane_inlier_mask(points, center0, normal0, s0, sigma_scale)
    inliers = points[mask0] if mask0.sum() >= 6 else points

    center, normal, s, _ = _svd_plane(inliers)
    mask1 = _plane_inlier_mask(inliers, center, normal, s, sigma_scale)
    inliers = inliers[mask1] if mask1.sum() >= 4 else inliers

    center, normal, _, _ = _svd_plane(inliers)
    return normal, center


def _area_delaunay(points: np.ndarray, sigma_scale: float = 3.0) -> float:
    '''
    Estimate surface area by Delaunay-triangulating the inlier point cloud projected
    onto the fitted plane, then summing the 3-D triangle areas.
    Overestimates for non-convex shapes (fills convex hull).
    '''
    if points.shape[0] < 6:
        raise ValueError(f'Too few valid points for Delaunay area: {points.shape[0]}')

    center0, normal0, s0, _ = _svd_plane(points)
    mask0 = _plane_inlier_mask(points, center0, normal0, s0, sigma_scale)
    inliers = points[mask0] if mask0.sum() >= 6 else points

    center, normal, s, vt = _svd_plane(inliers)
    mask1 = _plane_inlier_mask(inliers, center, normal, s, sigma_scale)
    inliers = inliers[mask1] if mask1.sum() >= 4 else inliers

    projected = (inliers - center) @ vt[:2].T
    try:
        tri = Delaunay(projected)
    except QhullError as e:
        raise ValueError(f'Could not triangulate mask points for area estimation: {e}')

    verts = inliers[tri.simplices]
    ab = verts[:, 1] - verts[:, 0]
    ac = verts[:, 2] - verts[:, 0]
    return float(0.5 * np.sum(np.linalg.norm(np.cross(ab, ac), axis=1)))


def _area_per_pixel(points: np.ndarray, mask: np.ndarray,
                    normal: np.ndarray, intrinsics: dict) -> float:
    '''
    Per-pixel real-world area.

    For each valid masked pixel with depth d (mm), the projected pixel footprint
    is d²/(fx·fy). Dividing by cos_tilt = |normal · camera_Z| = |normal[2]| corrects
    for surface inclination. Points are in camera frame (mm), so normal[2] is the
    component along the optical axis — no additional transform needed.
    '''
    fx, fy = intrinsics['fx'], intrinsics['fy']
    d = points[mask, 2]
    valid = np.isfinite(d) & (d > 0)
    d = d[valid]
    if d.size == 0:
        raise ValueError('No valid depth pixels in mask for per-pixel area')

    projected_area = float(np.sum(d ** 2)) / (fx * fy)

    cos_tilt = abs(float(normal[2]))
    cos_tilt = max(cos_tilt, 0.15)   # clamp: don't blow up beyond ~81° tilt

    return projected_area / cos_tilt




def verify_mask(points: np.ndarray, mask: np.ndarray, class_name: str, real_areas: dict,
                tolerance: float,
                intrinsics: Optional[dict] = None) -> tuple[bool, float, Optional[np.ndarray], Optional[np.ndarray]]:
    '''
    Estimates the masked region's real-world surface area and compares it against
    the known reference area for `class_name`.

    When `intrinsics` is provided the area is computed per-pixel (d²/(fx·fy) summed
    over valid masked pixels, corrected for tilt), which is more accurate than the
    Delaunay convex-hull approach for non-convex shapes. Falls back to Delaunay when
    intrinsics are not supplied.

    Returns:
        (accepted, area_mm2, surface_normal, center)
    '''
    pts = stamp_mask(points, mask)
    normal, center = fit_plane(pts)

    if intrinsics is not None:
        area = _area_per_pixel(points, mask, normal, intrinsics)
        method = 'per-pixel'
    else:
        area = _area_delaunay(pts)
        method = 'Delaunay'

    reference = real_areas.get(class_name, 0.0)

    if reference <= 0:
        logger.warning("No reference area for '%s' yet — skipping verification. "
                       "area=%.1f mm² (%s)", class_name, area, method)
        return True, area, normal, center

    rel_err = abs(area - reference) / reference
    accepted = rel_err <= tolerance
    logger.debug("area=%.1f mm²  ref=%.1f mm²  err=%.1f%%  method=%s  accepted=%s",
                 area, reference, rel_err * 100, method, accepted)
    return accepted, area, normal, center
