from __future__ import annotations

import numpy as np


def _ransac_circle_3d(bnd_proj: np.ndarray, normal: np.ndarray, p0: np.ndarray,
                      radius_mm: float, n_iter: int = 300,
                      inlier_thresh_mm: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    '''
    RANSAC circle fit with known radius on a planar 3D point set.
    Samples 2 points, computes both candidate centers (the two solutions of two
    intersecting circles of radius r), counts inliers, returns best center + mask.
    '''
    rng = np.random.default_rng(0)
    N   = len(bnd_proj)
    n   = normal / np.linalg.norm(normal)
    best_c, best_mask, best_n = None, None, 0

    for _ in range(n_iter):
        i, j = rng.choice(N, size=2, replace=False)
        A, B = bnd_proj[i], bnd_proj[j]
        ab   = B - A
        d2   = float(ab @ ab)
        if d2 < 1e-6 or d2 > (2 * radius_mm) ** 2:
            continue
        h2 = radius_mm ** 2 - d2 / 4.0
        if h2 < 0:
            continue
        mid  = (A + B) * 0.5
        perp = np.cross(n, ab)
        pl   = np.linalg.norm(perp)
        if pl < 1e-6:
            continue
        perp /= pl
        for sign in (1.0, -1.0):
            c_cand  = mid + sign * np.sqrt(h2) * perp
            c_cand -= ((c_cand - p0) @ n) * n
            dist    = np.linalg.norm(bnd_proj - c_cand, axis=1)
            inliers = np.abs(dist - radius_mm) < inlier_thresh_mm
            n_in    = int(inliers.sum())
            if n_in > best_n:
                best_n, best_c, best_mask = n_in, c_cand.copy(), inliers

    if best_c is None:
        raise ValueError('RANSAC found no valid circle candidate')
    return best_c, best_mask


def fit_circle_3d_icp(points: np.ndarray, mask: np.ndarray,
                      normal: np.ndarray, radius_mm: float,
                      max_iter: int = 30, tol_mm: float = 0.05,
                      inlier_thresh_mm: float = 3.0) -> np.ndarray:
    '''
    Fit a circle of known radius to the 3D boundary of a partial mask.

    1. Extract contour pixels → 3D positions → project onto fitted plane.
    2. RANSAC (known radius) for a robust initial center, discarding straight
       cut-edge points that appear on partially occluded chips.
    3. ICP refinement on the RANSAC inlier set.

    Returns the 3D center on the chip plane in camera frame (mm).
    '''
    import cv2

    n = normal.astype(np.float64)
    n /= np.linalg.norm(n)

    chip_pts = points[mask].astype(np.float64)
    chip_pts = chip_pts[~np.isnan(chip_pts).any(axis=1)]
    if len(chip_pts) < 4:
        raise ValueError(f'Too few valid chip points: {len(chip_pts)}')
    p0 = chip_pts.mean(axis=0)

    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError('No contour found in mask')
    px = max(contours, key=cv2.contourArea).squeeze()
    if px.ndim < 2 or len(px) < 5:
        raise ValueError(f'Too few contour pixels: {len(px)}')

    u   = np.clip(px[:, 0], 0, points.shape[1] - 1)
    v   = np.clip(px[:, 1], 0, points.shape[0] - 1)
    bnd = points[v, u].astype(np.float64)
    bnd = bnd[~np.isnan(bnd).any(axis=1)]
    if len(bnd) < 5:
        raise ValueError(f'Too few valid 3D contour points: {len(bnd)}')

    bnd_proj = bnd - np.outer((bnd - p0) @ n, n)

    c, inlier_mask = _ransac_circle_3d(bnd_proj, n, p0, radius_mm,
                                       inlier_thresh_mm=inlier_thresh_mm)
    pts = bnd_proj[inlier_mask] if inlier_mask.sum() >= 3 else bnd_proj

    for _ in range(max_iter):
        d    = pts - c
        d    = d - np.outer(d @ n, n)
        dist = np.linalg.norm(d, axis=1)
        ok   = dist > 1e-6
        if ok.sum() < 2:
            break
        d_hat = d[ok] / dist[ok, np.newaxis]
        c_new = np.mean(pts[ok] - radius_mm * d_hat, axis=0)
        c_new -= ((c_new - p0) @ n) * n
        if np.linalg.norm(c_new - c) < tol_mm:
            c = c_new
            break
        c = c_new

    return c
