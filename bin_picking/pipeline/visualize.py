"""
Non-blocking pick visualisation.

Spawns a daemon subprocess (via ``start_visualization``) that renders the
detection overlays and 3-D point cloud while the main process continues with
robot motion.  The subprocess is killed before the next visualisation starts,
so only one window is ever open.

Saved plots go to ``data/../plots/pick_<unix_timestamp>.png``.
"""
from __future__ import annotations

import multiprocessing
import time
from typing import Optional

import cv2 as cv
import numpy as np

from bin_picking.common.helper import get_project_dir
from bin_picking.pipeline.config import PickResult

_viz_proc: Optional[multiprocessing.Process] = None


def stop_visualization() -> None:
    global _viz_proc
    if _viz_proc is not None and _viz_proc.is_alive():
        _viz_proc.terminate()
        _viz_proc.join(timeout=2.0)
    _viz_proc = None


def start_visualization(color: np.ndarray, points: np.ndarray,
                        detections: list, rejected: list,
                        result: PickResult) -> None:
    global _viz_proc
    stop_visualization()
    _viz_proc = multiprocessing.Process(
        target=visualize_pick,
        args=(color, points, detections, rejected, result),
        daemon=True,
    )
    _viz_proc.start()


def visualize_pick(color: np.ndarray, points: np.ndarray,
                   detections: list, rejected: list,
                   result: PickResult) -> None:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    WINNER_COLOR  = (  0, 220,  80)
    NEUTRAL_COLOR = (255, 160,  40)
    REJECT_COLOR  = (100, 100, 100)

    rejected_ids = {id(d) for d in rejected}

    def _draw_mask(img, det, color, cross=False):
        colored = np.zeros_like(img)
        colored[det['mask']] = color
        img = cv.addWeighted(img, 1.0, colored, 0.4, 0)
        contours, _ = cv.findContours(
            det['mask'].astype(np.uint8) * 255, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        cv.drawContours(img, contours, -1, color, 2)
        ys, xs = np.where(det['mask'])
        if not len(xs):
            return img
        cx, cy = int(xs.mean()), int(ys.mean())
        if cross:
            s = 18
            cv.line(img, (cx - s, cy - s), (cx + s, cy + s), color, 3, cv.LINE_AA)
            cv.line(img, (cx + s, cy - s), (cx - s, cy + s), color, 3, cv.LINE_AA)
        cv.putText(img, f"{det['class_name']} {det['confidence']:.2f}",
                   (cx, cy - 22), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)
        return img

    def _is_winner(det):
        return (id(det) not in rejected_ids and
                det['class_name'] == result.class_name and
                abs(det['confidence'] - result.confidence) < 1e-4)

    overlay = color.copy()
    for det in detections:
        if id(det) in rejected_ids:
            overlay = _draw_mask(overlay, det, REJECT_COLOR, cross=True)
        elif _is_winner(det):
            overlay = _draw_mask(overlay, det, WINNER_COLOR)
        else:
            overlay = _draw_mask(overlay, det, NEUTRAL_COLOR)

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(cv.cvtColor(overlay, cv.COLOR_BGR2RGB))
    center_3d = result.transform[:3, 3]
    pts_flat  = points.reshape(-1, 3)
    valid     = ~np.isnan(pts_flat).any(axis=1)
    dists     = np.full(pts_flat.shape[0], np.inf)
    dists[valid] = np.linalg.norm(pts_flat[valid] - center_3d, axis=1)
    cy_px, cx_px = np.unravel_index(np.argmin(dists), points.shape[:2])
    ax1.plot(cx_px, cy_px, 'o', color='black', markersize=6)
    ax1.set_title('Detections: green=winner  orange=candidate  gray✕=rejected')
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    def _flip_z(pts: np.ndarray) -> np.ndarray:
        """Mirror a (N, 3) or (3,) array at the XY plane by negating Z."""
        out = pts.copy()
        out[..., 2] = -out[..., 2]
        return out

    # --- Full scene point cloud (RGB coloured, subsampled, mirrored) ---
    pts_flat   = points.reshape(-1, 3)
    valid_mask = ~np.isnan(pts_flat).any(axis=1) & (pts_flat[:, 2] > 0)
    valid_pts  = pts_flat[valid_mask]
    if len(valid_pts) > 6000:
        sub       = np.random.choice(len(valid_pts), 6000, replace=False)
        valid_pts = valid_pts[sub]
    valid_pts = _flip_z(valid_pts)
    ax2.scatter(valid_pts[:, 0], valid_pts[:, 1], valid_pts[:, 2],
                s=4, color=(0.75, 0.75, 0.75), alpha=0.2)

    # --- Detection mask overlays on top ---
    color_map = {
        'winner':  (0.0,  0.86, 0.31),
        'neutral': (1.0,  0.63, 0.16),
        'reject':  (0.39, 0.39, 0.39),
    }
    for det in detections:
        pts   = points[det['mask']]
        valid = pts[~np.isnan(pts).any(axis=1)]
        if not len(valid):
            continue
        idx = np.random.choice(len(valid), min(400, len(valid)), replace=False)
        s   = _flip_z(valid[idx])
        if id(det) in rejected_ids:
            c, alpha = color_map['reject'], 0.4
        elif _is_winner(det):
            c, alpha = color_map['winner'], 0.9
        else:
            c, alpha = color_map['neutral'], 0.6
        ax2.scatter(s[:, 0], s[:, 1], s[:, 2], s=4, color=c, alpha=alpha)

    # --- Grasp target ---
    center   = _flip_z(result.transform[:3, 3])
    approach = _flip_z(result.transform[:3, 2])
    ax2.scatter(*center, color='red', s=120, zorder=5, label='target')
    ax2.quiver(*center, *approach, length=30.0, color='red', linewidth=2)
    ax2.set_xlabel('X (mm)'); ax2.set_ylabel('Y (mm)'); ax2.set_zlabel('Z (mm)')
    ax2.set_title(f'Grasp: {result.class_name} (conf={result.confidence:.2f}, area={result.area:.1f}mm²)')
    ax2.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    out = get_project_dir() / 'plots' / f'pick_{int(time.time())}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=120)
    try:
        plt.show(block=True)
    except Exception:
        pass
