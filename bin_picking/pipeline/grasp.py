from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from bin_picking.camera.camera import RealSenseCamera
from bin_picking.common.eye_in_hand_calibration import get_robot_transform
from bin_picking.common.image_processing import ImageProcessing
from bin_picking.pipeline.config import MAX_APPROACH_ANGLE_DEG, PickResult
from bin_picking.pipeline.inference import run_inference
from bin_picking.pipeline.motion import capture
from bin_picking.pipeline.occlusion import fit_circle_3d_icp
from bin_picking.pipeline.pose import calculate_grip_transformation
from bin_picking.pipeline.postprocess import filter_masks, results_to_masks
from bin_picking.pipeline.preprocess import preprocess
from bin_picking.pipeline.reachability import is_reachable
from bin_picking.pipeline.verify import verify_mask
from bin_picking.pipeline.visualize import start_visualization
from bin_picking.robot.node import RobotNode

logger = logging.getLogger(__name__)

_REACH_LOG = '/tmp/reachability_samples.npy'


def _save_reachability_sample(transform: np.ndarray,
                               reachable: bool, bin_cfg: dict) -> None:
    try:
        existing = list(np.load(_REACH_LOG, allow_pickle=True))
    except Exception:
        existing = []
    existing.append({'transform': transform, 'reachable': reachable, 'bin_cfg': bin_cfg})
    np.save(_REACH_LOG, existing)


def _apply_approach_clamp(transform: np.ndarray) -> Optional[np.ndarray]:
    '''Clamp approach axis to MAX_APPROACH_ANGLE_DEG. Returns None if degenerate.'''
    approach_axis = transform[:3, 2]
    angle = np.degrees(np.arccos(np.clip(approach_axis @ np.array([0., 0., 1.]), -1., 1.)))
    if angle <= MAX_APPROACH_ANGLE_DEG:
        return transform
    rot_axis = np.cross(approach_axis, np.array([0., 0., 1.]))
    norm     = np.linalg.norm(rot_axis)
    if norm < 1e-6:
        return None
    correction = np.radians(angle - MAX_APPROACH_ANGLE_DEG)
    R_corr     = Rotation.from_rotvec(-correction * rot_axis / norm).as_matrix()
    out        = transform.copy()
    out[:3, :3] = R_corr @ transform[:3, :3]
    logger.info('Clamped approach angle %.1f° → %.1f°', angle, MAX_APPROACH_ANGLE_DEG)
    return out


def _find_reachable_approach(T_world: np.ndarray, T_cam_world: np.ndarray,
                              cfg: dict) -> Optional[np.ndarray]:
    '''
    Search for a reachable approach when the default orientation is blocked.

    Tries (in order):
      1. Straight-down approach (world -Z) at 8 azimuths every 45°.
      2. Current elevation at 8 azimuths every 45°.

    Returns the first reachable transform in camera frame, or None.
    '''
    pos = T_world[:3, 3]

    tool_z   = T_world[:3, 2]
    world_nz = np.array([0., 0., -1.])
    cross    = np.cross(tool_z, world_nz)
    sin_a    = np.linalg.norm(cross)
    cos_a    = float(tool_z @ world_nz)
    R_align  = (Rotation.from_rotvec(cross / sin_a * np.arctan2(sin_a, cos_a)).as_matrix()
                if sin_a > 1e-6 else np.eye(3))
    R_down   = R_align @ T_world[:3, :3]

    T_cam_world_inv = np.linalg.inv(T_cam_world)

    for base_R in (R_down, T_world[:3, :3]):
        for az in range(0, 360, 45):
            R_az  = Rotation.from_euler('z', az, degrees=True).as_matrix()
            T_try = np.eye(4)
            T_try[:3, :3] = R_az @ base_R
            T_try[:3,  3] = pos
            if is_reachable(cfg['chain'], T_try, cfg['home_pose'], cfg['bin']):
                label = 'straight-down' if np.array_equal(base_R, R_down) else 'original'
                logger.info('Found reachable approach: base=%s az=%d°', label, az)
                return T_cam_world_inv @ T_try

    return None


def find_grasp(cam: RealSenseCamera, processor: ImageProcessing, intrinsics: dict,
               cfg: dict, robot: RobotNode,
               hand_eye: np.ndarray) -> Optional[PickResult]:
    color, depth_m = capture(cam, processor)
    img6, points   = preprocess(color, depth_m, intrinsics, processor)

    results    = run_inference(img6)
    detections = results_to_masks(results)

    pick_class = cfg['filter'].get('pick')
    if pick_class is not None:
        detections = [d for d in detections if d['class_name'] == pick_class]

    detections = filter_masks(
        detections,
        conf=cfg['filter']['confidence'],
        alpha=cfg['filter']['percentile'],
        iou_threshold=cfg['filter']['iou_threshold'],
    )

    def _median_z(det):
        z     = points[det['mask'], 2]
        valid = z[np.isfinite(z)]
        return np.median(valid) if valid.size > 0 else np.inf

    rejected: list[dict]                          = []
    occluded_cands: list[tuple[dict, np.ndarray, float]] = []

    for det in sorted(detections, key=_median_z):
        try:
            accepted, area, normal, center = verify_mask(
                points, det['mask'], det['class_name'],
                cfg['real_areas'], cfg['filter']['area_tolerance'],
                intrinsics=intrinsics,
            )
        except ValueError as e:
            logger.info('Rejected %s (conf=%.2f): %s', det['class_name'], det['confidence'], e)
            rejected.append(det)
            continue

        if not accepted:
            logger.info('Rejected %s (conf=%.2f, area=%.1f mm²).',
                        det['class_name'], det['confidence'], area)
            ref = cfg['real_areas'].get(det['class_name'], 0.0)
            if ref > 0 and area < ref * (1.0 - cfg['filter']['area_tolerance']):
                occluded_cands.append((det, normal, area))
            rejected.append(det)
            continue

        transform = _apply_approach_clamp(calculate_grip_transformation(normal, center))
        if transform is None:
            rejected.append(det)
            continue

        T_cam_world = get_robot_transform(robot) @ hand_eye
        T_world     = T_cam_world @ transform
        reachable   = is_reachable(cfg['chain'], T_world, cfg['home_pose'], cfg['bin'])
        _save_reachability_sample(T_world, reachable, cfg['bin'])
        if not reachable:
            transform = _find_reachable_approach(T_world, T_cam_world, cfg)
            if transform is None:
                logger.info('Rejected %s (conf=%.2f): no reachable approach found.',
                            det['class_name'], det['confidence'])
                rejected.append(det)
                continue

        result = PickResult(transform=transform, class_name=det['class_name'],
                            confidence=det['confidence'], area=area)
        start_visualization(color, points, detections, rejected, result)
        return result

    # Fallback: occluded chips via 3D ICP circle fit
    for det, normal, area in occluded_cands:
        ref       = cfg['real_areas'].get(det['class_name'], 0.0)
        radius_mm = float(np.sqrt(ref / np.pi))
        try:
            fitted_center = fit_circle_3d_icp(points, det['mask'], normal, radius_mm)
        except ValueError as e:
            logger.info('ICP circle fit failed for occluded %s: %s', det['class_name'], e)
            continue

        logger.info('Occluded %s (area=%.1f mm²): ICP center=%s',
                    det['class_name'], area, fitted_center.round(1).tolist())

        transform = _apply_approach_clamp(calculate_grip_transformation(normal, fitted_center))
        if transform is None:
            continue

        T_cam_world = get_robot_transform(robot) @ hand_eye
        T_world     = T_cam_world @ transform
        reachable   = is_reachable(cfg['chain'], T_world, cfg['home_pose'], cfg['bin'])
        _save_reachability_sample(T_world, reachable, cfg['bin'])
        if not reachable:
            transform = _find_reachable_approach(T_world, T_cam_world, cfg)
            if transform is None:
                logger.info('Occluded %s: no reachable approach found.', det['class_name'])
                continue

        result = PickResult(transform=transform, class_name='default',
                            confidence=det['confidence'], area=area)
        start_visualization(color, points, detections, rejected, result)
        return result

    return None
