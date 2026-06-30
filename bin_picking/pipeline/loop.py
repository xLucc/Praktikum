from __future__ import annotations

import logging

import rclpy

from bin_picking.camera.camera import RealSenseCamera
from bin_picking.common.calibration_errors import InvalidRobotTransform
from bin_picking.common.helper import get_project_dir, load_hand_eye, load_intrinsics
from bin_picking.common.image_processing import ImageProcessing
from bin_picking.pipeline.config import load_cfg
from bin_picking.pipeline.grasp import find_grasp
from bin_picking.pipeline.motion import deliver_chip, move_home, move_to_grasp, setup_robot
from bin_picking.pipeline.visualize import stop_visualization

logger = logging.getLogger(__name__)

# After this many consecutive empty cycles, filter thresholds are relaxed so
# the detector gets a better chance on hard-to-see chips before giving up.
_RELAX_AT_EMPTY = 1

# After this many consecutive empty cycles (with relaxed params for the later
# ones), the bin is declared clear and the loop exits.
_EMPTY_CYCLES_BEFORE_STOP = 3


def _relax_filter(filter_cfg: dict) -> dict:
    """Return a copy of filter_cfg with loosened thresholds for a near-empty bin."""
    relaxed = filter_cfg.copy()
    relaxed['confidence']    = max(0.10, filter_cfg['confidence']    * 0.75)
    relaxed['area_tolerance'] = min(0.80, filter_cfg['area_tolerance'] * 1.5)
    relaxed['percentile']    = max(0.0,  filter_cfg['percentile']    * 0.5)
    return relaxed


def main():
    '''
    Continuous variant of bin_picking.pipeline.run:main -- repeats the same
    single-shot capture/pose/move cycle (find_grasp + move_to_grasp) in a loop
    until the bin is empty or interrupted, instead of running once and exiting.

    Termination: exits automatically after _EMPTY_CYCLES_BEFORE_STOP consecutive
    cycles with no verified grasp (bin cleared), or on Ctrl+C.
    '''
    logging.basicConfig(level=logging.INFO)
    cfg_dir = get_project_dir() / 'data' / 'cfg'

    processor = ImageProcessing(cfg_path=str(cfg_dir / 'processing_cfg.json'))
    intrinsics = load_intrinsics(get_project_dir() / 'data' / 'calibration' / 'intrinsics.hdf5')
    hand_eye = load_hand_eye(get_project_dir() / 'data' / 'calibration' / 'hand_eye.hdf5')
    cfg = load_cfg()

    cam = RealSenseCamera(align=True, adv=str(cfg_dir / 'high_density.json'))
    robot = setup_robot()

    logger.info('Starting continuous pick loop (Ctrl+C to stop).')

    consecutive_empty = 0

    try:
        while True:
            # Eye-in-hand setup: the camera moves with the robot, so it must be back at the
            # known home/viewing pose before every capture -- called every iteration, not
            # just once at startup.
            if not move_home(robot, cfg['home_pose']):
                logger.warning('Move to home pose failed or timed out.')

            if consecutive_empty >= _RELAX_AT_EMPTY:
                active_cfg = {**cfg, 'filter': _relax_filter(cfg['filter'])}
                logger.info(
                    'Relaxed filter params: conf=%.2f area_tol=%.2f percentile=%.2f',
                    active_cfg['filter']['confidence'],
                    active_cfg['filter']['area_tolerance'],
                    active_cfg['filter']['percentile'],
                )
            else:
                active_cfg = cfg

            result = find_grasp(cam, processor, intrinsics, active_cfg, robot, hand_eye)

            if result is None:
                consecutive_empty += 1
                logger.info(
                    'No verified grasp found (%d/%d consecutive empty cycles).',
                    consecutive_empty, _EMPTY_CYCLES_BEFORE_STOP,
                )
                if consecutive_empty >= _EMPTY_CYCLES_BEFORE_STOP:
                    logger.info('Bin appears empty — stopping after %d consecutive empty cycles.',
                                _EMPTY_CYCLES_BEFORE_STOP)
                    break
                continue

            # Successful detection resets the empty-cycle counter and restores normal params.
            consecutive_empty = 0

            logger.info(
                'Grasp: %s (conf=%.2f, area=%.1f mm²)\n%s',
                result.class_name, result.confidence, result.area, result.transform,
            )

            try:
                if not move_to_grasp(robot, hand_eye, result.transform, cfg['bin']):
                    logger.warning('Move to grasp pose failed or timed out.')
                robot.turn_digital_output(6, 1)
                deliver_chip(robot, result.class_name, cfg['bin'], cfg['motion'])
                stop_visualization()
            except InvalidRobotTransform:
                logger.warning("Couldn't read the robot's current pose -- skipping move.")

            move_home(robot, cfg['home_pose'])
    except KeyboardInterrupt:
        logger.info('Stopped.')
    finally:
        move_home(robot, cfg['home_pose'])
        rclpy.shutdown()


if __name__ == '__main__':
    main()
