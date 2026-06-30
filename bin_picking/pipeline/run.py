"""
Single-shot pick entry point.

Runs exactly one capture → pose → pick cycle, then shuts down:

    move_home → find_grasp → move_to_grasp → suction on → deliver_chip → move_home

For continuous (looping) operation see ``bin_picking.pipeline.loop``.
"""
from __future__ import annotations

import logging
import time

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


def main():
    """Initialise all resources, execute one pick cycle, then shut down cleanly."""
    logging.basicConfig(level=logging.INFO)
    cfg_dir = get_project_dir() / 'data' / 'cfg'

    processor  = ImageProcessing(cfg_path=str(cfg_dir / 'processing_cfg.json'))
    intrinsics = load_intrinsics(get_project_dir() / 'data' / 'calibration' / 'intrinsics.hdf5')
    hand_eye   = load_hand_eye(get_project_dir() / 'data' / 'calibration' / 'hand_eye.hdf5')
    cfg        = load_cfg()

    cam   = RealSenseCamera(align=True, adv=str(cfg_dir / 'high_density.json'))
    robot = setup_robot()

    try:
        if not move_home(robot, cfg['home_pose']):
            logger.warning('Move to home pose failed or timed out.')

        start  = time.perf_counter()
        result = find_grasp(cam, processor, intrinsics, cfg, robot, hand_eye)

        if result is None:
            logger.info('No verified grasp found.')
            return

        logger.info(
            'Grasp: %s (conf=%.2f, area=%.1fmm²)\n%s',
            result.class_name, result.confidence, result.area, result.transform,
        )
        logger.info('Took: %.3f s', time.perf_counter() - start)

        try:
            if not move_to_grasp(robot, hand_eye, result.transform, cfg['bin']):
                logger.warning('Move to grasp pose failed or timed out.')
            robot.turn_digital_output(6, 1)
            time.sleep(0.35)
            deliver_chip(robot, result.class_name, cfg['bin'], cfg['motion'])
            stop_visualization()
        except InvalidRobotTransform:
            logger.warning("Couldn't read the robot's current pose -- skipping move.")
    finally:
        move_home(robot, cfg['home_pose'])
        rclpy.shutdown()


if __name__ == '__main__':
    main()
