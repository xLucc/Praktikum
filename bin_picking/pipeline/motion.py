from __future__ import annotations

import logging
import threading
import time

import numpy as np
import rclpy
from scipy.spatial.transform import Rotation

from bin_picking.camera.camera import RealSenseCamera
from bin_picking.common.eye_in_hand_calibration import get_robot_transform
from bin_picking.common.image_processing import ImageProcessing
from bin_picking.pipeline.config import TCP_OFFSET, TCP_TUBE_END_MM
from bin_picking.robot.node import RobotNode

logger = logging.getLogger(__name__)


def setup_robot() -> RobotNode:
    '''
    Starts the ROS2 robot node, spins it in a background thread, waits for its
    services, then sets the TCP offset and jogging frame.
    '''
    rclpy.init(args=None)
    robot = RobotNode()
    threading.Thread(target=robot.spin, daemon=True).start()
    robot.wait_for_service()
    robot.set_sys_frame(pos=list(TCP_OFFSET), rot=[0.0, 0.0, 0.0])
    robot.select_jf(0)
    robot.resume_motion()
    robot.readable = False
    return robot


def _poll_until_close(get_current, target, tol: float, robot: RobotNode,
                      timeout: float, poll_interval: float) -> bool:
    '''
    Polls `get_current()` until within `tol` of `target`, or `timeout` elapses.
    Aborts early if the robot signals a halt/collision via robot.readable.
    '''
    target = np.asarray(target, dtype=np.float64)
    start  = time.monotonic()

    while time.monotonic() - start < timeout:
        if robot.readable:
            robot.readable = False
            logger.warning('Robot reported a halt/collision while waiting for arrival.')
            return False
        current = get_current()
        if current is not None and len(current) == len(target):
            if np.linalg.norm(np.asarray(current, dtype=np.float64) - target) <= tol:
                return True
        time.sleep(poll_interval)

    return False


def move_home(robot: RobotNode, home_pose: list,
              tol: float = 0.5, timeout: float = 30.0) -> bool:
    '''Joint-space PTP move to home pose, blocking until the robot arrives.'''
    if not robot.move_pnp(home_pose):
        return False

    def get_current():
        try:
            return robot.pose
        except IndexError:
            return None

    return _poll_until_close(get_current, home_pose, tol, robot, timeout, poll_interval=0.05)


def move_to_grasp(robot: RobotNode, hand_eye: np.ndarray,
                  T_camera_target: np.ndarray, bin_cfg: dict,
                  tol: float = 1.0, timeout: float = 30.0) -> bool:
    '''
    Two-step Cartesian approach: hover above the bin rim, then drop to grasp Z.
    '''
    T_tfp_world    = get_robot_transform(robot)
    T_world_target = T_tfp_world @ hand_eye @ T_camera_target

    pos = T_world_target[:3, 3].copy()
    pos[2] += 4.0
    rot = Rotation.from_matrix(T_world_target[:3, :3]).as_euler('xyz', degrees=True)

    bin_top_z = bin_cfg['world_pos'][2] + bin_cfg['size'][2] / 2
    hover_pos    = pos.copy()
    hover_pos[2] = bin_top_z + 180.0

    if not robot.move({'pos': list(hover_pos), 'rot': list(rot), 'ref': 0}):
        return False
    if not _poll_until_close(lambda: robot.robot_pose.get('pos'), hover_pos,
                             tol, robot, timeout, poll_interval=0.2):
        return False

    if not robot.move({'pos': list(pos), 'rot': list(rot), 'ref': 0}):
        return False
    return _poll_until_close(lambda: robot.robot_pose.get('pos'), pos,
                             tol, robot, timeout, poll_interval=0.2)


def retreat_from_grasp(robot: RobotNode, bin_cfg: dict, drop_pos: list,
                       tol: float = 1.0, timeout: float = 30.0) -> bool:
    '''
    Three-step retreat: lift until suction tip clears the bin rim, rotate to
    world -Z, then translate to drop_pos.
    '''
    T_tfp_world       = get_robot_transform(robot)
    current_pos       = T_tfp_world[:3, 3]
    current_rot_euler = Rotation.from_matrix(T_tfp_world[:3, :3]).as_euler('xyz', degrees=True)

    bin_top_z    = bin_cfg['world_pos'][2] + bin_cfg['size'][2] / 2
    clear_tfc_z  = bin_top_z + TCP_TUBE_END_MM + 50.0
    lift_pos     = current_pos.copy()
    lift_pos[2]  = max(current_pos[2] + 10.0, clear_tfc_z)

    if not robot.move({'pos': list(lift_pos), 'rot': list(current_rot_euler), 'ref': 0}):
        return False
    if not _poll_until_close(lambda: robot.robot_pose.get('pos'), lift_pos,
                             tol, robot, timeout, poll_interval=0.2):
        return False

    tool_z   = T_tfp_world[:3, 2]
    target_z = np.array([0.0, 0.0, -1.0])
    cross    = np.cross(tool_z, target_z)
    sin_a    = np.linalg.norm(cross)
    cos_a    = float(tool_z @ target_z)
    R_align  = (Rotation.from_rotvec(cross / sin_a * np.arctan2(sin_a, cos_a)).as_matrix()
                if sin_a > 1e-6 else np.eye(3))
    down_rot = Rotation.from_matrix(R_align @ T_tfp_world[:3, :3]).as_euler('xyz', degrees=True).tolist()

    if not robot.move({'pos': list(lift_pos), 'rot': down_rot, 'ref': 0}):
        return False
    if not _poll_until_close(lambda: robot.robot_pose.get('pos'), lift_pos,
                             tol, robot, timeout, poll_interval=0.2):
        return False

    if not robot.move({'pos': drop_pos, 'rot': down_rot, 'ref': 0}):
        return False
    return _poll_until_close(lambda: robot.robot_pose.get('pos'), drop_pos,
                             tol, robot, timeout, poll_interval=0.2)


def deliver_chip(robot: RobotNode, class_name: str, bin_cfg: dict,
                 motion_cfg: dict, tol: float = 1.0, timeout: float = 30.0) -> bool:
    '''Retreat to the class-specific drop location and release the chip.'''
    class_poses = motion_cfg.get('class_drop_poses', {})
    drop_pos    = class_poses.get(class_name, motion_cfg['drop_pose'])['pos']

    reached = retreat_from_grasp(robot, bin_cfg, drop_pos, tol=tol, timeout=timeout)
    if not reached:
        logger.warning(f"Move to drop location for {class_name!r} failed or timed out.")

    robot.turn_digital_output(6, 0)
    return reached


def capture(cam: RealSenseCamera,
            processor: ImageProcessing) -> tuple[np.ndarray, np.ndarray]:
    '''9-frame temporal-median depth + 1 color frame.'''
    depth_frames = cam.get_depth(num_frames=11)
    depth_raw    = processor.median_filtering_over_time(depth_frames)
    color        = cam.get_color(num_frames=1)[0]
    depth_raw    = np.where((depth_raw > 100) & (depth_raw < 1000), depth_raw, 0)
    return color, depth_raw * 0.001
