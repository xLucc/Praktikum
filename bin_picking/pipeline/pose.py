from __future__ import annotations

import numpy as np


def calculate_grip_transformation(surface_normal: np.ndarray, center: np.ndarray, eps: float = 5e-5) -> np.ndarray:
    '''
    Builds a 4x4 grip transform (camera frame) from a surface normal + center point.

    The gripper approaches straight down along the normal; the in-plane (x/y) axes are
    arbitrary since the gripper itself is rotationally symmetric -- only the approach
    axis (the normal) matters here. Any standoff/offset motion is handled by the robot
    motion layer, outside this function.

    `center` (and therefore T's translation column) is in millimeters -- matches this
    codebase's robot-frame convention (see get_poses() in eye_in_hand_calibration.py),
    so the result can be handed to RobotNode.move()/move_pnp() without further conversion.
    '''
    support_vec = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    temp = np.cross(surface_normal, support_vec)
    y_axis = temp / np.linalg.norm(temp)

    temp = np.cross(y_axis, surface_normal)
    x_axis = temp / np.linalg.norm(temp)

    # Flip the normal if it doesn't face the camera.
    camera_origin = np.zeros(3)
    if surface_normal @ (center - camera_origin) < 0:
        surface_normal = -surface_normal

    # Ensure it's right-handed.
    if not (1 - eps <= np.linalg.det([x_axis, y_axis, surface_normal]) <= 1 + eps):
        y_axis = -y_axis

    T = np.eye(4)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = surface_normal
    T[:3, 3] = center

    return T
