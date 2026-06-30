import rclpy
import argparse
import sys
import h5py
import logging
import threading
import os
import numpy as np
import cv2 as cv
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union, Tuple, Set, List
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Semaphore, Event
from multiprocessing.shared_memory import SharedMemory
from bin_picking.robot.node import RobotNode
from bin_picking.camera.camera import RealSenseCamera
from bin_picking.common.image_processing import ImageProcessing
from bin_picking.common.helper import get_project_dir
from bin_picking.common.calibration_errors import InvalidRobotTransform

# --- Aruco ---
ARUCO_DICT = aruco.DICT_4X4_250
BOARD_SIZE = (5,7)
MARKER_LENGTH = 22.0
SQUARE_LENGTH = 29.7


def main(**kwargs):
    """
    Main calibration loop.

    Continuously captures color frames and depth maps from the RealSense camera
    and reads the current TCP transform from the robot. Each (color, depth, transform)
    triple is buffered. On KeyboardInterrupt the buffer is processed in parallel to
    extract 3D point clouds and circle positions for each frame, after which
    OpenCV's hand-eye calibration is run and the result is persisted.
    """
    kwargs = get_args(**kwargs)
    project_dir      = get_project_dir()
    data_path        = project_dir / 'data'
    cfg_path         = data_path   / 'cfg'
    calibration_path = data_path   / 'calibration'

    for path in [data_path, cfg_path, calibration_path]:
        path.mkdir(parents=True, exist_ok=True)

    hand_eye_path = calibration_path / 'hand_eye.hdf5'
    buf_path      = data_path        / 'buf.hdf5'
    logger        = logging.getLogger(__name__)

    rclpy.init(args=None)

    if not kwargs['skip']:
        cam       = RealSenseCamera(align=True, adv=str(cfg_path / 'high_density.json'))
        robot     = RobotNode()
        node      = threading.Thread(target=robot.spin, daemon=True)
        node.start()
        robot.wait_for_service()

    processor = ImageProcessing(cfg_path=str(cfg_path / 'processing_cfg.json'))

    if kwargs['cam']:
        intrinsics = load_intrinsics(calibration_path / 'intrinsics.hdf5')

    count = 0
    # Buffer storing Data objects (transform, depth, color) for deferred parallel processing.
    buf: List[Data] = []

    if not kwargs['skip']:
        h, w = cam.color_resolution
        shape = (w, h, 3) # Turn the resolution into np shape.
        size = int(np.prod(shape)) * np.dtype(np.uint8).itemsize # w * h * 3 * 1

        shm = SharedMemory(create=True, size=size)
        lock = Semaphore(1)
        frame_event = Event()
        exit_event = Event()

        poses = get_poses()

        for pos in poses:
            _   = robot.move_pnp(pos)
            p = Process(target=cam.stream_parallel, args=(shm.name, lock, frame_event, shape, exit_event), daemon=True)
            logger.info(f'Iteration: {count}')
            cam.stop()
            p.start()
            frame_event.clear()
            img = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
            clean = None

            while True:
                frame_event.wait(timeout=1.0)

                if not frame_event.is_set():
                    continue

                frame_event.clear()

                with lock:
                    frame = img.copy()
                
                clean = frame.copy()

                cv.imshow('clean', clean)
                key = cv.waitKey(1) & 0xFF

                if key == ord("q"):
                    break


            exit_event.set()
            logger.info('Close process')
            p.join(timeout=3.0)

            if p.is_alive():
                p.terminate()
                p.join()
            
            exit_event.clear()
            cv.destroyAllWindows()
            cam.start()
            
            cmd = prompt_cmd('Discard the image or use it.', {Cmd.DISCARD, Cmd.KEEP, Cmd.EXIT})
            
            if cmd == Cmd.EXIT:
                break
            elif cmd == Cmd.DISCARD:
                continue

            count += 1

            logger.info('Take pictures.')
            depth = take_pic(cam, processor)

            logger.info('Get transform.')
            T     = get_robot_transform(robot)
            clean = clean if clean is not None else np.zeros(shape)

            data = Data(T=T, depth=depth, color=clean)
            buf.append(data)
        shm.close()
        shm.unlink()
    else: 
        buf = load_buf(buf_path)

    if not kwargs['skip']:
        store_buf(buf, buf_path)

    if not kwargs['cam']:
        intrinsics = calibrate_camera_charuco(images=[d.color for d in buf])

    # Process all buffered frames once the capture loop has ended.
    T_cam_2_target_list = process_buf(buf, intrinsics)
    T_base_2_tfp_list = get_robot_info_from_buf(buf)

    R_target_2_cam, t_target_2_cam = deconstruct_T_into_R_and_t(T_cam_2_target_list)
    R_tfp_2_base,   t_tfp_2_base   = deconstruct_T_into_R_and_t(T_base_2_tfp_list)

    R_cam_2_tfp, t_cam_2_tfp = cv.calibrateHandEye(
        R_gripper2base=R_tfp_2_base,
        t_gripper2base=t_tfp_2_base,
        R_target2cam=R_target_2_cam,
        t_target2cam=t_target_2_cam,
    )
    logger.info(f"Rotation: {R_cam_2_tfp}")
    logger.info(f"Translation: {t_cam_2_tfp}")

    R_cam_2_tfp = np.asarray(R_cam_2_tfp)
    plot_calib_params(R_cam_2_tfp, t_cam_2_tfp)
    store_calib(R_cam_2_tfp, t_cam_2_tfp, hand_eye_path)
    X = np.eye(4)
    X[:3, :3] = R_cam_2_tfp
    X[:3, 3] = t_cam_2_tfp.ravel()
    _,_,_  = axb_consistency_check(T_base_2_tfp_list, T_cam_2_target_list, X, logger)


def calibrate_camera_charuco(
    images: List[np.ndarray],
    board_size: Tuple[int, int] = BOARD_SIZE,
    square_length_mm: float = SQUARE_LENGTH,
    marker_length_mm: float = MARKER_LENGTH,
    aruco_dict_type = ARUCO_DICT,
) -> dict:
    """
    Calibrate a camera using ChArUco board images.

    Args:
        images:            List of grayscale or BGR images containing the ChArUco board.
        board_size:        (squares_x, squares_y) — number of chessboard squares.
        square_length_mm:  Side length of one chessboard square in millimeters.
        marker_length_mm:  Side length of one ArUco marker in millimeters.
        aruco_dict_type:   ArUco dictionary to use.

    Returns:
        dict with keys: fx, fy, cx, cy, dist, rms
        dist is the full distortion coefficient vector (k1, k2, p1, p2 [, k3, ...]).

    Raises:
        ValueError: if fewer than 4 images yield usable corners.
    """
    aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)
    board = aruco.CharucoBoard(
        board_size,
        square_length_mm,
        marker_length_mm,
        aruco_dict,
    )
    charuco_detector = aruco.CharucoDetector(
        board,
        aruco.CharucoParameters(),
        aruco.DetectorParameters(),
    )

    all_charuco_corners: list = []
    all_charuco_ids: list = []
    image_size = None

    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img
        image_size = gray.shape[::-1]  # (width, height)

        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)

        if charuco_ids is None or len(charuco_ids) < 4:
            continue

        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)

    if len(all_charuco_corners) < 4:
        raise ValueError(
            f"Only {len(all_charuco_corners)} usable frames — need at least 4. "
            "Check board visibility and ArUco dictionary."
        )
    distCoeffs = np.zeros((5,1), dtype=np.float64)
    rms, camera_matrix, dist_coeffs, _, _ = aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=distCoeffs,
    )

    return {
        "fx":   float(camera_matrix[0, 0]),
        "fy":   float(camera_matrix[1, 1]),
        "ppx":   float(camera_matrix[0, 2]),
        "ppy":   float(camera_matrix[1, 2]),
        "dist": dist_coeffs.flatten(),
        "rms":  float(rms),
    }


def store_calib(rotation: np.ndarray, translation: np.ndarray, path: Path) -> None:
    """
    Persist the hand-eye calibration result as a 4x4 homogeneous transform in an HDF5 file.

    Args:
        rotation:    3x3 rotation matrix (camera-to-TFP).
        translation: 3-element translation vector (camera-to-TFP).
        path:        Destination .hdf5 file path.
    """
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3]  = translation.ravel()

    with h5py.File(path, 'w') as f:
        f.create_dataset('T_cam2tfp', data=T)


def store_buf(buf: list, path: Path) -> None:
    """
    Write all buffered Data objects to an HDF5 file for offline inspection.

    Each frame is stored in a separate group named ``arr_<i>`` containing
    three datasets: ``T`` (robot transform), ``depth``, and ``color``.

    Args:
        buf:  List of Data objects captured during the calibration run.
        path: Destination .hdf5 file path.
    """
    with h5py.File(path, 'w') as f:
        for i, d in enumerate(buf):
            grp = f.create_group(f'arr_{i}')
            grp.create_dataset('T',     data=d.T)
            grp.create_dataset('depth', data=d.depth)
            grp.create_dataset('color', data=d.color)

def load_buf(buf_path: Path) -> List['Data']:
    with h5py.File(buf_path, 'r') as f:
        buf = []
        for key in f.keys():
            grp = f[key]
            T     = grp['T'][:]
            depth = grp['depth'][:]
            color = grp['color'][:]
            buf.append(Data(T=T, depth=depth, color=color))
    return buf

def get_robot_info_from_buf(buf: list) -> list:
    """
    Extract the robot TCP transforms from the capture buffer.

    Args:
        buf: List of Data objects.

    Returns:
        List of 4x4 homogeneous transform matrices (base-to-TFP), one per frame.
    """
    return [data.T for data in buf]


def deconstruct_T_into_R_and_t(T_list: list) -> Tuple[list, list]:
    """
    Split a list of 4x4 homogeneous transforms into separate rotation and translation lists.

    Args:
        T_list: List of 4x4 numpy arrays.

    Returns:
        Tuple (R_list, t_list) where each element is a list of 3x3 rotation
        matrices and 3-element translation vectors respectively.
    """
    R_list, t_list = [], []
    for T in T_list:
        R_list.append(T[:3, :3])
        t_list.append(T[:3, 3])
    return R_list, t_list


def process_buf(buf: list, intrinsics: dict) -> list:
    """
    Process all buffered frames in parallel.

    Each frame's ChArUco target pose is estimated in its own thread; the degree
    of parallelism is capped at the number of logical CPU cores.

    Args:
        buf:        List of Data objects (color, depth, robot transform).
        intrinsics: Camera intrinsic parameters (fx, fy, ppx, ppy, dist).

    Returns:
        List of 4x4 camera-to-target transforms, one per buffered frame.
    """
    cpu_count   = os.cpu_count() or 1
    max_workers = min(len(buf), cpu_count)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures_charuco = [executor.submit(get_charuco_pose, d.color, intrinsics) for d in buf]
        charuco_list   = [f.result() for f in futures_charuco]

    return charuco_list


def get_robot_transform(robot: RobotNode) -> np.ndarray:
    """
    Query the current TCP pose from the robot and convert it to a 4x4 transform matrix.

    The robot reports Euler angles (XYZ convention, degrees) and a translation
    vector. Both are combined into a homogeneous transformation matrix.

    Args:
        robot: RobotNode instance connected to the Kassow controller.

    Returns:
        4x4 homogeneous transform matrix [R | t; 0 0 0 1] in robot base frame units.

    Raises:
        InvalidRobotTransform: If the robot returns None or an all-zero pose,
                               which indicates a communication or initialisation error.
    """
    euler_cords = robot.sys_frame('tfc', 'world')

    if euler_cords is None:
        raise InvalidRobotTransform

    # An all-zero response typically means the robot has not been homed yet.
    if np.all(euler_cords == 0):
        raise InvalidRobotTransform

    angles = euler_cords[1, :]
    r      = R.from_euler('xyz', angles, degrees=True)

    T = np.eye(4)
    T[:3, :3] = r.as_matrix()
    T[:3, 3]  = euler_cords[0, :]

    return T


def take_pic(camera: RealSenseCamera, processor: ImageProcessing, num_depth: int = 9, num_color: int = 1) -> np.ndarray:
    """
    Capture a temporally filtered depth image from the RealSense camera.

    Multiple depth frames are median-filtered to reduce per-pixel noise before
    returning a single representative depth map.

    Args:
        camera:     RealSenseCamera instance.
        processor:  ImageProcessing instance providing median_filtering_over_time.
        num_depth:  Number of depth frames to capture and median-filter (default 15).
        num_color:  Number of color frames to capture (currently unused, reserved).

    Returns:
        2D float array containing the median-filtered depth image.
    """
    depth_frames = camera.get_depth(num_depth)
    depth_img    = processor.median_filtering_over_time(depth_frames)
    return depth_img


def load_intrinsics(path: Union[str, Path]) -> dict:
    """
    Load camera intrinsic parameters from a supported file format.

    Currently supported formats:
        - HDF5 (.hdf5): expects datasets at 'intrinsics/mtx' and 'intrinsics/dist'.

    Args:
        path: Path to the intrinsics file.

    Returns:
        Dict with keys: fx, fy, ppx, ppy, dist.

    Raises:
        ValueError:   If path is not a str or Path.
        RuntimeError: If the file extension is not supported.
    """
    if not isinstance(path, (str, Path)):
        raise ValueError(f'Expected path to be either str or Path, got: {type(path)}')

    path = path if isinstance(path, Path) else Path(path)

    if path.suffix == '.hdf5':
        return load_hdf5(path)
    else:
        raise RuntimeError('Unsupported file type.')


def load_hdf5(path: Path) -> dict:
    """
    Read camera intrinsics from an HDF5 file.

    Expected HDF5 layout:
        intrinsics/
            mtx   -- 3x3 camera matrix  [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            dist  -- distortion coefficients, shape (1, 5) or (5,)

    Args:
        path: Path to the .hdf5 file.

    Returns:
        Dict with keys fx, fy, ppx, ppy, dist (first row of dist array).
    """
    with h5py.File(path, 'r') as f:
        mtx:  np.ndarray = f['intrinsics']['mtx'][:]
        dist: np.ndarray = f['intrinsics']['dist'][:]

    return {
        "fx":   mtx[0, 0],
        "fy":   mtx[1, 1],
        "ppx":  mtx[0, 2],
        "ppy":  mtx[1, 2],
        "dist": dist.ravel(),
    }


def undistort_img(img: np.ndarray, intrinsics: dict) -> np.ndarray:
    """
    Remove radial and tangential lens distortion from a BGR image.

    Reconstructs the 3x3 camera matrix from the intrinsics dict and applies
    OpenCV's undistort using the stored distortion coefficients.

    Args:
        img:        Distorted BGR input image.
        intrinsics: Dict with keys fx, fy, ppx, ppy, dist.

    Returns:
        Undistorted BGR image of the same resolution.
    """
    mtx = np.array([
        [intrinsics["fx"],              0, intrinsics["ppx"]],
        [             0,  intrinsics["fy"], intrinsics["ppy"]],
        [             0,              0,              1      ],
    ], dtype=np.float64)

    return cv.undistort(img, mtx, intrinsics["dist"])


def get_charuco_pose(color: np.ndarray, intrinsics: dict, board_size: tuple = BOARD_SIZE, square_length: float = SQUARE_LENGTH, marker_length: float = MARKER_LENGTH, aruco_dict_id: int = ARUCO_DICT
):
    """
    Estimate target pose in camera frame using a self-created ChArUco board.

    Args:
        color: BGR image
        intrinsics: dict with fx, fy, ppx, ppy, dist
        board_size: (cols, rows) of chessboard
        square_length: size of chessboard squares [m]
        marker_length: size of ArUco markers [m]
        aruco_dict_id: OpenCV dictionary ID

    Returns:
        4x4 homogeneous transform T_camera_target or None
    """

    # Build camera matrix K
    K = np.array([
        [intrinsics["fx"], 0, intrinsics["ppx"]],
        [0, intrinsics["fy"], intrinsics["ppy"]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist = intrinsics["dist"]

    # Create ArUco dictionary
    aruco_dict = cv.aruco.getPredefinedDictionary(aruco_dict_id)

    # Create ChArUco board (THIS replaces your missing board variable)
    board = cv.aruco.CharucoBoard(
        board_size,
        square_length,
        marker_length,
        aruco_dict
    )

    # Detect markers
    gray = cv.cvtColor(color, cv.COLOR_BGR2GRAY)

    detector = cv.aruco.ArucoDetector(aruco_dict)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) < 4:
        return None

    # Interpolate ChArUco corners
    charuco_detector = cv.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(
        image=gray, markerCorners=corners, markerIds=ids
    )

    if charuco_ids is None or len(charuco_ids) < 4:
        return None

    # Solve PnP
    rvec = np.zeros((3, 1), dtype=np.float64) # Due to inconsistencies in OpenCV's API, these must be initialized as non-None arrays. Inplace vals and return vals will be the same.
    tvec = np.zeros((3, 1), dtype=np.float64) # Due to inconsistencies in OpenCV's API, these must be initialized as non-None arrays. Inplace vals and return vals will be the same.

    success, rvec, tvec = cv.aruco.estimatePoseCharucoBoard(
        charuco_corners,
        charuco_ids,
        board,
        K,
        dist,
        rvec,
        tvec
    )

    if not success:
        return None

    # Homogeneous transform
    R_mat, _ = cv.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = tvec.flatten()
    return T


@dataclass
class Data:
    """Container for a single synchronised capture: robot pose, depth image, color image."""
    T:     np.ndarray   # 4x4 homogeneous robot TCP transform in base frame
    depth: np.ndarray   # Temporally filtered depth image
    color: np.ndarray   # Undistorted BGR color image


# An enum for the possible user commands, with a helper function to parse them from input.
class Cmd(Enum):
    CONTINUE = "c"
    QUIT = "q"
    DISCARD = "d"
    KEEP = "k"
    EXIT = "e"

    @classmethod
    def from_input(cls, raw: str) -> Optional['Cmd']:
        raw = raw.strip().lower()
        for cmd in cls:
            if cmd.value == raw:
                return cmd
        return None

# Prompt the user for a command until a valid one is given or the maximum number of tries is reached.
def prompt_cmd(prompt: str, valid: Set["Cmd"], max_tries: int = 5) -> Cmd:
    options = "/".join(cmd.value for cmd in valid)
    for _ in range(max_tries):
        cmd = Cmd.from_input(input(f"{prompt} [{options}]: "))
        if cmd in valid:
            return cmd
        print("Invalid input.")
    raise RuntimeError(f"No valid input after {max_tries} tries.")


def get_args(**kwargs):
    filtered = rclpy.utilities.remove_ros_args(sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument('--skip', action='store_true', default=False)
    parser.add_argument('--cam', action='store_true', default=False)
    args   = parser.parse_args(filtered[1:])

    kwargs.update(vars(args))

    return kwargs

def plot_calib_params(rotation: np.ndarray, translation: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    translation = translation.ravel()
    ax.scatter(0,0,0)
    ax.scatter(*translation)
    colors = ['red', 'blue', 'green']
    for i, c in enumerate(colors):
        direction = np.eye(3)[i]
        ax.quiver(0,0,0, *direction, color=c, length=30)

    for i, c in enumerate(colors):
        ax.quiver(*translation, *rotation[:3, i], color=c, normalize=True, length=30)

    ax.set_aspect('equal')
    # ax.quiver(0,0,0, 1,0,0, color='red')
    # ax.quiver(0,0,0, 0,1,0, color='blue')
    # ax.quiver(0,0,0, 0,0,1, color='green')
    # ax.quiver(translation, rotation[:3, 0][0], rotation[:3, 0][1], rotation[:3, 0][2], color='blue', normalize=True)
    # ax.quiver(translation.ravel(), rotation[:3, 0][0], rotation[:3, 1][1], rotation[:3, 1][2], color='red', normalize=True)
    # ax.quiver(translation.ravel(), rotation[:3, 2], color='green', normalize=True)
    plt.show()


def axb_consistency_check(
    T_gripper2base: List[np.ndarray],
    T_target2cam: List[np.ndarray],
    X: np.ndarray,
    logger: logging.Logger
) -> Tuple[float, float, np.ndarray]:
    """
    Evaluate the pairwise AX = XB residual error for a hand-eye calibration result.

    For every frame pair (i, j), the relative gripper motion A_ij and the
    corresponding relative target motion B_ij must satisfy A·X = X·B if X
    (the camera-to-TCP transform) is correct *and* the input data is
    consistent.  Large residuals indicate systematic errors in the inputs
    (wrong Euler convention, unit mismatch, unsynchronised poses, …).

    Args:
        T_gripper2base: List of N 4×4 transforms (gripper pose in base frame).
        T_target2cam:   List of N 4×4 transforms (target pose in camera frame).
        X:              4×4 hand-eye calibration result (camera-to-gripper).

    Returns:
        mean_err:   Mean translational residual over all pairs [same unit as inputs].
        max_err:    Worst-case translational residual.
        all_errs:   1-D array of per-pair errors (length N*(N-1)/2), sorted ascending.
    """
    n = len(T_gripper2base)
    assert n == len(T_target2cam), "Input lists must have the same length."
    assert n >= 2, "Need at least 2 poses."

    errs = []
    for i in range(n):
        for j in range(i + 1, n):
            A = np.linalg.inv(T_gripper2base[j]) @ T_gripper2base[i]
            B = T_target2cam[j] @ np.linalg.inv(T_target2cam[i])

            lhs = A @ X
            rhs = X @ B
            errs.append(np.linalg.norm(lhs[:3, 3] - rhs[:3, 3]))

    all_errs = np.sort(errs)
    mean_err = float(all_errs.mean())
    std_err = float(all_errs.std())
    max_err = float(all_errs.max())

    logger.info(
        f"AX=XB check  —  pairs: {len(all_errs)},\n"
        f"mean: {mean_err:.2f}, median: {float(np.median(all_errs)):.2f},\n"
        f'std: {std_err:.2f} \n'
        f"max: {max_err:.2f}"
    )

    return mean_err, max_err, all_errs

def get_poses() -> List[List[float]]:
    return [[303.96, -1.44, 34.017, 116.6, -102.57, 67.25, 23.0],
            [303.96, -4.82, 25.72, 77.06, -70.928, 65.283, -26.0], 
            [303.88, -45.27, -12.89, 117.86, -68.93, 25.78, -19.0],
            [304.184, -36.914, -48.26, 129.05, -72.485, -96.02, -7.0],
            [303.21, -28.26, 47.72, 106.65, -120.66, -99.99, -24.0],
            [303.22, -28.671, 48.33, 111.33, - 94.4, -8.629, -17.0], 
            [303.246, 22.437, 52.847, 86.99, -178.609, -22.909, 61.0],
            [296.356, 45.564, 23.91, 44.607, -134.786, -56.17, 51.0],
            [295.96, -3.58, 62.334, 146.93, -201.735, 47.28, 58.0],
            [299.08, 24.59, 72.737, 119.25, -205.541, -13.916, 61.0],
            [299.178, 31.37, 52.86, 107.391, -222.389, -13.874, 106.0],
            [298.061, 42.588, 39.251, 71.808, -177.096, -38.584, 73.0],
            [288.662, 36.777, 23.168, 72.504, -149.748, -23.665, 72.0],
            [311.0, 14.811, 0.655, 114.82, -155.7, 7.19, 94.9],
            [310.586, -30.542, -5.491, 151.394, -125.278, 46.706, 84.5],
            [310.535, -3.147, -4.045, 159.86, -228.854, 98.951, 102.0],
            [295.611, 1.108, -28.090, 121.25, -286.025, 11.882, 217.0],
            [321.209, 9.887, 15.792, 121.163, -284.549, 54.833, 148.0],
            [320.824, -37.0, 12.0, 153.0, -265.0, 65.0, 118.0],
            [315.0, -47.0, -55.0, 149.0, -262.0, 92.0, 148.0],
            [299.0, 10.0, -0.3, 128.0, 46.0, 128.0, 101.0],
            [297.0, 40.0, -55.0, 93.0, 12.0, -135.0, 59.0]]