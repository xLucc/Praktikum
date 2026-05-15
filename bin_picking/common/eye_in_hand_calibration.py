import rclpy
import argparse
import sys
import h5py
import logging
import threading
import time
import os
import numpy as np
import cv2 as cv
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from pyzbar.pyzbar import decode as decode_qr
from pathlib import Path
from typing import Optional, Union, Tuple, Set, List, Dict
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Semaphore, Event
from multiprocessing.shared_memory import SharedMemory
from bin_picking.robot.node import RobotNode
from bin_picking.camera.camera import RealSenseCamera
from bin_picking.common.image_processing import ImageProcessing
from bin_picking.common.helper import get_project_dir, load_dict_from_json
from bin_picking.common.calibration_errors import *

# --- Hough circle detection radius bounds (0 = unlimited) ---
MIN_RADIUS = 0
MAX_RADIUS = 0

# --- HSV bounds for red (two ranges required due to hue wrap-around at 0°/180°) ---
RED_LOW1  = [0,   100,  50]
RED_HIGH1 = [20,  255, 255]

RED_LOW2  = [160, 100,  50]
RED_HIGH2 = [179, 255, 255]

# --- HSV bounds for blue and green ---
BLUE_LOW  = [100,  80,  50]
BLUE_HIGH = [130, 255, 255]

GREEN_LOW  = [40,  50,  50]
GREEN_HIGH = [80, 255, 255]

# Maps color names to their HSV range tuples (lower, upper bound).
# Red requires two ranges since its hue wraps around the 0°/180° boundary in HSV.
COLOR_RANGES = {
    "red":   [(RED_LOW1, RED_HIGH1), (RED_LOW2, RED_HIGH2)],
    "green": [(GREEN_LOW, GREEN_HIGH)],
    "blue":  [(BLUE_LOW, BLUE_HIGH)],
}


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

    intrinsics = load_intrinsics(calibration_path / 'intrinsics.hdf5')
    mapping    = get_mapping(cfg_path / 'mapping.json')

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
            
                detections = get_aruco_codes(frame)

                if detections:
                    draw_aruco_codes(frame, detections)

                clone = undistort_img(frame, intrinsics)
                cv.imshow('circles', clone)
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

            data = Data(T=T, depth=depth, color=undistort_img(clean, intrinsics))
            buf.append(data)
        shm.close()
        shm.unlink()
    else: 
        buf = load_buf(buf_path)

    if not kwargs['skip']:
        store_buf(buf, buf_path)

    # Process all buffered frames once the capture loop has ended.
    point_3d_list, T_cam_2_target_list = process_buf(buf, intrinsics, processor, mapping)
    # axis_list   = map_marker_to_3d(point_3d_list, marker_list)
    # invert_axis = mapping['invert_axis']
    # T_cam_2_target_list = [
    #     construct_coordinate_system(Px=Px, Py=Py, Pz=Pz, invert_axis=invert_axis)
    #     for Px, Py, Pz in axis_list
    # ]
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
               
def get_qr_codes(color: np.ndarray) -> np.ndarray:
    """"
    Detect QR codes in a BGR image and return their bounding circle descriptors.

    For each detected QR code the bounding polygon is used to compute a
    minimum enclosing circle, matching the [cx, cy, radius] format of the
    former Hough-circle output so that all downstream code remains unchanged.

    Args:
        color: BGR input image.

    Returns:
        Nx3 uint16 array of detected QR codes as circles [cx, cy, radius],
        or a zero array of shape (1, 3) if nothing is detected.
    """
    gray    = cv.cvtColor(color, cv.COLOR_BGR2GRAY)
    decoded = decode_qr(gray)

    if not decoded:
        return np.zeros((1, 3), dtype=np.uint16)

    result = []
    for qr in decoded:
        pts    = np.array(qr.polygon, dtype=np.float32)
        pts_xy = np.array([[p.x, p.y] for p in pts], dtype=np.float32)
        (cx, cy), radius = cv.minEnclosingCircle(pts_xy)
        result.append([int(cx), int(cy), int(radius)])

    return np.array(result, dtype=np.uint16)

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


def map_marker_to_3d(point_3d_list: list, circle_list: list) -> list:
    """
    Convert 2D circle descriptors to 3D axis points using the reconstructed point cloud.

    For each circle the centre pixel is used to read (X, Y) from the point cloud,
    while Z is estimated as the median of all valid depth values inside the circle
    to reduce the influence of outliers.

    Args:
        point_3d_list: List of HxWx3 point cloud arrays, one per frame.
        circle_list:   List of circle triplets (x, y, z marker), one per frame.
                       Each marker is a [cx, cy, radius] array.

    Returns:
        List of (Px, Py, Pz) tuples of 3D points corresponding to the X, Y, Z
        axis markers for every frame.
    """
    def map_uvr_to_xyz(vec: tuple, point_3d: np.ndarray) -> np.ndarray:
        """Back-project a single circle descriptor to a 3D point."""
        cx, cy, r = vec

        x, y = point_3d[cy, cx, :2]

        h, w = point_3d.shape[:2]
        us, vs = np.meshgrid(np.arange(w), np.arange(h))
        mask = (us - cx) ** 2 + (vs - cy) ** 2 <= r ** 2

        z_vals  = point_3d[mask, 2]
        z_valid = z_vals[z_vals > 0]
        z       = np.median(z_valid)

        return np.array([x, y, z])

    axis_points_list = []
    for point_3d, circle in zip(point_3d_list, circle_list):
        xyz = [map_uvr_to_xyz(vec, point_3d) for vec in circle]
        axis_points_list.append(xyz)

    return axis_points_list

def draw_aruco_codes(frame: np.ndarray, detections: Dict[int, np.ndarray]) -> None:
    """
    Draw a circle and ID label for each detected ArUco marker onto the frame (in-place).

    Args:
        frame:      BGR image to draw on.
        detections: Dict of marker_id -> [cx, cy, radius] from get_aruco_codes.
    """
    for marker_id, c in detections.items():
        cv.circle(frame, center=(c[0], c[1]), radius=c[2], color=(0, 0, 0), thickness=2)
        cv.putText(frame, str(marker_id), (c[0] + c[2] + 5, c[1]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
def get_aruco_codes(color: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Detect ArUco markers in a BGR image and return their centre and radius.

    Each detected marker is described by its ID (key) and a [cx, cy, radius]
    array (value), where the radius is derived from the minimum enclosing circle
    of the four marker corners — matching the downstream format.

    Args:
        color: BGR input image.

    Retur
        Dict mapping marker ID (int) to uint16 array [cx, cy, radius].
        Empty dict if no markers are detected.
    """
    gray            = cv.cvtColor(color, cv.COLOR_BGR2GRAY)
    dictionary      = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    params          = aruco.DetectorParameters()
    detector        = aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return {}

    result = {}
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        pts          = marker_corners[0].astype(np.float32)  # shape (4, 2)
        (cx, cy), r  = cv.minEnclosingCircle(pts)
        result[int(marker_id)] = np.array([int(cx), int(cy), int(r)], dtype=np.uint16)

    return result


def find_points(color: np.ndarray, mapping: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Locate the three ArUco axis markers in a color image using the ID-to-axis mapping.

    Args:
        color:   BGR image containing the calibration target.
        mapping: Dict mapping axis labels to ArUco IDs
                 (e.g. {"x_axis": 0, "y_axis": 1, "z_axis": 2, ...}).

    Returns:
        Tuple (x, y, z) where each element is a [cx, cy, radius] array.

    Raises:
        CalibrationError: If any required marker ID is not detected in the image.
    """
    detections = get_aruco_codes(color)

    x_id = mapping['x_axis']
    y_id = mapping['y_axis']
    z_id = mapping['z_axis']

    missing = [ax for ax, mid in [('x', x_id), ('y', y_id), ('z', z_id)]
               if mid not in detections]
    if missing:
        raise CalibrationError(f'ArUco marker(s) not detected for axis: {missing}')

    return detections[x_id], detections[y_id], detections[z_id]


def process_buf(buf: list, intrinsics: dict, processor: ImageProcessing, mapping: dict) -> Tuple[list, list]:
    """
    Process all buffered frames in parallel.

    For each frame two independent tasks are submitted to a thread pool:
      1. Guided-filter depth refinement followed by full 3D point cloud reconstruction.
      2. Circle detection and axis-marker identification.

    The degree of parallelism is capped at the number of logical CPU cores.

    Args:
        buf:        List of Data objects (color, depth, robot transform).
        intrinsics: Camera intrinsic parameters (fx, fy, ppx, ppy, dist).
        processor:  ImageProcessing helper (guided filter, temporal median, etc.).
        mapping:    Dict mapping color names to axis labels.

    Returns:
        Tuple (points_3d_list, circles_list) where each element is a list with one
        entry per buffered frame.
    """
    cpu_count   = os.cpu_count() or 1
    max_workers = min(len(buf), cpu_count)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures_3d = [
            executor.submit(filtering_and_construct_3d, d.color, d.depth, intrinsics, processor)
            for d in buf
        ]
        # futures_marker = [
        #     executor.submit(find_points, d.color, mapping)
        #     for d in buf
        # ]
        futures_charuco = [executor.submit(get_charuco_pose, d.color, intrinsics) for d in buf]
        points_3d_list = [f.result() for f in futures_3d]
        # marker_list   = [f.result() for f in futures_marker]
        charuco_list   = [f.result() for f in futures_charuco]


    return points_3d_list, charuco_list


def filtering_and_construct_3d(color: np.ndarray, depth: np.ndarray, intrinsics: dict, processor: ImageProcessing) -> np.ndarray:
    """
    Refine the depth image and back-project every pixel into 3D camera space.

    A guided filter (using the color image as guidance) is applied first to
    suppress noise while preserving depth edges. The filtered depth is then
    unprojected using the pinhole camera model:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

    Args:
        color:      BGR image used as guidance for the filter.
        depth:      Raw depth image (metric, same resolution as color).
        intrinsics: Dict with keys fx, fy, ppx, ppy.
        processor:  ImageProcessing instance providing apply_guided_filter.

    Returns:
        HxWx3 float array where the last dimension contains (X, Y, Z) in
        camera frame coordinates for every pixel.
    """
    depth_filtered = processor.apply_guided_filter(depth, color)

    h, w   = depth_filtered.shape
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["ppx"], intrinsics["ppy"]

    # Build pixel-coordinate grids once — avoids a Python loop over every pixel.
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    X = (u - cx) * depth_filtered / fx
    Y = (v - cy) * depth_filtered / fy
    Z = depth_filtered

    return np.stack([X, Y, Z], axis=-1)


def construct_coordinate_system(Px: np.ndarray, Py: np.ndarray, Pz: np.ndarray, invert_axis: list = [0, 0, 0]) -> np.ndarray:
    """
    Build a 4x4 rigid-body transform that defines a right-handed coordinate frame
    from three 3D points.

    Convention:
        - Origin : Pz
        - X-axis : direction from Pz to Px (normalised)
        - Z-axis : perpendicular to the plane spanned by (Px-Pz) and (Py-Pz)
        - Y-axis : Z cross X  (completes the right-handed system)

    Args:
        Px:          Point defining the X-axis direction relative to the origin.
        Py:          Additional point used to determine the plane (and hence Z-axis).
        Pz:          Origin of the new coordinate frame.
        invert_axis: Three-element list of 0/1 flags; set to 1 to flip the
                     corresponding axis (X, Y, Z order). Y is recomputed after
                     flipping X or Z to preserve right-handedness.

    Returns:
        4x4 homogeneous transformation matrix [R | t; 0 0 0 1].
    """
    x = Px - Pz
    v = Py - Pz

    X = x / np.linalg.norm(x)

    # Z is orthogonal to the plane containing all three points.
    z = np.cross(X, v)
    Z = z / np.linalg.norm(z)

    # Y completes the right-handed triad.
    Y = np.cross(Z, X)

    axes = [X, Y, Z]

    # Flip each axis where invert_axis[i] == 1.
    axes = [-ax if invert_axis[i] else ax for i, ax in enumerate(axes)]

    # After flipping X or Z, recompute Y to preserve right-handedness.
    if invert_axis[0] or invert_axis[2]:
        axes[1] = np.cross(axes[2], axes[0])

    rotation = np.column_stack(axes)
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3]  = Pz

    return T


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
    # start = time.time()
    depth_img    = processor.median_filtering_over_time(depth_frames)
    # print(f'Filtering took {(time.time() - start)} sec.')
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
        "dist": dist[0],
    }


def get_circles(color: np.ndarray) -> np.ndarray:
    """
    Detect circular markers in a BGR image using the Hough Gradient method.

    A bilateral filter is applied first to smooth color noise while keeping
    edges sharp, which improves Hough circle detection robustness.

    Args:
        color: BGR input image.

    Returns:
        Nx3 uint16 array of detected circles, each row [cx, cy, radius].
    """
    img  = cv.bilateralFilter(color, 12, 65, 65)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    circles = cv.HoughCircles(
        gray, cv.HOUGH_GRADIENT,
        dp=1, minDist=50,
        param1=70, param2=40,
        minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS,
    )

    if circles is not None:
        return np.around(circles[0]).astype(np.uint16)
    else:
        return np.zeros((1,3)).astype(np.uint16)


def extract_color(color: np.ndarray, circles: np.ndarray) -> list:
    """
    Sample the dominant BGR color inside each detected circle.

    For each circle a filled circular mask is created. The median pixel value
    within that mask is taken as the representative color, making the estimate
    robust against highlights and shadow gradients.

    Args:
        color:   BGR image.
        circles: Nx3 array of circles [cx, cy, radius].

    Returns:
        List of uint8 BGR arrays, one per circle.

    Raises:
        FalseIndexing: If the number of decoded colors does not match the
                       number of input circles (should never happen in practice).
    """
    decoded_circles = []

    for c in circles:
        mask = np.zeros(color.shape[:2], dtype=np.uint8)
        cv.circle(mask, (c[0], c[1]), c[2], 255, -1)
        roi_pixel = color[mask == 255]
        decoded_circles.append(np.median(roi_pixel, axis=0).astype(np.uint8))

    if len(decoded_circles) != circles.shape[0]:
        raise FalseIndexing

    return decoded_circles


def get_mapping(path: Path) -> dict:
    mapping = load_dict_from_json(path)

    should_include = {'x_axis', 'y_axis', 'z_axis', 'invert_axis'}
    missing = should_include - mapping.keys()
    if missing:
        raise KeyError(f'Missing key(s): {missing}')

    # Axis values must now be ints (ArUco IDs)
    for k in ('x_axis', 'y_axis', 'z_axis'):
        if not isinstance(mapping[k], int):
            raise ValueError(f'{k} must be an int (ArUco ID), got: {type(mapping[k])}')

    invert = mapping['invert_axis']
    if not isinstance(invert, list) or len(invert) != 3 or not all(v in (0, 1) for v in invert):
        raise ValueError(f'invert_axis must be a list of 3 booleans, got: {invert}')

    return mapping


def convert_bgr_to_hsv(bgr: np.ndarray) -> np.ndarray:
    """
    Convert a single BGR pixel to HSV using OpenCV's 8-bit convention.

    OpenCV HSV convention:
        H in [0, 179]   (degrees / 2)
        S in [0, 255]
        V in [0, 255]

    Args:
        bgr: 1D uint8 array [B, G, R].

    Returns:
        1D uint8 array [H, S, V] matching OpenCV's range convention.
    """
    b, g, r = bgr / 255.0

    cmax  = max(r, g, b)
    cmin  = min(r, g, b)
    delta = cmax - cmin

    # --- Hue: angle in the color hexagon ---
    if delta == 0:
        h = 0
    elif cmax == r:
        h = 60 * (((g - b) / delta) % 6)
    elif cmax == g:
        h = 60 * (((b - r) / delta) + 2)
    else:
        h = 60 * (((r - g) / delta) + 4)

    # --- Saturation: normalised chroma ---
    s = 0 if cmax == 0 else delta / cmax

    # --- Value: brightness ---
    v = cmax

    # Scale to OpenCV's 8-bit HSV convention (H/2, S*255, V*255).
    return np.array([h / 2, s * 255, v * 255], dtype=np.uint8)


def get_points(mapping: dict, hsv_list: list, color: np.ndarray, circles: np.ndarray) -> list:
    """
    Assign an Axis label to each detected circle based on its classified color.

    The inverse mapping (color -> axis) is derived from the provided mapping dict.
    If classification is ambiguous (duplicate or missing axes), the user is prompted
    to select the points manually as a fallback.

    Args:
        mapping:   Dict mapping axis label strings to Color enum values.
        hsv_list:  List of HSV arrays, one per detected circle.
        color:     Original BGR image (forwarded to manual selection if needed).
        circles:   Nx3 circle array (forwarded to manual selection if needed).

    Returns:
        List of Axis enum values (or None for unclassified circles), one per circle.
    """
    # Invert the mapping so we can look up the axis by its assigned color.
    inverse    = {v: k for k, v in mapping.items() if k != 'invert_axis'}
    classified = [classify_color(hsv) for hsv in hsv_list]
    result     = [Axis.from_string(inverse[c]) if c in inverse else None for c in classified]

    # Verify that exactly three distinct axes have been identified.
    valid = [r for r in result if r is not None]
    if len(set(valid)) != 3:
        print('PROBLEM occurred, the points are overdetermined.')
        return select_manually(color, circles)

    return result


def select_manually(color: np.ndarray, circles: np.ndarray):
    """
    Fallback: let the user manually assign axis labels to detected circles.

    Called when automatic color classification fails to produce exactly three
    distinct axis labels.

    Args:
        color:   BGR image displayed to the user for reference.
        circles: Detected circles the user should pick from.
    """
    raise NotImplementedError


def classify_color(hsv: np.ndarray) -> Optional['Color']:
    """
    Classify a single HSV pixel into one of the predefined Color categories.

    Each color may define multiple HSV ranges (e.g. red wraps around the hue
    circle). A pixel matches if it falls within *any* range of a color.

    Args:
        hsv: 1D uint8 array [H, S, V] in OpenCV convention.

    Returns:
        The matching Color enum member, or None if no range matches.
    """
    for color, ranges in COLOR_RANGES.items():
        if any(np.all((hsv >= lo) & (hsv <= hi)) for lo, hi in ranges):
            return Color.from_string(color)
    return None


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


def get_charuco_pose(color: np.ndarray, intrinsics: dict, board_size: tuple = (5, 7), square_length: float = 29.7, marker_length: float = 22.0, aruco_dict_id: int = cv.aruco.DICT_4X4_50
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

    # dist = intrinsics["dist"]
    dist = np.zeros(5)  # Assuming no distortion for simplicity; replace with actual dist if available.

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


class Color(Enum):
    """Supported calibration marker colors."""
    RED   = 'red'
    BLUE  = 'blue'
    GREEN = 'green'

    @classmethod
    def from_string(cls, raw: str) -> 'Color':
        """
        Instantiate a Color from its string representation.

        Args:
            raw: Case-insensitive color name (e.g. 'Red', 'BLUE').

        Returns:
            Matching Color enum member.

        Raises:
            ValueError: If the string does not match any supported color.
        """
        raw = raw.strip().lower()
        for color in cls:
            if color.value == raw:
                return color
        raise ValueError(f'Color {raw!r} not supported.')


class Axis(Enum):
    """Coordinate axes used to label calibration marker circles."""
    X = 'x_axis'
    Y = 'y_axis'
    Z = 'z_axis'

    @classmethod
    def from_string(cls, raw: str) -> 'Axis':
        """
        Instantiate an Axis from its string key (as used in mapping.json).

        Args:
            raw: Axis key string, e.g. 'x_axis'.

        Returns:
            Matching Axis enum member.

        Raises:
            ValueError: If the string does not match any supported axis.
        """
        for ax in cls:
            if ax.value == raw:
                return ax
        raise ValueError(f'Unsupported axis: {raw!r}')
    
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
    args   = parser.parse_args(filtered[1:])

    kwargs.update(vars(args))

    return kwargs

def plot_calib_params(rotation: np.ndarray, translation: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    translation = translation.ravel()
    ax.scatter(0,0,0, translation)
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
    return [[303.96, -1.44, 34.017, 116.6, -102.57, 67.25, -162.96], [303.96, -4.82, 25.72, 77.06, -70.928, 65.283, -212.66], [303.88, -45.27, -12.89, 117.86, -68.93, 25.78, -183.575], [304.184, -36.914, -48.26, 129.05, -72.485, -96.02, -199.15], [303.21, -28.26, 47.72, 106.65, -120.66, -99.99, -209.596], [303.22, -28.671, 48.33, 111.33, - 94.4, -8.629, -209.156], [303.246, 22.437, 52.847, 86.99, -178.609, -22.909, -117.668], [296.356, 45.564, 23.91, 44.607, -134.786, -56.17, -117.038], [295.96, -3.58, 62.334, 146.93, -201.735, 47.28, -122.426],[299.08, 24.59, 72.737, 119.25, -205.541, -13.916, -122.384], [299.178, 31.37, 52.86, 107.391, -222.389, -13.874, -85.276], [298.061, 42.588, 39.251, 71.808, -177.096, -38.584, -85.304], [288.662, 36.777, 23.168, 72.504, -149.748, -23.665, -87.585], [311.0, 14.811, 0.655, 114.82, -155.7, 7.19, -87.562], [310.586, -30.542, -5.491, 151.394, -125.278, 46.706, -87.465], [310.535, -3.147, -4.045, 159.86, -228.854, 98.951, -71.52], [295.611, 1.108, -28.090, 121.25, -286.025, 11.882, 30.828], [321.209, 9.887, 15.792, 121.163, -284.549, 54.833, -26.15],[320.824, -42.622, 7.936, 155.648, -270.687, 67.925, -54.392], [315.0, -47.0, -55.0, 149.0, -262.0, 92.0, -39.0], [299.0, 11.0, -1.0, 128.0, 46.0, 129.0, -84.0], [297.0, 40.0, -55.0, 93.0, 12.0, -135.0, -121.0]]