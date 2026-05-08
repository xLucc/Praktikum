import rclpy
import argparse
import sys
import h5py
import logging
import threading
import os
import numpy as np
import cv2 as cv
import copy
from pathlib import Path
from typing import Optional, Union, Tuple, Set
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Lock, Event
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

    cam       = RealSenseCamera(align=True, adv=str(cfg_path / 'high_density.json'))
    robot     = RobotNode()
    node      = threading.Thread(target=robot.spin, daemon=True)
    node.start()
    processor = ImageProcessing(cfg_path=str(cfg_path / 'processing_cfg.json'))

    intrinsics = load_intrinsics(calibration_path / 'intrinsics.hdf5')
    mapping    = get_mapping(cfg_path / 'mapping.json')

    count = 0
    # Buffer storing Data objects (transform, depth, color) for deferred parallel processing.
    buf: list[Data] = []
    h, w = cam.color_resolution
    shape = (w, h, 3) # Turn the resolution into np shape.
    size = int(np.prod(shape)) * np.dtype(np.uint8).itemsize # w * h * 3 * 1

    shm = SharedMemory(create=True, size=size)
    lock = Lock()
    frame_event = Event()

    while True:
        p = Process(target=cam.stream_parallel, args=(shm.name, lock, frame_event, shape), daemon=True)
        logger.info(f'Iteration: {count}')
        p.start()
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
        
            circles = get_circles(frame)

            if not np.all(circles == 0):
                temp = extract_color(circles=circles, color=frame)
                hsv_circles = [(convert_bgr_to_hsv(bgr)) for bgr in temp]
                classified_circles = [classify_color(hsv) for hsv in hsv_circles]
                
                for c, text in zip(circles, classified_circles):
                    cv.circle(frame, center=(c[0], c[1]), radius=c[2], color=(0,0,0), thickness=2)
                    cv.putText(frame, text, (c[0] + c[2] + 5, c[1]), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            else:
                logger.info("Couldn't find any circles.")

            clone = undistort_img(frame, intrinsics)
            cv.imshow('circles', clone)
            cv.imshow('clean', clean)
            key = cv.waitKey(1) & 0xFF

            if key == ord("q"):
                break


        p.terminate()
        p.join()
        cv.destroyAllWindows()
        
        cmd = prompt_cmd('Discard the image or use it.', {Cmd.DISCARD, Cmd.KEEP, Cmd.EXIT})
        
        if cmd == Cmd.EXIT:
            break
        elif cmd == Cmd.DISCARD:
            continue

        count += 1

        depth = take_pic(cam, processor)
        T     = get_robot_transform(robot)
        clean = clean if clean is not None else np.zeros(shape)

        data = Data(T=T, depth=depth, color=undistort_img(clean, intrinsics))
        buf.append(data)
    

    shm.close()
    shm.unlink()

    store_buf(buf, buf_path)

    # Process all buffered frames once the capture loop has ended.
    point_3d_list, circle_list = process_buf(buf, intrinsics, processor, mapping)
    axis_list   = map_circle_to_3d(point_3d_list, circle_list)
    invert_axis = mapping['invert_axis']
    T_cam_2_target_list = [
        construct_coordinate_system(Px=Px, Py=Py, Pz=Pz, invert_axis=invert_axis)
        for Px, Py, Pz in axis_list
    ]
    T_base_2_tfp_list = get_robot_info_from_buf(buf)

    R_target_2_cam, t_target_2_cam = deconstruct_T_into_R_and_t(T_cam_2_target_list)
    R_tfp_2_base,   t_tfp_2_base   = deconstruct_T_into_R_and_t(T_base_2_tfp_list)

    R_cam_2_tfp, t_cam_2_tfp = cv.calibrateHandEye(
        R_gripper2base=R_tfp_2_base,
        t_gripper2base=t_tfp_2_base,
        R_target2cam=R_target_2_cam,
        t_target2cam=t_target_2_cam,
    )
    store_calib(R_cam_2_tfp, t_cam_2_tfp, hand_eye_path)


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


def map_circle_to_3d(point_3d_list: list, circle_list: list) -> list:
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


def find_points(color: np.ndarray, mapping: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Locate the three calibration-marker circles (X, Y, Z axis markers) in a color image.

    The function detects all circles via Hough transform, extracts their dominant color,
    converts to HSV, and maps each detected color to the corresponding coordinate axis
    using the provided color-to-axis mapping.

    Args:
        color:   BGR image containing the calibration target.
        mapping: Dict loaded from mapping.json assigning color names to axis labels
                 (e.g. {"x_axis": "red", "y_axis": "blue", "z_axis": "green", ...}).

    Returns:
        Tuple (x, y, z) where each element is a circle descriptor array [cx, cy, r]
        corresponding to the X, Y, and Z axis markers respectively.
    """
    circles = get_circles(color=color)

    bgr_decoded_circles = extract_color(color=color, circles=circles)
    hsv_decoded_circles = [convert_bgr_to_hsv(bgr) for bgr in bgr_decoded_circles]

    # Ensure all string values in the mapping are converted to Color enum instances.
    mapping = {k: Color.from_string(v) if isinstance(v, str) else v for k, v in mapping.items()}

    axis_list = get_points(mapping, hsv_decoded_circles, color, circles)

    x = circles[axis_list.index(Axis.X)]
    y = circles[axis_list.index(Axis.Y)]
    z = circles[axis_list.index(Axis.Z)]

    return x, y, z


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
        futures_circles = [
            executor.submit(find_points, d.color, mapping)
            for d in buf
        ]
        points_3d_list = [f.result() for f in futures_3d]
        circles_list   = [f.result() for f in futures_circles]

    return points_3d_list, circles_list


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
    euler_cords = robot.sys_frame('tfc', 'base')

    if euler_cords is None:
        raise InvalidRobotTransform

    # An all-zero response typically means the robot has not been homed yet.
    if np.all(euler_cords == 0):
        raise InvalidRobotTransform

    angles = euler_cords[1, :]
    r      = R.from_euler('XYZ', angles, degrees=True)

    T = np.eye(4)
    T[:3, :3] = r.as_matrix()
    T[:3, 3]  = euler_cords[0, :]

    return T


def take_pic(camera: RealSenseCamera, processor: ImageProcessing, num_depth: int = 15, num_color: int = 1) -> np.ndarray:
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
        return np.zeros((1,3))


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
    """
    Load and validate the color-to-axis mapping from a JSON file.

    The JSON must contain exactly the keys: x_axis, y_axis, z_axis, invert_axis.
    All axis values must be strings; invert_axis must be a list of three 0/1 integers.

    Args:
        path: Path to mapping.json.

    Returns:
        Validated mapping dict.

    Raises:
        KeyError:   If any required key is missing.
        ValueError: If any value has an unexpected type or invert_axis is malformed.
    """
    mapping = load_dict_from_json(path)

    should_include = {'x_axis', 'y_axis', 'z_axis', 'invert_axis'}
    missing = should_include - mapping.keys()
    if missing:
        raise KeyError(f'Missing key(s): {missing}')

    for v in mapping.values():
        if not isinstance(v, (str, list)):
            raise ValueError(f'Wrong type for value {v!r}. Expected str or list, got {type(v)}')

    invert = mapping['invert_axis']
    if not isinstance(invert, list) or len(invert) != 3 or not all(v in (0, 1) for v in invert):
        raise ValueError(f'invert_axis must be a list of 3 booleans, e.g. [1, 0, 0], got: {invert}')

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


if __name__ == '__main__':
    filtered = rclpy.utilities.remove_ros_args(sys.argv)

    parser = argparse.ArgumentParser()
    args   = parser.parse_args(filtered[1:])

    main(**vars(args))