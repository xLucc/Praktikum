import logging
import warnings
import numpy as np
import cv2 as cv
import open3d as o3d
from typing import Optional
from pathlib import Path
from bin_picking.common.helper import load_dict_from_json


class ImageProcessing:
    """
    This class uses the guided filter to smooth the depth image while preserving edges.
    """

    def __init__(self, cfg_path: Optional[str] = None):

        self._cfg = load_dict_from_json(Path(cfg_path)) if cfg_path else {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._cfg = self._verify_cfg(self._cfg)

        self.guided_radius = self._cfg.get("guided_radius")
        self.guided_eps = self._cfg.get("guided_eps")
        self.sharp_lambda = self._cfg.get("sharp_lambda")
        self.residual_clamp = self._cfg.get("residual_clamp")

        self.logger.info(f"Configuration: {self._cfg}")
        self.logger.info("ImageProcessing initialized with the above configuration.")

    # This function applies the guided filter to the input depth image using the color image as a guide.
    def apply_guided_filter(
        self, depth_m: np.ndarray, color: np.ndarray, cfg: Optional[dict] = None
    ) -> np.ndarray:
        """
        Applies the guided filter to the input depth image using the color image as a guide.

        Parameters:

            depth_m (np.ndarray): The input depth image as a 2D array of shape (H, W) with dtype float32. Unit should be in meters.
            color (np.ndarray): The input color image as a 3D array of shape (H, W, 3) with dtype uint8.
            cfg (dict, optional): A dictionary containing filter configuration parameters.
                Expected keys:
                    - "sobel_ksize": int, kernel size for Sobel filter (must be odd).
                    - "gaussian_ksize": tuple, kernel size for Gaussian blur (must be odd, e.g. (3, 3)).

        Returns:
            np.ndarray: The sharpened depth image after applying the guided filter, with the same shape and dtype as the input depth image.

        Raises:
            TypeError: If the input depth or color images are not numpy arrays, or if the color image is not of type uint8, or if the depth image is not of type float32.
            ValueError: If the depth and color images do not have the same shape, or if the kernel sizes in cfg are not valid (e.g., not odd).
            KeyError: If the required keys are missing in the cfg dictionary.
        """

        self._verify_input(depth_m, color)

        if cfg is None:
            cfg = {}
        elif cfg:
            self._verify_filter_cfg(cfg)

        sobel_ksize = cfg.get("sobel_ksize", 5)
        gaussian_ksize = cfg.get("gaussian_ksize", (3, 3))

        gray = cv.cvtColor(color, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=sobel_ksize)
        gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=sobel_ksize)
        edge_mag = np.sqrt(gx * gx + gy * gy)
        edge_mag = edge_mag / (edge_mag.max() + 1e-6)
        guide = cv.GaussianBlur(edge_mag, gaussian_ksize, 0)

        smooth = cv.ximgproc.guidedFilter(
            guide=guide, src=depth_m, radius=self.guided_radius, eps=self.guided_eps
        )
        residual = depth_m - smooth
        residual = np.clip(residual, -self.residual_clamp, self.residual_clamp)

        return depth_m + self.sharp_lambda * residual * guide

    def median_filtering_over_time(self, img_list) -> np.ndarray:
        """
        Applies median filtering over a list of images to reduce noise.

        Parameters:
            img_list (list or np.ndarray): A list or numpy array of images (2D arrays) to be filtered. Each image should have the same shape.

        Returns:
            np.ndarray: A single image resulting from the median filtering of the input images, with the same shape as the input images and dtype float32.

        Raises:
            TypeError: If img_list is not a list or numpy array.
            ValueError: If img_list is empty.
        """

        if not isinstance(img_list, (np.ndarray, list)):
            self.logger.error(
                f"Expected img_list to be a list or numpy array, got {type(img_list)}"
            )
            raise TypeError

        if len(img_list) == 0:
            self.logger.warning("Received an empty img_list. Returning an empty array.")
            raise ValueError

        img_list = img_list if isinstance(img_list, list) else img_list.tolist()

        images = np.stack(img_list, axis=-1).astype(np.float32)
        images = np.where(images == 0, np.nan, images)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nan_to_num(np.nanmedian(images, axis=2), nan=0).astype(np.float32)

    # This function generates a point cloud from the input depth and color images.
    # This function is a inplace operation, meaning it modifies the input point cloud object directly.
    def generate_point_cloud(
        self,
        depth_m: np.ndarray,
        color: np.ndarray,
        intrinsics: dict,
        pcd: o3d.geometry.PointCloud,
        zmin=0.3,
        zmax=3.0,
    ):
        """
        Generates a point cloud from the input depth and color images. The point cloud is generated in-place, meaning the input point cloud object is modified directly.

        Parameters:
            depth_m (np.ndarray): The input depth image as a 2D array of shape (H, W) with dtype float32. Unit should be in meters.
            color (np.ndarray): The input color image as a 3D array of shape (H, W, 3) with dtype uint8.
            intrinsics (dict): A dictionary containing the camera intrinsic parameters with keys "fx", "fy", "ppx", and "ppy".
            pcd (o3d.geometry.PointCloud): An Open3D PointCloud object that will be modified in-place to contain the generated point cloud.
            zmin (float, optional): The minimum depth value to be included in the point cloud. Default is 0.3 meters.
            zmax (float, optional): The maximum depth value to be included in the point cloud. Default is 3.0 meters.

        Raises:
            TypeError: If the input depth or color images are not numpy arrays, if the color image is not of type uint8, if the depth image is not of type float32, if intrinsics is not a dict, if pcd is not an Open3D PointCloud object, or if zmin and zmax are not numbers.
            ValueError: If the depth and color images do not have the same shape, if required keys are missing in intrinsics, if zmin or zmax are not positive, or if zmin is greater than or equal to zmax.
        """

        self._verify_point_cloud_params(depth_m, color, intrinsics, pcd, zmin, zmax)

        # Setup for point cloud generation
        h, w = depth_m.shape
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["ppx"], intrinsics["ppy"]
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        mask = (depth_m > zmin) & (depth_m < zmax)

        # Convert color image to RGB and normalize, required for Open3D
        pcd_color = cv.cvtColor(color, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Calculate 3D coordinates from depth image
        X = ((u - cx) * depth_m / fx).flatten()
        Y = ((v - cy) * depth_m / fy).flatten()
        Z = depth_m.flatten()

        # Stack X, Y, Z to get the point cloud coordinates and reshape color to match the point cloud
        points = np.column_stack([X, Y, Z])
        colors = pcd_color.reshape(-1, 3)

        # Set the points and colors of the point cloud, applying the mask to filter out invalid points
        pcd.points = o3d.utility.Vector3dVector(points[mask.flatten()])
        pcd.colors = o3d.utility.Vector3dVector(colors[mask.flatten()])

    def _verify_cfg(self, cfg: dict) -> dict:

        default_cfg = {
            "guided_radius": 5,
            "guided_eps": 1e-3,
            "sharp_lambda": 0.5,
            "residual_clamp": 0.01,
        }

        should_keys = {"guided_radius", "guided_eps", "sharp_lambda", "residual_clamp"}

        if should_keys - cfg.keys():
            self.logger.warning(f"Missing keys in cfg: {should_keys - cfg.keys()}")
            self.logger.warning("Using default values for the config.")
            return default_cfg

        checks = [
            ("guided_radius", cfg["guided_radius"], 1, 15),
            ("guided_eps", cfg["guided_eps"], 1e-4, 2e-1),
            ("sharp_lambda", cfg["sharp_lambda"], 0.0, 1.0),
            ("residual_clamp", cfg["residual_clamp"], 0.01, 0.5),
        ]

        for name, value, min_val, max_val in checks:
            if not (min_val <= value <= max_val):
                self.logger.warning(
                    f"Invalid value for {name}: {value}. Should be in [{min_val}, {max_val}]"
                )
                self.logger.warning(f"Clamping {name} to valid range.")
                cfg[name] = max(min(value, max_val), min_val)

        return cfg

    # This function checks if the input depth and color images are valid for processing.
    def _verify_input(self, depth_m, color):

        if not isinstance(depth_m, np.ndarray) or not isinstance(color, np.ndarray):
            self.logger.error("Depth image and edge guide must be numpy arrays.")
            raise TypeError

        if not depth_m.shape == color.shape[:2]:
            self.logger.error("Depth image and edge guide must have the same shape.")
            raise ValueError

        if color.dtype != np.uint8:
            self.logger.error(
                f"Expected color image to be of type uint8, got {color.dtype}"
            )
            raise TypeError

        if depth_m.dtype != np.float32:
            self.logger.error(
                f"Expected depth image to be of type float32, got {depth_m.dtype}"
            )
            raise TypeError

    # Function expects user to input useful cfg parameters for the filter, e.g. kernel sizes for Sobel and Gaussian filters.
    def _verify_filter_cfg(self, cfg):

        if not isinstance(cfg, dict):
            self.logger.error(f"Expected cfg to be a dict, got {type(cfg)}")
            raise TypeError

        must_keys = {"sobel_ksize", "gaussian_ksize"}

        if must_keys - cfg.keys():
            self.logger.error(f"Missing keys in filter cfg: {must_keys - cfg.keys()}")
            raise KeyError

        # Check if sobel_ksize is a positive integer
        if not isinstance(cfg["sobel_ksize"], int):
            self.logger.error(
                f"Expected sobel_ksize to be an integer, got {type(cfg['sobel_ksize'])}"
            )
            raise TypeError

        if cfg["sobel_ksize"] <= 0:
            self.logger.error(
                f"Expected sobel_ksize to be a positive integer, got {cfg['sobel_ksize']}"
            )
            raise ValueError

        if not (
            isinstance(cfg["gaussian_ksize"], tuple) and len(cfg["gaussian_ksize"]) == 2
        ):
            self.logger.error(
                f"Invalid gaussian_ksize: {cfg['gaussian_ksize']}. Must be a tuple of (width, height)."
            )
            raise TypeError

        if not (cfg["sobel_ksize"] & 1):
            self.logger.error(
                f"Sobel kernel size must be odd, got {cfg['sobel_ksize']}"
            )
            raise ValueError

        if not (cfg["gaussian_ksize"][0] & 1) or not (cfg["gaussian_ksize"][1] & 1):
            self.logger.error(
                f"Gaussian kernel size must be odd, got {cfg['gaussian_ksize']}"
            )
            raise ValueError

    # This function checks if the parameters for point cloud generation are valid.
    def _verify_point_cloud_params(self, depth_m, color, intrinsics, pcd, zmin, zmax):

        if not isinstance(depth_m, np.ndarray) or not isinstance(color, np.ndarray):
            self.logger.error("Depth and color must be numpy arrays.")
            raise TypeError

        if depth_m.shape != color.shape[:2]:
            self.logger.error(
                "Depth and color images must have the same height and width."
            )
            raise ValueError

        if not isinstance(intrinsics, dict):
            self.logger.error(
                f"Expected intrinsics to be a dict, got {type(intrinsics)}"
            )
            raise TypeError

        required_intrinsic_keys = {"fx", "fy", "ppx", "ppy"}
        if required_intrinsic_keys - intrinsics.keys():
            self.logger.error(
                f"Missing keys in intrinsics: {required_intrinsic_keys - intrinsics.keys()}"
            )
            raise KeyError

        if not isinstance(pcd, o3d.geometry.PointCloud):
            self.logger.error(
                f"Expected pcd to be an Open3D PointCloud object, got {type(pcd)}"
            )
            raise TypeError

        if not (isinstance(zmin, (int, float)) and isinstance(zmax, (int, float))):
            self.logger.error(
                f"Expected zmin and zmax to be numbers, got {type(zmin)} and {type(zmax)}"
            )
            raise TypeError

        if zmin <= 0 or zmax <= 0:
            self.logger.error(
                f"Expected zmin and zmax to be positive, got {zmin} and {zmax}"
            )
            raise ValueError

        if zmin >= zmax:
            self.logger.error(
                f"Expected zmin to be less than zmax, got zmin={zmin} and zmax={zmax}"
            )
            raise ValueError
