import h5py
import open3d as o3
import cv2 as cv
import numpy as np
import debugpy
import sys
import matplotlib.pyplot as plt
from cv2.ximgproc import jointBilateralFilter, guidedFilter
from sklearn.cluster import DBSCAN
from pathlib import Path
from bin_picking.common.helper import load_dict_from_json


def main():
    # Loading data and format it
    # debugpy.listen(('0.0.0.0', 4865))
    # debugpy.wait_for_client()
    path = Path.cwd() / 'data' / 'image_processing'
    rgb = cv.imread(str(path / 'color_image.png'))
    rgb_gray = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)
    file = h5py.File(str(path / 'data_set.hdf5'), 'r')
    depth_data: np.ndarray = file['depth'][:].astype(np.float32)



    # valid = (depth_data < 800) & (depth_data > 400)
    # depth_data[~valid] = 0


    depth_u8 = cv.normalize(depth_data, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)


    edges = cv.Canny(rgb_gray, 50, 150)
    guide = cv.addWeighted(rgb_gray.astype(np.float32), 0.6, edges.astype(np.float32), 0.4, 0)
    # depth_data = depth_data.astype(np.uint8)
    file.close()

    depth_camera_info = load_dict_from_json(path / 'depth_camera_info.json')
    # rgb_camera_info = load_dict_from_json(path / 'color_camera_info.json') # Uncomment if needed.

    rgb_resolution = rgb_gray.shape
    depth_resolution = depth_data.shape

    intrinsic = depth_camera_info['intrinsic']
    fx, cx, fy, cy = intrinsic[0], intrinsic[2], intrinsic[4], intrinsic[5]
    depth_img1 = cv.normalize(depth_data, None, 0, 255, cv.NORM_MINMAX)
    # depth = jointBilateralFilter(joint=rgb.astype(np.float32), src=depth_data, d=9, sigmaColor=0.1, sigmaSpace=5)
    depth = guidedFilter(guide=rgb_gray.astype(np.float32), src=depth_data, radius=200, eps=1e-5)
    depth_img2 = cv.normalize(depth, None, 0, 255, cv.NORM_MINMAX)
    cv.imshow('depth2', depth_img2)
    cv.imshow('depth1', depth_img1)
    cv.waitKey(0)

    def convert(data): 
        # valid = (data < 800) & (data > 400)
        # data[~valid] = 0
        c, v = np.meshgrid(np.arange(depth_resolution[1]),np.arange(depth_resolution[0]))

        X  = ((c - cx) / fx * data).flatten()
        Y = ((v - cy) / fy * data).flatten()
        Z = data.flatten()


        xyz = np.column_stack([X,Y,Z])
        return xyz


    pcd = o3.geometry.PointCloud(points=o3.utility.Vector3dVector(convert(depth_data)))
    # points = np.asarray(convert(depth_data))
    # db = DBSCAN(eps=3, min_samples=10).fit(points)
    # labels = db.labels_
    # unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    # chip_label = unique[np.argmin(counts)]
    # chip_points = points[labels == chip_label]
    # pcd = o3.geometry.PointCloud(points= o3.utility.Vector3dVector(chip_points))
    o3.visualization.draw_geometries([pcd])



if __name__ == '__main__':
    main()
