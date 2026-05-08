import argparse
import h5py
import threading
import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import os
import time
from enum import Enum
from pathlib import Path
from typing import Optional, Set


def main(
    event: Optional[threading.Event] = None,
    quit: Optional[threading.Event] = None,
    **kwargs,
):
    # Initialize the camera pipeline.
    if not kwargs['skip']:
        pipeline = init_pipeline(**kwargs)

    # Get the path to save the images, either from the repo root or relative to this file.
    try:
        path = find_repo_root() / "data" / "calibration"
    except FileNotFoundError:
        path = Path(__file__).parent / "data" / "calibration"

    path.mkdir(parents=True, exist_ok=True)

    # Capture images until the user decides to quit.
    try:
        if not kwargs['skip']:
            gather_images(pipeline, path, event, **kwargs)

    except KeyboardInterrupt:
        print("Quitting.")

    except SystemExit:
        print("Exiting.")
        pipeline.stop()
        cv.destroyAllWindows()
        os._exit(0)

    except RuntimeError as e:
        print(f"Error: {e}")

    finally:
        if not kwargs["skip"]:
            pipeline.stop()
        cv.destroyAllWindows()
        if quit is not None:
            quit.set()
        calibrate(path=path, **kwargs)


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


# Capture a single frame from the camera pipeline, displaying it in a window and waiting for the user to either save it or quit.
def capture_frame(pipeline: rs.pipeline) -> np.ndarray:
    while True:
        frame = pipeline.wait_for_frames().get_color_frame()
        if not frame:
            continue

        img = np.asanyarray(frame.get_data())
        cv.imshow("img", img)
        key = cv.waitKey(1) & 0xFF
        if key == ord("s"):
            return img
        elif key == ord("q"):
            raise KeyboardInterrupt
        elif key == ord("e"):
            raise SystemExit


# Prompt the user for a command until a valid one is given or the maximum number of tries is reached.
def prompt_cmd(prompt: str, valid: Set["Cmd"], max_tries: int = 5) -> Cmd:
    options = "/".join(cmd.value for cmd in valid)
    for _ in range(max_tries):
        cmd = Cmd.from_input(input(f"{prompt} [{options}]: "))
        if cmd in valid:
            return cmd
        print("Invalid input.")
    raise RuntimeError(f"No valid input after {max_tries} tries.")


# Initialize the camera pipeline with the given parameters.
def init_pipeline(**kwargs) -> rs.pipeline:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color,
        kwargs.get("width", 1280),
        kwargs.get("height", 720),
        rs.format.bgr8,
        kwargs.get("fps", 30),
    )
    pipeline.start(config)
    return pipeline


# Calibrate the camera using the captured images and openCV's checkboard detection and calibration functions.
# The calibration results are saved to an hdf5 file for later use.
def calibrate(path, **kwargs):
    print("Calibrating...")
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    shape = kwargs.get("check_shape", (8, 6))

    objectPoints = []
    imagePoints = []
    valid_files= []

    object_points = np.zeros((1, shape[0] * shape[1], 3), np.float32)
    object_points[0, :, :2] = np.mgrid[0 : shape[0], 0 : shape[1]].T.reshape(-1, 2)

    images = path.glob("*.png")

    if not images:
        print(f"No images found in {path}.")
        return

    # Find the chessboard corners in each image and add the corresponding object and image points to the lists.
    for fname in images:
        img = cv.imread(str(fname))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, (shape[0],shape[1]), None)

        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objectPoints.append(object_points)
            imagePoints.append(corners2)
            valid_files.append(fname)

            if kwargs.get("show", True):
                cv.drawChessboardCorners(img, (shape[0], shape[1]), corners2, ret)
                cv.imshow("img", img)
                cv.waitKey(500)
        else:
            print(f'Discarded: {fname}')

    cv.destroyAllWindows()

    # Calibrate the camera and print the reprojection error.
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objectPoints, imagePoints, gray.shape[::-1], None, None, flags=cv.CALIB_FIX_K3
    )

    print(f"Reprojection error: {ret:.4f} px")


    per_image_errors = []
    for i, fname in enumerate(valid_files):
        projected, _ = cv.projectPoints(objectPoints[i], rvecs[i], tvecs[i], mtx, dist)
        err = cv.norm(imagePoints[i], projected, cv.NORM_L2) / np.sqrt(len(projected))
        per_image_errors.append((fname, err))
        print(f"{err:.4f} px  {fname}")
    
    print(f'Number of pics: {len(valid_files)}')

    # Save the calibration results to an hdf5 file.
    file = h5py.File(path / "intrinsics.hdf5", "w")
    file.create_dataset("intrinsics/mtx", data=mtx)
    file.create_dataset("intrinsics/dist", data=dist)
    file.flush()
    file.close()


def find_repo_root(start: Path = Path(__file__).parent) -> Path:
    for parent in [start, *start.parents]:
        if (parent / ".git").exists():
            return parent
    raise FileNotFoundError(f"No git repo found from {start}")


def gather_images(
    pipeline: rs.pipeline, path: Path, event: Optional[threading.Event] = None, **kwargs
):
    count = 0
    while True:

        img = capture_frame(pipeline)

        cmd = prompt_cmd("Keep the image?", {Cmd.KEEP, Cmd.DISCARD, Cmd.QUIT, Cmd.EXIT})

        if cmd == Cmd.KEEP:
            count += 1
            cv.imwrite(
                str(
                    path
                    / f"Image_{count}_resolution{kwargs.get('width', 1280)}x{kwargs.get('height', 720)}_{str(time.time()).strip()}.png"
                ),
                img,
            )

            if event is not None:
                event.set()

        elif cmd == Cmd.DISCARD:
            print("Discarded.")
        elif cmd == Cmd.QUIT:
            break
        elif cmd == Cmd.EXIT:
            raise SystemExit

        cmd = prompt_cmd("Continue?", {Cmd.CONTINUE, Cmd.QUIT, Cmd.EXIT})

        if cmd == Cmd.QUIT:
            break
        elif cmd == Cmd.EXIT:
            raise SystemExit

        print("Please reposition.\n")
        print(f"Image number {count}\n")


if __name__ == "__main__":
    # Parse command line arguments and run the main function.
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--check_shape", type=int, nargs=2, default=(8, 6))
    parser.add_argument("--show", action="store_true", default=True)
    parser.add_argument("--no-show", dest="show", action="store_false")
    parser.add_argument("--skip", action='store_true', default=False)

    main(**vars(parser.parse_args()))
