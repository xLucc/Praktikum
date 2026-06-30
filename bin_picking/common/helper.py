import json
import logging
import h5py
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def get_package_root() -> Path:
    """Walk ancestor directories until a ``src/`` sibling is found and return it."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / 'src').is_dir():
            return parent / 'src'
    raise RuntimeError(f'Found no src directory in ancestors of {current}')


def get_project_dir() -> Path:
    """Return the path to the ``bin_picking`` package root inside the workspace ``src/``."""
    return get_package_root() / 'bin_picking'


def load_dict_from_json(path: Path) -> dict:
    """Load and return a JSON file as a Python dict."""
    with open(path, 'r') as f:
        return json.load(f)


def load_intrinsics(path: Path) -> dict:
    '''
    Loads camera intrinsics from an HDF5 file (expects 'intrinsics/mtx' and 'intrinsics/dist').

    Returns:
        dict with keys: fx, fy, ppx, ppy, dist.
    '''
    with h5py.File(path, 'r') as f:
        mtx = f['intrinsics']['mtx'][:]
        dist = f['intrinsics']['dist'][:]

    return {
        'fx': float(mtx[0, 0]),
        'fy': float(mtx[1, 1]),
        'ppx': float(mtx[0, 2]),
        'ppy': float(mtx[1, 2]),
        'dist': dist.ravel(),
    }


def load_hand_eye(path: Path) -> np.ndarray:
    '''Loads the camera-to-TCP hand-eye calibration transform (4x4) from an HDF5 file.'''
    with h5py.File(path, 'r') as f:
        return f['T_cam2tfp'][:]


def write_json_from_dict(data: dict, path: Path) -> bool:
    try:
        json_data = json.dumps(data)
    except Exception as e:
        logger.error("Can't convert the data to JSON: %s", e)
        return False

    try:
        with open(path, 'w') as f:
            f.write(json_data)
    except Exception as e:
        logger.error('Error occurred while writing the file: %s', e)
        return False

    return True
