from __future__ import annotations

import numpy as np
import open3d as o3d
from ultralytics import YOLO
from ultralytics.engine.results import Results

from bin_picking.common.helper import get_project_dir

_MODEL: YOLO | None = None


def _weights_path():
    return get_project_dir() / 'data' / 'weights' / 'best.pt'


def get_model() -> YOLO:
    '''Loads the segmentation model once and caches it -- never reloaded per shot.'''
    global _MODEL

    if _MODEL is None:
        path = _weights_path()

        if not path.exists():
            raise FileNotFoundError(
                f"Model weights not found at '{path}'. Drop the trained 'best.pt' there "
                'once training has converged.'
            )

        _MODEL = YOLO(str(path))

    return _MODEL


def run_inference(img6: np.ndarray) -> list[Results]:
    '''
    Runs the 6-channel BGRXYZ model.

    Feed [0,255]-range values as-is -- this patched ultralytics fork (see PIPELINE.md
    stage 3) has no /255 normalization step and no channel-order swap for 6-channel
    input, both verified directly against the fork's predictor code and empirically
    against the model's actual forward() input tensor.
    '''
    model = get_model()
    return model(img6, imgsz=1280, rect=False, retina_masks=True, verbose=False)
