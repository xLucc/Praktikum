from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bin_picking.common.helper import get_project_dir, load_dict_from_json
from bin_picking.pipeline.reachability import load_chain

# Tool-tip offset from the flange (tfc) frame, in tfc-local coordinates.
TCP_OFFSET = np.array([4.587, 0.906, 187.924])

# Suction tube tip distance from TFC along tool Z.
TCP_TUBE_END_MM = 197.5

# Max allowed deviation of the approach axis from camera +Z.
MAX_APPROACH_ANGLE_DEG = 25.0


@dataclass
class PickResult:
    transform: np.ndarray
    class_name: str
    confidence: float
    area: float


def load_cfg() -> dict:
    cfg_dir = get_project_dir() / 'data' / 'cfg'
    return {
        'filter':     load_dict_from_json(cfg_dir / 'filter_cfg.json'),
        'real_areas': load_dict_from_json(cfg_dir / 'real_areas.json'),
        'home_pose':  load_dict_from_json(cfg_dir / 'home_pose.json')['joint_config'],
        'motion':     load_dict_from_json(cfg_dir / 'motion_cfg.json'),
        'bin':        load_dict_from_json(cfg_dir / 'bin_cfg.json'),
        'chain':      load_chain(),
    }
