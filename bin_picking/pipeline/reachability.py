from __future__ import annotations

import logging
import time
from typing import Optional

import h5py
import numpy as np
from ikpy.chain import Chain

from bin_picking.common.helper import get_project_dir

logger = logging.getLogger(__name__)

# TCP offset from TFC in TFC-local coordinates (mm) — must match TCP_OFFSET in run.py.
# Subtracting this from the TCP target gives the TFC target for IK.
_TCP_OFFSET_MM = np.array([4.587, 0.906, 187.924])

# Chain link indices that correspond to actuated joints (joint0..joint6).
# Fixed parallel-mechanism joints (jointPL2/4/6) and the base are inactive.
_ACTIVE_MASK = [False, True, True, False, True, True, False, True, True, False, True]
_ACTIVE_IDXS = [i for i, a in enumerate(_ACTIVE_MASK) if a]  # [1,2,4,5,7,8,10]

# Conservative capsule radii per chain link index (mm).
_RADII_MM: dict[int, float] = {
    0:  0.0,   # base frame (fixed origin)
    1: 80.0,   # joint0 → link1 (shoulder column)
    2: 70.0,   # joint1 → linkJ2 (elbow / parallel pivot)
    3: 65.0,   # jointPL2 (fixed offset, outer arm)
    4: 65.0,   # joint2 → link3 (forearm 1)
    5: 60.0,   # joint3 → linkJ4 (wrist pivot / parallel)
    6: 55.0,   # jointPL4 (fixed offset)
    7: 50.0,   # joint4 → link5 (forearm 2)
    8: 45.0,   # joint5 → linkJ6 (wrist 2)
    9: 40.0,   # jointPL6 (fixed offset)
    10: 40.0,  # joint6 → end_effector (TFC)
}

# Custom TCP assembly sections from TFC along tool Z (top → suction tip):
#   (start_mm_from_tfc, end_mm_from_tfc, radius_mm)
_TCP_SECTIONS = [
    (0.0,   11.0,  120.0),   # wide mounting plate
    (11.0,  27.5,   80.0),   # transition piece
    (27.5, 197.5,   12.0),   # suction tube (enters bin)
]
_TCP_SECTION_NAMES = ['mounting plate', 'transition piece', 'suction tube']

_IK_TOL_MM = 15.0  # IK convergence threshold (mm)

# Camera (D430 module) capsule in camera frame (mm).
# Axis along camera X; origin is 12 mm from the right end, 11 mm from the bottom.
# Body: 70.7 x 14.0 x 10.52 mm → radius = max(14.0, 10.52)/2 = 7.0 mm.
_CAM_CAPSULE_P1 = np.array([-12.0,  4.0, 0.0])   # right end in camera frame
_CAM_CAPSULE_P2 = np.array([ 58.7,  4.0, 0.0])   # left  end in camera frame
_CAM_CAPSULE_R  = 7.0                              # mm

_hand_eye: Optional[np.ndarray] = None

# Track which collision types have been plotted this session to avoid flooding plots/.
_plotted_types: set[str] = set()


def _get_hand_eye() -> np.ndarray:
    global _hand_eye
    if _hand_eye is None:
        p = get_project_dir() / 'data' / 'calibration' / 'hand_eye.hdf5'
        with h5py.File(p, 'r') as f:
            _hand_eye = f['T_cam2tfp'][:]
    return _hand_eye


def load_chain() -> Chain:
    urdf = get_project_dir() / 'data' / 'urdf' / 'kr1205.urdf'
    return Chain.from_urdf_file(
        str(urdf),
        base_elements=['base'],
        active_links_mask=_ACTIVE_MASK,
    )


def _to_ikpy_joints(angles_deg: list) -> np.ndarray:
    joints = np.zeros(len(_ACTIVE_MASK))
    for i, idx in enumerate(_ACTIVE_IDXS):
        joints[idx] = np.deg2rad(angles_deg[i])
    return joints


def _solve_ik(chain: Chain, T_world_mm: np.ndarray,
              home_joints_deg: list) -> Optional[np.ndarray]:
    """Solve IK for a TCP target in world frame (mm). Returns full chain joint vector or None."""
    T_m = T_world_mm.copy().astype(float)
    # Shift from TCP target to TFC target: TFC = TCP - R * tcp_offset
    T_m[:3, 3] -= T_m[:3, :3] @ _TCP_OFFSET_MM
    T_m[:3, 3] /= 1000.0  # mm → m, only at the ikpy boundary

    initial = _to_ikpy_joints(home_joints_deg)
    sol = chain.inverse_kinematics(
        T_m[:3, 3],
        target_orientation=T_m[:3, :3],
        initial_position=initial,
        orientation_mode='all',
    )

    err_mm = np.linalg.norm(chain.forward_kinematics(sol)[:3, 3] - T_m[:3, 3]) * 1000.0
    return sol if err_mm <= _IK_TOL_MM else None


def ik_joints(chain: Chain, T_world_mm: np.ndarray,
              home_joints_deg: list) -> Optional[list]:
    """
    Solve IK for a target given in world frame (mm).
    Returns 7 joint angles in degrees, or None if IK didn't converge.
    All inputs and outputs in robot-native units (mm, degrees) — no metres in callers.
    """
    sol = _solve_ik(chain, T_world_mm, home_joints_deg)
    if sol is None:
        return None
    return [np.degrees(sol[idx]) for idx in _ACTIVE_IDXS]


def _bin_faces(bin_cfg: dict) -> list:
    cx, cy, cz = bin_cfg['world_pos']
    w, d, h    = bin_cfg['size']
    hw, hd, hh = w / 2, d / 2, h / 2
    z_floor, z_top = cz - hh, cz + hh
    return [
        {'axis': 2, 'pos': z_floor, 'bounds': [(cx - hw, cx + hw), (cy - hd, cy + hd)], 'name': 'floor'},
        {'axis': 0, 'pos': cx - hw, 'bounds': [(cy - hd, cy + hd), (z_floor, z_top)],   'name': 'X- wall'},
        {'axis': 0, 'pos': cx + hw, 'bounds': [(cy - hd, cy + hd), (z_floor, z_top)],   'name': 'X+ wall'},
        {'axis': 1, 'pos': cy - hd, 'bounds': [(cx - hw, cx + hw), (z_floor, z_top)],   'name': 'Y- wall'},
        {'axis': 1, 'pos': cy + hd, 'bounds': [(cx - hw, cx + hw), (z_floor, z_top)],   'name': 'Y+ wall'},
    ]


def _seg_face_dist(p1: np.ndarray, p2: np.ndarray, face: dict, n: int = 8) -> float:
    ax    = face['axis']
    other = [i for i in range(3) if i != ax]
    min_d = np.inf
    for t in np.linspace(0.0, 1.0, n):
        p = p1 + t * (p2 - p1)
        c = np.empty(3)
        c[ax]       = face['pos']
        c[other[0]] = np.clip(p[other[0]], *face['bounds'][0])
        c[other[1]] = np.clip(p[other[1]], *face['bounds'][1])
        d = np.linalg.norm(p - c)
        if d < min_d:
            min_d = d
    return min_d


def _plot_collision(all_T: list, bin_cfg: dict, coll: dict) -> None:
    """Save a 3D rejection plot highlighting the colliding element and the hit bin face."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401

        pos_mm  = np.array([T[:3, 3] * 1000.0 for T in all_T])
        tfc_pos = pos_mm[-1]
        tfc_z   = all_T[-1][:3, 2]

        cx, cy, cz = bin_cfg['world_pos']
        hw, hd, hh = [s / 2 for s in bin_cfg['size']]
        x0, x1 = cx - hw, cx + hw
        y0, y1 = cy - hd, cy + hd
        z0, z1 = cz - hh, cz + hh

        bin_walls = [
            [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],   # floor
            [(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)],   # X- wall
            [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],   # X+ wall
            [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],   # Y- wall
            [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],   # Y+ wall
        ]
        wall_face_defs = [
            (2, z0), (0, x0), (0, x1), (1, y0), (1, y1),
        ]

        hit_axis = coll['face']['axis']
        hit_pos  = coll['face']['pos']
        wall_colors = [
            (1.0, 0.2, 0.2, 0.55) if (fa == hit_axis and abs(fp - hit_pos) < 1)
            else (0.3, 0.6, 0.9, 0.12)
            for fa, fp in wall_face_defs
        ]

        # Camera capsule endpoints in world frame
        T_tfc_world = np.eye(4)
        T_tfc_world[:3, :3] = all_T[-1][:3, :3]
        T_tfc_world[:3,  3] = tfc_pos
        T_cam_world = T_tfc_world @ _get_hand_eye()
        cam_p1 = (T_cam_world @ np.append(_CAM_CAPSULE_P1, 1.0))[:3]
        cam_p2 = (T_cam_world @ np.append(_CAM_CAPSULE_P2, 1.0))[:3]

        fig = plt.figure(figsize=(16, 6))
        views = [(25, -55), (0, 90), (90, 0)]
        for col_idx, (elev, azim) in enumerate(views, 1):
            ax = fig.add_subplot(1, 3, col_idx, projection='3d')

            # Arm skeleton
            ax.plot(pos_mm[:, 0], pos_mm[:, 1], pos_mm[:, 2],
                    'o-', color='steelblue', lw=2, ms=4, zorder=2)

            # TCP sections
            for sec_idx, (start, end, r) in enumerate(_TCP_SECTIONS):
                p1 = tfc_pos + start * tfc_z
                p2 = tfc_pos + end   * tfc_z
                hit = coll['type'] == 'tcp' and coll.get('sec_idx') == sec_idx
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        lw=max(1, r / 12), color='red' if hit else 'orange',
                        solid_capstyle='round', zorder=3)

            # Camera capsule
            cam_hit = coll['type'] == 'camera'
            ax.plot([cam_p1[0], cam_p2[0]], [cam_p1[1], cam_p2[1]], [cam_p1[2], cam_p2[2]],
                    lw=4, color='red' if cam_hit else 'mediumpurple',
                    solid_capstyle='round', label='camera', zorder=3)

            # Colliding segment highlighted
            if 'p1' in coll and 'p2' in coll:
                cp1, cp2 = coll['p1'], coll['p2']
                ax.plot([cp1[0], cp2[0]], [cp1[1], cp2[1]], [cp1[2], cp2[2]],
                        lw=6, color='red', alpha=0.5, zorder=4)

            # Bin walls
            poly = Poly3DCollection(bin_walls, facecolors=wall_colors,
                                    edgecolors='gray', linewidths=0.4, zorder=1)
            ax.add_collection3d(poly)

            # Mark TCP target
            tcp_target = tfc_pos + _TCP_OFFSET_MM[2] * tfc_z
            ax.scatter(*tcp_target, color='green', s=60, zorder=5, label='TCP target')

            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z (mm)')
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f'elev={elev} azim={azim}')

        label = coll.get('label', coll['type'])
        fig.suptitle(
            f"REJECTED  –  {label}  (r={coll['radius']:.1f} mm)  "
            f"→  {coll['face']['name']}  "
            f"|  clearance {coll['clearance']:.1f} mm",
            fontsize=11,
        )
        plt.tight_layout()

        out = get_project_dir() / 'plots' / f'reject_{coll["type"]}_{int(time.time() * 1000)}.png'
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=100, bbox_inches='tight')
        plt.close(fig)
        logger.info('Rejection plot saved → %s', out.name)

    except Exception as exc:
        logger.warning('Could not save rejection plot: %s', exc)


def is_reachable(chain: Chain, T_world_target_mm: np.ndarray,
                 home_joints_deg: list, bin_cfg: dict) -> bool:
    """
    Returns True if the grasp pose is kinematically reachable AND no robot link
    or TCP assembly section collides with the bin walls or floor.
    """
    target_pos = T_world_target_mm[:3, 3].round(1).tolist()

    sol = _solve_ik(chain, T_world_target_mm, home_joints_deg)
    if sol is None:
        logger.debug('REJECT IK  target=%s  (no convergence)', target_pos)
        return False

    all_T  = chain.forward_kinematics(sol, full_kinematics=True)
    pos_mm = [t[:3, 3] * 1000.0 for t in all_T]
    faces  = _bin_faces(bin_cfg)

    # Robot arm links
    for i in range(len(pos_mm) - 1):
        r = _RADII_MM.get(i, 40.0)
        if r == 0.0:
            continue
        for face in faces:
            dist = _seg_face_dist(pos_mm[i], pos_mm[i + 1], face)
            if dist < r:
                label = f'arm link {i} (r={r:.0f} mm)'
                logger.info(
                    'REJECT collision  %s → %s  clearance=%.1f mm  target=%s',
                    label, face['name'], dist, target_pos,
                )
                coll = dict(type='arm', label=label, sec_idx=None,
                            radius=r, face=face, clearance=dist,
                            p1=pos_mm[i], p2=pos_mm[i + 1])
                if 'arm' not in _plotted_types:
                    _plotted_types.add('arm')
                    _plot_collision(all_T, bin_cfg, coll)
                return False

    # Custom TCP assembly sections
    bin_top_z  = bin_cfg['world_pos'][2] + bin_cfg['size'][2] / 2
    wall_faces = [f for f in faces if f['axis'] != 2]
    tfc_pos = pos_mm[-1]
    tfc_z   = all_T[-1][:3, 2]
    for sec_idx, (start, end, r) in enumerate(_TCP_SECTIONS):
        p1 = tfc_pos + start * tfc_z
        p2 = tfc_pos + end   * tfc_z
        if min(p1[2], p2[2]) > bin_top_z:
            continue
        for face in wall_faces:
            dist = _seg_face_dist(p1, p2, face)
            if dist < r:
                name  = _TCP_SECTION_NAMES[sec_idx]
                label = f'TCP {name} (r={r:.0f} mm)'
                logger.info(
                    'REJECT collision  %s → %s  clearance=%.1f mm  target=%s',
                    label, face['name'], dist, target_pos,
                )
                plot_key = f'tcp_{sec_idx}'
                coll = dict(type='tcp', label=label, sec_idx=sec_idx,
                            radius=r, face=face, clearance=dist,
                            p1=p1, p2=p2)
                if plot_key not in _plotted_types:
                    _plotted_types.add(plot_key)
                    _plot_collision(all_T, bin_cfg, coll)
                return False

    # Camera body capsule
    T_tfc_world = np.eye(4)
    T_tfc_world[:3, :3] = all_T[-1][:3, :3]
    T_tfc_world[:3,  3] = tfc_pos
    T_cam2tfp = _get_hand_eye()
    T_cam_world = T_tfc_world @ T_cam2tfp
    cam_p1 = (T_cam_world @ np.append(_CAM_CAPSULE_P1, 1.0))[:3]
    cam_p2 = (T_cam_world @ np.append(_CAM_CAPSULE_P2, 1.0))[:3]
    if min(cam_p1[2], cam_p2[2]) <= bin_top_z:
        for face in wall_faces:
            dist = _seg_face_dist(cam_p1, cam_p2, face)
            if dist < _CAM_CAPSULE_R:
                label = f'camera capsule (r={_CAM_CAPSULE_R:.0f} mm)'
                logger.info(
                    'REJECT collision  %s → %s  clearance=%.1f mm  target=%s',
                    label, face['name'], dist, target_pos,
                )
                coll = dict(type='camera', label=label, sec_idx=None,
                            radius=_CAM_CAPSULE_R, face=face, clearance=dist,
                            p1=cam_p1, p2=cam_p2)
                if 'camera' not in _plotted_types:
                    _plotted_types.add('camera')
                    _plot_collision(all_T, bin_cfg, coll)
                return False

    return True
