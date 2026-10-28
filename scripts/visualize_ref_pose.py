#!/usr/bin/env python
"""Visualize the reference pinch pose from the screenshot with red skeleton overlay."""
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from anydexrt.kinematics import HandKinematics
from anydexrt.hands import get_hand


def draw_skeleton(scene, points, connections, color, radius=0.002, offset=np.zeros(3)):
    """Draw skeleton with capsules and spheres like debug_skeleton.py does."""
    points = np.asarray(points) + offset

    # Draw connections as capsules
    for conn in connections:
        if scene.ngeom >= scene.maxgeom:
            break
        i, j = conn
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.array([radius, 0, 0]),
            pos=np.zeros(3),
            mat=np.eye(3).flatten(),
            rgba=color
        )
        mujoco.mjv_connector(
            geom,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            width=radius,
            from_=points[i],
            to=points[j]
        )
        scene.ngeom += 1

    # Draw joints as spheres
    for pt in points:
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([radius * 1.5, 0, 0]),
            pos=pt,
            mat=np.eye(3).flatten(),
            rgba=color
        )
        scene.ngeom += 1


def main():
    hand = get_hand('linker_l20')
    kin = HandKinematics('linker_l20')

    def set_ctrl(name, value):
        aid = mujoco.mj_name2id(kin.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        kin.data.ctrl[aid] = value

    # Set the reference pose from screenshot
    mujoco.mj_resetData(kin.model, kin.data)
    set_ctrl('thumb_cmc_roll', 0.605)
    set_ctrl('thumb_cmc_yaw', 1.06)
    set_ctrl('thumb_cmc_pitch', 0.249)
    set_ctrl('thumb_mcp', 0.637)
    set_ctrl('index_mcp_roll', -0.0023)
    set_ctrl('index_mcp_pitch', 1.22)
    set_ctrl('index_pip', 0.674)

    # Let physics settle
    for _ in range(500):
        mujoco.mj_step(kin.model, kin.data)

    def janchor(name):
        jid = mujoco.mj_name2id(kin.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return kin.data.xanchor[jid].copy()

    thumb_tip = kin._current_tips()[0]
    index_tip = kin._current_tips()[1]
    gap = np.linalg.norm(thumb_tip - index_tip)
    print(f'拇指尖-食指尖间隙: {gap*1000:.2f}mm')

    # Build skeleton points and connections
    wrist = janchor('thumb_cmc_roll')
    thumb_pts = [
        wrist,
        janchor('thumb_cmc_roll'),
        janchor('thumb_mcp'),
        janchor('thumb_ip'),
        thumb_tip
    ]
    thumb_conns = [(0, 1), (1, 2), (2, 3), (3, 4)]

    index_pts = [
        janchor('index_mcp_pitch'),
        janchor('index_pip'),
        janchor('index_dip'),
        index_tip
    ]
    index_conns = [(0, 1), (1, 2), (2, 3)]

    # Red with 30% opacity
    RED = np.array([1.0, 0.1, 0.1, 0.3], dtype=np.float32)

    print('正在启动 MuJoCo viewer...')
    print('操作: 鼠标左键旋转, 右键平移, 滚轮缩放, 按 Esc 关闭')

    with mujoco.viewer.launch_passive(kin.model, kin.data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        viewer.cam.lookat[:] = (thumb_tip + index_tip) / 2
        viewer.cam.distance = 0.2
        viewer.cam.elevation = -20

        while viewer.is_running():
            mujoco.mj_step(kin.model, kin.data)
            viewer.user_scn.ngeom = 0
            draw_skeleton(viewer.user_scn, thumb_pts, thumb_conns, RED)
            draw_skeleton(viewer.user_scn, index_pts, index_conns, RED)
            viewer.sync()
            time.sleep(0.01)

    print('Viewer closed.')


if __name__ == "__main__":
    main()
