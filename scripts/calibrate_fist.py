#!/usr/bin/env python
"""Interactive fist pose calibration tool.

在 MuJoCo viewer 里手动摆出"自然握拳"的姿态(拇指搭在四指外侧),
然后在终端里敲 's' 回车保存。写入 calibrated_fist_pose.yaml。

为什么需要单独标定握拳:
  CrossFingerMapper 的拇指子网络把其它手指位置作为输入。为了解决"拇指
  被其它手指牵连"的问题,推理时会把蜗曲手指在 context 里归零 —— 但训练
  时 context 从来都是真实位置,全零是网络没见过的分布外输入,它对此的
  响应没有定义,表现为握拳时拇指被甩得太远。标定一个明确的握拳目标,
  就不必赌网络在分布外输入上的瞎猜。

操作流程:
1. 运行脚本,打开 MuJoCo viewer
2. 在 Control 面板调整关节,摆出自然握拳(四指蜗曲 + 拇指搭在外侧)
3. 回到这个终端,敲 's' 回车保存并退出

终端命令:
  s  = 保存当前姿态并退出
  q  = 不保存直接退出
"""
import sys
import threading
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from anydexrt.kinematics import HandKinematics
from anydexrt.hands import get_hand

CALIBRATION_FILE = PROJECT_ROOT / "calibrated_fist_pose.yaml"


def main():
    hand_name = 'linker_l20'
    hand = get_hand(hand_name)
    kin = HandKinematics(hand_name)

    state = {'quit': False, 'pending_save': False, 'saved': False}
    lock = threading.Lock()

    def read_drive_qpos():
        """MJCF qpos -> canonical drive values (see calibrate_pinch.py)."""
        raw_qpos = kin.data.qpos.copy()
        qpos = np.zeros(kin.n_joints)
        for i, resolved in enumerate(kin._drive_joints):
            _name, (adr, mult) = next(iter(resolved.items()))
            qpos[i] = raw_qpos[adr] / mult
        return qpos

    def do_save():
        qpos = read_drive_qpos()
        tips = kin._current_tips()
        reach = np.linalg.norm(tips, axis=-1) * 1000
        gaps = np.linalg.norm(tips[1:] - tips[0], axis=-1) * 1000

        payload = {
            'qpos': [round(float(v), 4) for v in qpos],
            'joint_names': list(hand.joint_names),
            'thumb_reach_mm': round(float(reach[0]), 2),
            'finger_reach_mm': [round(float(v), 2) for v in reach[1:]],
            'thumb_to_finger_gap_mm': [round(float(v), 2) for v in gaps],
        }
        with open(CALIBRATION_FILE, 'w') as f:
            yaml.dump(payload, f, default_flow_style=False, sort_keys=False)

        print(f"\n[已保存] {CALIBRATION_FILE}")
        print(f"  拇指离手掌: {reach[0]:.1f}mm")
        print(f"  四指离手掌: {np.round(reach[1:], 1).tolist()} mm")
        print(f"  拇指到各指: {np.round(gaps, 1).tolist()} mm")
        print(f"  拇指关节: {[f'{v:.3f}' for v in qpos[hand.joint_slice('thumb')]]}")
        state['saved'] = True
        state['quit'] = True

    def stdin_reader():
        while not state['quit']:
            try:
                line = input().strip().lower()
            except EOFError:
                break
            with lock:
                if line == 's':
                    state['pending_save'] = True
                elif line == 'q':
                    state['quit'] = True

    print("=" * 60)
    print("MuJoCo 自然握拳姿态标定")
    print("=" * 60)
    print("\n请在 Control 面板摆出【自然握拳】:四指蜗曲,拇指搭在四指外侧")
    print("\n操作:")
    print("  1. 在 viewer 的 Control 面板调整关节滑条")
    print("  2. 回到这个终端,敲 s 回车 → 保存并退出")
    print("  3. 敲 q 回车 → 不保存直接退出")
    print()
    print(f"输出文件: {CALIBRATION_FILE}")
    print("=" * 60)

    mujoco.mj_resetData(kin.model, kin.data)

    reader = threading.Thread(target=stdin_reader, daemon=True)
    reader.start()

    with mujoco.viewer.launch_passive(kin.model, kin.data) as viewer:
        viewer.cam.lookat[:] = [0.1, 0.0, 0.2]
        viewer.cam.distance = 0.3
        viewer.cam.elevation = -15

        print("\n摆好自然握拳后,在这个终端敲 s 回车保存")

        while viewer.is_running() and not state['quit']:
            mujoco.mj_step(kin.model, kin.data)
            viewer.sync()

            with lock:
                should_save = state['pending_save']
                state['pending_save'] = False
            if should_save:
                do_save()

    if not state['saved']:
        print('\n[未保存] 退出')
    print('Done.')


if __name__ == "__main__":
    main()
