#!/usr/bin/env python
"""Interactive pinch pose calibration tool.

在 MuJoCo viewer 里手动调整关节,摆出好的捏合姿态,然后在终端里敲 's' 回车
保存当前姿态并自动切换到下一对手指。全部标定完自动写入 yaml。

操作流程:
1. 运行脚本,打开 MuJoCo viewer
2. 在 viewer 的 Control 面板调整关节,摆出 thumb-index 的好姿态
3. 回到运行脚本的终端,敲 's' 回车 → 保存 thumb-index,自动切到 thumb-middle
4. 继续调整,敲 's' 保存 ... 一直到 thumb-pinky
5. 4 组都保存后自动写文件并退出

终端命令:
  s  = 保存当前姿态,切到下一对
  q  = 保存已完成的并退出
  (直接回车 / 其它输入 = 忽略,继续调整)
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


CALIBRATION_FILE = PROJECT_ROOT / "calibrated_pinch_poses.yaml"
FINGER_ORDER = ['index', 'middle', 'ring', 'pinky']
FINGER_IDX = {'index': 1, 'middle': 2, 'ring': 3, 'pinky': 4}


def main():
    hand_name = 'linker_l20'
    hand = get_hand(hand_name)
    kin = HandKinematics(hand_name)

    thumb_sl = hand.joint_slice(hand.fingers[0].name)
    finger_slices = {
        name: hand.joint_slice(hand.fingers[idx].name)
        for name, idx in FINGER_IDX.items()
    }

    calibrated = {}
    state = {'step': 0, 'quit': False, 'pending_save': False}
    lock = threading.Lock()

    def current_finger():
        if state['step'] < len(FINGER_ORDER):
            return FINGER_ORDER[state['step']]
        return None

    def write_file():
        if not calibrated:
            print('\n[警告] 没有任何标定数据,不写文件')
            return
        with open(CALIBRATION_FILE, 'w') as f:
            yaml.dump(calibrated, f, default_flow_style=False, sort_keys=False)
        print(f"\n[已写入] {CALIBRATION_FILE}")
        for name in FINGER_ORDER:
            if name in calibrated:
                print(f"  thumb-{name}: gap={calibrated[name]['gap_mm']:.2f}mm")

    def do_save():
        finger_name = current_finger()
        if finger_name is None:
            print('\n[提示] 4 组已全部标定完成,输入 q 回车退出')
            return

        # Read current joint values straight from the simulated qpos, then
        # convert MJCF qpos back to canonical drive values.
        raw_qpos = kin.data.qpos.copy()
        qpos = np.zeros(kin.n_joints)
        for i, resolved in enumerate(kin._drive_joints):
            # resolved maps joint_name -> (qpos_address, multiplier); a coupled
            # drive writes several joints from one drive value, so any single
            # child (divided by its own multiplier) recovers the drive value.
            _name, (adr, mult) = next(iter(resolved.items()))
            qpos[i] = raw_qpos[adr] / mult

        thumb_q = qpos[thumb_sl].tolist()
        finger_q = qpos[finger_slices[finger_name]].tolist()

        tips = kin._current_tips()
        finger_tip_idx = FINGER_IDX[finger_name]
        gap = float(np.linalg.norm(tips[0] - tips[finger_tip_idx]) * 1000)

        calibrated[finger_name] = {
            'thumb_q': [round(v, 4) for v in thumb_q],
            'finger_q': [round(v, 4) for v in finger_q],
            'gap_mm': round(gap, 3),
        }
        state['step'] += 1

        print(f"\n[保存] thumb-{finger_name}  gap={gap:.2f}mm")
        print(f"  拇指: {[f'{v:.3f}' for v in thumb_q]}")
        print(f"  {finger_name}: {[f'{v:.3f}' for v in finger_q]}")

        nxt = current_finger()
        if nxt is not None:
            print(f"\n[下一对] 请调整关节摆出 thumb-{nxt} 的姿态,然后敲 s 回车")
        else:
            print("\n[完成] 4 组全部标定完毕,正在写文件...")
            write_file()
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
    print("MuJoCo 捏合姿态标定工具")
    print("=" * 60)
    print("\n标定顺序: thumb-index → thumb-middle → thumb-ring → thumb-pinky")
    print("\n操作:")
    print("  1. 在 viewer 的 Control 面板调整关节滑条,摆出好的捏合姿态")
    print("  2. 回到这个终端,敲 s 回车 → 保存当前姿态并切到下一对")
    print("  3. 敲 q 回车 → 保存已完成的并退出")
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

        print(f"\n请调整关节摆出 thumb-{current_finger()} 的姿态,然后在这个终端敲 s 回车")

        while viewer.is_running() and not state['quit']:
            mujoco.mj_step(kin.model, kin.data)
            viewer.sync()

            with lock:
                should_save = state['pending_save']
                state['pending_save'] = False
            if should_save:
                do_save()

    if not state['quit']:
        write_file()
    print('Done.')


if __name__ == "__main__":
    main()
