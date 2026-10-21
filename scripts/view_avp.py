"""Replay avp1.pkl through AnyDexRT in the interactive MuJoCo viewer.

Keys (focus the viewer window):
    space   pause / resume
    ->  <-  step one frame while paused

Usage:
    python scripts/view_avp.py --hand linker_l20
    python scripts/view_avp.py --hand wuji_hand
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anydexrt.data import load_avp_pkl  # noqa: E402
from anydexrt.inference import load_bundle  # noqa: E402

KEY_SPACE, KEY_RIGHT, KEY_LEFT = 32, 262, 263


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand", default="linker_l20", choices=["linker_l20", "wuji_hand"])
    ap.add_argument("--results", default=str(ROOT / "results/avp"))
    ap.add_argument("--avp", default="/home/qsq/qqsq12321/AnyDexRetarget/example/data/avp1.pkl")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--pinch", action="store_true",
                    help="enable f_c pinch-template snapping (off by default: see "
                         "scripts/teleop_avp.py --pinch help for why)")
    args = ap.parse_args()

    rt = load_bundle(Path(args.results) / args.hand / f"{args.hand}_bundle.pt", use_contacts=args.pinch)
    kin = rt.kin
    rec = load_avp_pkl(args.avp, kin.hand)
    qpos, stats = rt.retarget_sequence(rec.positions)
    print(f"[{args.hand}] {rec.n_frames} frames retargeted at {stats['hz']:.0f} Hz; replaying at {args.fps:.0f} fps")
    print("keys: space pause | <- -> step")

    model = kin.model
    data = mujoco.MjData(model)
    origin, R = kin.palm_frame()
    state = {"i": 0, "paused": False}

    def on_key(key):
        if key == KEY_SPACE:
            state["paused"] = not state["paused"]
        elif key == KEY_RIGHT:
            state["i"] = (state["i"] + 1) % len(qpos)
        elif key == KEY_LEFT:
            state["i"] = (state["i"] - 1) % len(qpos)

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        # Look at the palm side of the hand, framed by its own palm frame.
        is_l20 = "l20" in args.hand
        viewer.cam.lookat[:] = origin + R @ np.array([0.15 if is_l20 else 0.10, 0.0, 0.0])
        viewer.cam.distance = 0.45 if is_l20 else 0.35
        fwd = -(R @ np.array([-0.2, -0.3, -0.9]))
        fwd /= np.linalg.norm(fwd)
        viewer.cam.azimuth = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
        viewer.cam.elevation = float(np.degrees(np.arcsin(fwd[2])))

        while viewer.is_running():
            t0 = time.perf_counter()
            data.qpos[:] = kin._expand_drives(qpos[state["i"]])
            mujoco.mj_forward(model, data)
            viewer.sync()
            if not state["paused"]:
                state["i"] = (state["i"] + 1) % len(qpos)
            time.sleep(max(0.0, 1.0 / args.fps - (time.perf_counter() - t0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
