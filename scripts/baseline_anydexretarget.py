"""Baseline: the existing AnyDexRetarget optimizer on the same held-out AVP frames.

AnyDexRetarget (`/home/qsq/qqsq12321/AnyDexRetarget`) is the optimization-based
retargeter already used for AVP teleoperation here -- AdaptiveOptimizerAnalytical,
configs `example/config/adaptive/avp/avp_{wuji_hand,linker_l20}.yaml`, run as
shipped. It is the paper's "optimization-based retargeting" baseline in spirit:
hand-designed objectives with tuned weights, segment scalings and an input
rotation.

To compare like with like, its joint output is mapped by name onto this
package's drive vector, pushed through the SAME MJCF forward kinematics, and
scored with the SAME functions and held-out blocks as `scripts/run_avp.py`.
Only active drives are taken; coupled joints (L20 DIP/IP) are re-derived from
the coupling table, and the optimizer's own output already satisfies them.

Usage:
    python scripts/baseline_anydexretarget.py --hands wuji_hand linker_l20
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANYDEX = Path("/home/qsq/qqsq12321/AnyDexRetarget")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ANYDEX))

from anydexretarget.retarget import Retargeter  # noqa: E402

from anydexrt.data import build_robot_clouds, load_avp_pkl  # noqa: E402
from anydexrt.hands import get_hand  # noqa: E402
from run_avp import DEFAULT_AVP, block_split, gmc, pinch_stats  # noqa: E402

CONFIGS = {
    "wuji_hand": ANYDEX / "example/config/adaptive/avp/avp_wuji_hand.yaml",
    "linker_l20": ANYDEX / "example/config/adaptive/avp/avp_linker_l20.yaml",
}


def run_hand(hand: str, args) -> dict:
    spec = get_hand(hand)
    rec = load_avp_pkl(args.avp, spec)
    with open(args.avp, "rb") as fh:
        raw_frames = pickle.load(fh)
    raw = np.stack([np.asarray(f["right_fingers"], dtype=np.float64) for f in raw_frames])
    # load_avp_pkl drops all-zero tracking frames; keep the same frames here.
    raw = raw[np.abs(raw).sum(axis=(1, 2)) > 1e-9]
    assert len(raw) == rec.n_frames

    _, test_slices = block_split(rec.n_frames, args.blocks, args.test_blocks)
    clouds, kin = build_robot_clouds(spec, n_samples=2000, seed=0, frame="palm")
    lo, hi = kin.limits()

    retargeter = Retargeter.from_yaml(str(CONFIGS[hand]), hand_side="right")
    opt = retargeter.optimizer
    names = list(getattr(opt, "joint_names", None) or opt.robot.dof_joint_names)
    if len(names) != retargeter.num_joints:
        raise RuntimeError(f"joint name list ({len(names)}) != output size ({retargeter.num_joints})")
    take = np.array([names.index(d) for d in spec.joint_names])

    human_segs, robot_segs, qs = [], [], []
    n, seconds = 0, 0.0
    for s in test_slices:
        retargeter.reset()
        t0 = time.perf_counter()
        q_theirs = np.stack([retargeter.retarget(raw[t]) for t in range(s.start, s.stop)])
        seconds += time.perf_counter() - t0
        n += s.stop - s.start
        q = np.clip(q_theirs[:, take], lo, hi)
        qs.append(q)
        robot_segs.append(clouds.to_cloud_frame(kin.fingertip_positions_batch(q)))
        human_segs.append(rec.positions[s])

    report = {
        "hand": hand,
        "config": str(CONFIGS[hand]),
        "gmc_robot": gmc(human_segs, robot_segs, args.stride),
        "pinch": pinch_stats(np.concatenate(human_segs), np.concatenate(robot_segs), spec.finger_names),
        "jitter_rad": float(np.mean(np.concatenate([np.abs(np.diff(q, axis=0)).mean(1) for q in qs]))),
        "hz": n / seconds,
    }
    pi = report["pinch"]["index"]
    print(f"{hand:11s} AnyDexRetarget  gmc_robot={report['gmc_robot']:6.1f} "
          f"idx_pinch(n={pi['n_pinch']}) gap={pi['robot_gap_mm_on_pinch']:.1f}mm "
          f"closed={pi['robot_closed_on_pinch']:.2f} false={pi['false_pinch_rate']:.3f} "
          f"jitter={report['jitter_rad']:.4f} {report['hz']:.0f}Hz")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hands", nargs="+", default=["wuji_hand", "linker_l20"])
    ap.add_argument("--avp", default=DEFAULT_AVP)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--test-blocks", type=int, nargs="+", default=[5, 9])
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", default="results/avp/baseline_anydexretarget.json")
    args = ap.parse_args()
    reports = [run_hand(h, args) for h in args.hands]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"args": vars(args), "hands": reports}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
