"""Train one deployable CrossFingerMapper bundle from a hand recording.

    python scripts/train_bundle.py --rec data/pico4_check.pkl --out results/pico4
    python scripts/train_bundle.py --rec data/avp_20261016_143808.pkl --out results/avp_cf

Works for both AVP and PICO 4 recordings -- scripts/record_avp.py writes the
same (21, 3) MediaPipe-landmark pkl for either device.

The data setup is run_avp.py's, line for line, because every piece of it is
load-bearing and load_bundle() rebuilds the robot side the same way at
inference:

  * robot clouds in the PALM frame (`frame="palm"`) with the robot's OWN
    normalizer -- a base-frame cloud, or one normalized with the human
    normalizer, trains a mapper into a different space than the one
    load_bundle() looks up in, and every finger comes out rotated;
  * skeleton anchors (`strategy="skeleton"` with the recording's landmarks and
    the robot palm frame);
  * identity human orientations with L_align still ON -- the recordings are
    positions only, and identity makes L_align the global-frame objective,
    which is the intended default, not a reason to drop the term.

What differs from run_avp.py: one model (the `full` objective, CrossFingerMapper
for continuous pinch) instead of the paper's ablation ladder, and the
bundle's extra['open_gap_mm'] is estimated from this recording so
teleop_avp.py's pinch blend uses this operator/device's own open-hand spread.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anydexrt.anchors import build_anchor_set  # noqa: E402
from anydexrt.data import build_robot_clouds, load_avp_pkl, normalize_human  # noqa: E402
from anydexrt.hands import get_hand  # noqa: E402
from anydexrt.inference import save_bundle  # noqa: E402
from anydexrt.train import TrainConfig, train  # noqa: E402

FULL_WEIGHTS = {"p_chamfer": 1.0, "dist": 1.0, "motion": 1.0, "align": 1.0}


def block_split(n: int, n_blocks: int, test_blocks: list[int]) -> np.ndarray:
    """Train indices after removing contiguous held-out blocks (run_avp.py's split)."""
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    mask = np.ones(n, dtype=bool)
    for b in test_blocks:
        mask[edges[b]:edges[b + 1]] = False
    return np.where(mask)[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", required=True, help="AVP or PICO 4 recording (scripts/record_avp.py pkl)")
    ap.add_argument("--hand", default="linker_l20")
    ap.add_argument("--side", default="right", choices=["right", "left"])
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--cloud", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--test-blocks", type=int, nargs="*", default=[5, 9],
                    help="contiguous blocks held out of training, as in run_avp.py; "
                         "pass nothing (--test-blocks) to train on every frame")
    args = ap.parse_args()

    spec = get_hand(args.hand)
    rec = load_avp_pkl(args.rec, spec, side=args.side)
    train_idx = block_split(rec.n_frames, args.blocks, args.test_blocks)
    h_train = rec.positions[train_idx]
    print(f"{rec.path.name}: {rec.n_frames} frames, training on {len(train_idx)} "
          f"(held out blocks {args.test_blocks})")

    samples = normalize_human(h_train, rec.identity_quats()[train_idx], spec.finger_names)
    clouds, kin = build_robot_clouds(spec, n_samples=args.cloud, seed=0, frame="palm")
    anchors = build_anchor_set(
        spec, h_train, samples.normalizer, kin,
        robot_normalizer=clouds.normalizer, strategy="skeleton",
        landmarks=rec.landmarks[train_idx],
        robot_frame=(clouds.frame_origin, clouds.frame_rot),
    )

    cfg = TrainConfig(hand=args.hand, epochs=args.epochs, seed=args.seed,
                      weights=FULL_WEIGHTS, use_anchors=True, mapper_type="cross_finger")
    t0 = time.time()
    result = train(samples, clouds, anchors, cfg, raw_fingertips=h_train, train_classifier=False)
    print(f"trained {args.epochs} epochs in {(time.time() - t0) / 60:.1f} min")

    # Per-finger open-hand thumb-to-finger gap, 90th percentile (not max, so
    # one outlier frame cannot set the scale) -- teleop_avp.py reads this as
    # the pinch blend's alpha=0 reference.
    gaps = np.linalg.norm(h_train[:, 1:] - h_train[:, :1], axis=-1)
    open_gap_mm = [float(np.percentile(gaps[:, i], 90) * 1000) for i in range(gaps.shape[1])]
    print(f"open-hand gap (p90) per finger: {[round(g, 1) for g in open_gap_mm]} mm")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.hand}_C_cross_finger_bundle.pt"
    save_bundle(path, result, args.hand, samples.normalizer, args.cloud, cloud_seed=0,
                extra={"avp": str(rec.path), "side": args.side, "open_gap_mm": open_gap_mm,
                       "test_blocks": args.test_blocks, "blocks": args.blocks})
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
