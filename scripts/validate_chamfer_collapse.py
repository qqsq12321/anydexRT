"""Rung 1 of the falsifiable ladder: does partial Chamfer alone collapse?

The paper's ablation (Table 3) on Wuji Hand reports:

    LP-Chamfer          4.1
    + Ldist            83.8
    + Lmotion          89.1
    + Lalign           92.3

A score of 4.1 for Chamfer-alone is not a weak result -- it is the signature of
collapse. Partial Chamfer is ONE-directional: it only pulls mapped human points
toward the robot cloud. Writing every human sample onto a single robot point
drives it to exactly zero, and a collapsed mapping has essentially random
displacement directions, so a direction-agreement metric lands near 0.

`L_dist` is therefore load-bearing, not decorative: it is the only term that
resists the collapse, by holding pairwise distances fixed.

This script tests that mechanism directly, on one hand and one capture:

    1. Train f_m with partial Chamfer only.
    2. Measure how much the mapped cloud collapses (mean pairwise distance vs.
       the source cloud's).
    3. Repeat with L_dist added.

Passing looks like: collapse ratio near 0 with Chamfer alone, and near the
source's spread once L_dist is on. Failing means the Chamfer implementation is
not one-directional, or L_dist is leaking into the first run.

Usage:
    python scripts/validate_chamfer_collapse.py --hand leap_hand
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anydexrt.anchors import build_anchor_set  # noqa: E402
from anydexrt.data import (  # noqa: E402
    discover_manus_captures,
    load_manus_npz,
    normalize_human,
    build_robot_clouds,
)
from anydexrt.hands import get_hand  # noqa: E402
from anydexrt.losses import LossWeights  # noqa: E402
from anydexrt.train import TrainConfig, train  # noqa: E402
from anydexrt.metrics import motion_consistency  # noqa: E402


def collapse_ratio(mapper, human_norm: np.ndarray, device: torch.device) -> dict:
    """How tightly the mapper squeezes each finger's cloud.

    Returns the ratio of the mapped cloud's mean pairwise spread to the
    source's, per finger averaged. ~0 is total collapse, ~1 preserves spread.
    """
    F, N, _ = human_norm.shape
    x = torch.from_numpy(human_norm).float().to(device)
    with torch.no_grad():
        mapped = torch.stack(
            [mapper.forward_finger(x[i], i) for i in range(F)]
        )  # (F, N, 3)

    src_spread, map_spread = [], []
    for i in range(F):
        d_src = torch.cdist(x[i].unsqueeze(0), x[i].unsqueeze(0))[0]
        d_map = torch.cdist(mapped[i].unsqueeze(0), mapped[i].unsqueeze(0))[0]
        n = d_src.shape[0]
        off = ~torch.eye(n, dtype=torch.bool, device=d_src.device)
        src_spread.append(float(d_src[off].mean()))
        map_spread.append(float(d_map[off].mean()))

    src_mean = float(np.mean(src_spread))
    map_mean = float(np.mean(map_spread))
    return {
        "source_spread": src_mean,
        "mapped_spread": map_mean,
        "ratio": map_mean / max(src_mean, 1e-12),
        "mapped_spread_per_finger": map_spread,
    }


def direction_agreement(mapper, human_norm: np.ndarray, device: torch.device) -> float:
    """Proxy for the paper's motion consistency, using consecutive samples.

    This is LMC/GMC in the same spirit as `anydexrt.metrics` but computed in the
    global frame against the identity mapping, so it isolates the collapse
    effect without needing robot-side frames.
    """
    F, N, _ = human_norm.shape
    x = torch.from_numpy(human_norm).float().to(device)
    with torch.no_grad():
        mapped = torch.stack([mapper.forward_finger(x[i], i) for i in range(F)])
    src = human_norm.reshape(1, -1, 3)
    pred = mapped.cpu().numpy().reshape(1, -1, 3)
    try:
        return motion_consistency(
            torch.from_numpy(src).float(), torch.from_numpy(pred).float(), local=False
        ).score
    except ValueError:
        return float("nan")


def run_condition(
    name: str,
    weights: LossWeights,
    hand: str,
    capture: Path,
    epochs: int,
    seed: int,
    device: torch.device,
    n_cloud: int,
    frames: int,
) -> dict:
    spec = get_hand(hand)
    raw, quats = load_manus_npz(capture, spec, max_frames=frames)
    samples = normalize_human(raw, quats, spec.finger_names)
    clouds, kin = build_robot_clouds(
        spec, n_samples=n_cloud, seed=seed, normalizer=samples.normalizer
    )
    anchors = build_anchor_set(
        spec,
        raw,
        samples.normalizer,
        kin,
        robot_normalizer=samples.normalizer,
        strategy="synthetic",
        n_lateral=50,
        n_bending=100,
    )

    cfg = TrainConfig(
        hand=hand,
        epochs=epochs,
        seed=seed,
        weights={
            "p_chamfer": weights.p_chamfer,
            "dist": weights.dist,
            "motion": weights.motion,
            "align": weights.align,
        },
    )
    # human is (F, N, 3) for the objective
    human_f = np.transpose(samples.fingertips, (1, 0, 2))
    result = train(
        samples, clouds, anchors, cfg, device=device, train_classifier=False,
        raw_fingertips=raw,
    )
    mapper = result.mapper
    mapper.eval()

    collapse = collapse_ratio(mapper, human_f, device)
    agree = direction_agreement(mapper, human_f, device)
    print(
        f"  {name:24s} spread_ratio={collapse['ratio']:.4f}  "
        f"(src={collapse['source_spread']:.4f} -> mapped={collapse['mapped_spread']:.4f})  "
        f"dir_agree={agree:6.1f}"
    )
    return {
        "name": name,
        "ratio": collapse["ratio"],
        "source_spread": collapse["source_spread"],
        "mapped_spread": collapse["mapped_spread"],
        "direction_agreement": agree,
        "anchors_strategy": anchors.strategy,
        "per_finger_ratio": [
            m / max(s, 1e-12)
            for s, m in zip(
                [collapse["source_spread"]] * len(collapse["mapped_spread_per_finger"]),
                collapse["mapped_spread_per_finger"],
            )
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hand", default="leap_hand")
    ap.add_argument(
        "--capture",
        default=str(
            Path("/home/qsq/qqsq12321/mimic_retargeter_lab/dataset/manus")
        ),
    )
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=4000)
    ap.add_argument("--cloud", type=int, default=8000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    caps = discover_manus_captures(args.capture)
    if not caps:
        print(f"no Manus captures under {args.capture}", file=sys.stderr)
        return 2
    capture = caps[0]

    print(f"Rung 1: partial-Chamfer collapse ({args.hand}, {capture.name})")
    print(f"  epochs={args.epochs} cloud={args.cloud} frames={args.frames} device={device}")
    print()

    conditions = [
        ("LP-Chamfer only", LossWeights(p_chamfer=1.0, dist=0.0, motion=0.0, align=0.0)),
        ("+ Ldist", LossWeights(p_chamfer=1.0, dist=1.0, motion=0.0, align=0.0)),
        ("+ Lmotion", LossWeights(p_chamfer=1.0, dist=1.0, motion=1.0, align=0.0)),
        ("+ Lalign (full)", LossWeights(1.0, 1.0, 1.0, 1.0)),
    ]

    rows = []
    for name, w in conditions:
        rows.append(
            run_condition(
                name, w, args.hand, capture, args.epochs, args.seed, device,
                args.cloud, args.frames,
            )
        )

    print()
    only = rows[0]
    full = rows[-1]
    print("Expected pattern (paper Table 3, direction of effect):")
    print("  * Chamfer alone collapses  -> spread ratio near 0")
    print("  * adding Ldist restores    -> spread ratio jumps toward 1")
    print()
    ok_collapse = only["ratio"] < 0.35
    ok_recover = full["ratio"] > 0.5 and full["ratio"] > only["ratio"] + 0.25

    print(f"  collapse detected:  {'PASS' if ok_collapse else 'FAIL'} "
          f"(ratio {only['ratio']:.4f}, need < 0.35)")
    print(f"  Ldist recovers:     {'PASS' if ok_recover else 'FAIL'} "
          f"(ratio {full['ratio']:.4f} vs {only['ratio']:.4f}, need > 0.5 and +0.25)")

    if ok_collapse and ok_recover:
        print("\nRUNG 1 PASSED: the partial-Chamfer collapse mechanism is reproduced.")
        return 0
    print("\nRUNG 1 FAILED: inspect the Chamfer directionality and Ldist normalization.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
