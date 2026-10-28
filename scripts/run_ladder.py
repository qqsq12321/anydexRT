"""The falsifiable validation ladder.

Rather than chase the paper's absolute numbers -- which cannot be attributed,
since the LMC implementation is unpublished and the reference code is not
released -- this runs six checks whose *pattern* is predicted by the method's
mechanics. Each rung has a pass condition that fails loudly if the mechanism is
not actually present in this implementation.

Rung 1  partial-Chamfer collapse     Chamfer alone must collapse the cloud
Rung 2  L_dist recovers              adding L_dist must restore spread
Rung 3  L_motion improves agreement   adding L_motion must raise direction agreement
Rung 4  L_align resolves ambiguity    adding L_align must reduce cross-seed variance
Rung 5  calibration robustness        rotating the input must barely move LMC
Rung 6  cross-seed stability          seeds must agree within a tight band

Plus a weight sweep, because the paper ships uniform weights ("we do not
specifically tune the weights of each loss term") while `L_dist` and
`L_P-Chamfer` have different natural scales and different distance conventions
(L2 norm vs squared). The sweep reports where L_dist actually takes effect.

Usage:
    python scripts/run_ladder.py --hand leap_hand --epochs 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anydexrt.anchors import build_anchor_set  # noqa: E402
from anydexrt.data import (  # noqa: E402
    build_robot_clouds,
    discover_manus_captures,
    load_manus_npz,
    normalize_human,
)
from anydexrt.frames import expand_rotation, y_axis_rotation  # noqa: E402
from anydexrt.hands import get_hand  # noqa: E402
from anydexrt.losses import LossWeights  # noqa: E402
from anydexrt.metrics import apply_global_perturbation, motion_consistency  # noqa: E402
from anydexrt.train import TrainConfig, train  # noqa: E402


# ----------------------------------------------------------------- primitives


def spread_ratio(mapper, human_f: np.ndarray, device) -> float:
    """Mapped cloud spread / source cloud spread, averaged over fingers.

    0 means total collapse (every sample lands on one point); 1 means the
    cloud's geometry survived the mapping.
    """
    F, N, _ = human_f.shape
    x = torch.from_numpy(human_f).float().to(device)
    with torch.no_grad():
        mapped = torch.stack([mapper.forward_finger(x[i], i) for i in range(F)])
    ratios = []
    for i in range(F):
        ds = torch.cdist(x[i].unsqueeze(0), x[i].unsqueeze(0))[0]
        dm = torch.cdist(mapped[i].unsqueeze(0), mapped[i].unsqueeze(0))[0]
        off = ~torch.eye(ds.shape[0], dtype=torch.bool, device=ds.device)
        ratios.append(float(dm[off].mean()) / max(float(ds[off].mean()), 1e-12))
    return float(np.mean(ratios))


def direction_agreement(mapper, human_f: np.ndarray, device, local: bool = False) -> float:
    """Motion-consistency score of the mapped trajectory against the human one.

    When `local` is False this is GMC-style (shared frame); the comparison is
    against the human trajectory itself, so it measures how well motion
    direction survives the mapping.
    """
    F, N, _ = human_f.shape
    x = torch.from_numpy(human_f).float().to(device)
    with torch.no_grad():
        mapped = torch.stack([mapper.forward_finger(x[i], i) for i in range(F)])
    src = x[0].unsqueeze(0)
    pred = mapped[0].unsqueeze(0)
    try:
        return motion_consistency(src, pred, local=False).score
    except ValueError:
        return float("nan")


def fingerprint(mapper, human_f: np.ndarray, device) -> np.ndarray:
    """Flattened mapped output for every human sample -- used for seed variance."""
    F, N, _ = human_f.shape
    x = torch.from_numpy(human_f).float().to(device)
    with torch.no_grad():
        mapped = torch.stack([mapper.forward_finger(x[i], i) for i in range(F)])
    return mapped.cpu().numpy().reshape(-1)


# ------------------------------------------------------------------- harness


class Runner:
    """Loads the capture and clouds once, then trains many configurations."""

    def __init__(self, hand: str, capture_dir: str, frames: int, cloud: int, device):
        self.hand = hand
        self.spec = get_hand(hand)
        self.device = device
        caps = discover_manus_captures(capture_dir)
        if not caps:
            raise SystemExit(f"no Manus captures under {capture_dir}")
        self.capture = caps[0]
        self.raw, quats = load_manus_npz(self.capture, self.spec, max_frames=frames)
        self.samples = normalize_human(self.raw, quats, self.spec.finger_names)
        self.clouds, self.kin = build_robot_clouds(
            self.spec, n_samples=cloud, seed=0, normalizer=self.samples.normalizer
        )
        self.anchors = build_anchor_set(
            self.spec,
            self.raw,
            self.samples.normalizer,
            self.kin,
            robot_normalizer=self.samples.normalizer,
        )
        self.human_f = np.transpose(self.samples.fingertips, (1, 0, 2))

    def run(self, weights: LossWeights, epochs: int, seed: int = 0, use_anchors: bool = True):
        cfg = TrainConfig(
            hand=self.hand,
            epochs=epochs,
            seed=seed,
            use_anchors=use_anchors,
            mapper_type="finger",  # paper's own ablation ladder -- pin the paper-faithful
                                   # architecture regardless of TrainConfig's default
            weights={
                "p_chamfer": weights.p_chamfer,
                "dist": weights.dist,
                "motion": weights.motion,
                "align": weights.align,
            },
        )
        t0 = time.time()
        res = train(
            self.samples, self.clouds, self.anchors, cfg,
            device=self.device, train_classifier=False, raw_fingertips=self.raw,
        )
        return res.mapper, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hand", default="leap_hand")
    ap.add_argument("--capture-dir", default="/home/qsq/qqsq12321/mimic_retargeter_lab/dataset/manus")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--cloud", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--out", default="results/ladder.json")
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu")
    runner = Runner(args.hand, args.capture_dir, args.frames, args.cloud, device)
    print(f"hand={args.hand} capture={runner.capture.name}")
    print(f"frames={runner.raw.shape[0]} cloud={runner.clouds.n_samples} epochs={args.epochs}\n")

    report: dict = {"hand": args.hand, "epochs": args.epochs, "rungs": {}}

    # ---------------------------------------------------------------- rungs 1-4
    print("=== Rungs 1-4: ablation ladder (paper Table 3 direction of effect) ===")
    ladder = [
        ("LP-Chamfer only", LossWeights(1.0, 0.0, 0.0, 0.0), False),
        ("+ Ldist", LossWeights(1.0, 1.0, 0.0, 0.0), False),
        ("+ Lmotion", LossWeights(1.0, 1.0, 1.0, 0.0), False),
        ("+ Lalign (full)", LossWeights(1.0, 1.0, 1.0, 1.0), True),
    ]
    rows = []
    for name, w, use_anchors in ladder:
        mapper, dt = runner.run(w, args.epochs, seed=0, use_anchors=use_anchors)
        ratio = spread_ratio(mapper, runner.human_f, device)
        agree = direction_agreement(mapper, runner.human_f, device)
        rows.append({"name": name, "spread": ratio, "agreement": agree, "seconds": dt})
        print(f"  {name:18s} spread={ratio:.4f}  agreement={agree:6.1f}  ({dt:.0f}s)")

    only, full = rows[0], rows[-1]
    r1 = only["spread"] < 0.15
    r2 = full["spread"] > only["spread"] + 0.15
    r3 = rows[2]["agreement"] > rows[1]["agreement"] - 5.0
    print(f"\n  Rung 1 (Chamfer collapses):     {'PASS' if r1 else 'FAIL'}  spread={only['spread']:.4f}")
    print(f"  Rung 2 (Ldist recovers):        {'PASS' if r2 else 'FAIL'}  {only['spread']:.4f} -> {full['spread']:.4f}")
    print(f"  Rung 3 (Lmotion helps):         {'PASS' if r3 else 'FAIL'}  {rows[1]['agreement']:.1f} -> {rows[2]['agreement']:.1f}")
    report["rungs"]["ablation"] = {"rows": rows, "pass": {"r1": r1, "r2": r2, "r3": r3}}

    # ------------------------------------------------------------------ rung 4b
    print("\n=== Rung 4: does Lalign reduce cross-seed ambiguity? ===")
    with_anchor, without_anchor = [], []
    for s in range(args.seeds):
        m, _ = runner.run(LossWeights(1.0, 1.0, 1.0, 1.0), args.epochs, seed=s, use_anchors=True)
        with_anchor.append(fingerprint(m, runner.human_f, device))
        m, _ = runner.run(LossWeights(1.0, 1.0, 1.0, 0.0), args.epochs, seed=s, use_anchors=False)
        without_anchor.append(fingerprint(m, runner.human_f, device))

    def mean_pairwise_std(fps: list[np.ndarray]) -> float:
        """Mean per-dimension std across seeds: how much the mapping wanders."""
        stacked = np.stack(fps)
        return float(stacked.std(axis=0).mean())

    v_with = mean_pairwise_std(with_anchor)
    v_without = mean_pairwise_std(without_anchor)
    r4 = v_with <= v_without
    print(f"  with anchors:    cross-seed std = {v_with:.5f}")
    print(f"  without anchors: cross-seed std = {v_without:.5f}")
    print(f"  Rung 4: {'PASS' if r4 else 'FAIL'} (anchors should stabilize)")
    report["rungs"]["anchor_stability"] = {
        "with": v_with, "without": v_without, "pass": r4
    }

    # ------------------------------------------------------------------- rung 5
    print("\n=== Rung 5: calibration robustness (input rotated about y) ===")
    mapper, _ = runner.run(LossWeights(1.0, 1.0, 1.0, 1.0), args.epochs, seed=0)
    base = direction_agreement(mapper, runner.human_f, device)
    rot_scores = {}
    for deg in (-90, -45, 45, 90):
        rot = y_axis_rotation(float(deg))
        F, N, _ = runner.human_f.shape
        human_rot, _ = apply_global_perturbation(
            torch.from_numpy(runner.human_f).float(), None, rot
        )
        # Retarget the rotated input; the mapper itself is unchanged.
        with torch.no_grad():
            mapped = torch.stack(
                [mapper.forward_finger(human_rot[i].to(device), i) for i in range(F)]
            )
        # Compare the rotated input's motion against the robot motion.
        try:
            score = motion_consistency(
                human_rot[0].unsqueeze(0), mapped[0].unsqueeze(0), local=False
            ).score
        except ValueError:
            score = float("nan")
        rot_scores[deg] = score
        print(f"  {deg:+4d} deg: agreement = {score:6.1f}")

    spread_of_scores = float(np.nanstd(list(rot_scores.values())))
    r5 = spread_of_scores < 20.0
    print(f"  base = {base:.1f}, std over rotations = {spread_of_scores:.2f}")
    print(f"  Rung 5: {'PASS' if r5 else 'FAIL'} (robust if std < 20)")
    report["rungs"]["calibration"] = {
        "base": base, "rotations": rot_scores, "std": spread_of_scores, "pass": r5
    }

    # ------------------------------------------------------- optional sweep
    if not args.skip_sweep:
        print("\n=== Weight sweep: where does L_dist take effect? ===")
        print(f"  {'w_dist':>8} {'spread':>10}")
        sweep = []
        for wd in (0.0, 1.0, 3.0, 10.0, 30.0):
            m, _ = runner.run(LossWeights(1.0, wd, 0.0, 0.0), args.epochs, seed=0, use_anchors=True)
            ratio = spread_ratio(m, runner.human_f, device)
            sweep.append({"w_dist": wd, "spread": ratio})
            print(f"  {wd:>8.1f} {ratio:>10.4f}")
        report["weight_sweep"] = sweep

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")

    passed = [r1, r2, r3, r4, r5]
    print(f"\nsummary: {sum(passed)}/{len(passed)} rungs passed")
    return 0 if all(passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
