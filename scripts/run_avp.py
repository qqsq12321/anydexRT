"""AnyDexRT on Apple Vision Pro input, for Wuji Hand and Linker L20.

Pipeline under test:

    AVP landmarks (wrist frame) --skeleton palm frame--> human fingertips (palm frame)
      --f_m--> robot targets --f_s (per-finger NNS)--> drives --FK--> robot fingertips

The recording is split into contiguous blocks; `--test-blocks` are held out
(default 5 and 9 of 10, chosen so the held-out set contains the recording's
second pinch sequence -- see the argument help). f_m and f_c train on the rest.
Every metric below is computed on held-out frames only, on the robot side after
forward kinematics unless stated.

Metrics
  gmc_mapped   motion consistency of f_m's targets vs the human, palm frame.
               Measures f_m alone.
  gmc_robot    the same for the FK fingertips the robot actually reaches.
               Includes f_s's table quantization.
  nns_mm       f_s residual: distance from f_m's target to the chosen table tip.
  pinch        on frames where the human thumb-to-finger gap is < 2 cm, the
               robot's gap; plus false pinches (human gap > 5 cm, robot < 2 cm).
  contact      f_c precision / recall against the geometric labels.
  jitter       mean |dq| between consecutive frames, rad.
  hz           end-to-end frames per second on CPU.

Conditions
  identity     no f_m: per-finger normalization alone (the natural baseline)
  chamfer      L_P-Chamfer only
  +dist, +motion, full (+align)   the paper's ablation ladder

Usage:
    python scripts/run_avp.py --hands wuji_hand linker_l20 --epochs 300
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
from anydexrt.data import build_robot_clouds, load_avp_pkl, normalize_human  # noqa: E402
from anydexrt.hands import get_hand  # noqa: E402
from anydexrt.inference import AnyDexRT, make_pinch_templates, save_bundle  # noqa: E402
from anydexrt.metrics import motion_consistency  # noqa: E402
from anydexrt.train import TrainConfig, derive_contact_labels, train  # noqa: E402

DEFAULT_AVP = "/home/qsq/qqsq12321/AnyDexRetarget/example/data/avp1.pkl"

CONDITIONS = {
    "identity": None,
    "chamfer": {"p_chamfer": 1.0, "dist": 0.0, "motion": 0.0, "align": 0.0},
    "+dist": {"p_chamfer": 1.0, "dist": 1.0, "motion": 0.0, "align": 0.0},
    "+motion": {"p_chamfer": 1.0, "dist": 1.0, "motion": 1.0, "align": 0.0},
    "full": {"p_chamfer": 1.0, "dist": 1.0, "motion": 1.0, "align": 1.0},
}


def gmc(segments_h: list[np.ndarray], segments_r: list[np.ndarray], stride: int) -> float:
    """Pooled GMC over fingers and contiguous segments; each segment (T, F, 3).

    Segments are scored separately and combined weighted by their number of
    valid displacements, so no displacement spans the gap between two
    held-out blocks.
    """
    total, n = 0.0, 0
    for human, robot in zip(segments_h, segments_r):
        h = torch.from_numpy(np.transpose(human[::stride], (1, 0, 2))).float()
        r = torch.from_numpy(np.transpose(robot[::stride], (1, 0, 2))).float()
        try:
            res = motion_consistency(h, r, local=False, min_displacement=1e-4)
        except ValueError:
            continue
        total += res.score * res.n_samples
        n += res.n_samples
    return total / n if n else float("nan")


def pinch_stats(human: np.ndarray, robot: np.ndarray, names) -> dict:
    out = {}
    for i in range(1, human.shape[1]):
        dh = np.linalg.norm(human[:, i] - human[:, 0], axis=-1)
        dr = np.linalg.norm(robot[:, i] - robot[:, 0], axis=-1)
        pinch, apart = dh < 0.02, dh > 0.05
        out[names[i]] = {
            "n_pinch": int(pinch.sum()),
            "robot_gap_mm_on_pinch": float(np.median(dr[pinch]) * 1000) if pinch.any() else None,
            "robot_closed_on_pinch": float((dr[pinch] < 0.02).mean()) if pinch.any() else None,
            "false_pinch_rate": float((dr[apart] < 0.02).mean()) if apart.any() else None,
        }
    return out


def contact_stats(rt: AnyDexRT, human: np.ndarray, names) -> dict | None:
    if rt.classifier is None:
        return None
    labels = derive_contact_labels(human, 0.2)
    pred = np.stack([rt.contact_scores(h) >= 0.5 for h in human])
    out = {}
    for j in range(labels.shape[1]):
        y, p = labels[:, j] > 0.5, pred[:, j]
        tp = int((p & y).sum())
        out[names[j + 1]] = {
            "n_pos": int(y.sum()),
            "precision": tp / max(int(p.sum()), 1),
            "recall": tp / max(int(y.sum()), 1),
        }
    return out


def evaluate(rt: AnyDexRT, segments: list[np.ndarray], clouds, kin, names, stride: int) -> dict:
    """Run each held-out segment from a fresh state (no prior across the gap)."""
    qs, robots, mappeds, resid, contacts, n_frames, seconds = [], [], [], [], [], 0, 0.0
    for seg in segments:
        qpos, stats = rt.retarget_sequence(seg)
        qs.append(qpos)
        robots.append(clouds.to_cloud_frame(kin.fingertip_positions_batch(qpos)))
        mappeds.append(np.stack([clouds.normalizer.inverse(rt.map_fingertips(h)) for h in seg]))
        resid.append(stats["nns_residual_norm"])
        contacts.append(stats["contacts"])
        n_frames += len(seg)
        seconds += len(seg) / stats["hz"]
    scale = np.array([n.scale for n in clouds.normalizer.normalizers])
    nns_mm = np.concatenate(resid) * scale[None] * 1000
    human_all, robot_all = np.concatenate(segments), np.concatenate(robots)
    return {
        "gmc_mapped": gmc(segments, mappeds, stride),
        "gmc_robot": gmc(segments, robots, stride),
        "nns_mm_mean": float(nns_mm.mean()),
        "nns_mm_p95": float(np.percentile(nns_mm, 95)),
        "pinch": pinch_stats(human_all, robot_all, names),
        "jitter_rad": float(np.mean(np.concatenate([np.abs(np.diff(q, axis=0)).mean(1) for q in qs]))),
        "hz": n_frames / seconds,
        "contact_fires": float(np.concatenate(contacts).any(axis=1).mean()),
    }, qs


def block_split(n: int, n_blocks: int, test_blocks: list[int]) -> tuple[np.ndarray, list[slice]]:
    """Contiguous blocks; returns (train indices, list of test slices)."""
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    test = [slice(edges[b], edges[b + 1]) for b in sorted(test_blocks)]
    mask = np.ones(n, dtype=bool)
    for s in test:
        mask[s] = False
    return np.where(mask)[0], test


def run_hand(hand: str, args) -> dict:
    spec = get_hand(hand)
    rec = load_avp_pkl(args.avp, spec, side=args.side)
    train_idx, test_slices = block_split(rec.n_frames, args.blocks, args.test_blocks)
    h_train = rec.positions[train_idx]
    h_test = [rec.positions[s] for s in test_slices]
    n_train, n_test = len(train_idx), sum(len(s) for s in h_test)
    print(f"\n######## {hand}  ({spec.n_fingers} fingers, {spec.n_joints} drives)")
    print(f"AVP {rec.path.name} side={args.side}: {rec.n_frames} frames in {args.blocks} blocks, "
          f"test blocks {args.test_blocks} ({n_test} frames), train {n_train}; "
          f"palm-frame spread {rec.frame_spread_deg.mean():.2f} deg mean")

    samples = normalize_human(h_train, rec.identity_quats()[train_idx], spec.finger_names)
    clouds, kin = build_robot_clouds(spec, n_samples=args.cloud, seed=0, frame="palm")
    anchors = build_anchor_set(
        spec, h_train, samples.normalizer, kin,
        robot_normalizer=clouds.normalizer, strategy="skeleton",
        landmarks=rec.landmarks[train_idx],
        robot_frame=(clouds.frame_origin, clouds.frame_rot),
    )
    templates = make_pinch_templates(clouds, kin)
    gaps = {spec.finger_names[i]: round(float(np.median(t.gap) * 1000), 2) for i, t in templates.items()}
    print(f"pinch templates: {sum(len(t.gap) for t in templates.values())}, median gap mm {gaps}")
    res = anchors.residual_deg
    bend = [j for j, t in enumerate(anchors.types) if t == "bending"]
    print(f"anchor residual deg (bending, median per finger): "
          f"{np.round(np.median(res[:, bend], axis=1), 1).tolist()}")

    report = {"hand": hand, "n_train": n_train, "n_test": n_test,
              "anchor_bending_residual_deg": np.median(res[:, bend], axis=1).tolist(),
              "conditions": {}}

    # Setup checkpoint directory if interval is specified
    checkpoint_dir = None
    if args.checkpoint_interval is not None and args.checkpoint_interval > 0:
        checkpoint_dir = Path(args.out) / hand / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for name, weights in CONDITIONS.items():
        t0 = time.time()
        cfg = TrainConfig(
            hand=hand, epochs=args.epochs if weights is not None else 0, seed=0,
            weights=weights or CONDITIONS["full"],
            use_anchors=(name == "full"),
        )
        result = train(samples, clouds, anchors, cfg, raw_fingertips=h_train,
                       train_classifier=(name == "full"),
                       checkpoint_interval=args.checkpoint_interval,
                       checkpoint_dir=checkpoint_dir / name if checkpoint_dir else None)
        rt = AnyDexRT(result.mapper, clouds, kin, samples.normalizer,
                      classifier=result.classifier, pinch_templates=templates,
                      identity_mapper=(weights is None))
        metrics, qpos = evaluate(rt, h_test, clouds, kin, spec.finger_names, args.stride)
        if name == "full":
            full_mapped = np.concatenate([
                np.stack([clouds.normalizer.inverse(rt.map_fingertips(h)) for h in seg])
                for seg in h_test
            ])
            metrics["contact"] = contact_stats(rt, np.concatenate(h_test), spec.finger_names)
            rt_nofc = AnyDexRT(result.mapper, clouds, kin, samples.normalizer,
                               classifier=None, pinch_templates=None)
            metrics["without_fc"], _ = evaluate(rt_nofc, h_test, clouds, kin, spec.finger_names, args.stride)
            out = Path(args.out) / f"{hand}_bundle.pt"
            save_bundle(out, result, hand, samples.normalizer, args.cloud,
                        extra={"avp": str(rec.path), "side": args.side,
                               "test_blocks": args.test_blocks, "blocks": args.blocks})
            np.save(Path(args.out) / f"{hand}_test_qpos.npy", np.concatenate(qpos))
            metrics["bundle"] = str(out)
        pi = metrics["pinch"]["index"]
        print(f"  {name:9s} gmc_mapped={metrics['gmc_mapped']:6.1f} gmc_robot={metrics['gmc_robot']:6.1f} "
              f"nns={metrics['nns_mm_mean']:.2f}mm idx_pinch(n={pi['n_pinch']}) "
              f"gap={pi['robot_gap_mm_on_pinch'] or float('nan'):.1f}mm "
              f"closed={pi['robot_closed_on_pinch'] or 0:.2f} jitter={metrics['jitter_rad']:.4f} "
              f"{metrics['hz']:.0f}Hz ({time.time() - t0:.0f}s)")
        report["conditions"][name] = metrics

    # Seed stability of the full objective (paper Fig. 6 analogue): std of the
    # mapped targets on the test frames across seeds, in mm. Seed 0 is the
    # 'full' condition above.
    outs = [full_mapped]
    test_all = np.concatenate(h_test)
    for seed in range(1, args.seeds):
        cfg = TrainConfig(hand=hand, epochs=args.epochs, seed=seed, weights=CONDITIONS["full"])
        result = train(samples, clouds, anchors, cfg, raw_fingertips=h_train, train_classifier=False)
        rt = AnyDexRT(result.mapper, clouds, kin, samples.normalizer)
        outs.append(np.stack([clouds.normalizer.inverse(rt.map_fingertips(h)) for h in test_all]))
    std_mm = float(np.stack(outs).std(axis=0).mean() * 1000)
    report["seed_std_mm"] = std_mm
    print(f"  seed stability (full, {args.seeds} seeds): mean std of mapped targets = {std_mm:.2f} mm")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hands", nargs="+", default=["wuji_hand", "linker_l20"])
    ap.add_argument("--avp", default=DEFAULT_AVP)
    ap.add_argument("--side", default="right", choices=["right", "left"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--cloud", type=int, default=20000)
    # The shipped avp1.pkl contains two passes of an index->middle->ring->pinky
    # pinch sequence, in deciles 0-1 and 5; a plain tail split holds out no
    # pinch at all. Holding out blocks 5 and 9 tests the second pass with a
    # model that only saw the first.
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--test-blocks", type=int, nargs="+", default=[5, 9])
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results/avp")
    ap.add_argument("--checkpoint-interval", type=int, default=None,
                    help="save mapper every N epochs (e.g. 10); checkpoints go to <out>/<hand>/checkpoints/")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    reports = [run_hand(h, args) for h in args.hands]
    path = Path(args.out) / "report.json"
    path.write_text(json.dumps({"args": vars(args), "hands": reports}, indent=2))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
