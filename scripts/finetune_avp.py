"""Fine-tune an existing f_m (fingertip mapper) on a short, supplementary AVP recording.

Unlike scripts/run_avp.py (which trains f_m + f_c from scratch across the full
ablation ladder), this is for the common follow-up case: you already have a
working bundle, and a short new recording to adjust f_m's coverage with --
e.g. a few seconds of open/close motion the original recording under-covered.
It warm-starts f_m from the existing bundle's weights instead of random init,
and skips f_c entirely (the classifier is untouched; see --carry-classifier).

Why this is a separate script rather than a flag on run_avp.py: run_avp.py's
CONDITIONS loop (identity/chamfer/+dist/+motion/full) exists to produce the
ablation table, which is meaningless for a fine-tune -- there is no "from
scratch with only L_P-Chamfer" step here, only a short continuation of the
existing full-objective mapper.

The robot-side cloud, NNS table and pinch templates are regenerated
deterministically from the bundle's own (hand, cloud_samples, cloud_seed), so
they are identical to what the base bundle used -- fine-tuning only touches
f_m's weights, nothing on the robot side.

Usage:
    python scripts/finetune_avp.py \\
        --bundle results/avp3/linker_l20/linker_l20_bundle.pt \\
        --avp data/avp_20261019_134551.pkl \\
        --epochs 50 --out results/avp4/linker_l20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anydexrt.anchors import build_anchor_set  # noqa: E402
from anydexrt.data import build_robot_clouds, load_avp_pkl, normalize_human  # noqa: E402
from anydexrt.hands import get_hand  # noqa: E402
from anydexrt.inference import AnyDexRT, make_pinch_templates, save_bundle  # noqa: E402
from anydexrt.normalize import HandNormalizer  # noqa: E402
from anydexrt.train import TrainConfig, train  # noqa: E402

# Same loss weights as run_avp.py's "full" condition -- a fine-tune continues
# training the complete objective, not an ablation stage.
FULL_WEIGHTS = {"p_chamfer": 1.0, "dist": 1.0, "motion": 1.0, "align": 1.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, help="existing bundle to warm-start f_m from")
    ap.add_argument("--avp", required=True, help="new, short recording to fine-tune on")
    ap.add_argument("--side", default="right", choices=["right", "left"])
    ap.add_argument("--epochs", type=int, default=50,
                    help="fine-tune epochs -- short recordings need far fewer than "
                         "the 300 used for from-scratch training; watch the loss log")
    ap.add_argument("--lr", type=float, default=None,
                    help="override TrainConfig.lr; default keeps the base bundle's own lr "
                         "(1e-4), which is deliberately gentle for a warm start")
    ap.add_argument(
        "--normalizer", default="keep", choices=["keep", "refit"],
        help="'keep' (default) reuses the base bundle's per-finger normalizer, which is "
             "almost always right for a short supplementary clip: a clip with a narrower "
             "range of motion (e.g. no pinches) would refit a SMALLER per-finger scale, "
             "silently distorting every other config that loads this bundle. 'refit' is "
             "only for deliberately replacing the operator/calibration baseline.",
    )
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="fraction of the new recording held out to report fine-tuned "
                         "GMC on; the rest is used for training")
    ap.add_argument("--out", required=True, help="output directory for the fine-tuned bundle")
    ap.add_argument("--carry-classifier", action="store_true", default=True,
                    help="copy f_c unchanged from the base bundle into the new one (default). "
                         "The new recording is not used to retrain f_c.")
    args = ap.parse_args()

    base_path = Path(args.bundle)
    payload = torch.load(base_path, map_location="cpu", weights_only=False)
    hand = payload["hand"]
    spec = get_hand(hand)
    base_cfg = payload["config"]
    print(f"base bundle: {base_path}  (hand={hand}, base lr={base_cfg['lr']}, "
          f"base epochs={base_cfg['epochs']})")

    rec = load_avp_pkl(args.avp, spec, side=args.side)
    n_val = max(1, int(args.val_frac * rec.n_frames))
    h_train, h_val = rec.positions[:-n_val], rec.positions[-n_val:]
    print(f"fine-tune recording: {rec.path.name}, {rec.n_frames} frames "
          f"(train {len(h_train)} / val {len(h_val)}); palm-frame spread "
          f"{rec.frame_spread_deg.mean():.2f} deg mean")
    if len(h_train) < 50:
        print(f"WARNING: only {len(h_train)} training frames -- this is a short clip; "
              "keep --epochs small (default 50) to avoid overfitting to it.")

    if args.normalizer == "keep":
        normalizer = HandNormalizer.from_dict(payload["human_normalizer"])
        print("reusing base bundle's per-finger normalizer")
    else:
        samples_tmp = normalize_human(h_train, rec.identity_quats()[: len(h_train)], spec.finger_names)
        normalizer = samples_tmp.normalizer
        print("refit a new per-finger normalizer from the fine-tune recording")

    # normalize_human fits its own normalizer by default; override with the one
    # we actually want (base or refit) so training and evaluation agree.
    # HandNormalizer.transform expects (F, N, 3) -- finger axis first -- hence
    # the transpose round-trip, matching normalize_human's own convention.
    samples = normalize_human(h_train, rec.identity_quats()[: len(h_train)], spec.finger_names)
    samples.normalizer = normalizer
    samples.fingertips = np.transpose(
        normalizer.transform(np.transpose(h_train, (1, 0, 2))), (1, 0, 2)
    ).astype(np.float32)

    clouds, kin = build_robot_clouds(
        spec, n_samples=payload["cloud_samples"], seed=payload["cloud_seed"], frame="palm"
    )
    # Anchors re-derived from the fine-tune clip's own skeleton coverage. With a
    # short, motion-limited clip these will have high residuals (the anchor
    # search can't find poses the clip never contains) -- see the printed
    # residuals below. L_align's contribution is correspondingly weaker here
    # than in a full from-scratch run; this is expected for a short fine-tune.
    anchors = build_anchor_set(
        spec, h_train, normalizer, kin,
        robot_normalizer=clouds.normalizer, strategy="skeleton",
        landmarks=rec.landmarks[: len(h_train)],
        robot_frame=(clouds.frame_origin, clouds.frame_rot),
    )
    res = anchors.residual_deg
    bend = [j for j, t in enumerate(anchors.types) if t == "bending"]
    print(f"anchor residual deg (bending, median per finger): "
          f"{np.round(np.median(res[:, bend], axis=1), 1).tolist()}  "
          "(high values are expected -- see script docstring)")

    cfg = TrainConfig(
        hand=hand, epochs=args.epochs, seed=base_cfg.get("seed", 0),
        lr=args.lr if args.lr is not None else base_cfg["lr"],
        weights=FULL_WEIGHTS, use_anchors=True,
        width=base_cfg["width"], depth=base_cfg["depth"],
        activation=base_cfg["activation"], output_tanh=base_cfg["output_tanh"],
        # Must match the base bundle's own architecture, not TrainConfig's
        # default -- init_mapper_state below is that bundle's state_dict, and
        # a mismatched mapper_type makes load_state_dict fail on shape.
        mapper_type=base_cfg.get("mapper_type", "finger"),
        thumb_idx=base_cfg.get("thumb_idx", 0),
    )

    mapper_init_state = payload["mapper"]
    t0 = time.time()
    result = train(
        samples, clouds, anchors, cfg,
        train_classifier=False,  # f_c is carried over untouched, see below
        init_mapper_state=mapper_init_state,
        log_fn=print,
    )
    dt = time.time() - t0
    print(f"fine-tuned f_m in {dt:.1f}s")

    # Validation on the held-out tail of the NEW recording.
    from anydexrt.metrics import motion_consistency

    rt = AnyDexRT(result.mapper, clouds, kin, normalizer, identity_mapper=False)
    rt.reset()
    qpos = np.stack([rt.retarget(h) for h in h_val])
    robot = clouds.to_cloud_frame(kin.fingertip_positions_batch(qpos))
    try:
        gmc = motion_consistency(
            torch.from_numpy(np.transpose(h_val, (1, 0, 2))).float(),
            torch.from_numpy(np.transpose(robot, (1, 0, 2))).float(),
            local=False, min_displacement=1e-4,
        ).score
    except ValueError:
        gmc = float("nan")
    print(f"val GMC on held-out tail of the fine-tune clip: {gmc:.1f}")

    # Carry the classifier over unchanged -- this script never touches f_c.
    classifier_state = payload["classifier"] if args.carry_classifier else None
    if classifier_state is not None:
        from anydexrt.models import ContactClassifier

        clf = ContactClassifier(
            n_fingers=spec.n_fingers, width=base_cfg["width"], depth=base_cfg["depth"],
            activation=base_cfg["activation"],
        )
        clf.load_state_dict(classifier_state)
        result.classifier = clf

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{hand}_bundle.pt"
    save_bundle(
        out_path, result, hand, normalizer, payload["cloud_samples"],
        cloud_seed=payload["cloud_seed"],
        extra={
            "fine_tuned_from": str(base_path),
            "fine_tune_avp": str(rec.path),
            "fine_tune_epochs": args.epochs,
            "fine_tune_val_gmc": gmc,
            "normalizer_source": args.normalizer,
        },
    )
    print(f"wrote {out_path}")

    report = {
        "base_bundle": str(base_path),
        "fine_tune_avp": str(rec.path),
        "fine_tune_frames": {"train": len(h_train), "val": len(h_val)},
        "epochs": args.epochs,
        "val_gmc": gmc,
        "anchor_bending_residual_deg": np.median(res[:, bend], axis=1).tolist(),
        "normalizer_source": args.normalizer,
        "history_tail": result.history[-5:],
    }
    (out_dir / "finetune_report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out_dir / 'finetune_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
