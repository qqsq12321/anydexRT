"""Render held-out AVP frames: human skeleton next to Wuji and L20 results.

Loads the bundles written by `scripts/run_avp.py`, picks representative frames
from the held-out blocks (each thumb-finger pinch, the most open hand, the most
closed hand), and writes one contact sheet:

    rows = frames;  columns = human skeleton (palm frame) | one render per hand

The robot camera is placed in each hand's own palm frame, so the views are
comparable across hands whose MJCF bases are oriented differently.

Usage:
    MUJOCO_GL=osmesa python scripts/render_avp.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anydexrt.anchors import MEDIAPIPE_CHAINS  # noqa: E402
from anydexrt.data import load_avp_pkl  # noqa: E402
from anydexrt.inference import load_bundle  # noqa: E402

FINGER_COLORS = {"thumb": "tab:red", "index": "tab:orange", "middle": "tab:green",
                 "ring": "tab:blue", "pinky": "tab:purple"}


def pick_frames(landmarks: np.ndarray, idx: np.ndarray) -> list[tuple[str, int]]:
    """Representative held-out frames: each pinch, most open, most closed."""
    L = landmarks[idx]
    tips = L[:, [4, 8, 12, 16, 20]]
    picks = []
    for i, name in enumerate(["index", "middle", "ring", "pinky"], start=1):
        gap = np.linalg.norm(tips[:, i] - tips[:, 0], axis=-1)
        picks.append((f"thumb-{name} pinch ({gap.min() * 1000:.0f} mm)", int(idx[np.argmin(gap)])))
    reach = np.linalg.norm(tips[:, 1:], axis=-1).mean(1)
    picks.append(("most open", int(idx[np.argmax(reach)])))
    picks.append(("most closed", int(idx[np.argmin(reach)])))
    return picks


def draw_skeleton(ax, L: np.ndarray, title: str) -> None:
    for name, chain in MEDIAPIPE_CHAINS.items():
        pts = L[[0, *chain]]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", ms=2, lw=2, color=FINGER_COLORS[name])
    ax.set_title(title, fontsize=8)
    lim = 0.1
    ax.set_xlim(0, 2 * lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    # Same palm-side viewpoint as the robot renders (looking along +z).
    ax.view_init(elev=-70, azim=-125)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def render_robot(rt, qpos: np.ndarray, size=(320, 320)) -> np.ndarray:
    kin = rt.kin
    kin.set_qpos(qpos)
    origin, R = kin.palm_frame(qpos)
    kin.set_qpos(qpos)  # palm_frame re-ran FK at qpos; keep the pose
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    is_l20 = "l20" in kin.hand.name
    cam.lookat[:] = origin + R @ np.array([0.15 if is_l20 else 0.10, 0.0, 0.0])
    # Look at the palm (volar) side, -z of the palm frame, slightly off-axis.
    view = R @ np.array([-0.2, -0.3, -0.9])
    view /= np.linalg.norm(view)
    fwd = -view
    cam.azimuth = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
    cam.elevation = float(np.degrees(np.arcsin(fwd[2])))
    cam.distance = 0.45 if is_l20 else 0.35
    renderer = mujoco.Renderer(kin.model, size[0], size[1])
    renderer.update_scene(kin.data, camera=cam)
    img = renderer.render()
    renderer.close()
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--avp", default="/home/qsq/qqsq12321/AnyDexRetarget/example/data/avp1.pkl")
    ap.add_argument("--results", default="results/avp")
    ap.add_argument("--hands", nargs="+", default=["wuji_hand", "linker_l20"])
    ap.add_argument("--out", default="results/avp/contact_sheet.png")
    args = ap.parse_args()

    bundles = {h: load_bundle(Path(args.results) / h / f"{h}_bundle.pt") for h in args.hands}
    first = next(iter(bundles.values()))
    extra = torch.load(
        Path(args.results) / args.hands[0] / f"{args.hands[0]}_bundle.pt", weights_only=False
    )["extra"]
    rec = load_avp_pkl(args.avp, first.kin.hand)
    edges = np.linspace(0, rec.n_frames, extra["blocks"] + 1).astype(int)
    held = np.concatenate([np.arange(edges[b], edges[b + 1]) for b in extra["test_blocks"]])
    picks = pick_frames(rec.landmarks, held)

    cols = 1 + len(args.hands)
    fig = plt.figure(figsize=(3.0 * cols, 2.9 * len(picks)))
    for r, (label, t) in enumerate(picks):
        ax = fig.add_subplot(len(picks), cols, r * cols + 1, projection="3d")
        draw_skeleton(ax, rec.landmarks[t], f"AVP frame {t}: {label}")
        for c, (hand, rt) in enumerate(bundles.items(), start=2):
            if rt.kin.hand.finger_names != first.kin.hand.finger_names:
                raise ValueError("all hands on one sheet must share a finger set")
            rt.reset()
            # Warm the continuity prior on the 10 preceding frames, as in streaming.
            for frame in rec.positions[max(0, t - 10) : t + 1]:
                q = rt.retarget(frame)
            contact = rt.last_contacts
            ax = fig.add_subplot(len(picks), cols, r * cols + c)
            ax.imshow(render_robot(rt, q))
            tag = ""
            if contact is not None and contact.any():
                tag = " | f_c: " + ",".join(
                    rt.kin.hand.finger_names[i + 1] for i in np.where(contact)[0]
                )
            ax.set_title(f"{hand}{tag}", fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")
    for label, t in picks:
        print(f"  frame {t:5d}  {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
