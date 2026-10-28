"""Data pipeline: human fingertip samples and robot fingertip clouds.

Human side comes from Manus glove captures in mimic_retargeter_lab's dataset
layout: an .npz with `data` of shape (frames, 25, 7) whose channels are
"px, py, pz, qw, qx, qy, qz" (per the sidecar .json), plus `keypoint_ids`.

Robot side is sampled by forward kinematics: uniform joint samples inside the
MJCF joint limits, evaluated per finger so that each finger's cloud describes
that finger's own reachable fingertip space. That per-finger factorization
matters -- the mapper has one sub-net per finger, and the training target for
finger i is generated with the other fingers at neutral.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .hands import HandSpec, get_hand
from .kinematics import HandKinematics
from .normalize import HandNormalizer


@dataclass
class HandSamples:
    """Normalized fingertip samples plus the normalizers that produced them."""

    fingertips: np.ndarray  # (T, F, 3), normalized
    quats_wxyz: np.ndarray  # (T, F, 4), raw (un-normalized) orientations
    normalizer: HandNormalizer
    finger_names: tuple[str, ...]

    @property
    def n_frames(self) -> int:
        return self.fingertips.shape[0]

    @property
    def n_fingers(self) -> int:
        return self.fingertips.shape[1]


def load_manus_npz(
    path: str | Path,
    hand: HandSpec | str,
    max_frames: int | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Load raw human fingertips + orientations for a given robot hand.

    Args:
        path: .npz written by the Manus capture tool.
        hand: HandSpec or name; selects which human keypoints are used.
        max_frames: optional cap on frames read.
        stride: subsample every `stride` frames.

    Returns:
        positions (T, F, 3) in raw glove space, orientations (T, F, 4) wxyz.
    """
    hand = get_hand(hand) if isinstance(hand, str) else hand
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manus capture not found: {path}")

    with np.load(path) as npz:
        data = np.asarray(npz["data"], dtype=np.float32)  # (T, 25, 7)
    if data.ndim != 3 or data.shape[2] != 7:
        raise ValueError(
            f"expected (T, 25, 7) with px,py,pz,qw,qx,qy,qz; got {data.shape}"
        )

    ids = hand.human_ids()
    frames = data[::stride]
    if max_frames is not None:
        frames = frames[:max_frames]

    positions = frames[:, ids, :3]  # (T, F, 3)
    quats = frames[:, ids, 3:7]  # (T, F, 4) wxyz
    return positions, quats


def normalize_human(
    positions: np.ndarray,
    quats: np.ndarray,
    finger_names: tuple[str, ...],
) -> HandSamples:
    """Fit the per-finger normalizer on the human data and apply it.

    Fitting on the human cloud (not the robot cloud) is what makes the mapping
    scale-agnostic: the operator's hand size is divided out, so the same
    mapper configuration transfers between people.
    """
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 3:
        raise ValueError(f"expected (T, F, 3), got {positions.shape}")
    # HandNormalizer expects per-finger clouds: transpose (T,F,3) -> (F,T,3).
    normalizer = HandNormalizer.fit(finger_names, np.transpose(positions, (1, 0, 2)))
    normalized = normalizer.transform(np.transpose(positions, (1, 0, 2)))  # (F, T, 3)
    return HandSamples(
        fingertips=np.transpose(normalized, (1, 0, 2)).astype(np.float32),
        quats_wxyz=np.asarray(quats, dtype=np.float32),
        normalizer=normalizer,
        finger_names=tuple(finger_names),
    )


@dataclass
class RobotClouds:
    """Sampled robot fingertip space, normalized.

    `tips` are in the frame given by (frame_origin, frame_rot) -- world by
    default, the palm frame when built with frame='palm'. `to_cloud_frame`
    applies the same transform to fresh FK output so evaluation compares like
    with like.
    """

    qpos: np.ndarray  # (N, J) canonical joint order
    tips: np.ndarray  # (N, F, 3) in the cloud frame
    tips_norm: np.ndarray  # (N, F, 3) normalized
    tips_rot: np.ndarray  # (N, F, 3, 3) fingertip orientation, cloud frame
    normalizer: HandNormalizer
    frame_origin: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    frame_rot: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))

    def to_cloud_frame(self, world_points: np.ndarray) -> np.ndarray:
        return to_frame(world_points, self.frame_origin, self.frame_rot)

    @property
    def n_samples(self) -> int:
        return self.tips.shape[0]

    @property
    def n_fingers(self) -> int:
        return self.tips.shape[1]

    def per_finger_cloud(self, i: int) -> np.ndarray:
        """(N, 3) normalized cloud for finger i, in the order used by anchors/IK."""
        return self.tips_norm[:, i]

    def per_finger_rot(self, i: int) -> np.ndarray:
        return self.tips_rot[:, i]


def fingertip_orientations(
    kin: HandKinematics, qpos: np.ndarray
) -> np.ndarray:
    """Fingertip body orientations for each sampled configuration.

    Returns (N, F, 3, 3). Used only for the robot side of local frames; the
    nearest-neighbour rule in the paper attaches one of these to each mapped
    point.
    """
    import mujoco

    qpos = np.asarray(qpos, dtype=np.float64)
    out = np.empty((qpos.shape[0], kin.n_fingers, 3, 3), dtype=np.float32)
    for b in range(qpos.shape[0]):
        kin.set_qpos(qpos[b])
        for i, bid in enumerate(kin._tip_bids):
            out[b, i] = np.asarray(kin.data.xmat[bid], dtype=np.float32).reshape(3, 3)
    return out


def build_robot_clouds(
    hand: HandSpec | str,
    n_samples: int = 20000,
    seed: int = 0,
    asset_root: Path | None = None,
    normalizer: HandNormalizer | None = None,
    frame: str = "world",
) -> tuple[RobotClouds, HandKinematics]:
    """Sample the robot's reachable fingertip space.

    Args:
        normalizer: if given, reuse it (e.g. the human-side normalizer, which is
            what the early Manus/Leap experiments did). If None, fit on the robot
            cloud itself -- per finger, as Appendix B prescribes ("for each
            human/robot finger, we centralize ... and normalize"). The mapper
            then maps normalized human space to normalized robot space, and
            each side's translation and scale are removed independently.
        frame: 'world' keeps MJCF world coordinates. 'palm' expresses tips (and
            their orientations) in the hand's skeleton-derived palm frame, the
            same convention used for the human input; see
            `kinematics.palm_frame_from_points`.
    """
    if frame not in {"world", "palm"}:
        raise ValueError(f"frame must be 'world' or 'palm', got {frame!r}")
    hand_spec = get_hand(hand) if isinstance(hand, str) else hand
    kin = HandKinematics(hand_spec, asset_root=asset_root)
    rng = np.random.default_rng(seed)
    qpos, tips = kin.sample_fingertip_cloud(n_samples, rng)
    tips_rot = fingertip_orientations(kin, qpos)

    origin = np.zeros(3)
    rot = np.eye(3)
    if frame == "palm":
        origin, rot = kin.palm_frame()
        tips = to_frame(tips, origin, rot)
        tips_rot = np.einsum("ji,nfjk->nfik", rot, tips_rot).astype(np.float32)

    if normalizer is None:
        normalizer = HandNormalizer.fit(
            hand_spec.finger_names, np.transpose(tips, (1, 0, 2))
        )
    tips_norm = np.transpose(
        normalizer.transform(np.transpose(tips, (1, 0, 2))), (1, 0, 2)
    ).astype(np.float32)

    clouds = RobotClouds(
        qpos=qpos,
        tips=tips,
        tips_norm=tips_norm,
        tips_rot=tips_rot,
        normalizer=normalizer,
        frame_origin=np.asarray(origin, dtype=np.float32),
        frame_rot=np.asarray(rot, dtype=np.float32),
    )
    return clouds, kin


def to_frame(points: np.ndarray, origin: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Express (..., 3) world points in a frame given by (origin, R columns = axes)."""
    pts = np.asarray(points, dtype=np.float64)
    return ((pts - np.asarray(origin, dtype=np.float64)) @ np.asarray(rot, dtype=np.float64)).astype(
        np.float32
    )


# ------------------------------------------------------------- Apple Vision Pro


@dataclass
class AVPRecording:
    """A right- or left-hand stream from an AVP recording, in the palm frame.

    `positions` are fingertips for the requested robot hand's fingers, shape
    (T, F, 3), metres, expressed in the human's skeleton-derived palm frame.
    `palm_rot` is the single rotation used for that (see `load_avp_pkl`), and
    `frame_spread_deg` records how far the per-frame palm frames stray from it.
    """

    positions: np.ndarray
    landmarks: np.ndarray  # (T, 21, 3) full skeleton, palm frame
    palm_rot: np.ndarray
    frame_spread_deg: np.ndarray  # (T,)
    side: str
    path: Path

    @property
    def n_frames(self) -> int:
        return self.positions.shape[0]

    def identity_quats(self) -> np.ndarray:
        """(T, F, 4) wxyz identity orientations.

        The recordings store positions only (MediaPipe layout), so there is no
        per-joint orientation to give the human side of T(x). Identity is
        returned so the global-frame objective -- the default -- can run.
        """
        q = np.zeros(self.positions.shape[:2] + (4,), dtype=np.float32)
        q[..., 0] = 1.0
        return q


def palm_tips_from_avp_poses(poses: np.ndarray, hand: HandSpec | str) -> np.ndarray:
    """Live AVP frame (25, 4, 4) -> (F, 3) fingertips in this frame's palm frame.

    The 25-joint skeleton shares the Manus layout (wrist, thumb 1-4, then four
    5-joint fingers), so fingertips are `human_ids('manus')` and the palm is
    built from joints (0, 6, 11, 21) -- the same physical points the 21-landmark
    recordings use (MediaPipe 0, 5, 9, 17 map to AVP 0, 6, 11, 21 via
    VP_TO_MEDIAPIPE).
    """
    from .hands import HUMAN_PALM_IDS
    from .kinematics import palm_frame_from_points

    hand = get_hand(hand) if isinstance(hand, str) else hand
    p = np.asarray(poses, dtype=np.float64)
    if p.ndim != 3 or p.shape[0] < 25 or p.shape[1:] != (4, 4):
        raise ValueError(f"expected (25, 4, 4) AVP poses, got {p.shape}")
    pos = p[:, :3, 3]
    w, i, m, k = HUMAN_PALM_IDS["manus"]
    rot = palm_frame_from_points(pos[w], pos[i], pos[m], pos[k])
    return to_frame(pos[hand.human_ids("manus")], pos[w], rot)


def palm_tips_from_mediapipe_points(points21: np.ndarray, hand: HandSpec | str) -> np.ndarray:
    """Live single-frame (21, 3) MediaPipe-order, wrist-relative points ->
    (F, 3) fingertips in this frame's palm frame.

    For any source that natively outputs MediaPipe landmarks instead of AVP's
    25-joint skeleton -- e.g. PICO 4 hand tracking
    (AnyDexRetarget/example/input/pico4.py), whose JOINT_21_INDICES already
    produces this exact (21, 3), wrist-origin format. Mirrors
    `palm_tips_from_avp_poses`'s per-frame approach and `load_avp_pkl`'s
    per-frame palm construction -- same four palm points, same
    `HUMAN_PALM_IDS["mediapipe"]` / `human_ids("mediapipe")` indices, just a
    single live frame instead of a recording. No side-mirroring here, to stay
    symmetric with `palm_tips_from_avp_poses` (callers mirror left-hand input
    externally, same convention `teleop_avp.py` already uses for AVP).
    """
    from .hands import HUMAN_PALM_IDS
    from .kinematics import palm_frame_from_points

    hand = get_hand(hand) if isinstance(hand, str) else hand
    pos = np.asarray(points21, dtype=np.float64)
    if pos.shape != (21, 3):
        raise ValueError(f"expected (21, 3) MediaPipe landmarks, got {pos.shape}")
    w, i, m, k = HUMAN_PALM_IDS["mediapipe"]
    rot = palm_frame_from_points(pos[w], pos[i], pos[m], pos[k])
    return to_frame(pos[hand.human_ids("mediapipe")], pos[w], rot)


def thumb_straightness(mcp: np.ndarray, ip: np.ndarray, tip: np.ndarray) -> float:
    """How straight the thumb is, in [0, 1]: 1 = MCP/IP/TIP collinear.

    `|tip - mcp| / (|ip - mcp| + |tip - ip|)`, i.e. end-to-end distance over
    path length along the bones. A SHAPE ratio, so unlike a distance in mm it
    does not change with the operator's hand size or the tracker's own bone
    conventions -- which is the point. Measured on curled-fingers frames, the
    thumb's distance from the palm origin separates a fist from an extended
    thumb on AVP (95-106mm vs 119-142mm) but NOT on PICO 4, whose thumb reach
    is compressed into 90-122mm with no gap in between; PICO's OpenXR thumb
    root (ThumbMetacarpal) sits further out than AVP's THUMB_CMC, making the
    thumb measure 25% shorter end to end. This ratio splits both devices at
    the same place: curled < 0.88, extended > 0.94 on each.
    """
    path = np.linalg.norm(ip - mcp) + np.linalg.norm(tip - ip)
    if path < 1e-9:
        return 0.0
    return float(np.linalg.norm(tip - mcp) / path)


# Index of each MediaPipe landmark in the AVP 25-joint skeleton
# (AnyDexRetarget/example/input/visionpro.py: VP_TO_MEDIAPIPE).
VP_TO_MEDIAPIPE = (0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24)

# MediaPipe landmark indices of the three thumb joints `thumb_straightness`
# needs (THUMB_MCP, THUMB_IP, THUMB_TIP). The same indices address the AVP
# 25-joint skeleton directly -- VP_TO_MEDIAPIPE is the identity for 0..4.
THUMB_MCP_IP_TIP = (2, 3, 4)


def load_avp_pkl(
    path: str | Path,
    hand: HandSpec | str,
    side: str = "right",
    stride: int = 1,
    max_frames: int | None = None,
) -> AVPRecording:
    """Load an AVP recording saved as a list of {'t', 'left_fingers', 'right_fingers'}.

    Each `*_fingers` entry is (21, 3) in MediaPipe order, already relative to
    the wrist (landmark 0 is identically zero), as produced by
    AnyDexRetarget/example/input/visionpro.py.

    The palm frame is built from the MEAN wrist/index/middle/pinky MCP
    positions over the recording rather than per frame. On avp1.pkl the
    per-frame frame strays from it by 0.8 deg on average (max 3.1), so a fixed
    rotation loses nothing and avoids injecting per-frame jitter. The mean, not
    the coordinate-wise median: the median of rotated coordinates is not the
    rotated median, which would make the rotation invariance below approximate
    (~0.3 mm drift was measured) instead of exact.
    """
    import pickle

    from .hands import HUMAN_PALM_IDS
    from .kinematics import palm_frame_from_points

    if side not in {"left", "right"}:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    hand = get_hand(hand) if isinstance(hand, str) else hand
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"AVP recording not found: {path}")
    with open(path, "rb") as fh:
        frames = pickle.load(fh)
    key = f"{side}_fingers"
    raw = np.stack([np.asarray(f[key], dtype=np.float64) for f in frames])  # (T, 21, 3)
    if raw.ndim != 3 or raw.shape[1:] != (21, 3):
        raise ValueError(f"expected (T, 21, 3) MediaPipe landmarks, got {raw.shape}")
    raw = raw[::stride]
    if max_frames is not None:
        raw = raw[:max_frames]

    # Drop frames where tracking was lost (all-zero placeholder).
    valid = np.abs(raw).sum(axis=(1, 2)) > 1e-9
    raw = raw[valid]

    w, i, m, p = HUMAN_PALM_IDS["mediapipe"]
    if side == "left":
        # Mirror to a right hand so the same palm-frame rule and the same
        # right-handed robot models apply.
        raw = raw * np.array([-1.0, 1.0, 1.0])
    rot = palm_frame_from_points(
        raw[:, w].mean(0), raw[:, i].mean(0), raw[:, m].mean(0), raw[:, p].mean(0)
    )
    per_frame = np.stack(
        [palm_frame_from_points(f[w], f[i], f[m], f[p]) for f in raw]
    )
    rel = np.einsum("ji,tjk->tik", rot, per_frame)
    spread = np.degrees(
        np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1))
    )

    landmarks = to_frame(raw, raw[:, w].mean(0), rot)
    positions = landmarks[:, hand.human_ids("mediapipe")]
    return AVPRecording(
        positions=positions.astype(np.float32),
        landmarks=landmarks.astype(np.float32),
        palm_rot=rot.astype(np.float32),
        frame_spread_deg=spread.astype(np.float32),
        side=side,
        path=path,
    )


def load_manus_metadata(path: str | Path) -> dict:
    """Read the sidecar .json next to a Manus .npz."""
    path = Path(path)
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    with open(sidecar) as fh:
        return json.load(fh)


def discover_manus_captures(dataset_dir: str | Path) -> list[Path]:
    """List Manus captures in a dataset directory, sorted by name."""
    d = Path(dataset_dir)
    if not d.is_dir():
        return []
    return sorted(d.glob("*.npz"))
