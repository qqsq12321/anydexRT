"""Few-shot anchor generation, following Appendix D.

The paper's process has three steps:

  (P1) Human Anchor Collection. Sample K0 joint configurations under each anchor
       type's constraint, and have the operator wear the glove and pose each
       finger accordingly. K0 = 5 per type; for bending, beta_1 rises from 0 to
       pi/2 in increments of pi/8. Only this step needs a human.

  (P2) Human Anchor Interpolation. Linearly interpolate between neighbouring
       human anchors to reach K anchors. K = 50 for lateral rotation, K = 100
       for bending.

  (P3) Robot Anchor Generation. "generated in simulation by uniformly sampling
       K joint configurations and collecting the corresponding fingertip
       positions."

Two anchor types (Fig. 9), parametrized by per-finger angles (alpha, beta_1,
beta_2, beta_3):
  * Lateral rotation: beta_1 = beta_2 = beta_3 = 0, vary alpha.
  * Bending: alpha = 0, vary beta with beta_1 = beta_2 = lambda*beta_3, lambda = 2.

IMPORTANT CAVEAT. (P1) in the paper is a live capture session with a specific
protocol -- the operator imitates reference gestures. We do not have such a
session. We therefore offer two human-side strategies, recorded in the returned
object so results can be labelled honestly:

  'reference' -- pick the frames from an existing glove capture that best match
                 each target anchor constraint. Cheap and uses real data, but
                 only valid if the capture actually covers the anchor range.
  'synthetic' -- build the anchor poses geometrically from a per-finger plane
                 model fitted to the capture (base point, axis, segment lengths).
                 Always available and monotonic in the anchor parameter, at the
                 cost of being an approximation of a real hand.

Either way, (P2) and (P3) follow the paper exactly. Notes on the mapping from
the paper's abstract angles (alpha, beta_1..3) to concrete poses: the paper's
lambda=2 relation assumes a 3-segment finger model where beta_3 drives the
distal joint. For a 4-DoF finger we fold the metacarpal into the base angle,
which is what the 'synthetic' model below does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hands import HandSpec, get_hand
from .kinematics import HandKinematics
from .normalize import HandNormalizer

# Paper's constants (Appendix D).
K0 = 5
K_LATERAL = 50
K_BENDING = 100
LAMBDA = 2.0
BETA1_STEP = np.pi / 8  # bending anchors rise 0 -> pi/2 in pi/8 steps


@dataclass
class AnchorSet:
    """Paired human/robot anchors for one hand.

    human: (F, K, 3) normalized human fingertip anchor positions.
    robot: (F, K, 3) normalized robot fingertip anchor positions.
    params: (F, K, 4) the (alpha, beta1, beta2, beta3) that produced each anchor.
    """

    human: np.ndarray
    robot: np.ndarray
    params: np.ndarray
    types: list[str]  # 'lateral' | 'bending', length K
    strategy: str  # 'reference' | 'synthetic' | 'skeleton'
    finger_names: tuple[str, ...] = field(default=())
    residual_deg: np.ndarray | None = None  # (F, K) for 'skeleton' anchors

    @property
    def n_fingers(self) -> int:
        return self.human.shape[0]

    @property
    def n_anchors(self) -> int:
        return self.human.shape[1]


# --------------------------------------------------------------------- params


def lateral_params(n: int = K_LATERAL) -> np.ndarray:
    """(n, 4) anchors for lateral rotation: alpha spans, betas pinned to 0.

    The paper does not give alpha's range. We use [-pi/2, pi/2], which covers
    the abduction/adduction range of a human finger.
    """
    alpha = np.linspace(-np.pi / 2, np.pi / 2, n, dtype=np.float32)
    zeros = np.zeros_like(alpha)
    return np.stack([alpha, zeros, zeros, zeros], axis=-1)


def bending_params(n: int = K_BENDING, lam: float = LAMBDA) -> np.ndarray:
    """(n, 4) anchors for bending: beta_1 = beta_2 = lam*beta_3, alpha = 0.

    beta_1 rises from 0 to pi/2 (the paper's (P1) constraint). The relation
    beta_1 = beta_2 = lambda*beta_3 is inverted to give beta_3 = beta_1/lambda,
    so the distal joint bends half as much as the proximal ones at lambda=2.
    """
    beta1 = np.linspace(0.0, np.pi / 2, n, dtype=np.float32)
    beta2 = beta1.copy()
    beta3 = beta1 / lam
    zeros = np.zeros_like(beta1)
    return np.stack([zeros, beta1, beta2, beta3], axis=-1)


# ------------------------------------------------------------------ synthetic


@dataclass
class FingerPlaneModel:
    """A crude plane-chain model of one human finger, fitted from a capture.

    The finger is modelled as three segments hinging in a plane. Bending
    accumulates along the chain; lateral rotation swings the whole plane about
    the finger's base axis. Segment lengths are taken from the capture's mean
    base-to-tip distance, split 0.4/0.35/0.25 (proximal/middle/distal), which
    is representative of human finger proportions.
    """

    base: np.ndarray  # (3,) finger base position
    tip_mean: np.ndarray  # (3,) mean fingertip position
    length: float
    axis: np.ndarray  # (3,) unit base->tip direction
    side: np.ndarray  # (3,) unit in-plane swing direction
    normal: np.ndarray  # (3,) unit plane normal

    SEGMENT_SPLIT = (0.4, 0.35, 0.25)

    @classmethod
    def fit(cls, tips: np.ndarray, base_hint: np.ndarray | None = None) -> "FingerPlaneModel":
        """Fit from a finger's fingertip samples, shape (T, 3)."""
        tips = np.asarray(tips, dtype=np.float32)
        tip_mean = tips.mean(axis=0)
        if base_hint is None:
            # Use the sample farthest from the mean as a stand-in for the base
            # direction origin; the exact base is not recoverable from tips alone.
            base_hint = tips[np.argmax(np.linalg.norm(tips - tip_mean, axis=1))]
        base = np.asarray(base_hint, dtype=np.float32)
        axis = tip_mean - base
        length = float(np.linalg.norm(axis))
        axis = axis / max(length, 1e-8)
        # Swing direction: principal in-plane spread of the samples about the axis.
        centred = tips - tips.mean(axis=0)
        cov = centred.T @ centred
        vals, vecs = np.linalg.eigh(cov)
        side = vecs[:, -1]
        side = side - axis * float(side @ axis)
        n = np.linalg.norm(side)
        side = side / n if n > 1e-8 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        normal = np.cross(axis, side)
        normal = normal / max(float(np.linalg.norm(normal)), 1e-8)
        return cls(base=base, tip_mean=tip_mean, length=length, axis=axis, side=side, normal=normal)

    def tip_for(self, alpha: float, beta1: float, beta2: float, beta3: float) -> np.ndarray:
        """Fingertip position for the given anchor angles.

        Bend angles rotate within the (axis, side) plane, accumulating outward.
        `alpha` rotates the whole plane about the base axis.
        """
        l1 = self.length * self.SEGMENT_SPLIT[0]
        l2 = self.length * self.SEGMENT_SPLIT[1]
        l3 = self.length * self.SEGMENT_SPLIT[2]

        # Direction of each segment accumulates the joint angles before it.
        # beta3 is the distal (DIP) joint, beta2 the PIP, beta1 the MCP.
        a1 = beta1
        a2 = beta1 + beta2
        a3 = beta1 + beta2 + beta3
        p = self.base.copy()
        for length, ang in ((l1, a1), (l2, a2), (l3, a3)):
            direction = np.cos(ang) * self.axis + np.sin(ang) * self.side
            p = p + length * direction

        if alpha != 0.0:
            # Rodrigues rotation of the final offset about the base axis.
            off = p - self.base
            c, s = float(np.cos(alpha)), float(np.sin(alpha))
            k = self.axis
            off = off * c + np.cross(k, off) * s + k * float(k @ off) * (1 - c)
            p = self.base + off
        return p.astype(np.float32)


# ------------------------------------------------------------------ reference


def _match_reference_frames(
    tips: np.ndarray, model: FingerPlaneModel, params: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the real frame closest to each target anchor.

    Distance is measured in a rotation-invariant way: for each candidate target
    pose we synthesize the ideal tip and select the captured frame whose tip is
    nearest. Returns (selected_tips (K, 3), residual (K,)) so the caller can
    decide whether the capture actually covers the requested range.
    """
    ideal = np.stack([model.tip_for(*p) for p in params])  # (K, 3)
    d = np.linalg.norm(ideal[:, None, :] - tips[None, :, :], axis=-1)  # (K, T)
    idx = d.argmin(axis=1)
    return tips[idx], d.min(axis=1)


def build_human_anchors(
    positions: np.ndarray,
    finger_names: tuple[str, ...],
    params: np.ndarray,
    strategy: str = "synthetic",
    coverage_tol: float = 0.25,
) -> tuple[np.ndarray, str]:
    """Build (F, K, 3) human anchors.

    Args:
        positions: (T, F, 3) raw human fingertips from the capture.
        params: (K, 4) anchor angles.
        strategy: 'synthetic' or 'reference'.
        coverage_tol: for 'reference', if the mean matching residual exceeds
            this fraction of the finger's length, the capture is judged not to
            cover the anchor range and we fall back to 'synthetic'. The value
            that is actually used is returned so it can be reported.

    Returns:
        (anchors (F, K, 3), strategy_actually_used)
    """
    if strategy not in {"synthetic", "reference"}:
        raise ValueError(f"unknown strategy {strategy!r}")
    positions = np.asarray(positions, dtype=np.float32)
    T, F, _ = positions.shape
    K = len(params)

    out = np.empty((F, K, 3), dtype=np.float32)
    used = strategy
    for i in range(F):
        model = FingerPlaneModel.fit(positions[:, i])
        synth = np.stack([model.tip_for(*p) for p in params])
        if strategy == "reference":
            matched, residual = _match_reference_frames(positions[:, i], model, params)
            if float(residual.mean()) > coverage_tol * model.length:
                synth = synth  # keep synthetic; capture does not cover the range
                out[i] = synth
                used = "synthetic"
            else:
                out[i] = matched
        else:
            out[i] = synth
    return out, used


# --------------------------------------------------------------- robot anchors


def anchor_fractions(params: np.ndarray, types: list[str]) -> np.ndarray:
    """Position of each anchor along its own type's sweep, in [0, 1].

    Anchors are paired human-to-robot BY THIS FRACTION: the j-th lateral anchor
    is the j-th step of the lateral sweep on both sides. The paper's P1-P3
    sample both sides "uniformly ... according to the constraint of each anchor
    type", so equal fractions are what correspond.
    """
    s = np.empty(len(types), dtype=np.float32)
    for kind in ("lateral", "bending"):
        idx = [j for j, t in enumerate(types) if t == kind]
        if idx:
            s[idx] = np.linspace(0.0, 1.0, len(idx), dtype=np.float32)
    return s


def finger_joint_roles(kin: HandKinematics) -> list[tuple[int, list[int]]]:
    """For each finger: (abduction drive, [flexion drives in chain order]).

    Derived from the joint axes in the palm frame instead of a hand-written
    table: the drive whose axis is most aligned with the palm normal (z) is the
    lateral one, the rest are the flexion chain. Indices are within the
    finger's own slice. On Wuji and L20 every positive abduction moves the
    finger toward the thumb (+y), so the lateral sweep has a common direction.
    """
    import mujoco

    _, rot = kin.palm_frame()
    roles = []
    for finger in kin.hand.fingers:
        z_align = []
        for drive in finger.joints:
            # A coupled drive's axis is that of its first (master) joint.
            joint = next(iter(kin.hand.drive_targets(drive)))
            jid = mujoco.mj_name2id(kin.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            body = kin.model.jnt_bodyid[jid]
            axis = kin.data.xmat[body].reshape(3, 3) @ kin.model.jnt_axis[jid]
            z_align.append(abs(float((rot.T @ axis)[2])))
        abd = int(np.argmax(z_align))
        flex = [j for j in range(len(finger.joints)) if j != abd]
        roles.append((abd, flex))
    return roles


def build_robot_anchors(
    kin: HandKinematics,
    params: np.ndarray,
    types: list[str],
    lam: float = LAMBDA,
    frame: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """(P3): robot anchors under the same constraint as the human ones.

    Lateral: sweep the abduction drive across its full range, flexion at rest.
    Bending: abduction at rest, flexion chain set to (b1, b1, b1/lambda) with b1
    rising 0 -> pi/2, each clipped to its limits; a two-drive chain (L20, whose
    DIP is slaved to PIP) gets (b1, b1). The thumb, for which the paper says
    bending is "hard to precisely define", sweeps its whole flexion chain
    linearly from rest to the upper limits -- the "pre-generated bending
    trajectory of the robotic thumb" that Appendix D has operators imitate.

    An earlier version sampled K configurations uniformly at random,
    independent of the anchor parameters, which paired the j-th human anchor
    with an unrelated robot pose. That made L_align fit noise.

    Returns (F, K, 3), in `frame` coordinates if given, else world.
    """
    lo, hi = kin.limits()
    rest = kin.rest_qpos()
    s = anchor_fractions(params, types)
    roles = finger_joint_roles(kin)
    F, K = kin.n_fingers, len(types)

    out = np.empty((F, K, 3), dtype=np.float32)
    for i, finger in enumerate(kin.hand.fingers):
        sl = kin.hand.joint_slice(finger.name)
        f_lo, f_hi, f_rest = lo[sl], hi[sl], rest[sl]
        abd, flex = roles[i]
        qpos = np.tile(rest, (K, 1))
        for j in range(K):
            q = f_rest.copy()
            if types[j] == "lateral":
                q[abd] = f_lo[abd] + s[j] * (f_hi[abd] - f_lo[abd])
            elif finger.name == "thumb":
                q[flex] = f_rest[flex] + s[j] * (f_hi[flex] - f_rest[flex])
            else:
                b1 = s[j] * np.pi / 2
                pattern = [b1, b1, b1 / lam][: len(flex)]
                q[flex] = pattern
            qpos[j, sl] = np.clip(q, f_lo, f_hi)
        tips = kin.fingertip_positions_batch(qpos)[:, i]
        if frame is not None:
            tips = (tips - frame[0]) @ frame[1]
        out[i] = tips
    return out


# ------------------------------------------------------- human anchors (skeleton)

# MediaPipe 21-landmark chains: (base, joint1, joint2, tip) per finger.
MEDIAPIPE_CHAINS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


def _angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    cos = (a * b).sum(-1) / np.maximum(na * nb, 1e-12)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def skeleton_angles(landmarks: np.ndarray, finger: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (alpha, beta (T, 3)) for one finger from palm-frame landmarks.

    beta_k are the unsigned flexion angles at the MCP, PIP and DIP (between
    consecutive bone vectors, the metacarpal being wrist -> MCP). alpha is the
    signed in-palm-plane angle of the proximal phalanx relative to the
    metacarpal, positive toward +y (the thumb side), matching the robots'
    abduction sign. For the thumb the chain is CMC -> MCP -> IP -> tip and alpha
    is the in-plane heading of CMC -> MCP.
    """
    L = np.asarray(landmarks, dtype=np.float64)
    c0, c1, c2, c3 = MEDIAPIPE_CHAINS[finger]
    meta = L[:, c0] - L[:, 0]
    s1, s2, s3 = L[:, c1] - L[:, c0], L[:, c2] - L[:, c1], L[:, c3] - L[:, c2]
    beta = np.stack([_angle(meta, s1), _angle(s1, s2), _angle(s2, s3)], axis=-1)
    if finger == "thumb":
        alpha = np.arctan2(s1[:, 1], s1[:, 0])
    else:
        cross = meta[:, 0] * s1[:, 1] - meta[:, 1] * s1[:, 0]
        dot = meta[:, 0] * s1[:, 0] + meta[:, 1] * s1[:, 1]
        alpha = np.arctan2(cross, dot)
    return alpha, beta


def build_human_anchors_skeleton(
    landmarks: np.ndarray,
    finger_names: tuple[str, ...],
    params: np.ndarray,
    types: list[str],
    lam: float = LAMBDA,
    flat_quantile: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """(P1)+(P2) from a recording: pick the real frame that best meets each anchor.

    This stands in for the live session -- the operator imitating reference
    poses -- by searching a free-motion recording for the frames whose measured
    skeleton angles best satisfy each anchor's constraint:

      lateral  among the flattest `flat_quantile` of frames (smallest total
               flexion), the frame whose alpha is nearest the target; targets
               span the 2nd-98th percentile of alpha in that pool.
      bending  the frame minimizing ||beta - (b1, b1, b1/lambda)|| + |alpha - median|.
      thumb bending is matched ORDINALLY (by quantile of total curl), since the
               paper itself does not parameterize thumb bending by angle.

    Returns (anchors (F, K, 3), residual_deg (F, K)): the angular distance
    between the chosen frame and its target, so coverage is measured, not
    assumed. `flat_quantile` and the percentile span are choices of this
    reproduction.
    """
    L = np.asarray(landmarks, dtype=np.float32)
    s = anchor_fractions(params, types)
    K = len(types)
    tip_idx = {f: MEDIAPIPE_CHAINS[f][3] for f in MEDIAPIPE_CHAINS}
    out = np.empty((len(finger_names), K, 3), dtype=np.float32)
    resid = np.zeros((len(finger_names), K), dtype=np.float32)

    for i, finger in enumerate(finger_names):
        alpha, beta = skeleton_angles(L, finger)
        curl = beta.sum(-1)
        flat = np.where(curl <= np.quantile(curl, flat_quantile))[0]
        a_lo, a_hi = np.percentile(alpha[flat], [2, 98])
        a_med = float(np.median(alpha))
        order = np.argsort(curl)
        for j in range(K):
            if types[j] == "lateral":
                target = a_lo + s[j] * (a_hi - a_lo)
                d = np.abs(alpha[flat] - target)
                k = int(flat[np.argmin(d)])
                resid[i, j] = np.degrees(d.min())
            elif finger == "thumb":
                lo_i, hi_i = int(0.02 * len(order)), int(0.98 * (len(order) - 1))
                k = int(order[lo_i + int(round(s[j] * (hi_i - lo_i)))])
                resid[i, j] = 0.0
            else:
                b1 = s[j] * np.pi / 2
                target = np.array([b1, b1, b1 / lam])
                cost = np.linalg.norm(beta - target, axis=-1) + np.abs(alpha - a_med)
                k = int(np.argmin(cost))
                resid[i, j] = np.degrees(np.linalg.norm(beta[k] - target))
            out[i, j] = L[k, tip_idx[finger]]
    return out, resid


# ------------------------------------------------------------------- assemble


def build_anchor_set(
    hand: HandSpec | str,
    human_positions: np.ndarray,
    human_normalizer: HandNormalizer,
    kin: HandKinematics,
    robot_normalizer: HandNormalizer | None = None,
    strategy: str = "synthetic",
    n_lateral: int = K_LATERAL,
    n_bending: int = K_BENDING,
    seed: int = 0,
    landmarks: np.ndarray | None = None,
    robot_frame: tuple[np.ndarray, np.ndarray] | None = None,
) -> AnchorSet:
    """Assemble the full paired anchor set for one hand.

    Human anchors are normalized with the human normalizer and robot anchors
    with the robot normalizer. With per-finger self-normalization on both sides
    (Appendix B) these differ; the robot normalizer falls back to the human one
    only for the older shared-normalizer setup.

    strategy 'skeleton' requires `landmarks` (T, 21, 3) in the palm frame and
    uses `build_human_anchors_skeleton`; 'synthetic'/'reference' use the
    fingertip-only plane model. `robot_frame` expresses robot anchors in the same
    frame as the robot cloud (e.g. `clouds.frame_origin, clouds.frame_rot`).
    """
    hand = get_hand(hand) if isinstance(hand, str) else hand
    robot_normalizer = robot_normalizer or human_normalizer

    p_lat = lateral_params(n_lateral)
    p_bend = bending_params(n_bending)
    params = np.concatenate([p_lat, p_bend], axis=0)
    types = ["lateral"] * len(p_lat) + ["bending"] * len(p_bend)

    residual = None
    if strategy == "skeleton":
        if landmarks is None:
            raise ValueError("strategy 'skeleton' needs the (T, 21, 3) landmarks")
        human_raw, residual = build_human_anchors_skeleton(
            landmarks, hand.finger_names, params, types
        )
        used = "skeleton"
    else:
        human_raw, used = build_human_anchors(
            human_positions, hand.finger_names, params, strategy=strategy
        )
    robot_raw = build_robot_anchors(kin, params, types, frame=robot_frame)
    # Both builders return (F, K, 3), which is what HandNormalizer wants.
    human_norm = human_normalizer.transform(human_raw).astype(np.float32)
    robot_norm = robot_normalizer.transform(robot_raw).astype(np.float32)

    return AnchorSet(
        human=human_norm,
        robot=robot_norm,
        params=params,
        types=types,
        strategy=used,
        finger_names=hand.finger_names,
        residual_deg=residual,
    )
