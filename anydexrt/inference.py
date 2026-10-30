"""Inference: f_m -> f_s (NNS) -> optional pinch refinement with f_c.

Per the paper's Inference paragraph:

  "The retargeted fingertip positions predicted by f_m need to be transformed to
   joint configurations. We simply use nearest neighbor search (NNS) to get the
   corresponding joint configurations from the given fingertip positions and the
   last joint configurations."

  "We use a threshold of 0.5 for the scores predicted by f_c. For the predicted
   contact pattern of two fingers, we simply search the nearest fingertip
   positions from pre-generated templates using the corresponding positions
   predicted by f_m."

Spaces involved, so they do not get mixed up:

  human palm frame (m) --human normalizer--> human normalized
        --f_m--> robot normalized --robot normalizer^-1--> robot cloud frame (m)

`f_s` runs in robot-normalized space against the normalized cloud. Pinch
templates live in the robot cloud frame in metres: with per-finger
normalization each finger has its own centre and scale, so a thumb-to-finger
distance is only physical in metres.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

from .data import RobotClouds
from .kinematics import HandKinematics, NearestNeighborIK
from .models import ContactClassifier, CrossFingerMapper, FingerMapper
from .normalize import HandNormalizer

_CALIBRATION_FILE = Path(__file__).resolve().parents[1] / "calibrated_pinch_poses.yaml"
_FIST_CALIBRATION_FILE = Path(__file__).resolve().parents[1] / "calibrated_fist_pose.yaml"


@dataclass
class PinchTemplates:
    """Pre-generated robot pinch poses for one (thumb, finger i) pair.

    `tips` is (N, 6): thumb tip then finger tip, robot cloud frame, metres.
    `thumb_q` / `finger_q` are the drive values for those two fingers.

    `calibrated_thumb_q` / `calibrated_finger_q` are an optional extra pose,
    hand-tuned in the MuJoCo viewer (scripts/calibrate_pinch.py) and verified
    visually -- unlike the sampled rows above, it is known to look like a
    proper pinch. It is NOT part of the NNS tree (its fingertip positions are
    the ones the robot actually reaches, so it would not necessarily be the
    nearest neighbour of what f_m predicts); `refine_pinch*` uses it as the
    look-alike target once the pinch is fully engaged.
    """

    finger: int
    tips: np.ndarray
    thumb_q: np.ndarray
    finger_q: np.ndarray
    gap: np.ndarray  # (N,) thumb-to-finger distance, metres
    calibrated_thumb_q: np.ndarray | None = None
    calibrated_finger_q: np.ndarray | None = None
    calibrated_thumb_tip: np.ndarray | None = None
    calibrated_finger_tip: np.ndarray | None = None

    def __post_init__(self):
        self.tree = cKDTree(self.tips)

    def nearest(self, thumb: np.ndarray, finger: np.ndarray) -> int:
        _, idx = self.tree.query(np.concatenate([thumb, finger]))
        return int(idx)


def make_pinch_templates(
    clouds: RobotClouds,
    kin: HandKinematics,
    per_finger: int = 400,
    pool_factor: int = 5,
) -> dict[int, PinchTemplates]:
    """Pre-generate pinch poses by pairing the thumb table with each finger table.

    For every finger, start from a closest-gap pool (nearest finger-i cloud
    row per thumb cloud row, by raw fingertip distance -- same pool as the
    plain baseline). If scripts/calibrate_pinch.py has written a hand-tuned,
    visually-verified pose for this finger (see calibrated_pinch_poses.yaml),
    that pose becomes row 0 with its own exact FK tip positions, and the rest
    of the table is the pool rows CLOSEST TO IT IN (normalized) JOINT SPACE --
    not closest-gap, not direction-aligned-by-dot-product (both tried and
    rejected: gap alone and gap+alignment both looked wrong to the eye). This
    keeps the verified pose as the answer at full pinch while still giving
    refine_pinch_continuous nearby rows to interpolate through as the human
    hand's position varies. Fingers without a calibrated pose fall back to
    plain closest-gap, matching behaviour before calibration existed.
    """
    tips = clouds.tips  # (N, F, 3), cloud frame, metres
    thumb_sl = kin.hand.joint_slice(kin.hand.fingers[0].name)
    lo, hi = kin.limits()
    joint_range = np.maximum(hi - lo, 1e-9)

    calibration: dict = {}
    if _CALIBRATION_FILE.exists():
        with open(_CALIBRATION_FILE) as f:
            calibration = yaml.safe_load(f) or {}

    out: dict[int, PinchTemplates] = {}
    for i in range(1, kin.n_fingers):
        sl = kin.hand.joint_slice(kin.hand.fingers[i].name)
        finger_name = kin.hand.fingers[i].name

        dist, idx = cKDTree(tips[:, i]).query(tips[:, 0])
        pool_size = min(len(dist), per_finger * pool_factor)
        pool = np.argsort(dist)[:pool_size]
        a_pool, b_pool = pool, idx[pool]

        calib = calibration.get(finger_name)
        if calib is not None and (
            len(calib["thumb_q"]) != (thumb_sl.stop - thumb_sl.start)
            or len(calib["finger_q"]) != (sl.stop - sl.start)
        ):
            # Calibration was recorded for a different hand's joint layout
            # (e.g. linker_l20's 4-DoF thumb vs wuji_hand's); joint counts
            # don't line up, so it cannot apply here -- fall back below.
            calib = None
        if calib is None:
            a, b = a_pool[:per_finger], b_pool[:per_finger]
            out[i] = PinchTemplates(
                finger=i,
                tips=np.concatenate([tips[a, 0], tips[b, i]], axis=1).astype(np.float32),
                thumb_q=clouds.qpos[a][:, thumb_sl].copy(),
                finger_q=clouds.qpos[b][:, sl].copy(),
                gap=dist[a].astype(np.float32),
            )
            continue

        cal_thumb_q = np.asarray(calib["thumb_q"], dtype=np.float32)
        cal_finger_q = np.asarray(calib["finger_q"], dtype=np.float32)

        pool_thumb_q = clouds.qpos[a_pool][:, thumb_sl]
        pool_finger_q = clouds.qpos[b_pool][:, sl]
        d_thumb = (pool_thumb_q - cal_thumb_q) / joint_range[thumb_sl]
        d_finger = (pool_finger_q - cal_finger_q) / joint_range[sl]
        joint_dist = np.linalg.norm(np.concatenate([d_thumb, d_finger], axis=1), axis=1)
        nearest = np.argsort(joint_dist)[: per_finger - 1]
        a, b = a_pool[nearest], b_pool[nearest]

        qpos_full = np.zeros(kin.n_joints, dtype=np.float32)
        qpos_full[thumb_sl] = cal_thumb_q
        qpos_full[sl] = cal_finger_q
        kin.set_qpos(qpos_full)
        cal_tips = kin._current_tips()
        cal_thumb_tip, cal_finger_tip = cal_tips[0].astype(np.float32), cal_tips[i].astype(np.float32)
        cal_gap = float(np.linalg.norm(cal_thumb_tip - cal_finger_tip))

        out[i] = PinchTemplates(
            finger=i,
            tips=np.concatenate(
                [
                    np.concatenate([cal_thumb_tip, cal_finger_tip])[None],
                    np.concatenate([tips[a, 0], tips[b, i]], axis=1),
                ],
                axis=0,
            ).astype(np.float32),
            thumb_q=np.concatenate([cal_thumb_q[None], clouds.qpos[a][:, thumb_sl]], axis=0),
            finger_q=np.concatenate([cal_finger_q[None], clouds.qpos[b][:, sl]], axis=0),
            gap=np.concatenate([[cal_gap], dist[a]]).astype(np.float32),
            calibrated_thumb_q=cal_thumb_q,
            calibrated_finger_q=cal_finger_q,
            calibrated_thumb_tip=cal_thumb_tip,
            calibrated_finger_tip=cal_finger_tip,
        )
    return out


@dataclass
class JointLowPassFilter:
    """First-order low-pass (EMA) over a joint-space vector, variable dt.

    y[n] = y[n-1] + alpha * (x[n] - y[n-1]),  alpha = dt / (tau + dt),  tau = 1/(2*pi*fc)

    This is the standard fix for two distinct sources of jitter in this
    pipeline: NNS "hunting" between nearby table entries as a continuously
    moving target crosses a Voronoi boundary, and the discrete jump `refine_pinch`
    makes when it swaps in a stored template. Both become gradual instead of
    instantaneous.

    `apply` with no `dt` measures wall-clock time between calls, which is only
    physically meaningful for callers running in real time (a live/paced
    teleoperation loop). A tight offline batch loop would measure near-zero dt
    and over-smooth, so batch callers (`retarget_sequence`) do not use this.
    """

    cutoff_hz: float
    _y: np.ndarray | None = None
    _t: float | None = None

    def reset(self) -> None:
        self._y = None
        self._t = None

    def apply(self, x: np.ndarray, dt: float | None = None) -> np.ndarray:
        now = time.perf_counter()
        if dt is None:
            dt = 0.0 if self._t is None else now - self._t
        self._t = now
        if self._y is None:
            self._y = np.asarray(x, dtype=np.float32).copy()
            return self._y
        tau = 1.0 / (2.0 * np.pi * self.cutoff_hz)
        alpha = dt / (tau + dt) if (tau + dt) > 0 else 1.0
        self._y = self._y + alpha * (np.asarray(x, dtype=np.float32) - self._y)
        return self._y


class AnyDexRT:
    """End-to-end retargeter: palm-frame human fingertips in, robot drives out."""

    def __init__(
        self,
        mapper: FingerMapper,
        clouds: RobotClouds,
        kin: HandKinematics,
        human_normalizer: HandNormalizer,
        classifier: ContactClassifier | None = None,
        pinch_templates: dict[int, PinchTemplates] | None = None,
        device: str | torch.device = "cpu",
        use_contacts: bool = True,
        contact_threshold: float = 0.5,
        identity_mapper: bool = False,
        smoothing_hz: float | None = None,
        pinch_mode: str = "classifier",
        pinch_open_gap_mm: float | list[float] = 80.0,
        pinch_closed_gap_mm: float | list[float] = 6.0,
        pinch_min_extension_mm: float = 110.0,
        fist_thumb_rest_mm: float = 110.0,
        fist_thumb_extended_mm: float = 125.0,
        fist_thumb_bent: float = 0.92,
        fist_thumb_straight: float = 0.95,
        pinch_thumb_combine: int = 2,
        pinch_thumb_sharpness: float = 3.0,
        fist_open_reach_mm: float = 110.0,
        fist_curled_reach_mm: float = 80.0,
    ):
        """`identity_mapper=True` skips f_m (normalized human = normalized robot).

        That is the natural ablation baseline: per-finger normalization alone,
        i.e. align each finger's workspace by centre and scale and do nothing
        else. Whatever f_m adds is measured against it.

        `smoothing_hz` applies a joint-space low-pass to the output of
        `retarget()` (see `JointLowPassFilter`). Off by default: it uses
        wall-clock time between calls, which only makes sense for a caller
        pacing itself to real time (teleoperation), not for the tight batch loop
        in `retarget_sequence` -- enabling it there would over-smooth, since
        measured dt would be near zero. `scripts/teleop_avp.py` turns it on.

        `pinch_mode`:
            'classifier' (default) -- f_c decides contact on/off, refine_pinch
                does a hard pose swap. Needs a trained, working classifier;
                see README for the ring/pinky failure mode.
            'continuous' -- no classifier. `refine_pinch_continuous` blends
                toward the pinch template in proportion to the real thumb-finger
                distance (see that method). `classifier` can be None here.
            'off' -- f_m's raw output only, no pinch assist at all.
        `pinch_open_gap_mm`: the thumb-finger distance (mm) treated as alpha=0
            ("fully open") for 'continuous' mode, per finger (index, middle,
            ring, pinky order) or a single float applied to all four. Pick this
            from the operator's own comfortable open-hand spread; it does not
            need to be precise -- alpha is clamped to [0, 1] either side.
        `pinch_closed_gap_mm`: the thumb-finger distance (mm) treated as
            alpha=1 ("fully pinched"). Real AVP tracking never reports 0mm
            even when the operator pinches as hard as they can -- the two
            fingertip meshes have physical thickness and tracking noise, so
            the measured gap bottoms out around 4-6mm (verified against
            recorded sessions). Requiring gap==0 for alpha=1 meant the blend
            was always a few percent short of the calibrated pose, which
            compounds through the joint-space blend + FK into a visibly
            "not fully pinched" result. Defaults to 6mm; same per-finger-or-
            single-float shape as pinch_open_gap_mm.
        `pinch_min_extension_mm`: a finger must reach at least this far from
            the palm origin before it is even considered for the pinch blend.
            Without this, a closed fist can accidentally trigger it: a curled
            finger's tip can end up only 1-8mm from the thumb (verified on
            recordings) even though it is curled INTO the palm, not reaching
            OUT toward the thumb -- gap alone can't tell those apart. Palm-
            distance can: curled fingers measured ~60-85mm from the palm
            origin, fingers genuinely reaching toward a pinch ~150-185mm.
            Defaults to 110mm, the midpoint of that gap in recorded sessions.
        """
        self.device = torch.device(device)
        self.mapper = mapper.to(self.device).eval()
        self.classifier = None if classifier is None else classifier.to(self.device).eval()
        self.human_normalizer = human_normalizer
        self.clouds = clouds
        self.kin = kin
        self.solver = NearestNeighborIK(kin, clouds.qpos, clouds.tips_norm)
        self.pinch_templates = pinch_templates or {}
        self.use_contacts = use_contacts
        self.contact_threshold = contact_threshold
        self.identity_mapper = identity_mapper
        self.smoother = JointLowPassFilter(smoothing_hz) if smoothing_hz else None
        if pinch_mode not in {"classifier", "continuous", "off"}:
            raise ValueError(f"pinch_mode must be 'classifier', 'continuous', or 'off', got {pinch_mode!r}")
        self.pinch_mode = pinch_mode
        n_other = mapper.n_fingers - 1

        def _per_finger(value, name):
            if isinstance(value, (int, float)):
                return np.full(n_other, value / 1000.0, dtype=np.float32)
            arr = np.asarray(value, dtype=np.float32) / 1000.0
            if arr.shape != (n_other,):
                raise ValueError(f"{name} must have {n_other} entries, got {arr.shape}")
            return arr

        self.pinch_open_gap = _per_finger(pinch_open_gap_mm, "pinch_open_gap_mm")
        self.pinch_closed_gap = _per_finger(pinch_closed_gap_mm, "pinch_closed_gap_mm")
        self.pinch_min_extension = pinch_min_extension_mm / 1000.0
        self.fist_thumb_rest = fist_thumb_rest_mm / 1000.0
        self.fist_thumb_extended = fist_thumb_extended_mm / 1000.0
        self.fist_thumb_bent = fist_thumb_bent
        self.fist_thumb_straight = fist_thumb_straight
        self.pinch_thumb_combine = max(1, int(pinch_thumb_combine))
        self.pinch_thumb_sharpness = float(pinch_thumb_sharpness)
        self.last_pinch: dict | None = None
        self.fist_open_reach = fist_open_reach_mm / 1000.0
        self.fist_curled_reach = fist_curled_reach_mm / 1000.0
        self.last_fist: dict | None = None
        self.fist_thumb_q: np.ndarray | None = None
        if _FIST_CALIBRATION_FILE.exists():
            with open(_FIST_CALIBRATION_FILE) as f:
                fist_cal = yaml.safe_load(f) or {}
            fist_q = np.asarray(fist_cal.get("qpos", []), dtype=np.float32)
            thumb_sl = kin.hand.joint_slice(kin.hand.fingers[0].name)
            # Calibrated on one specific hand; a different hand's joint layout
            # makes those numbers meaningless, so only apply on a size match.
            if fist_q.size == kin.n_joints:
                self.fist_thumb_q = fist_q[thumb_sl].copy()
        self.last_qpos: np.ndarray | None = None
        self.last_residual: np.ndarray | None = None
        self.last_contacts: np.ndarray | None = None

    @torch.no_grad()
    def map_fingertips(self, human_tips: np.ndarray) -> np.ndarray:
        """Palm-frame human fingertips (F, 3) -> robot-normalized targets (F, 3)."""
        human_tips = np.asarray(human_tips, dtype=np.float32)
        if human_tips.shape != (self.mapper.n_fingers, 3):
            raise ValueError(
                f"expected ({self.mapper.n_fingers}, 3) fingertips, got {tuple(human_tips.shape)}"
            )
        norm = self.human_normalizer.transform(human_tips)
        if self.identity_mapper:
            return norm
        x = torch.from_numpy(norm).float().to(self.device)
        if isinstance(self.mapper, CrossFingerMapper):
            return self._map_cross_finger_gated(x, human_tips)
        return self.mapper(x[None])[0].cpu().numpy()

    def _map_cross_finger_gated(self, x: torch.Tensor, human_tips: np.ndarray) -> np.ndarray:
        """CrossFingerMapper's thumb net always sees the other fingers'
        positions as context -- useful for a genuine pinch, but misleading
        when a finger is curled into the palm rather than reaching toward the
        thumb (making a fist, say). Training data pairs real recorded hand
        poses, so "thumb extended alone, every other finger fully curled"
        rarely appears; the thumb net has no learned response for that
        combination and extrapolates from what it has seen -- poses where the
        thumb extends roughly in step with the other fingers. That shows up
        as the thumb needing the other fingers to extend before it will.

        Substitute a curled finger's slot in the context with a neutral
        "open" reference (the normalizer's center, i.e. the mean fingertip
        position across training data, typically a naturally open hand)
        instead of the real curled position. That way the thumb net sees
        "other fingers open, thumb at X" -- a combination it actually saw
        during training -- rather than "thumb extended, others curled" (rare
        in paired recordings) or all-zero context (never seen, out of
        distribution). The thumb can then extend independently without being
        held back by the curled fingers' real positions.
        """
        mapper = self.mapper
        thumb_idx = mapper.thumb_idx
        reach = np.linalg.norm(human_tips, axis=-1)
        context_parts = []
        for i in range(mapper.n_fingers):
            if i == thumb_idx:
                continue
            if reach[i] < self.pinch_min_extension:
                # Neutral open position in normalized space (center maps to origin)
                neutral = torch.zeros(1, 3, device=self.device, dtype=x.dtype)
                context_parts.append(neutral)
            else:
                context_parts.append(x[i : i + 1])
        context = torch.cat(context_parts, dim=-1)

        out = torch.empty_like(x)
        for i in range(mapper.n_fingers):
            if i == thumb_idx:
                out[i] = mapper.forward_finger(x[i : i + 1], i, context=context)[0]
            else:
                out[i] = mapper.forward_finger(x[i : i + 1], i)[0]
        return out.cpu().numpy()

    @torch.no_grad()
    def contact_scores(self, human_tips: np.ndarray) -> np.ndarray | None:
        """(F-1,) scores for fingers 1..F-1, or None without a classifier."""
        if self.classifier is None:
            return None
        norm = self.human_normalizer.transform(np.asarray(human_tips, dtype=np.float32))
        x = torch.from_numpy(norm).float().to(self.device)
        return self.classifier(x[None])[0].cpu().numpy()

    def refine_pinch(self, qpos: np.ndarray, mapped_norm: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Overwrite thumb + one finger with the nearest pinch template.

        Only the highest-scoring contact above threshold is applied: the thumb
        can pinch one finger at a time in a template, and applying several in
        sequence would let the last one silently win.
        """
        best = int(np.argmax(scores))
        if scores[best] < self.contact_threshold:
            return qpos
        finger = best + 1
        templates = self.pinch_templates.get(finger)
        if templates is None:
            return qpos
        mapped = self.clouds.normalizer.inverse(mapped_norm)  # (F, 3) cloud frame, metres
        k = templates.nearest(mapped[0], mapped[finger])
        out = qpos.copy()
        out[self.kin.hand.joint_slice(self.kin.hand.fingers[0].name)] = templates.thumb_q[k]
        out[self.kin.hand.joint_slice(self.kin.hand.fingers[finger].name)] = templates.finger_q[k]
        return out

    def refine_pinch_continuous(
        self, qpos: np.ndarray, mapped_norm: np.ndarray, human_tips: np.ndarray
    ) -> np.ndarray:
        """Blend each finger toward its pinch template by real thumb-finger distance.

        Unlike `refine_pinch` (a binary classifier gate -> instantaneous pose
        swap), this computes alpha_i = clamp(1 - human_gap_i / open_gap_i, 0, 1)
        directly from the human geometry, with NO classifier involved -- it
        sidesteps f_c entirely, including its dead ring/pinky heads. alpha=0 at
        the operator's natural open-hand distance, alpha=1 at full pinch, and
        varies continuously in between (so a 1cm/2cm half-pinch produces a
        proportionally half-closed robot pose, not a jump).

        The interpolation is done in JOINT space (qpos), not Cartesian space:
        blending Cartesian targets and then re-solving NNS per blend step would
        risk crossing unreachable configurations; blending the two already-valid
        qpos solutions stays inside the robot's reachable set for any alpha
        (convex combination of two feasible joint vectors is not guaranteed
        feasible in general, but is a good approximation for the small, local
        thumb/finger motions pinch poses involve -- see README caveat).

        Only the DOMINANT finger (largest alpha) blends the thumb. An earlier
        version blended the thumb slice once per finger with alpha > 0, so a
        real index pinch (finger 1 at alpha~1) got its thumb target overwritten
        by middle/ring/pinky's much weaker partial pinches (alpha~0.3-0.5,
        since their gap is still well under their own open-hand reference) --
        measured: thumb-index gap stuck at 55.9mm regardless of how close the
        real fingers got, because the thumb target was an unintended blend of
        several fingers' templates, none of which matched where the thumb
        itself needed to be. Non-dominant fingers still blend their OWN slice
        independently (their finger doesn't care what the thumb does), just not
        the shared thumb slice.

        Args:
            human_tips: (F, 3) RAW (un-normalized) palm-frame human fingertips,
                thumb first -- needed to compute real-world distances in metres,
                which normalized/mapped coordinates do not preserve directly.
        """
        thumb = human_tips[0]
        mapped = self.clouds.normalizer.inverse(mapped_norm)
        out = qpos.copy()

        alphas, nearest_k = {}, {}
        for finger in range(1, self.mapper.n_fingers):
            templates = self.pinch_templates.get(finger)
            if templates is None:
                continue
            finger_reach = float(np.linalg.norm(human_tips[finger]))
            if finger_reach < self.pinch_min_extension:
                # A curled finger (fist) can land thumb-close by accident --
                # measured on real recordings: thumb-finger gap during a fist
                # drops to 1-8mm, well inside the pinch band, even though the
                # finger is curled INTO the palm, not reaching OUT to meet the
                # thumb. Distance from the palm origin reliably separates the
                # two (curled ~60-85mm vs. extended-toward-thumb ~150-185mm on
                # recorded sessions), so gate on it before even considering
                # gap; a curled finger never engages the pinch blend.
                alphas[finger] = 0.0
                continue
            gap = float(np.linalg.norm(human_tips[finger] - thumb))
            open_gap = self.pinch_open_gap[finger - 1]
            closed_gap = self.pinch_closed_gap[finger - 1]
            # alpha=1 at closed_gap (realistic "fully pinched" reading, not
            # exactly 0 -- see pinch_closed_gap_mm docstring), alpha=0 at
            # open_gap, clamped outside that range.
            span = max(open_gap - closed_gap, 1e-6)
            alphas[finger] = float(np.clip((open_gap - gap) / span, 0.0, 1.0))
            if templates.calibrated_thumb_q is not None:
                # Row 0 is the hand-verified calibrated pose (see
                # make_pinch_templates); use it directly rather than a
                # nearest-neighbour search on the network's raw mapped tip
                # position. That search is blind to alpha -- it would often
                # pick one of the nearby-in-joint-space rows instead, whose
                # tip position (not joint values) can be far from the
                # calibrated pose, so alpha=1 would blend toward the wrong
                # target instead of the one actually verified in the
                # simulator.
                nearest_k[finger] = 0
            else:
                nearest_k[finger] = templates.nearest(mapped[0], mapped[finger])

        active = {f: a for f, a in alphas.items() if a > 0.0}
        self.last_pinch = None
        if not active:
            return out
        # Every finger blends its OWN slice with its own alpha: a finger does
        # not care what the thumb is doing.
        for finger, alpha in active.items():
            templates = self.pinch_templates[finger]
            finger_sl = self.kin.hand.joint_slice(self.kin.hand.fingers[finger].name)
            out[finger_sl] = (
                (1 - alpha) * out[finger_sl] + alpha * templates.finger_q[nearest_k[finger]]
            )

        # The thumb is shared, so it gets one blend built from the strongest
        # `pinch_thumb_combine` fingers rather than from the single dominant
        # one. Moving between two pinches, the human thumb is briefly far from
        # both fingers, so both alphas sag (measured: the max drops to ~0.43
        # mid-transition) and a max-weighted blend hands most of the weight
        # back to f_m's raw thumb pose -- whose roll sits at 0.057 rad against
        # the calibrated pinch poses' 0.60-1.19, i.e. almost fully abducted.
        # The thumb therefore swung outward between every pair of fingers
        # (measured over 45 transitions: it left the two calibrated values'
        # range every single time, by 17 deg on average and up to 24 deg).
        #
        # Two parts fix it. The TARGET is an alpha^p weighted average of the
        # calibrated thumb poses, so a transition interpolates along the
        # calibrated arc instead of detouring through f_m's pose. The WEIGHT is
        # the sum of those alphas, clamped, so two half-engaged fingers keep
        # the thumb on the arc rather than releasing it.
        #
        # This is not the earlier defect the docstring above describes (the
        # thumb slice being overwritten once per active finger): that applied
        # each finger's alpha in sequence, so a weak partial pinch could undo
        # a strong one. Here there is a single blend, and p=3 keeps a genuine
        # single-finger pinch dominant -- alphas of 1.0 and 0.4 give the
        # leader 94% of the target.
        ranked = sorted(active.items(), key=lambda kv: -kv[1])[: self.pinch_thumb_combine]
        weights = np.array([a ** self.pinch_thumb_sharpness for _, a in ranked])
        total = float(weights.sum())
        thumb_sl = self.kin.hand.joint_slice(self.kin.hand.fingers[0].name)
        if total <= 0.0:
            return out
        weights = weights / total
        target = np.zeros(thumb_sl.stop - thumb_sl.start, dtype=out.dtype)
        for (finger, _), w in zip(ranked, weights):
            target += w * self.pinch_templates[finger].thumb_q[nearest_k[finger]]
        thumb_alpha = float(min(sum(a for _, a in ranked), 1.0))
        out[thumb_sl] = (1 - thumb_alpha) * out[thumb_sl] + thumb_alpha * target
        self.last_pinch = {
            "alphas": dict(alphas),
            "thumb_alpha": thumb_alpha,
            "leaders": [self.kin.hand.finger_names[f] for f, _ in ranked],
        }
        return out

    def refine_fist(
        self, qpos: np.ndarray, human_tips: np.ndarray,
        straightness: float | None = None,
    ) -> np.ndarray:
        """Drive the thumb toward a calibrated natural-fist pose when the human
        is making a fist, blended by how curled the other fingers are.

        `_map_cross_finger_gated` zeroes a curled finger's slot in the thumb
        net's context, which fixes the thumb being dragged by the other
        fingers -- but zeros are an input the net never saw in training
        (context was always real positions), so its response there is
        undefined, and in practice the thumb lands too far out on a fist.
        Rather than bet on an out-of-distribution guess, aim at a pose
        calibrated by hand in the viewer (scripts/calibrate_fist.py).

        A fist and "thumb extended, other fingers curled" both have all four
        fingers curled, so curl alone can't separate them. Neither can the
        thumb-to-finger gap: with the thumb extended it is still only 50-82mm
        from the nearest curled fingertip, inside the fist range (29-81mm).

        What does separate them, on every device tested, is how STRAIGHT the
        thumb is (`data.thumb_straightness`): on both AVP and PICO 4 the curled
        and extended frames form two peaks with the valley at 0.90, which is
        where `fist_thumb_bent` sits -- the extended peak only starts at 0.94,
        so `fist_thumb_straight` stays at 0.95 to keep a genuinely extended
        thumb at zero weight. A distance in mm does not transfer -- it splits
        AVP's fist (95-106mm) from its extended thumb (119-142mm), but PICO 4
        compresses the same two poses into 90-122mm with no gap between them,
        because its OpenXR thumb root sits further out and the thumb measures
        25% shorter. `straightness` is passed down from the live entry points;
        when it is absent (bare `retarget` on palm-frame tips, which carry no
        thumb MCP/IP) this falls back to the reach threshold, which is correct
        for AVP-scale input only.

        Both weights blend rather than snap, so the thumb eases into / out of
        the fist pose instead of jumping when a threshold trips.
        """
        self.last_fist = None
        if self.fist_thumb_q is None or self.fist_thumb_extended <= 0.0:
            return qpos
        others = human_tips[1:]
        if others.shape[0] == 0:
            return qpos
        reach = np.linalg.norm(others, axis=-1)
        thumb_reach = float(np.linalg.norm(human_tips[0]))
        # alpha_curl=1 once every finger is as curled as a real fist gets, 0
        # once one is open. A curled fingertip still sits ~73-94mm from the palm
        # origin (fingers have length), so normalizing against 0 would cap this
        # near 0.3. Deliberately NOT normalized against pinch_min_extension,
        # even though that is also a curled/open boundary: that one has to drop
        # to ~90mm on PICO 4 (its fingers reach 20-35mm less far than AVP's, so
        # 110 blocks the ring/pinky pinch entirely), which would squeeze this
        # span from 30mm to 10mm and make a fist lose half its weight over 5mm
        # of fingertip wobble. Separate knob, same default as AVP's gate.
        span = max(self.fist_open_reach - self.fist_curled_reach, 1e-6)
        alpha_curl = float(np.clip((self.fist_open_reach - reach.max()) / span, 0.0, 1.0))
        if straightness is None:
            thumb_span = max(self.fist_thumb_extended - self.fist_thumb_rest, 1e-6)
            alpha_thumb = float(
                np.clip((self.fist_thumb_extended - thumb_reach) / thumb_span, 0.0, 1.0)
            )
        else:
            bend_span = max(self.fist_thumb_straight - self.fist_thumb_bent, 1e-6)
            alpha_thumb = float(
                np.clip((self.fist_thumb_straight - straightness) / bend_span, 0.0, 1.0)
            )
        alpha = alpha_curl * alpha_thumb
        self.last_fist = {
            "finger_reach_max_mm": float(reach.max()) * 1000,
            "thumb_reach_mm": thumb_reach * 1000,
            "straightness": straightness,
            "alpha": alpha,
        }
        if alpha <= 0.0:
            return qpos
        out = qpos.copy()
        thumb_sl = self.kin.hand.joint_slice(self.kin.hand.fingers[0].name)
        out[thumb_sl] = (1 - alpha) * out[thumb_sl] + alpha * self.fist_thumb_q
        return out

    def retarget(
        self, human_tips: np.ndarray, dt: float | None = None,
        straightness: float | None = None,
    ) -> np.ndarray:
        """Full pipeline for one frame. Returns (J,) drive values.

        `dt` is only used if smoothing is enabled, and only needs to be passed
        explicitly by non-real-time callers (see `JointLowPassFilter`).

        `straightness`: this frame's `data.thumb_straightness`, if the caller
        has the raw thumb joints. Only `refine_fist` uses it, to tell a fist
        from an extended thumb across devices; see there for why the mm
        fallback is AVP-only.
        """
        mapped = self.map_fingertips(human_tips)
        qpos, residual = self.solver.solve(mapped, last_qpos=self.last_qpos)
        self.last_contacts = None
        if self.pinch_mode == "classifier" and self.use_contacts and self.pinch_templates:
            scores = self.contact_scores(human_tips)
            if scores is not None:
                self.last_contacts = scores >= self.contact_threshold
                qpos = self.refine_pinch(qpos, mapped, scores)
        elif self.pinch_mode == "continuous" and self.pinch_templates:
            qpos = self.refine_pinch_continuous(qpos, mapped, human_tips)
        qpos = self.refine_fist(qpos, human_tips, straightness=straightness)
        if self.smoother is not None:
            qpos = self.smoother.apply(qpos, dt=dt)
        # Smoothed qpos feeds back as the continuity prior for next frame's NNS
        # lookup, so the solver chases the commanded (smoothed) pose rather than
        # the raw, jumpier one -- the two effects compound.
        self.last_qpos = qpos
        self.last_residual = residual
        return qpos

    def reset(self) -> None:
        self.last_qpos = None
        self.last_residual = None
        if self.smoother is not None:
            self.smoother.reset()

    def retarget_sequence(self, human_tips: np.ndarray) -> tuple[np.ndarray, dict]:
        """(T, F, 3) palm-frame fingertips -> ((T, J) drives, stats)."""
        human_tips = np.asarray(human_tips, dtype=np.float32)
        if human_tips.ndim != 3:
            raise ValueError(f"expected (T, F, 3), got {tuple(human_tips.shape)}")
        self.reset()
        out, residuals, contacts = [], [], []
        t0 = time.perf_counter()
        for t in range(human_tips.shape[0]):
            out.append(self.retarget(human_tips[t]))
            residuals.append(self.last_residual)
            contacts.append(
                self.last_contacts
                if self.last_contacts is not None
                else np.zeros(self.mapper.n_fingers - 1, dtype=bool)
            )
        dt = time.perf_counter() - t0
        stats = {
            "hz": human_tips.shape[0] / max(dt, 1e-9),
            "nns_residual_norm": np.stack(residuals),  # (T, F), normalized units
            "contacts": np.stack(contacts),
        }
        return np.stack(out), stats

    # ------------------------------------------------------------- live AVP

    def retarget_avp_poses(self, poses: np.ndarray, dt: float | None = None) -> np.ndarray:
        """One live Apple Vision Pro frame -> drives.

        `poses` is the (25, 4, 4) wrist-relative skeleton `avp_stream` returns
        (see anydexrt/avp.py). The palm frame is rebuilt from this frame's own
        skeleton (wrist, index/middle/pinky MCP = 0, 6, 11, 21), so no stored
        calibration is needed and a moved headset or re-seated wrist frame
        cannot misalign the input. The model was trained in the fixed mean palm
        frame of its recording; per-frame frames differ from it by < 3.1 deg on
        avp1.pkl.
        """
        from .data import THUMB_MCP_IP_TIP, palm_tips_from_avp_poses, thumb_straightness

        mcp, ip, tip = (np.asarray(poses)[i, :3, 3] for i in THUMB_MCP_IP_TIP)
        return self.retarget(
            palm_tips_from_avp_poses(poses, self.kin.hand), dt=dt,
            straightness=thumb_straightness(mcp, ip, tip),
        )

    def retarget_mediapipe_points(self, points21: np.ndarray, dt: float | None = None) -> np.ndarray:
        """One live frame of (21, 3) MediaPipe-order wrist-relative points -> drives.

        For sources that natively output MediaPipe landmarks instead of AVP's
        25-joint skeleton -- e.g. PICO 4 hand tracking, see
        `anydexrt.data.palm_tips_from_mediapipe_points`.
        """
        from .data import (
            THUMB_MCP_IP_TIP, palm_tips_from_mediapipe_points, thumb_straightness,
        )

        pts = np.asarray(points21)
        mcp, ip, tip = (pts[i] for i in THUMB_MCP_IP_TIP)
        return self.retarget(
            palm_tips_from_mediapipe_points(points21, self.kin.hand), dt=dt,
            straightness=thumb_straightness(mcp, ip, tip),
        )


# ------------------------------------------------------------------- bundles


def save_bundle(
    path,
    result,
    hand: str,
    human_normalizer: HandNormalizer,
    cloud_samples: int,
    cloud_seed: int = 0,
    extra: dict | None = None,
) -> None:
    """Everything needed to rebuild a working AnyDexRT, in one file.

    The robot cloud is not stored: it is regenerated deterministically from
    (hand, cloud_samples, cloud_seed), which keeps the bundle small.
    """
    from dataclasses import asdict

    torch.save(
        {
            "hand": hand,
            "mapper": result.mapper.state_dict(),
            "classifier": None if result.classifier is None else result.classifier.state_dict(),
            "config": asdict(result.config),
            "n_fingers": result.mapper.n_fingers,
            "human_normalizer": human_normalizer.to_dict(),
            "cloud_samples": cloud_samples,
            "cloud_seed": cloud_seed,
            "extra": extra or {},
        },
        path,
    )


def load_bundle(
    path, device: str = "cpu", use_contacts: bool = True, smoothing_hz: float | None = None,
    pinch_mode: str = "classifier", pinch_open_gap_mm: float | list[float] = 80.0,
    pinch_closed_gap_mm: float | list[float] = 6.0, pinch_min_extension_mm: float = 110.0,
    fist_thumb_rest_mm: float = 110.0, fist_thumb_extended_mm: float = 125.0,
    fist_thumb_bent: float = 0.92, fist_thumb_straight: float = 0.95,
    fist_open_reach_mm: float = 110.0,
    pinch_thumb_combine: int = 2, pinch_thumb_sharpness: float = 3.0,
) -> AnyDexRT:
    """Rebuild a ready-to-run AnyDexRT from `save_bundle` output."""
    from .data import build_robot_clouds
    from .models import CrossFingerMapper

    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload["config"]
    F = payload["n_fingers"]
    # Older bundles (saved before CrossFingerMapper existed) have no
    # "mapper_type" key; TrainConfig's own default ("finger") is the correct
    # read for them, so .get() rather than a hard requirement keeps them loadable.
    mapper_type = cfg.get("mapper_type", "finger")
    if mapper_type == "cross_finger":
        mapper = CrossFingerMapper(
            F, cfg.get("thumb_idx", 0), cfg["width"], cfg["depth"], cfg["activation"], cfg["output_tanh"]
        )
    elif mapper_type == "finger":
        mapper = FingerMapper(F, cfg["width"], cfg["depth"], cfg["activation"], cfg["output_tanh"])
    else:
        raise ValueError(f"bundle has unknown mapper_type {mapper_type!r}")
    mapper.load_state_dict(payload["mapper"])
    clf = None
    if payload["classifier"] is not None:
        clf = ContactClassifier(F, cfg["width"], cfg["depth"], cfg["activation"])
        clf.load_state_dict(payload["classifier"])
    clouds, kin = build_robot_clouds(
        payload["hand"], n_samples=payload["cloud_samples"], seed=payload["cloud_seed"], frame="palm"
    )
    return AnyDexRT(
        mapper, clouds, kin, HandNormalizer.from_dict(payload["human_normalizer"]),
        classifier=clf, pinch_templates=make_pinch_templates(clouds, kin),
        device=device, use_contacts=use_contacts, smoothing_hz=smoothing_hz,
        pinch_mode=pinch_mode, pinch_open_gap_mm=pinch_open_gap_mm,
        pinch_closed_gap_mm=pinch_closed_gap_mm, pinch_min_extension_mm=pinch_min_extension_mm,
        fist_thumb_rest_mm=fist_thumb_rest_mm, fist_thumb_extended_mm=fist_thumb_extended_mm,
        fist_thumb_bent=fist_thumb_bent, fist_thumb_straight=fist_thumb_straight,
        fist_open_reach_mm=fist_open_reach_mm,
        pinch_thumb_combine=pinch_thumb_combine,
        pinch_thumb_sharpness=pinch_thumb_sharpness,
    )
