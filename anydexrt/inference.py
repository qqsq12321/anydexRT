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

import numpy as np
import torch
from scipy.spatial import cKDTree

from .data import RobotClouds
from .kinematics import HandKinematics, NearestNeighborIK
from .models import ContactClassifier, FingerMapper
from .normalize import HandNormalizer


@dataclass
class PinchTemplates:
    """Pre-generated robot pinch poses for one (thumb, finger i) pair.

    `tips` is (N, 6): thumb tip then finger tip, robot cloud frame, metres.
    `thumb_q` / `finger_q` are the drive values for those two fingers.
    """

    finger: int
    tips: np.ndarray
    thumb_q: np.ndarray
    finger_q: np.ndarray
    gap: np.ndarray  # (N,) thumb-to-finger distance, metres

    def __post_init__(self):
        self.tree = cKDTree(self.tips)

    def nearest(self, thumb: np.ndarray, finger: np.ndarray) -> int:
        _, idx = self.tree.query(np.concatenate([thumb, finger]))
        return int(idx)


def make_pinch_templates(
    clouds: RobotClouds,
    kin: HandKinematics,
    per_finger: int = 400,
) -> dict[int, PinchTemplates]:
    """Pre-generate pinch poses by pairing the thumb table with each finger table.

    The fingers are independent chains, so any thumb row combined with any
    finger-i row of the sampled cloud is a valid whole-hand configuration. For
    every thumb sample the nearest finger-i tip is found with a KD-tree, and the
    `per_finger` closest pairs are kept. This gives genuine contacts (a few mm),
    whereas picking whole-hand random samples by thumb-finger distance yields
    near-misses: random poses almost never pinch.

    The paper says templates are "pre-generated" without saying how; this is a
    reproduction choice.
    """
    tips = clouds.tips  # (N, F, 3), cloud frame, metres
    thumb_sl = kin.hand.joint_slice(kin.hand.fingers[0].name)
    out: dict[int, PinchTemplates] = {}
    for i in range(1, kin.n_fingers):
        sl = kin.hand.joint_slice(kin.hand.fingers[i].name)
        dist, idx = cKDTree(tips[:, i]).query(tips[:, 0])  # nearest finger-i per thumb row
        keep = np.argsort(dist)[:per_finger]
        a, b = keep, idx[keep]
        out[i] = PinchTemplates(
            finger=i,
            tips=np.concatenate([tips[a, 0], tips[b, i]], axis=1).astype(np.float32),
            thumb_q=clouds.qpos[a][:, thumb_sl].copy(),
            finger_q=clouds.qpos[b][:, sl].copy(),
            gap=dist[keep].astype(np.float32),
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
        return self.mapper(x[None])[0].cpu().numpy()

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

    def retarget(self, human_tips: np.ndarray, dt: float | None = None) -> np.ndarray:
        """Full pipeline for one frame. Returns (J,) drive values.

        `dt` is only used if smoothing is enabled, and only needs to be passed
        explicitly by non-real-time callers (see `JointLowPassFilter`).
        """
        mapped = self.map_fingertips(human_tips)
        qpos, residual = self.solver.solve(mapped, last_qpos=self.last_qpos)
        self.last_contacts = None
        if self.use_contacts and self.pinch_templates:
            scores = self.contact_scores(human_tips)
            if scores is not None:
                self.last_contacts = scores >= self.contact_threshold
                qpos = self.refine_pinch(qpos, mapped, scores)
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
        from .data import palm_tips_from_avp_poses

        return self.retarget(palm_tips_from_avp_poses(poses, self.kin.hand), dt=dt)


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
) -> AnyDexRT:
    """Rebuild a ready-to-run AnyDexRT from `save_bundle` output."""
    from .data import build_robot_clouds

    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload["config"]
    F = payload["n_fingers"]
    mapper = FingerMapper(F, cfg["width"], cfg["depth"], cfg["activation"], cfg["output_tanh"])
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
    )
