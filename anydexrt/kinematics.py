"""MJCF-backed forward kinematics for robot hands.

Provides fingertip positions for a given joint configuration, in the canonical
joint order defined by `hands.HandSpec`. Everything downstream (fingertip
clouds, anchors, nearest-neighbour inverse kinematics) goes through this class,
so the joint ordering is defined in exactly one place.

The class also exposes a batched path used during training to sample robot
fingertip clouds, and a differentiable-free NNS "inverse" solver.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .hands import HandSpec, get_hand


class HandKinematics:
    """Forward kinematics for one robot hand."""

    def __init__(self, hand: HandSpec | str, asset_root: Path | None = None):
        self.hand = get_hand(hand) if isinstance(hand, str) else hand
        mjcf = self.hand.resolved_mjcf(asset_root)
        if not mjcf.exists():
            raise FileNotFoundError(
                f"MJCF not found for hand {self.hand.name!r}: {mjcf}. "
                "Pass asset_root or fix HandSpec.mjcf."
            )
        self.model = mujoco.MjModel.from_xml_path(str(mjcf))
        self.data = mujoco.MjData(self.model)

        # Write targets for each drive. A drive is usually one MJCF joint, but
        # can be a tendon-style coupling spanning several (see HandSpec.couplings).
        self._joint_adrs: dict[str, int] = {}
        self._drive_joints: list[dict[str, tuple[int, float]]] = []
        for name in self.hand.joint_names:
            targets = self.hand.drive_targets(name)
            resolved: dict[str, tuple[int, float]] = {}
            for joint_name, mult in targets.items():
                if joint_name not in self._joint_adrs:
                    jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    if jid < 0:
                        raise ValueError(
                            f"drive {name!r} targets joint {joint_name!r}, which is "
                            f"not in {mjcf.name}; available: {self._joint_names()}"
                        )
                    self._joint_adrs[joint_name] = int(self.model.jnt_qposadr[jid])
                resolved[joint_name] = (self._joint_adrs[joint_name], float(mult))
            self._drive_joints.append(resolved)

        # A joint written by two drives would silently accumulate; reject it.
        seen: dict[int, str] = {}
        for name, resolved in zip(self.hand.joint_names, self._drive_joints):
            for joint_name, (adr, _) in resolved.items():
                if adr in seen:
                    raise ValueError(
                        f"joint {joint_name!r} is actuated by both {seen[adr]!r} and "
                        f"{name!r}; the coupling table must assign each joint once"
                    )
                seen[adr] = name

        # Fingertip body ids, in canonical finger order.
        self._tip_bids = np.zeros(self.hand.n_fingers, dtype=np.int32)
        for i, name in enumerate(self.hand.tip_bodies):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(
                    f"fingertip body {name!r} not found in {mjcf.name}; "
                    f"available: {self._body_names()}"
                )
            self._tip_bids[i] = bid
        self._tip_offsets = np.asarray(self.hand.tip_offsets, dtype=np.float64)  # (F, 3)

        self._palm_bids: np.ndarray | None = None
        if self.hand.palm_bodies is not None:
            self._palm_bids = np.array(
                [self._body_id(n, mjcf) for n in self.hand.palm_bodies], dtype=np.int32
            )

        self._limits_lo, self._limits_hi = self._read_limits()
        # A neutral reference qpos used to build local frames from FK.
        self._neutral = (self._limits_lo + self._limits_hi) / 2.0

    # ---------------------------------------------------------------- helpers

    def _body_id(self, name: str, mjcf: Path) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(
                f"body {name!r} not found in {mjcf.name}; available: {self._body_names()}"
            )
        return int(bid)

    def _current_tips(self) -> np.ndarray:
        """Fingertip positions from the current mj_data, offsets applied. (F, 3)."""
        pos = self.data.xpos[self._tip_bids]  # (F, 3)
        rot = self.data.xmat[self._tip_bids].reshape(-1, 3, 3)  # (F, 3, 3)
        return (pos + np.einsum("fij,fj->fi", rot, self._tip_offsets)).astype(np.float32)

    def _joint_names(self) -> list[str]:
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]

    def _body_names(self) -> list[str]:
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(self.model.nbody)
        ]

    def _joint_range(self, name: str) -> tuple[float, float]:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(
                f"joint {name!r} not found in {self.hand.name}; "
                f"available: {self._joint_names()}"
            )
        if self.model.jnt_limited[jid]:
            return float(self.model.jnt_range[jid][0]), float(self.model.jnt_range[jid][1])
        # Unlimited (rare on these hands) -> use a sane symmetric range.
        return -1.5, 1.5

    def _read_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Limits of each *drive*.

        For a plain joint this is the joint's own range. For a coupled drive the
        drive value maps to child joints scaled by `mult`, so the drive's range
        is the tightest child range mapped back through the multiplier -- a
        coupled drive can never push its children past their limits.
        """
        lo = np.zeros(len(self.hand.joint_names), dtype=np.float32)
        hi = np.zeros(len(self.hand.joint_names), dtype=np.float32)
        for i, name in enumerate(self.hand.joint_names):
            targets = self.hand.drive_targets(name)
            cands_lo, cands_hi = [], []
            for joint_name, mult in targets.items():
                jlo, jhi = self._joint_range(joint_name)
                if mult == 0.0:
                    continue
                a, b = jlo / mult, jhi / mult
                cands_lo.append(min(a, b))
                cands_hi.append(max(a, b))
            if not cands_lo:
                raise ValueError(f"drive {name!r} actuates no joints")
            lo[i], hi[i] = max(cands_lo), min(cands_hi)
            if lo[i] > hi[i]:
                raise ValueError(
                    f"drive {name!r} has empty limit range from couplings "
                    f"{targets}; check the multipliers"
                )
        return lo, hi

    # ------------------------------------------------------------------- API

    @property
    def n_joints(self) -> int:
        return len(self.hand.joint_names)

    @property
    def n_fingers(self) -> int:
        return self.hand.n_fingers

    def limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._limits_lo.copy(), self._limits_hi.copy()

    def _expand_drives(self, drive_values: np.ndarray) -> np.ndarray:
        """Expand drive values into MJCF joint values via the coupling table.

        Two drives writing the same joint would accumulate, which is why the
        coupling table must not list a joint under two drivers; that is checked
        in the constructor.
        """
        out = np.zeros(self.model.nq, dtype=np.float64)
        for i, resolved in enumerate(self._drive_joints):
            value = drive_values[i]
            for adr, mult in resolved.values():
                out[adr] += mult * value
        return out

    def set_qpos(self, qpos: np.ndarray) -> None:
        """Write a canonical-order drive vector into the MJCF data buffer."""
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos.size != self.n_joints:
            raise ValueError(f"expected {self.n_joints} joints, got {qpos.size}")
        self.data.qpos[:] = self._expand_drives(qpos)
        mujoco.mj_forward(self.model, self.data)

    def fingertip_positions(self, qpos: np.ndarray) -> np.ndarray:
        """Fingertips for one configuration, shape (F, 3), in canonical order."""
        self.set_qpos(qpos)
        return self._current_tips()

    def fingertip_positions_batch(self, qpos_batch: np.ndarray) -> np.ndarray:
        """Fingertips for many configurations at once.

        Args:
            qpos_batch: (B, n_joints) in canonical joint order.

        Returns:
            (B, F, 3) fingertip positions.
        """
        qpos_batch = np.asarray(qpos_batch, dtype=np.float64)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.n_joints:
            raise ValueError(
                f"expected (B, {self.n_joints}) qpos batch, got {qpos_batch.shape}"
            )
        out = np.empty((qpos_batch.shape[0], self.n_fingers, 3), dtype=np.float32)
        for b in range(qpos_batch.shape[0]):
            self.data.qpos[:] = self._expand_drives(qpos_batch[b])
            mujoco.mj_forward(self.model, self.data)
            out[b] = self._current_tips()
        return out

    def fingertip_per_finger(self, qpos_batch: np.ndarray) -> np.ndarray:
        """Fingertips produced by each finger's joints *independently*.

        This is how the per-finger mapper is trained: every finger owns its own
        sub-network and only ever sees its own joints, so the training target
        must be generated with the other fingers held at their neutral pose.

        Args:
            qpos_batch: (B, n_joints) canonical joint order.

        Returns:
            (B, F, 3) where entry [b, i] is finger i's tip with finger i set to
            qpos_batch[b, finger_i] and all other joints at neutral.
        """
        qpos_batch = np.asarray(qpos_batch, dtype=np.float64)
        B = qpos_batch.shape[0]
        out = np.empty((B, self.n_fingers, 3), dtype=np.float32)
        full = np.tile(self._neutral.astype(np.float64), (B, 1))
        for i, finger in enumerate(self.hand.fingers):
            sl = self.hand.joint_slice(finger.name)
            full[:, sl] = qpos_batch[:, sl]
            tips = self.fingertip_positions_batch(full)
            out[:, i] = tips[:, i]
            full[:, sl] = self._neutral[sl]
        return out

    def sample_qpos(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Uniform samples inside joint limits, shape (n, n_joints)."""
        lo, hi = self._limits_lo, self._limits_hi
        return rng.uniform(lo, hi, size=(n, self.n_joints)).astype(np.float32)

    def sample_fingertip_cloud(
        self, n: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample configurations and return (qpos, fingertips).

        Returns:
            qpos: (n, n_joints), fingertips: (n, F, 3). The fingertip entry for
            finger i is generated with finger i's own joints only, so the cloud
            per finger is a faithful sample of that finger's reachable space.
        """
        qpos = self.sample_qpos(n, rng)
        tips = self.fingertip_per_finger(qpos)
        return qpos, tips

    def rest_qpos(self) -> np.ndarray:
        """Zero pose clipped into limits: the open, flat hand on these models."""
        return np.clip(0.0, self._limits_lo, self._limits_hi).astype(np.float32)

    def palm_frame(self, qpos: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Canonical palm frame from the hand's own skeleton.

        Returns (origin, R) with R's columns the frame axes, so a world point p
        maps to canonical coordinates as R.T @ (p - origin). Computed at the
        rest pose by default. See `palm_frame_from_points` for the convention.
        """
        if self._palm_bids is None:
            raise ValueError(f"{self.hand.name} has no palm_bodies; cannot build a palm frame")
        self.set_qpos(self.rest_qpos() if qpos is None else qpos)
        wrist, index, middle, pinky = (self.data.xpos[b].copy() for b in self._palm_bids)
        return wrist, palm_frame_from_points(wrist, index, middle, pinky)


def palm_frame_from_points(
    wrist: np.ndarray, index_mcp: np.ndarray, middle_mcp: np.ndarray, pinky_mcp: np.ndarray
) -> np.ndarray:
    """Rotation of the canonical palm frame, (3, 3), axes as columns.

    x  wrist -> middle MCP          (along the fingers)
    y  pinky MCP -> index MCP, made orthogonal to x   (across the palm)
    z  x cross y                     (palm normal)

    The same rule applied to a human skeleton and to a robot's bodies gives both
    sides semantically matching axes, which is what replaces the hand-tuned
    input rotations (e.g. `mediapipe_rotation: {x: -7}`) used elsewhere. It uses
    only the skeleton, so it introduces no tuned parameter.
    """
    x = np.asarray(middle_mcp, dtype=np.float64) - np.asarray(wrist, dtype=np.float64)
    nx = np.linalg.norm(x)
    if nx < 1e-9:
        raise ValueError("degenerate palm: wrist and middle MCP coincide")
    x /= nx
    y = np.asarray(index_mcp, dtype=np.float64) - np.asarray(pinky_mcp, dtype=np.float64)
    y -= x * float(y @ x)
    ny = np.linalg.norm(y)
    if ny < 1e-9:
        raise ValueError("degenerate palm: index-pinky axis parallel to the fingers")
    y /= ny
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


class NearestNeighborIK:
    """`f_s`: inverse kinematics by nearest-neighbour lookup, one table per finger.

    The paper: "We simply use nearest neighbor search (NNS) to get the
    corresponding joint configurations from the given fingertip positions and
    the last joint configurations."

    The lookup is factorized by finger. On these hands every finger is an
    independent kinematic chain off the palm, so finger i's tip depends only on
    finger i's drives, and a table of (finger-i drives, finger-i tip) pairs is
    exact. An earlier version looked up whole-hand samples and voted across
    fingers; with 16-20 DoF a single random sample essentially never lies near
    the target for all fingers at once, so that version returned a compromise
    pose whose per-finger error grew with the number of fingers.

    "The last joint configurations" enters as a continuity prior: among the k
    nearest table entries, the one closest to the previous solution in that
    finger's joint space wins, which suppresses frame-to-frame flips between
    distant configurations that reach the same tip.
    """

    def __init__(
        self,
        kin: HandKinematics,
        qpos: np.ndarray,
        tips: np.ndarray,
        n_candidates: int = 8,
        prior_weight: float = 0.05,
    ):
        from scipy.spatial import cKDTree

        self.kin = kin
        self.qpos = np.asarray(qpos, dtype=np.float32)
        self.tips = np.asarray(tips, dtype=np.float32)
        if self.qpos.ndim != 2 or self.tips.ndim != 3:
            raise ValueError("expected qpos (N, J) and tips (N, F, 3)")
        if self.tips.shape[1] != kin.n_fingers:
            raise ValueError(f"tips have {self.tips.shape[1]} fingers, hand has {kin.n_fingers}")
        self.n_candidates = n_candidates
        self.prior_weight = prior_weight
        self.slices = [kin.hand.joint_slice(f.name) for f in kin.hand.fingers]
        self.trees = [cKDTree(self.tips[:, i, :]) for i in range(kin.n_fingers)]
        # Joint-space scale per finger, so the prior is unit-free.
        lo, hi = kin.limits()
        self._joint_span = [np.maximum(hi[s] - lo[s], 1e-6) for s in self.slices]

    @classmethod
    def build(
        cls, kin: HandKinematics, n_samples: int = 20000, seed: int = 0, **kwargs
    ) -> "NearestNeighborIK":
        rng = np.random.default_rng(seed)
        qpos, tips = kin.sample_fingertip_cloud(n_samples, rng)
        return cls(kin, qpos, tips, **kwargs)

    def solve(
        self, target_tips: np.ndarray, last_qpos: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve for one target pose.

        Args:
            target_tips: (F, 3) desired fingertips, in the same space as `tips`.
            last_qpos: (J,) previous solution for the continuity prior, or None.

        Returns:
            (qpos (J,), residual (F,)): the solution and, per finger, the
            distance between the target and the table tip that was chosen --
            the error `f_s` itself contributes.
        """
        target_tips = np.asarray(target_tips, dtype=np.float32)
        if target_tips.shape != (self.kin.n_fingers, 3):
            raise ValueError(
                f"expected ({self.kin.n_fingers}, 3) target tips, got {target_tips.shape}"
            )
        out = np.empty(self.qpos.shape[1], dtype=np.float32)
        residual = np.empty(self.kin.n_fingers, dtype=np.float32)
        k = min(self.n_candidates, len(self.qpos))
        for i, (tree, sl) in enumerate(zip(self.trees, self.slices)):
            dist, idx = tree.query(target_tips[i], k=k)
            dist, idx = np.atleast_1d(dist), np.atleast_1d(idx)
            cost = dist.copy()
            if last_qpos is not None and self.prior_weight > 0:
                dq = (self.qpos[idx, sl] - last_qpos[sl]) / self._joint_span[i]
                cost = cost + self.prior_weight * np.linalg.norm(dq, axis=1)
            best = int(np.argmin(cost))
            out[sl] = self.qpos[idx[best], sl]
            residual[i] = dist[best]
        return out, residual
