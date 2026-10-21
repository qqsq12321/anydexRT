"""Motion consistency metrics: GMC and LMC.

Neither metric has a public implementation in this repository, so this is a
reconstruction. Two distinct conventions are involved, per the paper:

  GMC (global motion consistency) -- "compares displacement directions in a
  shared coordinate frame, assuming an ideal, well-calibrated setting."

  LMC (local motion consistency) -- "compares directions in local frames,
  making it less calibration-dependent and more aligned with operator control."

Both are reported as percentages in [0, 100]. The scale is read off the reported
numbers: the ablation's collapsed case scores 4.1, and a collapsed mapping maps
every human sample to essentially one robot point, so its displacement
directions are noise and the mean cosine of noise against real human motion is
~0. Hence score = 100 * mean(cosine), which puts a random mapping at ~0 and a
direction-preserving mapping at ~90 for real data.

This reading is stated explicitly because the absolute values depend on it, and
the whole point of the validation ladder is to compare *patterns* rather than
to match the paper's absolute numbers -- see README.

Conventions we are forced to choose (the paper does not specify them):
  * Aggregation is over all samples of all fingers, pooled, rather than
    per-finger-averaged. Pooling is used because it is the simpler reading of
    "average over seven hands" style reporting.
  * Displacements are measured between consecutive samples in the provided
    sequence, per finger.
  * Cosine is averaged directly (not clipped at 0) unless clip_negative=True.
    `clip_negative` is exposed because a mean cosine near 0.04 is the
    signature of collapse either way.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .frames import nearest_neighbor_rotation, rotate_into_local


@dataclass
class MotionConsistency:
    """Result of a GMC/LMC evaluation."""

    score: float  # in [0, 100]
    mean_cosine: float  # in [-1, 1]
    n_samples: int
    per_finger: list[float]

    def __str__(self) -> str:
        pf = " ".join(f"{x:.1f}" for x in self.per_finger)
        return f"{self.score:.1f} (cos={self.mean_cosine:+.4f}, n={self.n_samples}, per-finger [{pf}])"


def _as_batched(traj: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Promote (N, 3) to (1, N, 3); report whether it was promoted."""
    if traj.dim() == 2:
        if traj.shape[-1] != 3:
            raise ValueError(f"expected last dim 3, got {tuple(traj.shape)}")
        return traj.unsqueeze(0), True
    if traj.dim() == 3 and traj.shape[-1] == 3:
        return traj, False
    raise ValueError(f"expected (N, 3) or (B, N, 3), got {tuple(traj.shape)}")


def displacements(traj: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-step displacement vectors and their magnitudes.

    Args:
        traj: (B, N, 3) (or (N, 3), promoted).

    Returns:
        (delta, magnitude) both (B, N-1, 3) and (B, N-1, 1). Returning both from
        one place keeps the validity mask and the normalized directions
        consistent -- computing them separately lets a sample be excluded by one
        and included by the other.
    """
    traj, _ = _as_batched(traj)
    if traj.shape[1] < 2:
        raise ValueError("need at least two frames")
    delta = traj[:, 1:] - traj[:, :-1]
    mag = torch.linalg.norm(delta, dim=-1, keepdim=True)
    return delta, mag


def _unit(delta: torch.Tensor, mag: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return delta / mag.clamp_min(eps)


def motion_consistency(
    human: torch.Tensor,
    robot: torch.Tensor,
    human_rot: torch.Tensor | None = None,
    robot_rot: torch.Tensor | None = None,
    local: bool = False,
    min_displacement: float = 1e-5,
    clip_negative: bool = False,
) -> MotionConsistency:
    """Cosine agreement between human and robot displacement directions.

    Args:
        human: (B, N, 3) human fingertip trajectory (normalized space).
        robot: (B, N, 3) the corresponding retargeted trajectory, same shape.
        human_rot: (B, N, 3, 3) local frames per human sample. Required if local.
        robot_rot: (B, N, 3, 3) local frames per robot sample. Required if local.
        local: False gives GMC (shared frame), True gives LMC (local frames).
        min_displacement: displacements shorter than this are excluded, since
            their direction is numerically meaningless.
        clip_negative: clamp cosine at 0 before averaging.

    Returns:
        MotionConsistency with `score` in [0, 100].
    """
    human, _ = _as_batched(human)
    robot, _ = _as_batched(robot)
    if human.shape != robot.shape:
        raise ValueError(f"shape mismatch: {tuple(human.shape)} vs {tuple(robot.shape)}")

    d_h, mag_h = displacements(human)
    d_r, mag_r = displacements(robot)

    valid = (mag_h.squeeze(-1) > min_displacement) & (
        mag_r.squeeze(-1) > min_displacement
    )

    dh = _unit(d_h, mag_h)
    dr = _unit(d_r, mag_r)

    if local:
        if human_rot is None or robot_rot is None:
            raise ValueError("local=True requires both human_rot and robot_rot")
        # Rotate each displacement into its own sample's frame. Frames are given
        # per sample; the displacement spans samples j and j+1, so the frame at
        # the earlier sample is used (the tangent frame convention).
        dh = rotate_into_local(dh, human_rot[:, :-1])
        dr = rotate_into_local(dr, robot_rot[:, :-1])

    cos = (F.normalize(dh, dim=-1, eps=1e-8) * F.normalize(dr, dim=-1, eps=1e-8)).sum(-1)

    if valid.sum() == 0:
        raise ValueError("no valid displacements; check min_displacement and input")

    cos_valid = cos[valid]
    if clip_negative:
        cos_valid = cos_valid.clamp_min(0.0)

    per_finger = [float(cos_valid[i].mean()) for i in range(cos_valid.shape[0])]
    return MotionConsistency(
        score=100.0 * float(cos_valid.mean()),
        mean_cosine=float(cos_valid.mean()),
        n_samples=int(valid.sum()),
        per_finger=per_finger,
    )


def robot_frames_from_cloud(
    mapped: torch.Tensor, cloud: torch.Tensor, cloud_rot: torch.Tensor
) -> torch.Tensor:
    """Robot-side local frames: the nearest neighbor's rotation, per the paper."""
    return nearest_neighbor_rotation(mapped, cloud, cloud_rot)


def apply_global_perturbation(
    human: torch.Tensor, human_rot: torch.Tensor | None, rot: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Rotate inputs about the origin to simulate glove calibration error.

    This reproduces the paper's calibration-sensitivity experiment, which
    rotates the input human data around the y-axis by -90, -45, 45 and 90
    degrees. When the mapping is learned in local frames, a global rotation
    should leave LMC essentially unchanged; a globally-defined mapping degrades.

    Args:
        human: (B, N, 3) or (N, 3).
        human_rot: (B, N, 3, 3) or None.
        rot: (3, 3) rotation left-multiplied onto positions.

    Returns:
        (rotated human, rotated frames or None)
    """
    if rot.shape != (3, 3):
        raise ValueError(f"rot must be (3, 3), got {tuple(rot.shape)}")
    rotated = human @ rot.T
    if human_rot is None:
        return rotated, None
    if human_rot.shape[-2:] != (3, 3):
        raise ValueError(f"human_rot must end in (3, 3), got {tuple(human_rot.shape)}")
    rotated_rot = torch.einsum("ij,...jk->...ik", rot, human_rot)
    return rotated, rotated_rot
