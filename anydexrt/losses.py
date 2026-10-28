"""The four mapping losses plus the contact loss, transcribed from the paper.

Equations as printed in the paper:

  (1) L_P-Chamfer(C^{H,i}, C^{R,i}) = 1/|C^{H,i}| * sum_j min_k || f_m^i(x_j) - x_k ||

  (2) L_dist(C^{H,i}) = 1/(|C|(|C|-1)) * sum_{j1!=j2}
                        ( ||f_m(x_j1) - f_m(x_j2)|| - ||x_j1 - x_j2|| )^2

  (3) L_motion(C^{H,i}) = -1/|C| * sum_j
        < T^-1(x_j) * dx/||dx|| , T^-1(f_m(x_j)) * df/||df|| >

  (4) L_align = 1/M * sum_j || f_m(x_bar_j) - x_bar_j^R ||

  (5) L_contact = 1/N * sum_j BCE( y_j^i , f_c^i(x_j^0, x_j^i) )

and the total is

  L_mapping = L_P-Chamfer + L_dist + L_motion + L_align

with, per Appendix B, "we do not specifically tune the weights of each loss
term" -- so every weight is 1.

Two details worth stating because they drive whether this reproduces at all:

* L_P-Chamfer is ONE-directional and uses the L2 NORM (the paper says so
  explicitly: "||.|| stands for L2 normalization"), not the squared distance
  that GeoRT's bidirectional chamfer uses. Norm, not squared norm, and no
  reverse term. Its unconstrained optimum is a point mass -- mapping every
  human sample onto a single robot point drives the loss to zero -- which is
  exactly why L_dist is load-bearing rather than decorative, and why the
  ablation starts at 4.1 before L_dist lifts it to 83.8.

* L_dist IS squared (the paper's (..)^2 is on the difference of distances).
  So the two terms deliberately use different distance conventions, matching
  the printed equations rather than any attempt to "unify" them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .frames import nearest_neighbor_rotation, rotate_into_local


@dataclass
class LossWeights:
    """Per-term weights. Defaults are all 1, per Appendix B."""

    p_chamfer: float = 1.0
    dist: float = 1.0
    motion: float = 1.0
    align: float = 1.0

    def active_terms(self) -> dict[str, float]:
        """Terms with non-zero weight; used to run the ablation ladder."""
        out = {
            "p_chamfer": self.p_chamfer,
            "dist": self.dist,
            "motion": self.motion,
            "align": self.align,
        }
        return {k: v for k, v in out.items() if v != 0.0}


def partial_chamfer_loss(
    mapped: torch.Tensor,
    robot_cloud: torch.Tensor,
    chunk: int = 512,
    reduction: str = "mean",
) -> torch.Tensor:
    """Eq (1): one-directional Chamfer, L2 norm, mean over human samples.

    Args:
        mapped: (B, N, 3) mapped human fingertips for one finger.
        robot_cloud: (M, 3) that finger's reachable robot fingertip positions.
        chunk: rows of `mapped` processed per cdist call, for memory.
        reduction: 'mean' (paper) or 'none' to keep per-sample values.

    Returns:
        Scalar loss, or (B,) if reduction='none'.
    """
    if robot_cloud.dim() != 2 or robot_cloud.shape[1] != 3:
        raise ValueError(f"robot_cloud must be (M, 3), got {tuple(robot_cloud.shape)}")
    B, N, _ = mapped.shape
    mins = []
    for start in range(0, N, chunk):
        block = mapped[:, start : start + chunk]  # (B, c, 3)
        d = torch.cdist(block.reshape(-1, 3), robot_cloud)  # (B*c, M), L2 by default
        mins.append(d.min(dim=1).values.reshape(B, -1))
    per_sample = torch.cat(mins, dim=1)  # (B, N)
    if reduction == "none":
        return per_sample.mean(dim=1)
    return per_sample.mean()


def distance_preservation_loss(
    mapped: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    """Eq (2): match pairwise distances before and after mapping. Squared form.

    Normalized by |C|(|C|-1), i.e. the number of ordered pairs, so this is a
    mean over all ordered pairs that the paper sums over.
    """
    B, N, _ = mapped.shape
    if source.shape != mapped.shape:
        raise ValueError(f"source {tuple(source.shape)} != mapped {tuple(mapped.shape)}")
    if N < 2:
        return mapped.sum() * 0.0

    d_mapped = torch.cdist(mapped, mapped)  # (B, N, N)
    d_source = torch.cdist(source, source)
    diff = (d_mapped - d_source) ** 2
    # Zero the diagonal so the j1 == j2 terms drop out of the mean.
    eye = torch.eye(N, dtype=torch.bool, device=diff.device)
    diff = diff.masked_fill(eye, 0.0)
    return diff.sum(dim=(1, 2)) / (N * (N - 1))


def local_motion_loss(
    mapped: torch.Tensor,
    source: torch.Tensor,
    mapper,
    finger: int,
    delta: torch.Tensor,
    human_rot: torch.Tensor | None = None,
    robot_cloud: torch.Tensor | None = None,
    robot_cloud_rot: torch.Tensor | None = None,
    context: torch.Tensor | None = None,
) -> torch.Tensor:
    """Eq (3): motion directions must agree after rotating into local frames.

    Both directions are unit-normalized and compared by inner product, so the
    loss is the negative mean cosine similarity and lives in [-1, 1].

    Args:
        mapped: (B, N, 3) f_m(x_j) for this finger (needed only for shapes).
        source: (B, N, 3) x_j, the un-jittered human fingertips.
        mapper: module with forward_finger(x, finger) -> (B, 3).
        finger: index of the finger being trained.
        delta: (B, N, 3) perturbation applied to the human points.
        human_rot: (B, N, 3, 3) local frames for the human side. None means
            global frame (identity) -- the GeoRT convention, kept so the
            global-vs-local comparison can be run directly.
        robot_cloud, robot_cloud_rot: (M, 3) and (M, 3, 3) for the robot side.
            When provided, the robot frame is the nearest neighbour's rotation,
            per the paper. When None, the robot side defaults to global too.
        context: (N, C) other-fingers context for `mapper.forward_finger`, only
            meaningful when `finger` is a CrossFingerMapper thumb net and
            `mapper` accepts the `context` kwarg. Held FIXED across both the
            jittered and un-jittered evaluation (only this finger's own point
            moves), then broadcast over `B`. None for a plain FingerMapper.

    Returns:
        Scalar loss for this finger.
    """
    B, N, _ = source.shape

    def _map(pts: torch.Tensor) -> torch.Tensor:
        flat = pts.reshape(-1, 3)
        if context is None:
            return mapper.forward_finger(flat, finger).reshape(B, N, 3)
        ctx = context.unsqueeze(0).expand(B, N, -1).reshape(-1, context.shape[-1])
        return mapper.forward_finger(flat, finger, context=ctx).reshape(B, N, 3)

    d_x = delta
    d_f = _map(source + delta) - _map(source)

    eps = 1e-5
    d_x_hat = F.normalize(d_x, dim=-1, eps=eps)
    d_f_hat = F.normalize(d_f, dim=-1, eps=eps)

    if human_rot is not None:
        d_x_hat = rotate_into_local(d_x_hat.squeeze(0), human_rot.squeeze(0))
    if robot_cloud is not None and robot_cloud_rot is not None:
        rot = nearest_neighbor_rotation(mapped, robot_cloud, robot_cloud_rot)
        d_f_hat = rotate_into_local(d_f_hat.squeeze(0), rot.squeeze(0))

    cos = (F.normalize(d_x_hat, dim=-1, eps=eps) * F.normalize(d_f_hat, dim=-1, eps=eps)).sum(-1)
    return -cos.mean()


def alignment_loss(
    mapper, finger: int, human_anchor: torch.Tensor, robot_anchor: torch.Tensor
) -> torch.Tensor:
    """Eq (4): mean L2 over M paired anchors.

    Args:
        human_anchor: (M, 3) normalized human anchor positions.
        robot_anchor: (M, 3) normalized robot anchor positions.
    """
    if human_anchor.shape != robot_anchor.shape:
        raise ValueError(
            f"anchor shape mismatch: {tuple(human_anchor.shape)} vs "
            f"{tuple(robot_anchor.shape)}"
        )
    pred = mapper.forward_finger(human_anchor, finger)
    return torch.linalg.norm(pred - robot_anchor, dim=-1).mean()


def contact_loss(
    scores: torch.Tensor, labels: torch.Tensor, pos_weight: float | None = None
) -> torch.Tensor:
    """Eq (5): BCE over samples for one finger pair.

    Args:
        scores: (N,) predicted scores in [0, 1] (post-Sigmoid).
        labels: (N,) targets in {0, 1}.
        pos_weight: optional positive-class weight, applied to the positive term
            manually because `binary_cross_entropy` (the probability-input
            variant, which is what a Sigmoid-terminated net needs) has no such
            keyword -- only the `_with_logits` variant does. Contact is rare, so
            an unweighted BCE can settle on the all-negative solution; the paper
            does not mention reweighting, so this defaults to None.
    """
    if scores.shape != labels.shape:
        raise ValueError(f"shape mismatch: {tuple(scores.shape)} vs {tuple(labels.shape)}")
    scores = scores.clamp(1e-6, 1 - 1e-6)
    targets = labels.float()
    loss = F.binary_cross_entropy(scores, targets, reduction="none")
    if pos_weight is not None and pos_weight != 1.0:
        weight = torch.where(
            targets > 0.5,
            torch.full_like(loss, float(pos_weight)),
            torch.ones_like(loss),
        )
        loss = loss * weight
    return loss.mean()


def jitter_delta(
    x: torch.Tensor,
    generator: torch.Generator | None = None,
    base_scale: float = 0.001,
    rand_scale: float = 0.01,
) -> torch.Tensor:
    """Random perturbation for L_motion, with per-sample magnitude.

    The paper says samples are "randomly perturb[ed] ... with a generated delta
    movement" without giving a magnitude. GeoRT's direction loss uses
    scale = 0.001 + rand(0, 1) * 0.01 in the same normalized space, so that is
    the default here and is exposed as a knob.

    Returns: (B, N, 3) deltas with uniformly random direction and per-sample scale.
    """
    direction = F.normalize(
        torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype),
        dim=-1,
        eps=1e-8,
    )
    if rand_scale == 0.0:
        return direction * base_scale
    rand = torch.rand(
        x.shape[:-1] + (1,), generator=generator, device=x.device, dtype=x.dtype
    )
    scale = base_scale + rand * rand_scale
    return direction * scale


def mapping_loss(
    mapper,
    human: torch.Tensor,
    robot_cloud: torch.Tensor,
    weights: LossWeights,
    generator: torch.Generator | None = None,
    human_rot: torch.Tensor | None = None,
    robot_cloud_rot: torch.Tensor | None = None,
    anchors: tuple[torch.Tensor, torch.Tensor] | None = None,
    jitter_base: float = 0.001,
    jitter_rand: float = 0.01,
    human_paired: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Total L_mapping over all fingers, plus a detached per-term breakdown.

    Args:
        mapper: FingerMapper, or CrossFingerMapper (duck-typed via a
            `thumb_idx` attribute -- when present, the thumb finger's forward
            pass is given the other fingers' same-frame context from
            `human_paired`).
        human: (F, N, 3) normalized human fingertip samples: one set of N
            samples per finger. N is the objective's sample set, so it must be
            >= 2 for L_dist to have any pairs.
        robot_cloud: (F, M, 3) normalized robot fingertip clouds.
        weights: which terms to include and how heavily.
        anchors: optional ((F, M_a, 3) human, (F, M_a, 3) robot) paired anchors.
        human_rot, robot_cloud_rot: (F, N, 3, 3) and (F, M, 3, 3) local frames.
        human_paired: (N, F, 3) SAME-FRAME human fingertips across the finger
            axis (i.e. `human.permute(1, 0, 2)`, same N samples, same frame
            order). Required when `mapper` is cross-finger-aware (gives the
            thumb sub-net its context, used by every active loss term that
            touches the thumb -- not just one of them). None is fine for a
            plain FingerMapper.

    Returns:
        (total_loss, {'p_chamfer': float, 'dist': float, ...}) summed over
        fingers, matching the paper's per-finger sum in L_mapping.
    """
    n_fingers = human.shape[0]
    active = weights.active_terms()
    total = human.sum() * 0.0
    log: dict[str, float] = {}

    is_cross_finger = hasattr(mapper, "thumb_idx")
    thumb_idx = getattr(mapper, "thumb_idx", 0)

    for i in range(n_fingers):
        x = human[i].unsqueeze(0)  # (1, N, 3)
        finger_context = None
        if is_cross_finger and i == thumb_idx and human_paired is not None:
            finger_context = mapper.context_from_paired(human_paired)
        if finger_context is not None:
            mapped = mapper.forward_finger(x.reshape(-1, 3), i, context=finger_context).reshape(x.shape)
        else:
            mapped = mapper.forward_finger(x.reshape(-1, 3), i).reshape(x.shape)
        cloud = robot_cloud[i]

        if "p_chamfer" in active:
            term = partial_chamfer_loss(mapped, cloud)
            total = total + weights.p_chamfer * term
            log["p_chamfer"] = float(term.detach())

        if "dist" in active:
            term = distance_preservation_loss(mapped, x).mean()
            total = total + weights.dist * term
            log["dist"] = float(term.detach())

        if "motion" in active:
            delta = jitter_delta(x, generator=generator, base_scale=jitter_base, rand_scale=jitter_rand)
            term = local_motion_loss(
                mapped=mapped,
                source=x,
                mapper=mapper,
                finger=i,
                delta=delta,
                # human_rot is (F, N, 3, 3): finger on axis 0, frames on axis 1.
                human_rot=None if human_rot is None else human_rot[i].unsqueeze(0),
                robot_cloud=cloud,
                robot_cloud_rot=None if robot_cloud_rot is None else robot_cloud_rot[i],
                context=finger_context,
            )
            total = total + weights.motion * term
            log["motion"] = float(term.detach())

        if "align" in active and anchors is not None:
            h_anchor, r_anchor = anchors[0][i], anchors[1][i]
            term = alignment_loss(mapper, i, h_anchor, r_anchor).mean()
            total = total + weights.align * term
            log["align"] = float(term.detach())

    return total, log
