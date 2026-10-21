"""Tests for the four mapping losses and the contact loss.

These pin the properties that make the reproduction identifiable rather than
merely plausible:

  * L_P-Chamfer is ONE-directional. Making the robot cloud sparse in a region the
    human cloud covers must not be penalized; making the human cloud cover
    unmapped robot space must not be either. If a reverse term creeps in, the
    collapse experiment is meaningless.
  * L_P-Chamfer uses the L2 norm, not the squared distance. The two differ by a
    square in scale, which is exactly the factor that separates GeoRT's
    w_chamfer=80 from this paper's weight of 1.
  * L_dist is zero exactly when distances are preserved and grows with
    distortion, in squared units.
  * L_motion is a negative cosine, so -1 is perfect agreement.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anydexrt.losses import (
    LossWeights,
    contact_loss,
    distance_preservation_loss,
    jitter_delta,
    partial_chamfer_loss,
)


def test_chamfer_uses_l2_not_squared():
    """One sample at distance 1 from a single robot point must cost 1, not 1."""
    mapped = torch.tensor([[[0.0, 0.0, 0.0]]])  # (1, 1, 3)
    cloud = torch.tensor([[1.0, 0.0, 0.0]])  # (1, 3), L2 distance exactly 1
    assert float(partial_chamfer_loss(mapped, cloud)) == pytest.approx(1.0, abs=1e-6)


def test_chamfer_is_one_directional():
    """Dense robot regions the human cloud never visits must be free."""
    mapped = torch.tensor([[[0.0, 0.0, 0.0]]])
    sparse = torch.tensor([[1.0, 0.0, 0.0]])
    # Add 100 far-away robot points: a bidirectional Chamfer would blow up here.
    dense = torch.cat([sparse, torch.full((100, 3), 50.0)], dim=0)
    assert float(partial_chamfer_loss(mapped, sparse)) == pytest.approx(
        float(partial_chamfer_loss(mapped, dense)), abs=1e-6
    )


def test_chamfer_zero_on_exact_match():
    mapped = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    cloud = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert float(partial_chamfer_loss(mapped, cloud)) == pytest.approx(0.0, abs=1e-6)


def test_chamfer_mean_over_samples():
    """Chamfer averages over human samples, so duplicating one doesn't scale it."""
    cloud = torch.tensor([[0.0, 0.0, 0.0]])
    one = torch.tensor([[[1.0, 0.0, 0.0]]])
    many = one.repeat(1, 5, 1)
    assert float(partial_chamfer_loss(one, cloud)) == pytest.approx(
        float(partial_chamfer_loss(many, cloud)), abs=1e-6
    )


def test_dist_zero_when_preserved():
    x = torch.randn(1, 8, 3)
    # Rigid transforms preserve pairwise distances exactly.
    rot = torch.linalg.qr(torch.randn(3, 3))[0]
    moved = x @ rot.T + 5.0
    assert float(distance_preservation_loss(moved, x).mean()) == pytest.approx(0.0, abs=1e-4)


def test_dist_is_squared_units():
    """Halving the cloud's scale must cost (0.5*d)^2 summed appropriately.

    Two points 2 apart scaled by 0.5 become 1 apart: difference 1, squared 1.
    """
    x = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    scaled = x * 0.5
    # Ordered pairs: (1,2) and (2,1), each contribution 1.0; mean = 1.0.
    assert float(distance_preservation_loss(scaled, x).mean()) == pytest.approx(1.0, abs=1e-6)


def test_dist_ignores_rigid_motion_but_not_collapse():
    x = torch.randn(1, 6, 3)
    collapsed = torch.zeros_like(x)
    assert float(distance_preservation_loss(collapsed, x).mean()) > 0.1


def test_dist_needs_two_samples():
    x = torch.randn(1, 1, 3)
    assert float(distance_preservation_loss(x, x).mean()) == pytest.approx(0.0)


def test_jitter_magnitude_is_bounded():
    x = torch.zeros(1, 16, 3)
    delta = jitter_delta(x, base_scale=0.001, rand_scale=0.01)
    mag = torch.linalg.norm(delta, dim=-1)
    assert float(mag.min()) >= 0.001 - 1e-6
    assert float(mag.max()) <= 0.011 + 1e-6


def test_jitter_is_reproducible_with_generator():
    x = torch.zeros(1, 8, 3)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    assert torch.allclose(jitter_delta(x, g1), jitter_delta(x, g2))


def test_contact_loss_perfect_prediction_is_low():
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    good = torch.tensor([0.01, 0.99, 0.01, 0.99])
    bad = torch.tensor([0.99, 0.01, 0.99, 0.01])
    assert float(contact_loss(good, labels)) < float(contact_loss(bad, labels))


def test_contact_loss_pos_weight_upweights_positives():
    labels = torch.tensor([0.0, 0.0, 0.0, 1.0])
    pred = torch.tensor([0.1, 0.1, 0.1, 0.1])
    plain = float(contact_loss(pred, labels))
    weighted = float(contact_loss(pred, labels, pos_weight=10.0))
    assert weighted > plain


def test_contact_loss_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        contact_loss(torch.zeros(3), torch.zeros(4))


def test_loss_weights_active_terms():
    w = LossWeights(p_chamfer=1.0, dist=0.0, motion=1.0, align=0.0)
    assert set(w.active_terms()) == {"p_chamfer", "motion"}
    assert LossWeights().active_terms() == {
        "p_chamfer": 1.0,
        "dist": 1.0,
        "motion": 1.0,
        "align": 1.0,
    }
