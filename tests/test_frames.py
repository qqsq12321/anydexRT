"""Tests for local coordinate frames.

These exist because the human-side frame of T(x) is the single most
consequential unspecified detail in the reproduction. Eq. (3) applies T^-1 to
both sides of the inner product, and the paper defines only the robot half. A
mismatched pair of bases does not merely add noise -- it drives the inner
product negative, so the term actively fights the objective it is meant to
support. That failure mode is silent: the loss curve still descends.

The tests below pin the algebra that any correct frame pair must satisfy.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anydexrt.frames import (
    expand_rotation,
    nearest_neighbor_rotation,
    quat_wxyz_to_matrix,
    rotate_into_local,
    rotation_angle_deg,
    y_axis_rotation,
)


# ------------------------------------------------------------- rotate_into_local


def test_identity_frame_is_a_noop():
    v = torch.randn(5, 7, 3)
    rot = torch.eye(3).expand(5, 7, 3, 3)
    assert torch.allclose(rotate_into_local(v, rot), v, atol=1e-6)


def test_frame_matching_rank_is_required():
    """A singleton batch dim must be squeezed, not broadcast.

    (1, N, 3) against (N, 3, 3) fails in matplotlib-style broadcast because
    matmul reads the leading 1 as a batch size. Raising beats silently
    aligning the wrong axes.
    """
    v = torch.randn(1, 6, 3)
    rot = torch.eye(3).expand(6, 3, 3)
    with pytest.raises(ValueError, match="does not match"):
        rotate_into_local(v, rot)


def test_local_frame_cancels_a_common_rotation():
    """(R T)^-1 (R v) == T^-1 v for any R: the whole point of Eq. (3)."""
    v = torch.randn(3, 9, 3)
    T = torch.linalg.qr(torch.randn(3, 3))[0].expand(3, 9, 3, 3).clone()
    R = y_axis_rotation(72.0)

    plain = rotate_into_local(v, T)
    rotated = rotate_into_local(v @ R.T, R @ T)
    assert torch.allclose(plain, rotated, atol=1e-5)


def test_shapes_of_frame_are_validated():
    with pytest.raises(ValueError):
        rotate_into_local(torch.randn(4, 2), torch.eye(3))  # last dim 2
    with pytest.raises(ValueError):
        rotate_into_local(torch.randn(4, 3), torch.eye(4))  # not 3x3


# ------------------------------------------------------- nearest_neighbor_rotation


def test_nearest_neighbor_picks_the_closest_frame():
    cloud = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    rots = torch.stack([torch.eye(3), torch.eye(3) * 2])
    query = torch.tensor([[[9.5, 0.0, 0.0], [0.5, 0.0, 0.0]]])  # (1, 2, 3)
    out = nearest_neighbor_rotation(query, cloud, rots)
    assert torch.allclose(out[0, 0], rots[1])
    assert torch.allclose(out[0, 1], rots[0])


def test_nearest_neighbor_validates_cloud_shape():
    with pytest.raises(ValueError):
        nearest_neighbor_rotation(
            torch.randn(1, 3, 3), torch.randn(5, 4), torch.randn(5, 3, 3)
        )
    with pytest.raises(ValueError):
        nearest_neighbor_rotation(
            torch.randn(1, 3, 3), torch.randn(5, 3), torch.randn(5, 2, 2)
        )


# --------------------------------------------------------------- quaternions


def test_quat_identity_and_y_rotation():
    assert np.allclose(
        quat_wxyz_to_matrix(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
        np.eye(3),
        atol=1e-6,
    )
    half = np.pi / 4
    q = np.array([np.cos(half), 0.0, np.sin(half), 0.0], dtype=np.float32)
    assert np.allclose(
        quat_wxyz_to_matrix(q), y_axis_rotation(90.0).numpy(), atol=1e-5
    )


def test_quat_outputs_are_proper_rotations():
    q = np.random.default_rng(0).standard_normal((6, 4)).astype(np.float32)
    m = quat_wxyz_to_matrix(q)
    assert m.shape == (6, 3, 3)
    assert np.allclose(np.linalg.det(m), 1.0, atol=1e-4)
    assert np.allclose(m @ m.transpose(0, 2, 1), np.eye(3), atol=1e-4)


def test_quat_normalization_is_applied():
    """Unnormalized quaternions must still yield a rotation matrix."""
    q = np.array([[2.0, 2.0, 0.0, 0.0]], dtype=np.float32)
    m = quat_wxyz_to_matrix(q)
    assert np.allclose(np.linalg.det(m), 1.0, atol=1e-5)


def test_quat_rejects_wrong_last_dim():
    with pytest.raises(ValueError):
        quat_wxyz_to_matrix(np.zeros((4, 3), dtype=np.float32))


# --------------------------------------------------------------- utilities


def test_expand_rotation_matches_manual_broadcast():
    rot = y_axis_rotation(30.0)
    out = expand_rotation(rot, 2, 5)
    assert out.shape == (2, 5, 3, 3)
    assert torch.allclose(out[1, 3], rot)


def test_expand_rotation_rejects_non_single():
    with pytest.raises(ValueError):
        expand_rotation(torch.eye(3).expand(4, 3, 3), 1, 1)


def test_rotation_angle_reports_the_right_value():
    for deg in (0.0, 45.0, 90.0, 180.0):
        rot = y_axis_rotation(deg)
        assert float(rotation_angle_deg(rot)) == pytest.approx(deg, abs=1e-3)


def test_y_axis_rotation_is_orthonormal():
    rot = y_axis_rotation(37.0)
    assert torch.allclose(rot @ rot.T, torch.eye(3), atol=1e-6)
    assert float(torch.linalg.det(rot)) == pytest.approx(1.0, abs=1e-6)
