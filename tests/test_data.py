"""Tests for normalization, finger mapping, and hand kinematics.

The normalization tests matter more than they look: the paper's Appendix B
specifies a *single* max-axis range applied to all three axes, and switching to
per-axis normalization silently stretches the cloud. That would change every
downstream number while leaving the code looking reasonable.
"""

from __future__ import annotations

import numpy as np
import pytest

from anydexrt.hands import available_hands, get_hand
from anydexrt.kinematics import HandKinematics
from anydexrt.normalize import FingerNormalizer, HandNormalizer


# ------------------------------------------------------------------ normalize


def test_normalizer_uses_single_max_axis_range():
    """A cloud elongated on x must be scaled by x's range, not per axis.

    Concretely: the longest axis spans exactly 1.0 after scaling (that is what
    "the selected range is used to normalize" means), and the other axes keep
    their true proportion relative to it. Per-axis normalization would instead
    drive every axis to a span of 1.0 and destroy the aspect ratio.
    """
    rng = np.random.default_rng(0)
    pts = np.stack(
        [rng.uniform(0, 4, 500), rng.uniform(0, 1, 500), rng.uniform(0, 1, 500)], axis=-1
    ).astype(np.float32)
    norm = FingerNormalizer.fit(pts)
    assert norm.scale == pytest.approx(4.0, abs=0.2)

    out = norm.transform(pts)
    spans = out.max(axis=0) - out.min(axis=0)
    assert spans[0] == pytest.approx(1.0, abs=0.02), "longest axis spans 1.0"
    # y and z are a quarter of x's extent in the source, so a quarter here too.
    assert spans[1] == pytest.approx(0.25, abs=0.05)
    assert spans[2] == pytest.approx(0.25, abs=0.05)
    # The distinguishing property: not every axis was scaled to full range.
    assert spans[0] > 3 * spans[1]


def test_normalizer_preserves_shape():
    """No geometric distortion: the cloud's aspect ratio must survive."""
    rng = np.random.default_rng(1)
    pts = rng.standard_normal((400, 3)).astype(np.float32) * np.array([3.0, 1.0, 0.5])
    norm = FingerNormalizer.fit(pts)
    out = norm.transform(pts)
    before = np.linalg.norm(pts[0] - pts[1])
    after = np.linalg.norm(out[0] - out[1])
    assert after / before == pytest.approx(1.0 / norm.scale, rel=1e-5)


def test_normalizer_roundtrip():
    rng = np.random.default_rng(2)
    pts = rng.standard_normal((100, 3)).astype(np.float32)
    norm = FingerNormalizer.fit(pts)
    assert np.allclose(norm.inverse(norm.transform(pts)), pts, atol=1e-5)


def test_normalizer_centers_on_mean():
    rng = np.random.default_rng(3)
    pts = (rng.standard_normal((100, 3)) + 10.0).astype(np.float32)
    norm = FingerNormalizer.fit(pts)
    assert np.allclose(norm.transform(pts).mean(axis=0), 0.0, atol=1e-5)


def test_hand_normalizer_accepts_both_ranks():
    """(F, N, 3) clouds and (F, 3) single poses must both work."""
    fingers = ("thumb", "index", "middle")
    rng = np.random.default_rng(4)
    clouds = rng.standard_normal((3, 50, 3)).astype(np.float32)
    hn = HandNormalizer.fit(fingers, clouds)

    out3 = hn.transform(clouds)
    assert out3.shape == (3, 50, 3)

    single = clouds[:, 0, :]
    out2 = hn.transform(single)
    assert out2.shape == (3, 3)
    # A single pose must normalize identically either way.
    assert np.allclose(out2, out3[:, 0, :], atol=1e-5)


def test_hand_normalizer_inverse_roundtrip():
    fingers = ("a", "b")
    rng = np.random.default_rng(5)
    clouds = rng.standard_normal((2, 30, 3)).astype(np.float32)
    hn = HandNormalizer.fit(fingers, clouds)
    assert np.allclose(hn.inverse(hn.transform(clouds)), clouds, atol=1e-5)


def test_hand_normalizer_rejects_wrong_finger_count():
    hn = HandNormalizer.fit(("a", "b"), np.zeros((2, 10, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        hn.transform(np.zeros((3, 10, 3), dtype=np.float32))


def test_hand_normalizer_serialization_roundtrip():
    fingers = ("thumb", "index")
    rng = np.random.default_rng(6)
    clouds = rng.standard_normal((2, 20, 3)).astype(np.float32)
    hn = HandNormalizer.fit(fingers, clouds)
    restored = HandNormalizer.from_dict(hn.to_dict())
    assert np.allclose(restored.transform(clouds), hn.transform(clouds), atol=1e-6)


# ----------------------------------------------------------------------- hands


def test_available_hands_includes_four_and_five_fingered():
    hands = available_hands()
    assert "leap_hand" in hands
    assert get_hand("leap_hand").n_fingers == 4
    assert get_hand("wuji_hand").n_fingers == 5


def test_four_fingered_hand_omits_pinky():
    """The dropped-pinky choice must be visible, not implicit.

    Four-fingered hands exclude the pinky entirely -- both from the robot finger
    list and from the Manus keypoints requested -- which is what makes thumb-pinky
    pinch unavailable for them.
    """
    leap = get_hand("leap_hand")
    assert "pinky" not in leap.finger_names
    assert 24 not in leap.human_ids()  # the Manus pinky keypoint
    assert leap.human_ids() == [4, 9, 14, 19]

    wuji = get_hand("wuji_hand")
    assert "pinky" in wuji.finger_names
    assert 24 in wuji.human_ids()


def test_shadow_drives_are_fewer_than_mjcf_joints():
    """Shadow's tendon couplings mean a drive is not a joint.

    The four non-thumb F0 actuators each drive two joints at 0.5, so the model
    has 22 joints but only 18 drives. Conflating the two would make qpos vectors
    the wrong length.
    """
    spec = get_hand("shadow_hand")
    kin = HandKinematics(spec)
    assert spec.n_joints == 18
    assert kin.model.njnt == 22
    assert spec.drive_targets("rh_FFJ0") == {"rh_FFJ1": 0.5, "rh_FFJ2": 0.5}
    assert spec.drive_targets("rh_FFJ4") == {"rh_FFJ4": 1.0}


def test_coupled_drive_limits_are_tightened():
    """A coupled drive cannot push its children past their limits."""
    spec = get_hand("shadow_hand")
    kin = HandKinematics(spec)
    lo, hi = kin.limits()
    idx = spec.joint_names.index("rh_FFJ0")
    for child, mult in spec.drive_targets("rh_FFJ0").items():
        jid = kin.model.joint("rh_FFJ0" if False else child).id
        clo, chi = kin.model.jnt_range[jid]
        assert lo[idx] * mult >= clo - 1e-5
        assert hi[idx] * mult <= chi + 1e-5


@pytest.mark.parametrize("hand", available_hands())
def test_forward_kinematics_shapes_and_limits(hand):
    spec = get_hand(hand)
    kin = HandKinematics(spec)
    lo, hi = kin.limits()
    assert lo.shape == (spec.n_joints,)
    assert (hi > lo).all()

    mid = (lo + hi) / 2
    tips = kin.fingertip_positions(mid)
    assert tips.shape == (spec.n_fingers, 3)
    assert np.isfinite(tips).all()

    # Fingertips must be distinct points, not a degenerate stack.
    for i in range(spec.n_fingers):
        for j in range(i + 1, spec.n_fingers):
            assert np.linalg.norm(tips[i] - tips[j]) > 1e-4


@pytest.mark.parametrize("hand", available_hands())
def test_per_finger_cloud_isolates_each_finger(hand):
    """f_m's training target for finger i must depend only on finger i's joints.

    `fingertip_per_finger` holds every other finger at neutral while one finger
    varies. If it did not, finger i's cloud would carry the other fingers'
    motion and the per-finger mapper would be trained against a target it cannot
    possibly produce from a 3-vector input.
    """
    spec = get_hand(hand)
    kin = HandKinematics(spec)
    rng = np.random.default_rng(0)
    qpos, tips = kin.sample_fingertip_cloud(24, rng)
    assert qpos.shape == (24, spec.n_joints)
    assert tips.shape == (24, spec.n_fingers, 3)

    # Perturb finger 0's joints only; every other finger's target must hold.
    sl = spec.joint_slice(spec.finger_names[0])
    perturbed = qpos.copy()
    perturbed[:, sl] += 0.3

    before = kin.fingertip_per_finger(qpos)
    after = kin.fingertip_per_finger(perturbed)
    assert not np.allclose(before[:, 0], after[:, 0]), "finger 0 should have moved"
    for i in range(1, spec.n_fingers):
        assert np.allclose(before[:, i], after[:, i]), (
            f"finger {i}'s per-finger target moved when only finger 0 changed"
        )


def test_sample_fingertip_cloud_respects_dof_count():
    for hand in available_hands():
        spec = get_hand(hand)
        kin = HandKinematics(spec)
        rng = np.random.default_rng(1)
        qpos, tips = kin.sample_fingertip_cloud(8, rng)
        assert qpos.shape == (8, spec.n_joints)
        assert tips.shape == (8, spec.n_fingers, 3)


def test_unknown_hand_rejected():
    with pytest.raises(KeyError):
        get_hand("not_a_hand")
