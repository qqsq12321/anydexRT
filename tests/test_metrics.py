"""Self-tests for GMC/LMC.

The metrics have no public reference implementation in this workspace, so the
only way to trust them is to feed them inputs whose answer is known
analytically:

  * identical trajectories            -> cosine exactly +1, score 100
  * reversed trajectory               -> cosine exactly -1, score -100 (unclipped)
  * deterministic global rotation applied identically to both sides
                                      -> directions rotate together, score stays 100
  * a mapping that collapses every frame to one point
                                      -> displacement ~0 everywhere, so every
                                         sample is filtered out and the metric
                                         must raise rather than report 0
  * a random (uncorrelated) mapping   -> mean cosine ~0

The last two are the ones that matter for reading the paper's 4.1. A collapsed
mapping has no measurable displacement at all, so 4.1 cannot come from a fully
degenerate point mass -- it must come from a mapping that still moves but whose
motion is uncorrelated with the human's. `test_random_mapping_is_near_zero`
pins that reading down.
"""

from __future__ import annotations

import numpy as np
import torch
import pytest

from anydexrt.frames import (
    expand_rotation,
    quat_wxyz_to_matrix,
    rotate_into_local,
    y_axis_rotation,
)
from anydexrt.metrics import apply_global_perturbation, motion_consistency


def _trajectory(seed: int = 0, frames: int = 64) -> torch.Tensor:
    """A smooth, non-degenerate 3D trajectory."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, frames, dtype=np.float32)
    base = np.stack(
        [np.sin(t), np.cos(1.3 * t), 0.4 * np.sin(0.7 * t)], axis=-1
    ).astype(np.float32)
    return torch.from_numpy(base + 0.01 * rng.standard_normal(base.shape).astype(np.float32))


def test_identical_trajectories_score_100():
    traj = _trajectory()
    res = motion_consistency(traj, traj.clone())
    assert res.mean_cosine == pytest.approx(1.0, abs=1e-5)
    assert res.score == pytest.approx(100.0, abs=1e-3)


def test_reversed_trajectory_is_negative():
    traj = _trajectory()
    res = motion_consistency(traj, traj.flip(0))
    # Not exactly -1 because the reversed samples pair different displacements,
    # but every dot product is negative.
    assert res.mean_cosine < 0
    res_clipped = motion_consistency(traj, traj.flip(0), clip_negative=True)
    assert res_clipped.score >= 0


def test_common_rotation_does_not_change_score():
    """Rotating everything -- data and the frames attached to it -- is a no-op.

    If R multiplies both the positions and their local frames, every local
    direction is unchanged, so LMC is exactly what it was before.
    """
    traj = _trajectory()
    robot = traj * 1.5
    n_frames = traj.shape[0]
    rot = y_axis_rotation(37.0)
    eye_frames = expand_rotation(torch.eye(3), 1, n_frames)
    rot_frames = expand_rotation(rot, 1, n_frames)

    base = motion_consistency(traj.unsqueeze(0), robot.unsqueeze(0), local=False)
    assert base.score == pytest.approx(100.0, abs=1e-3)

    base_local = motion_consistency(
        traj.unsqueeze(0), robot.unsqueeze(0),
        human_rot=eye_frames, robot_rot=eye_frames, local=True,
    )

    # Everything rotated together: positions by R, frames by R.
    rotated_h, rotated_h_frames = apply_global_perturbation(
        traj.unsqueeze(0), eye_frames, rot
    )
    rotated_r, rotated_r_frames = apply_global_perturbation(
        robot.unsqueeze(0), eye_frames, rot
    )
    both = motion_consistency(
        rotated_h, rotated_r,
        human_rot=rotated_h_frames, robot_rot=rotated_r_frames, local=True,
    )
    assert both.score == pytest.approx(base_local.score, abs=1e-3)
    assert both.score == pytest.approx(100.0, abs=1e-3)


def test_global_frame_degrades_under_frame_misalignment():
    """A global frame misalignment must degrade GMC, monotonically in the angle.

    The mean cosine is NOT cos(theta): each displacement d is compared against
    R*d, and the angle between those depends on where d sits relative to the
    rotation axis, so the exact value is data-dependent. What is guaranteed is
    that rotating further costs more. That is the property the paper's
    calibration-sensitivity figure relies on, and the one the local-frame fix
    removes.
    """
    traj = _trajectory().unsqueeze(0)
    robot = traj * 1.5  # a direction-preserving map: GMC should be perfect

    gmc_clean = motion_consistency(traj, robot, local=False)
    assert gmc_clean.score == pytest.approx(100.0, abs=1e-3)

    scores = []
    for deg in (0.0, 45.0, 90.0):
        if deg == 0.0:
            scores.append(gmc_clean.score)
            continue
        rot = y_axis_rotation(deg)
        rot4 = expand_rotation(rot, 1, traj.shape[1])
        human_bad, _ = apply_global_perturbation(traj, rot4, rot)
        gmc_bad = motion_consistency(human_bad, robot, local=False)
        scores.append(gmc_bad.score)

    assert scores == sorted(scores, reverse=True), f"expected monotone decay, got {scores}"
    assert scores[2] < scores[1] < 100.0
    # A 90-degree rotation about y zeroes the component of motion along x and z
    # for a trajectory dominated by those axes, so agreement must collapse well
    # below the clean score.
    assert scores[2] < 60.0, f"90 deg misalignment should hurt badly, got {scores[2]:.1f}"


def test_local_frames_survive_frame_misalignment():
    """The fix, stated precisely: local agreement is invariant when each side is
    read in the frame that travelled with it.

    The perturbation rotates the human input while the robot output stays in the
    world frame -- the misalignment a glove calibration error actually produces.
    Measured globally, the two displacement sets are now separated by that
    rotation. Measured locally, with the human frame rotated by the same amount,
    the rotation cancels exactly: (R)^-1 (R d) = d. Frames are built explicitly
    here so the algebra is visible rather than implied.
    """
    traj = _trajectory().unsqueeze(0)
    robot = traj * 1.5  # direction-preserving map, so any loss is the frame's fault
    n_frames = traj.shape[1]

    rot = y_axis_rotation(90.0)
    eye_frames = expand_rotation(torch.eye(3), 1, n_frames)
    rot_frames = expand_rotation(rot, 1, n_frames)

    human_bad = traj @ rot.T  # human input rotated by the calibration error
    gmc_bad = motion_consistency(human_bad, robot, local=False)

    # Identical data and frames, read locally: the rotation cancels per side.
    lmc_matched = motion_consistency(
        human_bad, robot, human_rot=rot_frames, robot_rot=eye_frames, local=True
    )
    # Sanity: with no perturbation at all, local and global agree at 100.
    lmc_plain = motion_consistency(
        traj, robot, human_rot=eye_frames, robot_rot=eye_frames, local=True
    )

    assert lmc_plain.score == pytest.approx(100.0, abs=1e-2)
    assert lmc_matched.score == pytest.approx(100.0, abs=1e-2), (
        "when each side is read in its own frame the rotation cancels"
    )
    assert gmc_bad.score < 60.0, (
        "the same perturbation must damage the global measure, which is the "
        "entire justification for local frames"
    )


def test_collapse_is_filtered_not_scored_zero():
    """A true point-mass collapse has no displacement, so it must raise."""
    traj = _trajectory()
    collapsed = traj[:1].expand_as(traj).clone()
    with pytest.raises(ValueError):
        motion_consistency(traj, collapsed, min_displacement=1e-5)


def test_random_mapping_is_near_zero():
    """An uncorrelated mapping lands near 0, which is how 4.1 is read."""
    traj = _trajectory()
    rng = np.random.default_rng(123)
    noise = torch.from_numpy(
        np.cumsum(rng.standard_normal(traj.shape).astype(np.float32), axis=0)
    )
    res = motion_consistency(traj, noise)
    assert abs(res.mean_cosine) < 0.2, f"expected ~0, got {res.mean_cosine}"


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        motion_consistency(_trajectory(), _trajectory(frames=32))


def test_single_frame_rejected():
    with pytest.raises(ValueError):
        motion_consistency(_trajectory(frames=1), _trajectory(frames=1))


def test_quat_identity_conversion():
    ident = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    m = quat_wxyz_to_matrix(ident)
    assert np.allclose(m, np.eye(3), atol=1e-6)


def test_quat_90deg_y_matches_rotation_matrix():
    half = np.pi / 4
    q = np.array([np.cos(half), 0.0, np.sin(half), 0.0], dtype=np.float32)
    m = quat_wxyz_to_matrix(q)
    assert np.allclose(m, y_axis_rotation(90.0).numpy(), atol=1e-5)


def test_quat_batch_shapes():
    q = np.random.default_rng(0).standard_normal((5, 7, 4)).astype(np.float32)
    m = quat_wxyz_to_matrix(q)
    assert m.shape == (5, 7, 3, 3)
    # Every output must be a proper rotation.
    det = np.linalg.det(m)
    assert np.allclose(det, 1.0, atol=1e-4)


def test_rotate_into_local_identity_is_noop():
    delta = _trajectory()
    rot = expand_rotation(torch.eye(3), 1, delta.shape[0])
    out = rotate_into_local(delta.unsqueeze(0), rot)
    assert torch.allclose(out.squeeze(0), delta, atol=1e-6)


def test_single_trajectory_accepted():
    """(N, 3) input must be promoted rather than rejected."""
    traj = _trajectory()
    res = motion_consistency(traj, traj.clone())
    assert res.score == pytest.approx(100.0, abs=1e-3)


def test_rotate_into_local_inverts_global_rotation():
    """Local-frame directions must be invariant to a common global rotation.

    Rotating both the vector and its frame by the same R must leave the local
    representation unchanged: (R T)^-1 (R v) = T^-1 v.
    """
    delta = _trajectory().unsqueeze(0)
    rot = y_axis_rotation(53.0)
    rot4 = expand_rotation(rot, 1, delta.shape[1])

    delta_rot = delta @ rot.T
    local_plain = rotate_into_local(delta, rot4)
    local_rotated = rotate_into_local(delta_rot, expand_rotation(rot, 1, delta.shape[1]) @ rot4)
    # The frame must rotate too, otherwise this is a different test.
    local_matched = rotate_into_local(delta_rot, torch.einsum("ij,...jk->...ik", rot, rot4))
    assert torch.allclose(local_plain, local_matched, atol=1e-5)
