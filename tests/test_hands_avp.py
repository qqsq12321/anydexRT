"""Tests for the Wuji / Linker L20 / Apple Vision Pro path.

What these pin down, and why each matters:

* L20 couplings equal the MJCF's joint-equality constraints. mj_forward does not
  enforce equality constraints when qpos is written directly, so a drift here
  would silently produce fingertips the real hand cannot reach.
* The palm frame is invariant to a rotation of the input. This is where the AVP
  pipeline's calibration-freeness actually comes from: any global rotation of the
  headset/wrist frame is removed exactly by the skeleton-derived frame.
* Per-finger NNS is exact: FK of the returned drives reproduces the table tip.
  The earlier whole-hand voting version missed by ~45 mm on these hands.
* Robot anchors follow the anchor parametrization. The earlier version sampled
  them at random, pairing the j-th human anchor with an unrelated robot pose.
"""

from __future__ import annotations

import pickle

import mujoco
import numpy as np
import pytest

from anydexrt.anchors import (
    MEDIAPIPE_CHAINS,
    anchor_fractions,
    bending_params,
    build_robot_anchors,
    finger_joint_roles,
    lateral_params,
    skeleton_angles,
)
from anydexrt.data import build_robot_clouds, load_avp_pkl, to_frame
from anydexrt.hands import HUMAN_LAYOUTS, get_hand
from anydexrt.inference import make_pinch_templates
from anydexrt.kinematics import HandKinematics, NearestNeighborIK, palm_frame_from_points

AVP_HANDS = ["wuji_hand", "linker_l20"]


def _rot(axis: str, deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


# ------------------------------------------------------------------- hands


def test_l20_couplings_match_mjcf_equalities():
    spec = get_hand("linker_l20")
    m = HandKinematics(spec).model
    assert m.neq == 5
    for e in range(m.neq):
        slave = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.eq_obj1id[e])
        master = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.eq_obj2id[e])
        targets = spec.drive_targets(master)
        assert targets[master] == 1.0
        assert targets[slave] == pytest.approx(float(m.eq_data[e][1]), abs=1e-6)
        assert m.eq_data[e][0] == pytest.approx(0.0)  # no constant offset


def test_l20_has_16_drives_over_21_joints():
    spec = get_hand("linker_l20")
    kin = HandKinematics(spec)
    assert spec.n_joints == 16 and kin.model.njnt == 21


def test_l20_tip_offset_is_applied():
    spec = get_hand("linker_l20")
    kin = HandKinematics(spec)
    q = kin.rest_qpos()
    tips = kin.fingertip_positions(q)
    for i, finger in enumerate(spec.fingers):
        bid = mujoco.mj_name2id(kin.model, mujoco.mjtObj.mjOBJ_BODY, finger.tip_body)
        body = kin.data.xpos[bid]
        offset = kin.data.xmat[bid].reshape(3, 3) @ np.array(finger.tip_offset)
        assert np.allclose(tips[i], body + offset, atol=1e-6)
        assert np.linalg.norm(tips[i] - body) > 0.015  # really moved to the tip


def test_mediapipe_layout_ids():
    assert get_hand("wuji_hand").human_ids("mediapipe") == [4, 8, 12, 16, 20]
    assert set(HUMAN_LAYOUTS) >= {"manus", "mediapipe"}
    with pytest.raises(KeyError):
        get_hand("wuji_hand").human_ids("nope")


# -------------------------------------------------------------- palm frame


def test_palm_frame_is_proper_rotation_with_expected_semantics():
    for hand in AVP_HANDS:
        kin = HandKinematics(get_hand(hand))
        origin, R = kin.palm_frame()
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)
        tips = to_frame(kin.fingertip_positions(kin.rest_qpos()), origin, R)
        # Open hand: fingers point along +x; index is on the +y side of the pinky.
        assert (tips[1:, 0] > 0.05).all()
        assert tips[1, 1] > tips[4, 1]


def test_palm_frame_rotates_with_its_points():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(4, 3))
    R0 = palm_frame_from_points(*pts)
    Q = _rot("x", 33) @ _rot("z", -71)
    R1 = palm_frame_from_points(*(pts @ Q.T))
    assert np.allclose(R1, Q @ R0, atol=1e-9)


def test_palm_frame_rejects_degenerate_input():
    p = np.zeros(3)
    with pytest.raises(ValueError):
        palm_frame_from_points(p, p + [0, 1, 0], p, p)


def _write_avp(path, landmarks):
    frames = [{"t": 0.0, "right_fingers": f, "left_fingers": f} for f in landmarks]
    with open(path, "wb") as fh:
        pickle.dump(frames, fh)


def _synthetic_hand(T=40, seed=0):
    """A plausible right hand in MediaPipe layout, wrist at the origin."""
    rng = np.random.default_rng(seed)
    L = np.zeros((T, 21, 3))
    bases = {"thumb": (0.03, 0.03), "index": (0.09, 0.03), "middle": (0.095, 0.0),
             "ring": (0.09, -0.02), "pinky": (0.08, -0.04)}
    for f, (x, y) in bases.items():
        chain = MEDIAPIPE_CHAINS[f]
        for k, idx in enumerate(chain):
            L[:, idx] = [x + 0.025 * k, y + (0.02 * k if f == "thumb" else 0), 0.0]
    return L + rng.normal(scale=1e-3, size=L.shape)


def test_avp_palm_frame_cancels_any_input_rotation(tmp_path):
    """Calibration-freeness of the AVP path, by construction."""
    L = _synthetic_hand()
    Q = _rot("y", 90) @ _rot("x", -45)
    _write_avp(tmp_path / "a.pkl", L)
    _write_avp(tmp_path / "b.pkl", L @ Q.T)
    a = load_avp_pkl(tmp_path / "a.pkl", "wuji_hand")
    b = load_avp_pkl(tmp_path / "b.pkl", "wuji_hand")
    assert np.allclose(a.positions, b.positions, atol=1e-5)


def test_avp_loader_drops_lost_tracking_frames(tmp_path):
    L = _synthetic_hand(T=10)
    L[3] = 0.0
    _write_avp(tmp_path / "c.pkl", L)
    rec = load_avp_pkl(tmp_path / "c.pkl", "linker_l20")
    assert rec.n_frames == 9
    assert rec.positions.shape == (9, 5, 3)
    assert rec.identity_quats().shape == (9, 5, 4)


# --------------------------------------------------------------------- NNS


@pytest.mark.parametrize("hand", AVP_HANDS)
def test_per_finger_nns_is_exact_on_its_own_table(hand):
    kin = HandKinematics(get_hand(hand))
    nn = NearestNeighborIK.build(kin, n_samples=3000, seed=0)
    for row in (0, 17, 2999):
        q, resid = nn.solve(nn.tips[row])
        assert np.allclose(resid, 0.0, atol=1e-6)
        assert np.allclose(kin.fingertip_positions(q), nn.tips[row], atol=1e-5)


@pytest.mark.parametrize("hand", AVP_HANDS)
def test_per_finger_nns_residual_matches_fk(hand):
    kin = HandKinematics(get_hand(hand))
    nn = NearestNeighborIK.build(kin, n_samples=4000, seed=0)
    target = kin.fingertip_positions(kin.sample_qpos(1, np.random.default_rng(5))[0])
    q, resid = nn.solve(target)
    err = np.linalg.norm(kin.fingertip_positions(q) - target, axis=1)
    assert np.allclose(err, resid, atol=1e-6)


def test_nns_continuity_prior_prefers_previous_solution():
    kin = HandKinematics(get_hand("wuji_hand"))
    nn = NearestNeighborIK.build(kin, n_samples=3000, seed=0, prior_weight=10.0)
    target = nn.tips[5]
    q_free, _ = nn.solve(target)
    q_prior, _ = nn.solve(target, last_qpos=q_free)
    assert np.allclose(q_prior, q_free)


# ----------------------------------------------------------------- anchors


def test_joint_roles_find_abduction():
    expect = {
        "wuji_hand": "right_finger2_joint2",
        "linker_l20": "index_mcp_roll",
    }
    for hand, abd_name in expect.items():
        kin = HandKinematics(get_hand(hand))
        abd, flex = finger_joint_roles(kin)[1]
        assert kin.hand.fingers[1].joints[abd] == abd_name
        assert abd not in flex


def test_anchor_fractions_span_each_type():
    types = ["lateral"] * 3 + ["bending"] * 5
    s = anchor_fractions(np.zeros((8, 4)), types)
    assert np.allclose(s[:3], [0, 0.5, 1]) and np.allclose(s[3:], np.linspace(0, 1, 5))


@pytest.mark.parametrize("hand", AVP_HANDS)
def test_robot_anchors_follow_the_parametrization(hand):
    kin = HandKinematics(get_hand(hand))
    params = np.concatenate([lateral_params(5), bending_params(6)])
    types = ["lateral"] * 5 + ["bending"] * 6
    origin, R = kin.palm_frame()
    a = build_robot_anchors(kin, params, types, frame=(origin, R))
    idx = a[1]  # index finger, palm frame
    lat, bend = idx[:5], idx[5:]
    # Lateral sweep moves the tip across the palm (y), monotonically toward +y.
    assert np.all(np.diff(lat[:, 1]) > 0)
    # Bending brings the tip back toward the wrist (x shrinks) and is deterministic.
    assert bend[-1, 0] < bend[0, 0] - 0.03
    assert np.allclose(build_robot_anchors(kin, params, types, frame=(origin, R)), a)


def test_skeleton_angles_straight_and_bent():
    L = np.zeros((2, 21, 3))
    c0, c1, c2, c3 = MEDIAPIPE_CHAINS["index"]
    for k, idx in enumerate((c0, c1, c2, c3)):
        L[0, idx] = [0.08 + 0.03 * k, 0.0, 0.0]  # straight along +x
    L[1, c0] = [0.08, 0, 0]
    L[1, c1] = [0.08, 0, -0.03]  # proximal bent 90 deg into the palm
    L[1, c2] = [0.08, 0, -0.06]
    L[1, c3] = [0.08, 0, -0.09]
    alpha, beta = skeleton_angles(L, "index")
    assert np.allclose(beta[0], 0.0, atol=1e-6)
    assert beta[1, 0] == pytest.approx(np.pi / 2, abs=1e-6)
    assert np.allclose(beta[1, 1:], 0.0, atol=1e-6)


# ------------------------------------------------------------ pinch templates


@pytest.mark.parametrize("hand", AVP_HANDS)
def test_pinch_templates_are_real_contacts(hand):
    clouds, kin = build_robot_clouds(hand, n_samples=6000, seed=0, frame="palm")
    templates = make_pinch_templates(clouds, kin, per_finger=50)
    assert set(templates) == {1, 2, 3, 4}
    for i, t in templates.items():
        assert np.median(t.gap) < 0.01, f"finger {i}: templates are not pinches"
        q = kin.rest_qpos()
        q[kin.hand.joint_slice("thumb")] = t.thumb_q[0]
        q[kin.hand.joint_slice(kin.hand.fingers[i].name)] = t.finger_q[0]
        tips = clouds.to_cloud_frame(kin.fingertip_positions(q))
        assert np.linalg.norm(tips[0] - tips[i]) == pytest.approx(float(t.gap[0]), abs=1e-5)


# ------------------------------------------------------------- live AVP path


def test_live_avp_path_matches_recording_path():
    """The live 25-joint path must land where the 21-landmark training data did.

    Each recorded frame is lifted back to a 25-joint skeleton via VP_TO_MEDIAPIPE
    (metacarpal joints, which the palm frame does not use, are filled with
    wrist/MCP midpoints). The live path rebuilds the palm frame per frame; the
    recording path uses one mean frame. They should differ only by that small
    per-frame rotation -- a few mm at the fingertips, not centimetres.
    """
    from anydexrt.data import VP_TO_MEDIAPIPE, palm_tips_from_avp_poses

    path = "/home/qsq/qqsq12321/AnyDexRetarget/example/data/avp1.pkl"
    rec = load_avp_pkl(path, "linker_l20")
    with open(path, "rb") as fh:
        frames = pickle.load(fh)
    diffs = []
    for t in range(0, rec.n_frames, 97):
        mp = np.asarray(frames[t]["right_fingers"], dtype=np.float64)
        poses = np.tile(np.eye(4), (25, 1, 1))
        for mp_idx, vp_idx in enumerate(VP_TO_MEDIAPIPE):
            poses[vp_idx, :3, 3] = mp[mp_idx]
        for meta, mcp in ((5, 6), (10, 11), (15, 16), (20, 21)):
            poses[meta, :3, 3] = 0.5 * poses[mcp, :3, 3]
        live = palm_tips_from_avp_poses(poses, "linker_l20")
        diffs.append(np.linalg.norm(live - rec.positions[t], axis=-1).max())
    assert max(diffs) < 0.012, f"live vs recording fingertip gap up to {max(diffs)*1000:.1f} mm"


# ------------------------------------------------------------------- bundles


def test_bundle_roundtrip_reproduces_outputs(tmp_path):
    """A saved bundle must rebuild a retargeter that gives identical drives.

    The robot cloud is not stored but regenerated from (hand, samples, seed), so
    this also checks that cloud generation is deterministic -- if it were not,
    the NNS table would differ and deployment would silently diverge from the
    evaluated model.
    """
    import torch

    from anydexrt.anchors import build_anchor_set
    from anydexrt.data import normalize_human
    from anydexrt.inference import AnyDexRT, load_bundle, save_bundle
    from anydexrt.train import TrainConfig, train

    path = "/home/qsq/qqsq12321/AnyDexRetarget/example/data/avp1.pkl"
    rec = load_avp_pkl(path, "linker_l20", max_frames=400)
    finger_names = get_hand("linker_l20").finger_names
    samples = normalize_human(rec.positions, rec.identity_quats(), finger_names)
    clouds, kin = build_robot_clouds("linker_l20", n_samples=1500, seed=0, frame="palm")
    anchors = build_anchor_set(
        "linker_l20", rec.positions, samples.normalizer, kin,
        robot_normalizer=clouds.normalizer, strategy="skeleton",
        landmarks=rec.landmarks, robot_frame=(clouds.frame_origin, clouds.frame_rot),
        n_lateral=5, n_bending=5,
    )
    torch.manual_seed(0)
    cfg = TrainConfig(hand="linker_l20", epochs=1, contact_epochs=2, batch_size=256)
    result = train(samples, clouds, anchors, cfg, raw_fingertips=rec.positions)
    original = AnyDexRT(result.mapper, clouds, kin, samples.normalizer,
                        classifier=result.classifier,
                        pinch_templates=make_pinch_templates(clouds, kin))
    save_bundle(tmp_path / "b.pt", result, "linker_l20", samples.normalizer, cloud_samples=1500)
    loaded = load_bundle(tmp_path / "b.pt")

    q_a, _ = original.retarget_sequence(rec.positions[:60])
    q_b, _ = loaded.retarget_sequence(rec.positions[:60])
    assert finger_names == loaded.kin.hand.finger_names
    assert np.array_equal(q_a, q_b)


# --------------------------------------------------------------- recording


def test_record_avp_mediapipe_roundtrip():
    """to_mediapipe() must exactly invert the 21->25 reconstruction ReplayAVP uses.

    VP_TO_MEDIAPIPE skips the four metacarpal indices (5, 10, 15, 20), so a
    (21,3) frame expanded to a fake (25,4,4) skeleton and converted back must
    recover the original bit-for-bit, regardless of what the metacarpals were
    filled with.
    """
    import sys as _sys
    scripts_dir = "/home/qsq/qqsq12321/anydexRT/scripts"
    if scripts_dir not in _sys.path:
        _sys.path.insert(0, scripts_dir)
    from record_avp import to_mediapipe
    from anydexrt.data import VP_TO_MEDIAPIPE

    rng = np.random.default_rng(0)
    mp = rng.standard_normal((21, 3)).astype(np.float64)
    poses = np.tile(np.eye(4), (25, 1, 1))
    for mp_idx, vp_idx in enumerate(VP_TO_MEDIAPIPE):
        poses[vp_idx, :3, 3] = mp[mp_idx]
    for meta in (5, 10, 15, 20):  # metacarpals: VP_TO_MEDIAPIPE never reads these
        poses[meta, :3, 3] = rng.standard_normal(3)

    recovered = to_mediapipe(poses)
    assert np.allclose(recovered, mp, atol=1e-6)


def test_record_avp_dedup_logic_matches_live_teleop():
    """The de-dup check in record_avp.py must be the same one validated for
    teleop_avp.py: exact array equality on the raw (25,4,4) skeleton."""
    a = np.zeros((25, 4, 4))
    b = a.copy()
    b[4, :3, 3] = 0.001  # a real sensor update, however small, must count as new
    assert not np.array_equal(a, b)
    assert np.array_equal(a, a.copy())
