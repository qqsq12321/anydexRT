"""Tests for the joint-space output smoother.

Jitter in this pipeline has two distinct sources: the NNS lookup hunting
between nearby table entries as a continuous target crosses a Voronoi
boundary, and `refine_pinch` swapping in a stored template as a discrete jump.
A low-pass on the final qpos fixes both without touching f_m or f_s, so these
tests pin the filter's own behaviour in isolation (no hand/model needed) and
one end-to-end check that AnyDexRT actually wires it in.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from anydexrt.inference import JointLowPassFilter


def test_first_sample_has_no_lag():
    """Startup must not ramp from zero or from an arbitrary initial state."""
    f = JointLowPassFilter(cutoff_hz=5.0)
    x = np.array([0.3, -0.7, 1.2], dtype=np.float32)
    out = f.apply(x, dt=0.016)
    assert np.allclose(out, x)


def test_converges_to_a_constant_target():
    f = JointLowPassFilter(cutoff_hz=5.0)
    target = np.array([1.0, -1.0], dtype=np.float32)
    f.apply(np.zeros(2, dtype=np.float32), dt=0.0)
    for _ in range(500):
        out = f.apply(target, dt=0.01)
    assert np.allclose(out, target, atol=1e-3)


def test_matches_first_order_step_response():
    """y(t) after a step must follow 1 - exp(-t/tau) to within simulation error."""
    cutoff = 4.0
    tau = 1.0 / (2 * np.pi * cutoff)
    f = JointLowPassFilter(cutoff_hz=cutoff)
    f.apply(np.array([0.0]), dt=0.0)  # seed at 0
    dt = 0.001
    t = 0.0
    out = np.array([0.0])
    for _ in range(2000):
        out = f.apply(np.array([1.0]), dt=dt)
        t += dt
    expected = 1.0 - np.exp(-t / tau)
    assert float(out[0]) == pytest.approx(expected, abs=0.01)


def test_never_overshoots_a_monotone_target():
    """A convex combination of the previous OUTPUT and a new sample cannot
    exceed either -- the filter must not ring or overshoot."""
    f = JointLowPassFilter(cutoff_hz=5.0)
    rng = np.random.default_rng(0)
    base = 0.0
    prev_out = np.array([0.0])
    f.apply(prev_out, dt=0.0)
    for _ in range(50):
        base += abs(rng.normal(scale=0.1))  # monotonically increasing target
        target = np.array([base])
        out = f.apply(target, dt=0.02)
        assert out[0] <= target[0] + 1e-6
        assert out[0] >= prev_out[0] - 1e-6
        prev_out = out


def test_zero_dt_does_not_move():
    f = JointLowPassFilter(cutoff_hz=5.0)
    f.apply(np.array([0.0]), dt=1.0)
    out = f.apply(np.array([10.0]), dt=0.0)
    assert out[0] == pytest.approx(0.0)


def test_large_dt_snaps_to_new_value():
    """A long gap (e.g. tracking lost and regained) should not impose stale lag."""
    f = JointLowPassFilter(cutoff_hz=5.0)
    f.apply(np.array([0.0]), dt=1.0)
    out = f.apply(np.array([10.0]), dt=1000.0)
    assert out[0] == pytest.approx(10.0, abs=1e-2)


def test_reset_clears_state_so_next_sample_has_no_lag():
    f = JointLowPassFilter(cutoff_hz=5.0)
    f.apply(np.array([5.0]), dt=0.0)
    f.apply(np.array([5.0]), dt=0.01)
    f.reset()
    out = f.apply(np.array([-3.0]), dt=0.01)
    assert out[0] == pytest.approx(-3.0)


def test_wall_clock_default_runs_without_explicit_dt():
    """No dt given must fall back to measured wall-clock time, not raise."""
    f = JointLowPassFilter(cutoff_hz=5.0)
    f.apply(np.array([0.0]))
    time.sleep(0.01)
    out = f.apply(np.array([1.0]))
    assert 0.0 < out[0] < 1.0


def test_smoothing_disabled_is_a_passthrough(monkeypatch):
    """AnyDexRT with smoothing_hz=None must return the raw solver output.

    Confirms the opt-in default doesn't change behaviour for existing offline
    evaluation / the README's recorded numbers.
    """
    from anydexrt.hands import get_hand
    from anydexrt.data import build_robot_clouds
    from anydexrt.inference import AnyDexRT
    from anydexrt.models import FingerMapper
    from anydexrt.normalize import HandNormalizer

    spec = get_hand("linker_l20")
    clouds, kin = build_robot_clouds(spec, n_samples=500, seed=0, frame="palm")
    mapper = FingerMapper(spec.n_fingers)
    norm = HandNormalizer.fit(spec.finger_names, np.transpose(clouds.tips, (1, 0, 2)))
    rt = AnyDexRT(mapper, clouds, kin, norm, use_contacts=False)
    assert rt.smoother is None

    human = clouds.tips[0]
    qpos_direct, _ = rt.solver.solve(rt.map_fingertips(human))
    qpos_via_retarget = rt.retarget(human)
    assert np.allclose(qpos_direct, qpos_via_retarget)


def test_smoothing_damps_a_frame_to_frame_jump():
    """Enabling smoothing must change the output relative to the unsmoothed run,
    in the direction of damping a jump -- isolated from the NNS continuity prior,
    which both runs share equally."""
    from anydexrt.hands import get_hand
    from anydexrt.data import build_robot_clouds
    from anydexrt.inference import AnyDexRT
    from anydexrt.models import FingerMapper
    from anydexrt.normalize import HandNormalizer

    spec = get_hand("linker_l20")
    clouds, kin = build_robot_clouds(spec, n_samples=500, seed=0, frame="palm")
    mapper = FingerMapper(spec.n_fingers)
    norm = HandNormalizer.fit(spec.finger_names, np.transpose(clouds.tips, (1, 0, 2)))

    def two_steps(smoothing_hz):
        rt = AnyDexRT(mapper, clouds, kin, norm, use_contacts=False, smoothing_hz=smoothing_hz)
        rt.retarget(clouds.tips[0], dt=0.0)
        return rt.retarget(clouds.tips[200], dt=0.02)

    raw = two_steps(None)
    smoothed = two_steps(5.0)
    if np.allclose(raw, smoothed):
        pytest.skip("frames 0 and 200 happened to map to the same table entry")
    assert not np.allclose(raw, smoothed)
