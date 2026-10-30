"""Tests for the on-disk cache behind `build_robot_clouds`.

The cache exists because sampling the robot cloud costs ~4.4s on every start.
Two things have to hold, and the second is the dangerous one:

  * a hit must return exactly what a miss computes, bit for bit;
  * anything that changes the cloud must change the key -- a stale cloud
    trains or infers against a different space than the one it was built for,
    and nothing downstream checks that they agree.

All tests point the cache at a tmp_path, so they never touch the real one.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from anydexrt.data import (
    build_robot_clouds,
    cache_dir,
    cache_load,
    cache_store,
    content_hash,
    mjcf_fingerprint,
)
from anydexrt.hands import get_hand

HAND = "linker_l20"
SMALL = 300  # enough to exercise the paths; the point here is not cloud quality


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ANYDEXRT_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("ANYDEXRT_NO_CACHE", raising=False)
    return tmp_path


def test_hit_matches_miss_bit_for_bit(cache):
    """The whole point: a cached cloud is indistinguishable from a fresh one."""
    miss, _ = build_robot_clouds(HAND, n_samples=SMALL, seed=0, frame="palm")
    assert list(cache.glob("clouds/*.npz")), "nothing was cached"
    hit, _ = build_robot_clouds(HAND, n_samples=SMALL, seed=0, frame="palm")

    for name in ("qpos", "tips", "tips_norm", "tips_rot", "frame_origin", "frame_rot"):
        assert np.array_equal(getattr(miss, name), getattr(hit, name)), name
    for a, b in zip(miss.normalizer.normalizers, hit.normalizer.normalizers):
        assert np.array_equal(a.center, b.center)
        assert a.scale == b.scale


def test_disabled_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ANYDEXRT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("ANYDEXRT_NO_CACHE", "1")
    assert cache_dir("clouds") is None
    build_robot_clouds(HAND, n_samples=SMALL, seed=0)
    assert not list(tmp_path.rglob("*.npz")), "wrote a cache entry while disabled"


def test_one_entry_serves_both_frames(cache):
    """Entries are world-frame, so `frame` must not be part of the key."""
    world, _ = build_robot_clouds(HAND, n_samples=SMALL, seed=0, frame="world")
    palm, _ = build_robot_clouds(HAND, n_samples=SMALL, seed=0, frame="palm")
    assert len(list(cache.glob("clouds/*.npz"))) == 1
    # Same underlying cloud, genuinely different output coordinates.
    assert np.array_equal(world.qpos, palm.qpos)
    assert not np.array_equal(world.tips, palm.tips)


def test_frame_transform_still_applies_on_a_hit(cache):
    """A hit must not accidentally return world tips for frame='palm'."""
    build_robot_clouds(HAND, n_samples=SMALL, seed=0, frame="world")   # prime
    palm_hit, _ = build_robot_clouds(HAND, n_samples=SMALL, seed=0, frame="palm")
    monkey = os.environ.pop("ANYDEXRT_CACHE_DIR")
    try:
        os.environ["ANYDEXRT_NO_CACHE"] = "1"
        palm_fresh, _ = build_robot_clouds(HAND, n_samples=SMALL, seed=0, frame="palm")
    finally:
        del os.environ["ANYDEXRT_NO_CACHE"]
        os.environ["ANYDEXRT_CACHE_DIR"] = monkey
    assert np.array_equal(palm_hit.tips, palm_fresh.tips)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_samples": SMALL + 1},
        {"seed": 1},
    ],
)
def test_sample_count_and_seed_are_in_the_key(cache, kwargs):
    base = dict(n_samples=SMALL, seed=0, frame="palm")
    build_robot_clouds(HAND, **base)
    build_robot_clouds(HAND, **{**base, **kwargs})
    assert len(list(cache.glob("clouds/*.npz"))) == 2, f"{kwargs} reused the key"


def test_hand_is_in_the_key(cache):
    build_robot_clouds(HAND, n_samples=SMALL, seed=0, frame="palm")
    build_robot_clouds("wuji_hand", n_samples=SMALL, seed=0, frame="palm")
    assert len(list(cache.glob("clouds/*.npz"))) == 2


def test_editing_the_mjcf_changes_the_fingerprint(tmp_path):
    """Content, not path or mtime: a model edit has to invalidate."""
    original = get_hand(HAND).resolved_mjcf()
    copy = tmp_path / "hand.xml"
    shutil.copy(original, copy)
    assert mjcf_fingerprint(copy) == mjcf_fingerprint(original)

    copy.write_bytes(copy.read_bytes() + b"<!-- edited -->")
    assert mjcf_fingerprint(copy) != mjcf_fingerprint(original)


def test_fingerprint_follows_includes(tmp_path):
    """An `<include>`d file is part of the model, so it is part of the hash."""
    child = tmp_path / "child.xml"
    child.write_text("<mujoco/>")
    parent = tmp_path / "parent.xml"
    parent.write_text(f'<mujoco><include file="{child.name}"/></mujoco>')
    before = mjcf_fingerprint(parent)

    child.write_text("<mujoco><!-- changed --></mujoco>")
    assert mjcf_fingerprint(parent) != before


def test_fingerprint_survives_a_missing_include(tmp_path):
    """Must not raise -- MuJoCo will report the real error, not the hash."""
    parent = tmp_path / "parent.xml"
    parent.write_text('<mujoco><include file="nope.xml"/></mujoco>')
    assert isinstance(mjcf_fingerprint(parent), str)


def test_fingerprint_handles_include_cycles(tmp_path):
    """A -> B -> A must terminate rather than spin."""
    a, b = tmp_path / "a.xml", tmp_path / "b.xml"
    a.write_text('<mujoco><include file="b.xml"/></mujoco>')
    b.write_text('<mujoco><include file="a.xml"/></mujoco>')
    assert isinstance(mjcf_fingerprint(a), str)


def test_store_reports_success_and_roundtrips(cache):
    """`cache_store` returning True is load-bearing: the first version always
    failed silently, because np.savez appends `.npz` to a name without it."""
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert cache_store("k", "probe", values=data) is True
    assert (cache / "probe" / "k.npz").exists()
    assert not list((cache / "probe").glob("*.tmp*")), "left a temp file behind"

    loaded = cache_load("k", "probe")
    assert loaded is not None and np.array_equal(loaded["values"], data)


def test_store_is_a_noop_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("ANYDEXRT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("ANYDEXRT_NO_CACHE", "1")
    assert cache_store("k", "probe", values=np.zeros(1)) is False
    assert cache_load("k", "probe") is None


def test_missing_entry_loads_as_none(cache):
    assert cache_load("never-written", "clouds") is None


def test_corrupt_entry_falls_back_to_recompute(cache):
    """A truncated file must not crash a run, and must not be trusted."""
    fresh, _ = build_robot_clouds(HAND, n_samples=SMALL, seed=0, frame="palm")
    entry = next(iter(cache.glob("clouds/*.npz")))
    entry.write_bytes(entry.read_bytes()[: entry.stat().st_size // 2])

    assert cache_load(entry.stem, "clouds") is None
    again, _ = build_robot_clouds(HAND, n_samples=SMALL, seed=0, frame="palm")
    assert np.array_equal(again.tips, fresh.tips)


def test_content_hash_is_stable_and_separates_parts(cache):
    assert content_hash("a", "b") == content_hash("a", "b")
    # Not a plain concatenation: "ab" + "" must not collide with "a" + "b".
    assert content_hash("ab", "") != content_hash("a", "b")
    assert content_hash(b"bytes") == content_hash(b"bytes")
