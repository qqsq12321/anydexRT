"""Tests for the Linker L20 RS485 output (anydexrt/real_l20.py).

These cover the qpos->register mapping and the Modbus framing, both of which
move real hardware when `teleop_avp.py --real --confirm-real` runs, so a wrong
byte here is a wrong finger position. No serial port is opened: the mapping is
pure, and the transport is exercised against a fake port.
"""

from __future__ import annotations

import numpy as np
import pytest

from anydexrt.hands import get_hand
from anydexrt.kinematics import HandKinematics
from anydexrt.real_l20 import (
    LinkerL20Output,
    LinkerL20RS485,
    _ACTIVE_JOINT_ORDER,
    _INVERTED_JOINTS,
    _JOINT_LIMITS,
    _QPOS_ALIGNMENT_POINTS,
)

HAND = "linker_l20"


@pytest.fixture(scope="module")
def names():
    return list(get_hand(HAND).joint_names)


@pytest.fixture(scope="module")
def limits():
    return HandKinematics(HAND).limits()


@pytest.fixture
def out():
    return LinkerL20Output(dry_run=True)


def test_joint_limits_match_the_hand_spec(limits, names):
    """The reason qpos needs no unit conversion: the limits are the same.

    If this ever fails, the driver is normalizing against a different range
    than the solver produced, and every register is silently off.
    """
    lo, hi = limits
    for suffix, (d_lo, d_hi) in _JOINT_LIMITS.items():
        idx = [i for i, n in enumerate(names) if n.upper().endswith(suffix)]
        assert len(idx) == 1, f"{suffix} matched {idx}"
        i = idx[0]
        assert lo[i] == pytest.approx(d_lo, abs=1e-4), suffix
        assert hi[i] == pytest.approx(d_hi, abs=1e-4), suffix


def test_register_count_and_distal_repeat(out, names, limits):
    """30 registers, with the distal block repeated three times."""
    regs = out.qpos_to_registers(limits[0], names)
    assert len(regs) == LinkerL20RS485.POSITION_COUNT == 30
    assert regs[15:20] == regs[20:25] == regs[25:30]


def test_registers_are_bytes(out, names, limits):
    lo, hi = limits
    rng = np.random.default_rng(0)
    for q in [lo, hi, (lo + hi) / 2, *rng.uniform(lo, hi, size=(50, len(lo)))]:
        regs = out.qpos_to_registers(q, names)
        assert all(isinstance(r, int) and 0 <= r <= 255 for r in regs)


def test_only_thumb_has_a_roll_register(out, names, limits):
    """Registers 1-4 are the neutral constant: the fingers have no roll drive."""
    lo, hi = limits
    for q in (lo, hi, (lo + hi) / 2):
        assert out.qpos_to_registers(q, names)[1:5] == [128] * 4


def test_inverted_joints_run_backwards(out, names, limits):
    """A joint at its lower limit vs upper limit, for one example of each.

    index_mcp_pitch is inverted (register falls as qpos rises), index_mcp_roll
    is deliberately not.
    """
    lo, hi = limits
    i_pitch = names.index("index_mcp_pitch")
    i_roll = names.index("index_mcp_roll")
    assert "INDEX_MCP_PITCH" in _INVERTED_JOINTS
    assert "INDEX_MCP_ROLL" not in _INVERTED_JOINTS

    q = (lo + hi) / 2
    at_lo, at_hi = q.copy(), q.copy()
    at_lo[i_pitch], at_hi[i_pitch] = lo[i_pitch], hi[i_pitch]
    # root1 block holds the MCP pitches; index is its second entry (register 11)
    assert out.qpos_to_registers(at_lo, names)[11] > out.qpos_to_registers(at_hi, names)[11]

    at_lo, at_hi = q.copy(), q.copy()
    at_lo[i_roll], at_hi[i_roll] = lo[i_roll], hi[i_roll]
    # yaw block holds the MCP rolls; index is its second entry (register 6)
    assert out.qpos_to_registers(at_lo, names)[6] < out.qpos_to_registers(at_hi, names)[6]


def test_monotonic_in_each_joint(out, names, limits):
    """Sweeping one joint moves its register monotonically, nothing else moves.

    Catches a mis-wired register index, which a single-pose check would not:
    two joints swapped still produces plausible-looking bytes.
    """
    lo, hi = limits
    base = (lo + hi) / 2
    for suffix in _ACTIVE_JOINT_ORDER:
        i = [k for k, n in enumerate(names) if n.upper().endswith(suffix)][0]
        seq = []
        for v in np.linspace(lo[i], hi[i], 12):
            q = base.copy()
            q[i] = v
            seq.append(out.qpos_to_registers(q, names))
        arr = np.asarray(seq)
        moved = [c for c in range(30) if np.ptp(arr[:, c]) > 0]
        assert moved, f"{suffix} moved no register"
        for c in moved:
            d = np.diff(arr[:, c].astype(int))
            assert (d >= 0).all() or (d <= 0).all(), f"{suffix} register {c} not monotonic"


def test_thumb_alignment_curve_is_used(out, names, limits):
    """The measured thumb curve, not linear normalization.

    These differ by up to 0.10 of full scale (~26 counts), so this asserts the
    calibration is actually applied rather than quietly bypassed.
    """
    lo, hi = limits
    i = names.index("thumb_mcp")
    q = (lo + hi) / 2
    q[i] = 0.3
    # distal block, thumb is its first entry (register 15); THUMB_MCP is inverted
    got = out.qpos_to_registers(q, names)[15]

    sim, cmd = zip(*_QPOS_ALIGNMENT_POINTS["THUMB_MCP"])
    expected = int(round((1.0 - float(np.interp(0.3, sim, cmd))) * 255))
    assert got == expected

    j_lo, j_hi = _JOINT_LIMITS["THUMB_MCP"]
    linear = int(round((1.0 - (0.3 - j_lo) / (j_hi - j_lo)) * 255))
    assert abs(got - linear) > 10, "alignment curve indistinguishable from linear"


def test_rejects_nan_and_inf(out, names, limits):
    q = (limits[0] + limits[1]) / 2
    for bad in (np.nan, np.inf):
        q_bad = q.copy()
        q_bad[0] = bad
        with pytest.raises(ValueError, match="NaN or Inf"):
            out.qpos_to_registers(q_bad, names)


def test_missing_or_ambiguous_joint_names(out, limits):
    q = (limits[0] + limits[1]) / 2
    with pytest.raises(ValueError, match="no joint ending in"):
        out.qpos_to_registers(q, ["wrong"] * 16)
    dup = list(get_hand(HAND).joint_names) + ["also_thumb_mcp"]
    with pytest.raises(ValueError, match="ambiguous"):
        out.qpos_to_registers(np.r_[q, 0.0], dup)


def test_open_palm_is_fully_extended():
    regs = LinkerL20Output.open_palm_registers()
    assert len(regs) == 30
    assert regs[:5] == [128] * 5          # roll neutral
    assert regs[5:10] == [128] * 5        # yaw neutral
    assert regs[10:] == [255] * 20        # extended (pitch/PIP are inverted)


def test_dry_run_sends_nothing(out, names, limits):
    assert out._hand is None
    out.send((limits[0] + limits[1]) / 2, names)
    assert out.send_count == 1
    out.close()  # must not raise with no port open


def test_identical_commands_are_skipped(out, names, limits):
    q = (limits[0] + limits[1]) / 2
    out.send(q, names)
    out.send(q, names)
    assert out.send_count == 1
    out.send(limits[1], names)
    assert out.send_count == 2


# ------------------------------------------------------------------ transport


def test_crc16_known_values():
    """Modbus CRC-16: the standard check value, plus a real register frame."""
    assert LinkerL20RS485.crc16(b"\x00") == 0x40BF
    assert LinkerL20RS485.crc16(b"123456789") == 0x4B37
    assert LinkerL20RS485.crc16(b"") == 0xFFFF


class _FakeSerial:
    """Captures what the driver writes and replies like the hand would."""

    EIGHTBITS = 8
    PARITY_NONE = "N"
    STOPBITS_ONE = 1

    def __init__(self):
        self.written = bytearray()
        self.reply = b""
        self.is_open = True

    def reset_input_buffer(self):
        pass

    def write(self, data):
        self.written.extend(data)

    def flush(self):
        pass

    def read(self, n):
        out, self.reply = self.reply[:n], self.reply[n:]
        return out

    def close(self):
        self.is_open = False


def _driver_with_fake(monkeypatch, slave_id=0x29):
    """A LinkerL20RS485 whose serial port is a _FakeSerial."""
    import sys
    import types

    fake = _FakeSerial()
    module = types.SimpleNamespace(
        Serial=lambda **kw: fake,
        EIGHTBITS=8, PARITY_NONE="N", STOPBITS_ONE=1,
    )
    monkeypatch.setitem(sys.modules, "serial", module)
    drv = LinkerL20RS485(slave_id=slave_id)
    return drv, fake


def _ok_reply(slave_id, func, start, count):
    body = bytes([slave_id, func, (start >> 8) & 0xFF, start & 0xFF,
                  (count >> 8) & 0xFF, count & 0xFF])
    crc = LinkerL20RS485.crc16(body)
    return body + bytes([crc & 0xFF, (crc >> 8) & 0xFF])


def test_write_registers_frame_layout(monkeypatch):
    drv, fake = _driver_with_fake(monkeypatch)
    fake.reply = _ok_reply(0x29, 0x10, 0, 30)
    drv.set_positions(list(range(30)))

    sent = bytes(fake.written)
    assert sent[0] == 0x29                      # slave id
    assert sent[1] == 0x10                      # write multiple registers
    assert sent[2:4] == b"\x00\x00"             # start address 0
    assert sent[4:6] == b"\x00\x1e"             # 30 registers
    assert sent[6] == 60                        # byte count
    assert sent[7:67] == bytes(b for v in range(30) for b in (0x00, v))
    assert LinkerL20RS485.crc16(sent[:-2]) == (sent[-2] | (sent[-1] << 8))


def test_position_command_needs_exactly_30(monkeypatch):
    drv, _ = _driver_with_fake(monkeypatch)
    with pytest.raises(ValueError, match="30 registers"):
        drv.set_positions([0] * 16)


def test_short_response_raises_timeout(monkeypatch):
    drv, fake = _driver_with_fake(monkeypatch)
    fake.reply = b"\x29\x10"            # truncated
    with pytest.raises(TimeoutError, match="incomplete"):
        drv.set_positions([0] * 30)


def test_crc_mismatch_raises(monkeypatch):
    drv, fake = _driver_with_fake(monkeypatch)
    good = _ok_reply(0x29, 0x10, 0, 30)
    fake.reply = good[:-2] + b"\x00\x00"
    with pytest.raises(RuntimeError, match="CRC mismatch"):
        drv.set_positions([0] * 30)


def test_wrong_slave_id_raises(monkeypatch):
    drv, fake = _driver_with_fake(monkeypatch)
    fake.reply = _ok_reply(0x2A, 0x10, 0, 30)   # left hand answered
    with pytest.raises(RuntimeError, match="slave id"):
        drv.set_positions([0] * 30)


def test_modbus_exception_raises(monkeypatch):
    drv, fake = _driver_with_fake(monkeypatch)
    body = bytes([0x29, 0x10 | 0x80, 0x02, 0x00, 0x00, 0x00])
    crc = LinkerL20RS485.crc16(body)
    fake.reply = body + bytes([crc & 0xFF, (crc >> 8) & 0xFF])
    with pytest.raises(RuntimeError, match="exception code 2"):
        drv.set_positions([0] * 30)


def test_register_values_are_clamped(monkeypatch):
    drv, fake = _driver_with_fake(monkeypatch)
    fake.reply = _ok_reply(0x29, 0x10, 0, 3)
    drv.write_registers(0, [-5, 300, 128])
    payload = bytes(fake.written)[7:13]
    assert payload == bytes([0x00, 0, 0x00, 255, 0x00, 128])


def test_thumb_roll_offset_is_tunable(names, limits):
    """Only thumb_cmc_roll's register moves, and it moves the right way.

    The offset feeds the alignment curve, so this also guards the order:
    offset first, then interpolation.
    """
    lo, hi = limits
    q = (lo + hi) / 2
    q[names.index("thumb_cmc_roll")] = 0.2

    regs = {
        off: LinkerL20Output(dry_run=True, thumb_cmc_roll_offset=off).qpos_to_registers(q, names)
        for off in (0.0, 0.10, 0.15)
    }
    # THUMB_CMC_ROLL is inverted, so a smaller offset -> larger register
    assert regs[0.0][0] > regs[0.10][0] > regs[0.15][0]
    for off in (0.0, 0.10):
        assert regs[off][1:] == regs[0.15][1:], "offset touched another joint"


def test_default_thumb_roll_offset_keeps_both_ends(names, limits):
    """Default 0.075 buys abduction without giving up full adduction.

    Both halves are pinned, since they pull against each other: a larger
    offset loses abduction, and 0 loses the end of travel (qpos 1.39 stops
    reaching register 0, so the thumb cannot quite close for a pinch).
    """
    from anydexrt.real_l20 import _DEFAULT_THUMB_CMC_ROLL_OFFSET_RAD as default

    assert default == pytest.approx(0.075)
    lo, hi = limits
    i = names.index("thumb_cmc_roll")
    out = LinkerL20Output(dry_run=True)
    original = LinkerL20Output(dry_run=True, thumb_cmc_roll_offset=0.15)
    off = LinkerL20Output(dry_run=True, thumb_cmc_roll_offset=0.0)
    q = (lo + hi) / 2

    q[i] = hi[i]
    assert out.qpos_to_registers(q, names)[0] == 0, "full adduction no longer reachable"
    assert off.qpos_to_registers(q, names)[0] > 0, "0 was expected to lose the end of travel"

    q[i] = 0.2
    assert (off.qpos_to_registers(q, names)[0]
            > out.qpos_to_registers(q, names)[0]
            > original.qpos_to_registers(q, names)[0])

    sim, cmd = zip(*_QPOS_ALIGNMENT_POINTS["THUMB_CMC_ROLL"])
    assert float(np.interp(0.0 + 0.15, sim, cmd)) < 0.02   # original: at the stop
