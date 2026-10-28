"""Protocol round-trip and end-to-end loopback tests for the PICO 4 driver.

No real headset needed: these build the exact byte/JSON shapes the headset's
own firmware would send (per the protocol spec reverse-engineered from
AnyDexRetarget's working implementation) and verify anydexRT's independent
reimplementation (anydexrt.pico, scripts/pico4_daemon.py) parses them
correctly end to end.
"""

import socket
import struct
import sys
import time

import numpy as np
import pytest

from anydexrt.pico import (
    JOINT_21_INDICES,
    Pico4,
    _CMD_CONNECT,
    _CMD_DEVICE_STATE_JSON,
    _CMD_HEARTBEAT,
    _CLIENT_HEAD,
    _TAIL,
    _DirectFrameParser,
    _RelayFrameParser,
    _parse_hand_joints,
    _parse_tracking_json,
)

SCRIPTS_DIR = "/home/qsq/qqsq12321/anydexRT/scripts"
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
from pico4_daemon import Pico4Daemon, encode_relay_frame  # noqa: E402


def _joint(x: float, y: float, z: float) -> dict:
    return {"p": f"{x},{y},{z}"}


def _hand_json(is_active=1, n_joints: int = 26) -> dict:
    joints = [_joint(i, i + 0.1, i + 0.2) for i in range(n_joints)]
    return {"isActive": is_active, "HandJointLocations": joints}


def _tracking_payload(left=None, right=None) -> bytes:
    import json

    body = {"Hand": {}}
    if left is not None:
        body["Hand"]["leftHand"] = left
    if right is not None:
        body["Hand"]["rightHand"] = right
    return json.dumps(body).encode("utf-8")


def _direct_frame(cmd: int, payload: bytes) -> bytes:
    """[0x3F][cmd][4B len LE][payload][8B ts (unparsed)][0xA5]"""
    return bytes([_CLIENT_HEAD, cmd]) + struct.pack("<I", len(payload)) + payload + b"\x00" * 8 + bytes([_TAIL])


# ------------------------------------------------------------- unit parsing


def test_parse_hand_joints_wrist_relative_and_21_point_reduction():
    hand = _hand_json(is_active=1)
    out = _parse_hand_joints(hand)
    assert out is not None
    assert out.shape == (21, 3)
    # Joint 0 of the 21-point output is SDK joint JOINT_21_INDICES[0] == 1;
    # everything should be wrist(joint-0-of-21)-relative, so index 0 is zero.
    assert np.allclose(out[0], 0.0)
    expected = np.array([[i, i + 0.1, i + 0.2] for i in JOINT_21_INDICES], dtype=np.float32)
    expected -= expected[0].copy()
    assert np.allclose(out, expected)


@pytest.mark.parametrize("is_active", [0, "0", 2, None, "not-a-number"])
def test_parse_hand_joints_inactive_or_invalid_returns_none(is_active):
    hand = _hand_json(is_active=is_active)
    assert _parse_hand_joints(hand) is None


def test_parse_hand_joints_wrong_joint_count_returns_none():
    hand = _hand_json(is_active=1, n_joints=20)
    assert _parse_hand_joints(hand) is None


def test_parse_tracking_json_plain_and_escaped_wrapper():
    import json

    left = _hand_json(is_active=1)
    payload = _tracking_payload(left=left)
    l1, r1 = _parse_tracking_json(payload)
    assert l1 is not None and r1 is None

    # Some firmware wraps the real payload as an escaped JSON string under 'value'.
    inner = payload.decode("utf-8")
    escaped = inner.replace('"', '\\"')
    wrapped = json.dumps({"value": escaped}).encode("utf-8")
    l2, r2 = _parse_tracking_json(wrapped)
    assert l2 is not None and r2 is None
    assert np.allclose(l1, l2)


# --------------------------------------------------------- frame parsers


def test_direct_frame_parser_roundtrip_and_resync():
    parser = _DirectFrameParser()
    payload = b"hello-payload"
    frame = _direct_frame(_CMD_CONNECT, payload)

    # Garbage before the head byte should be skipped during resync.
    garbage = b"\x00\x01\x02"
    parser.feed(garbage + frame)
    parsed = parser.try_parse()
    assert parsed == {"cmd": _CMD_CONNECT, "payload": payload}
    assert parser.try_parse() is None  # buffer drained

    # Partial feed: nothing parses until the full frame has arrived.
    parser2 = _DirectFrameParser()
    parser2.feed(frame[:5])
    assert parser2.try_parse() is None
    parser2.feed(frame[5:])
    assert parser2.try_parse() == {"cmd": _CMD_CONNECT, "payload": payload}


def test_direct_frame_parser_bad_tail_byte_resyncs():
    parser = _DirectFrameParser()
    payload = b"xyz"
    frame = bytearray(_direct_frame(_CMD_HEARTBEAT, payload))
    frame[-1] = 0x00  # corrupt the tail byte
    good = _direct_frame(_CMD_CONNECT, b"ok")
    parser.feed(bytes(frame) + good)
    # The corrupted frame is dropped byte-by-byte until the next valid frame is found.
    parsed = parser.try_parse()
    assert parsed == {"cmd": _CMD_CONNECT, "payload": b"ok"}


def test_relay_frame_parser_roundtrip():
    payload = _tracking_payload(left=_hand_json())
    frame = encode_relay_frame("pico4", payload)
    parser = _RelayFrameParser()
    parser.feed(frame[:3])
    assert parser.try_parse() is None
    parser.feed(frame[3:])
    assert parser.try_parse() == payload


# --------------------------------------------- end-to-end loopback (daemon + client)


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_daemon_and_pico4_relay_client_loopback():
    """Simulate a headset's direct connection into our own daemon, and read
    the result back out through a relay-mode Pico4 client -- exercises
    daemon accept -> parse -> publish and client connect -> parse -> store,
    the full path, without any real hardware."""
    direct_port = _free_port()
    relay_port = _free_port()
    broadcast_port = _free_port()

    daemon = Pico4Daemon(
        direct_port=direct_port, relay_host="127.0.0.1", relay_port=relay_port,
        broadcast_port=broadcast_port, device_id="pico4-test",
    )
    import threading
    t = threading.Thread(target=daemon.run, daemon=True)
    t.start()
    time.sleep(0.3)  # let the daemon bind both ports

    client = Pico4(mode="relay", relay_host="127.0.0.1", relay_port=relay_port)
    try:
        # Act as the headset: connect directly and send a CONNECT heartbeat
        # followed by a device-state JSON frame.
        headset = socket.create_connection(("127.0.0.1", direct_port), timeout=5.0)
        headset.sendall(_direct_frame(_CMD_CONNECT, b""))
        right = _hand_json(is_active=1)
        payload = _tracking_payload(right=right)
        headset.sendall(_direct_frame(_CMD_DEVICE_STATE_JSON, payload))

        deadline = time.monotonic() + 5.0
        data = client.get_fingers_data()
        while not np.any(data["right_fingers"]) and time.monotonic() < deadline:
            time.sleep(0.05)
            data = client.get_fingers_data()

        expected = np.array([[i, i + 0.1, i + 0.2] for i in JOINT_21_INDICES], dtype=np.float32)
        expected -= expected[0].copy()
        assert np.allclose(data["right_fingers"], expected)
        assert np.allclose(data["left_fingers"], 0.0)
    finally:
        headset.close()
        client.stop()
        daemon._stop.set()
        t.join(timeout=3.0)
