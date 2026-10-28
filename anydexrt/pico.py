"""PICO 4 hand-tracking input device.

Independent reimplementation of the PICO 4 wire protocol (not an import of
any other project's code): a custom binary framing for the headset's direct
TCP connection, a JSON-over-length-prefixed-framing relay protocol for local
clients, and a UDP discovery broadcast. These byte layouts, special values,
and timeouts are what the headset's own tracking software actually speaks --
they are protocol facts, not implementation choices, so they are fixed here
exactly as observed, independent of how any particular client/daemon chooses
to structure its code around them.

Two operating modes, selected via `Pico4(mode=...)`:

  relay (default):
    Connects to a local relay daemon's TCP port (127.0.0.1:63902 by default,
    see scripts/pico4_daemon.py). The daemon is the thing that actually talks
    to the headset and republishes its tracking JSON to any number of relay
    clients -- use this so the headset stays paired across repeated restarts
    of whatever script is consuming the data.
    Frame format: [4B device_id_len LE][device_id][4B json_len LE][JSON]

  direct:
    This process itself is the TCP server (0.0.0.0:63901 by default) and UDP
    broadcaster (so the headset can discover this PC) -- no daemon needed,
    but the headset re-pairs every time this process restarts.
    Frame format: [0x3F][cmd][4B payload_len LE][payload][8B timestamp][0xA5]

Usage:
    dev = Pico4(mode="relay")                 # needs pico4_daemon.py running
    dev = Pico4(mode="direct")                # self-contained
    data = dev.get_fingers_data()              # non-blocking, latest cached frame
    # {'left_fingers': (21,3) float32, 'right_fingers': (21,3) float32}
    dev.stop()
"""

from __future__ import annotations

import json
import logging
import socket
import struct
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------- protocol

_CLIENT_HEAD = 0x3F
_SERVER_HEAD = 0xCF
_TAIL = 0xA5
_CMD_CONNECT = 0x19
_CMD_HEARTBEAT = 0x23
_CMD_DEVICE_STATE_JSON = 0x6D
_CMD_BATTERY = 0x1A
_CMD_SENSOR = 0x1B
_CMD_UDP_TCP_IP = 0x7E

_HEADER_SIZE = 6  # head(1) + cmd(1) + payload_len(4)
_TAIL_SIZE = 9  # timestamp(8, unparsed) + tail(1)
_MIN_FRAME = _HEADER_SIZE + _TAIL_SIZE

DEFAULT_RELAY_HOST = "127.0.0.1"
DEFAULT_RELAY_PORT = 63902
DEFAULT_DIRECT_PORT = 63901
DEFAULT_UDP_BROADCAST_PORT = 29888
_BROADCAST_INTERVAL_S = 5.0
_HEARTBEAT_TIMEOUT_S = 20.0
_RELAY_RECONNECT_S = 2.0

# 26 SDK joints -> 21 MediaPipe joints: removes Palm(0) and the four
# metacarpal joints (Index=6, Middle=11, Ring=16, Little=21).
JOINT_21_INDICES = [
    1, 2, 3, 4, 5,
    7, 8, 9, 10,
    12, 13, 14, 15,
    17, 18, 19, 20,
    22, 23, 24, 25,
]


# ------------------------------------------------------------- JSON parsing


def _parse_csv_floats(s: str) -> list[float]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if part:
            try:
                out.append(float(part))
            except ValueError:
                out.append(0.0)
    return out


def _parse_hand_joints(hand_j: dict) -> np.ndarray | None:
    """One hand's 26-joint SDK payload -> (21, 3) wrist-relative, or None."""
    if not isinstance(hand_j, dict):
        return None
    is_active = hand_j.get("isActive", 0)
    try:
        is_active = int(float(str(is_active)))
    except (ValueError, TypeError):
        return None
    if is_active != 1:
        return None

    joints = hand_j.get("HandJointLocations")
    if not isinstance(joints, list) or len(joints) != 26:
        return None

    result = np.zeros((21, 3), dtype=np.float32)
    for dst_i, src_i in enumerate(JOINT_21_INDICES):
        joint = joints[src_i]
        if not isinstance(joint, dict):
            continue
        p = joint.get("p")
        if not isinstance(p, str):
            continue
        vals = _parse_csv_floats(p)
        if len(vals) >= 3:
            result[dst_i] = vals[:3]

    result -= result[0].copy()
    return result


def _parse_tracking_json(payload: bytes) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Device-state JSON payload -> (left, right), each (21,3) or None."""
    try:
        outer = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, None

    if isinstance(outer, dict) and isinstance(outer.get("value"), str):
        # Some firmware wraps the real payload as an escaped JSON string
        # under a 'value' key; unescape and re-parse.
        inner_str = outer["value"].replace("\\", "")
        try:
            data = json.loads(inner_str)
        except json.JSONDecodeError:
            return None, None
    else:
        data = outer

    if not isinstance(data, dict):
        return None, None
    hand_section = data.get("Hand", {})
    if not isinstance(hand_section, dict):
        return None, None

    left = _parse_hand_joints(hand_section.get("leftHand", {}))
    right = _parse_hand_joints(hand_section.get("rightHand", {}))
    return left, right


# ------------------------------------------------------ relay frame parser
# Frame: [4B device_id_len LE][device_id bytes][4B json_len LE][JSON bytes]


class _RelayFrameParser:
    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> None:
        self._buf.extend(data)

    def try_parse(self) -> bytes | None:
        """Return the next JSON payload, or None if a full frame isn't buffered yet."""
        while len(self._buf) >= 8:
            id_len = struct.unpack_from("<I", self._buf, 0)[0]
            if id_len > 256 or len(self._buf) < 4 + id_len + 4:
                del self._buf[0]
                continue
            json_len = struct.unpack_from("<I", self._buf, 4 + id_len)[0]
            if json_len > 10_000_000:
                del self._buf[0]
                continue
            total = 4 + id_len + 4 + json_len
            if len(self._buf) < total:
                return None
            payload = bytes(self._buf[4 + id_len + 4 : total])
            del self._buf[:total]
            return payload
        return None


# ----------------------------------------------------- direct frame parser


class _DirectFrameParser:
    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> None:
        self._buf.extend(data)

    def try_parse(self) -> dict | None:
        """Return {'cmd': int, 'payload': bytes}, or None if incomplete."""
        while self._buf:
            try:
                idx = self._buf.index(_CLIENT_HEAD)
            except ValueError:
                self._buf.clear()
                return None
            if idx:
                del self._buf[:idx]
            if len(self._buf) < _MIN_FRAME:
                return None
            cmd = self._buf[1]
            plen = struct.unpack_from("<I", self._buf, 2)[0]
            if plen > 10_000_000:
                del self._buf[0]
                continue
            total = _HEADER_SIZE + plen + _TAIL_SIZE
            if len(self._buf) < total:
                return None
            if self._buf[total - 1] != _TAIL:
                del self._buf[0]
                continue
            payload = bytes(self._buf[_HEADER_SIZE : _HEADER_SIZE + plen])
            del self._buf[:total]
            return {"cmd": cmd, "payload": payload}
        return None


# --------------------------------------------------- UDP broadcast helpers


def _build_broadcast_packet(ip: str) -> bytes:
    ip_bytes = ip.encode("utf-8")
    ts_ms = int(time.time() * 1000) & 0xFFFFFFFFFFFFFFFF
    pkt = bytearray()
    pkt.append(_SERVER_HEAD)
    pkt.append(_CMD_UDP_TCP_IP)
    pkt.extend(struct.pack("<I", len(ip_bytes)))
    pkt.extend(ip_bytes)
    pkt.extend(struct.pack("<Q", ts_ms))
    pkt.append(_TAIL)
    return bytes(pkt)


def _get_local_ips() -> list[str]:
    ips = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            if info[0] == socket.AF_INET:
                ip = info[4][0]
                if not ip.startswith("127."):
                    ips.append(ip)
    except Exception:
        pass
    if not ips:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ips.append(s.getsockname()[0])
            s.close()
        except Exception:
            pass
    return list(dict.fromkeys(ips))


# -------------------------------------------------------------- main class


class Pico4:
    """PICO 4 hand tracking input device.

    Args:
        mode: 'relay' (default) or 'direct'.
        relay_host, relay_port: relay daemon address (relay mode only).
        port: TCP listen port (direct mode only).
        broadcast_port: UDP discovery broadcast port (direct mode only).
    """

    def __init__(
        self,
        mode: str = "relay",
        relay_host: str = DEFAULT_RELAY_HOST,
        relay_port: int = DEFAULT_RELAY_PORT,
        port: int = DEFAULT_DIRECT_PORT,
        broadcast_port: int = DEFAULT_UDP_BROADCAST_PORT,
    ) -> None:
        self._mode = mode.lower()
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._left_hand: np.ndarray | None = None
        self._right_hand: np.ndarray | None = None
        self._last_update: float = 0.0

        if self._mode == "relay":
            self._relay_host = relay_host
            self._relay_port = relay_port
            self._thread = threading.Thread(target=self._run_relay, daemon=True)
            logger.info("Pico4 relay mode: connecting to %s:%d", relay_host, relay_port)
        else:
            self._direct_port = port
            self._broadcast_port = broadcast_port
            self._thread = threading.Thread(target=self._run_direct_server, daemon=True)
            logger.info("Pico4 direct mode: listening on TCP %d", port)

        self._thread.start()

    # ------------------------------------------------------- public API

    def get_fingers_data(self) -> dict:
        empty = np.zeros((21, 3), dtype=np.float32)
        with self._lock:
            left = self._left_hand.copy() if self._left_hand is not None else empty.copy()
            right = self._right_hand.copy() if self._right_hand is not None else empty.copy()
        return {"left_fingers": left, "right_fingers": right}

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------- relay mode

    def _run_relay(self) -> None:
        while not self._stop.is_set():
            sock = None
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                sock.connect((self._relay_host, self._relay_port))
                sock.settimeout(1.0)
                logger.info("Pico4 connected to relay %s:%d", self._relay_host, self._relay_port)
                self._relay_loop(sock)
            except (ConnectionRefusedError, OSError) as e:
                logger.warning(
                    "Pico4 relay connect failed: %s, retry in %.1fs", e, _RELAY_RECONNECT_S
                )
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass
            if not self._stop.is_set():
                self._stop.wait(_RELAY_RECONNECT_S)

    def _relay_loop(self, sock: socket.socket) -> None:
        parser = _RelayFrameParser()
        while not self._stop.is_set():
            try:
                data = sock.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                break
            parser.feed(data)
            while True:
                payload = parser.try_parse()
                if payload is None:
                    break
                left, right = _parse_tracking_json(payload)
                with self._lock:
                    if left is not None:
                        self._left_hand = left
                    if right is not None:
                        self._right_hand = right
                    if left is not None or right is not None:
                        self._last_update = time.monotonic()
        logger.info("Pico4 relay connection closed")

    # ------------------------------------------------------ direct mode

    def _run_direct_server(self) -> None:
        broadcaster = threading.Thread(target=self._broadcast_loop, daemon=True)
        broadcaster.start()

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind(("0.0.0.0", self._direct_port))
        except OSError as e:
            logger.error("Pico4 direct mode: cannot bind port %d: %s", self._direct_port, e)
            return
        srv.listen(1)
        srv.settimeout(1.0)
        logger.info("Pico4 direct mode: TCP server on port %d", self._direct_port)
        try:
            while not self._stop.is_set():
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                logger.info("Pico4 direct: connected from %s", addr)
                t = threading.Thread(target=self._direct_client_loop, args=(conn,), daemon=True)
                t.start()
        finally:
            srv.close()

    def _direct_client_loop(self, conn: socket.socket) -> None:
        parser = _DirectFrameParser()
        conn.settimeout(1.0)
        last_hb = time.monotonic()
        try:
            while not self._stop.is_set():
                if time.monotonic() - last_hb > _HEARTBEAT_TIMEOUT_S:
                    break
                try:
                    data = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not data:
                    break
                parser.feed(data)
                while True:
                    frame = parser.try_parse()
                    if frame is None:
                        break
                    if frame["cmd"] in (_CMD_HEARTBEAT, _CMD_CONNECT, _CMD_BATTERY, _CMD_SENSOR):
                        last_hb = time.monotonic()
                    if frame["cmd"] == _CMD_DEVICE_STATE_JSON:
                        left, right = _parse_tracking_json(frame["payload"])
                        with self._lock:
                            if left is not None:
                                self._left_hand = left
                            if right is not None:
                                self._right_hand = right
                            if left is not None or right is not None:
                                self._last_update = time.monotonic()
        finally:
            conn.close()
            logger.info("Pico4 direct: client disconnected")

    def _broadcast_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            while not self._stop.is_set():
                for ip in _get_local_ips():
                    parts = ip.split(".")
                    if len(parts) == 4:
                        bcast = ".".join(parts[:3]) + ".255"
                        pkt = _build_broadcast_packet(ip)
                        try:
                            sock.sendto(pkt, (bcast, self._broadcast_port))
                        except OSError:
                            pass
                self._stop.wait(_BROADCAST_INTERVAL_S)
        finally:
            sock.close()
