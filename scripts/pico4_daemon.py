"""PICO 4 discovery/relay daemon.

Run this in its own terminal when you want the PICO 4 to discover and stay
connected to this PC without any particular teleop script running. It:

1. broadcasts this PC's IP on UDP so the headset can discover it
2. accepts the headset's direct TCP connection
3. republishes its tracking JSON to local relay clients on 127.0.0.1:63902

anydexrt.pico.Pico4(mode="relay") (the default used by scripts/teleop_avp.py
and scripts/record_avp.py) then just connects to the relay port here, so the
headset stays paired across as many restarts of those scripts as you like --
the alternative, `--pico4-mode direct`, re-pairs the headset every time the
consuming script itself restarts, since direct mode owns the TCP server.

Usage:
    python scripts/pico4_daemon.py
    python scripts/teleop_avp.py --hand linker_l20 --input pico4   # separate terminal

If the default ports (63901 direct, 63902 relay, 29888 broadcast UDP) are
already in use by something else on this machine, pick different ones here
and pass the matching --pico4-relay-port etc. to the consuming script.
"""

from __future__ import annotations

import argparse
import errno
import logging
import socket
import struct
import threading
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anydexrt.pico import (  # noqa: E402
    _CMD_BATTERY,
    _CMD_CONNECT,
    _CMD_DEVICE_STATE_JSON,
    _CMD_HEARTBEAT,
    _CMD_SENSOR,
    _DirectFrameParser,
    _build_broadcast_packet,
    _get_local_ips,
)

logger = logging.getLogger("pico4_daemon")

DEFAULT_DIRECT_PORT = 63901
DEFAULT_RELAY_HOST = "127.0.0.1"
DEFAULT_RELAY_PORT = 63902
DEFAULT_BROADCAST_PORT = 29888
DEFAULT_DEVICE_ID = "pico4"
HEARTBEAT_TIMEOUT_S = 20.0
BROADCAST_INTERVAL_S = 5.0


def encode_relay_frame(device_id: str, payload: bytes) -> bytes:
    device_id_bytes = device_id.encode("utf-8")
    return (
        struct.pack("<I", len(device_id_bytes))
        + device_id_bytes
        + struct.pack("<I", len(payload))
        + payload
    )


class PortBindingError(RuntimeError):
    """Raised when one or more daemon TCP ports cannot be reserved."""

    def __init__(self, failures: list[tuple[str, str, int, OSError]]) -> None:
        self.failures = failures
        details = []
        for role, host, port, exc in failures:
            reason = "address already in use" if exc.errno == errno.EADDRINUSE else (exc.strerror or str(exc))
            details.append(f"  - {role} TCP {host}:{port}: {reason}")
        super().__init__(
            "Cannot start Pico 4 daemon:\n"
            + "\n".join(details)
            + "\nAnother relay daemon (this one or AnyDexRetarget's) may already be "
            "running, or something else is using these ports. Check with: "
            f"ss -ltnp | grep -E ':({DEFAULT_DIRECT_PORT}|{DEFAULT_RELAY_PORT})\\b'"
        )


class RelayHub:
    """Local TCP server that republishes JSON frames to every connected client."""

    def __init__(self, host: str, port: int, device_id: str) -> None:
        self._host = host
        self._port = port
        self._device_id = device_id
        self._stop = threading.Event()
        self._clients: set[socket.socket] = set()
        self._lock = threading.Lock()
        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None

    def bind(self) -> None:
        """Reserve the relay port synchronously so startup errors reach main."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((self._host, self._port))
            server.listen(8)
            server.settimeout(1.0)
        except OSError:
            server.close()
            raise
        self._server = server

    def start(self) -> None:
        if self._server is None:
            raise RuntimeError("RelayHub.bind() must be called before start()")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        server, self._server = self._server, None
        if server is not None:
            try:
                server.close()
            except OSError:
                pass
        with self._lock:
            for client in list(self._clients):
                try:
                    client.close()
                except OSError:
                    pass
            self._clients.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def publish(self, payload: bytes) -> None:
        frame = encode_relay_frame(self._device_id, payload)
        dead_clients: list[socket.socket] = []
        with self._lock:
            for client in self._clients:
                try:
                    client.sendall(frame)
                except OSError:
                    dead_clients.append(client)
            for client in dead_clients:
                self._clients.discard(client)
                try:
                    client.close()
                except OSError:
                    pass

    def _run(self) -> None:
        server = self._server
        if server is None:
            return
        logger.info("Relay hub listening on %s:%d", self._host, self._port)
        while not self._stop.is_set():
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            logger.info("Relay client connected: %s", addr)
            with self._lock:
                self._clients.add(conn)


class Pico4Daemon:
    def __init__(
        self,
        direct_port: int,
        relay_host: str,
        relay_port: int,
        broadcast_port: int,
        device_id: str,
    ) -> None:
        self._direct_port = direct_port
        self._broadcast_port = broadcast_port
        self._hub = RelayHub(relay_host, relay_port, device_id)
        self._stop = threading.Event()

    def run(self) -> None:
        server, failures = self._bind_servers()
        if failures:
            raise PortBindingError(failures)
        assert server is not None

        self._hub.start()
        broadcast_thread = threading.Thread(target=self._broadcast_loop, daemon=True)
        broadcast_thread.start()
        logger.info("Direct server listening on 0.0.0.0:%d", self._direct_port)

        try:
            while not self._stop.is_set():
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue
                logger.info("Pico 4 connected from %s", addr)
                self._handle_direct_client(conn)
                logger.info("Pico 4 disconnected")
        except KeyboardInterrupt:
            logger.info("Stopping daemon...")
        finally:
            self._stop.set()
            server.close()
            self._hub.stop()
            broadcast_thread.join(timeout=2.0)

    def _bind_servers(self) -> tuple[socket.socket | None, list[tuple[str, str, int, OSError]]]:
        """Bind both TCP endpoints before any background thread is started."""
        failures: list[tuple[str, str, int, OSError]] = []
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind(("0.0.0.0", self._direct_port))
            server.listen(1)
            server.settimeout(1.0)
        except OSError as exc:
            failures.append(("direct", "0.0.0.0", self._direct_port, exc))
            server.close()
            server = None

        try:
            self._hub.bind()
        except OSError as exc:
            failures.append(("relay", self._hub._host, self._hub._port, exc))

        if failures:
            if server is not None:
                server.close()
            self._hub.stop()
            return None, failures
        return server, failures

    def _handle_direct_client(self, conn: socket.socket) -> None:
        parser = _DirectFrameParser()
        conn.settimeout(1.0)
        last_heartbeat = time.monotonic()
        try:
            while not self._stop.is_set():
                if time.monotonic() - last_heartbeat > HEARTBEAT_TIMEOUT_S:
                    logger.warning("Pico 4 heartbeat timeout")
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
                        last_heartbeat = time.monotonic()
                    if frame["cmd"] == _CMD_DEVICE_STATE_JSON:
                        self._hub.publish(frame["payload"])
        finally:
            conn.close()

    def _broadcast_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            while not self._stop.is_set():
                for ip in _get_local_ips():
                    parts = ip.split(".")
                    if len(parts) != 4:
                        continue
                    broadcast_ip = ".".join(parts[:3]) + ".255"
                    packet = _build_broadcast_packet(ip)
                    try:
                        sock.sendto(packet, (broadcast_ip, self._broadcast_port))
                    except OSError:
                        pass
                self._stop.wait(BROADCAST_INTERVAL_S)
        finally:
            sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--direct-port", type=int, default=DEFAULT_DIRECT_PORT)
    parser.add_argument("--relay-host", default=DEFAULT_RELAY_HOST)
    parser.add_argument("--relay-port", type=int, default=DEFAULT_RELAY_PORT)
    parser.add_argument("--broadcast-port", type=int, default=DEFAULT_BROADCAST_PORT)
    parser.add_argument("--device-id", default=DEFAULT_DEVICE_ID)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    daemon = Pico4Daemon(
        direct_port=args.direct_port,
        relay_host=args.relay_host,
        relay_port=args.relay_port,
        broadcast_port=args.broadcast_port,
        device_id=args.device_id,
    )
    try:
        daemon.run()
    except PortBindingError as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
