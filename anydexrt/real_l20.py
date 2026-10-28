"""RS485 Modbus output for the Linker L20 hand.

An independent reimplementation of the wire protocol and the GEORT-calibrated
qpos->register mapping, so the live path does not import from another
repository. The protocol constants (register layout, Modbus framing, CRC-16,
baud, slave ids) and the calibration numbers are what the hand's firmware
actually expects, so they are copied exactly; the code is written here.

The mapping is NOT "send qpos as-is". Three things sit between a simulation
joint value and the byte the hand receives, all of them measured against the
physical hand by GEORT rather than derived from the URDF:

  1. **Offsets** (`_JOINT_OFFSETS_RAD`). The four finger MCP pitches get
     -0.152 rad (-8.7 deg), i.e. simulation zero is not the real hand's zero.
     The thumb CMC roll offset is tunable and now OFF by default
     (`thumb_cmc_roll_offset`, was 0.15 in the original driver) because it
     decides how far out the thumb sits at rest and can only be judged on a
     physical hand.
  2. **Piecewise alignment** for the three thumb joints that were measured at
     three points each (`_QPOS_ALIGNMENT_POINTS`). Interpolating through them
     differs from linear limit normalization by up to 0.10 of full scale
     (thumb CMC roll at qpos 0, thumb MCP at 0.3) -- not a rounding
     difference, which is why the measured curve is used where it exists and
     linear normalization only elsewhere.
  3. **Direction inversion** (`_INVERTED_JOINTS`). Modbus counts the opposite
     way from URDF qpos for the thumb and for every pitch/PIP joint. Finger
     MCP roll is deliberately NOT inverted, matching the register protocol.

Joint limits are the one thing that needs no conversion: anydexRT's L20 limits
equal this driver's per-joint, checked before `--real` was added (thumb CMC
roll 1.39, PIP 1.7416 constrained by the DIP mimic 0.89, etc.), so a qpos that
is in range in simulation is in range here.

Each position command is 30 registers in the order
`roll(5) + yaw(5) + root1(5) + distal(5) + distal(5) + distal(5)` -- the distal
block is repeated three times because PIP, DIP and the fingertip segment are
driven by one value. Only the thumb has a real roll register; the other four
carry `_NEUTRAL_ROLL`.
"""

from __future__ import annotations

import time

import numpy as np

PORT = "/dev/ttyUSB0"
BAUDRATE = 460800
RIGHT_SLAVE_ID = 0x29
LEFT_SLAVE_ID = 0x2A
SPEED = 200
CURRENT_LIMIT = 200
_NEUTRAL_ROLL = 128
_NEUTRAL_YAW = 128

# L20 V10 active-joint order used by the real-hand mapper. Matched against
# anydexrt's own joint names by suffix, so `thumb_cmc_yaw` finds THUMB_CMC_YAW.
_ACTIVE_JOINT_ORDER = (
    "THUMB_CMC_YAW",
    "THUMB_CMC_ROLL",
    "THUMB_CMC_PITCH",
    "THUMB_MCP",
    "INDEX_MCP_ROLL",
    "INDEX_MCP_PITCH",
    "INDEX_PIP",
    "MIDDLE_MCP_ROLL",
    "MIDDLE_MCP_PITCH",
    "MIDDLE_PIP",
    "RING_MCP_ROLL",
    "RING_MCP_PITCH",
    "RING_PIP",
    "PINKY_MCP_ROLL",
    "PINKY_MCP_PITCH",
    "PINKY_PIP",
)

# Effective limits from the L20 V10 URDF; PIP is constrained by the DIP mimic
# (0.89). Equal to anydexrt's own per-joint limits -- the check that let qpos
# be used without a unit-conversion layer.
_JOINT_LIMITS = {
    "THUMB_CMC_YAW": (0.0, 1.57),
    "THUMB_CMC_ROLL": (0.0, 1.39),
    "THUMB_CMC_PITCH": (0.0, 0.83),
    "THUMB_MCP": (0.0, 1.25),
    "INDEX_MCP_ROLL": (-0.23, 0.23),
    "INDEX_MCP_PITCH": (0.0, 1.22),
    "INDEX_PIP": (0.0, 1.7415730337078652),
    "MIDDLE_MCP_ROLL": (-0.23, 0.23),
    "MIDDLE_MCP_PITCH": (0.0, 1.22),
    "MIDDLE_PIP": (0.0, 1.7415730337078652),
    "RING_MCP_ROLL": (-0.23, 0.23),
    "RING_MCP_PITCH": (0.0, 1.22),
    "RING_PIP": (0.0, 1.7415730337078652),
    "PINKY_MCP_ROLL": (-0.23, 0.23),
    "PINKY_MCP_PITCH": (0.0, 1.22),
    "PINKY_PIP": (0.0, 1.7415730337078652),
}

# Modbus counts opposite to URDF qpos for these. Finger MCP roll is
# intentionally absent: the register protocol agrees with qpos there.
_INVERTED_JOINTS = frozenset({
    "THUMB_CMC_YAW",
    "THUMB_CMC_ROLL",
    "THUMB_CMC_PITCH",
    "THUMB_MCP",
    "INDEX_MCP_PITCH",
    "INDEX_PIP",
    "MIDDLE_MCP_PITCH",
    "MIDDLE_PIP",
    "RING_MCP_PITCH",
    "RING_PIP",
    "PINKY_MCP_PITCH",
    "PINKY_PIP",
})

# Applied to qpos before normalization (and before the alignment curve, for the
# joints that have one): simulation zero is not the hand's zero.
#
# THUMB_CMC_ROLL is off by default, where the original driver had +0.15. That
# value pinned qpos=0 to 0.1396 -- the left point of this joint's alignment
# curve -- so a fully open thumb commanded 0.009, essentially the hard stop.
#
# The trade, which only a physical hand settles: with no offset the thumb sits
# further out through the usable range (register 220 -> 242 at qpos 0.2) but no
# longer reaches the end of travel, since qpos 1.39 now maps to register 9
# rather than 0. If the thumb cannot close far enough to complete a pinch, that
# is this. Raise it per run with LinkerL20Output(thumb_cmc_roll_offset=...),
# i.e. `teleop_avp.py --l20-thumb-roll-offset 0.05`.
_DEFAULT_THUMB_CMC_ROLL_OFFSET_RAD = 0.0

_JOINT_OFFSETS_RAD = {
    "THUMB_CMC_ROLL": _DEFAULT_THUMB_CMC_ROLL_OFFSET_RAD,
    "INDEX_MCP_PITCH": -0.15235988,
    "MIDDLE_MCP_PITCH": -0.15235988,
    "RING_MCP_PITCH": -0.15235988,
    "PINKY_MCP_PITCH": -0.15235988,
}

# Measured (simulation qpos -> normalized command) pairs for the thumb joints
# that were calibrated against the physical hand. Used instead of linear limit
# normalization for these three; they differ by up to 0.10 of full scale.
_QPOS_ALIGNMENT_POINTS = {
    "THUMB_CMC_ROLL": (
        (0.13962634015954636, 0.0),
        (0.7299065850398866, 0.5),
        (1.4423598775598299, 1.0),
    ),
    "THUMB_CMC_PITCH": (
        (0.05235987755982989, 0.0),
        (0.4499065850398866, 0.5),
        (0.8823598775598299, 1.0),
    ),
    "THUMB_MCP": (
        (0.13962634015954636, 0.0),
        (0.67735987755983, 0.5),
        (1.25, 1.0),
    ),
}


def _clamp_uint8(value: float) -> int:
    return int(np.clip(int(round(value)), 0, 255))


class LinkerL20RS485:
    """Modbus-over-RS485 transport: framing, CRC-16, register writes."""

    POSITION_START, POSITION_COUNT = 0, 30
    SPEED_START, SPEED_COUNT = 30, 30
    CLEAR_FAULT_START, CLEAR_FAULT_COUNT = 90, 30
    CURRENT_LIMIT_START, CURRENT_LIMIT_COUNT = 150, 30

    def __init__(
        self,
        port: str = PORT,
        baudrate: int = BAUDRATE,
        slave_id: int = RIGHT_SLAVE_ID,
        timeout: float = 0.05,
    ):
        try:
            import serial
        except ImportError as exc:
            raise ImportError(
                "pyserial is required to drive the Linker L20: pip install pyserial"
            ) from exc

        self.slave_id = int(slave_id)
        self.port = port
        self.baudrate = int(baudrate)
        self._serial = serial.Serial(
            port=port, baudrate=baudrate, bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=timeout,
        )

    @staticmethod
    def crc16(data: bytes) -> int:
        """Modbus CRC-16 (poly 0xA001, reflected), returned LSB-first by callers."""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
        return crc

    def _request(self, payload: bytes, expected_len: int) -> bytes:
        crc = self.crc16(payload)
        self._serial.reset_input_buffer()
        self._serial.write(payload + bytes([crc & 0xFF, (crc >> 8) & 0xFF]))
        self._serial.flush()
        response = self._serial.read(expected_len)
        if len(response) != expected_len:
            raise TimeoutError(
                f"incomplete Modbus response: expected {expected_len} bytes, got {len(response)}"
            )
        if self.crc16(response[:-2]) != (response[-2] | (response[-1] << 8)):
            raise RuntimeError("CRC mismatch in Modbus response")
        if response[0] != self.slave_id:
            raise RuntimeError(
                f"response from slave id {response[0]}, expected {self.slave_id}"
            )
        if response[1] & 0x80:
            raise RuntimeError(f"Modbus exception code {response[2]}")
        return response

    def write_registers(self, start_address: int, values: list[int]) -> None:
        """Function 0x10 (write multiple), one uint8 per register, high byte 0."""
        registers = [_clamp_uint8(v) for v in values]
        payload = bytearray([
            self.slave_id, 0x10,
            (start_address >> 8) & 0xFF, start_address & 0xFF,
            (len(registers) >> 8) & 0xFF, len(registers) & 0xFF,
            len(registers) * 2,
        ])
        for value in registers:
            payload.extend([0x00, value])
        self._request(bytes(payload), expected_len=8)

    def configure_motion_profile(
        self, speed: int, current_limit: int, clear_faults: bool = False
    ) -> None:
        self.write_registers(self.SPEED_START, [speed] * self.SPEED_COUNT)
        self.write_registers(self.CURRENT_LIMIT_START, [current_limit] * self.CURRENT_LIMIT_COUNT)
        if clear_faults:
            self.write_registers(self.CLEAR_FAULT_START, [1] * self.CLEAR_FAULT_COUNT)
            time.sleep(0.02)
            self.write_registers(self.CLEAR_FAULT_START, [0] * self.CLEAR_FAULT_COUNT)

    def set_positions(self, registers: list[int]) -> None:
        if len(registers) != self.POSITION_COUNT:
            raise ValueError(
                f"L20 position command needs {self.POSITION_COUNT} registers, got {len(registers)}"
            )
        self.write_registers(self.POSITION_START, registers)

    def close(self) -> None:
        if self._serial.is_open:
            self._serial.close()


class LinkerL20Output:
    """anydexRT drives (qpos) -> L20 registers, optionally over the wire.

    `dry_run=True` computes and reports registers without opening the serial
    port, which is what `teleop_avp.py --real` does unless `--confirm-real` is
    also given.
    """

    def __init__(
        self,
        port: str = PORT,
        baudrate: int = BAUDRATE,
        slave_id: int = RIGHT_SLAVE_ID,
        speed: int = SPEED,
        current_limit: int = CURRENT_LIMIT,
        clear_faults: bool = False,
        open_on_exit: bool = False,
        print_registers: bool = False,
        dry_run: bool = False,
        thumb_cmc_roll_offset: float = _DEFAULT_THUMB_CMC_ROLL_OFFSET_RAD,
    ):
        self.dry_run = bool(dry_run)
        self._offsets = dict(_JOINT_OFFSETS_RAD)
        self._offsets["THUMB_CMC_ROLL"] = float(thumb_cmc_roll_offset)
        self.print_registers = bool(print_registers)
        self._open_on_exit = bool(open_on_exit)
        self._last_registers: list[int] | None = None
        self._index_by_suffix: dict[str, int] = {}
        self._index_key: tuple[str, ...] | None = None
        self.send_count = 0

        if self.dry_run:
            self._hand = None
            print("L20 dry run: registers computed, serial output disabled.")
            return

        self._hand = LinkerL20RS485(port=port, baudrate=baudrate, slave_id=slave_id)
        try:
            self._hand.configure_motion_profile(speed, current_limit, clear_faults)
        except Exception as exc:
            self._hand.close()
            raise RuntimeError(
                f"Linker L20 did not respond on startup (port={port}, baudrate={baudrate}, "
                f"slave_id={slave_id}). Check hand power, RS485 wiring, and these settings."
            ) from exc
        print(
            f"Linker L20 connected: {port} @ {baudrate}, slave_id={slave_id}, "
            f"speed={speed}, current_limit={current_limit}"
        )

    def _resolve_joint_indices(self, joint_names) -> None:
        """Map each L20 register joint to its index in `joint_names`, by suffix."""
        key = tuple(str(n) for n in joint_names)
        if key == self._index_key:
            return
        upper = [n.upper() for n in key]
        resolved = {}
        for suffix in _ACTIVE_JOINT_ORDER:
            matches = [i for i, n in enumerate(upper) if n.endswith(suffix)]
            if not matches:
                raise ValueError(f"no joint ending in {suffix} among {key}")
            if len(matches) > 1:
                raise ValueError(f"joint suffix {suffix} is ambiguous: {[key[i] for i in matches]}")
            resolved[suffix] = matches[0]
        self._index_by_suffix, self._index_key = resolved, key

    def _normalized(self, qpos: np.ndarray, suffix: str) -> float:
        """One joint's qpos -> [0, 1] command, before direction inversion."""
        value = float(qpos[self._index_by_suffix[suffix]]) + self._offsets.get(suffix, 0.0)
        alignment = _QPOS_ALIGNMENT_POINTS.get(suffix)
        if alignment is not None:
            sim_qpos, commanded = zip(*alignment)
            return float(np.interp(value, sim_qpos, commanded))
        lower, upper = _JOINT_LIMITS[suffix]
        if upper <= lower:
            return 0.5
        return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))

    def qpos_to_registers(self, qpos, joint_names) -> list[int]:
        """(J,) drives -> the 30 position registers. Pure; safe to call dry."""
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(qpos)):
            raise ValueError("qpos contains NaN or Inf -- refusing to build a command")
        self._resolve_joint_indices(joint_names)

        def reg(suffix: str) -> int:
            ratio = self._normalized(qpos, suffix)
            if suffix in _INVERTED_JOINTS:
                ratio = 1.0 - ratio
            return _clamp_uint8(ratio * 255.0)

        roll = [reg("THUMB_CMC_ROLL")] + [_NEUTRAL_ROLL] * 4
        yaw = [reg(s) for s in ("THUMB_CMC_YAW", "INDEX_MCP_ROLL", "MIDDLE_MCP_ROLL",
                                "RING_MCP_ROLL", "PINKY_MCP_ROLL")]
        root1 = [reg(s) for s in ("THUMB_CMC_PITCH", "INDEX_MCP_PITCH", "MIDDLE_MCP_PITCH",
                                  "RING_MCP_PITCH", "PINKY_MCP_PITCH")]
        distal = [reg(s) for s in ("THUMB_MCP", "INDEX_PIP", "MIDDLE_PIP",
                                   "RING_PIP", "PINKY_PIP")]
        return roll + yaw + root1 + distal * 3

    @staticmethod
    def open_palm_registers() -> list[int]:
        return [_NEUTRAL_ROLL] * 5 + [_NEUTRAL_YAW] * 5 + [255] * 20

    def send(self, qpos, joint_names) -> None:
        """Build and (unless dry) write one position command.

        Identical consecutive commands are skipped -- the retargeter runs far
        faster than the hand needs and the registers are only 8-bit, so most
        frames produce the same bytes.
        """
        registers = self.qpos_to_registers(qpos, joint_names)
        if registers == self._last_registers:
            return
        if self.print_registers or self.dry_run:
            print(f"L20 registers: {registers}")
        if self._hand is not None:
            self._hand.set_positions(registers)
        self._last_registers = registers
        self.send_count += 1

    def close(self) -> None:
        if self._hand is None:
            return
        try:
            if self._open_on_exit:
                self._hand.set_positions(self.open_palm_registers())
                time.sleep(0.2)
        finally:
            self._hand.close()


__all__ = ["LinkerL20Output", "LinkerL20RS485", "PORT", "BAUDRATE",
           "RIGHT_SLAVE_ID", "LEFT_SLAVE_ID", "SPEED", "CURRENT_LIMIT",
           "_DEFAULT_THUMB_CMC_ROLL_OFFSET_RAD"]
