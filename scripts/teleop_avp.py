"""Live teleoperation in simulation: Apple Vision Pro / PICO 4 -> AnyDexRT -> MuJoCo.

Streams the hand skeleton from the Vision Pro (avp_stream / Tracking Streamer
app) or a PICO 4 (anydexrt.pico, an independent reimplementation of its wire
protocol), retargets every frame with a trained AnyDexRT bundle, and shows the
robot hand in the MuJoCo viewer.

    python scripts/teleop_avp.py --hand linker_l20 --avp-ip 192.168.50.127
    python scripts/teleop_avp.py --hand wuji_hand  --avp-ip 192.168.50.127
    python scripts/teleop_avp.py --hand linker_l20 --input pico4

Without a headset, `--replay` feeds a recording through the same code path --
both AVP and PICO 4 recordings are the same pkl format (see
scripts/record_avp.py), so pass `--input` to match whichever device the
recording came from (default: avp):

    python scripts/teleop_avp.py --hand linker_l20 --replay
    python scripts/teleop_avp.py --hand linker_l20 --input pico4 --replay data/pico4_20261022.pkl

No calibration step: the palm frame is rebuilt from each live frame's own
skeleton, so the headset's position and the wrist frame's orientation do not
matter. The bundle was trained on avp1.pkl (one operator); for another operator,
record their hand and retrain with scripts/run_avp.py --avp <file>.

PICO 4 (`--input pico4`): uses anydexrt.pico.Pico4, which already outputs the
same (21, 3) MediaPipe-order, wrist-relative point format the AVP recordings
use (see anydexrt.data.palm_tips_from_mediapipe_points) -- no retraining
needed, same palm-frame construction, just a different capture device feeding
it. Two modes:

    --pico4-mode relay (default)   connect to a relay daemon already running on
                                   127.0.0.1:63902 -- start it first in another
                                   terminal: `python scripts/pico4_daemon.py`.
                                   Convenient for repeated restarts of this
                                   script without re-pairing the headset.
    --pico4-mode direct            this script itself opens the TCP server
                                   (63901) and UDP discovery broadcast (29888)
                                   the headset connects to -- no extra process,
                                   but the headset re-pairs on every restart.

Output is low-pass filtered in joint space by default (`--smoothing-hz 6`,
disable with `--smoothing-hz 0`) -- the raw NNS lookup hunts between nearby
table entries as the target moves, and pinch-template snapping is a discrete
jump; both are visible as jitter without this. See
anydexrt.inference.JointLowPassFilter. The lookup's own continuity prior
(NearestNeighborIK: 32 candidates, weight 0.5) does most of the work now --
raised from 8/0.05, which measured too weak to decide anything; see the README's
"Jitter with a still hand".

Pinch assist is OFF by default (`--pinch-mode off`). Three modes:

    --pinch-mode classifier   f_c decides contact on/off, hard pose-swap on
                              trigger. Needs a trained, working classifier --
                              as measured on the current bundles, the
                              ring/pinky contact classifiers have
                              precision/recall of 0.00 (they never fire), and
                              even where it works the snap replaces the whole
                              pose in one frame (see README errata).
    --pinch-mode continuous  NO classifier; blends toward the pinch template
                              in proportion to the REAL thumb-finger distance
                              (dominant finger only), so a 1cm/2cm partial
                              pinch produces a proportionally part-closed pose
                              instead of a jump. Trained with a CrossFingerMapper
                              (TrainConfig.mapper_type, default since the pinch
                              comparison -- a plain FingerMapper's mapped gap
                              barely tracks the real one).
    --pinch-mode off          (default) raw f_m output, no pinch assist.

Real hardware (Linker L20 only), via anydexrt.real_l20:

    python scripts/teleop_avp.py --hand linker_l20 --real                  # dry run
    python scripts/teleop_avp.py --hand linker_l20 --real --confirm-real   # moves the hand

`--real` alone is NOT enough to move the physical hand -- it also requires
`--confirm-real` as a second, explicit opt-in, so a copy-pasted command line
cannot accidentally drive real hardware. Without `--confirm-real`, `--real`
runs dry: the 30 position registers are computed and printed, the serial port
is never opened. Run dry first and watch the registers move sensibly.

Joint LIMITS need no conversion -- anydexRT's L20 limits equal the driver's
per-joint (checked before this flag existed) -- but the qpos->register mapping
is not the identity: it applies GEORT's measured offsets (the four finger MCP
pitches are shifted -8.7 deg, i.e. simulation zero is not the hand's zero),
a measured piecewise curve for three thumb joints, and per-joint direction
inversion. See anydexrt/real_l20.py.

Default serial settings (port /dev/ttyUSB0, baud 460800, slave id 0x29 for the
right hand, speed 200, current_limit 200) are conservative values for first
bring-up; override with --l20-port, --l20-slave-id etc. Add --l20-open-on-exit
to leave the hand open rather than holding its last pose, and
--l20-clear-faults if it came up latched after a stall.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anydexrt.data import VP_TO_MEDIAPIPE  # noqa: E402
from anydexrt.inference import load_bundle  # noqa: E402

# A recording to fall back on for --replay. Data only -- no code from that
# repository is imported any more (the L20 driver was the last one; it now
# lives in anydexrt/real_l20.py). Override with --replay <file>.
DEFAULT_REPLAY = "/home/qsq/qqsq12321/AnyDexRetarget/example/data/avp1.pkl"

# How far a fingertip must reach from the palm origin before the pinch blend
# engages for it (--pinch-min-extension-mm), per capture device.
#
# PICO 4 needs a lower gate than AVP because it measures the same hand as
# reaching 20-35mm less far: on recordings, median fingertip reach is 140 /
# 153 / 137 / 139mm (index..pinky) vs AVP's 171 / 176 / 166 / 148mm. At the
# AVP gate of 110mm, pinching the ring or pinky -- where the finger is already
# partly curled -- drops below the gate and the blend never engages: the robot
# fingertips stay 12mm and 20mm apart at full human pinch, versus 7mm and 8mm
# at a 92mm gate. The cost of a lower gate is a curled finger occasionally
# reading as "reaching": on AVP, frames with a bent thumb that pick up a pinch
# weight above 0.3 go from 11.1% at 110mm to 14.8% at 90mm, which is why AVP
# keeps its own value rather than both sharing one.
PINCH_MIN_EXTENSION_MM = {"avp": 110.0, "pico4": 90.0}


class LiveAVP:
    """Latest (25, 4, 4) wrist-local hand skeleton from the Vision Pro."""

    def __init__(self, ip: str, side: str):
        from avp_stream import VisionProStreamer

        self.streamer = VisionProStreamer(ip=ip, record=False)
        self.key = f"{side}_fingers"

    def poses(self) -> np.ndarray | None:
        data = self.streamer.get_latest()
        if data is None:
            return None
        try:
            return np.asarray(data[self.key], dtype=np.float64)
        except (KeyError, TypeError):
            return None

    def close(self) -> None:
        stop = getattr(self.streamer, "_stop_pose_streaming", None)
        if callable(stop):
            stop()


class ReplayRecording:
    """A (21, 3) MediaPipe-landmark recording, looped -- same pkl format
    regardless of capture device (see scripts/record_avp.py), since the live
    path for both AVP and PICO 4 already reduces to this exact representation.

    `pad_to_avp=True` (AVP replay) additionally lifts the 21 points back to
    the 25-joint AVP skeleton for `retarget_avp_poses` -- metacarpal joints
    (5, 10, 15, 20), which the palm frame does not use, are filled with
    wrist-MCP midpoints. `pad_to_avp=False` (PICO 4 replay) returns the 21
    points directly, for `retarget_mediapipe_points` -- PICO's live path
    already consumes this exact format, no padding needed.
    """

    def __init__(self, path: str, side: str, pad_to_avp: bool = True):
        with open(path, "rb") as fh:
            frames = pickle.load(fh)
        self.frames = [np.asarray(f[f"{side}_fingers"], dtype=np.float64) for f in frames]
        self.i = 0
        self.pad_to_avp = pad_to_avp

    def poses(self) -> np.ndarray:
        mp = self.frames[self.i]
        self.i = (self.i + 1) % len(self.frames)
        if not self.pad_to_avp:
            return mp
        poses = np.tile(np.eye(4), (25, 1, 1))
        for mp_idx, vp_idx in enumerate(VP_TO_MEDIAPIPE):
            poses[vp_idx, :3, 3] = mp[mp_idx]
        for meta, mcp in ((5, 6), (10, 11), (15, 16), (20, 21)):
            poses[meta, :3, 3] = 0.5 * poses[mcp, :3, 3]
        return poses

    def close(self) -> None:
        pass


class LivePico:
    """Latest (21, 3) wrist-relative MediaPipe hand skeleton from a PICO 4.

    Thin wrapper around anydexrt.pico.Pico4, which already does the SDK's
    26-joint -> 21-point MediaPipe reduction and wrist-relative normalization.
    Constructing `Pico4(...)` starts its background connect/listen thread
    immediately -- no separate start() call.
    """

    def __init__(self, side: str, mode: str, relay_host: str, relay_port: int,
                 port: int, broadcast_port: int):
        from anydexrt.pico import Pico4

        self.dev = Pico4(mode=mode, relay_host=relay_host, relay_port=relay_port,
                          port=port, broadcast_port=broadcast_port)
        self.key = f"{side}_fingers"

    def poses(self) -> np.ndarray | None:
        arr = self.dev.get_fingers_data()[self.key]
        return arr if np.any(arr) else None  # all-zero = no frame received yet

    def close(self) -> None:
        self.dev.stop()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand", default="linker_l20", choices=["linker_l20", "wuji_hand"])
    ap.add_argument("--input", default="avp", choices=["avp", "pico4"],
                    help="live input device (default: avp). --replay is AVP-only.")
    ap.add_argument("--avp-ip", default="10.10.10.126")
    ap.add_argument("--side", default="right", choices=["right", "left"],
                    help="left is mirrored onto the right-hand model")
    ap.add_argument("--replay", nargs="?", const=DEFAULT_REPLAY, default=None,
                    help="feed a recording instead of the headset (default: avp1.pkl). Works "
                         "for both --input avp and --input pico4 recordings (same pkl format).")
    pico4 = ap.add_argument_group("pico4", "only used with --input pico4")
    pico4.add_argument("--pico4-mode", default="relay", choices=["relay", "direct"],
                       help="relay daemon (default; run scripts/pico4_daemon.py first) or "
                            "direct TCP server (self-contained, no extra process)")
    pico4.add_argument("--pico4-relay-host", default="127.0.0.1")
    pico4.add_argument("--pico4-relay-port", type=int, default=63902)
    pico4.add_argument("--pico4-port", type=int, default=63901)
    pico4.add_argument("--pico4-broadcast-port", type=int, default=29888)
    ap.add_argument("--bundle", default=None, help="default: results/avp/<hand>/<hand>_bundle.pt")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--smoothing-hz", type=float, default=6.0,
                    help="joint-space low-pass cutoff; 0 disables (raw NNS output, visibly jittery)")
    ap.add_argument("--pinch-mode", default="off", choices=["off", "classifier", "continuous"],
                    help="how f_c/pinch-templates refine the mapped pose (default: off). "
                         "'classifier': f_c decides contact on/off, hard pose-swap on trigger -- "
                         "ring/pinky heads currently have 0 precision/recall, and even when it "
                         "fires it snaps instantly (see README errata). "
                         "'continuous': NO classifier; blends toward the pinch template in "
                         "proportion to the REAL thumb-finger distance (dominant finger only), "
                         "so a 1cm/2cm partial pinch produces a proportionally part-closed pose "
                         "instead of a jump -- works best with a CrossFingerMapper bundle. "
                         "'off': raw f_m output, no pinch assist at all.")
    ap.add_argument("--pinch-open-gap-mm", type=float, default=80.0,
                    help="'continuous' mode only: thumb-finger distance (mm) treated as the "
                         "fully-open reference (alpha=0). If the bundle's extra['open_gap_mm'] "
                         "is present, that is used per-finger instead and this flag is ignored.")
    ap.add_argument("--pinch-closed-gap-mm", type=float, default=6.0,
                    help="'continuous' mode only: thumb-finger distance (mm) treated as the "
                         "fully-pinched reference (alpha=1). Real AVP tracking never reports "
                         "exactly 0mm even at a hard pinch (finger meshes have physical "
                         "thickness); defaults to 6mm based on recorded sessions. Raise this if "
                         "the robot still isn't closing all the way, lower it if it over-closes "
                         "before your fingers actually touch.")
    ap.add_argument("--pinch-min-extension-mm", type=float, default=None,
                    help="'continuous' mode only: a finger must reach at least this far from the "
                         "palm origin before the pinch blend engages for it. Without this, making "
                         "a fist can accidentally trigger a pinch (a curled finger's tip can land "
                         "within a few mm of the thumb even though it's curled INTO the palm, not "
                         "reaching OUT toward the thumb). Defaults per device (see "
                         "PINCH_MIN_EXTENSION_MM): 110mm for AVP, 90mm for PICO 4.")
    ap.add_argument("--fist-thumb-rest-mm", type=float, default=110.0,
                    help="'continuous' mode only: with all four fingers curled, a human thumb tip "
                         "this close to the palm origin (or closer) counts as wrapped in a fist and "
                         "the robot thumb goes fully to calibrated_fist_pose.yaml.")
    ap.add_argument("--fist-thumb-extended-mm", type=float, default=125.0,
                    help="...and at this distance or farther the thumb counts as extended, no fist "
                         "blend (recordings: fist 95-105mm, extended 130-145mm). 0 disables the "
                         "fist blend entirely. Only used as a FALLBACK -- the live paths pass a "
                         "thumb straightness ratio instead, which works on both AVP and PICO 4 "
                         "(a mm threshold only separates the two poses on AVP). The status line "
                         "prints thumb_reach live.")
    ap.add_argument("--fist-thumb-bent", type=float, default=0.92,
                    help="With all four fingers curled, a thumb straightness "
                         "(anydexrt.data.thumb_straightness, 1.0 = MCP/IP/TIP collinear) at or "
                         "below this counts as wrapped in a fist: the robot thumb goes fully to "
                         "calibrated_fist_pose.yaml.")
    ap.add_argument("--fist-thumb-straight", type=float, default=0.95,
                    help="...and at or above this the thumb counts as extended, no fist blend. "
                         "Measured on both devices: curled < 0.88, extended > 0.94. The status "
                         "line prints the live value as 'straight'. Both peaks are the same "
                         "on AVP and PICO 4: curled below 0.88, valley at 0.90, extended above "
                         "0.94. Lower --fist-thumb-bent toward 0.88 for a stricter fist.")
    ap.add_argument("--fist-open-reach-mm", type=float, default=110.0,
                    help="How far a fingertip must be from the palm origin to count as fully "
                         "open for the FIST blend (the other half of its weight, alongside "
                         "--fist-thumb-bent). Deliberately separate from --pinch-min-extension-mm: "
                         "that one drops to ~90 on PICO 4, which would leave the fist only a 10mm "
                         "window to ramp over instead of 30mm.")
    ap.add_argument("--real", action="store_true",
                    help="also drive the physical Linker L20 over RS485. Needs --confirm-real "
                         "too, or runs dry (prints registers, sends nothing).")
    ap.add_argument("--confirm-real", action="store_true",
                    help="required alongside --real to actually write to the serial port")
    ap.add_argument("--l20-port", default="/dev/ttyUSB0")
    ap.add_argument("--l20-baudrate", type=int, default=460800)
    ap.add_argument("--l20-slave-id", type=lambda s: int(s, 0), default=0x29,
                    help="0x29 right hand, 0x2A left (accepts 0x.. or decimal)")
    ap.add_argument("--l20-speed", type=int, default=200, help="conservative default for first bring-up")
    ap.add_argument("--l20-current-limit", type=int, default=200)
    ap.add_argument("--l20-clear-faults", action="store_true",
                    help="pulse the clear-fault registers during startup; use if the hand "
                         "came up latched after a stall or an e-stop")
    ap.add_argument("--l20-open-on-exit", action="store_true",
                    help="command an open palm before closing the serial port, so the hand "
                         "is not left holding a clenched pose")
    args = ap.parse_args()

    if args.real and args.hand != "linker_l20":
        print("--real currently only has a driver for linker_l20", file=sys.stderr)
        return 1

    bundle = args.bundle or ROOT / "results/avp" / args.hand / f"{args.hand}_bundle.pt"
    # A bundle may carry its own per-finger open-hand reference in
    # extra['open_gap_mm'] (estimated from that bundle's OWN training
    # recording's 90th-percentile spread); prefer that over the flat
    # --pinch-open-gap-mm default, which is a guess if the bundle doesn't say
    # otherwise.
    open_gap = args.pinch_open_gap_mm
    try:
        _extra = torch.load(bundle, map_location="cpu", weights_only=False).get("extra", {})
        if "open_gap_mm" in _extra:
            open_gap = _extra["open_gap_mm"]
            print(f"using bundle's own open-hand gap estimate: {[round(g, 1) for g in open_gap]} mm")
    except Exception:
        pass  # fall back to the flat default; load_bundle will raise its own error below if the file is bad
    rt = load_bundle(bundle, smoothing_hz=args.smoothing_hz or None,
                     use_contacts=(args.pinch_mode == "classifier"),
                     pinch_mode=args.pinch_mode, pinch_open_gap_mm=open_gap,
                     pinch_closed_gap_mm=args.pinch_closed_gap_mm,
                     pinch_min_extension_mm=(args.pinch_min_extension_mm
                                             if args.pinch_min_extension_mm is not None
                                             else PINCH_MIN_EXTENSION_MM[args.input]),
                     fist_thumb_rest_mm=args.fist_thumb_rest_mm,
                     fist_thumb_extended_mm=args.fist_thumb_extended_mm,
                     fist_thumb_bent=args.fist_thumb_bent,
                     fist_thumb_straight=args.fist_thumb_straight,
                     fist_open_reach_mm=args.fist_open_reach_mm)
    kin = rt.kin
    if args.replay:
        source = ReplayRecording(args.replay, args.side, pad_to_avp=(args.input == "avp"))
        print(f"{args.hand} <- replay {args.replay} ({args.input}, {args.side} hand)")
    elif args.input == "pico4":
        source = LivePico(args.side, args.pico4_mode, args.pico4_relay_host, args.pico4_relay_port,
                           args.pico4_port, args.pico4_broadcast_port)
        print(f"{args.hand} <- PICO 4 ({args.pico4_mode} mode) ({args.side} hand)")
    else:
        source = LiveAVP(args.avp_ip, args.side)
        print(f"{args.hand} <- AVP {args.avp_ip} ({args.side} hand)")

    real_driver = None
    if args.real:
        from anydexrt.real_l20 import LinkerL20Output

        dry_run = not args.confirm_real
        if dry_run:
            print("--real given WITHOUT --confirm-real: dry run, registers computed and "
                  "printed but NOT sent to the hand. Add --confirm-real to actually move it.")
        else:
            print(f"*** --confirm-real set: WILL send live commands to {args.l20_port} "
                  f"(slave {hex(args.l20_slave_id)}). Make sure the hand's range of motion is "
                  "clear before continuing. ***")
        real_driver = LinkerL20Output(
            port=args.l20_port, baudrate=args.l20_baudrate, slave_id=args.l20_slave_id,
            speed=args.l20_speed, current_limit=args.l20_current_limit,
            clear_faults=args.l20_clear_faults, open_on_exit=args.l20_open_on_exit,
            print_registers=dry_run, dry_run=dry_run,
        )

    model = kin.model
    data = mujoco.MjData(model)
    qpos = kin.rest_qpos()
    data.qpos[:] = kin._expand_drives(qpos)
    mujoco.mj_forward(model, data)
    origin, R = kin.palm_frame()
    mirror = np.array([-1.0, 1.0, 1.0]) if args.side == "left" else None

    frames, lost, t_report = 0, 0, time.perf_counter()
    # Both LiveAVP's and LivePico's poses() are non-blocking: they return
    # whatever snapshot is currently cached, not "the next new frame". If the
    # headset delivers slower than this loop polls, the same snapshot gets
    # read (and retargeted) several times in a row -- the robot holds a pose,
    # then jumps when a new sample finally lands, which looks identical to
    # over-smoothing but has nothing to do with the filter. Comparing
    # consecutive raw arrays tells the two apart: see the printed "new
    # samples" rate below.
    fresh, stale, window_t0 = 0, 0, time.perf_counter()
    prev_raw = None
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            is_l20 = "l20" in args.hand
            viewer.cam.lookat[:] = origin + R @ np.array([0.15 if is_l20 else 0.10, 0.0, 0.0])
            viewer.cam.distance = 0.45 if is_l20 else 0.35
            fwd = -(R @ np.array([-0.2, -0.3, -0.9]))
            fwd /= np.linalg.norm(fwd)
            viewer.cam.azimuth = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
            viewer.cam.elevation = float(np.degrees(np.arcsin(fwd[2])))

            while viewer.is_running():
                t0 = time.perf_counter()
                poses = source.poses()
                if poses is not None:
                    if prev_raw is not None and np.array_equal(poses, prev_raw):
                        stale += 1
                    else:
                        fresh += 1
                    prev_raw = poses
                    if mirror is not None:
                        poses = poses.copy()
                        if args.input == "pico4":
                            poses *= mirror  # (21, 3) points, no rotation component to skip
                        else:
                            poses[:, :3, 3] *= mirror
                    try:
                        qpos = (rt.retarget_mediapipe_points(poses) if args.input == "pico4"
                                else rt.retarget_avp_poses(poses))
                        frames += 1
                    except ValueError:
                        # Untracked hand: source sends a zero/identity/degenerate
                        # placeholder, whose palm is degenerate. Hold the last pose.
                        lost += 1
                data.qpos[:] = kin._expand_drives(qpos)
                mujoco.mj_forward(model, data)
                viewer.sync()
                if real_driver is not None:
                    real_driver.send(qpos, kin.hand.joint_names)

                if time.perf_counter() - t_report > 2.0:
                    elapsed = time.perf_counter() - window_t0
                    contact = rt.last_contacts
                    pinch = ",".join(kin.hand.finger_names[i + 1] for i in np.where(contact)[0]) \
                        if contact is not None and contact.any() else "-"
                    print(f"  {frames} frames, {lost} untracked | new samples: {fresh} "
                          f"({fresh / elapsed:.1f} Hz), repeated reads: {stale} | pinch: {pinch}")
                    fist = rt.last_fist
                    if fist is not None:
                        print(f"    fist: finger_reach_max={fist['finger_reach_max_mm']:.0f}mm "
                              f"thumb_reach={fist['thumb_reach_mm']:.0f}mm "
                              + (f"straight={fist['straightness']:.2f} "
                                 if fist['straightness'] is not None else "")
                              + f"alpha={fist['alpha']:.2f}")
                    fresh, stale, window_t0 = 0, 0, time.perf_counter()
                    t_report = time.perf_counter()
                time.sleep(max(0.0, 1.0 / args.fps - (time.perf_counter() - t0)))
    finally:
        source.close()
        if real_driver is not None:
            real_driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
