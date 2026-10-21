"""Live teleoperation in simulation: Apple Vision Pro -> AnyDexRT -> MuJoCo.

Streams the hand skeleton from the Vision Pro (avp_stream / Tracking Streamer
app), retargets every frame with a trained AnyDexRT bundle, and shows the robot
hand in the MuJoCo viewer.

    python scripts/teleop_avp.py --hand linker_l20 --avp-ip 192.168.50.127
    python scripts/teleop_avp.py --hand wuji_hand  --avp-ip 192.168.50.127

Without a headset, `--replay` feeds a recording through the same code path:

    python scripts/teleop_avp.py --hand linker_l20 --replay

No calibration step: the palm frame is rebuilt from each live frame's own
skeleton, so the headset's position and the wrist frame's orientation do not
matter. The bundle was trained on avp1.pkl (one operator); for another operator,
record their hand and retrain with scripts/run_avp.py --avp <file>.

Output is low-pass filtered in joint space by default (`--smoothing-hz 6`,
disable with `--smoothing-hz 0`) -- the raw NNS lookup hunts between nearby
table entries as the target moves, and pinch-template snapping is a discrete
jump; both are visible as jitter without this. See
anydexrt.inference.JointLowPassFilter.

f_c pinch-template snapping is OFF by default (`--pinch` to enable). As
measured on the current bundles: the ring/pinky contact classifiers have
precision/recall of 0.00 (they never fire), and even where it works the snap
replaces the whole pose in one frame at a 13mm-gap threshold, which looks like
the robot closing on the human fingers well before they visually touch. Needs
the fixes in README errata before this is worth turning back on.

Real hardware (Linker L20 only, via AnyDexRetarget's RS485 driver):

    python scripts/teleop_avp.py --hand linker_l20 --real --confirm-real

`--real` alone is NOT enough to move the physical hand -- it also requires
`--confirm-real` as a second, explicit opt-in, so a copy-pasted command line
cannot accidentally drive real hardware. Without `--confirm-real`, `--real`
runs in dry-run mode: registers are computed and printed, nothing is sent
over the wire. The joint convention was checked to match exactly
(anydexRT's L20 joint limits equal the driver's GEORT-calibrated limits,
checked per-joint before this flag was added) so qpos is sent as-is, no
unit conversion layer. Default serial settings (port /dev/ttyUSB0, baud
460800, slave id 0x29, speed 200, current_limit 200) are the driver's own
conservative defaults for first bring-up; override with --l20-port etc. if
your setup differs.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ANYDEXRETARGET_ROOT = Path("/home/qsq/qqsq12321/AnyDexRetarget")
sys.path.insert(0, str(ANYDEXRETARGET_ROOT))

from anydexrt.data import VP_TO_MEDIAPIPE  # noqa: E402
from anydexrt.inference import load_bundle  # noqa: E402

DEFAULT_REPLAY = "/home/qsq/qqsq12321/AnyDexRetarget/example/data/avp1.pkl"


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


class ReplayAVP:
    """A 21-landmark recording lifted back to the 25-joint skeleton, looped.

    Metacarpal joints (5, 10, 15, 20), which the palm frame does not use, are
    filled with wrist-MCP midpoints.
    """

    def __init__(self, path: str, side: str):
        with open(path, "rb") as fh:
            frames = pickle.load(fh)
        self.frames = [np.asarray(f[f"{side}_fingers"], dtype=np.float64) for f in frames]
        self.i = 0

    def poses(self) -> np.ndarray:
        mp = self.frames[self.i]
        self.i = (self.i + 1) % len(self.frames)
        poses = np.tile(np.eye(4), (25, 1, 1))
        for mp_idx, vp_idx in enumerate(VP_TO_MEDIAPIPE):
            poses[vp_idx, :3, 3] = mp[mp_idx]
        for meta, mcp in ((5, 6), (10, 11), (15, 16), (20, 21)):
            poses[meta, :3, 3] = 0.5 * poses[mcp, :3, 3]
        return poses

    def close(self) -> None:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand", default="linker_l20", choices=["linker_l20", "wuji_hand"])
    ap.add_argument("--avp-ip", default="10.10.10.126")
    ap.add_argument("--side", default="right", choices=["right", "left"],
                    help="left is mirrored onto the right-hand model")
    ap.add_argument("--replay", nargs="?", const=DEFAULT_REPLAY, default=None,
                    help="feed a recording instead of the headset (default: avp1.pkl)")
    ap.add_argument("--bundle", default=None, help="default: results/avp/<hand>/<hand>_bundle.pt")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--smoothing-hz", type=float, default=6.0,
                    help="joint-space low-pass cutoff; 0 disables (raw NNS output, visibly jittery)")
    ap.add_argument("--pinch", action="store_true",
                    help="enable f_c pinch-template snapping (off by default, see module docstring)")
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
    args = ap.parse_args()

    if args.real and args.hand != "linker_l20":
        print("--real currently only has a driver for linker_l20", file=sys.stderr)
        return 1

    bundle = args.bundle or ROOT / "results/avp" / args.hand / f"{args.hand}_bundle.pt"
    rt = load_bundle(bundle, smoothing_hz=args.smoothing_hz or None, use_contacts=args.pinch)
    kin = rt.kin
    source = ReplayAVP(args.replay, args.side) if args.replay else LiveAVP(args.avp_ip, args.side)
    print(f"{args.hand} <- {'replay ' + args.replay if args.replay else 'AVP ' + args.avp_ip} ({args.side} hand)")

    real_driver = None
    if args.real:
        from example.output.real.drivers_linker_l20 import LinkerL20Output

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
    # avp_stream's get_latest() is non-blocking: it returns whatever snapshot is
    # currently cached, not "the next new frame". If the Vision Pro delivers
    # slower than this loop polls, the same snapshot gets read (and retargeted)
    # several times in a row -- the robot holds a pose, then jumps when a new
    # sample finally lands, which looks identical to over-smoothing but has
    # nothing to do with the filter. Comparing consecutive raw arrays tells
    # the two apart: see the printed "new AVP samples" rate below.
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
                        poses[:, :3, 3] *= mirror
                    try:
                        qpos = rt.retarget_avp_poses(poses)
                        frames += 1
                    except ValueError:
                        # Untracked hand: avp_stream sends a zero/identity placeholder,
                        # whose palm is degenerate. Hold the last pose.
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
                    print(f"  {frames} frames, {lost} untracked | new AVP samples: {fresh} "
                          f"({fresh / elapsed:.1f} Hz), repeated reads: {stale} | pinch: {pinch}")
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
