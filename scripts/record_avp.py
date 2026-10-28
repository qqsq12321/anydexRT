"""Record an Apple Vision Pro or PICO 4 hand-tracking session for training AnyDexRT.

Fixes the duplicate-frame bug found in avp1.pkl: avp_stream's get_latest() is
non-blocking and returns whatever snapshot is currently cached, so polling
faster than the device delivers new data saves the same frame repeatedly.
Measured directly in avp1.pkl: 59.7% of its 2751 saved "frames" are
byte-identical to the one before them -- the true distinct-sample rate is
~24 Hz despite the file looking like a denser recording. This script only
appends a frame when it differs from the last one saved, and stamps real
wall-clock time (avp1.pkl's 't' field was always 0, unusable for rate checks).
PICO 4's `get_fingers_data()` is non-blocking the same way, so the same fix
applies to it.

Output is pickled in the same [{'t', 'left_fingers', 'right_fingers'}, ...]
format as avp1.pkl (MediaPipe 21-landmark positions, both hands every entry,
wrist-relative), regardless of which device recorded it -- PICO 4 already
outputs this exact format natively, AVP's native (25,4,4) skeleton is reduced
to it here (see `to_mediapipe`). Either way it drops straight into the
existing pipeline:

    python scripts/record_avp.py --ip 10.10.10.126 --out data/avp_session1.pkl
    python scripts/record_avp.py --input pico4 --out data/pico4_session1.pkl
    python scripts/run_avp.py --hands wuji_hand linker_l20 --avp data/avp_session1.pkl
    python scripts/teleop_avp.py --hand linker_l20 --replay data/avp_session1.pkl
    python scripts/teleop_avp.py --hand linker_l20 --input pico4 --replay data/pico4_session1.pkl

Ctrl+C stops and saves whatever was captured; --seconds stops automatically.

PICO 4 (`--input pico4`) relay mode (default) needs its relay daemon running
in another terminal first: `python scripts/pico4_daemon.py`. `--pico4-mode
direct` is self-contained instead (no extra process, but the headset
re-pairs on every restart) -- see scripts/teleop_avp.py's docstring for the
same tradeoff.

Recording protocol -- the anchor search (lateral/bending coverage) and the
contact classifier both need this range actually present in the capture, or
training falls back to approximations (see README "Where this reproduction
had to make a choice", sections 2 and 3):
  1. Open hand, fingers spread -- hold ~1s.
  2. Closed fist -- hold ~1s.
  3. Curl each finger alone a few times: index, middle, ring, pinky.
  4. Spread fingers apart, bring them together (a few times).
  5. Thumb-to-fingertip pinch for EACH finger (index, middle, ring, pinky),
     hold each pinch ~1s, repeat 2-3x per finger.
  6. A minute or so of natural, varied hand motion.
Longer and more varied beats shorter: training later holds out two blocks of
whatever you record, so more coverage means a more informative held-out test.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anydexrt.data import VP_TO_MEDIAPIPE  # noqa: E402

PROTOCOL = """Recording protocol (see this script's docstring for the reasoning):
  1. Open hand, fingers spread -- hold ~1s.
  2. Closed fist -- hold ~1s.
  3. Curl each finger alone a few times: index, middle, ring, pinky.
  4. Spread fingers apart, bring them together (a few times).
  5. Thumb-to-fingertip pinch for EACH finger, hold each ~1s, repeat 2-3x.
  6. A minute or so of natural, varied hand motion.
"""


def to_mediapipe(raw_25x4x4: np.ndarray) -> np.ndarray:
    """(25, 4, 4) native AVP skeleton -> (21, 3) MediaPipe-layout positions."""
    return raw_25x4x4[list(VP_TO_MEDIAPIPE), :3, 3].astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="avp", choices=["avp", "pico4"])
    ap.add_argument("--ip", default="10.10.10.126", help="AVP only")
    ap.add_argument("--out", default=None, help="default: data/<input>_<timestamp>.pkl")
    ap.add_argument("--seconds", type=float, default=None, help="stop automatically; default: run until Ctrl+C")
    ap.add_argument("--poll-interval-ms", type=float, default=2.0,
                    help="breather between polls; does not cap the saved rate")
    ap.add_argument("--connect-timeout", type=float, default=20.0)
    pico4 = ap.add_argument_group("pico4", "only used with --input pico4")
    pico4.add_argument("--pico4-mode", default="relay", choices=["relay", "direct"],
                       help="relay daemon (default; run scripts/pico4_daemon.py first) or "
                            "direct TCP server (self-contained, no extra process)")
    pico4.add_argument("--pico4-relay-host", default="127.0.0.1")
    pico4.add_argument("--pico4-relay-port", type=int, default=63902)
    pico4.add_argument("--pico4-port", type=int, default=63901)
    pico4.add_argument("--pico4-broadcast-port", type=int, default=29888)
    args = ap.parse_args()

    out = Path(args.out) if args.out else ROOT / "data" / f"{args.input}_{time.strftime('%Y%m%d_%H%M%S')}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)

    # `get_frame()` returns {'right_fingers', 'left_fingers'} as (21, 3)
    # MediaPipe-order, wrist-relative float32 arrays, or None if nothing new /
    # untracked -- this is the one thing that differs between devices; the
    # recording loop below is identical either way.
    if args.input == "avp":
        from avp_stream import VisionProStreamer

        print(f"Connecting to Vision Pro at {args.ip} ...")
        streamer = VisionProStreamer(ip=args.ip, record=False)

        def get_frame():
            data = streamer.get_latest()
            if data is None:
                return None
            try:
                right = np.asarray(data["right_fingers"], dtype=np.float64)
                left = np.asarray(data["left_fingers"], dtype=np.float64)
            except (KeyError, TypeError):
                return None
            return {"right_fingers": to_mediapipe(right), "left_fingers": to_mediapipe(left)}

        def stop_source():
            stop = getattr(streamer, "_stop_pose_streaming", None)
            if callable(stop):
                stop()
    else:
        from anydexrt.pico import Pico4

        print(f"Starting PICO 4 ({args.pico4_mode} mode) ...")
        dev = Pico4(mode=args.pico4_mode, relay_host=args.pico4_relay_host,
                    relay_port=args.pico4_relay_port, port=args.pico4_port,
                    broadcast_port=args.pico4_broadcast_port)

        def get_frame():
            data = dev.get_fingers_data()
            right, left = data["right_fingers"], data["left_fingers"]
            if not (np.any(right) or np.any(left)):
                return None  # all-zero = no frame received yet
            return {"right_fingers": right.astype(np.float32), "left_fingers": left.astype(np.float32)}

        def stop_source():
            dev.stop()

    t_connect = time.perf_counter()
    while get_frame() is None:
        if time.perf_counter() - t_connect > args.connect_timeout:
            if args.input == "avp":
                print(f"no data after {args.connect_timeout:.0f}s -- check the IP and that "
                      "Tracking Streamer is running and streaming.", file=sys.stderr)
            else:
                print(f"no data after {args.connect_timeout:.0f}s -- check the headset is "
                      f"connected ({args.pico4_mode} mode; relay needs "
                      "scripts/pico4_daemon.py running in another terminal).", file=sys.stderr)
            return 1
        time.sleep(0.1)
    print("connected.\n")
    print(PROTOCOL)
    print("Recording -- Ctrl+C to stop and save.\n")

    frames: list[dict] = []
    prev_right, prev_left = None, None
    t_start = time.perf_counter()
    t_report, saved_since_report = t_start, 0

    try:
        while args.seconds is None or time.perf_counter() - t_start < args.seconds:
            frame = get_frame()
            if frame is None:
                time.sleep(args.poll_interval_ms / 1000.0)
                continue
            right, left = frame["right_fingers"], frame["left_fingers"]

            changed = (prev_right is None or not np.array_equal(right, prev_right)) or \
                      (prev_left is None or not np.array_equal(left, prev_left))
            if changed:
                frames.append({
                    "t": time.perf_counter() - t_start,
                    "right_fingers": right,
                    "left_fingers": left,
                })
                prev_right, prev_left = right, left
                saved_since_report += 1

            if time.perf_counter() - t_report > 2.0:
                window = time.perf_counter() - t_report
                print(f"  {time.perf_counter() - t_start:5.1f}s | {len(frames):5d} frames saved "
                      f"| {saved_since_report / window:5.1f} Hz")
                t_report, saved_since_report = time.perf_counter(), 0
            time.sleep(args.poll_interval_ms / 1000.0)
    except KeyboardInterrupt:
        print("\nstopped by user.")
    finally:
        stop_source()

    if not frames:
        print("nothing captured.", file=sys.stderr)
        return 1

    with open(out, "wb") as fh:
        pickle.dump(frames, fh)

    duration = frames[-1]["t"] - frames[0]["t"]
    right_stack = np.stack([f["right_fingers"] for f in frames])
    dup = float(np.all(right_stack[1:] == right_stack[:-1], axis=(1, 2)).mean()) if len(frames) > 1 else 0.0
    print(f"\nwrote {out}")
    print(f"  {len(frames)} frames over {duration:.1f}s  ({len(frames) / max(duration, 1e-9):.1f} Hz average)")
    print(f"  duplicate-of-previous fraction: {dup * 100:.1f}% (should be ~0, confirms the fix held)")
    print(f"\ntrain with:  python scripts/run_avp.py --hands wuji_hand linker_l20 --avp {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
