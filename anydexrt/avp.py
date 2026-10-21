"""Apple Vision Pro input adapter for the AnyDexRT reproduction.

The ``avp_stream`` package exposes a 25-joint hand skeleton as 4x4 poses.
AnyDexRT only needs the five fingertip positions, so this module deliberately
keeps the transport layer separate from the retargeter and can be unit-tested
without an Apple device or the optional dependency.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from .hands import HUMAN_LAYOUTS, HandSpec, get_hand

# Apple Vision Pro / ARKit hand skeleton order: wrist, thumb 1..4,
# index 1..5, middle 1..5, ring 1..5, little 1..5.  The fingertip ids are
# therefore identical to the Manus 25-joint layout used by this reproduction.
AVP_TIP_IDS = (4, 9, 14, 19, 24)


def _as_pose_array(hand: Any) -> np.ndarray:
    """Convert common avp_stream hand representations to (25,4,4)."""
    if isinstance(hand, dict):
        for key in ("fingers", "joints", "landmarks", "hand"):
            if key in hand:
                hand = hand[key]
                break
    for attr in ("fingers", "joints", "landmarks", "poses"):
        if hasattr(hand, attr):
            hand = getattr(hand, attr)
            break
    arr = np.asarray(hand)
    if arr.ndim != 3 or arr.shape[0] < 25 or arr.shape[1:] != (4, 4):
        raise ValueError(f"expected at least 25 hand poses of shape (4,4), got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def extract_fingertips(hand: Any, hand_spec: HandSpec | str = "wuji_hand") -> np.ndarray:
    """Extract canonical ``(F,3)`` fingertip positions from an AVP hand.

    ``hand`` may be the raw ndarray returned by ``avp_stream`` or an object /
    dict containing it under ``fingers``, ``joints`` or ``landmarks``.
    """
    spec = get_hand(hand_spec) if isinstance(hand_spec, str) else hand_spec
    poses = _as_pose_array(hand)
    ids = spec.human_ids("manus")
    return poses[ids, :3, 3].copy()


def extract_fingertip_orientations(hand: Any, hand_spec: HandSpec | str = "wuji_hand") -> np.ndarray:
    """Return fingertip rotation matrices in canonical finger order (F,3,3)."""
    spec = get_hand(hand_spec) if isinstance(hand_spec, str) else hand_spec
    poses = _as_pose_array(hand)
    return poses[spec.human_ids("manus"), :3, :3].copy()


def _get_side(frame: Any, side: str) -> Any:
    """Read a left/right hand from dict, namespace, or avp_stream frame."""
    if isinstance(frame, dict):
        for key in (f"{side}_fingers", f"{side}_hand", side):
            if key in frame:
                return frame[key]
    for key in (f"{side}_fingers", f"{side}_hand", side):
        if hasattr(frame, key):
            return getattr(frame, key)
    return None


class VisionProInput:
    """Small polling adapter around the optional ``avp_stream`` package.

    It intentionally does not hide training or normalization.  Callers should
    fit an AnyDexRT ``HandNormalizer`` from their calibration/training capture,
    then pass each returned ``(F,3)`` pose to ``AnyDexRT.retarget``.
    """

    def __init__(self, ip: str, *, record: bool = False, streamer: Any = None):
        if streamer is None:
            try:
                from avp_stream import VisionProStreamer
            except ImportError as exc:
                raise ImportError(
                    "VisionProInput requires optional package avp_stream; "
                    "install it with `pip install avp-stream`."
                ) from exc
            streamer = VisionProStreamer(ip=ip, record=record)
        self.streamer = streamer
        self.latest: Any = None

    def poll(self) -> bool:
        """Fetch and store the latest frame; return whether a frame exists."""
        getter = getattr(self.streamer, "get_latest", None)
        frame = getter() if getter is not None else getattr(self.streamer, "latest", None)
        if frame is None:
            return False
        self.latest = frame
        return True

    def fingertips(self, side: str = "right", hand: HandSpec | str = "wuji_hand") -> np.ndarray | None:
        if self.latest is None:
            return None
        raw = _get_side(self.latest, side)
        if raw is None:
            return None
        try:
            return extract_fingertips(raw, hand)
        except ValueError:
            return None

    def close(self) -> None:
        stop = getattr(self.streamer, "_stop_pose_streaming", None)
        if callable(stop):
            stop()
