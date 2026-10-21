"""Robot hand registry: maps each hand to its fingertips and driving joints.

A hand is described by its MJCF path plus, for each finger, the leaf body used
as the fingertip query point and the ordered list of joints that drive it.
The joint grouping matches the convention already used by mimic_retargeter_lab
(config/retargeter_cfg/geort/human_hand_to_*_hand.yaml) so that fingertip
clouds sampled here are comparable with the existing GeoRT checkpoints.

Human keypoint ids depend on the input device, so they are kept per layout:

  manus      25 keypoints (wrist + thumb 1-4 + four fingers x 5, metacarpal first).
             This is also Apple Vision Pro's raw 25-joint skeleton.
  mediapipe  21 keypoints (wrist + 5 fingers x 4). The AVP recordings in this
             workspace are stored in this layout, converted by
             AnyDexRetarget/example/input/visionpro.py (VP_TO_MEDIAPIPE).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")

# Fingertip keypoint ids per input layout, in canonical finger order.
HUMAN_LAYOUTS: dict[str, dict[str, int]] = {
    "manus": {"thumb": 4, "index": 9, "middle": 14, "ring": 19, "pinky": 24},
    "mediapipe": {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20},
}

# Palm skeleton ids per layout: (wrist, index MCP, middle MCP, pinky MCP). These
# define the canonical palm frame -- see `palm_frame` in kinematics.py.
HUMAN_PALM_IDS: dict[str, tuple[int, int, int, int]] = {
    "manus": (0, 6, 11, 21),
    "mediapipe": (0, 5, 9, 17),
}

# Kept for backwards compatibility with the Manus-only API.
HUMAN_FINGERTIP_IDS: dict[str, int] = HUMAN_LAYOUTS["manus"]

# Default MJCF root. Overridable so the package can be relocated or pointed at
# a different asset tree.
DEFAULT_ASSET_ROOT = Path("/home/qsq/qqsq12321/mimic_retargeter_lab/assets/mjcf")
LINKER_L20_MJCF = Path(
    "/home/qsq/qqsq12321/AnyDexRetarget/assets/linker_l20/linker_l20_right_mujoco.xml"
)


@dataclass(frozen=True)
class FingerSpec:
    """One finger: where its fingertip is and which joints move it.

    `tip_offset` is expressed in the tip body's own frame. It is zero for hands
    whose MJCF already has a dedicated fingertip body, and non-zero for hands
    like the Linker L20 whose last body frame sits at the distal joint.
    """

    name: str
    tip_body: str
    joints: tuple[str, ...]
    tip_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class HandSpec:
    """A robot hand: MJCF file plus its fingers in canonical order.

    `couplings` maps a drive name to the MJCF joints it actuates and with what
    multiplier. Most entries are absent (a drive is one joint, multiplier 1).
    Shadow is the exception: its tendon actuators F0 each drive two joints at
    0.5, so a drive is not a joint there and the distinction has to be modelled
    explicitly. A coupling's `mult` values are also what makes a drive's own
    joint-limit range well defined.

    `palm_bodies` names (wrist, index MCP, middle MCP, pinky MCP) bodies. They
    define the hand's canonical palm frame, which is what lets a human input
    device and this hand agree on axes without a hand-tuned rotation.
    """

    name: str
    mjcf: Path
    fingers: tuple[FingerSpec, ...]
    couplings: dict[str, dict[str, float]] | None = None
    palm_bodies: tuple[str, str, str, str] | None = None

    @property
    def n_fingers(self) -> int:
        return len(self.fingers)

    @property
    def finger_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fingers)

    @property
    def n_joints(self) -> int:
        """Number of drives (DoF of every qpos vector), not raw MJCF joints."""
        return sum(len(f.joints) for f in self.fingers)

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Flattened drive order; this is the DoF order of every qpos vector."""
        out: list[str] = []
        for f in self.fingers:
            out.extend(f.joints)
        return tuple(out)

    def drive_targets(self, name: str) -> dict[str, float]:
        """MJCF joints a drive actuates, with multipliers. Defaults to itself."""
        if self.couplings and name in self.couplings:
            return self.couplings[name]
        return {name: 1.0}

    @property
    def tip_bodies(self) -> tuple[str, ...]:
        return tuple(f.tip_body for f in self.fingers)

    def joint_slice(self, finger: str) -> slice:
        """Return the slice of the flat qpos vector owned by `finger`."""
        start = 0
        for f in self.fingers:
            if f.name == finger:
                return slice(start, start + len(f.joints))
            start += len(f.joints)
        raise KeyError(f"{self.name} has no finger {finger!r}")

    @property
    def tip_offsets(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(f.tip_offset for f in self.fingers)

    def human_ids(self, layout: str = "manus") -> list[int]:
        """Human fingertip keypoint ids for this hand's fingers.

        Four-fingered hands simply have no 'pinky', so the human pinky is
        dropped. That choice is not specified by the paper -- see README.
        """
        if layout not in HUMAN_LAYOUTS:
            raise KeyError(f"unknown layout {layout!r}; available: {sorted(HUMAN_LAYOUTS)}")
        ids = HUMAN_LAYOUTS[layout]
        return [ids[f.name] for f in self.fingers]

    def resolved_mjcf(self, asset_root: Path | None = None) -> Path:
        """Absolute MJCF path.

        `mjcf` is stored relative to the asset tree (e.g. "leap_hand/right_hand.xml")
        so the package can be relocated; an absolute `mjcf` is honoured as-is.
        """
        if self.mjcf.is_absolute():
            return self.mjcf
        root = asset_root if asset_root is not None else DEFAULT_ASSET_ROOT
        return root / self.mjcf


def _wuji() -> HandSpec:
    fingers = tuple(
        FingerSpec(
            name=name,
            tip_body=f"right_finger{i}_tip_link",
            joints=tuple(f"right_finger{i}_joint{j}" for j in range(1, 5)),
        )
        for i, name in enumerate(FINGER_ORDER, start=1)
    )
    palm = (
        "right_palm_link",
        "right_finger2_link2",
        "right_finger3_link2",
        "right_finger5_link2",
    )
    return HandSpec("wuji_hand", Path("wuji_hand/right_hand.xml"), fingers, palm_bodies=palm)


def _linker_l20() -> HandSpec:
    # 21 MJCF joints, 16 actuators. The other five joints are slaved by joint
    # equality constraints in the MJCF (polycoef: slave = mult * master):
    #   thumb_ip = 1.03 * thumb_mcp,   <finger>_dip = 0.89 * <finger>_pip.
    # Setting qpos directly and calling mj_forward does NOT enforce equality
    # constraints, so they are modelled here as couplings exactly like Shadow's
    # tendons. tests/test_hands_avp.py checks these multipliers against the
    # MJCF so they cannot drift.
    couplings = {"thumb_mcp": {"thumb_mcp": 1.0, "thumb_ip": 1.03}}
    for f in ("index", "middle", "ring", "pinky"):
        couplings[f"{f}_pip"] = {f"{f}_pip": 1.0, f"{f}_dip": 0.89}

    # The distal body frames sit at the DIP/IP joints; these offsets (taken from
    # AnyDexRetarget's robot_configs.py 'linker_l20') move the query point to the
    # visible fingertip surface.
    thumb_tip = (-0.0110, 0.0, 0.0277)
    finger_tip = (-0.0135, 0.0, 0.0188)
    fingers = (
        FingerSpec(
            "thumb",
            "thumb_distal",
            ("thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch", "thumb_mcp"),
            thumb_tip,
        ),
    ) + tuple(
        FingerSpec(f, f"{f}_distal", (f"{f}_mcp_roll", f"{f}_mcp_pitch", f"{f}_pip"), finger_tip)
        for f in ("index", "middle", "ring", "pinky")
    )
    palm = ("hand_base_link", "index_proximal", "middle_proximal", "pinky_proximal")
    return HandSpec("linker_l20", LINKER_L20_MJCF, fingers, couplings, palm_bodies=palm)


def _leap() -> HandSpec:
    # Leap has no pinky; finger order and joint ids from the existing config.
    fingers = (
        FingerSpec("thumb", "thumb_tip_head", ("12", "13", "14", "15")),
        FingerSpec("index", "index_tip_head", ("1", "0", "2", "3")),
        FingerSpec("middle", "middle_tip_head", ("5", "4", "6", "7")),
        FingerSpec("ring", "ring_tip_head", ("9", "8", "10", "11")),
    )
    return HandSpec("leap_hand", Path("leap_hand/right_hand.xml"), fingers)


def _allegro() -> HandSpec:
    # Allegro has no pinky.
    fingers = (
        FingerSpec("thumb", "th_tip", ("thj0", "thj1", "thj2", "thj3")),
        FingerSpec("index", "ff_tip", ("ffj0", "ffj1", "ffj2", "ffj3")),
        FingerSpec("middle", "mf_tip", ("mfj0", "mfj1", "mfj2", "mfj3")),
        FingerSpec("ring", "rf_tip", ("rfj0", "rfj1", "rfj2", "rfj3")),
    )
    return HandSpec("wonik_allegro_hand", Path("wonik_allegro_hand/right_hand.xml"), fingers)


def _shadow() -> HandSpec:
    # The four non-thumb fingers are tendon-driven: actuator F0 drives FFJ1 and
    # FFJ2 at 0.5 each, so those drives are couplings rather than bare joints.
    # Multipliers match mimic_retargeter_lab/hand_models/shadow_hand.py.
    couplings = {
        f"rh_{p}J0": {f"rh_{p}J1": 0.5, f"rh_{p}J2": 0.5}
        for p in ("FF", "MF", "RF", "LF")
    }
    fingers = (
        FingerSpec("thumb", "rh_thtip", ("rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1")),
        FingerSpec("index", "rh_fftip", ("rh_FFJ4", "rh_FFJ3", "rh_FFJ0")),
        FingerSpec("middle", "rh_mftip", ("rh_MFJ4", "rh_MFJ3", "rh_MFJ0")),
        FingerSpec("ring", "rh_rftip", ("rh_RFJ4", "rh_RFJ3", "rh_RFJ0")),
        FingerSpec("pinky", "rh_lftip", ("rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ0")),
    )
    return HandSpec("shadow_hand", Path("shadow_hand/right_hand.xml"), fingers, couplings)


_BUILDERS = {
    "wuji_hand": _wuji,
    "linker_l20": _linker_l20,
    "leap_hand": _leap,
    "wonik_allegro_hand": _allegro,
    "shadow_hand": _shadow,
}


def get_hand(name: str) -> HandSpec:
    if name not in _BUILDERS:
        raise KeyError(f"unknown hand {name!r}; available: {sorted(_BUILDERS)}")
    return _BUILDERS[name]()


def available_hands() -> list[str]:
    return sorted(_BUILDERS)
