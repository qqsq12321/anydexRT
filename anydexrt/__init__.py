"""AnyDexRT reproduction: calibration-free dexterous hand retargeting.

Reimplements "AnyDexRT: Calibration-Free Dexterous Hand Retargeting with
Few-Shot Human Guidance" (arXiv:2607.08341).

Pipeline:  f = f_s o f_m
    f_m : human fingertips -> robot fingertips   (FingerMapper, self-supervised)
    f_s : robot fingertips -> robot joints       (NearestNeighborIK)

Deliberately depends only on mujoco + torch + numpy + scipy so it stays
independent of the heavier JAX/hydra stack in mimic_retargeter_lab.
"""

from .hands import (
    FINGER_ORDER,
    HUMAN_FINGERTIP_IDS,
    FingerSpec,
    HandSpec,
    available_hands,
    get_hand,
)
from .kinematics import HandKinematics, NearestNeighborIK
from .normalize import FingerNormalizer, HandNormalizer
from .avp import VisionProInput, extract_fingertips

__all__ = [
    "FINGER_ORDER",
    "HUMAN_FINGERTIP_IDS",
    "FingerSpec",
    "HandSpec",
    "available_hands",
    "get_hand",
    "HandKinematics",
    "NearestNeighborIK",
    "FingerNormalizer",
    "HandNormalizer",
    "VisionProInput",
    "extract_fingertips",
]

__version__ = "0.1.0"
