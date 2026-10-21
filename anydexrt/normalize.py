"""Per-finger normalization to [-1, 1].

Appendix B of the paper: "For each human/robot finger, we centralize the
fingertip position, compute the ranges along each axis, and select the axis
with the maximum range. The selected range is then used to normalize the
positions. The normalized positions are distributed in [-1, 1] without
geometric distortion."

The important detail is *one* range is selected (the max over axes) and used
for all three axes. Dividing each axis by its own range would stretch the
cloud non-uniformly, which the paper explicitly rules out. Treating the finger
as its own frame also removes per-finger offset, so the mapper never has to
learn where the finger happens to sit in the hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FingerNormalizer:
    """Affine (scale + offset) normalization fit on one finger's samples."""

    center: np.ndarray  # (3,)
    scale: float  # scalar, applied to every axis

    @classmethod
    def fit(cls, points: np.ndarray, eps: float = 1e-8) -> "FingerNormalizer":
        """Fit from samples of shape (N, 3)."""
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"expected (N, 3) point samples, got {points.shape}")
        center = points.mean(axis=0)
        ranges = points.max(axis=0) - points.min(axis=0)
        scale = float(ranges.max())
        return cls(center=center.astype(np.float32), scale=max(scale, eps))

    def transform(self, points: np.ndarray) -> np.ndarray:
        return ((np.asarray(points, dtype=np.float32) - self.center) / self.scale).astype(
            np.float32
        )

    def inverse(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points, dtype=np.float32) * self.scale + self.center).astype(
            np.float32
        )

    def to_dict(self) -> dict:
        return {"center": self.center.tolist(), "scale": self.scale}

    @classmethod
    def from_dict(cls, d: dict) -> "FingerNormalizer":
        return cls(
            center=np.asarray(d["center"], dtype=np.float32), scale=float(d["scale"])
        )


class HandNormalizer:
    """One FingerNormalizer per finger, in canonical finger order."""

    def __init__(self, fingers: tuple[str, ...], normalizers: list[FingerNormalizer]):
        self.fingers = tuple(fingers)
        self.normalizers = list(normalizers)
        if len(self.fingers) != len(self.normalizers):
            raise ValueError("finger names and normalizers must have equal length")

    @classmethod
    def fit(cls, fingers: tuple[str, ...], points: np.ndarray) -> "HandNormalizer":
        """Fit from a (F, N, 3) array: one cloud per finger."""
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 3 or points.shape[2] != 3:
            raise ValueError(f"expected (F, N, 3), got {points.shape}")
        if points.shape[0] != len(fingers):
            raise ValueError(
                f"expected {len(fingers)} fingers, got {points.shape[0]} clouds"
            )
        return cls(fingers, [FingerNormalizer.fit(points[i]) for i in range(len(fingers))])

    def _apply(self, points: np.ndarray, fn) -> np.ndarray:
        """Apply a per-finger transform, preserving the caller's rank.

        Accepted shapes, all with the finger axis first:
            (F, 3)    a single pose
            (F, N, 3) a cloud per finger
        The finger index is always axis 0, so the per-finger loop stays the same
        regardless of rank; rank is restored on the way out.
        """
        points = np.asarray(points, dtype=np.float32)
        if points.ndim not in (2, 3) or points.shape[-1] != 3:
            raise ValueError(
                f"expected (F, 3) or (F, N, 3) with last dim 3, got {points.shape}"
            )
        if points.shape[0] != len(self.fingers):
            raise ValueError(
                f"expected {len(self.fingers)} fingers on axis 0, got {points.shape[0]}"
            )
        return np.stack(
            [fn(self.normalizers[i], points[i]) for i in range(len(self.fingers))]
        ).astype(np.float32)

    def transform(self, points: np.ndarray) -> np.ndarray:
        """Normalize a (F, N, 3) cloud array or a (F, 3) single pose."""
        return self._apply(points, lambda norm, pts: norm.transform(pts))

    def inverse(self, points: np.ndarray) -> np.ndarray:
        """Undo normalization for a (F, N, 3) cloud array or a (F, 3) pose."""
        return self._apply(points, lambda norm, pts: norm.inverse(pts))

    def to_dict(self) -> dict:
        return {
            "fingers": list(self.fingers),
            "normalizers": [n.to_dict() for n in self.normalizers],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HandNormalizer":
        return cls(
            tuple(d["fingers"]),
            [FingerNormalizer.from_dict(x) for x in d["normalizers"]],
        )
