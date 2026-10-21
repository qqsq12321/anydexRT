"""Local coordinate frame helpers.

The paper's central geometric claim is that motion consistency should be
measured in *local* frames T(x) rather than a global hand-base frame, because
glove calibration errors rotate the global frame and thereby corrupt globally
measured directions.

Two frame sources are needed:

* Human side: T(x_j^{H,i}) is the measured fingertip orientation. The Manus
  glove reports a per-keypoint quaternion, so this is available directly. The
  paper does not spell this out for the human side; using the measured glove
  orientation is the natural reading, since the whole point is to cancel
  glove-induced frame error.

* Robot side: the paper is explicit -- "Since f_m^i(x_j^{H,i}) does not provide
  a local rotation, we assign it the rotation of its nearest neighbor in
  C^{R,i}." So the robot frame comes from the FK orientation of whichever
  sampled configuration is closest.

Both helpers here are pure and testable without a hand model.
"""

from __future__ import annotations

import numpy as np
import torch


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert (..., 4) w-x-y-z quaternions to (..., 3, 3) rotation matrices.

    Manus sidecar metadata declares `quaternion_order: wxyz`.
    """
    q = np.asarray(quat, dtype=np.float32)
    if q.shape[-1] != 4:
        raise ValueError(f"expected last dim 4, got {q.shape}")
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    norms = np.sqrt(w * w + x * x + y * y + z * z)
    norms = np.where(norms < 1e-12, 1.0, norms)
    w, x, y, z = w / norms, x / norms, y / norms, z / norms

    out = np.empty(q.shape[:-1] + (3, 3), dtype=np.float32)
    out[..., 0, 0] = 1 - 2 * (y * y + z * z)
    out[..., 0, 1] = 2 * (x * y - w * z)
    out[..., 0, 2] = 2 * (x * z + w * y)
    out[..., 1, 0] = 2 * (x * y + w * z)
    out[..., 1, 1] = 1 - 2 * (x * x + z * z)
    out[..., 1, 2] = 2 * (y * z - w * x)
    out[..., 2, 0] = 2 * (x * z - w * y)
    out[..., 2, 1] = 2 * (y * z + w * x)
    out[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return out


def nearest_neighbor_rotation(
    query: torch.Tensor, cloud: torch.Tensor, cloud_rot: torch.Tensor
) -> torch.Tensor:
    """Assign each query point the rotation of its nearest neighbor in `cloud`.

    Args:
        query: (B, N, 3) points needing a rotation.
        cloud: (M, 3) reference point cloud for this finger.
        cloud_rot: (M, 3, 3) rotations attached to `cloud`.

    Returns:
        (B, N, 3, 3) rotations.

    This implements the paper's rule for the robot side of T(x).
    """
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError(f"cloud must be (M, 3), got {tuple(cloud.shape)}")
    if cloud_rot.shape != (cloud.shape[0], 3, 3):
        raise ValueError(
            f"cloud_rot must be ({cloud.shape[0]}, 3, 3), got {tuple(cloud_rot.shape)}"
        )
    # cdist over the merged batch dimension keeps memory at (B*N, M).
    B, N, _ = query.shape
    flat = query.reshape(-1, 3)
    idx = torch.cdist(flat, cloud).argmin(dim=1)  # (B*N,)
    return cloud_rot[idx].reshape(B, N, 3, 3)


def rotate_into_local(delta: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """Express a world-frame vector in local coordinates: T(x)^-1 @ delta.

    Args:
        delta: (..., 3) world-frame vectors.
        rot: (..., 3, 3) local frame orientations (columns are the local axes).

    Returns:
        (..., 3) vectors expressed in the local frame.

    `delta` and `rot` must have the same leading shape. They are deliberately
    NOT broadcast against each other: matmul treats a leading 1 as a batch
    dimension, so a (1, N, 3) delta against an (N, 3, 3) frame fails with a
    batch-size mismatch rather than aligning the sample axes. Callers with a
    scalar leading dim squeeze it first.
    """
    if delta.shape[-1] != 3:
        raise ValueError(f"delta must end in 3, got {tuple(delta.shape)}")
    if rot.shape[-2:] != (3, 3):
        raise ValueError(f"rot must end in (3, 3), got {tuple(rot.shape)}")
    if rot.shape[:-2] != delta.shape[:-1]:
        raise ValueError(
            f"frame shape {tuple(rot.shape[:-2])} does not match vector batch "
            f"{tuple(delta.shape[:-1])}; squeeze singleton leading dims instead "
            "of relying on broadcast"
        )
    rot = rot.to(dtype=delta.dtype)
    return (rot.transpose(-1, -2) @ delta.unsqueeze(-1)).squeeze(-1)


def rotation_angle_deg(rot: torch.Tensor) -> torch.Tensor:
    """Rotation angle in degrees, for reporting perturbation magnitudes."""
    trace = rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2]
    cos = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos))


def y_axis_rotation(deg: float, dtype=torch.float32) -> torch.Tensor:
    """Rotation matrix about +y, used to simulate glove frame misalignment."""
    rad = np.deg2rad(deg)
    c, s = float(np.cos(rad)), float(np.sin(rad))
    return torch.tensor(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=dtype
    )


def expand_rotation(rot: torch.Tensor, batch: int, frames: int) -> torch.Tensor:
    """Broadcast a single (3, 3) rotation to one per (batch, frame) sample.

    Handy for tests and for applying one calibration perturbation across an
    entire trajectory.
    """
    if rot.shape != (3, 3):
        raise ValueError(f"expected a single (3, 3) rotation, got {tuple(rot.shape)}")
    return rot.unsqueeze(0).unsqueeze(0).expand(batch, frames, 3, 3).contiguous()
