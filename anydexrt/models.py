"""Networks: fingertip mapper f_m and contact classifier f_c.

Appendix B: "For a hand with F fingers, we employ F sub-nets for f_m and (F-1)
sub-nets for f_c. All sub-nets of f_m have the same size of (3, 128, 128, 3),
whose input and output are human fingertip positions and retargeted fingertip
positions, respectively. All sub-nets of f_c have the same size of
(6, 128, 128, 1) followed by the Sigmoid function. The input of f_c^i (i > 0) is
a fingertip position concatenation of the thumb and the i-th finger, and the
output is a one-dimensional score distributed in [0, 1]."

Two things the paper leaves unspecified and we make explicit:

* Activation. Not named. GeoRT (the direct ancestor) uses LeakyReLU, so that is
  the default here; ReLU is available.
* BatchNorm. GeoRT has BatchNorm1d after each hidden layer; the paper's size
  tuples list only Linear layers, so BatchNorm is omitted. This also avoids the
  train/eval discrepancy that matters at batch 2048, and it keeps the mapper a
  pure function at inference.

The output activation is also left open. The mapper's output lives in the same
normalized [-1, 1] space as the robot cloud, so a tanh is defensible, but the
paper's literal reading is a linear output. Default is linear; the online
solver's nearest-neighbour lookup keeps predictions in range anyway.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}


def _mlp(dims: list[int], activation: str, final_activation: nn.Module | None) -> nn.Sequential:
    """Build Linear -> act stacks from a list of layer widths."""
    if len(dims) < 2:
        raise ValueError("need at least input and output widths")
    act = ACTIVATIONS[activation]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        is_last = i == len(dims) - 2
        if not is_last:
            layers.append(act())
        elif final_activation is not None:
            layers.append(final_activation)
    return nn.Sequential(*layers)


class FingerMapper(nn.Module):
    """f_m: human fingertips -> robot fingertips, one sub-net per finger.

    Each sub-net sees only its own finger's 3D position, matching the paper's
    (3, 128, 128, 3) shape and its per-finger factorization. Cross-finger
    consistency comes from the shared anchors and from `f_s`, not from the
    network seeing other fingers.
    """

    def __init__(
        self,
        n_fingers: int,
        width: int = 128,
        depth: int = 2,
        activation: str = "leaky_relu",
        output_tanh: bool = False,
    ):
        super().__init__()
        dims = [3] + [width] * depth + [3]
        final = nn.Tanh() if output_tanh else None
        self.nets = nn.ModuleList(
            [_mlp(dims, activation, final) for _ in range(n_fingers)]
        )
        self.n_fingers = n_fingers
        self.width = width
        self.depth = depth
        self.activation = activation
        self.output_tanh = output_tanh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, F, 3) normalized human fingertips. Returns: (B, F, 3)."""
        if x.dim() != 3 or x.shape[1] != self.n_fingers or x.shape[2] != 3:
            raise ValueError(
                f"expected (B, {self.n_fingers}, 3) input, got {tuple(x.shape)}"
            )
        outs = [self.nets[i](x[:, i]) for i in range(self.n_fingers)]
        return torch.stack(outs, dim=1)

    def forward_finger(self, x: torch.Tensor, finger: int) -> torch.Tensor:
        """Map a single finger. Args: x (B, 3) -> (B, 3)."""
        return self.nets[finger](x)


class CrossFingerMapper(nn.Module):
    """f_m variant: the thumb sub-net also sees the other fingers' same-frame
    positions, so it can represent cross-finger (pinch) relationships that
    FingerMapper's strict per-finger factorization architecturally cannot.

    Measured on `FingerMapper`: mapped thumb-index gap barely moves (95.4mm ->
    75.7mm) as the real human gap goes from 80mm to 0mm, because the thumb
    sub-net never sees the index finger's position at all. This class widens
    only the thumb sub-net's input, from 3 to 3 + 3*(F-1): its own position
    plus every other finger's position, same frame, concatenated in finger
    index order (excluding itself). Every other finger keeps the paper's plain
    (3, width, width, 3) sub-net -- this is a minimal, targeted change aimed at
    the one cross-finger relationship this model needs (thumb-vs-finger pinch),
    not a general all-to-all mixing network.
    """

    def __init__(
        self,
        n_fingers: int,
        thumb_idx: int = 0,
        width: int = 128,
        depth: int = 2,
        activation: str = "leaky_relu",
        output_tanh: bool = False,
    ):
        super().__init__()
        if not (0 <= thumb_idx < n_fingers):
            raise ValueError(f"thumb_idx must be in [0, {n_fingers}), got {thumb_idx}")
        final = nn.Tanh() if output_tanh else None
        plain_dims = [3] + [width] * depth + [3]
        thumb_dims = [3 + 3 * (n_fingers - 1)] + [width] * depth + [3]
        nets = []
        for i in range(n_fingers):
            dims = thumb_dims if i == thumb_idx else plain_dims
            nets.append(_mlp(dims, activation, final))
        self.nets = nn.ModuleList(nets)
        self.n_fingers = n_fingers
        self.thumb_idx = thumb_idx
        self.width = width
        self.depth = depth
        self.activation = activation
        self.output_tanh = output_tanh
        # Zero placeholder for the other-fingers context when a caller maps the
        # thumb in isolation (forward_finger with no context, e.g. call sites
        # written against FingerMapper's single-finger signature). Zeros are a
        # neutral, shape-compatible stand-in, not a claim about real geometry --
        # the thumb's output degrades to FingerMapper-like blindness in that case.
        self.register_buffer(
            "context_placeholder", torch.zeros(3 * (n_fingers - 1)), persistent=False
        )

    def context_from_paired(self, paired: torch.Tensor) -> torch.Tensor:
        """paired: (B, F, 3) same-frame fingertips -> (B, 3*(F-1)) thumb context."""
        return torch.cat(
            [paired[:, i] for i in range(self.n_fingers) if i != self.thumb_idx], dim=-1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, F, 3) normalized human fingertips, SAME-FRAME (paired)
        across the finger dimension -- required for the thumb sub-net's
        context to be meaningful. Returns: (B, F, 3)."""
        if x.dim() != 3 or x.shape[1] != self.n_fingers or x.shape[2] != 3:
            raise ValueError(
                f"expected (B, {self.n_fingers}, 3) input, got {tuple(x.shape)}"
            )
        context = self.context_from_paired(x)
        outs = []
        for i in range(self.n_fingers):
            if i == self.thumb_idx:
                outs.append(self.nets[i](torch.cat([x[:, i], context], dim=-1)))
            else:
                outs.append(self.nets[i](x[:, i]))
        return torch.stack(outs, dim=1)

    def forward_finger(
        self, x: torch.Tensor, finger: int, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Map a single finger. Args: x (B, 3) -> (B, 3).

        `context`: required for the thumb (finger == thumb_idx) to use the
        other fingers' positions -- (B, 3*(F-1)), same frame as x. Falls back
        to a zero placeholder broadcast to batch size if omitted.
        """
        if finger == self.thumb_idx:
            if context is None:
                context = self.context_placeholder.unsqueeze(0).expand(x.shape[0], -1)
            return self.nets[finger](torch.cat([x, context], dim=-1))
        return self.nets[finger](x)


class ContactClassifier(nn.Module):
    """f_c: thumb-vs-finger contact score in [0, 1], one sub-net per non-thumb finger.

    Index 0 is the thumb by the paper's convention ("Let 0 be the thumb index
    and i > 0 be the index of other fingers"). Sub-net i (i > 0) consumes the
    concatenated thumb and i-th fingertip positions, i.e. 6 numbers.
    """

    def __init__(
        self,
        n_fingers: int,
        width: int = 128,
        depth: int = 2,
        activation: str = "leaky_relu",
    ):
        super().__init__()
        if n_fingers < 2:
            raise ValueError("contact classification needs at least two fingers")
        dims = [6] + [width] * depth + [1]
        self.nets = nn.ModuleList(
            [_mlp(dims, activation, nn.Sigmoid()) for _ in range(n_fingers - 1)]
        )
        self.n_fingers = n_fingers
        self.width = width
        self.depth = depth
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, F, 3) fingertips. Returns: (B, F-1) scores for fingers 1..F-1."""
        if x.dim() != 3 or x.shape[2] != 3:
            raise ValueError(f"expected (B, F, 3) input, got {tuple(x.shape)}")
        thumb = x[:, 0]
        outs = [
            self.nets[i](torch.cat([thumb, x[:, i + 1]], dim=-1))
            for i in range(self.n_fingers - 1)
        ]
        return torch.cat(outs, dim=-1)

    def contact_pattern(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Boolean contact pattern, shape (B, F-1)."""
        return self.forward(x) >= threshold
