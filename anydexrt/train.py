"""Training loop for f_m (fingertip mapper) and f_c (contact classifier).

Hyperparameters are the paper's (Appendix B), verbatim:
    20 epochs, lr 1e-4, batch size 2048, anchor batch size 32.
    "The finger mapper f_m and the contact classifier f_c can be trained in
    parallel since they are independent from each other."
    Loss: L_mapping = L_P-Chamfer + L_dist + L_motion + L_align, unweighted.

Contact labels are the one piece the paper does not describe. It says contact
"signals are easy to collect and annotate, and can directly reflect key
manipulation intentions such as pinching", but the shipped Manus captures carry
no contact channel. We therefore derive labels geometrically: the thumb and
finger i are labelled in contact when their fingertips are closer than a
threshold, expressed as a fraction of the palm span so it is subject-independent.
That threshold is a free parameter of this reproduction and is recorded in the
run config.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .anchors import AnchorSet
from .data import HandSamples, RobotClouds
from .frames import quat_wxyz_to_matrix
from .losses import LossWeights, contact_loss, mapping_loss
from .models import ContactClassifier, FingerMapper


@dataclass
class TrainConfig:
    """Everything that defines a run. Serialized next to the checkpoints."""

    hand: str = "leap_hand"
    epochs: int = 20
    lr: float = 1e-4
    # The contact classifier needs different settings from the mapper, and the
    # paper applies one setting to both. At lr 1e-4 the classifier converges to
    # the all-negative solution no matter how long it trains: scores park
    # permanently below the 0.5 inference threshold, `f_c` never fires, and
    # pinch refinement is dead code -- while the loss curve looks healthy.
    # Measured on Leap Hand against the geometric labels (true contact rate
    # 7-9% per finger):
    #
    #   lr=1e-4,  20 epochs -> predicted rate 0.000, recall 0.00
    #   lr=1e-4, 200 epochs -> predicted rate 0.000, recall 0.00
    #   lr=1e-3,  40 epochs -> recall 0.0-0.40
    #   lr=1e-3, 150 epochs -> predicted rate 0.07/0.09/0.08, recall 1.00/1.00/0.99
    #
    # Note the input representation is NOT the problem: per-finger normalized
    # coordinates (what the mapper consumes) separate contact as well as a
    # shared-frame representation once trained. Raw metres does worse. See README.
    contact_lr: float = 1e-3
    contact_epochs: int = 150
    batch_size: int = 2048
    anchor_batch_size: int = 32
    width: int = 128
    depth: int = 2
    activation: str = "leaky_relu"
    output_tanh: bool = False
    seed: int = 0
    jitter_base: float = 0.001
    jitter_rand: float = 0.01
    # Fraction of the hand span below which the thumb is treated as touching a
    # finger. 0.2 is chosen from the data: on both shipped captures every finger
    # lands in a 6-10% contact rate there, and the rate stops growing steeply
    # past it (i.e. the real pinches are already inside). See README.
    contact_threshold: float = 0.2
    contact_pos_weight: float | None = None
    weights: dict = field(
        default_factory=lambda: asdict(LossWeights())
    )
    use_anchors: bool = True
    # Eq. (3) is stated for local frames T(x), and the paper pins down the robot
    # side ("assign it the rotation of its nearest neighbor in C^{R,i}") but
    # says nothing about the human side. Both tried readings were measured on
    # Leap Hand at 60 epochs, against a global-frame direction-agreement score:
    #
    #   both sides global                      -> 99.1
    #   robot side local, human side global    -> -28.1
    #   both sides local (glove quaternion)    -> -28.1
    #
    # The robot side's frame is a fingertip body orientation, i.e. local to the
    # palm; the glove quaternion is absolute. Pairing them inserts a
    # ~30-degree-mean, widely-spread relative rotation and the inner product in
    # Eq. (3) goes negative -- worse than omitting the term. Since the paper
    # does not specify the human frame, no local variant can be reproduced
    # faithfully, so the global reading is the default and the local variant
    # stays available for the calibration experiments. See README.
    use_local_frames: bool = False
    strategy: str = "synthetic"

    def loss_weights(self) -> LossWeights:
        return LossWeights(**self.weights)


@dataclass
class TrainResult:
    mapper: FingerMapper
    classifier: ContactClassifier | None
    history: list[dict]
    config: TrainConfig
    contact_history: list[dict] = field(default_factory=list)


def hand_span(fingertips: np.ndarray) -> float:
    """A scale reference for the hand, in metres.

    Deliberately NOT the thumb-to-pinky distance. That distance is the very
    quantity being thresholded for the pinky, so using it as the denominator
    makes the pinky's normalized distance identically 1.0 and permanently
    labels the pinky as out of contact -- a silent, plausible-looking failure.
    The mean pairwise distance among all fingertips has no such self-reference
    and still tracks overall hand size.
    """
    fingertips = np.asarray(fingertips, dtype=np.float32)
    T, F, _ = fingertips.shape
    total = 0.0
    n_pairs = 0
    for i in range(F):
        for j in range(i + 1, F):
            total += float(np.linalg.norm(fingertips[:, i] - fingertips[:, j], axis=-1).mean())
            n_pairs += 1
    span = total / max(n_pairs, 1)
    if span < 1e-6:
        raise ValueError("degenerate hand span; check the fingertip input")
    return span


def derive_contact_labels(
    fingertips: np.ndarray, threshold: float
) -> np.ndarray:
    """Geometric thumb-contact labels, shape (T, F-1).

    A finger is in contact with the thumb when their fingertips are closer than
    `threshold` times the hand span. Normalizing keeps one threshold meaningful
    across operators with different hand sizes.

    Args:
        fingertips: (T, F, 3) raw (metres). Finger 0 must be the thumb.
        threshold: fraction of the hand span.
    """
    fingertips = np.asarray(fingertips, dtype=np.float32)
    T, F, _ = fingertips.shape
    span = hand_span(fingertips)
    thumb = fingertips[:, 0]  # (T, 3)
    labels = np.zeros((T, F - 1), dtype=np.float32)
    for i in range(1, F):
        d = np.linalg.norm(fingertips[:, i] - thumb, axis=-1)
        labels[:, i - 1] = (d < threshold * span).astype(np.float32)
    return labels


def train_contact_classifier(
    human_norm: np.ndarray,
    labels: np.ndarray,
    config: TrainConfig,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[ContactClassifier, list[dict]]:
    """Train f_c independently of f_m, per the paper.

    Args:
        human_norm: (T, F, 3) normalized fingertips.
        labels: (T, F-1) contact targets.
    """
    T, F, _ = human_norm.shape
    clf = ContactClassifier(
        n_fingers=F, width=config.width, depth=config.depth, activation=config.activation
    ).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=config.contact_lr)

    x = torch.from_numpy(human_norm).float().to(device)
    y = torch.from_numpy(labels).float().to(device)
    pos_weight = config.contact_pos_weight

    history: list[dict] = []
    for epoch in range(config.contact_epochs):
        perm = torch.randperm(T, generator=generator, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, T, config.batch_size):
            idx = perm[start : start + config.batch_size]
            if idx.numel() < 2:
                continue
            scores = clf(x[idx])  # (b, F-1)
            loss = sum(
                contact_loss(scores[:, j], y[idx][:, j], pos_weight) for j in range(F - 1)
            ) / (F - 1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        history.append({"epoch": epoch, "contact": epoch_loss / max(n_batches, 1)})
    return clf, history


def train_mapper(
    mapper: FingerMapper,
    samples: HandSamples,
    clouds: RobotClouds,
    anchors: AnchorSet | None,
    config: TrainConfig,
    device: torch.device,
    generator: torch.Generator,
    checkpoint_callback=None,
) -> tuple[FingerMapper, list[dict]]:
    """Train f_m on the self-supervised losses plus anchor alignment.

    Args:
        checkpoint_callback: optional callable(epoch, mapper, history) invoked
            after each epoch completes, for intermediate checkpoint saving.
    """
    human = torch.from_numpy(samples.fingertips).float().to(device)  # (T, F, 3)
    robot_cloud = torch.from_numpy(clouds.tips_norm).float().to(device)  # (N, F, 3)
    # (F, N, 3) so index i is finger i's cloud.
    robot_cloud = robot_cloud.permute(1, 0, 2).contiguous()
    robot_rot = torch.from_numpy(clouds.tips_rot).float().to(device).permute(1, 0, 2, 3)

    # Human-side local frames come from the glove's own per-keypoint
    # orientation. Eq. (3) applies T^-1 to BOTH sides; using a global frame on
    # one side and a local frame on the other aligns two differently-rotated
    # directions and drives the inner product negative, which is worse than not
    # using the term at all.
    # Stored as (F, T, 3, 3) so the frame axis leads, matching `human` after its
    # transpose: a batch built from frame indices `idx` slices axis 1.
    human_rot_np = quat_wxyz_to_matrix(samples.quats_wxyz)  # (T, F, 3, 3)
    human_rot = (
        torch.from_numpy(human_rot_np).float().to(device).permute(1, 0, 2, 3).contiguous()
    )  # (F, T, 3, 3)

    anchor_h = anchor_r = None
    if anchors is not None and config.use_anchors:
        anchor_h = torch.from_numpy(anchors.human).float().to(device)  # (F, K, 3)
        anchor_r = torch.from_numpy(anchors.robot).float().to(device)

    opt = torch.optim.Adam(mapper.parameters(), lr=config.lr)
    weights = config.loss_weights()
    T = human.shape[0]
    F = human.shape[1]
    if T < 2:
        raise ValueError("need at least two frames: L_dist is a pairwise term")

    history: list[dict] = []
    for epoch in range(config.epochs):
        perm = torch.randperm(T, generator=generator, device=device)
        acc: dict[str, float] = {}
        n_batches = 0
        # Split the epoch into chunks of `batch_size` SAMPLES PER FINGER. The
        # Chamfer and distance terms are set-to-set objectives, so the leading
        # dimension of `human` is the sample set N, not a gradient minibatch:
        # with N < 2 the pairwise L_dist term vanishes and the mapping collapses
        # to a point (the 4.1 regime in the paper's ablation).
        for start in range(0, T, config.batch_size):
            idx = perm[start : start + config.batch_size]
            if idx.numel() < 2:
                continue
            batch = human[idx].permute(1, 0, 2).contiguous()  # (F, N, 3)

            # A short sub-batch of anchors is fed alongside each step, per the
            # paper's "anchor batch size of 32".
            anchors_arg = None
            if anchor_h is not None:
                ka = min(config.anchor_batch_size, anchor_h.shape[1])
                aidx = torch.randperm(anchor_h.shape[1], generator=generator, device=device)[:ka]
                anchors_arg = (anchor_h[:, aidx], anchor_r[:, aidx])

            total, log = mapping_loss(
                mapper=mapper,
                human=batch,
                robot_cloud=robot_cloud,
                weights=weights,
                generator=generator,
                # Pass frames only when the local-frame variant is selected, so
                # the global (GeoRT-style) comparison stays runnable.
                human_rot=human_rot[:, idx] if config.use_local_frames else None,
                robot_cloud_rot=robot_rot if config.use_local_frames else None,
                anchors=anchors_arg,
                jitter_base=config.jitter_base,
                jitter_rand=config.jitter_rand,
            )
            opt.zero_grad()
            total.backward()
            opt.step()

            for k, v in log.items():
                acc[k] = acc.get(k, 0.0) + v
            n_batches += 1

        entry = {"epoch": epoch, "total": 0.0}
        for k, v in acc.items():
            entry[k] = v / max(n_batches, 1)
            entry["total"] += entry[k]
        history.append(entry)
        if checkpoint_callback is not None:
            checkpoint_callback(epoch, mapper, history)
    return mapper, history


def train(
    samples: HandSamples,
    clouds: RobotClouds,
    anchors: AnchorSet | None,
    config: TrainConfig,
    device: str | torch.device = "cpu",
    log_fn=None,
    train_classifier: bool = True,
    raw_fingertips: np.ndarray | None = None,
    checkpoint_interval: int | None = None,
    checkpoint_dir: str | Path | None = None,
    init_mapper_state: dict | None = None,
) -> TrainResult:
    """Train both networks. They are independent, so order does not matter.

    Args:
        checkpoint_interval: if set, save mapper every N epochs to checkpoint_dir.
        checkpoint_dir: directory for intermediate checkpoints (required if checkpoint_interval is set).
        init_mapper_state: if given, FingerMapper starts from this state_dict instead
            of random init (warm start / fine-tuning from an existing bundle). Must
            match (n_fingers, width, depth, activation, output_tanh) in config.
    """
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    F = samples.n_fingers
    mapper = FingerMapper(
        n_fingers=F,
        width=config.width,
        depth=config.depth,
        activation=config.activation,
        output_tanh=config.output_tanh,
    ).to(device)
    if init_mapper_state is not None:
        mapper.load_state_dict(init_mapper_state)
        if log_fn:
            log_fn("warm-started f_m from provided state_dict")

    # jitter_delta draws from the global RNG for direction; seed it so runs are
    # exactly reproducible.
    torch.manual_seed(config.seed + 1)

    # Setup checkpoint callback if requested
    checkpoint_callback = None
    if checkpoint_interval is not None and checkpoint_interval > 0:
        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir required when checkpoint_interval is set")
        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        def checkpoint_callback(epoch, mapper, history):
            if (epoch + 1) % checkpoint_interval == 0:
                ckpt_path = ckpt_dir / f"mapper_epoch_{epoch+1:04d}.pt"
                torch.save({
                    "mapper": mapper.state_dict(),
                    "config": asdict(config),
                    "history": history,
                    "epoch": epoch,
                }, ckpt_path)
                if log_fn:
                    log_fn(f"saved checkpoint: {ckpt_path}")

    t0 = time.time()
    mapper, m_hist = train_mapper(
        mapper, samples, clouds, anchors, config, device, generator, checkpoint_callback
    )
    train_time = time.time() - t0

    clf = None
    c_hist: list[dict] = []
    if train_classifier:
        torch.manual_seed(config.seed + 2)
        # Labels come from the RAW metric positions: the threshold is a physical
        # distance, so it must not be applied to normalized coordinates.
        if raw_fingertips is None:
            raw_fingertips = samples.normalizer.inverse(
                np.transpose(samples.fingertips, (1, 0, 2))
            ).transpose(1, 0, 2)
        labels = derive_contact_labels(np.asarray(raw_fingertips), config.contact_threshold)
        if labels.sum() == 0:
            if log_fn is not None:
                log_fn(
                    "WARNING: no contact labels at threshold "
                    f"{config.contact_threshold}; f_c would learn the trivial "
                    "all-negative solution. Raise the threshold."
                )
        clf, c_hist = train_contact_classifier(
            samples.fingertips, labels, config, device, generator
        )

    # The two networks run for different numbers of epochs, so the histories are
    # reported side by side rather than zipped: pairing mapper epoch 20 with
    # classifier epoch 20 would imply a correspondence that does not exist.
    history = [dict(entry) for entry in m_hist]
    contact_history = [dict(entry) for entry in c_hist]

    if log_fn is not None:
        log_fn(
            f"trained f_m in {train_time:.1f}s over {config.epochs} epochs "
            f"(F={F}, anchors={'yes' if anchors is not None and config.use_anchors else 'no'}, "
            f"local_frames={config.use_local_frames})"
        )

    return TrainResult(
        mapper=mapper,
        classifier=clf,
        history=history,
        config=config,
        contact_history=contact_history,
    )


def save_checkpoint(result: TrainResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mapper": result.mapper.state_dict(),
        "classifier": None if result.classifier is None else result.classifier.state_dict(),
        "config": asdict(result.config),
        "history": result.history,
        "contact_history": result.contact_history,
        "n_fingers": result.mapper.n_fingers,
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[FingerMapper, ContactClassifier | None, dict]:
    device = torch.device(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = TrainConfig(**{k: v for k, v in payload["config"].items() if k != "weights"})
    payload_cfg = payload["config"]
    cfg.weights = payload_cfg.get("weights", cfg.weights)
    F = payload["n_fingers"]
    mapper = FingerMapper(
        n_fingers=F,
        width=cfg.width,
        depth=cfg.depth,
        activation=cfg.activation,
        output_tanh=cfg.output_tanh,
    ).to(device)
    mapper.load_state_dict(payload["mapper"])
    mapper.eval()

    clf = None
    if payload.get("classifier") is not None:
        clf = ContactClassifier(
            n_fingers=F, width=cfg.width, depth=cfg.depth, activation=cfg.activation
        ).to(device)
        clf.load_state_dict(payload["classifier"])
        clf.eval()
    return mapper, clf, payload
