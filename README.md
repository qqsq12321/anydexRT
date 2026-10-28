# anydexRT

A reproduction of **AnyDexRT: Calibration-Free Dexterous Hand Retargeting with
Few-Shot Human Guidance** ([arXiv:2607.08341](https://arxiv.org/abs/2607.08341),
Wang, Feng, Fang, Xia, Yang, Wen, Lu — Noematrix / SJTU / Shanghai Innovation
Institute).

The paper's reference code is not released ("Code: Coming Soon" on the project
page), so this is an independent implementation from the paper text, built to be
**falsifiable** rather than to match the published numbers. See
[What this does not claim](#what-this-does-not-claim).

```
human fingertips --> f_m --> robot fingertips --> f_s (NNS) --> joints --> refine --> filter
                      ^                              ^                      ^
                 few-shot anchors            continuity prior        pinch / fist blend
```

## What is implemented

| Component | Paper reference | Where |
|---|---|---|
| Fingertip mapper `f_m`, F sub-nets of (3,128,128,3) | Appendix B | `anydexrt/models.py` |
| Contact classifier `f_c`, (F−1) sub-nets of (6,128,128,1)+Sigmoid | Appendix B | `anydexrt/models.py` |
| Partial (one-directional, L2-norm) Chamfer | Eq. (1) | `anydexrt/losses.py` |
| Distance preservation | Eq. (2) | `anydexrt/losses.py` |
| Local motion preservation | Eq. (3) | `anydexrt/losses.py` |
| Anchor alignment | Eq. (4) | `anydexrt/losses.py` |
| Contact BCE | Eq. (5) | `anydexrt/losses.py` |
| Per-finger normalize to [−1,1], single max-axis range | Appendix B | `anydexrt/normalize.py` |
| Anchor types + interpolation (λ=2, K=50/100) | Appendix D | `anydexrt/anchors.py` |
| NNS inverse kinematics + pinch template retrieval | Inference | `anydexrt/inference.py` |
| GMC / LMC | §4.1 | `anydexrt/metrics.py` |

### Added beyond the paper, for the AVP / PICO 4 → L20 deployment

The paper's pipeline alone did not produce a usable pinch on these hands, so the
deployed path adds the following. None of it is in the paper, and none of it is
claimed as a reproduction of it:

| Addition | Why | Where |
|---|---|---|
| `CrossFingerMapper` — thumb sub-net also sees the other fingers' positions | The paper's strict per-finger factorization means the thumb sub-net cannot see the index finger at all, so the mapped thumb-index gap barely moved (95→76 mm) as the real gap went 80→0 mm: the hand could not pinch | `anydexrt/models.py` |
| `pinch_mode="continuous"` — blend toward a pinch pose by the *real* thumb-finger distance, no classifier | `f_c` has precision/recall 0.00 on ring and pinky here (never fires), and where it does fire it replaces the whole pose in one frame | `anydexrt/inference.py` |
| Hand-calibrated pinch and fist poses | Every automatic template-scoring rule tried (gap, palm distance, direction alignment, joint engagement) was rejected on inspection; `_tip_offsets` put the measured "fingertip" 10–11 mm beyond the visible mesh surface, so all of them scored against a wrong reference | `scripts/calibrate_pinch.py`, `scripts/calibrate_fist.py` |
| Fist refinement, gated on thumb straightness | Zeroing a curled finger's slot in the thumb context is out-of-distribution for the net; the thumb landed far from the palm. A *mm* threshold cannot tell a fist from an extended thumb on PICO 4 (both 90–122 mm), a shape ratio can on both devices | `anydexrt/inference.py` |
| Joint-space low-pass + stronger NNS continuity prior | `f_s` is a discrete lookup, so tracking noise becomes joint jitter — see [Jitter with a still hand](#jitter-with-a-still-hand) | `anydexrt/inference.py`, `anydexrt/kinematics.py` |
| PICO 4 wire protocol client | Independent reimplementation, so the live path does not depend on another repository | `anydexrt/pico.py` |

Dependencies are deliberately narrow: `mujoco`, `torch`, `numpy`, `scipy`. This
does **not** depend on `mimic_retargeter_lab`, whose JAX/hydra/dash stack is
unrelated to the method.

## Quick start

```bash
conda activate anydex          # mujoco + torch + numpy + scipy
export PYTHONPATH=/home/qsq/qqsq12321/anydexRT

pytest tests/ -q                                   # 139 unit tests
python scripts/validate_chamfer_collapse.py        # rung 1 alone
python scripts/run_ladder.py --epochs 60           # the full ladder
```

For the AVP / PICO 4 → L20 path that is actually deployed, see
[Two kept bundles](#two-kept-bundles-one-per-capture-device) — train with
`scripts/train_bundle.py`, run live with `scripts/teleop_avp.py`.

The hand models come from `mimic_retargeter_lab/assets/mjcf`, and the human
captures from `mimic_retargeter_lab/dataset/manus`. Both paths are overridable
(`--capture-dir`, `HandSpec.resolved_mjcf`).

```python
from anydexrt import get_hand, HandKinematics
from anydexrt.data import load_manus_npz, normalize_human, build_robot_clouds
from anydexrt.anchors import build_anchor_set
from anydexrt.train import TrainConfig, train
from anydexrt.inference import AnyDexRT, make_pinch_templates

spec = get_hand("leap_hand")
raw, quats = load_manus_npz(capture_path, spec)
samples = normalize_human(raw, quats, spec.finger_names)
# The robot cloud gets its OWN per-finger normalizer (see errata 4) -- do not
# pass the human one here.
clouds, kin = build_robot_clouds(spec, n_samples=20000, frame="palm")
anchors = build_anchor_set(spec, raw, samples.normalizer, kin,
                           robot_normalizer=clouds.normalizer)

result = train(samples, clouds, anchors, TrainConfig(hand="leap_hand"), raw_fingertips=raw)
rt = AnyDexRT(result.mapper, clouds, kin, samples.normalizer,
              result.classifier, make_pinch_templates(clouds, kin))
qpos = rt.retarget_sequence(raw)          # (T, F, 3) -> (T, J)
```

## Where this reproduction had to make a choice

Every item below is something the paper does not specify. Each is a real fork in
the road, not a detail — changing one changes the numbers.

### 1. The human-to-robot finger correspondence for four-fingered hands

**This is the largest genuine gap in the paper.** It writes "for a hand with F
fingers, we employ F sub-nets for `f_m`", but never says which human finger maps
to which robot finger, nor what happens to the human pinky when the robot has
four fingers.

Leap Hand and Allegro are four-fingered. Leap is also exactly the hand where
GeoRT beats AnyDexRT on GMC (73.4 vs 54.5 in the paper's Table 1), so the choice
is not academic.

This implementation drops the pinky, matching the convention already used in
`mimic_retargeter_lab/config/retargeter_cfg/geort/human_hand_to_{leap,wonik_allegro}_hand.yaml`.
The correspondence is **never implicit in code**: it lives in
`anydexrt/hands.py`, one `FingerSpec` per finger. A consequence worth stating —
dropping the pinky also drops thumb-pinky pinch entirely, since `f_c` only has
`F−1` heads.

### 2. Contact labels

The paper says contact "signals are easy to collect and annotate" but the shipped
Manus captures carry no contact channel. Labels here are geometric: contact when
the thumb and finger i are closer than a fraction of the hand span.

The threshold is 0.2, chosen from the data rather than guessed: on both captures
every finger lands at a 6–10% contact rate there and the rate stops rising
steeply past it. The first attempt used 0.12 and produced **zero** labels for
subject 000's index finger — the classifier would have learned the trivial
all-negative solution while the loss curve looked perfectly healthy. `train()`
now warns loudly when a threshold yields no positives.

Two bugs were caught here that are worth naming, because both are silent:

- The span was originally defined as `‖thumb − pinky‖` and then used as the
  denominator for the pinky's own distance, making that ratio identically 1.0
  and permanently labelling the pinky as not-in-contact. `hand_span()` now uses
  the mean pairwise fingertip distance, which has no self-reference.
- The Manus wrist keypoint (index 0) is constant across all 5000 frames in both
  captures, i.e. an invalid channel. It is not used, but it is a reason not to
  fit anything on the raw 25-keypoint layout.

### 3. Few-shot anchor collection

Appendix D's step (P1) is a **live operator session** — someone wearing the glove
imitating reference gestures. There is no such session here, so
`build_human_anchors` offers two strategies:

- `reference` — pick the frames from an existing capture that best match each
  anchor constraint. Real data, but only valid if the capture covers the range.
- `synthetic` (default) — build anchors from a per-finger plane model fitted to
  the capture. Always available and monotonic in the anchor parameter, at the
  cost of approximating a real hand.

`reference` falls back to `synthetic` when the mean matching residual exceeds
`coverage_tol` of the finger's length, and the strategy actually used is recorded
in the `AnchorSet`. Steps (P2) and (P3) follow the paper exactly.

**This means `L_align` here is not the `L_align` the paper measured.** It is the
same objective on approximated anchors.

### 4. Loss weights

Appendix B is explicit: `L_mapping = L_P-Chamfer + L_dist + L_motion + L_anchor`
and "we do not specifically tune the weights of each loss term" — so every weight
is 1, and that is the default here.

But the four terms do not share a scale, and the paper's own ablation is the
evidence. `L_P-Chamfer` uses the **L2 norm** (the paper says so: "‖·‖ stands for
L2 normalization") while `L_dist` is **squared** (Eq. 2's `(…)²`). At a collapsed
configuration on Leap Hand, measured here:

| | value at collapse | value if distances preserved |
|---|---|---|
| `L_P-Chamfer` | 0.343 | 0.346 |
| `L_dist` | 0.196 | 0.000 |

Note `L_P-Chamfer` barely moves between the two states: it floors at the distance
from the collapse point to the nearest robot sample, not at zero. The collapse
equilibrium is therefore a real tug-of-war between two comparable terms, and
which way it settles depends on the relative weighting. `scripts/run_ladder.py
--skip-sweep false` reports the sweep; on Leap Hand at 60 epochs:

| `w_dist` | mapped/source spread |
|---|---|
| 0 | 0.019 |
| 1 (paper) | 0.390 |
| 3 | 0.694 |
| 10 | 0.792 |
| 30 | 0.825 |

With the paper's uniform weighting the cloud recovers only partially within this
implementation's training budget, and is still climbing at 60 epochs. The paper
reports 83.8/100 LMC at epoch 20. Uniform weights are kept as the default —
deviating would be tuning to a number this reproduction cannot validly target —
and this divergence is treated as a finding, not a bug to paper over.

### 5. Local frame `T(x)`

The paper fixes the robot side: "since `f_m^i(x_j^{H,i})` does not provide a
local rotation, we assign it the rotation of its nearest neighbour in `C^{R,i}`".
The **human side is not specified**. `anydexrt/frames.py` takes the Manus
per-keypoint quaternion, which is the natural reading — the whole point of local
frames is to cancel glove-induced frame error, and the glove does report
orientation. `motion_consistency(local=...)` exposes the global/local switch so
both the GMC and LMC conventions can be reproduced from the same code.

### 6. Jitter magnitude for `L_motion`

"Samples are randomly perturbed with a generated delta movement" — no magnitude
given. The default is `0.001 + rand(0,1) * 0.01`, taken from GeoRT's direction
loss, which lives in the same normalized space. Exposed as
`TrainConfig.jitter_base` / `jitter_rand`.

### 7. Other choices

- **Activation** not named; LeakyReLU (GeoRT's choice) is the default.
- **BatchNorm** omitted. GeoRT uses `BatchNorm1d`, but the paper's size tuples
  list only Linear layers, and BN would make the mapper non-deterministic at
  inference.
- **Mapper output activation** linear by default (`output_tanh` available).
- **Pinch templates**: "pre-generated" without saying how.
  `make_pinch_templates` pairs the thumb table with each finger table (the
  fingers are independent chains, so any pairing is a valid hand pose) and keeps
  the closest pairs, measured in metres in the palm frame. Median template gap
  is under 1 mm on Wuji and L20 with a 20k-sample cloud.
- **`f_s` solver**: per-finger NNS with a continuity prior toward the previous
  frame, which is what "from the given fingertip positions and the last joint
  configurations" implies. See the errata below — the first version of this
  solver was wrong.
- **Metrics**: `score = 100 × mean(cosine)`. This reading is forced by the
  reported 4.1 for the collapsed case — a collapsed mapping's displacements are
  uncorrelated with the human's, so the mean cosine of noise is ~0. See
  `tests/test_metrics.py`, which pins that interpretation down.

## The falsifiable ladder

`scripts/run_ladder.py` runs the checks below, each with a pass condition that
fails loudly if the mechanism is absent. The point is that these hold regardless
of whether this implementation reproduces the paper's absolute numbers.

Measured on Leap Hand, 3000 frames, 60 epochs (`results/ladder.json`):

| Condition | spread | direction agreement |
|---|---|---|
| `L_P-Chamfer` only | 0.019 | **4.5** |
| `+ L_dist` | 0.389 | 67.5 |
| `+ L_motion` | 0.179 | **99.2** |
| `+ L_align` (full) | 0.202 | 97.9 |

| Rung | Prediction | Result |
|---|---|---|
| 1 | Chamfer alone collapses (paper: LMC 4.1) | **PASS** — 0.019, agreement 4.5 |
| 2 | `L_dist` restores spread (paper: → 83.8) | **PASS** — 0.019 → 0.202 |
| 3 | `L_motion` improves agreement | **PASS** — 67.5 → 99.2 |
| 4 | `L_align` reduces cross-seed variance | **FAIL** — 0.0833 with anchors vs 0.0647 without |
| 5 | Input rotation ±45°/±90° barely moves the score | **PASS** — 96.5/95.9/96.3/95.1, std 0.54 |

**4/5.** The passing rungs are the strong signal:

- **Rung 1** is the sharpest: an independent implementation collapsed to a
  direction agreement of **4.5** where the paper reports **4.1**, and for the
  reason the ablation implies — one-directional Chamfer's optimum is a point
  mass and `L_dist` is the only thing resisting it.
- **Rung 2** confirms `L_dist` is load-bearing rather than decorative.
- **Rung 5** independently reproduces the paper's Fig. 5 claim. Rotating the
  input by ±90° moves the score by less than a point. With a deliberately
  mismatched frame pair the same measurement gives a base of −35.7 and a spread
  of 12.25, so this rung is discriminative, not vacuous.

**Rung 4 fails for a structural reason, not a bug.** The anchors are synthetic
(choice 3), and they are not plausible poses: 35–55% fall outside the real
workspace's bounding box, and their median distance to the nearest real sample
is 15–26% of the finger's span. `L_align` is therefore pulling the mapping
toward geometrically unreachable configurations — it adds error instead of
resolving ambiguity. The `reference` strategy also falls back to `synthetic`,
confirming that this free-swinging capture does not cover the anchor range. No
variant of this experiment can validate `L_align` without a live anchor session,
which is exactly the limitation the paper lists first.

Note also that `+ L_motion` *reduces* the spread ratio (0.389 → 0.179) while
raising agreement to 99.2. Spread and agreement are different criteria: `L_dist`
preserves the cloud's geometry, whereas `L_motion` only constrains the direction
of motion, and can do so while contracting the cloud. The paper's Table 3 only
reports LMC, so this trade-off is invisible there.

### One real bug the ladder caught

The first ladder run scored `L_motion` at **−24.3** — worse than not using the
term. The cause was the unspecified human-side frame (choice 5): the robot-side
frame is a fingertip body orientation local to the palm, the glove quaternion is
absolute, and pairing them inserts a relative rotation whose mean is ~30° and
whose spread is much larger, driving the inner product negative. Both attempted
local readings measured −28.1; the global reading measures 99.2. Since the paper
does not define the human frame, `use_local_frames` now defaults to `False` and
the finding is recorded in `TrainConfig`.

Two further silent failures were caught the same way and are documented in the
code: the contact classifier converging to the all-negative solution at the
paper's learning rate (choice in `TrainConfig.contact_lr`/`contact_epochs`), and
the hand-span reference being defined as thumb-to-pinky distance, which made the
pinky's own contact label identically negative.

## Apple Vision Pro → Wuji Hand / Linker L20

The current target. `scripts/run_avp.py` trains and evaluates; `scripts/baseline_anydexretarget.py`
runs the existing optimizer (`AnyDexRetarget`, AdaptiveOptimizerAnalytical, shipped
AVP configs) on the same frames; `scripts/render_avp.py` writes
`results/avp/contact_sheet.png`.

**Setup.** `avp1.pkl` (2751 frames, MediaPipe 21 landmarks, wrist frame, no
orientations), right hand. Ten contiguous blocks; blocks 5 and 9 (551 frames) are
held out, so the test set contains the recording's *second* pinch sequence
while training only saw the first. 300 epochs — at 2200 frames and batch 2048,
the paper's 20 epochs is ~40 gradient steps and leaves `L_dist` at 0.26 vs 0.003.
Every number below is on held-out frames, on the robot side after MJCF FK, in
the palm frame.

**Frames.** Human and robot are both expressed in a palm frame built from their
own skeletons (x wrist→middle MCP, y pinky MCP→index MCP, z normal). No tuned
rotation. Wrist→tip direction cosine human vs robot: 0.92–0.94 (Wuji),
0.94–0.97 (L20), against 0.40 … −0.68 unaligned. Any rotation of the input is
removed exactly (tested), which is where this pipeline's calibration-freeness
comes from.

| Wuji (20 drives) | GMC robot | `f_s` residual | pinch gap i/m/r/p (mm) | pinch closed | jitter (rad/frame) | Hz |
|---|---|---|---|---|---|---|
| AnyDexRetarget (existing) | 62.6 | — | 3.1 / 1.9 / 3.0 / 4.0 | 94/93/81/100 % | **0.0121** | 38 |
| identity (normalization only) | 87.6 | 13.1 mm | 30 / 43 / 46 / 54 | 0 % | 0.0198 | 3873 |
| `L_P-Chamfer` | −15.0 | 1.0 mm | 36 / 27 / 46 / 51 | 0 % | 0.0036 | 1826 |
| `+ L_dist` | 0.3 | 3.2 mm | 54 / 62 / 93 / 94 | 0 % | 0.0207 | 1874 |
| `+ L_motion` | 85.7 | 3.2 mm | 55 / 55 / 38 / 54 | 0 % | 0.0176 | 1765 |
| `+ L_align`, no `f_c` | **88.2** | 2.7 mm | 37 / 48 / 42 / 66 | 0 % | 0.0203 | 1860 |
| **full** (+ `f_c`) | 83.5 | 2.7 mm | **0.5 / 0.7 / 0.7 / 1.5** | **100 %** | 0.0227 | 1179 |

| Linker L20 (16 drives) | GMC robot | `f_s` residual | pinch gap i/m/r/p (mm) | pinch closed | jitter | Hz |
|---|---|---|---|---|---|---|
| AnyDexRetarget (existing) | 67.5 | — | 23.5 / 30.3 / 30.6 / 22.3 | 14/0/6/0 % | 0.0130 | 59 |
| identity | 82.4 | 17.3 mm | 81 / 84 / 62 / 58 | 0 % | 0.0142 | 3955 |
| `L_P-Chamfer` | 4.6 | 0.5 mm | 32 / 51 / 52 / 56 | 0 % | 0.0015 | 1879 |
| `+ L_dist` | −1.5 | 1.4 mm | 99 / 135 / 138 / 142 | 0 % | 0.0124 | 1607 |
| `+ L_motion` | 86.5 | 2.5 mm | 80 / 74 / 59 / 68 | 0 % | **0.0108** | 1856 |
| `+ L_align`, no `f_c` | **89.0** | 2.7 mm | 47 / 48 / 48 / 74 | 0 % | 0.0120 | 1827 |
| **full** (+ `f_c`) | 86.8 | 2.7 mm | **0.7 / 0.8 / 1.0 / 1.8** | **100 %** | 0.0145 | 1201 |

Pinch frames in the test set: 35 / 27 / 32 / 28. False pinches (human gap > 5 cm,
robot < 2 cm): 0 % for full on both hands. `f_c` precision / recall:
1.00/1.00, 0.84/1.00, 1.00/1.00, 0.89/1.00. Seed stability (3 seeds, std of
mapped targets): 1.11 mm Wuji, 0.89 mm L20.

What the tables say, read carefully:

- **Collapse reproduces on new hands and a new input.** Chamfer alone: mapped
  direction agreement 5.4 (Wuji) and 9.5 (L20), against the paper's 4.1.
- **`L_dist` does not restore direction, `L_motion` does.** `+ L_dist` leaves
  agreement near zero (2.5 / −3.7 mapped). Pairwise distances are invariant to
  rotation and reflection, so `L_dist` restores the cloud's shape but not its
  orientation; `+ L_motion` takes it to 99. The paper's ablation order makes
  `L_dist` look like the term that fixes direction (4.1 → 83.8); on this data it
  is `L_motion`. The earlier Leap run's 67.5 for `+ L_dist` was a lucky
  orientation.
- **`L_align` helps modestly** once anchors are paired correctly: robot GMC
  85.7 → 88.2 and 86.5 → 89.0, smaller pinch gaps.
- **Every pinch closure comes from `f_c` + templates.** Without `f_c` the robot
  never closes a pinch (gaps 37–74 mm). Caveat: with AVP there is no contact
  ground truth, so `f_c` is trained on geometric proximity labels computed from
  the same fingertips it sees. It is a learned proximity detector here, not the
  paper's "contact the sensor missed" detector, and 100 % closure is partly by
  construction. The snap to a template also costs ~5 GMC points and adds jitter.
- **Against the existing optimizer.** Higher GMC on both hands, and on L20 a far
  better pinch (its shipped AVP config closes 14 % / 0 % / 6 % / 0 %; a
  `.bak` of that config dated 2026-10-14 suggests it is mid-tuning). On Wuji the
  existing optimizer's pinch is already good (81–100 %) and **it is smoother**:
  AnyDexRT's Wuji jitter is ~1.9× higher, from NNS table quantization and
  template snapping. GMC also rewards direction preservation, which the optimizer
  deliberately trades away (segment scalings, pinch attraction), so it is not a
  neutral metric for that comparison.
- **Speed.** ~1300 Hz end to end on CPU vs 38–59 Hz for the SLSQP optimizer.

### Jitter with a still hand

`f_s` is a discrete table lookup, so fingertip tracking noise becomes joint
jitter even when the operator is not moving. Measured on a PICO 4 session with
the hand held still: 0.53 mm/frame of fingertip noise, amplified ~8× by `f_m`
to 4.2 mm/frame of target motion, which selected table entries up to **71 % of
a joint's range apart** in a single frame.

The continuity prior is what suppresses this, and its two defaults were both too
weak to: `prior_weight` 0.05 contributed 0.004–0.011 to the cost while the
candidates' own tip distances spanned 0.002–0.008 (same order, so the prior
rarely decided anything), and `n_candidates` 8 often did not even contain the
continuous-looking entry. Now **32 candidates, weight 0.5**, measured over three
recording/bundle combinations:

| | jitter, all frames | jitter, still frames | tip error |
|---|---|---|---|
| PICO rec / AVP bundle | 0.733 → **0.553** | 0.114 → **0.073** | 16.8 → 18.2 mm |
| AVP rec / AVP bundle | 1.439 → **1.166** | 0.377 → **0.261** | 17.7 → 19.0 mm |
| PICO rec / PICO bundle | 0.670 → **0.497** | 0.118 → **0.068** | 13.5 → 14.7 mm |

(% of joint range per frame, after the 6 Hz output filter.) Jitter drops ~25 %
overall and ~38 % when still, and the trade is real: the lookup now accepts a
slightly worse tip match to stay continuous, costing 1.1–1.4 mm of tip error,
and 32 candidates cost ~1586 → ~1282 Hz. Worst-case single-frame jump did not
improve (25.2 % → 26.9 %) — the prior reduces how often a far entry wins, not
how far the farthest one is.

A deadband on target motion (skip the lookup below a threshold) was tried and
**removed**: requiring all five fingers under 0.0008 fired on only 2.4 % of
frames and changed jitter by 0.001, because a hand that is "still" to the eye
still has one finger drifting above any threshold loose enough to be safe.

### Live use

```python
from anydexrt.inference import load_bundle
rt = load_bundle("results/compare/linker_l20_C_cross_finger_bundle.pt")
qpos = rt.retarget_avp_poses(avp_right_fingers)       # (25, 4, 4) from avp_stream -> (16,) drives
qpos = rt.retarget_mediapipe_points(pico_right_hand)  # (21, 3) from anydexrt.pico -> same
```

Both entry points rebuild the palm frame from each live frame's own skeleton, so
no stored calibration is needed. Drives are in `rt.kin.hand.joint_names` order;
coupled joints (L20 DIP/IP) follow via `rt.kin.hand.drive_targets`.

### Two kept bundles, one per capture device

`scripts/train_bundle.py` trains one deployable CrossFingerMapper bundle from a
recording — either device, since `scripts/record_avp.py` writes the same
(21, 3) MediaPipe pkl for both. Two are kept:

| | bundle | trained on | open-hand gap (p90, mm) |
|---|---|---|---|
| AVP | `results/compare/linker_l20_C_cross_finger_bundle.pt` | `data/avp_20261016_143808.pkl` | 105.6 / 123.7 / 125.0 / 137.2 |
| PICO 4 | `results/pico4/linker_l20_C_cross_finger_bundle.pt` | `data/pico4_check.pkl`, 2553 frames | 120.4 / 153.0 / 157.4 / 162.5 |

Same recipe for both (300 epochs, 20000-sample palm-frame cloud, skeleton
anchors, full loss, seed 0), so the recording is the only difference. Neither
the bundles nor the recordings are in git (`.gitignore`: `*.pt`, `results/`,
`data/*.pkl`) — record and train to reproduce them:

```bash
python scripts/record_avp.py --input pico4 --out data/pico4_check.pkl --seconds 40
python scripts/train_bundle.py --rec data/pico4_check.pkl --out results/pico4
```

Then run live:

```bash
# AVP bundle, AVP input
python scripts/teleop_avp.py --hand linker_l20 --input avp --avp-ip <headset-ip> \
  --bundle results/compare/linker_l20_C_cross_finger_bundle.pt --pinch-mode continuous
# PICO 4 bundle, PICO 4 input
python scripts/teleop_avp.py --hand linker_l20 --input pico4 \
  --bundle results/pico4/linker_l20_C_cross_finger_bundle.pt --pinch-mode continuous
```

`--input` also picks the pinch gate (`PINCH_MIN_EXTENSION_MM`: 110 mm for AVP,
90 mm for PICO 4). PICO measures the same hand as reaching 20–35 mm less far, so
at the AVP gate a ring or pinky pinch never engages at all. The bundle's own
`extra['open_gap_mm']`, printed at startup, is the pinch blend's alpha=0
reference — that is what the table's last column is, and why it differs by
device.

A bundle is **not** separable from how its robot cloud was built. `load_bundle`
rebuilds the cloud with `frame="palm"` and the robot's own normalizer, so a
mapper trained against a base-frame cloud, or against one normalized with the
*human* normalizer, lands in a different space than inference looks up in:
every finger comes out rotated and the thumb will not open. Train through
`scripts/train_bundle.py`, which mirrors `run_avp.py`'s data path for exactly
this reason, rather than assembling the setup by hand — this was hit twice while
adding PICO support, silently both times, with healthy-looking loss curves.

### Real hardware (Linker L20)

`anydexrt/real_l20.py` drives the physical hand over RS485/Modbus. Two opt-ins
are required, so a copy-pasted command cannot move hardware:

```bash
python scripts/teleop_avp.py --hand linker_l20 --input pico4 --real
#   -> dry run: the 30 position registers are computed and printed,
#      the serial port is never opened
python scripts/teleop_avp.py --hand linker_l20 --input pico4 --real --confirm-real
#   -> opens /dev/ttyUSB0 and moves the hand
```

Run dry first and watch the registers track the hand sensibly. Useful flags:
`--l20-port`, `--l20-slave-id` (0x29 right, 0x2A left), `--l20-speed`,
`--l20-current-limit`, `--l20-open-on-exit` (leave the hand open instead of
holding its last pose), `--l20-clear-faults` (if it came up latched after a
stall). Needs `pyserial`.

Joint **limits** need no conversion — anydexRT's L20 limits equal the driver's
per-joint, which `tests/test_real_l20.py` asserts — but the qpos→register
mapping is **not** the identity. Three measured corrections sit in between, all
calibrated against the physical hand by GEORT rather than derived from the URDF:

- **Offsets.** The four finger MCP pitches are shifted −0.152 rad (−8.7°):
  simulation zero is not the hand's zero. The thumb CMC roll offset
  (`--l20-thumb-roll-offset`) is **off by default**, where the original driver
  had 0.15 — that pinned qpos=0 to the alignment curve's left point, i.e. the
  hard stop. Off keeps the thumb further out through the usable range (register
  220 → 242 at qpos 0.2) but costs the end of travel: qpos 1.39 maps to
  register 9 rather than 0, so if the thumb cannot close far enough to complete
  a pinch, raise it to ~0.05. Only a physical hand settles this.
- **A piecewise curve** for three thumb joints, measured at three points each.
  It differs from linear limit normalization by up to 0.10 of full scale (~26
  counts) — e.g. thumb CMC roll at qpos 0, thumb MCP at 0.3 — so it is used
  where it exists and linear normalization only elsewhere.
- **Direction inversion** for the thumb and every pitch/PIP joint. Finger MCP
  roll is deliberately *not* inverted, matching the register protocol.

Each command is 30 registers: `roll(5) + yaw(5) + root1(5) + distal(5)×3`. The
distal block repeats because PIP, DIP and the fingertip segment share one
value, and only the thumb has a real roll register (the other four carry the
neutral 128). Identical consecutive commands are skipped — the retargeter runs
at ~1300 Hz while the registers are 8-bit, so most frames produce the same
bytes.

This is an independent reimplementation of the protocol and the calibration, so
nothing is imported from another repository (the same approach as
`anydexrt/pico.py`). It was checked register-for-register against the original
driver over 804 poses — 300 frames of real replay, 500 random in-range, and the
limit corners — with zero differences, plus matching CRC-16. With
`--l20-thumb-roll-offset 0.15` that equivalence still holds exactly, so the
thumb roll offset is the only intentional divergence. **It has not been
run against the physical hand from this repository**; the equivalence check and
`tests/test_real_l20.py` (20 tests, fake serial port) are what stands behind it.

## Errata (found while porting to Wuji / L20 / AVP)

Four defects in the first version, all silent — every loss curve looked healthy.
They affect the Leap Hand / Manus ladder above, which has **not** been re-run;
treat its numbers as superseded by the AVP section below.

1. **`f_s` returned wrong poses.** NNS looked up whole-hand samples and voted
   across fingers. With 16–20 DoF a single random sample is essentially never
   near the target for every finger at once, so it returned a compromise pose:
   on reachable targets the mean fingertip error was **48.6 mm on Wuji and
   42.6 mm on L20** (p95 > 110 mm). The per-finger lookup now used is exact by
   construction (each finger is an independent chain) and measures **1.5 mm and
   1.0 mm**. The earlier "958 Hz" figure was measured on the broken solver.
2. **Robot anchors were paired at random.** (P3) sampled K configurations
   uniformly, unrelated to the anchor parameters, so the j-th human "lateral
   30%" anchor was paired with an arbitrary robot pose and `L_align` fit noise.
   This, more than synthetic human anchors, explains rung 4's failure. Robot
   anchors now follow the same parametrization (abduction sweep / flexion chain
   (b1, b1, b1/λ)), with joint roles derived from the joint axes.
3. **The two sides were never in a common frame.** Manus world coordinates and
   the Leap MJCF frame were used as-is; direction terms (`L_motion`, GMC) compare
   vectors across those frames. On AVP → L20 the unaligned direction cosine of
   the pinky is −0.68. Both sides are now expressed in a palm frame derived from
   their own skeletons (see below).
4. **The robot cloud was normalized with the human normalizer.** Appendix B says
   "for each human/robot finger"; each side now gets its own per-finger
   normalizer. The pinch templates were also chosen by distances in that
   normalized space, where thumb and finger have different centres and scales;
   they are now chosen in metres.

Defect 4 **recurred** in 2026-10 while training the first PICO 4 bundle, from a
hand-written training script that passed the human normalizer to
`build_robot_clouds` and left the cloud in the base frame instead of
`frame="palm"`. Symptom on the robot: all four fingers skewed to one angle and
the thumb would not open, while training itself looked healthy. The lesson is
not about the normalizer specifically — it is that the mapper and
`load_bundle`'s cloud reconstruction have to agree, and nothing checks that they
do. `scripts/train_bundle.py` exists so the setup is written once.

## What this does not claim

- **Not the paper's numbers.** The LMC/GMC implementation is unpublished and the
  reference code is unreleased, so a difference between this implementation's
  score and the paper's cannot be attributed to the method. Rung 9's absolute
  values are therefore not targets.
- **Four of seven hands.** Wuji, Leap, Allegro and Shadow are available
  locally; Inspire, Ability and XHand are commercial units without public MJCF
  models. The paper's average over seven hands is not reproducible here.
- **No real-world results.** Table 4 needs a Manus glove, an HTC Vive Tracker, a
  Flexiv Rizon 4 and a Wuji Hand, plus eight human operators over four tasks.
  Nothing in this repository validates teleoperation performance.
- **Anchors are approximated.** See choice 3. `L_align`'s contribution here is
  measured on synthetic anchors, so it is not comparable to the paper's.
- **Downstream policies are untested.** The paper itself leaves this open, and so
  does this reproduction.
- **The deployed path's pinch/fist quality is judged by eye, not measured.** The
  calibrated poses, the gate thresholds and the fist discriminator were accepted
  or rejected by looking at the MuJoCo viewer. What *is* measured is narrower:
  jitter and tip error (above), and whether a threshold separates two pose
  groups on recordings. There is no metric here for "does this look like a
  proper pinch".
- **One operator, one short recording per device.** Every threshold in the
  deployed path comes from this operator's hands: `data/pico4_check.pkl` is 40
  seconds / 2553 frames, and the fist statistics behind `fist_thumb_bent` rest
  on 140 fist frames. Several thresholds are also absolute distances in mm
  (`pinch_min_extension_mm`, `fist_open_reach_mm`), which is why they already
  need separate values per device — another operator will likely need them
  retuned again.

## Layout

```
anydexrt/
  hands.py        hand registry: fingertips, drive joints, tendon couplings
  kinematics.py   MJCF forward kinematics; NNS inverse solver
  normalize.py    per-finger normalization to [-1, 1]
  frames.py       local coordinate frames, quaternion conversion
  models.py       FingerMapper, CrossFingerMapper, ContactClassifier
  losses.py       Eqs. (1)-(5) and the jitter
  anchors.py      Appendix D anchor generation
  data.py         Manus / AVP / MediaPipe loading, robot cloud sampling
  metrics.py      GMC / LMC
  train.py        training loop, checkpointing
  inference.py    end-to-end retargeting, pinch + fist refinement, bundles
  avp.py          Apple Vision Pro skeleton handling
  pico.py         PICO 4 wire protocol client (relay + direct), independent impl
  real_l20.py     Linker L20 RS485/Modbus output, independent impl
scripts/
  train_bundle.py                train one deployable bundle from a recording
  run_avp.py                     the AVP ablation ladder + evaluation
  teleop_avp.py                  live AVP / PICO 4 -> MuJoCo (and --real L20)
  record_avp.py                  record either device to a (21, 3) pkl
  pico4_daemon.py                PICO 4 relay daemon
  calibrate_pinch.py             hand-tune the four pinch poses in the viewer
  calibrate_fist.py              hand-tune the natural fist pose
  validate_chamfer_collapse.py   rung 1 in isolation
  run_ladder.py                  the full ladder
calibrated_pinch_poses.yaml      output of calibrate_pinch.py (L20)
calibrated_fist_pose.yaml        output of calibrate_fist.py (L20)
tests/           139 unit tests (losses, metrics, normalization, frames,
                 smoothing, AVP, PICO protocol round-trips, L20 registers)
```

## Licence note

`facebookresearch/GeoRT` is CC-BY-NC. No GeoRT code is vendored here — the
losses and models are written from the AnyDexRT paper — but the *design* of this
reproduction is informed by reading it, and the jitter magnitude default is
taken from it. Research use.
