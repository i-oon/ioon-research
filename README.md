# Cross-Embodiment Locomotion via Latent Action World Models

Can a model learn, **from video alone**, what movement is happening — separately from which body is
doing it — so that the same latent drives a robot it has never seen?

A locomotion policy is tied to the body it was trained on: shorten a leg, redistribute the mass, or
change the number of legs, and it stops working. Retraining costs hours to days, every time. This
project asks whether a world model trained on video can carry behaviour across that gap, with **no
morphology label and no kinematics given**.

## Why vision, stated carefully

The honest version of the claim, because the obvious one is refutable.

Morphology-agnostic *proprioceptive* control exists — joints as a token set over the kinematic
graph — so "proprioception cannot do this" is not defensible. **The defensible claim is that those
methods must be handed the kinematic tree, and a camera has to be handed nothing.** This pipeline is
given video of a Unitree B1 and knows nothing else about it.

That is also why the two robots are chosen to be *incomparable*. Three leg lengths share an 18-D
joint space, so proprioception transfers between them too and vision wins only on convenience. A
**12-DOF quadruped against an 18-DOF hexapod share no joint correspondence**, while one camera
describes both in `256×256×3` whatever the body.

## What it is for

**A robot about which nothing is known.** No kinematic tree, no URDF, no action labels — only video
of it moving. Morphology-agnostic *proprioceptive* control exists, but those methods must be handed
the kinematic graph. A camera has to be handed nothing.

What the world model supplies is knowledge of **how to drive joints so that the result is
locomotion** — the expensive part of bringing up a new robot, and the part that otherwise costs a
training run per body.

## The contribution, and the result

The action target is **joint space**, not task space. LAC-WM's targets — end-effector poses,
fingertip positions — live in one physical frame shared by every embodiment, so a fingertip at
`(x,y,z)` already means the same thing for a human hand and a gripper. **An 18-DOF hexapod and a
12-DOF quadruped share no such frame**: no dimension of one corresponds to any dimension of the
other. Decoding into the shared space would hand us the correspondence instead of making the model
learn it.

That poses the question the experiments answer — *can a joint-space action target work at all when
no shared action space exists?*

| | within-robot joint error | cross-robot transfer |
|---|---|---|
| joint target, no body term | 0.3517 | **−28.9 / −43.1** |
| joint target + shared body term | **0.2183** | **+0.610 / +0.573** |

**A joint-space target works within a robot on its own; it crosses robots only with a shared
body-motion term** — which also improves the within-robot decoding by 38%. The term supervises one
dimensionless number both robots share, and transfer appears **channel by channel**: yaw sits at
−5.2 when it is not supervised and +0.37 when it is, on identical data and architecture.

## The two stages

| | question | bodies |
|---|---|---|
| **Stage 1** | cross-**morphology** — does a latent transfer to an unseen leg geometry? | stick-insect hexapods, scaled coxa/femur/tibia |
| **Stage 2** | cross-**embodiment** — does it transfer to a different *kind* of robot? | the hexapod against a Unitree B1 quadruped |

Stage 1 is finished. Stage 2's transfer and few-shot goals are met; the open question is whether a
**shared body-motion target** gives the two robots a common language rather than a switch.

## Architecture

```
frozen V-JEPA2  ─→  e_t
                     ├── ITM (e_t, e_t+1) ──→  z    the latent action
                     ├── FTM (e_t, z)     ──→  ê_t+1   the world model
                     └── MotionDecoder (e_t, z) ──→ joint commands
                                                     per-embodiment heads
                                                     + one shared body head
```

`L = λ_recon·L_recon + λ_motion·L_motion + λ_body·L_body`, where **`L_body` is the only term that
asks the same `z` to decode the same way on both robots**. As trained: `1.0 / 1.0 / 0.5`, with the
body head supervising forward, lateral and yaw. Full breakdown, sizes and every hyperparameter in
[doc/MODEL_CONFIG.md](doc/MODEL_CONFIG.md).

## Documentation

Each document has one role and does not repeat another.

| | |
|---|---|
| [doc/direction_plan.md](doc/direction_plan.md) | **the plan as it stands today.** Edited in place, never stacked — read this first |
| [doc/FINDINGS.md](doc/FINDINGS.md) | every measurement, numbered `F1`…`F184`, with the trap each one avoids. Cited from everywhere else. **Not append-only** — a finding is corrected or withdrawn when a later one refutes it |
| [doc/OPEN_QUESTION.md](doc/OPEN_QUESTION.md) | only what is still undecided. A settled question moves to FINDINGS and leaves one line here |
| [doc/PROGRESS.md](doc/PROGRESS.md) | the dated engineering log, including what was tried and failed. Thai and English |
| [doc/SIM_GUIDE.md](doc/SIM_GUIDE.md) | how to actually run anything described above |
| [doc/MODEL_CONFIG.md](doc/MODEL_CONFIG.md) | one-page reference: architecture, loss terms and λ weights, hyperparameters, with `file:line` for each |
| [wm/README.md](wm/README.md) | the world model: lifecycle, modules, and how to read a training log |
| [scripts/README.md](scripts/README.md) | every diagnostic, the question it answers, the trap it avoids |
| [sim/README.md](sim/README.md) | building scenes, recording data, rendering |

**Papers this builds on** are in [doc/ref/](doc/ref/) — LAC-WM (the source architecture), V-JEPA 2,
DreamerV3, and the latent-action locomotion line. `F67` reads LAC-WM against our own design and
`F81` explains why its action projector is not optional.

## Layout

```
wm/        the world model      models/ data/ policy/ · train.py, losses.py, config.py
sim/       recording data       scene/ collect/ control/ render/ diagnostics/ env/ assets/
scripts/   measurement          diagnostics/ dataset/ figures/ finished/ run/ tools/ render/
data/      collected clips      frames + joint commands + contact + body pose
results/   figures and metrics
doc/       documentation and reference papers
```

`scripts/diagnostics/` is grouped by the question each script answers:

```
decoder/  latent/  forward_model/  setting/  shared_body_target/
cross_embodiment/  planning/  objective_experiments/  egocentric_view/
```

`scripts/run/` holds the shell run sheets (one per experiment); `scripts/tools/` holds simulator
calibration utilities that answer no research question.

## Running anything

```bash
.venv/bin/python3 -m wm.train --help
```

Always `.venv/bin/python3` from the repository root. Heavy training goes to a remote GPU box, not a
local card — see [doc/SIM_GUIDE.md](doc/SIM_GUIDE.md).

## A note on the record

`FINDINGS.md` contains several entries that report defects in this project's own pipeline — a
frame-rate mismatch between the two robots, a sign flip that had them turning opposite ways, a body
target measured in the world frame. They are kept because **each one invalidates numbers that were
previously believed**, and because the rule each establishes is more useful than the fix.
