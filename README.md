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

## The approach

The action latent is grounded in a **task-space target that every body shares: body motion in Froude
units**, forward and lateral speed divided by √(g·h) and yaw rate times √(h/g), with h the height of
the centre of mass. LAC-WM grounds its latent in end-effector and camera poses, which live in one
physical frame for every embodiment. An 18-DOF hexapod and a 12-DOF quadruped share no joint
correspondence, but Froude-scaled body motion means the same thing for both.

A frozen Froude head reads z alone, one head for all bodies. The only per-body part is the action
projector, which maps a body's own commands to z. To choose an action, the model either reads the
command's motion directly (projector → head) or imagines the future frame with the forward model and
reads the motion back out of it (rollout).

**Where it stands (2026-10).** Direct selection follows a goal on all three bodies. Rollout does not:
the model trained on one command per clip cannot read the future of a different action from the same
state. Round 1 trains on **counterfactual branches**, the same state under all 24 commands, built
identically for both bodies, and measures whether rollout recovers. Every number from before
2026-10-01 was measured on data later found faulty and is being re-measured (`doc/STATUS.md`).

## Bodies

| | role |
|---|---|
| hexapod c10f10t10 (stick-insect, 18 DOF) | pretraining |
| hexapod c08f09t09 (shorter legs) | test only, zero-shot |
| Unitree B1 quadruped (12 DOF) | joint pretraining, or adapted to a hexapod-only model |

## Architecture

```
frozen V-JEPA2  ─→  e_t
                     ├── ITM (e_t, e_t+5)  ──→  z         latent action, 5 frames (0.25 s) per step
                     ├── FTM (e_t, z)      ──→  ê_t+5     the world model
                     └── Froude head (z)   ──→  body motion (fwd, lat, yaw), shared by all bodies
action projector (per body): commands ──→ z
```

Losses: reconstruction, real-vs-null hinge, Froude head on z, frozen read-out, 2-step rollout
consistency. Side-by-side with LAC-WM and Egocentric VSM in [ARCHITECTURE.md](ARCHITECTURE.md);
every default in `wm/config.py`, every run's values in `wm/runs/<name>/config.yaml`.

## Documentation

| | |
|---|---|
| [doc/STATUS.md](doc/STATUS.md) | **start here**: what is done, what runs now, what is next, open decisions |
| [doc/DATA.md](doc/DATA.md) | which data is current, its splits, and the script that made it |
| [doc/DATA_PLAN.md](doc/DATA_PLAN.md) | how the counterfactual data was designed and checked |
| [ARCHITECTURE.md](ARCHITECTURE.md) | the model against LAC-WM and Egocentric VSM |
| [doc/FINDINGS.md](doc/FINDINGS.md) | every measurement, numbered, with the trap each one avoids. Corrected or withdrawn when refuted |
| [doc/PROGRESS.md](doc/PROGRESS.md) | dated engineering log, Thai and English |
| [doc/SIM_GUIDE.md](doc/SIM_GUIDE.md) | installing and running the simulators |
| [wm/README.md](wm/README.md), [scripts/README.md](scripts/README.md), [sim/README.md](sim/README.md) | per-directory guides |
| [doc/_archive/](doc/_archive/) | plans and guides for earlier pipelines (kept for the record, not current) |

**Papers this builds on** are in [doc/ref/](doc/ref/), with notes on LAC-WM and Egocentric VSM.

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

Always `.venv/bin/python3` from the repository root. Training: `scripts/run/round1_counterfactual.sh`;
every reported number: `scripts/run/eval_suite.sh`. Frames are read lazily from a memory-mapped cache
(`data/_frame_cache/`), so training needs a few GB of RAM, not the size of the data.

## A note on the record

`FINDINGS.md` contains several entries that report defects in this project's own pipeline — a
frame-rate mismatch between the two robots, a sign flip that had them turning opposite ways, a body
target measured in the world frame. They are kept because **each one invalidates numbers that were
previously believed**, and because the rule each establishes is more useful than the fix.
