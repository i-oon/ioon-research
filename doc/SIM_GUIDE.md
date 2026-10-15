# SIM_GUIDE

Setup and usage guide for this project on a new machine.

Scope: how to install, run the simulator, collect data, and train/evaluate the world model.
Related documents:

| Document | Contents |
|---|---|
| [START_HERE.md](START_HERE.md) | Where to begin, current state |
| [FINDINGS.md](FINDINGS.md) | Numbered evidence |
| [PROGRESS.md](PROGRESS.md) | Chronological log (history) |
| [STATUS.md](STATUS.md) | Current state, what runs now, what is next |
| [DATA.md](DATA.md) | Current data, splits, and the scripts that made them |
| [sim/SOURCES.md](sim/SOURCES.md) | File origins and attribution for migrated simulator assets |

Throughout this guide `$REPO` means the repository root and `$SIM` means the CoppeliaSim
install directory. Set them once per shell:

```bash
export REPO=$HOME/ioon-research
export SIM=$HOME/CoppeliaSim
```

---

## 1. Requirements

- Linux with an NVIDIA GPU (CUDA). Training was developed on an RTX 2080 Ti (11 GB); 8 GB is
  the practical minimum at the default batch size.
- Python 3.10
- CoppeliaSim 4.10.0 Edu

---

## 2. Install

### 2.1 CoppeliaSim

Download CoppeliaSim 4.10.0 Edu for Linux from `downloads.coppeliarobotics.com` and extract
it to `$SIM`. It is not part of this repository.

### 2.2 Python environment

```bash
cd $REPO
python3 -m venv .venv --system-site-packages
.venv/bin/pip install torch transformers coppeliasim_zmqremoteapi_client pyzmq msgpack cbor2 \
    pandas numpy opencv-python imageio imageio-ffmpeg scikit-learn umap-learn tensorboard
```

Reference versions known to work: torch 2.13.0+cu130, transformers 5.13.1, scikit-learn 1.7.2,
numpy 2.2.6.

Every command in this guide uses `.venv/bin/python3` explicitly. Activating the environment
(`source .venv/bin/activate`) also works, in which case `python3` is enough.

### 2.3 Point CoppeliaSim at the virtual environment

Required. CoppeliaSim runs scene scripts in a Python subprocess. If that subprocess is the
system interpreter it cannot import `zmq`/`cbor2`, the scene script raises, CoppeliaSim pauses
the simulation on script error, and training then runs against frozen physics without any
visible failure.

Edit `~/.CoppeliaSim/usrset.txt`:

```
defaultPython = /absolute/path/to/ioon-research/.venv/bin/python3
```

### 2.4 Verify

Start CoppeliaSim (section 3), then:

```bash
cd $REPO && .venv/bin/python3 -c "
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
sim = RemoteAPIClient('localhost', port=25780).require('sim')
print('connected, objects:', len(sim.getObjectsInTree(sim.handle_scene, sim.handle_all)))"
```

A number is printed on success. A hang means CoppeliaSim is not running or its ZMQ server did
not start.

---

## 3. Running CoppeliaSim

Headless, one instance per port. Use ports in the 25xxx range; **never 23000** (reserved), at most 6 own instances:

```bash
cd $SIM && setsid bash -c "tail -f /dev/null | ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort=25780 -GzmqRemoteApi.cntPort=25781" &
```

Several instances run from the one install on different ports (the queue scripts in `results/wm/logs/queue_*.sh` start
them this way, 8 s apart, and wait until each answers `getSimulationState`). No scene argument: every script calls
`sim.loadScene()` itself. After killing a script mid-run, restart its instances (otherwise "simulation is not stopped").

### Scenes in use

| scene | body |
|---|---|
| `sim/env/medauroidea_c10f10t10.ttt` | hexapod c10 (pretraining) |
| `sim/env/medauroidea_c08f09t09.ttt` | hexapod c08 (test only) |
| `sim/env/b1_flat.ttt` | B1 rendering (physics in MuJoCo) |

Other `medauroidea_c*f*t*.ttt` files are leg-length variants not used by the current data.

---

## 4. Data collection (current pipeline, 2026-10-15)

Rewritten 2026-10-15; the earlier IK-retarget / expert-CSV / allocentric / AMP instructions are history (git, PROGRESS).
Never use the expert CSV (memory). Everything below writes 0.05 s frames (20 Hz) for every body; data map in `doc/DATA.md`,
design in `doc/DATA_PLAN.md`. Own CoppeliaSim instances on ports 25xxx (never 23000), at most 6.

### 4.1 Locomotion clips and branches (`data/counterfactual_walks/`)

| body | script | physics |
|---|---|---|
| hexapod c10 (pretraining) | `scripts/dataset/collect_c10_walks_and_branches.py` | CoppeliaSim, CPG gait, scene reuse (§8) |
| B1 | `scripts/dataset/collect_b1_walks.py`, then `scripts/dataset/build_branches.py` | MuJoCo (walking policy), CoppeliaSim render |
| hexapod c08 (test only) | `scripts/dataset/collect_c08_test_set.py` | CoppeliaSim, the c10 walk plans |

24 behaviours per body; main clips = windows of one long walk per behaviour; branches = 3 branch points per main clip x 24
commands from the identical state (10 frames before the switch, 20 after). Labels: Froude (forward / lateral / yaw) at the
centre of mass, 1 s average, never across a command switch (`wm/data/embodiment.py`).

### 4.2 Wall data (`data/wall_b1/`, `data/wall_c10/`)

| step | script |
|---|---|
| B1 episodes + branches (physics, MuJoCo) / render | `scripts/dataset/collect_b1_wall_data.py physics|render ...` (`--group`, `--side`, `--corner`) |
| c10 live episodes + branches (CoppeliaSim) / render | `scripts/dataset/collect_c10_wall_data.py physics|render ...` (`--group`, `--side`, `--n_corner`) |
| equal cells across bodies | `scripts/dataset/balance_wall_data.py --body {b1,c10} [--apply]` |
| train subset (14 of 28 commands per state) | `collect_*_wall_data.py select --split train` |
| gates (must all pass) | `scripts/dataset/check_wall_data.py --body {b1,c10} --render [--ref_body b1]` |

Episode = approach a wall -> back up / step aside -> spin -> walk out; branch = every one of the 28 commands held from a
saved state (3 s, 5 s from far states), cut at the boundary crossing. Per-frame `clearance` / `status`; design in
`doc/DATA_PLAN.md` section 13, findings F332-F334, F338, F346.

---

## 5. Data checks

Run after any data change; every reported number assumes they pass.

| check | command |
|---|---|
| branch exactness, c10 / B1 / c08 | `scripts/dataset/check_branches.py` (`--c08`) |
| splits disjoint (files, frames, rooms) | `scripts/diagnostics/dataset/check_splits.py` |
| Froude labels | `tests/test_froude_labels.py` (`FROUDE_TEST_HEX=c08` for c08) |
| frames vs labels aligned in time | `scripts/diagnostics/dataset/check_frame_label_alignment.py` |
| wall data | `scripts/dataset/check_wall_data.py` (above) |

Look at the frames too (memory: verify rendered frames): sample sheets / videos before collecting at scale.

---

## 6. World model

`wm/` -- frozen V-JEPA2 encoder (grayscale for the current models), inverse transition model (ITM, two embeddings -> latent
action z), forward transition model (FTM, embedding + z -> next embedding), Froude prediction head (z -> forward / lateral /
yaw), one action projector per body (command chunk -> z). `ARCHITECTURE.md` explains the design.

### 6.1 Train

```bash
# arm B = joint c10 + B1, clips + branches; the current recipe adds grayscale, augmentation and the imagined-read term
RUN_TAG=_gray_imread EXTRA="<recipe flags, see scripts/run/hexonly_center_pair.sh>" bash scripts/run/round1_counterfactual.sh B
# H = hexapod only (B1 adapted later), W = arm B + the wall branches
```

30.5k gradient steps per run (batch 8, balanced bodies); `--batches_per_epoch` keeps that budget when data is added.
Outputs `wm/runs/<name>/` (`best.pt`, `summary/` tensorboard: per epoch `train/`, `val/`; every 500 steps `train_step/`,
every 2,000 steps the full validation `val_step/`). A temperature guard pauses training above 83 C.

### 6.2 Evaluate

```bash
bash scripts/run/eval_suite.sh NAME joint   wm/runs/NAME/best.pt    # B1 in pretraining
bash scripts/run/eval_suite.sh NAME hexonly wm/runs/NAME/best.pt    # B1 adapted (ADAPT_EXTRA="--lambda_imread 1.0")
```

Every reported number comes from here: projectors, selection (B1 recorded / video goal, c08 zero-shot, c10), shared latent,
z-dependence, counterfactual read-outs -> `results/eval/NAME/summary.txt`. Physics closed loop:
`scripts/run/physics_orig_b1.sh`, `scripts/run/physics_orig_hex.sh` (env `CK`, `OUT`, `MECHS`, `GA=imagined`,
`GS=vision`). Wall prediction check: `scripts/diagnostics/wall_pilot/wall_predcheck.py --body {b1,c10} --ckpt NAME`.

---

## 7. Rendering

Replays render the recorded link / body poses in a separate CoppeliaSim instance: `sim/render/render_hex_replay.py`,
`sim/render/render_b1_replay.py`; deck figures and videos in `scripts/figures/` (`weekly_visuals.py` for the physics
clips). Same render path as the data, so a figure shows exactly what the model saw.

---

## 8. Determinism

| | behaviour | how the data uses it |
|---|---|---|
| B1 (MuJoCo) | exact save / restore; a rerun repeats bit for bit | branches = restore the state, run each command |
| hexapod (CoppeliaSim) | reruns after `loadScene` are NOT identical; with the scene loaded once, runs fall into bit-identical **classes** that switch with the run count (F305) | `sim/collect/scene_reuse.py`: load once, warm up, match each branch to the class of the walk it continues; wall data: store an episode only if its replay equals the live run (F346) |

Consequences: hexapod collection replays the episode for every branch (15-30 min per wall episode vs minutes for B1); a
CoppeliaSim instance can hang (seen 2026-10-14): long jobs need a timeout and a retry with fresh instances. Spread in a
result has to come from different start states / goals for B1 and from different episodes for the hexapod.

---

## 9. Troubleshooting

**A script hangs at `client.require("sim")`.** CoppeliaSim is not running, or its ZMQ server
did not start. Check section 2.3 and the verification in section 2.4.

**Simulation is paused and training produces frozen or nonsense values.** A scene script
raised and CoppeliaSim paused on script error. Almost always the `defaultPython` setting in
section 2.3.

**A vision sensor renders black, or the GUI shows no scene.** GPU rendering is unavailable,
usually an NVIDIA driver and library version mismatch after an update without a reboot.
`nvidia-smi` reporting "Failed to initialize NVML" confirms it. Reboot.

**CUDA out of memory during evaluation while training runs.** Both processes load their own
copy of the frozen encoder, about 2 GB each. Run evaluation after training, or on another GPU.

**`ModuleNotFoundError: transformers`.** The system interpreter is being used instead of the
virtual environment. Use `.venv/bin/python3`.

**`git push` rejected for file size.** Model checkpoints exceed GitHub's 100 MB limit. They
are covered by `.gitignore`, but files committed before a rule was added remain tracked:
`git rm --cached <file>` then commit.

### Constraints already handled in the scripts

- Vision sensors view along +Z. Along -Z the frame is entirely black.
- `createVisionSensor` defaults to visibility layer 8 and must be set to `0xFFFF`, otherwise
  it renders nothing without reporting an error.
- `loadScene` and `saveScene` require absolute paths.
- The floor texture must be matte, low contrast and non-repeating. A checkerboard aliases
  under sub-pixel motion; a blank floor causes ViT embeddings to fluctuate more than the robot
  does. See PROGRESS.md section 4.
---

## 10. Moving to another machine

Cloning the repository is not sufficient. Three categories of file are deliberately not in git
because of GitHub's 100 MB limit, and must be copied across manually.

### Required, not in git

| Path | Size | Needed for |
|---|---|---|
| `data/counterfactual_walks/` | ~23 GB | locomotion training / evaluation |
| `data/wall_b1/`, `data/wall_c10/` | ~14 / ~20 GB | wall training / evaluation (incl. `_physics/`) |
| `wm/runs/<name>/best.pt` | ~400 MB each | continuing from or evaluating existing checkpoints |
| `results/wm/cache/` | varies | evaluation embedding caches (rebuilt if missing) |

Copy with `rsync -aL` (follows the symlinked dirs, e.g. the wall `train_branches` folders). The expert CSV is NOT used.

### In git and portable

Scene files (`sim/env/*.ttt`), the floor texture, and all code. Paths in the active scripts are
derived from the file's own location, so the repository can live anywhere.

### Not portable

Some scripts still hold absolute paths to assets outside this repository:

- `sim/collect/rollout_b1_mujoco.py` and `sim/scene/build_b1_scene.py` point at `~/Sim2Real-B1` and
  `sim/assets/b1_description` (the B1 walking policy and model): needed for any new B1 physics (walks, branches, wall
  data, the B1 physics loop).
- Anything under `sim/_archive/` or `scripts/_archive/`.

### Checklist

```bash
cd $REPO
python3 -c "import torch, transformers; print(torch.cuda.is_available())"   # expect True
ls data/counterfactual_walks/c10_clips_train | wc -l   # expect 48
.venv/bin/python3 tests/test_froude_labels.py          # expect 0 failed
```

Then start CoppeliaSim (section 3) and run the verification in section 2.4.
