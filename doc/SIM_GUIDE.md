# SIM_GUIDE

Setup and usage guide for this project on a new machine.

Scope: how to install, run the simulator, collect data, and train/evaluate the world model.
Related documents:

| Document | Contents |
|---|---|
| [PROGRESS.md](PROGRESS.md) | Full chronological research log, findings, dated sections |
| [direction_plan.md](direction_plan.md) | Current research direction, roadmap, experiment design |
| [sim/SOURCES.md](sim/SOURCES.md) | File origins and attribution for migrated simulator assets |
| [OPEN_QUESTION.md](OPEN_QUESTION.md) | Unresolved questions |

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
sim = RemoteAPIClient('localhost', port=23000).require('sim')
print('connected, objects:', len(sim.getObjectsInTree(sim.handle_scene, sim.handle_all)))"
```

A number is printed on success. A hang means CoppeliaSim is not running or its ZMQ server did
not start.

---

## 3. Running CoppeliaSim

Headless, one instance per port:

```bash
cd $SIM && ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort=23000
```

No scene argument is needed. Every script calls `sim.loadScene()` itself over ZMQ. Pass
`-f <scene.ttt>` only to inspect a scene manually in the GUI (omit `-h` for that).

Run all other commands from a second terminal.

### Multiple instances

Each instance needs its own install directory and port, because instances share state through
their install path. Copy the directory to run several:

```bash
cp -r $SIM ${SIM}_b
cd ${SIM}_b && ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort=23001
```

Apply section 2.3 to each copy.

### Scenes

| Scene | Morphology |
|---|---|
| `sim/env/medauroidea_stick_insect.ttt` | long, 1.0x leg length (base) |
| `sim/env/medauroidea_stick_insect_medium.ttt` | medium, 0.75x |
| `sim/env/medauroidea_stick_insect_short.ttt` | short, 0.5x |

Camera, lighting and floor must remain identical across all three. That render lock is what
makes the morphology comparison valid; see PROGRESS.md section 15.2 for its current limits.

---

## 4. Data collection

### 4.1 Stick insect, IK-retargeted forward walk

The primary Stage 1 dataset. One expert foot trajectory is retargeted per morphology with
`simIK`, so behaviour is held fixed while joint commands differ per body.

```bash
cd $REPO && .venv/bin/python3 sim/collect/collect_ik.py \
    --port 23000 \
    --episodes 6,20,22,28 \
    --repeats 1 \
    --scale 0.5 \
    --travel 0.8 \
    --out data/ik_walk_100
```

- `--episodes` selects rows from the expert CSV (`sim/env/expert_66k_aug3c_fcontact.csv`,
  1000 episodes of 66 frames). All three morphologies are collected in one call.
- `--repeats 1` is intended. Repeats of one episode share a bit-identical action sequence and
  near-identical frames, so they add no information.
- Throughput is about 33 s per episode for all three bodies.

Output: one `.npz` per body per episode containing `frames (66,256,256,3)`, `actions (66,18)`,
`forces`, `head`, and metadata.

### 4.2 Unitree B1 quadruped

Rolled out in MuJoCo, then replayed kinematically in CoppeliaSim. **MuJoCo is never re-run to
change how a clip looks** -- every clip stores `base_pos`, `base_quat` and `joint_pos`, so the
physics is replayed from the file and only the camera changes.

```bash
# speed and sideways: re-render the stored states
cd $REPO && .venv/bin/python3 scripts/dataset/rerender_b1_framing.py --out data/allocentric/beh12_b1_flat

# turning: re-roll, because the commands themselves change
cd $REPO && .venv/bin/python3 scripts/dataset/recollect_b1_turns.py --out data/allocentric/beh12_b1_turns
```

The current set is **`data/allocentric/beh12_b1_flat`**; `data/allocentric/beh12_b1_flat` is the superseded one and the two
must not be mixed. Its README lists what changed.

#### The configuration that has to stay fixed

**Every item below was wrong at some point, and none of the errors announced itself.** They are
defaults now (§4.2.1); this table is here so a future change is a decision rather than a drift.

| | value | what goes wrong otherwise |
|---|---|---|
| perspective angle | **24 deg** | the scene ships 15, at which the B1 touches an image edge in 61% of frames and 100% of every sideways clip, while the insect never does. The two scenes ship identical cameras and that is **not** the same as an identical view: the field is 2.11 m at the robot, the insect needs 1.75 m and the B1 needs 2.85 m |
| spawn | **`--spawn 0 0`** | without it the camera is pinned to wherever that rollout happened to start, so every clip carries its own background -- 2.79 grey levels of spread against the insect's 0.00 |
| floor | **`--floor_scale 3`** | a 24-deg view reaches the far edge of the scene's 15 m floor and draws a band across the upper third of every frame. `sim.scaleObjects` grows a box **without moving its centre**, so the surface must be translated back to z=0 or the robot stands 20 cm underground with its feet cut off -- the renderer prints `surface +0.000 -> +0.000` to show it did |
| policies | **both**, two clips each | `base_gait3` at 2.0 Hz and `base_1.7hz_sym` at 1.7. Four clips of one policy are one limit cycle at four phases; the pair gives a condition genuine spread. `gait3`'s lateral drift and its lean into turns are the *controller*, not the robot |
| turn commands | **per policy** | the same `--wz` gives different rates on the two policies -- at the weakest level `sym` needs -0.023 and `gait3` -0.081. Conditions are named for the **achieved** rate, never for a command |
| turn direction | **negative**, matching the insect | the two robots turned opposite ways for six days after it was diagnosed, which made every cross-embodiment turning result meaningless |
| clip windows | **non-overlapping**, rotated to a common heading | MuJoCo starts every run identically and is deterministic, so four runs of one command are one clip four times; windows of a longer rollout are the only source of variety, and each is rotated about its own first pose so it starts facing where the first one did |

Camera and floor are **part of the data**, not of the viewing. A loop that plans on frames
differing from its adaptation set in any static way measures that difference:
`close_loop_b1_physics.py` carries the same three defaults and they have to agree.

#### 4.2.1 Checking a collected set

Numbers, not eyes -- but **look at the video too**: five defects this project found in its B1 data
were caught by watching and none by the tables that were passing at the time.

| check | pass | how it failed before |
|---|---|---|
| frames touching an image edge | 0% on all 48 | 61%, and the metric that catches a *static* band is `worst background edge`, not this one |
| background spread between clips | < 0.5 grey levels | 2.79 |
| worst background edge | < 5 | 21.3 with the floor edge in shot |
| feet visible | bright pixels > 0 per frame | 0, with the robot buried under a lifted floor |
| turn sign | negative on all four levels | positive, i.e. opposite the insect |
| weakest turn | distinct from `speed_vx0.30` | identical in both channels |
| conditions x clips | 12 x 4, two per policy | -- |

Output per clip: `frames (66,256,256,3)`, `action (66,12)`, `foot_contact`, `base_pos`,
`base_quat`, `joint_pos`, `condition`, `behaviour`, `level`, `expert_episode`, **`policy`**.

### 4.3 Generating a new morphology

Scales each leg segment and repositions downstream joints. Requires a running CoppeliaSim,
since it edits the model over ZMQ and then saves.

```bash
cd $REPO && .venv/bin/python3 sim/scene/make_leg_morphology.py \
    --factor 0.625 --out sim/env/medauroidea_stick_insect_0625.ttt
```

---

## 5. Data checks

Run before trusting any dataset.

```bash
cd $REPO && .venv/bin/python3 scripts/dataset/render_lock_check.py \
    --data data/ik_walk_100 --out results/ik/render_lock
```

Reports whether morphology is present in the frozen encoder's embeddings (expected), and
whether repeated recordings of the same clip are distinguishable (should be near chance).
Read PROGRESS.md section 15.2 before interpreting the repeat test.

```bash
cd $REPO && .venv/bin/python3 scripts/dataset/audit_ik_dataset.py --data data/ik_walk_100
```

---

## 6. World model

Implementation lives in `wm/`. Architecture follows LAC-WM; optimisation is scaled down for a
single GPU. See `wm/config.py` for all hyperparameters.

| Module | Role |
|---|---|
| `wm/models/itm.py` | Inverse Transition Model: two frames to latent action `z` |
| `wm/models/ftm.py` | Forward Transition Model: predicts the next visual embedding |
| `wm/models/motion_decoder.py` | Decodes `z` to joint commands; shared backbone, per-embodiment head |
| `wm/data/` | Datasets, augmentation, embodiment adapters |
| `wm/train.py` | Training entry point |
| `wm/evaluate.py` | Latent validation and transfer evaluation |

### 6.1 Train

Run from the repository root so `-m wm.train` resolves.

```bash
cd $REPO && .venv/bin/python3 -m wm.train \
    --data_dir data/ik_walk_100 \
    --epochs 20 \
    --batch_size 8 \
    --val_episodes 5 \
    --train_morphs long short \
    --heldout_morph medium \
    --name run_name
```

Key options:

| Option | Meaning |
|---|---|
| `--train_morphs` | Bodies used for gradient updates |
| `--heldout_morph` | Body never trained on; reported each epoch as `heldout/motion` |
| `--val_episodes` | Episodes held out from the training bodies, used to select `best.pt` |
| `--frame_start`, `--frame_stop` | Restrict the frame range within each clip |
| `--lambda_recon`, `--lambda_motion` | Loss weights; the source paper reports no values |
| `--z_dim` | Latent action dimensionality, default 64 |

Splitting is by expert episode, never by repeat. Repeats of one episode share identical
actions, so holding one out measures nothing.

Outputs in `wm/runs/<name>/`: `best.pt`, `best_motion.pt`, periodic `epoch###.pt`, and
TensorBoard logs under `summary/`.

Approximate cost: 1543 steps per epoch on 100 episodes and two bodies, about 23 minutes per
epoch on an RTX 2080 Ti, 4.8 GB of GPU memory at batch size 8.

### 6.2 Monitor

```bash
python3 -m tensorboard.main --logdir $REPO/wm/runs --port 6006
```

`val/*` measures unseen episodes of the training bodies. `heldout/*` measures the held-out
body. These can diverge: a run reached `val/motion` 0.0016 while the held-out body degraded to
0.42. Treat `val/*` alone as insufficient evidence of generalisation.

`heldout/*` is for reporting only. Selecting checkpoints on it would leak the test body.

### 6.3 Evaluate

```bash
cd $REPO && .venv/bin/python3 -m wm.evaluate \
    --ckpt wm/runs/run_name/best.pt --out results/wm/run_name
```

Writes `evaluation.json` with:

- `motion_mse` per body, including `zero_z` and `shuffled_z` ablations and a `predict_mean`
  reference. Actions are standardised, so 1.0 corresponds to no skill.
- `morphology_structure`: decode accuracy, silhouette, and between-class variance for raw
  embeddings and for `z`. Report all three; decode measures whether a signal is present,
  silhouette whether it dominates, and they can disagree.
- `behaviour_transfer_macro_f1`: foot-contact transfer across bodies, from raw embeddings and
  from `z`.

Contact labels are used only for evaluation, never for training.

---

## 7. Rendering and gait diagnostics

Use a separate CoppeliaSim instance so training instances are not disturbed.

```bash
cd $REPO && .venv/bin/python3 scripts/amp/render_rollout.py \
    --port 23063 \
    --scene sim/env/medauroidea_stick_insect.ttt \
    --ckpt amp/logs/<run>/model/step<N> \
    --out results/rollouts/name.mp4
```

```bash
cd $REPO && .venv/bin/python3 scripts/amp/gait_report.py \
    --port 23063 \
    --scene sim/env/medauroidea_stick_insect.ttt \
    --ckpt amp/logs/<run>/model/step<N> \
    --tag name
```

`gait_report.py` compares duty factor, inter-leg phase and stride rate against the expert
without assuming a gait template. Aggregate returns and video alone are not sufficient to
judge gait quality.

---

## 8. Determinism and episode length

Contact-rich legged dynamics amplify floating-point differences.

| Condition | Spread across identical runs |
|---|---|
| Scene loaded once, stop and start only | 0.0000 m |
| Scene reloaded per run | about 1.8 m |

Consequences:

- For reproducibility, load the scene once and run many episodes from it.
- Any single-episode measurement needs error bars. Report mean and standard deviation over
  several episodes.
- The floor is 10 m across, the robot walks about 0.46 m/s, and the timestep is 50 ms (20 Hz).
  About 300 steps stays on the floor; 1000 steps walks off the edge.

Engine settings: Bullet 2.78, 20 Hz, 10 substeps, 100 solver iterations, realtime off.

**The two robots differ here and it decides where spread comes from.** The B1's physics is MuJoCo,
seeded explicitly, and **repeats bit for bit** -- choices, frames and body track. Rerunning one B1
configuration returns the identical number and carries no information; its spread has to come from
**different goal clips**. The insect runs in CoppeliaSim with the scene reloaded per run and
spreads 37-71% on one configuration, so its spread comes from **repeats**. Applying either recipe
to the other robot produces error bars that mean nothing.

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
| `sim/env/expert_66k_aug3c_fcontact.csv` | 132 MB | all IK collection; `sim/collect/collect_ik.py` reads it directly |
| `data/` | varies | training and evaluation. `data/ik_walk_100_framed` is 379 MB |
| `wm/runs/` | 366 MB per checkpoint | only if continuing from existing checkpoints |

Copy them directly, for example:

```bash
rsync -av --progress \
    aria@source-host:/home/aria/ioon-research/sim/env/expert_66k_aug3c_fcontact.csv \
    aria@source-host:/home/aria/ioon-research/data/ik_walk_100_framed \
    aria@source-host:/home/aria/ioon-research/data/b1 \
    aria@source-host:/home/aria/ioon-research/data/b1_traj \
    $REPO/
```

Datasets can also be regenerated from the expert CSV with section 4, which is slower (about
55 minutes for 100 episodes) but needs only the CSV.

### In git and portable

Scene files (`sim/env/*.ttt`), the floor texture, and all code. Paths in the active scripts are
derived from the file's own location, so the repository can live anywhere.

### Not portable

Some scripts still hold absolute paths to assets outside this repository:

- `sim/collect/rollout_b1_mujoco.py` and `sim/scene/build_b1_scene.py` point at `~/Sim2Real-B1` and
  `sim/assets/b1_description`. Only needed to regenerate B1 trajectories or rebuild the B1 scene;
  the existing `sim/env/b1_flat.ttt` and `data/b1_traj` avoid this.
- Anything under `sim/_archive/` or `scripts/_archive/`.

### Checklist

```bash
cd $REPO
python3 -c "import torch, transformers; print(torch.cuda.is_available())"   # expect True
ls sim/env/expert_66k_aug3c_fcontact.csv          # expect it to exist
ls data/ik_walk_100_framed | wc -l                 # expect 301 (300 clips + manifest)
```

Then start CoppeliaSim (section 3) and run the verification in section 2.4.
