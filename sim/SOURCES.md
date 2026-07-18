# Provenance

Files below were copied from Ajan YuChen's `airl-insect-walking` repo
(local path: `../airl-insect-walking/`), not authored in this project.
Copied rather than symlinked so this project doesn't depend on that repo's
state changing underneath it.

| File here | Source | Notes |
|---|---|---|
| `env/medauroidea_stick_insect.ttt` | `airl-insect-walking/env/medauroidea_stick_insect.ttt` | Base CoppeliaSim stick insect model (*Medauroidea extradentata*). No camera/vision sensor — state-only in the original. Leg-length morphology variants (short/medium/long) don't exist yet and need to be created from this base. |
| `env/main_script.py` | `airl-insect-walking/env/main_script.py` | The scene's embedded control script (drives the default gait replay). Required — the `.ttt` fails to load without it. **Modified on copy**: two hardcoded absolute paths (`/home/yuchen/airl-insect-walking/...`) rewritten to point at `sim/env/` locally. |
| `env/ds_loopsm.csv` | `airl-insect-walking/env/ds_loopsm.csv` | Gait trajectory data `main_script.py` reads on init. Required alongside it. |
| `coppeliasim_env.py` | `airl-insect-walking/common/normalized_env.py` | `CoppeliaSimEnv` class — ZMQ Remote API connection, `reset()`/`step()`. Renamed on copy for clarity in this repo. |
| `environment.yml` | `airl-insect-walking/environment.yml` | Known-working conda env for CoppeliaSim v4.10 + ZMQ, for reference. We did **not** install this as-is (see below) — it bundles an unrelated torch 2.7.1 that would've conflicted with the project's own torch in `.venv`. |

## How to Use This Sim

### Install (one-time)

CoppeliaSim itself (v4.10.0 Edu) is installed at `/home/aria/CoppeliaSim` —
downloaded from `downloads.coppeliarobotics.com`, not part of this repo.

Python-side connector packages live in the project's `.venv` (not the
`environment.yml` above):
```bash
source /home/aria/ioon-research/.venv/bin/activate
pip install coppeliasim_zmqremoteapi_client pyzmq msgpack cbor2 pandas
```

### Launch CoppeliaSim

**Always activate the venv first** — CoppeliaSim's ZMQ remote API server
internally spawns its own Python subprocess using whatever `python3` is on
`PATH`. If the venv isn't active, that subprocess can't find `zmq`/`cbor2`
and the remote API server silently fails to come up.

```bash
source /home/aria/ioon-research/.venv/bin/activate
cd /home/aria/CoppeliaSim
./coppeliaSim.sh -f /home/aria/ioon-research/sim/env/medauroidea_stick_insect.ttt
```

This opens the GUI window with the stick insect scene loaded, and starts
the ZMQ remote API server on port 23000.

**Use GUI mode, not headless (`-h`/`-H`), for now.** Headless mode does
start and briefly opens port 23000, but reliably segfaults a few seconds
in during a Python-subprocess cleanup step (`QProcess: Destroyed while
process is still running`) — reproduced with both `-h` and `-H` on this
machine. GUI mode is stable. This matters if you ever need to run
collection unattended/on a headless server — that crash needs solving
first (see the CoppeliaSim forums/GitHub for this subprocess-cleanup
pattern; not yet root-caused here).

### Connect from Python

With CoppeliaSim already running (see above), from the same venv:
```python
from sim.coppeliasim_env import CoppeliaSimEnv

env = CoppeliaSimEnv()       # connects on localhost:23000, homes the joints
obs = env.reset()
obs, reward, terminated, truncated, info = env.step(action)  # action: (18,) array in [-1, 1]
```
`action_space`/`observation_space` bounds and the joint/leg naming
convention are documented inline in `coppeliasim_env.py`.

### Verify the connection works standalone

```bash
python3 -c "
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
client = RemoteAPIClient('localhost', port=23000)
sim = client.require('sim')
print('objects in scene:', len(sim.getObjectsInTree(sim.handle_scene, sim.handle_all)))
"
```
Expect `123` objects for the unmodified base scene.

## Not migrated yet (available in `airl-insect-walking/` if needed later)

- Trained AIRL/PPO walking policies (`logs/Medauroidea*`) — could remove the need to write a data-collection controller from scratch, once we're past camera setup.
- Leg-loss scene variants (`env_legloss/*.ttt`) — reference/template for how the lab structurally modifies the base `.ttt`; useful when building the short/medium/long leg-length variants.
- Gait analysis notebooks (`gait_analysis.ipynb`, `intra-limb analysis.ipynb`) — duty factor / phase / cyclogram tooling, useful for Step -1 morphology gap validation.
- Real animal motion-capture data (`expert/expert.csv`, `env/Animal06_110919_00_31.csv`) — biological reference, not a direct training input for this project.

## Vision pipeline (added 2026-07-17) — the CoppeliaSim → V-JEPA2 link

The scene originally had **no camera at all** (state-only). These scripts add one and record
training data. Run them in this order; CoppeliaSim must be running (GUI mode) for all of them.

```bash
python sim/set_floor_texture.py --all     # 1. replace the checkerboard floor
python sim/add_camera.py --all            # 2. add the vision sensor
python sim/record_episode.py --scene sim/env/medauroidea_stick_insect.ttt \
    --steps 300 --out data/episodes/long_walk_000
```

| Script | What it does |
|---|---|
| `set_floor_texture.py` | Generates a matte, mildly-textured, non-repeating floor (fixed seed → identical everywhere) and applies it. **Not cosmetic** — the stock floor is a checkerboard, which aliases under sub-pixel motion (measured r=-0.16); a blank floor is equally bad (ViT register-token noise, r=-0.20). This targets the middle. |
| `add_camera.py` | Creates a 256×256 vision sensor programmatically, positioned by a fixed offset from the robot centre. **Scripted on purpose**: it makes the camera identical across all morphologies *by construction*, which is what the Step 1.5 render-lock requires. A hand-dragged viewport cannot guarantee that. |
| `record_episode.py` | Steps the sim; per tick renders the frame AND reads the 18 joint targets. Saves `frames.npy` (N,256,256,3) + `actions.npy` (N,18). Alignment is exact by construction — required for `L_motion`. |

**Camera config** (in `add_camera.py`, identical for every variant):
`DISTANCE=2.0m, ELEVATION=40°, AZIMUTH=90° (side), VIEW_ANGLE=45°, 256×256`.
Camera tracks the robot's x/y at a fixed offset; height and orientation stay fixed — so apparent size
stays constant (legs remain resolvable vs the 16px patch), body-bob survives as signal, and a turn reads
as a heading change in-frame rather than the world rotating.

### Gotchas found the hard way (do not re-discover these)
1. **Vision sensors view along +Z, not −Z.** Pointing −Z at the target gives an all-black frame with
   `min_depth = 1.0`. Verified empirically both ways.
2. **`createVisionSensor` defaults to visibility layer 8**, but the robot is on layer 1 and the floor on
   32768 → no overlap → renders nothing, silently. Must set layer to `0xFFFF`.
3. **Elevation 30° + FOV 60° put the horizon exactly at the frame's top edge**, leaving ~15% of every
   frame as pure black void — the blank-patch failure mode. `ELEVATION=40, VIEW_ANGLE=45` gives 0.00%
   void and a larger robot.
4. `loadScene`/`saveScene` require **absolute** paths.

### Verified
- Void: **0.00%** of pixels near-black.
- Render lock: camera offset `[0, 1.532, 1.286]` **byte-identical across all 3 variants**; frame
  brightness 128.3 / 129.0 / 129.4 (long/medium/short).
- Morphology gap reproduces in the recorded video: **1.372 / 1.080 / 0.831 m** over 60 steps
  (long/medium/short) under identical commands — monotonic with leg length, consistent with Step -1.

### Determinism: investigated and understood (2026-07-17)

**The sim is chaotic, not buggy.** Findings:

| Condition | Spread over 3 identical 200-step runs |
|---|---|
| Scene loaded **once**, stop/start only | **0.0000 m — bit-exact deterministic** |
| **Reload** scene each run | **1.84 m — diverges** |

Traced step-by-step: two reloaded runs differ by **4.4e-16 (machine epsilon) at step 0**, 1e-3 by step 10,
~1.8 m by step 200. So scene reload introduces a last-bit difference (almost certainly memory-layout-dependent
contact ordering in Bullet), and **contact-rich legged dynamics amplify it exponentially**. This is real
physics — legged locomotion with intermittent contact is chaotic. It is not fixable, only managed.

Engine config (for the record): **Bullet 2.78**, sim timestep **0.05 s (20 Hz)**, Bullet internal step 0.005 s
(10 substeps), 100 constraint-solver iterations, realtime off. The scene script contains **no randomness** —
every `noise` term in `main_script.py` is commented out.

**Practical consequences**
- Need bit-exact reproducibility? **Load the scene once and run many episodes from it** (don't reload).
- For data collection, the variation is harmless and arguably useful — free diversity across the ~100
  episodes per condition.
- **Closed-loop control damps chaos**; the current open-loop replay has nothing correcting heading, so it is
  maximally sensitive. Another reason for the planned yaw-feedback layer.
- ⚠️ **Any single-episode measurement needs error bars.** `step_minus1_morphology_gap.py` reloads per variant,
  so its headline "3.49 m vs 4.77 m" was one draw from a chaotic distribution.

**Does the morphology gap survive? YES** — 5 reloaded episodes x 200 steps each:

| Morphology | distance (mean ± std) | note |
|---|---|---|
| long 1.0× | **4.125 ± 0.434 m** | **bimodal** — lands on 4.479 *or* 3.593, two basins of attraction |
| medium 0.75× | **3.562 ± 0.015 m** | |
| short 0.5× | **2.646 ± 0.002 m** | |

`long_min (3.593) > short_max (2.648)` → **no overlap. Step -1's PASS is robust.**
Note variance scales with leg length (σ: 0.434 / 0.015 / 0.002) — longer legs = more leverage = more chaotic.
**Step -1 should be re-reported as mean ± std over N episodes**, not a single run.

### ⚠️ Open: episode length vs floor size
The floor is **10 m** across; the robot walks **~0.46 m/s**. `direction_plan.md` specifies "1000 steps
(~16s at 60Hz)" — but **CoppeliaSim's timestep is 50 ms (20 Hz)**, so 1000 steps is ~50 s ≈ **23 m**, and
the robot would walk off the floor. ~300 steps (~15 s, ~7 m) keeps it on-surface and still yields
300 pairs/episode × 100 episodes × 3 morphologies × 3 behaviors ≈ 270k pairs (target was ~200k).
**Needs a decision**: shorter episodes, a larger floor, or a re-centring reset.

## Generated Morphology Variants

Created from the base model via `make_leg_morphology.py` (scales the local
long axis of each leg's coxa/femur/tibia segments, and repositions every
downstream joint to match — see the script for the "segment origin is at
its center, not one end" gotcha that caused an early bug here).

| File | Scale factor | Status |
|---|---|---|
| `env/medauroidea_stick_insect.ttt` | 1.0 (base, unmodified) | = "long" leg per plan |
| `env/medauroidea_stick_insect_medium.ttt` | 0.75 | generated, numerically verified (all 6 legs scale to exactly 0.75 reach ratio), not yet visually confirmed |
| `env/medauroidea_stick_insect_short.ttt` | 0.5 | generated, numerically verified (all 6 legs scale to exactly 0.5 reach ratio), not yet visually confirmed |

(Originally generated at 0.7/0.85 — revised to 0.5/0.75 for a more visually
noticeable difference between morphologies.)

Regenerate with:
```bash
python sim/make_leg_morphology.py --factor 0.5 --out sim/env/medauroidea_stick_insect_short.ttt
```

**Note**: this script requires a live CoppeliaSim instance already running
and connected on port 23000 (it scales the model inside the running sim via
the ZMQ API, then saves the result) — it is not a standalone file
transformation. Launch CoppeliaSim first (see "Launch CoppeliaSim" above),
*then* run this script in a separate terminal.
