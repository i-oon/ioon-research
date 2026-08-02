# SIM_GUIDE — running the CoppeliaSim → V-JEPA2 pipeline

The one usage guide for the sim. File origins/attribution live in
[sim/SOURCES.md](sim/SOURCES.md); everything about *running* it is here.

- CoppeliaSim v4.10.0 Edu installed at `/home/aria/CoppeliaSim` (not in this repo)
- Python connector lives in the project venv `/home/aria/ioon-research/.venv`
- All sim scripts talk to CoppeliaSim over the ZMQ Remote API on `localhost:23000`

---

## 1. One-time install

CoppeliaSim itself is downloaded from `downloads.coppeliarobotics.com` to
`/home/aria/CoppeliaSim`. The Python-side connector packages go in the venv:

```bash
source /home/aria/ioon-research/.venv/bin/activate
pip install coppeliasim_zmqremoteapi_client pyzmq msgpack cbor2 pandas
```

---

## 2. Launch CoppeliaSim (do this first, every time)

**Always activate the venv *before* launching.** CoppeliaSim's ZMQ server spawns
its own `python3` using whatever is on `PATH`; if the venv isn't active it can't
find `zmq`/`cbor2` and the server silently never comes up (then every script
hangs at `client.require("sim")`).

```bash
source /home/aria/ioon-research/.venv/bin/activate
cd /home/aria/CoppeliaSim
./coppeliaSim.sh
```

That's it — **no scene needed.** The pipeline scripts each call `sim.loadScene()`
themselves over ZMQ, so whatever is open at launch gets replaced by the scene the
script wants. Only pass a scene if *you* want to eyeball one in the GUI:

```bash
./coppeliaSim.sh -f /home/aria/ioon-research/sim/env/medauroidea_stick_insect.ttt   # optional
```

Run the pipeline scripts from a **second terminal** (venv active there too).

**Use GUI mode, not headless** (`-h`/`-H`). Headless opens port 23000 briefly
then segfaults during Python-subprocess cleanup on this machine. GUI is stable.

### Verify the connection is alive

```bash
python3 -c "from coppeliasim_zmqremoteapi_client import RemoteAPIClient; \
c=RemoteAPIClient('localhost',port=23000); s=c.require('sim'); \
print('objects:', len(s.getObjectsInTree(s.handle_scene, s.handle_all)))"
```

Prints a number (~123 for the base scene) → good. Hangs → CoppeliaSim isn't up,
or the venv wasn't active when you launched it.

---

## 3. Vision pipeline (run in order, CoppeliaSim must be running)

```bash
# 1. floor: matte, mildly-textured, non-repeating (NOT cosmetic — see below)
python3 sim/set_floor_texture.py --all

# 2. camera: fixed world-frame telephoto side view, added programmatically
python3 sim/add_camera.py --all --preview        # --preview writes PNGs to /tmp

# 3. record an episode (per scene): frames.npy + actions.npy, exactly aligned
python3 sim/record_episode.py \
    --scene sim/env/medauroidea_stick_insect.ttt \
    --steps 300 --out data/episodes/long_walk_000

# 4. encode frames with frozen V-JEPA2  (needs working CUDA)
python3 scripts/step0_encode.py

# 5. analyse / probe / figures
python3 scripts/step0_analyze_v2.py
python3 scripts/plot_morphology_evidence.py
python3 scripts/plot_sanity_check.py
```

Run `set_floor_texture` / `add_camera` on **all three** scenes so every
morphology is byte-identical (that render-lock is what the morphology-vs-behaviour
comparison depends on):

| Scene file | Scale |
|---|---|
| `sim/env/medauroidea_stick_insect.ttt` | long 1.0× (base) |
| `sim/env/medauroidea_stick_insect_medium.ttt` | 0.75× |
| `sim/env/medauroidea_stick_insect_short.ttt` | 0.5× |

### Why the floor script is not cosmetic
A **checkerboard** floor aliases under sub-pixel motion (pixels change where
nothing moved; measured corr(pixel-motion, embedding-change) r = −0.16). A
**blank** floor is equally bad — ViTs repurpose featureless patches as scratch
space, so their embeddings fluctuate *more* than the robot (r = −0.20).
`set_floor_texture.py` targets the middle: matte, low-contrast, non-repeating,
fixed seed (identical across every variant).

---

## 4. Camera config (fixed world-frame, telephoto side)

Set in `sim/add_camera.py`. The camera is **bolted to the world** — it does not
follow the robot — so the robot visibly travels through a static frame. That
world-frame travel is exactly the outcome a joint encoder cannot report.

| Param | Value | Why |
|---|---|---|
| `DISTANCE` | 8.0 m | far → perspective compressed → apparent size ~constant across the run |
| `VIEW_ANGLE` | 18° | narrow "telephoto"; near-constant size without orthographic (which V-JEPA2 never saw) |
| `ELEVATION` | 40° | >30° keeps horizon/void out of frame |
| `AZIMUTH` | 90° | pure side view; +x travel reads left↔right |
| `RUNWAY_AIM` | 1.0 m | aim ahead of body start, so it enters near an edge and crosses centre |
| resolution | 256×256 | V-JEPA2 native input |

**Preview checklist** (from the `--preview` PNGs in `/tmp`, before recording):
1. no void (no black sky, no floor edge in frame)
2. robot ~35–45 px tall (legs resolvable vs the 16 px patch)
3. framing identical across the 3 bodies (only the robot differs)
4. ~2–2.5 m of runway visible

> **In progress — fixed-camera migration.** `add_camera.py` is updated.
> `record_episode.py` still has the old per-step camera-follow — remove that
> block (and add distance-gating) before recording the fixed-cam dataset.

---

## 5. Connect from Python (for custom collection / control)

```python
from sim.coppeliasim_env import CoppeliaSimEnv

env = CoppeliaSimEnv()      # connects on localhost:23000, homes the joints
obs = env.reset()
obs, reward, terminated, truncated, info = env.step(action)   # action: (18,) in [-1,1]
```

Action/observation bounds and the joint/leg naming convention are documented
inline in `sim/coppeliasim_env.py`.

### Regenerate a morphology variant
Scales each leg's coxa/femur/tibia and repositions downstream joints. **Requires a
running CoppeliaSim** (it edits the model live over ZMQ, then saves):

```bash
python3 sim/make_leg_morphology.py --factor 0.5 --out sim/env/medauroidea_stick_insect_short.ttt
```

---

## 6. Determinism & episode length (know this before collecting)

**The sim is chaotic, not buggy.** Contact-rich legged dynamics amplify last-bit
differences exponentially:

| Condition | Spread over identical runs |
|---|---|
| Scene loaded **once**, stop/start only | 0.0000 m — bit-exact |
| **Reload** scene each run | ~1.8 m — diverges |

Practical consequences:
- Need reproducibility? **Load the scene once, run many episodes from it** (don't reload).
- For data collection the per-reload variation is harmless — free diversity across episodes.
- **Any single-episode measurement needs error bars.** Report mean ± std over N episodes.

**Episode length vs floor size (open):** the floor is **10 m** across; the robot
walks ~0.46 m/s; timestep is 50 ms (20 Hz). ~300 steps (~15 s, ~7 m) keeps it on
the floor; ~1000 steps would walk ~23 m off the edge. If you need longer runs:
larger floor, or a re-centring reset. Engine, for the record: Bullet 2.78, 20 Hz,
10 substeps, 100 solver iterations, realtime off; the scene script has no
randomness.

---

## 7. Troubleshooting (the ones we actually hit)

**Script hangs at `client.require("sim")`** — CoppeliaSim isn't running, or the
ZMQ server didn't start. Launch it from an activated venv (§2), confirm with the
verify one-liner.

**Vision sensor renders pure black (`mean=0.00`) and/or the GUI shows garbage /
no scene** — GPU rendering is down, almost always an **NVIDIA driver/library
version mismatch** after a driver update without a reboot:

```bash
nvidia-smi     # "Failed to initialize NVML: Driver/library version mismatch" == this bug
```

Fix: `sudo reboot` (reloads the matching kernel module; also restores CUDA for the
encoder step). After reboot, `nvidia-smi` should print the GPU table.

**Remoting via AnyDesk?** Before rebooting, force the autologin session to **X11**
(AnyDesk is unreliable on Wayland → black screen):

```bash
sudo sed -i 's/#WaylandEnable=false/WaylandEnable=false/' /etc/gdm3/custom.conf
```

Autologin (`AutomaticLogin=aria`) and the `anydesk` service are already enabled,
so the desktop returns on its own after boot.

### Rendering gotchas already handled in the scripts (don't re-break)
- Vision sensors view along **+Z** (−Z gives an all-black frame, `min_depth=1.0`).
- `createVisionSensor` defaults to visibility layer 8 → must set `0xFFFF`, else it renders nothing silently.
- Elevation 30° + FOV 60° puts the horizon at the top edge → ~15% void; the current 40°/18° avoids it.
- `loadScene`/`saveScene` require **absolute** paths.
