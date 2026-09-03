# Provenance & empirical notes

**For how to run the sim, see [../SIM_GUIDE.md](../SIM_GUIDE.md).** This file
covers only where the files came from (attribution) and the empirical findings
worth keeping on record.

## File origins

Files below were copied from Ajan YuChen's `airl-insect-walking` repo
(local path: `../airl-insect-walking/`), not authored in this project.
Copied rather than symlinked so this project doesn't depend on that repo's
state changing underneath it.

| File here | Source | Notes |
|---|---|---|
| `env/medauroidea_stick_insect.ttt` | `airl-insect-walking/env/medauroidea_stick_insect.ttt` | Base CoppeliaSim stick insect model (*Medauroidea extradentata*). No camera/vision sensor — state-only in the original. Leg-length morphology variants (short/medium/long) don't exist yet and need to be created from this base. |
| `env/main_script.py` | `airl-insect-walking/env/main_script.py` | The scene's embedded control script (drives the default gait replay). Required — the `.ttt` fails to load without it. **Modified on copy**: two hardcoded absolute paths (`/home/yuchen/airl-insect-walking/...`) rewritten to point at `sim/env/` locally. |
| `env/ds_loopsm.csv` | `airl-insect-walking/env/ds_loopsm.csv` | Gait trajectory data `main_script.py` reads on init. Required alongside it. |
| `coppeliasim_env.py` *(removed)* | `airl-insect-walking/common/normalized_env.py` | `CoppeliaSimEnv` class — ZMQ Remote API connection, `reset()`/`step()`. Renamed on copy for clarity in this repo. Never imported by anything; removed in the 2026-09-03 scripts/sim cleanup. |
| `environment.yml` | `airl-insect-walking/environment.yml` | Known-working conda env for CoppeliaSim v4.10 + ZMQ, for reference. We did **not** install this as-is — it bundles an unrelated torch 2.7.1 that would've conflicted with the project's own torch in `.venv`. |

## Not migrated yet (available in `airl-insect-walking/` if needed later)

- Trained AIRL/PPO walking policies (`logs/Medauroidea*`) — could remove the need to write a data-collection controller from scratch.
- Leg-loss scene variants (`env_legloss/*.ttt`) — reference/template for how the lab structurally modifies the base `.ttt`.
- Gait analysis notebooks (`gait_analysis.ipynb`, `intra-limb analysis.ipynb`) — duty factor / phase / cyclogram tooling.
- Real animal motion-capture data (`expert/expert.csv`, `env/Animal06_110919_00_31.csv`) — biological reference, not a direct training input.

## Generated morphology variants

Created from the base model via `make_leg_morphology.py` (scales the local long
axis of each leg's coxa/femur/tibia segments and repositions every downstream
joint to match — see the script for the "segment origin is at its center, not one
end" gotcha that caused an early bug). Regenerate command is in SIM_GUIDE.md §5.

| File | Scale factor | Status |
|---|---|---|
| `env/medauroidea_stick_insect.ttt` | 1.0 (base, unmodified) | "long" leg per plan |
| `env/medauroidea_stick_insect_medium.ttt` | 0.75 | generated, numerically verified (all 6 legs → 0.75 reach ratio) |
| `env/medauroidea_stick_insect_short.ttt` | 0.5 | generated, numerically verified (all 6 legs → 0.5 reach ratio) |

(Originally generated at 0.7/0.85 — revised to 0.5/0.75 for a more visually
noticeable difference between morphologies.)

## Empirical findings on record

### Vision pipeline verified (2026-07-17)
- Void: **0.00%** of pixels near-black (under the then-current follow-cam config).
- Render lock: camera offset **byte-identical across all 3 variants**; frame brightness 128.3 / 129.0 / 129.4 (long/medium/short).
- Morphology gap reproduces in recorded video: **1.372 / 1.080 / 0.831 m** over 60 steps (long/medium/short) under identical commands — monotonic with leg length.

### Determinism: chaotic, not buggy (2026-07-17)

| Condition | Spread over 3 identical 200-step runs |
|---|---|
| Scene loaded **once**, stop/start only | **0.0000 m — bit-exact** |
| **Reload** scene each run | **1.84 m — diverges** |

Two reloaded runs differ by **4.4e-16 (machine epsilon) at step 0**, ~1e-3 by
step 10, ~1.8 m by step 200. Scene reload introduces a last-bit difference
(memory-layout-dependent contact ordering in Bullet), and contact-rich legged
dynamics amplify it exponentially. Real physics, not a bug. Engine: Bullet 2.78,
20 Hz timestep (0.005 s internal, 10 substeps), 100 solver iterations, realtime
off; `main_script.py` has all `noise` terms commented out.

Practical consequences are captured in SIM_GUIDE.md §6 (load once for
reproducibility; per-reload variation is free diversity; single-episode
measurements need error bars).

### Does the morphology gap survive the chaos? Yes

5 reloaded episodes × 200 steps each:

| Morphology | distance (mean ± std) | note |
|---|---|---|
| long 1.0× | **4.125 ± 0.434 m** | bimodal — lands on 4.479 *or* 3.593, two basins of attraction |
| medium 0.75× | **3.562 ± 0.015 m** | |
| short 0.5× | **2.646 ± 0.002 m** | |

`long_min (3.593) > short_max (2.648)` → **no overlap; the morphology gap is
robust.** Variance scales with leg length (σ: 0.434 / 0.015 / 0.002) — longer
legs = more leverage = more chaotic. Step -1 should be reported as mean ± std
over N episodes, not a single run.
