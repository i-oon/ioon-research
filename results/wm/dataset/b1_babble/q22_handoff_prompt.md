# B1 in CoppeliaSim — current handoff (2026-09-12)

## Scope

This is no longer a Q22 test. The work only establishes a usable B1 model under native
CoppeliaSim–Bullet dynamics and begins controller/babble development for a future clean F194
2x2x2 rerun.

**Important:** no clean/expert controller exists yet. The present CPG is experimental and is not
yet an approved babble generator or dataset.

## What is fixed

- Use `sim/env/b1_flat_convex.ttt`, not `b1_flat.ttt`.
  - Original B1: 13 non-convex respondable shapes; Bullet hangs/becomes unusable.
  - Convex B1: 0 non-convex respondable shapes; engine remains Bullet (`0`).
  - No additional `.ttt` is needed for expert versus babble; both must use this same scene.
- The standing collapse was an initialization bug. Motor targets were set, but the 12 actual
  joints remained at zero. Set both `sim.setJointPosition(DEFAULT_IL)` and the target before
  starting dynamics.
- Stable stand verified for 100 steps: final height `0.5358 m`, minimum `up.z=0.9998`.
- Track `trunk_respondable` for physical state. `base_visual` is not the dynamics root.
- Coppelia PID values are controller coefficients, not MuJoCo `kp/kv`. Current stable values are
  `P=300`, `D=5`, with segment-specific force limits `91/93/140 Nm`.
- Scene audit confirms B1 original/convex cameras are identical. Insect and B1 share camera
  offset, orientation, 15-degree authored FOV, and 256x256 resolution; the far plane differs
  (insect 20 m, B1 30 m).
- Dataset-compatible runtime rendering is implemented: B1 FOV 24 degrees, floor scale 3 with
  surface height preserved, plus the existing paired egocentric room/texture convention.

## Native CPG status

Controller: `scripts/diagnostics/objective_experiments/b1_coppelia_cpg_controller.py`

Reproduction wrapper: `scripts/run/b1_coppelia_three_cpg.sh`

The controller records real live-physics frames and telemetry: action, joint state, base pose,
quaternion, contacts, body motion, stability, and optional allocentric/egocentric MP4.

Current experimental presets all survived 160 steps / 8 s:

| preset | result | mean body-frame Froude | status |
|---|---|---|---|
| forward | `+0.172 to +0.208 m` | best `[+0.0101,+0.0025,-0.0006]` | stable, slow trot |
| lateral | `+0.027 m` lateral | `[+0.0031,+0.0035,-0.0006]` | barely lateral-dominant |
| yaw | `+9.3 deg` | `[+0.0040,-0.0038,+0.0052]` | yaw-dominant but coupled |

Forward defaults to thigh amplitude `0.23`, calf amplitude `0.25`, frequency `0.5 Hz`.
Amplitudes `0.30–0.40` rolled over even with stronger PID gains. Do not increase amplitude again
without adding feedback/IK; the open-loop stability boundary has been measured.

Lateral required left-pair/right-pair phasing and mirrored left/right hip signs. Yaw uses the
measured same-sign hip-wave convention. These are family-designed motion primitives, not
undirected motor babble.

## What is not done

- **No Coppelia-native clean/expert policy or controller.**
- **No valid Coppelia babble dataset.** Current presets are only feasibility probes.
- No robust behavior bank across both signs, magnitudes, and repeated seeds.
- Lateral/yaw separation is weak; predicted-Froude margin has not been checked.
- No environment-grade reset, command, observation, termination, or batch collection interface.
- No new world-model fitting and no 2x2x2 rerun on Coppelia-native data.
- Do not transplant `model_600.pt`; it was trained for MuJoCo and already failed sim-to-sim.
- Do not train RL with the present WM reward; F195 remains below chance locally.
- `b1_flat_convex.ttt` and the new controller scripts are currently untracked. Make scene creation
  reproducible or version the artifact before the final experiment.

## Collector audit (2026-09-12)

`sim/collect/collect_b1_coppelia_babble.py` correctly uses live Bullet, the convex scene, physical
root, joint initialization, real frames, actions, contacts, and body-frame Froude. It now also:

- supports the exact egocentric camera/room convention;
- writes a YAML record for every attempt, including falls, so survival rate remains measurable;
- identifies itself as designed CPG primitives, not undirected/no-prior-knowledge babble.

It is **pilot infrastructure, not an approved collector**. Existing live-Bullet forward runs reach
only Froude `0.0080–0.0094` (best earlier run `0.0101`), while useful positive hexapod goals reach
`0.12–0.19`. This gap motivates deliberate over-collection with a broader, harder excitation
envelope: tolerate and record falls, repeat settings, and retain the rare stable high-motion tail.
Do not confuse this target-aware search with duplicating the same weak settings.

Aggressive pilot completed 2026-09-12: 50 configurations x 3 repetitions, 150 full 160-step
attempts (`coppelia_aggressive_pilot/{config.yaml,attempts.csv,ranking.csv}`). Overall survival was
17/150. Forward: 10/60 survived, 8/60 remained forward-dominant, strongest stable forward Froude
`0.0129`. Lateral: 7/54 survived, but 0/54 were lateral-dominant; target-scale lateral values only
appeared during falls. Yaw: 0/36 survived. Thus brute over-collection with this open-loop waveform
did not discover the required high-motion stable tail. The best forward config was rerun with the
real egocentric setup; two attempts fell and the third survived at forward Froude `0.0100`, saved
as `coppelia_aggressive_pilot/rendered/best_forward_try3.{npz,mp4,yaml}`. This pilot is screening
evidence only and must not enter training.

## Generic motor-babble contract and pilots (2026-09-12)

The valid reference is Egocentric VSM: their collector uses a **structured sinusoidal gait seed**
plus per-step Gaussian action noise. It is not independent random motor targets. We may use the
same protocol, but not copy their Atlas-specific offsets/amplitudes or their stored gait vector.

The claim-honest B1 collector is therefore a **predeclared generic quadruped CPG family + sampled
CPG parameters + per-step motor noise**. Its fixed leg phases and hip/thigh/calf roles are a
generic locomotion prior; it has no requested forward/lateral/yaw command, body-state feedback,
demonstration, retargeting, or B1 outcome-dependent adjustment. Precommit the parameter
distribution before collection. Retain every rollout, including falls. Log Froude only for
post-collection reporting; never use it to select clips or alter a rollout.

Implementation: `sim/collect/collect_b1_coppelia_generic_babble.py` (shared-frequency sine) and
`sim/collect/collect_b1_coppelia_stance_swing_babble.py` (generic diagonal stance/swing prior),
Bullet only. The initial independent-frequency pilot is invalid and labelled as such. The corrected
shared-frequency random-phase pilot (`coppelia_generic_pilot_v2/`) retained all 4 runs: 2 upright,
2 falls, with weak uncommanded motion. Equal-amplitude diagonal sine (`coppelia_generic_trot_pilot/`)
fell 4/4 because it excited hip ab/adduction too strongly. A generic quadruped joint-role prior
(`coppelia_quadruped_trot_pilot/`: hip `0.1x`, thigh/calf `1x`) retained all four runs; one stood
but shuffled backwards.

The original generic stance/swing diagnostic retained every result. The controlled 18-cell
lift/frequency grid had 12 upright runs; its best cell (`1.75 Hz`, amplitude `0.18`, calf ratio
`2.5`) logged Froude `[+0.0243,+0.0011,-0.0085]`. Foot telemetry explained the weakness: FL/FR
height varied only ~`0.4–2.1 mm`, versus rear `38–44 mm`; front-calf amplitude alone has little
vertical leverage at this pose.

**Current working seed:** `collect_b1_coppelia_generic_duty_cycle_babble.py` fixes a generic
diagonal duty-cycle CPG (65% planted stance, 35% raised return) with the same joint role/phase
prior and per-step motor noise. It replaced the weak sinusoidal stance movement; it is not a
forward/lateral/yaw controller. Across 10 live-Bullet development checks, `0.75–2.0 Hz` at
amplitude `0.18–0.20`, calf ratio `2.5`, all remained upright. A paired rendered run at `1.5 Hz`,
amplitude `0.18`, seed 0 traveled `0.46 m / 8 s`, with Froude `+0.0253` allocentric and `+0.0261`
egocentric; see `coppelia_generic_duty_cycle_pilot/`. The higher-amplitude boundary (`0.24+`) fell.
This makes the motor babble physically usable and honest, but still ~4–5x below the useful
positive expert/pretraining forward range. These are development diagnostics, **not yet an
approved training dataset**, because the final parameter distribution must be frozen before
collection.

Structured-sine check (`coppelia_trot_sine_diagnostic/`): explicitly tested the generic CPG the
user proposed—diagonal leg phase, hip amplitude `0.05–0.10x`, thigh `1x`, swing-only calf
`2.5x`, with a shared sine on every leg. At the stable setting it reached `+0.021–0.024` forward
Froude. The calf phase sweep confirmed `-pi/2` is the useful sign: zero phase was stable but only
`+0.0096`, `+pi/2` moved backward, and `pi` almost stalled. Raising shared amplitude/lift or
frequency again caused falls. This validates the generic structured CPG form, but not an
expert-speed gait.

## Required next work

1. Build a genuinely good Coppelia-native expert controller: feedback/IK or fresh training using
   a trustworthy physics/task reward, not the failed WM reward.
2. Freeze one generic CPG parameter distribution before final collection; then collect it without
   Froude/outcome filtering. This is the no-demonstration babble condition.
3. Collect expert and babble with identical scene, Bullet version, timestep, initialization,
   camera/room, duration, signs, seeds, and output schema.
4. Report the measured goal-source vocabulary alongside babble coverage: forward `0.12–0.19`,
   lateral `-0.12–+0.07`; current data has no useful yaw-dominant goal. Do not reject babble clips
   for weak/near-zero Froude.
5. Render every candidate body/data source for user inspection and keep YAML beside every run.
6. Fit/reuse a pilot checkpoint, then verify cross-family nearest-neighbour risk in **predicted**
   Froude (`body_head(proj(actions))`). True-Froude margin is not an acceptance substitute.
7. Fit expert and babble pipelines separately against their own data/candidate pools, then rerun
   the original 2x2x2: pipeline x goal source x locked/free-offset.

## Guardrails

- Run CoppeliaSim with GUI and exactly one instance on port 23000.
- Validate every accepted gait for at least 100–160 steps; short runs hide rollover failure.
- Use `trunk_respondable` for dynamics measurements and `b1_flat_convex.ttt` for every live run.
- A simulator run or attractive video is not evidence that expert/babble comparison is ready.
