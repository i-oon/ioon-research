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

## Claim-honest generic babble rule and pilot (2026-09-12)

The aggressive target-aware/designed-primitive sweep above is **not valid babble for the paper's
claim**. It remains diagnostic only. Claim-honest babble is fixed to one equation on every
normalized joint: `A*sin(2*pi*f*t + phase_j) + per_step_noise_j`; no behavior mechanisms,
demonstrations, retargeting, target-Froude tuning, or outcome-based selection. Every precommitted
attempt—including falls—must remain recorded. Froude is evaluation after collection, never a
retention criterion.

Implementation: `sim/collect/collect_b1_coppelia_generic_babble.py`, Bullet only. The first pilot
was marked invalid because it mistakenly used independent per-joint frequencies. Corrected v2
uses one shared frequency and amplitude plus per-joint phases and mandatory noise. Its unchanged
four-seed precommit produced 2 upright and 2 fallen rollouts; all four NPZ/MP4/YAML artifacts were
retained under `coppelia_generic_pilot_v2/`. Upright motions were weak and uncommanded, with mean
Froude `[-0.0125,+0.0073,-0.0115]` and `[+0.0044,+0.0031,+0.0077]`. This is pilot evidence only,
not approved training data.

A subsequent fixed diagonal-trot phase pilot (`coppelia_generic_trot_pilot/`) tested the user's
approved generic gait-cycle prior at shared `1–3 Hz`, without changing the amplitude distribution,
seeds, retention, or Froude rules. All 4/4 fell. The equal-amplitude oscillator drives the
ab/adduction hips as hard as the sagittal joints, causing lateral collapse (`|lateral Froude|
0.071–0.142`) rather than a walkable trot. Phase coordination alone is therefore insufficient.
The next possible prior is stronger and must be named honestly: generic **quadruped** joint-role
structure with small/neutral hips and coordinated thigh/calf motion.

That joint-role pilot (`coppelia_quadruped_trot_pilot/`) was then run with the same four-seed,
retain-everything rule: hip amplitude `0.1x`, thigh/calf `1x`, diagonal phases, shared `1–3 Hz`.
Three of four fell; seed 3 stayed upright but shuffled backward at forward Froude `-0.0102`.
Therefore small hips plus phase offsets still do not make the symmetric sine a walkable gait. The
remaining missing locomotion prior is an explicit stance/swing shape (e.g. slow planted push plus
short lifted swing/rectified knee), which is stronger than the same-sine-per-joint premise and must
be approved and named before testing.

Approved stance/swing follow-up (`coppelia_stance_swing_pilot/`): two precommitted calf-lift ratios
(`1.5x`, `2.0x`) x four unchanged seeds, rectified swing-only calf lift, 1 s ramp, all outcomes
retained. Each ratio survived only 1/4 seeds. On the shared stable seed 3, higher lift improved
forward Froude `0.0121 -> 0.0179` (~48%) with lateral/yaw near zero. This supports higher,
swing-only calf lift but does not solve robustness or reach the pretraining range. Side-by-side
allocentric replay: `coppelia_stance_swing_pilot/allocentric/seed3_calf_lift_comparison.mp4`.

Controlled clearance grid (`coppelia_lift_frequency_grid/`): 18 fixed cells over frequency
`1.5/1.75/2.0 Hz`, amplitude `0.14/0.18/0.22`, calf ratio `2.0/2.5`; 12/18 stayed upright. Best
stable cell was `1.75 Hz, 0.18, 2.5x`, Froude `[+0.0243,+0.0011,-0.0085]`. Foot telemetry confirms
the user's visual observation but identifies the mechanism: FL/FR world-height ranges were only
~`0.4–2.1 mm` while rear ranges were ~`38–44 mm`, despite similar actual calf joint motion on all
legs. Near the standing pose, front calf angle has almost no vertical leverage; more calf amplitude
alone cannot fix front clearance. The next controller must reshape front thigh/calf coordination
or command foot-space lift through IK.

## Required next work

1. Build a genuinely good Coppelia-native expert controller: feedback/IK or fresh training using
   a trustworthy physics/task reward, not the failed WM reward.
2. After the controller can reach the goal vocabulary, use the CPG collector only as a
   randomized **designed-primitive** baseline; do not call it undirected babble.
3. Collect expert and babble with identical scene, Bullet version, timestep, initialization,
   camera/room, duration, signs, seeds, and output schema.
4. Target the measured goal-source vocabulary first: forward `0.12–0.19`, lateral
   `-0.12–+0.07`; the current set has no useful yaw-dominant goal. Reject weak near-zero clips.
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
