# B1 in CoppeliaSim — current handoff (2026-09-12)

## Scope

This is no longer a Q22 test. The work only establishes a usable B1 model under native
CoppeliaSim–Bullet dynamics and begins controller/babble development for a future clean F194
2x2x2 rerun.

**Important:** no clean/expert controller exists yet. The generic duty-cycle CPG is now a usable
claim-honest babble preview, but it is **not the final babble dataset**: forward Froude is still
well below the pretraining/expert band, and the final collection distribution has not been frozen.

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
- **No final Coppelia babble dataset.** Current best is a valid preview seed, not a frozen
  collection.
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
- separates designed diagnostic primitives from generic CPG + noise babble.

It is **usable pilot infrastructure**, not the final dataset collector. Earlier designed forward
runs reached only Froude `0.0080–0.0101`; the current generic duty-cycle preview reaches `0.0328`
upright in egocentric render. Useful positive hexapod/expert goals are still around `0.10–0.20`,
so the gap remains. Final collection must freeze the generic parameter distribution first, then
retain every rollout including falls; do not select clips by measured Froude.

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

**Current best preview seed:** `collect_b1_coppelia_fast_duty_preview.py` fixes a generic diagonal
duty-cycle CPG (65% planted stance, 35% raised return) at `2.0 Hz`, amplitude `0.24`, calf ratio
`2.5`, noise `0.03`. It is not a forward/lateral/yaw controller and uses no policy, demo,
retargeting, body feedback, or outcome selection. Egocentric render:
`coppelia_fast_duty_candidate/f2.0_a0.24_s9_ego.mp4`. Result: upright 8 s, displacement
`+0.596 m`, mean Froude `[+0.0328,+0.0026,-0.0062]`.

Prior stable duty-cycle seed: `collect_b1_coppelia_generic_duty_cycle_babble.py` at `1.5 Hz`,
amplitude `0.18`, seed 0 traveled `0.46 m / 8 s`, Froude `+0.0253` allocentric and `+0.0261`
egocentric; see `coppelia_generic_duty_cycle_pilot/`.

The old MuJoCo-scale sine amplitudes were tested in Coppelia. They can produce
larger instantaneous motion but mostly dump energy into lateral roll and fall. Added fixed generic
phase/sign-convention knobs for `trot-sine` and `duty-cycle` diagnostics:
`--generic-trot-pairing`, `--generic-thigh-sign-layout`, `--generic-calf-sign-layout`. The best
signed sine variants reached forward Froude `~0.03–0.05` briefly but lost height before 8 s. The
harder duty-cycle edge at amplitude `0.30` reached `[+0.0366,+0.0029,-0.0115]` allocentric, but
the paired egocentric rerun fell, so use `0.24` as the current safe preview.

Status: babble is now visibly walkable and claim-honest as a preview, but still below the
pretraining/expert forward range (`~0.10–0.20`). Do not treat it as the final babble dataset until
the parameter distribution is frozen and collected without outcome filtering.

Structured-sine check (`coppelia_trot_sine_diagnostic/`): explicitly tested the generic CPG the
user proposed—diagonal leg phase, hip amplitude `0.05–0.10x`, thigh `1x`, swing-only calf
`2.5x`, with a shared sine on every leg. At the stable setting it reached `+0.021–0.024` forward
Froude. The calf phase sweep confirmed `-pi/2` is the useful sign: zero phase was stable but only
`+0.0096`, `+pi/2` moved backward, and `pi` almost stalled. Raising shared amplitude/lift or
frequency again caused falls. This validates the generic structured CPG form, but not an
expert-speed gait.

## Actuator-vs-dynamics diagnosis (2026-09-12, continuation)

Before spending more effort on CPG parameter sweeps, checked whether the frozen preview
(`collect_b1_coppelia_fast_duty_preview.py`, forward Froude `+0.0316-0.0328`) is actually
actuator-limited — the same question that, for gecko this session, turned out to have a real,
fixable answer (a too-weak position-PID gain, not the gait shape). **For B1 the answer is
different: it is not actuator-limited.**

Measured directly from one run's own `joint_targets`/`joint_pos` (SDK order, `seed=9`):

- Commanded joint range per cycle is small (thigh `~0.13-0.18 rad` peak-to-peak) and the required
  tracking velocity implied by it is well under `1 rad/s` — nowhere near saturating any joint's
  rated speed, unlike gecko's `~20 rad/s` demand against a `6 rad/s` cap.
- Achieved/commanded range ratio is `0.61-0.85` across the 12 joints — a real but modest
  tracking shortfall, not gecko's severe `~0.5x` saturation.
- **Raising `--pid-p 300->500 --pid-d 5->8` at the identical frozen config did not close the gap
  and instead caused a fall** (`up.z` collapsed from `0.995` to `-0.998` around step 140, worst
  joint error jumped to `0.68 rad`) — tighter tracking under contact disturbances traded away
  stability rather than adding authority, the same failure mode found and rejected for gecko's
  active joints earlier this session (raising P without matched damping fights the very
  disturbance it needs to absorb).

**Read: the forward-Froude ceiling here is a genuine open-loop dynamic-balance limit, not an
actuator-authority problem.** This independently confirms (with a mechanism, not just the
empirical "amplitudes 0.30-0.40 rolled over" observation already on record) that Required next
work #1 — a real feedback/IK controller — is the correct next step. Further CPG amplitude/gain
sweeps on the open-loop controller are very unlikely to close the `0.03` vs `0.10-0.20` gap; that
gap needs balance feedback, not a better-tuned open-loop waveform.

## Amplitude/duty-factor push (2026-09-12, second continuation)

User's direction: focus on generic babble, not the expert controller yet, and push the CPG
harder even at fall risk (accepting open-loop instability as expected). A real multi-seed sweep
(2-3 seeds per cell, `--screen-only`, fast iteration -- single-shot results are not trustworthy
here, see the actuator-vs-dynamics section above) rather than more single-run guesses.

**Amplitude alone, at the frozen shape (duty=0.65): survival collapses fast past ~0.28.**
`0.28`: 2/3 upright. `0.32`: 1/3. `0.36`: 0/3. `0.40`: 0/3. Every fall past 0.28 is
**lateral-dominant**, with a suspiciously consistent lateral Froude (~0.12-0.13) across very
different configs -- almost certainly the signature of a fully-flipped resting pose, not
meaningfully different physics each time (same pattern found for gecko's own fallen-state
artifact earlier this session).

**Hip channel ruled out as the driver.** Tested `--generic-gait-shape trot-sine
--generic-hip-ratio 0.0` (hip fully zeroed) at amplitude 0.30-0.40: fell 6/6, same ~0.125-0.128
lateral signature. The lateral instability comes from the thigh/calf trot mechanics themselves,
not hip ab/adduction -- disproves the natural first hypothesis. `trot-sine` (pure sine) is also
simply less stable than `duty-cycle` at every amplitude tested, confirming duty-cycle as the
right base shape (already suspected, now confirmed by a losing counter-test).

**`duty_factor` (stance/swing split) is the real lever, same mechanism as gecko's wave-gait
fix.** Swept at amplitude 0.32 (partial-survival zone): `0.55`/`0.60` made it worse (0/2 each,
more lateral). `0.70`: 1/2. **`0.75`: 3/3 upright** (confirmed with a third seed), forward-
dominant, lateral Froude down to `~0.006` from `0.03-0.13` -- a real, large stability
improvement from more of the cycle spent with feet planted (bigger support margin), not a
speed increase. Same principle as gecko's diagonal-trot-to-wave-gait fix earlier this session.

**But the ceiling itself did not move.** Re-swept amplitude at the improved `duty=0.75`:
`0.32` survives (3/3, forward Froude `0.024-0.028`, close to the old best `0.0316-0.0328`, not
higher). `0.34`: 1/3. `0.36`: 0/3. `0.38`: 1/3 (inconsistent). Past `~0.34`, falls return with
the same ~0.12+ lateral signature regardless of the duty-factor fix. **Forward Froude tops out
around `0.024-0.034` across every stable configuration found in this entire push, both before
and after the duty-factor improvement.** The extra amplitude headroom `duty=0.75` bought went
into stability margin, not speed.

**Read.** This independently reinforces the actuator-vs-dynamics diagnosis above with a much
larger sweep (not one config): the `~0.03` forward-Froude ceiling is a real wall for this
open-loop CPG family, not an artifact of one under-tuned parameter set. `duty=0.75, amp=0.32` is
a genuine improvement over the old frozen preview (same speed, better survival, 10x less lateral
drift) and is worth adopting as the new default generic preview config -- but it does not close
the gap to the `0.10-0.20` target. Further amplitude/duty/shape tuning on an open-loop controller
is very unlikely to close it; this is now evidence from a real sweep, not a single-run guess.

**New default preview**: `sim/collect/collect_b1_coppelia_wide_stance_preview.py` fixes this
config (`freq=2.0, amp=0.32, duty=0.75, calf_ratio=2.5`). Full egocentric render across 4 more
seeds (not screen-only, real frames/video, `results/wm/dataset/b1_babble/
coppelia_wide_stance_preview/`): **3/4 upright** (seeds 6,7,8; forward Froude `0.0276-0.0283`,
lateral `0.0013-0.0079`), **1/4 fell** (seed 5, kept as evidence, not discarded -- forward
`0.0208`, lateral `0.0551` before falling). Consistent with the sweep's own survival rate; this
config is meaningfully more reliable than the old preview but still not risk-free, matching the
genuine open-loop marginal-stability story throughout this section.

## Coupled joint mechanism, ported from Egocentric VSM's actual reference code (2026-09-12, third continuation)

User pushback, correctly: assuming a competing paper secretly used closed-loop feedback (to
explain why their open-loop CPG is stable) without checking is not a finding, it's an excuse.
Their reference implementation is available locally (`doc/ref/Egocentric_VSM/env_agent.py`,
`move_altas`) and was read directly rather than assumed.

**Their "CPG" is not a per-joint sinusoid at all.** It is a 3-phase discrete cycle where only ONE
number per leg is actually randomized (hip); the other two joints are fixed LINEAR functions of
it (`knee = 0.6 - hip`, `ankle = -(hip+knee)`) that keep the foot's orientation coherent through
the whole stride, by construction. This is a design-time kinematic prior, not real-time feedback
-- it never reads robot state. Every B1 gait shape tried before this (including the
`duty=0.75, amp=0.32` config adopted above) independently modulates hip/thigh/calf with separate
sines/ratios/phases; nothing enforced the calf staying kinematically coherent with the thigh's
own swing.

**Ported directly**: `generic_coupled_action_at` (`sim/collect/collect_b1_coppelia_babble.py`),
`calf = -coupling_ratio * thigh`, pure coupling, no independent calf motion at all. Tested first:
only `coupling_ratio=0.6` survived (of 0.3/0.6/1.0/1.5), and even that barely moved (forward
Froude `0.0039`) -- the foot never actively lifts, so it likely drags the whole stride.

**`generic_coupled_duty_action_at`: coupling during stance, one active clearance arc during
swing** (the same stance/swing split as the duty-cycle shape, but the calf follows the thigh
algebraically while planted instead of staying at a fixed neutral). This is the real result:

- `coupling_ratio=0.6, duty_factor=0.65, freq=2.0`, amplitude swept 0.20-0.30, 4 seeds each.
- **Amplitude 0.20-0.28: 20/20 upright.** Lateral Froude consistently under `0.003` across
  every single run -- roughly 10-40x straighter than any uncoupled config in the amplitude/
  duty-factor push above (lateral there ranged `0.006-0.13`). Forward Froude `0.017-0.026`.
- Amplitude 0.29-0.30: survival starts dropping (6/8) -- a real, sharp edge, not a gradual one.
- **This does not break the `~0.03` forward-Froude ceiling** -- speed is comparable to, not
  higher than, the uncoupled best. The win is reliability and straightness, not raw speed.

Confirmed with real egocentric render, not just screen-only: 3/3 more seeds (10,11,12) upright,
`results/wm/dataset/b1_babble/coppelia_coupled_duty_preview/`. **New default preview**:
`sim/collect/collect_b1_coppelia_coupled_duty_preview.py`
(`freq=2.0, amp=0.28, coupling_ratio=0.6, duty=0.65`) -- 23/23 survival across every render and
sweep run at this setting, the most reliable generic config found in this entire investigation.

**Read.** The coupling mechanism is real and load-bearing for stability -- confirms the user's
instinct that Egocentric VSM's method was worth copying, once actually read rather than assumed.
It does not, on its own, close the `0.03` vs `0.10-0.20` speed gap; that gap's diagnosis (real
open-loop dynamic-balance ceiling, actuator-vs-dynamics section above) still stands. Next test
worth running: does the coupling mechanism's stability margin allow a HIGHER frequency (more
strides/second at the same safe amplitude) to close some of the speed gap, since frequency was
not yet swept combined with coupling.

## Frequency push on the coupled-duty gait (2026-09-12, fourth continuation)

User's direction: Froude has to actually match or close the gap toward the target distribution,
not just be reliable. The coupled gait's stability margin (previous section) had not yet been
spent on speed -- pushed frequency next, since amplitude alone was already shown to plateau.

**Frequency, at the reliable amp=0.28: real gains, then a sharp wall.** 2.5/3.0/3.5/4.0/4.5 Hz
all 3/3 upright with forward Froude climbing smoothly (`0.024 -> 0.027 -> 0.031 -> 0.032 ->
0.036`). **5.0 Hz: falls outright (0/3)**, a sharp edge, not gradual. 4.6 Hz confirmed as the
reliable ceiling at this amplitude (3/3, Froude `0.034-0.037`).

**Trading amplitude for frequency unlocked a real, substantial jump.** Lower amplitude bought
back stability margin at higher frequency: `amp=0.22, freq=5.0` -- 3/3 upright, Froude
`0.046-0.050`, a genuine step up, not noise. Below that (amp `0.15-0.18`) frequency alone stopped
helping (Froude fell back to `0.016-0.037`) -- confirms this is a real sweet spot, not a
monotonic amplitude-down/frequency-up trend. Refined around it: `amp=0.24, freq=5.0` -> Froude
`0.053-0.054` (3/3). **`amp=0.26, freq=5.0` -> Froude `0.055-0.058`** (3/3 on the first seed
batch). `amp=0.28` at this frequency falls outright (0/3) -- confirms `0.26-0.28` is the real edge
at `freq=5.0`, matching the same kind of sharp cliff found throughout this whole investigation.

**Honest correction, not held back**: a second seed batch (20-22) at the `amp=0.26, freq=5.0`
"best" point only survived **1/3**, not 3/3 -- combined across both batches this is
**4/6 (67%) survival**, not the clean win the first batch suggested. Genuine seed-to-seed
non-determinism (same root cause characterized in the actuator-vs-dynamics section: Bullet's
contact solver, amplified by this gait being right at its stability edge) means this specific
peak is real but not fully reliable. Froude at this setting, across all 6 seeds tried:
`0.043-0.058`.

**Read.** Forward Froude nearly doubled from the original `~0.03` ceiling to `0.05-0.06` at the
best point found (`amp=0.26, freq=5.0`) -- genuine, substantial progress toward the `0.10-0.20`
target, not there yet but meaningfully closer. The speed/reliability trade-off is now real and
explicit: `amp=0.28, freq=4.6` is the reliable choice (Froude `~0.035`, high survival across
every seed tried), `amp=0.26, freq=5.0` is the fast choice (Froude `~0.05`, `~67%` survival).
Given this project's own retain-every-rollout babble philosophy, the faster/less-reliable point
may still be an acceptable final distribution choice -- falls are valid data, not discarded --
but that is a real decision to make explicitly, not a free win.

## Airtime (duty-factor) beats amplitude as the speed lever (2026-09-12, fifth continuation)

User's instinct, directly correct: "higher amp or airtime." Amplitude alone (previous section)
found a real but unreliable peak (`amp=0.26, freq=5.0, duty=0.65`: 4/6, 67%). Tested airtime
(lowering `duty_factor` -- more swing/less stance) at the same frequency instead of pushing
amplitude further.

**`duty=0.55/0.60` at the already-reliable `amp=0.22, freq=5.0`: better speed AND better
reliability than pushing amplitude.** 8/8 upright (both duty values, 4 seeds each), Froude
`0.052-0.056`, lateral drift tiny (`0.002-0.010`) -- beats the amplitude-pushed peak on every
axis at once. `duty=0.50` (max airtime) actually gave slightly LESS speed (`0.045-0.049`) than
0.55/0.60 -- confirms `~0.55-0.60` is a real local optimum, not "more airtime is always better."

**Combined with amplitude back up: the actual best config found in this whole investigation.**
`amp=0.26, duty=0.55, freq=5.0, coupling_ratio=0.6`: **11/11 upright** across every seed tested
(8 screen-only + 3 full egocentric renders, not just fast screening) -- fully reliable, not a
lucky batch. **Forward Froude 0.050-0.063.** `amp=0.28` at `duty=0.55` starts failing again
(1/4) -- the amplitude ceiling itself did not move, but duty=0.55 lets amp=0.26 be reliably used
where duty=0.65 could not.

**New default preview**: `sim/collect/collect_b1_coppelia_fast_air_preview.py`. Rendered,
`results/wm/dataset/b1_babble/coppelia_coupled_duty_air_preview/`.

**Read.** Forward Froude has now roughly DOUBLED from the original uncoupled ceiling (`~0.03` ->
`0.05-0.06`), fully reliably, not as an unstable peak. Still short of the `0.10-0.20` target, but
this is the closest and most solid point found across this entire investigation. Lateral/yaw
drift (`0.006-0.028`) is higher than the straightest configs (duty=0.65's `~0.003`) but still far
below anything the uncoupled gaits produced (`0.03-0.13`) -- a real, worthwhile trade for the
speed gained.

## Qualified babble generator checklist learned from failures

This is the current bar for calling a babble source usable. It is not all solved yet; it is the
checklist distilled from failure cases in this session.

1. **Stable, smooth motion must be measured directly.** F202 showed that erratic babble can make
   adaptation worse than doing nothing. A candidate needs a quantitative smoothness/contact gate
   such as joint-acceleration variance and contact consistency, not only Froude and occasional
   visual inspection.
2. **The generator must stay genuinely generic.** One broad CPG parameterization is allowed:
   sampled frequency, amplitude, phase offsets, joint-role scales, and per-step noise. Per-body or
   per-behavior primitives such as separate strafe/yaw mechanisms are premise drift, not undirected
   babble.
3. **Separation must be checked in the fitted model's predicted space.** F194's raw/true-Froude
   margin check was the wrong acceptance test. The margin must be checked through the actual fitted
   checkpoint prediction, e.g. `body_head(proj(actions))`.
4. **Coverage must match the actual goal vocabulary.** Collection effort should be calibrated
   against the source body's goal Froude range, not an arbitrary target. Current Coppelia B1 preview
   remains below the useful pretraining/expert band.
5. **Stage-1 adaptability is a required pre-check.** Before downstream experiments, run the
   candidate babble through `wm.adapt` and verify held-out ratio improves. This cheap gate would
   have caught F202 before the rest of the pipeline depended on it.

Bottom line: the current B1 Coppelia babble work has produced a much better live-physics preview
generator, but it is not yet a qualified final babble dataset until these gates pass.

## Quality-metric pilot after tip-toe diagnosis (2026-09-12)

Added reporting-only motion-quality metrics to `sim/collect/collect_b1_coppelia_babble.py`; every
new rollout YAML/NPZ now records joint velocity/acceleration/jerk, target/action acceleration,
tracking error, contact switches, support count, and foot-height ranges. These metrics do **not**
filter or discard rollouts; they only make the "stable/smooth/contact-consistent" gate measurable.

Also added `sim/collect/collect_b1_coppelia_swing_distance_preview.py`: a reproducible fixed
preview for the user's "more swing distance, less tip-toe" idea (`freq=4.5 Hz`, `amp=0.32`,
`duty=0.55`, `coupling=0.6`, `clearance=1.1`). It is the same generic coupled-duty CPG family,
not a new behavior primitive.

Rendered comparison, seed 11, ego + allo:
`results/wm/dataset/b1_babble/coppelia_quality_pilot/`.

| preview | view | status | mean body Froude | joint acc/jerk RMS | contact/support read |
|---|---|---|---|---|---|
| soft-air `5.0Hz/a0.26/d0.55/c1.1` | allo | upright | `[+0.0485,-0.0006,-0.0037]` | `16.69 / 552.11` | support mean `0.49`, `<=1 foot` frac `1.00` |
| soft-air `5.0Hz/a0.26/d0.55/c1.1` | ego | upright | `[+0.0462,-0.0002,-0.0015]` | `16.66 / 549.95` | support mean `0.49`, `<=1 foot` frac `1.00` |
| swing-distance `4.5Hz/a0.32/d0.55/c1.1` | allo | upright | `[+0.0398,+0.0036,+0.0040]` | `14.89 / 454.36` | support mean `0.53`, `<=1 foot` frac `1.00` |
| swing-distance `4.5Hz/a0.32/d0.55/c1.1` | ego | upright | `[+0.0392,+0.0046,+0.0023]` | `14.89 / 454.53` | support mean `0.52`, `<=1 foot` frac `1.00` |

Read: increasing swing distance and reducing Hz helps smoothness (`~18%` lower joint
acceleration, `~17%` lower jerk), but it costs speed (`~0.046-0.049` down to `~0.039`) and does
**not** fix the deeper tip-toe/contact problem. The low support-count metric makes the visual
failure mode concrete: these previews move forward while barely registering stable foot support.
Next babble work should target contact-consistent stance, not more speed-only tuning.

## Working Coppelia B1 babble preview: target-scale forward + lateral/yaw knobs (2026-09-12)

Added generic CPG parameters to `sim/collect/collect_b1_coppelia_babble.py`:

- `--generic-stance-calf-bias`: planted-leg calf preload/extension. Negative values extend the
  stance leg and were the missing speed/contact lever.
- `--generic-hip-sign-layout`, `--generic-hip-phase`, `--generic-hip-clock`: one generic hip
  oscillator parameterization for lateral/yaw coverage. This is not separate behavior scripting;
  it is the same coupled-duty CPG with sampled hip sign/clock/ratio parameters.

This finally made the Froude target achievable in native Coppelia/Bullet. Screen validation:

- Forward target candidate: `freq=5.0`, `amp=0.26`, `duty=0.55`, `coupling=0.6`,
  `clearance=1.1`, `stance_calf_bias=-1.2`, no hip oscillator. Seeds 10-14: **5/5 upright**,
  forward Froude `0.1187-0.1266`.
- Turn candidate: same base but `stance_calf_bias=-1.0`, global hip oscillator,
  `hip_sign_layout=left-right`, `hip_ratio=0.8`. Seeds 10-14: **5/5 upright**, forward
  `0.1013-0.1103`, lateral about `-0.022 to -0.028`, yaw `0.0445-0.0515`.
- Lateral/mixed candidate: same base but `hip_sign_layout=diagonal`, `hip_ratio=0.3`.
  Rendered upright in both views at seed 12, forward `~0.103-0.104`, lateral `~0.029-0.030`,
  yaw `~-0.021 to -0.024`. Stronger diagonal hip (`0.5-1.0`) can produce lateral `~0.04-0.09`
  and yaw `~0.05-0.10`, but is a genuine fall-boundary sample under Bullet and should not be the
  safe default.

Named preview scripts:

- `sim/collect/collect_b1_coppelia_target_forward_preview.py`
- `sim/collect/collect_b1_coppelia_turn_preview.py`
- `sim/collect/collect_b1_coppelia_lateral_preview.py`
- `sim/collect/collect_b1_coppelia_side_yaw_edge_preview.py` (diagnostic edge only; not safe)

Rendered proof, seed 12: `results/wm/dataset/b1_babble/coppelia_working_babble_preview/`.

| preview | view | status | mean body Froude | note |
|---|---|---|---|---|
| target-forward | allo | upright | `[+0.1220,+0.0137,+0.0117]` | inside lower useful forward band |
| target-forward | ego | upright | `[+0.1232,+0.0136,+0.0116]` | same camera/room convention |
| turn-left/right-hip | allo | upright | `[+0.1095,-0.0231,+0.0472]` | reliable yaw while moving fast |
| turn-left/right-hip | ego | upright | `[+0.1073,-0.0221,+0.0454]` | reliable yaw while moving fast |
| lateral-diagonal-hip | allo | upright | `[+0.1044,+0.0299,-0.0236]` | safer lateral/mixed preview |
| lateral-diagonal-hip | ego | upright | `[+0.1028,+0.0290,-0.0205]` | safer lateral/mixed preview |

Read: the babble generator now has usable preview coverage for forward, yaw/turn, and moderate
lateral/mixed motion at Coppelia-native dynamics. It is still **not a final qualified babble
dataset**: contact/support metrics remain imperfect (`support_count_mean` often below 1), strong
lateral/yaw is unstable, and no final parameter distribution has been frozen.

## Required next work

1. Build a genuinely good Coppelia-native expert controller: feedback/IK or fresh training using
   a trustworthy physics/task reward, not the failed WM reward.
2. Freeze one generic CPG parameter distribution before final collection; then collect it without
   Froude/outcome filtering. This is the no-demonstration babble condition.
3. Add and enforce the qualified-babble gates above: smoothness/contact metrics, predicted-space
   separation, vocabulary coverage report, and `wm.adapt` held-out improvement.
4. Collect expert and babble with identical scene, Bullet version, timestep, initialization,
   camera/room, duration, signs, seeds, and output schema.
5. Report the measured goal-source vocabulary alongside babble coverage: forward `0.12–0.19`,
   lateral `-0.12–+0.07`; current data has no useful yaw-dominant goal. Do not reject babble clips
   for weak/near-zero Froude.
6. Render every candidate body/data source for user inspection and keep YAML beside every run.
7. Fit/reuse a pilot checkpoint, then verify cross-family nearest-neighbour risk in **predicted**
   Froude (`body_head(proj(actions))`). True-Froude margin is not an acceptance substitute.
8. Fit expert and babble pipelines separately against their own data/candidate pools, then rerun
   the original 2x2x2: pipeline x goal source x locked/free-offset.

## Guardrails

- Run CoppeliaSim with GUI and exactly one instance on port 23000.
- Validate every accepted gait for at least 100–160 steps; short runs hide rollover failure.
- Use `trunk_respondable` for dynamics measurements and `b1_flat_convex.ttt` for every live run.
- A simulator run or attractive video is not evidence that expert/babble comparison is ready.
