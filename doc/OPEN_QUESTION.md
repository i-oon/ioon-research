# Cross-embodiment plan — status & open questions

> **Role**: What still has to be decided.
>
> Only genuinely open items. When a measurement settles one it moves to `FINDINGS.md` and leaves a single line in the settled table at the bottom, so no result is written out twice.

Living doc. Supersedes the old "argue vs prove / terrain / leg-length" questions,
which are now settled (see bottom). Updated 2026-09-10.

Stage 1 measurements that constrain everything below are in **[FINDINGS.md](FINDINGS.md)**;
this file carries only what is still undecided.

## Q23. Does the vision-goal pathway carry a hexapod-CoppeliaSim-vs-B1-MuJoCo physics-origin bias, separate from a generically weak signal? (new, 2026-09-11 — not started)

**User-raised, sharper than Q22's framing, not yet tested.** Q22 asked whether the frozen encoder
is sensitive to physics-engine signature in general. This is a more specific, more dangerous
version of that question, aimed at exactly the pathway that matters: **`vision_goal()`** (used in
mode D throughout this session's B1 work) reads a hexapod goal clip's froude value from frames that
were genuinely, dynamically simulated in **CoppeliaSim/Bullet** (`sim/collect/collect_ik.py`).
Every B1 candidate's own froude, by contrast, traces back to **MuJoCo**-computed motion. Both get
mapped into the same froude number line by the same `body_head`. **If that mapping carries a
systematic (not random-noise) bias tied to which physics engine originally produced the motion, no
amount of B1 candidate-selection quality can close the resulting gap** -- B1's froude values could
be structurally unable to reach the hexapod-goal's froude scale, independent of whether the right
candidate was picked.

**Distinguishes two hypotheses this session's `vision_goal` findings (F192) never separated**:
F192 attributed hexapod's own weak lateral vision-signal to a generically weak/compressed channel.
This raises a second, different candidate explanation for the SAME symptom: not "the signal is
weak everywhere," but "the signal is fine for hexapod-on-hexapod, and the mismatch specifically
appears when a MuJoCo-sourced body's froude is compared against it" -- a physics-origin bias in the
goal-reading pathway itself, not a general representation weakness.

**Partial mitigation already in place, not yet verified sufficient**: stage 4's rehearsal fitting
(F186/F191, `--also hexapod=data/egocentric/beh12_c10f10t10_ego_flat`) fits `body_head` jointly on
BOTH bodies' data, which would absorb a bias if it is a simple, linear scale/offset -- but would
NOT fully correct a bias that is a nonlinear function of the source engine's specific signature.
Whether the rehearsal fit's known-partial fix (F191: lateral misread 4/4 -> 3/4, not 0/4) reflects
a residual weak-signal problem or a residual physics-origin-bias problem is exactly what is
undetermined here.

**Not yet tested, first step**: check whether a candidate's REAL froude distance to a goal (using
`beh12_b1_ego_flat`'s own EXPERT clips, which -- like B1's babble -- also trace back to MuJoCo, per
`sim/collect/rollout_b1_mujoco.py`) shows the same lateral degradation pattern F192 found for
hexapod's own goal-reading, or whether it is specifically worse/different when the CANDIDATE side
is MuJoCo-sourced and the GOAL side is CoppeliaSim-sourced (the actual cross-origin case) versus
when checking hexapod-goal-reading against hexapod-sourced candidates alone (same-origin,
untested this way so far). This isolates origin-bias from generic weak-signal without needing any
new simulator engineering (unlike Q22) -- it is a re-analysis of data and checkpoints that already
exist.

**Second, related, and not yet started: a genuinely cross-embodiment stage-3 InfoNCE has never been
built, so it cannot be the explanation for anything measured so far (see FINDINGS.md F197).**
`wm/adapt3.py`'s positive/negative pairs are drawn entirely from one embodiment's own recorded
condition labels at matching timesteps -- Froude values are never read inside that file. The
Froude-correspondence idea this project's own claim rests on (hexapod Froude 0.1 <-> B1 Froude 0.1
as an InfoNCE positive pair, the retargeting-style correspondence used in
[Latent Cross-Embodiment Policies](https://arxiv.org/html/2506.14608v4)) has not been implemented.
If it were, this file's origin-bias question would become load-bearing at *training* time (does a
cross-origin Froude match actually correspond to the same behaviour, or does hexapod-Bullet-Froude
vs B1-MuJoCo-Froude carry a scale/engine bias that would teach the wrong positive pairs) rather than
only at evaluation, as it is today. Building and testing this is the concrete next step for this
question; no script exists yet. See `doc/ref/literature_review3_infonce_modality_gap.md` for the
literature this design should follow (reconstruction + contrastive hybrid, controlled/mixed negative
sampling across embodiments, not vanilla independent-encoder InfoNCE).

## Q22. Obsolete as a reward-gap test; B1-in-CoppeliaSim engineering continues for future native RL (settled 2026-09-11)

**Progress update, same day, from a parallel session** (full detail in
`results/wm/dataset/b1_babble/q22_handoff_prompt.md`, kept current there, not duplicated here):
the standing failure diagnosed earlier turned out to be a simple initialization bug, not a
gain/force-sensor problem -- the imported scene stores every joint at 0 rad, motor TARGETS were
being set correctly but the ACTUAL joint positions were never initialized to match before
`startSimulation()`, so Bullet started with straight legs punched through the floor. Fixed
(`sim.setJointPosition` before dynamics start); stable 100-step stand confirmed
(final z=0.5358m, up.z>=0.9998). A hand-designed native CPG controller (not a MuJoCo-policy
transplant) now survives 160 steps for forward/lateral/yaw presets -- forward is solid, lateral and
yaw are weak/coupled (same shape of problem as F190's original MuJoCo babble). **Explicitly not
yet done**: a real Coppelia-native expert controller, a proper (non-family-targeted) babble
collector, and any new 2x2x2 rerun -- do not treat the working CPG presets as ready data. See the
handoff file for the complete, currently-accurate status and required next steps.

**The original Q22 is no longer the reason for this work.** The current vision+action pipeline
does not mix live physics engines: CoppeliaSim only kinematically displays B1 trajectories already
computed by MuJoCo, and candidate scoring is action-only. A physics-signature gap can matter once
training needs live simulator feedback, but it does not explain F195 in the present pipeline.
Also, `rollout_b1_mujoco.py` already documents that transplanting the MuJoCo-trained policy into
CoppeliaSim was tried and failed. Do not repeat that experiment. See F196 for the now-working
Bullet stand and the first CoppeliaSim-native hand-designed controller.

**Current engineering result:** the apparent 0.23 m standing collapse was caused by a missing
initial condition, not force-sensor compliance or weak motors. The imported scene stores all
joints at zero; `setJointTargetPosition` does not set the initial joint state. Initializing each
joint with `setJointPosition(DEFAULT_IL)` before starting Bullet gives a stable 100-step stand.
The base force sensor is documented by CoppeliaSim as an initially rigid link, and its exposed
properties contain break thresholds but no stiffness setting. CoppeliaSim's position PID is also
not MuJoCo kp/kv: it drives a constrained motor command while `targetForce` is separate.

The first native diagonal-trot CPG (`b1_coppelia_cpg_controller.py`) stayed upright for 160 steps
(8 s), advanced 0.177 m, drifted 0.015 m laterally, held z >= 0.535 m and up.z >= 0.998, with
worst joint tracking error 0.046 rad. This establishes a conservative controllable baseline for
future environment/reset/action plumbing; it does not revive the blocked WM-reward RL plan.

**User-raised, confirmed factually in code (not yet tested for actual effect).** The hexapod/insect
pretraining data (`sim/collect/collect_ik.py`) is genuinely, dynamically simulated IN CoppeliaSim:
`sim.setJointTargetPosition()` (a motor target) followed by `sim.step()` in a live simulation loop
-- real gravity, contact, compliance, computed by CoppeliaSim's own physics engine. B1's data (all
of this session's babble/closed-loop work) is different: MuJoCo computes the entire trajectory,
and CoppeliaSim only poses the result kinematically for rendering ("KINEMATIC: the body is posed,
not simulated. It cannot fall" -- printed on every B1 closed-loop run). B1's own motion is
genuinely physically simulated too, just by a DIFFERENT engine than the one that renders it.

**The open question**: MuJoCo's and CoppeliaSim's contact solvers are different numerical
implementations. Even for the same commanded gait, the resulting motion's fine-grained signature
(foot-slip pattern, bounce frequency, compliance-driven jitter) could genuinely differ between
engines. The frozen encoder was pretrained exclusively on CoppeliaSim-physics-signature motion; B1's
fine-tuning data carries MuJoCo's signature instead. If the encoder is sensitive to this
engine-specific fingerprint, that is a real, previously invisible domain gap -- distinct from the
"pretrain representation is generically coarse" story (F102/F110/old-F136's type-vs-magnitude wall)
-- and could be a contributing cause of F195's reward-quality-gate failure and the broader
B1-adaptation difficulty this session's work sits on top of.

**Not yet tested, and a real build, not a quick check**: B1 has no motorized/dynamic setup in
`b1_flat.ttt` currently, only kinematic posing. A first test would need B1 implemented natively as
a dynamic body in CoppeliaSim (motorized joints, `sim.step()`, matching the hexapod's own collection
method) for a handful of already-collected MuJoCo gaits, then comparing the resulting motion's
low-level signature and/or the encoder's own embedding statistics against the MuJoCo-then-
kinematically-posed version of the same commanded gait.

**Considered and NOT recommended**: moving B1's physics wholesale to CoppeliaSim, replacing MuJoCo
entirely. MuJoCo's contact solver is the standard, more battle-tested choice for legged-robot RL
specifically (matching Unitree's own `unitree_mujoco` tooling), not an accidental detour to correct.
The recommended path is testing the SIGNATURE GAP directly on a small scale, not replacing the
physics engine project-wide.

**Feasibility investigation, 2026-09-11: real progress, not yet a working stand.** Checked whether
an existing pre-tuned quadruped (Laikago or similar) ships with CoppeliaSim to skip this work --
it does not (only `hexapod.ttm`/`ant hexapod.ttm`, matching the insect body already in this
project); a paper simulating Laikago in CoppeliaSim almost certainly did the same URDF-import +
tuning work from scratch, not from a ready asset. Diagnosed, not guessed:

1. **Bullet + non-convex collision shapes hangs** (`Detected dynamically enabled, non-convex
   shapes` warning) -- switching the scene's dynamic engine to Newton
   (`sim.setInt32Param(sim.intparam_dynamic_engine, 3)`) removes the hang; ran 60 steps without
   freezing.
2. **Default joint PID gains are far too weak** (`pid_p=0.1` on every joint vs MuJoCo's kp
   550-970 for the same joints) -- explains an early attempt sinking steadily instead of holding
   a standing pose. Setting matched gains stopped the steady sink but produced bouncing
   instability instead (0.54->0.34->0.22->0.43->0.36->0.24 m over 60 steps) -- not a clean stand
   either.
3. **Mass import is correct** -- summed only the `_respondable` (dynamic/collision) shapes:
   62.574 kg, matching the MuJoCo XML's 62.58 kg almost exactly. The `_visual` shapes also carry
   nonzero mass values but are confirmed `static=1` (kinematic, non-dynamic), so they do not
   double-count into the physics -- checked directly, not assumed.
4. **The base connection is a force sensor, not a plain joint or rigid weld**: the hierarchy is
   `base_visual` (dynamic, top, no parent) -> `floating_base` (object type 12 = FORCE SENSOR,
   confirmed via `sim.getObjectType`) -> `trunk_respondable` (dynamic, the body actually carrying
   the leg/joint chain). A force sensor link can be configured rigid or compliant/breakable --
   which this one is has NOT been checked yet, and is the most likely remaining cause of the
   bouncing instability (a compliant or under-configured link between the tracked root and the
   actual dynamic body would look exactly like this).

**Stopped here, not because it is unsolvable, but because it had become an open-ended,
multi-layered sim-engineering investigation** (engine choice, then gain magnitude, then mass
fidelity, then hierarchy/force-sensor rigidity) rather than a bounded check, and continuing to
iterate without a firm diagnosis budget was starting to look like guessing.

**Critical correction, caught before more work was wasted on it (user-flagged, verified
immediately): switching to Newton to dodge the Bullet/non-convex hang INVALIDATES the whole
comparison.** Checked directly: the hexapod pretraining scene (`sim/env/medauroidea_c10f10t10.ttt`)
uses engine=0 (**Bullet**), not Newton. If B1's dynamic test ran on Newton while hexapod's
pretraining ran on Bullet, any difference found would conflate TWO confounds -- engine choice
(Bullet vs Newton) and the actual thing Q22 wants to isolate (dynamic-vs-kinematic-render) -- and
the result would say nothing trustworthy about either one. **The comparison must run B1 on
Bullet, matching hexapod exactly.** This makes the convex-decomposition path (fixing the actual
non-convex-collision cause of the hang, not routing around it with a different engine) a REQUIRED
step, not an optional alternative -- Newton is not a valid substitute for this test.

**Next concrete step, precisely scoped**: run convex decomposition (CoppeliaSim's own
`Convex decomposition hacd.lua` add-on, already loaded in every scene this project uses) on B1's
collision meshes, re-verify the standing test on BULLET specifically, then separately still check
`floating_base`'s force-sensor rigidity (that diagnosis stands regardless of engine). Script:
`scripts/diagnostics/objective_experiments/b1_coppelia_dynamics_probe.py` -- must be re-run with
`intparam_dynamic_engine` left at 0 (Bullet) once the collision meshes are fixed, never switched
to Newton for this specific comparison.

**Convex decomposition done, saved, and it fixed the hang -- new state, same day.**
`sim.convexDecompose(handle, 25, [1,650,400,4,0,0,0,0,0,0], [0.01,30.0,0.25,0,0,0,0,0,0,0])`
(the HACD method, code 25 -- exact parameter counts and values found via web search, not guessed)
applied to all of B1's `*_respondable` shapes that weren't already convex (trunk + all four legs'
hip/thigh/calf; rotor and foot shapes were already convex). Saved as a new scene,
`sim/env/b1_flat_convex.ttt` -- use this file, not `b1_flat.ttt`, for any further Q22 dynamics
work, so the decomposition never has to be redone. **Result on Bullet (engine=0, correctly matching
hexapod), MuJoCo-matched PID gains, `trunk_respondable` tracked instead of the disconnected
`base_visual`**: no hang, no chaotic bouncing -- the body settles cleanly to a STABLE but WRONG
height (0.60m start -> 0.23m by step 20, then flat 0.2286-0.2321 for the remaining 40 steps). This
is real progress: convex decomposition fully resolved the instability/hang (confirms that
diagnosis was correct), and the remaining problem is now a clean, isolated one -- the standing
pose is not being held against real gravity, not numerical chaos. **Foot-ground friction checked
and ruled out** (`bullet_body_friction`: foot 0.5, floor 1.0 -- both reasonable, not the cause).
**Not yet found**: why MuJoCo's own kp/kv values don't hold the pose here when they hold it in
MuJoCo itself -- candidates not yet checked are (a) `floating_base`'s force-sensor rigidity
(diagnosed earlier, still unfixed), (b) whether CoppeliaSim's PID gain units/scaling actually
correspond 1:1 to MuJoCo's kp/kv the way this session assumed, (c) whether the DEFAULT_IL pose's
centre of mass is genuinely balanced over the feet once real Bullet contact geometry (the new
convex hulls, not MuJoCo's exact collision geometry) is in play. **Stopped here for this session**
-- the handoff prompt (`results/wm/dataset/b1_babble/q22_handoff_prompt.md`) has been updated to
match this exact state for whoever continues it next.

**Second engine-precision concern, user-flagged, NOT resolved (2026-09-11).** This install has TWO
Bullet library versions (`libsimBullet-2-78.so`, `libsimBullet-2-83.so`). Checked whether they
could reintroduce the SAME confound the Newton mistake did (Q22's whole point is a controlled
comparison, so a sub-version mismatch between hexapod's scenes and B1's scene would be just as
invalidating as an engine-family mismatch). **Could not confirm which sub-version is active for
either scene**: `sim.intparam_dynamic_engine`'s enum has only ONE "Bullet" value
(`physics_bullet=0`, confirmed via `dir(sim)`/`getattr`), no separate 2.78/2.83 slot; both `.so`
files are loaded into every CoppeliaSim process regardless of scene (checked via `/proc/<pid>/maps`
grep, both present) so that doesn't distinguish which is ACTIVE; no plaintext version string found
in either `.ttt` file via `strings`. **This is a genuinely open, unresolved uncertainty, not
something ruled out** -- the most likely explanation (this CoppeliaSim version has deprecated the
2.78/2.83 choice at the scripting-API level and defaults everything to 2.83, keeping 2.78 only for
loading old scenes) is a guess, not a checked fact. **Before trusting any Q22 comparison result,
verify this properly** -- likely needs the GUI's scene "Common properties" dialog checked visually
(`DISPLAY=:0`), or CoppeliaSim's own release notes/source for when this became a single choice, not
a remote-API check.

## Q21. Week 15 advisor review: the candidate source is the bottleneck, and the ask is a real Controller (new, 2026-09-10 — blocking, and it reorders everything below)

Source: `feedbacks/feedback_ajan_go.md`, Week 15. Four asks landed, and they change what "done"
means for this thesis. Recorded here because they are decisions, not results.

**What P'Nine identified, and he is right.** The closed-loop result (Slide 30, F188) scores
candidates drawn from **B1's own recorded expert clips**. On a body that has never had a controller
there are no such clips, so the mechanism as demonstrated does not transfer. His question -- *where
do candidates come from on an unseen body?* -- has no answer in the current pipeline. The gecko arc
(F189) is the same problem from the other side: gecko has only babble, no expert library at all.

**His two options, and what we already hold of each:**

| option | what it needs | what exists already |
|---|---|---|
| 1. Action selection + candidate generator | CPG/noise candidates for the unseen body | **`collect_gecko_dataset.py` IS this** — CPG + noise, no controller. Also matches Egocentric VSM's own published method (CPG + 50 noised copies, FTM ranks, argmax) |
| 2. **RL policy + WM reward** (he recommends) | Froude error from the WM as a reward term in vanilla PPO | nothing built |

**Option 2 is NOT the thing F179 killed, and the distinction is the whole point.** F179 trained the
policy *inside* the FTM's imagination -- roll the forward model ~100 steps, bootstrap a critic
through it -- and died of compounding prediction error (MC-check 0.281, and four Dreamer critic
variants plus PPO all landed on the same ~0.27-0.28 wall). **Option 2 never rolls the FTM at all**:
real physics supplies the next state, the WM supplies only the per-step reward. It sidesteps
exactly the failure that killed F179 and is genuinely untested.

### The ordering, and why it is not the order he gave

**1. B1 motor babble first (his own W15-5 ask) -- cheapest, and it is the only test that can be
graded.** B1 is the one body with **both** a babble option and ground truth, so it is the only place
"do babble candidates work as well as expert-library candidates?" can be asked *and checked*. Run the
existing loop twice on the same goals, changing only the candidate source. If babble candidates hold
up, the bottleneck dissolves and option 2 may be unnecessary; if they do not, that is the evidence
that justifies option 2 rather than following advice on faith.

> **Trap, verified in the repo:** `scripts/dataset/recollect_b1_noisy.py` is **not** babble -- it is
> B1's trained PPO policy plus `--cmd_noise 0.0137`. Using it would reproduce the exact "ground
> truth + noise" flaw P'Nine flagged. A CPG/random collector for B1 has to be written, mirroring
> `collect_gecko_dataset.py`.

**2. Reward-quality gate -- CHECKED (2026-09-11, F195), and it FAILS.** F136 measured the insect
teacher ranking *local perturbations of one action* at 33% against a 50% coin. Rebuilt on B1's own
MuJoCo physics (F136's own script is tied to the insect/teacher-student pipeline and could not be
reused directly), judged in real physics not by the model: **4.2% hit rate against 5.9% chance,
at or below chance at every perturbation size tested (sigma 0.1/0.3/0.5).** Same shape as F136, on
a different body, checkpoint, and physics engine. **Do not build the RL controller (step 3) on
this checkpoint** -- the reward has no local gradient for PPO-style exploration to climb, and
doing so now would very likely reproduce F179's collapse. **F179's whole arc was built on an
unverified version of exactly this assumption and cost weeks; this gate existing and being
checked before step 3, not after, is the fix for that mistake.** What is still open: whether a
different checkpoint, objective, or fitting procedure could produce a locally-discriminative
reward -- not attempted yet, and the natural next question rather than proceeding to step 3.

**3. RL controller + WM reward** (W15-2) -- **blocked**, step 2 failed. Do not start this until a
reward is found that clears step 2's gate.

**4. History-state ablation** (W15-4) -- runs independently of the above, and would improve the
reward in (3) if it works. See the contradiction note below before scoping it.

### The contradiction that has to be resolved before either is written up

The Week 15 notes say history-state "clearly improved" prediction. `FINDINGS.md`'s sequence-context
kill-gates say all four architectures **failed**. Both readings come from the same numbers:

| architecture | gap | vs stateless baseline 0.042 | vs pre-registered bar 0.110 |
|---|---|---|---|
| R0: mean-pool -> GRU | 0.036 | worse | fail |
| full token grid + self-attention | 0.048 | better | fail |
| attention-pool -> GRU | 0.058 | better | fail |
| **ConvGRU (spatial + recurrent)** | **0.069** | **1.6x** | fail, **by the smallest margin** |

**And `FINDINGS.md` states this backwards**: it calls ConvGRU *"the widest margin from the bar of
any variant tried, not the closest"* when 0.069 is the **closest** of the four (0.041 from the bar).
The "ruled out, not merely unconfirmed" verdict rests on that inverted sentence. **What is not in
dispute: every kill-gate was a 2,000-iteration probe on one body, not a full pretrain** -- so
whether a properly-resourced sequence model clears 0.110 is genuinely unknown, and the ablation the
Week 15 notes propose is a real open question rather than a reopened dead one.

### Status as of 2026-09-11: step 1 (B1 motor babble) is substantially done, three real sub-problems found, next is a clean redo not more patching

**A genuine, no-policy CPG babble source for B1 now exists and covers all three families** (F190) --
`sim/collect/collect_b1_cpg_babble.py`, forward/lateral solid (12/12, 12/12 on `batch2`), yaw real
but weaker (5/12 whole-clip-dominant, the rest a real, acknowledged residual from a genuinely narrow
gait-degeneracy fight, not swept under the rug). This closes the "does a babble collector even exist
for B1" half of step 1.

**The full 4-stage refit + hexapod rehearsal (F191) gives an honest, mixed verdict, not a clean
pass**: forward fixed (100%), lateral real-but-partial (4/4 broken goal-reads -> 3/4, F186's
rehearsal mechanism confirmed necessary a second time), yaw untestable with the current hexapod
goal set (F192 explains why "untestable" and not "0%" is correct -- hexapod's own turning is
forward-dominant, not a spin, at every measured condition). **The lateral goal-reading problem is
real and NOT fixable by more horizon-tuning** (F192's sweep: at or below chance at every horizon 1
to 20) -- it is a weak-but-real, magnitude-compressed visual signal, consistent with this project's
own earlier insect-to-B1 probe transfer numbers (lateral correlation 0.39-0.43 against forward's
0.50-0.63 in every view tested).

**A second mechanism, `free_offset`, was built and tested (F193) and is a SEPARATE axis from
goal-reading, not a fix for it.** It only changes which window of a *candidate's* actions gets
scored -- it cannot repair a goal that was already misread. **Two real bugs were found and fixed in
this mechanism, the second one from a rendered video, not a table** (a total freeze, then a
catastrophic compounding drift that tipped a body onto its back over ~60 steps) -- read F193 for
the numbers and do not quote any free_offset figure from before that entry's correction. Corrected
conclusion: a small, real family-accuracy gain for babble over locked, but still worse than locked
on the continuous distance metric in both goal modes -- opt-in flag on both `DirectFroudePlanner`
and `close_loop_direct_froude.py --free_offset`, off by default.

**Why the next move is a clean redo, not another one-off patch.** Across this arc, two different
"free_offset=False, mode A, expert-fit+expert-candidate" numbers were produced by two different
scripts (42% whole-clip-mean vs 96% windowed) and treated as the same cell before the mismatch was
caught (F193). That is exactly the failure mode a scattered set of one-off diagnostic scripts
produces. **The next experiment is a single, clean, end-to-end 2x2x2, built and run as one
reproducible pipeline (one bash script, one saved log), not assembled after the fact from whatever
scripts happened to exist:**

| axis | values |
|---|---|
| fit/candidate pipeline (never crossed -- expert-fit only scored on expert candidates, babble-fit only on babble candidates) | expert-fit + expert-cand / babble-fit + babble-cand |
| goal source | A (physics, privileged) / D (vision, via ITM) |
| free_offset | False (locked) / True |

All 8 cells measured with the SAME scoring mechanism throughout (windowed, matching what
`DirectFroudePlanner` actually does -- not the older whole-clip-mean convention, which stays a
separate, previously-reported number and must not be mixed into this table). Per-family (fwd/lat/
yaw) breakdown reported for every cell, not just the aggregate, per F192's own lesson about
aggregates hiding a channel-specific failure. The babble-fit leg of this should start from a fresh,
from-scratch 4-stage fit (collection -> render -> stage 1-4 -> rehearsal -> scoring) run as one
script, so the exact data/checkpoint provenance behind every one of the 8 numbers is reproducible
and logged, not reconstructed from memory across several ad-hoc runs the way this session's numbers
were.

**Deferred, not blocking**: whether a future recurrent/persistent-history architecture would need
`free_offset` and the current shuffle-tolerant per-transition fitting convention revisited (F193's
closing note) -- explicitly not in scope until the adjacent-frame (current, memoryless) pipeline is
finished being characterized.

**CLOSED, 2026-09-11: babble is left open, not passed or failed -- and the methodology itself has a
problem beyond the numbers, so no further babble-generation tuning is planned.** Two independent
signals converge:

1. **The numbers are ambiguous and metric-sensitive.** Family-match accuracy gives
   `capture_ratio = 0.60` (clears 0.5); the continuous Froude-distance metric -- the more
   trustworthy of the two, since family-match is a coarse proxy that was shown to actively hide a
   real quality regression from `free_offset` (F193, confirmed on the corrected, bug-fixed numbers
   too) -- gives `capture_ratio = 0.45` (fails 0.5) on the identical cell.
2. **Watching the actual rendered closed-loop video (not the numbers), babble looks visually
   unstable** -- worse in lateral than forward, and `free_offset` does not reliably fix or
   correlate with the visual quality either. Neither the family-accuracy nor the Froude-distance
   metric was ever built to catch this (both only check aggregate speed/direction, never joint
   motion smoothness or stability), so this is a real, independent negative signal the numbers
   cannot see or contradict.
3. **The generation methodology quietly drifted from its own stated premise.** F190's own
   constraint was "generic babble, no prior knowledge, not tuned to B1's known dynamics" (matching
   gecko's/Egocentric VSM's CPG+noise method). In practice, reliably producing three DISTINCT,
   LABELLED behaviour families required designing three SEPARATE, family-targeted mechanisms (a
   strafe hip channel for lateral, a hip differential for yaw) -- injecting knowledge of how
   quadruped locomotion produces different motion directions, even though no B1-specific tuning
   was ever done. **This is a materially weaker claim than "undirected babbling produces usable
   candidates"** -- it is closer to "three hand-designed motion primitives," which is a
   per-body-designed library, not a body-agnostic no-prior-knowledge generator, and doesn't answer
   the question Q21 originally asked.

**Conclusion: stop tuning the babble generator further** (three iterations already tried -- CPG
redesign F190, margin-fix v2, free_offset bugfix -- each ambiguous or negative) **and do not
proceed to the reward-quality gate / RL controller work on the strength of the old 0.60 number.**
Re-opening babble later would need either a genuinely undirected generation method (no
family-targeted mechanisms at all) or an honest reframing of the claim to "designed motion
primitives," not a resumption of parameter tuning.

**Follow-up, same day: root-caused the D-mode gap as candidate-pool margin (expert 0% cross-family
nearest-neighbour risk vs babble 19%), then tried and FAILED to fix it (v2 attempt, F194).**
Pushing lateral's amplitude for a better margin measured in true/label Froude space did not move
the real number at all -- margin has to be checked through a fitted checkpoint's PREDICTED Froude,
not raw motion (the projector/body_head does not preserve true-space distances linearly). This is
now a corrected standing rule (see memory), not resolved -- fixing the margin for real remains
open and would need a margin-in-predicted-space-aware collection loop, not attempted yet.

**Pre-registered bar for `scripts/run/b1_babble_clean_redo.sh`, before running (2026-09-11).**
**Headline (this IS step 1's answer, read on its own, first)**: babble substitutes for the teacher
library if `(babble, goal=A, free_offset=False)` >= `(expert, goal=A, free_offset=False)` AND
clears `pool_chance()`, both numbers from this same script/run -- not F184's old whole-clip-mean
42%, which is a different convention (F193). A good result in any OTHER cell does not substitute
for this comparison. **Secondary reads, informative but not the headline**: (a) does
`free_offset=True` help babble more than it helps expert -- the pre-registered test of the
babble-messiness hypothesis (a messier candidate pool may need the free-offset search more than a
clean one does); if babble only matches expert WITH free_offset, that is "babble needs free_offset
to compete," a weaker claim than "babble replaces teacher," and must be reported as such, not
folded into the headline. (b) does the vision goal (D) hold up against the physics goal (A) on
each pipeline. When results land: report the full 8-cell table, but state the headline verdict
from its own cell explicitly, separate from the two secondary reads.

---

## Q20. Gecko's remaining gap is the video, not the method -- is a body this slow inside the claim's scope? (2026-09-10, rewritten same day after F189)

**This question's original three candidates (adaptation budget / yaw telemetry / physics noise) are
all void.** They were reasoning from two measurement bugs, both since found and fixed (F189): the
stage-1 "gate" was read with the sign inverted, and gecko's body frame was built on a near-vertical
axis that scrambled forward, lateral and yaw at once. What is left is smaller and much better
located.

**Fixed, and their effect measured:**

| fix | effect |
|---|---|
| body frame `-(body y)`, not body x | yaw ceiling **-0.042 (dead) -> +0.501**; heading jumps >90 deg/step 16.8% -> **0%** |
| projector sees 20 action frames, not 1 | stage-4 held-out **1.016 -> 0.970** (below 1.0 for the first time), projector-path rho **0.032 -> 0.249** |

**The chain, measured end to end by ridge on ground truth:** gecko's actions carry **0.736**, its
egocentric video carries **0.374**, the pipeline delivers **0.249** (67% of what the video allows).
B1's video carries 0.747. **The video is the binding constraint and the cause is physical**: gecko's
forward Froude is **0.038-0.048** against B1's 0.126, and it is flat across gait frequencies 2-6 Hz,
so the robot barely translates between frames while its legs fill the view.

**The actual open question, and it is a scoping decision rather than a debugging one:**

1. **Make gecko genuinely faster.** Frequency is ruled out by measurement. Stride length, joint
   gains, or scene scale are untested. If gecko can be brought near B1's Froude, the video ceiling
   should rise with it -- that is the prediction to pre-register. **Cheapest real test.**
2. **Accept the ceiling and report gecko as a bounded case.** "The method grounds a novel body when
   that body's motion is visible to its own camera; a body 2.6x slower than the reference sits below
   that threshold" is a defensible, measured claim -- and it is a claim about the SENSOR, not about
   the shared coordinate, which is the thesis's actual subject.
3. ~~**Re-run the full pipeline on the corrected frame.**~~ **Done** (`wm/runs/gecko_fixed`):
   stage 1 improves at every horizon for the first time, stage 4 lands at **1.004** with the K=1
   projector the pipeline currently ships. **The frame fix alone does not rescue stage 4** -- the
   windowed projector is what takes it to 0.970. Neither fix is sufficient alone, and together they
   reach 0.970 against B1's 0.751, with the video ceiling (0.374) accounting for the remainder.
   **The decision this leaves is whether to wire windowing into the real projector**, which touches
   the validated B1 path (F183-F188) and so has not been done unilaterally.

**Do not** re-collect babble for coverage: `babble_v3` was built for a frequency lever that does not
exist, and it is measurably the WORSE dataset (action ceiling +0.146 vs the original's +0.736),
because constant-gait clips remove the within-clip noise the mapping is actually fitted on.

---

## Q19. Distill the vision-based teacher into a proprioception-only student (new, 2026-09-09 — scoped, not started)

Deployment step, downstream of F183-F188 (the direct-Froude controller now validated cross-
embodiment, live, closed-loop). Not the same thing as the "proprioception cannot do this" claim
already settled below (that's about the *comparison*; this is about *deploying* the working
vision-based controller cheaply). Not the same thing as `sim/control/teacher_student_insect.py`'s
existing `Student` either -- that one still takes a pooled VISION embedding as input, so it removes
the planning loop but not the camera. This asks for a student with no camera in the loop at all.

**Scope:**
1. **Teacher**: the validated direct-Froude controller (`wm/runs/b1_adapt/body_head_b1_hex.pt`,
   mode A/D) -- the only mechanism this project has shown actually works (F188).
2. **Student input**: proprioception only -- `joint_pos`/`joint_vel` (already recorded), plus base
   orientation/angular velocity if available. No vision embedding anywhere in the student's forward
   pass -- a new architecture, not a reuse of `teacher_student_insect.py`'s `Student`.
3. **Goal handling**: compute the goal once (vision or physics, matching mode A/D's own "fixed for
   the whole episode" design), pass it as a fixed vector alongside proprioception every step. Vision
   is allowed once, offline, before deployment -- never per-step.
4. **Data**: roll the teacher's own closed loop (F185-F188's mechanism) across many episodes and
   goals, recording `(proprioceptive state, teacher's chosen action)` pairs as behaviour-cloning
   targets.
5. **Training**: supervised regression, student mimics the teacher's picks -- same spirit as
   `teacher_student_insect.py`'s `clone()`, new student class taking proprioception instead of
   pooled embeddings.
6. **Evaluation**: same S.R. metrics as F188 (survival / behaviour class / speed within 15%),
   student vs. teacher, teacher's F188 numbers as the ceiling.

No blocker identified -- working teacher, closed-loop scoring infra, and an architecturally-close
(if wrong-input) `Student` to fork from all already exist. Not started.

## Q12. Which bodies belong in the dataset at all? (new, 2026-08-11 — blocking)

Measured in **F36**: two of the nine bodies in `data/allocentric/fwd_hex8body` do not walk — one moves 0.057 m
in an episode, the other **walks backwards** — and two more crab sideways 2 to 6 times more than
any sound body. Every Stage 2 run globs the whole directory, so about a fifth of the hexapod
gradient went to a robot that does not locomote.

The cause is the reach limit: a two-link leg cannot get closer to its shoulder than
`|femur − tibia|`, and the closest commanded target is at 92.5 mm. Ratio alone is harmless —
bodies at 1.04–1.10 with dead zones of 12–26 mm walk normally.

Open, in the order they have to be decided:

1. ~~**Exclude the two non-walking bodies.**~~ **Done** — `EXCLUDED_BODIES` in
   `wm/data/dataset.py`, applied wherever clips are globbed. Stage 2's hexapod pairs drop from
   15,755 to 12,285. The reruns are the remaining work.
2. **Keep the two 94.6 mm veering bodies — decided, keep.** They walk; the gait differs but it is
   locomotion, not a collapse. Not raised in the current deck. Revisit only if there is time to
   retrain, and note that dropping them would leave three bodies all at ratio 0.83, which ties
   femur to tibia perfectly and destroys the very coverage slide 8 is about.
3. **How to get femur/tibia diversity without leaving the sound range.** The coxa is the answer:
   it positions the shoulder without entering `|femur − tibia|`, and it is behaviourally almost
   free — ARI +0.038 with the gait split, and a 40% coxa change leaves contact patterns agreeing
   at 0.984. Rescaling the foot trajectory per body would work geometrically and **must not be
   done**: `lambda_cross` is well defined only because every body walks identical expert episodes,
   so per-body targets turn a shared intent into a wrong label.
4. ~~**Whether ratio > 1 is admissible at all.**~~ **Resolved, already logged**:
   `sim/scene/make_leg_morphology.py`'s docstring states it directly -- "femur longer than tibia
   inverts the animal's own proportion, so a body above 1.0 is a robot morphology, not a stick
   insect." Ratio > 1 is not geometrically broken (bodies at 1.04-1.10 walk normally, per the
   dead-zone measurement in the same docstring), it is just not a stick insect anymore. If the
   thesis frames this as stick insects specifically, the honest range is ratio ≤ 1.

**Resolved 2026-08-12.** Items 1 and 3 are done and `stage2_clean` has been trained and measured
on two seeds — see F37. The data questions are closed; what they uncovered is not:

**Q13. RESOLVED 2026-08-12: drop it.** The cross-embodiment variance decomposition is not
under-sampled, it is built on a phase label too coarse to be one. Stance fraction takes **8
distinct values** across both embodiments, dominated by 0.5, so quantile edges collapse: asking
for 3 bins gives 2, asking for 4 gives 2 (identical numbers), asking for 6 gives 3. The grid was
never 2 x 6 x 6 = 72 cells; it was 24 to 36. The embodiment share therefore reads **32.0% at three
bins and 12.0% at six**, same checkpoint, same data -- a 2.7x swing from a parameter that was
supposed to be cosmetic.

**F32's headline 33.0% came from this measurement.** Replace it everywhere with the probe (0.994 /
0.992 across seeds) and the identity ablation (1.03x / 1.04x against a random control), which
reproduce to three decimals and say something stronger anyway: the identity is fully present and
nothing uses it.

Stage 1's `z_body_share` is not affected -- insect bodies walk identical expert episodes, so its
grid can use the timestep directly instead of inventing a shared phase label.

*Original question below, kept for the reasoning.*

**Q13 (as asked). Is the variance decomposition salvageable, or should it be dropped?**

`two_way` balances its grid to the smallest cell, which holds six latents, so the whole
measurement rests on 72 points. Two seeds of one config give **12.0% and 6.7%** for the embodiment
share. F32's headline 33.0% rested on the same 72 points and is in the deck.

- `--bins 3` doubles the latents per cell. Does that make the seeds agree? One command, decides
  whether the measurement is under-sampled or unusable.
- If it stays unstable, every claim moves to the probe (0.994 / 0.992) and the ablation
  (1.03x / 1.04x), which reproduce to three decimals. That is a stronger claim anyway — presence
  plus non-use, rather than a share of variance.
- The same question applies to `z_body_share`'s Stage 1 numbers, which use the same machinery.

**Numbers stay as they are until the clean retrain.** Deliberate: the deck keeps its current
figures rather than being patched twice. What is known is the *direction*, and it is favourable
everywhere it has been checked — the veering bodies were making our own claims look worse, not
better:

| measured on | with the veering bodies | sound bodies only |
|---|---|---|
| body share in `z`, control | 11.3% | **5.8%** |
| body share in `z`, cross loss | 1.2% | **0.2%** |
| gait phase share, cross loss | 88.6% | **94.7%** |
| held-out error, `tib_cross` | 27.8 deg, R² −3.16 | **11–13 deg, R² −0.42 to −1.07** |

So cleaning the data should *strengthen* Stage 1's claims and only Stage 2's absolute numbers are
at risk. Re-measure after the retrain rather than editing figures now.

**Loose end to settle at the same time**: `z_content.py` reports the control's body share as 8.8%
and `z_body_share.py` reports 11.3% for the same checkpoint, while both agree on 1.2% for the
cross-loss run. Two of our own scripts disagree by 28% on a figure that is in the deck. Trace it
when the numbers are being redone, not before.

## Q0. What Stage 2 can and cannot claim, given Stage 1 (new, 2026-08-09)

Stage 1 found that the decoder identifies the body from a code in `z` and looks up, rather than
inferring morphology from the frame (FINDINGS F16-F19). Two claims were being run together and
have to be separated, because Stage 1 supports one and predicts the other will fail.

**Claim A — vision forms a shared model across incomparable joint spaces, proprioception cannot.**
Survives, and Stage 1 supports it. One model reconstructs five bodies to 0.5 deg with morphology
never supplied. This is a statement about a shared representation existing, not about
generalisation. An 18-DOF hexapod and a 12-DOF quadruped cannot be fed to one proprioceptive
model at all, so the asymmetry does not depend on transfer succeeding.

**Claim B — that model transfers to an unseen embodiment.** Stage 1 predicts failure. Training on
hexapod + B1 and testing on a 4-leg insect is two training points, which is the configuration F5
and F15 show does not work, and a third embodiment cannot be generated the way extra bodies were.

**Step 3's sample-efficiency framing is not claim B.** Pretrain, fine-tune on N clips of the new
embodiment, compare against from-scratch: the shared backbone carries gait phase and visual
processing, so it can start ahead even when zero-shot fails. Untested and not contradicted.

**Stage 2 has now been run once, and the premise did not hold on its own (F32).** One shared
trunk across the hexapod and the B1, per-embodiment heads, no cross-embodiment term -- which is
what the source method specifies. The latent came out **33.0% embodiment identity** against 39.6%
gait phase, with embodiment decodable at 1.000. For comparison, `lambda_cross` holds the *body*
share at 0.8-1.2% within the insect family.

Training did pull the two together: silhouette **+0.671 to +0.140**, cluster separation
**4.01x to 0.77x**. That is a large real compression and it is visible in the projection. **But
the latent still separates into two clean clusters in the UMAP and the probe is still 1.000** --
so weight sharing alone gets most of the way and does not finish. The question below is no longer
a prediction; it has a number.

**The mechanism that worked in Stage 1 does not port to Stage 2 as written.** `lambda_cross`
decodes body A's latent against body B's frame supervised by B's command, and it is well defined
only because every insect body walks the same expert episodes: `dataset.py` pairs on
`clip["episode"]`, so at a given timestep two bodies share the intent exactly and differ only in
geometry. **The hexapod and B1 share no episodes.** B1's clips come from MuJoCo rollouts under
different policies, so `self.partners` would be empty and `L_cross` would never be computed.

Without it, Stage 2 gets `lag1_ctrl`'s behaviour: body identity decodable from `z` at 0.638
against 0.470 with the term on, the decoder leaning on `z` rather than the frame (z-gap 11.3x
against x-gap 3.6x), and 1.65x worse reconstruction. **`z` becoming an embodiment code is exactly
what Stage 2 must not allow**, since that code has no entry for a third embodiment.

Pairing does not actually require shared episodes -- it requires knowing that two frames show the
same intent. Two measurable stands-in, both available:

| pair on | needs | cost |
|---|---|---|
| body velocity alone | simulator positions | phase mismatched, so the target command is wrong |
| velocity + gait phase from foot forces | force sensors, at training time only | most accurate; privileged data |
| **velocity + gait phase estimated from the frame** | **nothing extra** | phase wrong ~18% of the time |

Force sensors would be used to **build the pairs**, never as model input, so vision-only inference
is unaffected -- the same standing as the ground-truth commands already used as targets. And the
third row is available regardless: one frame identifies which feet are swinging at 0.815 against a
chance of 0.5 (F26).

**The real risk is not the sensor, it is that mis-paired frames are wrong labels**, not merely
noisy ones. Stage 1's pairing is exact; part of why `L_cross` works may be that exactness. Across
embodiments no pairing can be exact, and what counts as "the same phase" for a six-leg tripod and
a four-leg trot has no physically correct answer -- it is a design decision that has to be stated
and defended. ~~**This is the largest untested risk in Stage 2 and there is currently no plan for
it.**~~

**MEASURED 2026-08-14, and the answer is that none of the three rows works on current data (F39).**
The risk above is no longer untested. Per-leg contact was the most promising signal -- no shared
period needed, corner legs correspond anatomically -- and it fails in a specific, informative way:

| label | overlap | hexapod frames pairable | intent, hexapod | intent, b1 |
|---|---|---|---|---|
| `n_feet_down` | 0.572 | 98.9% | 0.913 | **0.998** |
| `diagonal` | 0.711 | 100% | **0.918** | 0.605 |
| `corner_pattern`, 16-way | **0.240** | **33.8%** | 0.630 | 0.524 |

`intent` = matched-pair over random-pair command distance within one body; 1.0 means the label
says nothing. The fine label means something on both robots but pairs only a third of the hexapod;
the coarse labels pair everything and mean nothing on one side. **Coverage and meaning trade
directly**, because coarsening is what destroys the meaning. The B1 spends 84.6% of its time in
the two trot diagonals while the hexapod spreads over all sixteen patterns, nine of them
hexapod-only.

Velocity-alone, the first row of the table above, is worse still and needed no new measurement:
the hexapod walks **one speed, forward only**, so velocity has no variation to pair *on* from the
insect side at all.

**So the plan is no longer "pick a pairing and defend it".** Either (a) accept that Stage 2
follows the paper without a cross term -- which Q11 notes is what the source method actually does,
making `lambda_cross` our addition rather than a missing piece -- or (b) broaden behavioural
coverage first, since the overlap failure is partly the one-gait-one-speed constraint (F26)
rather than a fact about hexapods and quadrupeds. Option (b) is the AMP-dataset question already
open in Q11, and F39 is now a second, independent reason to take it seriously.

Practical consequence: report claim A as the result, claim B as a measured limit with its
mechanism, and treat sample efficiency as the transfer claim actually being made.


---

## Q11. What else differs from the source paper, and does it matter? (open)

Read against the paper, the reimplementation matches on the things that define the method --
frozen V-JEPA2, the ITM/FTM/MD decomposition, cross-augmentation and its stated purpose, and,
after the `action_lag` correction, the action's time index. Four differences remain:

| | paper | ours | worth acting on |
|---|---|---|---|
| **action chunking** | actions grouped into **5-step** sequences, stated to improve world-model learning | one step | **yes, and now quantified** -- F27: widening the gap to five steps nearly doubles the reconstruction target's real signal, and combined with dropping the crop it moves the signal-to-noise ratio from 0.24x to 0.89x |
| latent dimension | 512 | 64 | maybe; ours is 8x tighter |
| module size | ITM 47M, FTM 94M | about 5M each | probably not at this data scale |
| behavioural diversity | 3 datasets, 150k trajectories, 22 object categories, a deliberate left-or-right choice in the task, 80 percent failures | one gait, one speed, forward only | this is the F26 constraint, restated |

**One difference removes a risk rather than adding one.** The paper has **no cross-embodiment
pairing term**: the shared latent space emerges from sharing the ITM, FTM and MD weights across
embodiments. So `lambda_cross` is **our addition**, and Stage 2 can follow the paper without
solving the pairing problem in Q0.

**Their setting probably does not need it, and that is the point.** The shortcut we measured is
"recognise which body this is and recall its commands", and it only pays when knowing the body
tells you the command. In our data each body does exactly one thing, so body identity is nearly
the whole answer. In theirs, one embodiment performs thousands of different manipulations across
22 object categories, with a deliberate left-or-right choice and 80 percent failures, so knowing
it is a Franka arm says almost nothing about what to do next -- the shortcut buys little and the
model has to read the scene regardless. Their Motion Decoder is also auxiliary rather than the
system's output, so a shortcut there costs them less than it costs us.

**We therefore cannot claim their method has this problem, and should not.** We have not run in
their regime. What we can say is scoped: LAC-WM is tested across embodiments that differ radically
with behaviourally rich data; applied to **cross-morphology** -- bodies that differ slightly, one
behaviour -- the auxiliary motion loss admits a shortcut that defeats transfer, and `lambda_cross`
closes it. That is a regime the paper does not test, so this is an extension, not a correction.

**The concrete proposal this points to**, to put to the professor rather than decide alone: rebuild
the main experiment with a five-step gap and photometric jitter only, which is the first setting in
which the forward model's target is mostly signal rather than augmentation noise (F27). Cost: the
Motion Decoder outputs 5 x 18 = 90 dimensions instead of 18, every number becomes incomparable with
the runs recorded so far, and the copying shortcut the augmentation was there to block has to be
re-measured rather than assumed away. That is a rebuild of Stage 1's main comparison, so it should
be decided before Stage 2 starts, not during.

**And the paper's transfer is not zero-shot.** Adapting to the unseen embodiment is a three-stage
LoRA finetune on 7,265 trajectories of the target robot. Q0's claim B, tested as zero-shot, is
stricter than what the method claims. The sample-efficiency framing in Step 3 is the comparable one.


---

## Q14. Does behavioural overlap make a channel shareable? (open — three runs deciding it)

The lever list of 2026-08-15 has been worked through and only one item is still standing:
**make the two robots' behaviour distributions overlap**. Shared supervision is blocked by F39 (no
usable frame pairing), architecture was measured to sharpen per-robot codes rather than share them
(F37, F40), three invariance methods moved nothing (F38), and leg-removal bodies read as the body
they were cut from (F41).

**Overlap in speed alone was tested and did not work.** Five matched speeds gave cross-embodiment
readouts of -4.16 and -5.60; more diversity gave the trunk more to partition by (F50, F53).

**Overlap across three behaviours now exists** and the untrained answer is still no: on the frozen
encoder, forward transfers at **+0.36 +/- 0.10** and lateral and yaw sit at zero (F67). But the
frozen encoder is the *before* condition, and forward speed itself reads **0.31 frozen against
0.85-0.92 trained** (F59) -- so a frozen zero does not decide the question (F68).

**What is open**, in the order it gets answered:

1. **Does training on yaw make it transfer?** Three arms running: control, body head forward-only,
   body head forward+yaw. The middle arm is what makes the third attributable to the channel rather
   than to the new dataset.
2. ~~**Does F59's 0.85-0.92 survive the frame-rate fix?**~~ **Settled, and re-asking it is a trap.**
   The runs give +0.701 / +0.667 by clip and +0.610 / +0.573 by condition, at the top of the old
   range and under a harder test -- but **the two datasets are not comparable** and three attempts to
   force it each found a different confound (F74). The claim rests on the controlled within-dataset
   comparison instead: -16.7 to +0.70 from the body term alone.
3. **Why do the channels compete?** Adding yaw costs forward 68% (F73). Three explanations were
   tested and rejected -- yaw carrying less signal (identical signal share, 0.86 both), a mismatched
   length scale (an affine rescale cancels against a standardised target, F68), and a longer
   smoothing window (degrades monotonically, F73). **Capacity is the remaining candidate and the
   cheapest test is one training run with a wider body head**, data held fixed.
4. **Is twelve behaviours enough to resolve effects of this size?** Held out by condition, about
   four test behaviours remain and the spreads run +/- 0.2 to 1.3. If the arms disagree weakly the
   answer may be power rather than substance.
5. **Should the yaw length scale be the stance radius rather than hip height?** Physically the
   moment arm of a turn is where the feet meet the ground, and the two scales differ 4.4x in the
   ratio between the robots. It does not change what transfers -- an affine rescale cancels against
   a standardised target -- but it does change how much the channel identifies the robot, 0.637
   against 0.571 (F68). Switching means re-solving the four `--spin` levels first, since the
   collection is matched on the height version.
6. **Does pretraining on two embodiments make a third one cheap?** Unanswerable as things stand and
   the most valuable thing left. LAC-WM's scaling result -- downstream performance rising with the
   number of pretraining embodiments -- needs at least three, and we have two, so it is declared as
   a limitation rather than attempted (F99, step 2o). A third body chosen for **incomparable
   topology** (a biped, or a different leg count) rather than for convenience would turn the claim
   from "these two robots transfer" into "these two are pretraining data". Costs what the B1 cost.
7. **Is lateral permanently out of the target?** It fails the robot gate at 0.68 even with the
   frame corrected, and half the B1 clips carry a per-policy lateral artefact (F70, F71). Excluded
   for now; the exclusion is a measurement, not a principle.

---

## Q1. Which cross-embodiment framing? (the main open choice)

- **(A) 6-leg → B1** — feasible **now** with data in hand. Pretrain hexapod, test
  transfer to B1 (or vice-versa). Clean incomparable proof (18-D vs 12-D). 1→1 transfer.
- **(B) Compositional: {6-leg + B1} → 4-leg insect** — *better story.* Train on two
  incomparable topologies (proprioception-can't is baked into the training set), then
  test transfer to a **4-leg stick insect**, which shares *appearance* with the hexapod
  and *leg-count* with B1 → tests whether the model **composes** them. Ablation
  (train 6-leg-only / B1-only / both) shows the composition explicitly.

**Lean:** (B) is the headline if we can produce a 4-leg walker (see Q2); (A) is the
guaranteed-feasible fallback and a good first result. Likely do (A) first, then (B).

**Update 2026-08-14 (F41): (B) was built, but the body chosen does not test composition.**
The 4-leg is the *base* insect with the middle legs removed, so its geometry is `c10f10t10`'s --
a training body's -- and its commands are that body's corner columns bit-identically. The latent
sits 0.578 from the base body's against a chance of 0.981, so the model reads it as the base body
at that phase and barely registers the missing legs. Leg count is the only novel axis; the
compositional claim needs at least two.

**Which axes a held-out embodiment can be novel on**, and where the current one stands:

| axis | hexapod -> B1 | hexapod -> 4-leg, as built | fix |
|---|---|---|---|
| action dimensionality | 18 vs 12 | 18 vs 12, **novel** | -- |
| leg count | 6 vs 4 | 6 vs 4, **novel** | -- |
| segment geometry | n/a | **in distribution** | remove legs from a *held-out* body |
| gait / dynamics | wave vs trot | unchanged insect wave | needs a 4-leg controller (Q2) |
| appearance | insect vs quadruped | still a stick insect | a second quadruped, new assets |

**Cheapest correct fix, one collection run and no new tooling**: ghost-remove the middle legs from
`c08f09t09`, already withheld from Stage 2 training, so geometry and leg count are both unseen.
Only middle-loss walks (front-loss tips, hind-loss rears, F38), so the variant is forced, and the
body must be rendered before collecting -- a geometry change can break a gait that worked on the
base scene.

**DONE 2026-08-14 (F41).** `data/ik_4leg_c08f09t09_clean10`, 10 clips from a 30-episode sweep.
The margin is **unchanged**: 2.85x against the base body's 2.86x (1.91 +/- 0.08 deg against a
random backbone's 5.45 +/- 0.16). Geometry and leg count are now both novel and the claim holds.
**Rows 1-3 of the table above are satisfied; rows 4 and 5 -- gait and appearance -- are not.**

**Leg-loss itself is not the problem** and should not be abandoned: it produces a genuinely
incomparable 12-D action space against the hexapod's 18-D, which is the asymmetry the thesis rests
on. What was wrong is the *geometry* it was built from, and that is a one-line change to which
scene the collector loads.

**A stronger version, if there is time**: make the held-out body quadruped-like in *behaviour* as
well as topology, so it shares appearance with the insect and gait with the B1 -- which is what
"composition" was supposed to mean. That needs a 4-leg controller rather than the unchanged
six-leg IK gait, which is Q2's open item. Strongest of all would be a second real quadruped, but
that is new assets and a new policy, not a collection run.


---

## Q2. The 4-leg walker — we build our own (no dependency on yuchen)

We do **not** need yuchen's `cutlegs` policy. The world model only needs 4-leg **frames
+ command**, not that policy's internal obs. The 6-leg CSV gait does **not** propel a
4-leg body (tested: 0.000 m — front-leg removal breaks propulsion), so we need *a*
4-leg controller — and we make it ourselves with a config we own:
- **Retrain via PPO** (`train_ppo.py`, reward = forward velocity) — **no expert demos
  needed** (unlike AIRL), so the missing `expert_cutlegs.csv` is irrelevant. We choose
  the obs and which legs. No config-drift.
- Or a hand-tuned / CPG 4-leg gait.

**Design decision:** likely cut the **front leg pair** (leave middle+hind) so the body
reads clearly as a quadruped → strongest "insect + quadruped → 4-leg insect" composition.


---

## Q3. 6-leg controller: CSV gait vs policy

`hexapod_v1` uses the **CSV gait** (walks properly, ready). Driving it with the AIRL
policy for "consistency" is **parked**: the AIRL policies aren't faithfully runnable
here — their obs *normalization* config is drifted from the trained weights (tried all
3 candidate obs fields; all give a stationary stance). For the vision dataset the CSV
gait is fully valid (V-JEPA2 sees a hexapod walking either way). **Lean: keep CSV.**


---

## Q4. To confirm / minor

- Metric = **reconstruction-loss sample efficiency (no policy)**.
- Data volume: current clip counts are a start; may scale the command sweeps.
- Writing caveats (Tee): single-step Markov is deliberate; which modules fine-tune on a
  new body; large-model fine-tuning/scaling limitation.


---

## Q16. Dreamer or candidate scoring — the pipeline has to commit (new, Week 13 — blocking)

**The advisor's framing.** We are sitting between two designs and should pick the one that spends
its effort where it pays.

| | what the world model does | what it needs |
|---|---|---|
| **Dreamer-style** | imagines rollouts to train an actor-critic, cutting sample complexity | long-horizon rollout accuracy |
| **candidate scoring** | scores actions someone else proposed | **an answer to where the candidates come from** |

**The second question is the one that bites, and it bit twice independently.** The advisor asked it
as *"if the policy generating candidates is already an expert, why score it at all, and would we
not have to train one policy per morphology?"* We reached the same place from the data: our
candidate library is twelve recorded behaviours of the target robot, so **something already made
that robot walk, turn and strafe** -- which contradicts the premise that the robot is unknown.

**The source paper does not escape this either.** LAC-WM §6: *"random action sequence sampling...
is inefficient for action optimization, especially for a difficult dexterous manipulation task"*,
so they draw **N=500 candidates from a pretrained VLA** by injecting noise seeds into its
flow-matching head. **Their prior is a policy; ours is a set of clips. Both have to come from
somewhere.** That also answers the advisor's second question -- diversity from a fixed network
comes from noise injection, not from the network being stochastic.

**What the 2026-08-26/27 measurements say about the choice.**

| | candidate scoring, as built | what distillation changes |
|---|---|---|
| library | required, and circular | not needed -- the policy emits continuous joint targets |
| run-time cost | twelve forward-model rolls per step | one forward pass |
| distribution | scored on recorded states, deployed on self-driven ones | trained on the states it reaches |
| camera at run time | required | not required -- the student reads proprioception |

**The world model is not the part in doubt.** Deleting the rollout from the scoring rule costs
**24 points** of selection accuracy and lands within five of not using the goal at all (F90), so
it is predicting rather than pattern-matching either way.

**Decided 2026-08-28: teacher-student, and the reason is not efficiency.** Sampling `(vx, vy, wz)`
from the B1's locomotion policy was proposed as a way to replace the twelve curated clips. It does
not rescue the claim -- **it makes the pipeline identical to LAC-WM's**, which is the thing this
project is supposed to improve on, and it still means a policy that already walks, turns and
strafes exists for the "unknown" robot.

**The distinction that settles it:**

| | what it requires of the target robot |
|---|---|
| scoring candidates | **a policy that already performs the behaviours.** If no candidate turns, no scoring rule can produce a turn |
| learning a policy | **only the ability to actuate.** Data where the robot falls is still data |

**So the way to stop depending on a policy is to emit one, not to find a better one to sample
from.** That also changes what has to be proven: not "what fraction of steps picked the right
candidate", but **whether a robot the world model never saw ends up with a usable controller from
video alone** -- which is a deliverable LAC-WM does not produce, since it stops at selecting among
what its VLA proposed.

**Start on the hexapod**: the approach leans on long-horizon rollout, this project's weakest
measurement -- the hexapod forward model holds to half a second, the B1's does not. Motor babbling
stops being a prerequisite under this design, because the policy is trained rather than selected
from; it remains the experiment that would let the *candidate-scoring* results claim an unknown
robot. See Q18 for why the same reasoning does not let us call turning and sideways failures.

---

## Q17. What is the research gap, in one sentence (new, Week 13 — the most important item)

**The advisor's homework, stated as the thing to settle before building anything else.** What gap
does this close, and how is the pipeline different from what exists?

**Draft, for confirmation rather than as a settled answer.**

> LAC-WM demonstrates cross-embodiment latent actions in a setting where **a shared task space
> already exists**: end-effector pose is meaningful for every arm in their corpus, so the alignment
> problem is answered by the choice of coordinate rather than by the method. **Locomotion across
> leg counts has no such space** -- 18 and 12 joint targets share no dimension, and no coordinate
> choice creates one without a kinematic model per robot, which is the cost the project exists to
> avoid.

**Two consequences follow that are not in the source work, and both are measured.**

| | measured |
|---|---|
| **(a)** a joint-space action target crosses incomparable embodiments **only when a shared body-motion term is present** | cross-robot speed readout **-28.9 -> +0.61**, and per-robot joint error improves at the same time, 0.3517 -> 0.2183 |
| **(b)** MSE adaptation is **insufficient across families**: the forward model improves its predictions while discarding the action channel entirely. A discriminative term is required | quadruped behaviour selection **19% -> 57%** against a 28% chance rate, everything else held fixed |

**(b) was found rather than sought**, and it is the answer to Q7 -- the objective was what was left.

**What weakens the claim, and should be said before it is asked.** The candidate library is
circular (Q16). Speed is not controlled on either robot. And the scaling result LAC-WM leads with
-- performance rising with the number of pretraining embodiments -- **cannot be reproduced with
two.**

### The axis the comparison should be made on

**"Nobody did it exactly this way" is not a defence.** A difference only counts if it changes what
the method has to be *told* about a new robot. Proposed axis, to be agreed with the advisor before
the literature is read against it:

| method family | what it must be given about the new robot |
|---|---|
| graph-based / modular policies | the kinematic tree -- which joint connects to which |
| universal-morphology policies | a URDF or equivalent body description |
| task-space latent actions (LAC-WM) | a shared coordinate that already means the same thing on both robots |
| **ours, as claimed** | **video, and the joint count** |
| **ours, as it actually stands** | video, the joint count, **and twelve clips of the robot already walking, turning and strafing** |

**The last two rows are not the same, and the gap between them is the honest state of the work.**
Anyone who knows this literature will find it immediately: a controller that can already produce
those twelve behaviours is *more* than a URDF, not less. **Motor babbling is not an extension --
it is what makes the fourth row true.**

### Three novelty checks, unverified — IOON to confirm

Believed clear as of Week 13 and **not yet checked against the literature.** Written down so the
check is against a fixed list rather than a memory of a conversation.

| | what to establish | what it costs us if it is already done |
|---|---|---|
| **1** | Do existing cross-morphology **locomotion** methods all require a kinematic graph or URDF? | If any works from video alone, the axis above stops separating us and the positioning has to change |
| **2** | Has anyone learned a latent action space **across different leg counts** from video? | If yes, read how they handled the action space -- that is the whole problem here, and their answer either supersedes ours or contrasts with it |
| **3** | Is a **contrastive term in an action-conditioned world model** already published? | Least damaging. The contribution would narrow from *the fix* to *the finding* -- that MSE adaptation silently discards the action channel across morphology families, invisible in its own loss curve. **That finding survives even if InfoNCE here is standard** |

**Check 3 is the safe one; 1 and 2 are the ones that decide the framing.** Until they are done, the
gap statement above is a hypothesis with measurements behind it, not a settled claim.

---

## Q18. The three limitations are untried, not established (new, Week 13)

Three things are currently reported as failures. **None of them has been tested where it would
actually be decided**, and saying "we cannot do X" before trying the one lever that addresses X is
the wrong claim to put in front of a committee.

| reported as | what was actually tested | what has never been tried |
|---|---|---|
| turning does not cross embodiments | the planner, the library, the scoring, three warm-start settings | **yaw has never been in the training target.** F73 moved it from -5.23 to +0.37 by supervising it, on a different checkpoint family, and that was never carried into a loop |
| sideways is at or below chance | nine hypotheses: library coarseness, score blindness, amplitude, switch rate, gait phase, lock-in, camera angle, condition labels, joint replay | **all nine are on the deployment side.** Lateral has never been in the training target either |
| the candidate library is recorded clips, so "a camera is the only thing it needs" is not earned | -- | **`rollout_b1_mujoco.py` takes `--vx --vy --wz`**, so a B1 library can be sampled without P'Jo's CPG -- but see below: this changes the prior rather than removing it |

**The first two collapse into one run**: stage 2 on `beh12_c10f10t10_flat` + `beh12_b1_flat` with
`body_channels ['0','1','2']` against a `lambda_body 0.0` control, then stage 3 and the loop on
both. It is the run that would carry F73's mechanism into a controller for the first time, and it
answers turning and sideways together. Heavy -- fibo7.

**The third is cheap and runnable here, and it does not do what it first appears to.** Sampling
`(vx, vy, wz)` still goes through the B1's trained locomotion policy, so the robot walks because a
policy makes it walk, not because a camera does. **What changes is the *level* of the prior:**

| prior | what it assumes | exposure |
|---|---|---|
| twelve curated conditions | a person decided what the behaviours *are* | "did you hand it the answer?" |
| random `(vx, vy, wz)` | a velocity-conditioned locomotion policy exists | **the same assumption LAC-WM makes**, which samples 500 candidates from a pretrained VLA |
| random joint-space actions | nothing | not achievable on a legged robot -- it falls immediately, and no published method does this |

So the honest ceiling on "a camera is the only thing it needs" is **a generic prior instead of a
task-specific one**, and the sentence has to be written that way rather than dropped or overclaimed.
**It also relocates Q16**: if a policy already supplies the candidates, the question is not "policy
or Dreamer" but *where in the pipeline the policy sits* -- generating candidates that a world model
selects among, or being trained by the world model and acting alone.

**And on all three, the honest present-tense sentence is "not yet demonstrated", not "does not
transfer"** -- every slide that says otherwise is overclaiming a negative.

## Q15. "How is this different from Diffusion?" (asked Week 11, answered — keep the answer)

Asked in review and answered on a deck slide that has since been cut, so it is kept here. It comes
back every time the work is described to someone new.

**The latent is inferred, not sampled.** A diffusion model starts from noise that is isotropic
Gaussian *by construction* -- structureless by design. Our `z` comes from the ITM given an observed
pair of frames, with no sampling anywhere at inference. The pipeline is deterministic.

Ask what a latent is made of and a diffusion prior answers "nothing, by design". Ours answers:

| | share of `z`'s variance |
|---|---|
| gait phase | **92.6%** |
| which body | **3.4%** |
| interaction | 4.1% |

**The requirement differs in kind.** A diffusion policy is trained for one robot and never has to
satisfy a cross-body constraint. Ours must decode to *different joint values for different bodies
from the same latent*, which is what `lambda_cross` enforces and the held-out body tests.

**The part of the question that stands, and should be conceded rather than argued.** At the level
of "a conditioned generator produces motion" the two are swappable, and swapping generators would
be plumbing rather than a contribution. **The differentiator is the claim, not the architecture:
transfer to a body, or an embodiment, never in the training set.** Diffusion policies, Sora and
animation pipelines do not attempt that -- which is also why the evaluation is a held-out body
rather than sample quality.

The other two Week 11 questions -- sensor count against speed, and whether proprioception can be
removed entirely -- are answered elsewhere in this file and in FINDINGS.

---

## Settled, and where the evidence lives

Each of these was an open question that a measurement closed. The full argument and the numbers
are in `FINDINGS.md` at the finding named; nothing is repeated here.

| | question | what settled it | finding |
|---|---|---|---|
| **Q5** | Does removing the body code from `z` make the decoder read the frame? | Yes, and it does not help: the decoder used the frame 2x more and transfer got 1.21x worse. | F18 |
| **Q6** | Can the decoder be given the view that works? | Yes, and it uses it 7.6x less. Access was never the constraint. | F19 |
| **Q7** | Is the objective the constraint? | Yes. `lambda_cross` is the only intervention of six that improved transfer. | F21 |
| **Q8** | What is the latent for, once the decoder stops needing it? | Gait, and only gait: 88.7% of its variance, with body down to 1.2%. | F22 |
| **Q9** | Does the corrected target make the latent do its job? | It triples the transition's contribution (11% to 36%) and changes transfer not at all. The constraint is the data, not the target. | F25, F26 |
| **Q10** | Is the forward model worth keeping? | Yes. It rolls the world forward 1.2-1.5x better than a frozen world out to ten steps; we had only ever scored it on a task the method does not assign it. | F46 |
| **Q17** | Does the 4-leg body test a new embodiment? | No. The latent places it 0.578 from the body it was cut from, against a chance level of 0.981. It tests a new action space; the B1 held out entirely is the real test. | F41 |
| **Q15** | Does anything transfer to a genuinely different robot? | Yes, and all of it travels through `z`: 1.28x on a held-out B1, dropping to 0.98x -- random weights -- when the latent is zeroed. The decoder's use of the frame carries nothing. | F43 |
| **Q16** | Can the forward model be made to work on a new robot? | Not frozen (0.57-0.71x), and coverage does not fix it (5-8%). But **one target clip clears break-even and nine clear every horizon tested**, about 7x fewer clips than from cold. The claim is cheap adaptation, not zero-shot transfer. | F44, F45 |

---

## Settled / obsolete (was Q1–Q4 in the old version)

- **Argue vs prove** → decided: **prove**, via cross-embodiment (above).
- **How to word the proprioception claim** → settled 2026-08-16: **not** "proprioception cannot do
  this". Morphology-agnostic proprioceptive control exists (joints as a token set over the
  kinematic graph), so that sentence is refutable. The defensible form is that those methods must
  be handed the **kinematic tree** and a camera has to be handed nothing. Four places in the deck
  and FINDINGS were corrected; references still need verifying before they are cited.
- **Leg amputation (nested, weak proof) vs different body** → chose a genuinely
  different body (B1 quadruped). Amputation reused only as the *4-leg test*, not the proof.
- **Terrain experiment** → dropped (open-loop can't traverse it; poor cost/benefit).
- **Leg-length range (0.5/0.75/1.0 vs 0.7/0.85/1.0)** → moot; leg-length variants are
  now just pretraining diversity, not the core axis.
- **AIRL policy reuse** → parked (config drift; only yuchen's exact obs config +
  normalization would unblock; action bounds + 4-leg=LFRF config already recovered).
