# Research Direction — Cross-Morphology Locomotion via Latent Action World Models

> **Role**: The plan as it stands today.
>
> Edited in place, never stacked with updates -- if a step changes, the old text is replaced. For how a step came to be what it is, read `PROGRESS.md`.
>
> **Start at §0.** Everything from §1 onward is a layered history of superseding updates
> (§1 → §1.1 → §1.2 → §1.3 → historical Stage 1/2 detail from July-August) -- real provenance, kept
> for how the plan got here, but not the place to read for current status. §0 is that place, rewritten
> fresh each time it goes stale rather than stacked.

## 0. Current status (2026-09-12) — read this first

**The core claim still holds, unchanged since §1.2**: a shared Froude body-motion coordinate
transfers across embodiments, confirmed on hexapod and B1 (F127 and everything built on it since).
Nothing this section covers touches that.

**The advisor's Week 15 ask (§1.3) is what's actually driving work right now**: stop at ranking, go
to a real Controller. `doc/OPEN_QUESTION.md` Q21 orders it — (1) B1 motor babble as candidate
source, (2) a reward-quality gate, (3) RL controller, (4) history-state ablation. Deliberately not
building the controller first (F179's arc did that and cost weeks on an unchecked assumption).

**Step 1 (babble as candidate source) — genuinely open, not resolved (F190-F194).** A diverse
3-family CPG babble generator was built (F190, after finding and fixing a real turn-primitive
degeneracy). The clean 2x2x2 test (F194): babble clears its own chance on family accuracy (69% vs
45%, capture ratio 0.60 against expert) but **fails the more trustworthy continuous Froude-distance
metric** (capture ratio 0.45). The verdict flips depending on which metric is trusted — this is not
settled, and the distance metric is the one to believe. `free_offset` (F193) gives babble a small,
real gain over locked candidates but is still worse on distance; two real bugs in that mechanism
were found and fixed along the way (a freeze, then a catastrophic compounding-drift tip-over caught
from a rendered video, not a table).

**Step 2 (reward-quality gate) — checked, and it fails (F195).** `body_head(proj(action))`, B1's
own MuJoCo physics, scores at or below chance (4.2% vs 5.9% chance) ranking local action
perturbations — the exact ability PPO-style RL needs to explore. This blocks step 3 outright: an RL
controller built on this reward would very likely reproduce F179's collapse, not from a bad
algorithm but because the reward has no local gradient to climb.

**Fixing the reward-quality gate has consumed most of this session, and it converges on a real,
mapped structural cause, not "nothing works, unexplained" (F142, F195-F199).** The action's causal
weight on the next frame is under 3% (F142) — almost everything about `e_{t+1}` is explained by
`e_t` alone. MSE-dominated training always takes that cheap win first. Five independent,
mechanistically distinct fixes have now been tried at real/exact strength:

| fix | mechanism | result |
|---|---|---|
| stage-3 contrastive (`wm.adapt3`, this session) | single-body condition-ID InfoNCE | full 15k-step budget on B1: `family` 30% vs chance 27% — **not selection** |
| F141 hinge (ActSWM-style) | rollout-level real-vs-null separation | diverged 3-4x past a frozen-frame baseline by horizon 2 — the pre-registered failure case, worse |
| F178 counterfactual targets | remove the copy-the-input shortcut from the target | exact test on B1's own MuJoCo state-resume — still null |
| recurrence, 4 variants (R0, ConvGRU probe, full RSSM, ConvGRU at full budget) | give the model more temporal context | **F198: all fail.** Full-budget ConvGRU (-0.006 gap) is *worse* than its own 2000-iteration probe (+0.069); the field-standard full RSSM (pooled + stochastic, matching DreamerV3's own recipe) scored worst of every scorer measured in this project's history |
| **F141's own missed fix: multi-step recon anchor + hinge** | `recon` extended to match the hinge's horizon (not just step 1), using existing `lambda_rollout` -- zero new code | **F199: passes all 3 of F141's pre-registered criteria for the first time.** No divergence, `/mean-z` within-family 0.886-0.938 and holds flat across horizon, separation rises and holds. A follow-up ceiling check (F199 coda) shows this is close to what the data allows (natural family separation in `z` is itself near 1.0) — real, modest, not a weak model leaving signal on the table |

**Resolved, 2026-09-12, and it closes negative (F199's coda, F201).** The launch mistake was fixed
and the full chain rerun: `beh12_hinge_multistep_anchor_v2` (with `lambda_body`) → full B1 adaptation
→ `reward_quality_gate_b1.py`. Result: **8.3%, no real margin over F195's 5.9% chance, not an
improvement on the 16.7%/4.2% expert/babble baselines already on record.** F199's fix is real on
hexapod pretraining but does not propagate to B1. **This makes six independent, mechanistically
distinct fixes (contrastive, hinge without an anchor, counterfactual targets, four recurrence
variants, and now the anchored hinge) that have all failed to produce a usable B1 reward function
(F201).** Per this project's standing rule for this shape of result: stop trying fixes, write this
up as a characterized negative result. `doc/OPEN_QUESTION.md` Q21 step 3 (RL controller) stays
blocked, permanently for this checkpoint family, not pending another attempt.

**A separate, corrected understanding, not yet acted on (F197).** A prior claim in this project's
own reasoning — that stage-3 InfoNCE "has a specific history of not transferring across bodies" —
was overstated and has been corrected. `wm/adapt3.py`'s positive/negative pairs are drawn entirely
from ONE embodiment's own condition labels; Froude values and any second body never enter that
file. **A genuinely cross-embodiment stage-3 (Froude-matched positive pairs across hexapod and B1)
has never been built or tested** — this is a real, open, not-yet-attempted experiment (Track 2),
separate from and not blocking the reward-quality-gate line above. See
`doc/ref/literature_review3_infonce_modality_gap.md` for what a non-naive design needs
(reconstruction hybrid, controlled negative sampling) before building it.

**Q22/Q23, parked, not abandoned.** B1 now stands and walks under native CoppeliaSim-Bullet dynamics
(F196), and the native generic motor-babble preview is now physically usable but not final (F200):
fixed generic diagonal duty-cycle CPG, `2.0 Hz`, amplitude `0.24`, calf ratio `2.5`, per-step
noise, egocentric render upright for 8 s at forward Froude `+0.0328`. This is claim-honest as a
preview (`generic CPG + noise`, no demo/policy/retarget/outcome filtering), but still below the
pretraining/expert forward band (`~0.10-0.20`) and is **not yet the final babble dataset**. Q23
(does the vision-goal pathway carry a hexapod-CoppeliaSim-vs-B1-MuJoCo physics-origin bias,
separate from F192's generic-weak-signal story) is still open and untested.

**What this means for the thesis if the reward-quality line never clears**: the fallback described
in §1.3 stands unchanged — grounding and goal-conditioned selection (no RL, no rollout) both work on
a body absent from pretrain, using B1's own behaviour library. That is real, measured, and does not
depend on anything in this section resolving.

---

## The problem

A locomotion policy maps the robot's state to a joint command. It is tied to the body it was
trained on: shorten a leg, redistribute the mass, or break a limb, and the policy no longer
works. Retraining from scratch costs hours to days, every time.

The question this project asks is whether a model can learn a **latent action** from video alone,
with no morphology label and no kinematics given, that separates *what movement is happening* from
*which body is doing it* -- so that the same latent drives a body it has never seen.

**Why vision rather than proprioception.** The committee's question was why vision is worth the
trouble. Three leg lengths cannot answer it: they share an 18-D joint space, so proprioception
transfers between them too, and vision wins only on convenience. Answering it needs a body whose
action space is **disjoint** -- a 12-DOF quadruped against an 18-DOF hexapod share no joint
correspondence, while one camera describes both in `256x256x3` whatever the body. Hence the two
stages.

> **State this carefully.** Morphology-agnostic proprioceptive control does exist -- joints as a
> token set over the kinematic graph -- so "proprioception cannot do this" is not defensible and
> should not appear in the deck or the thesis. The defensible claim is that those methods must be
> **handed the kinematic tree**, and a camera has to be handed nothing: this pipeline is given
> video of a B1 and knows nothing else about it. Verify the specific references before citing
> them; the distinction stands regardless of which papers are named.

**Target**: Stage 1 (cross-morphology) then Stage 2 (cross-embodiment), then a
deployment loop. See `PROGRESS.md` for the dated engineering log and `SIM_GUIDE.md` for how to run
anything described here.

**Vocabulary** — three separate things, kept distinct:

| Term | Meaning |
|---|---|
| **Stage** | research stage. Stage 1 = cross-morphology (3 leg lengths). Stage 2 = cross-embodiment (hexapod + quadruped) |
| **Step** | a numbered task inside a stage, listed in section 5 |
| **Pretraining / Deployment** | training the world model, versus using it to control a body. Deployment is out of thesis scope but constrains design |

**Conventions** — this file states the *current* direction only; history lives in `PROGRESS.md`.
Status is one of `done`, `in progress`, `blocked`, `open`. Every number carries its unit and a
reference point. Blockquotes are reserved for caveats.

---

## Contents

0. [Current status — read this first](#0-current-status-2026-09-12--read-this-first)
1. [Claim](#1-claim)
2. [Approach](#2-approach)
3. [Pipeline](#3-pipeline)
4. [Roadmap](#4-roadmap)
5. [Steps (pointer only)](#5-steps)
6. [Risks and confounds (settled)](#6-risks-and-confounds-stage-1-era-settled)
7. [Fallbacks (not needed)](#7-fallbacks-not-needed)
8. [Deployment — superseded](#8-deployment----superseded-by-what-was-actually-built)
9. [Baselines and references](#9-baselines-and-references)
10. [Open decisions](#10-open-decisions)

---

## 1. Claim

### Positioning against three named neighbours

**The scoped contribution, verbatim, and the only version to be written:**

> In periodic visual locomotion, the joint action is inverse-recoverable from a SINGLE frame (F145:
> insect R2 0.78, turning 0.93) because gait phase makes the pose encode the command -- so the
> action is forward-redundant (F144: <3% of prediction error). This closes the loop between two
> prior observations -- adjacent-frame redundancy (AHA-WAM) and the action-invariant teacher-forcing
> solution (UWM-JEPA) -- with a measured mechanism, and shows the objective-level fix-family fails
> on it (ActSWM hinge, F141) and that the residual-target route is closed (F144), since the
> residual of an action-blind prediction carries no action beyond the frame -- because there is no
> action-dependent forward signal to recover, only a redundant one.

**Never the unscoped version** -- "we discover visual world models cannot be action-conditioned in
locomotion". AHA-WAM and UWM-JEPA each state part of it already.

| paper | theirs | ours |
|---|---|---|
| **AHA-WAM** 2606.09811 | adjacent frames "redundant for control", used as architectural motivation | we **measure** it and attribute it to **gait periodicity** |
| **UWM-JEPA** 2605.25313 Sec 4 | the action-invariant solution as a training-target problem; fix is counterfactual targets | in locomotion that solution is **near-optimal**, so target fixes have nothing better to reach |
| **Yeom et al.** 2606.07687 | V-JEPA inverse-recoverable action signal; CALVIN's static-scene note | **inverse-recoverable is not forward-necessary**; periodicity is the severe form of their exception |

**The claim is deliberately narrow about target-level fixes.** F144 tested a *residual* target;
UWM-JEPA's *counterfactual* target has never been trained and is **not** claimed to fail. Widening
that clause requires running the arm -- see F147.

### Positioning against Yeom et al. (2606.07687)

**Their result:** V-JEPA carries action-relevant structure that is inverse-dynamics recoverable --
R2 0.40 frozen, 0.85 with an ID head -- and they observe that static-environment benchmarks such as
CALVIN let per-frame appearance stand in for temporal context.

**The contribution statement, as owned by the user:**

> งานก่อนหน้า (Yeom et al. 2606.07687) แสดงว่า V-JEPA เก็บ action-relevant structure ที่
> inverse-dynamics recoverable ได้ดี และตั้งข้อสังเกตว่า static-environment benchmarks (CALVIN)
> ทำให้ per-frame appearance แทน temporal context ได้ เราแสดงว่าใน legged locomotion กรณีนี้รุนแรงกว่าและเป็นเชิงโครงสร้าง:
> gait periodicity ทำให้ joint action อ่านได้เกือบสมบูรณ์จาก single-frame pose ดังนั้นแม้ inverse-dynamics R2
> จะสูง (0.78) action ก็แทบไม่ช่วย forward prediction (residual เหนือ frame-alone <1%) เราแสดงว่าสิ่งนี้ทำให้
> rollout-level action-sensitivity objectives (ActSWM-style) ล้มเหลวโดยหลักการ -- วัดข้ามสองร่าง ทุก horizon
> และผ่าน residual-recoverability -- เพราะไม่มี action-dependent signal ใน forward transition ให้ amplify
> แม้ signal นั้นจะ inverse-recoverable

**Four things it does:** acknowledges their result without reclaiming it; extends the exception they
left open, periodicity being the severe form of CALVIN's static tabletop; adds the distinction
**inverse-recoverable is not forward-necessary**; and ties that distinction to a measured objective
failure.

**One sentence in it must be narrowed, and F145 is why.** "อ่านได้เกือบสมบูรณ์จาก single-frame pose"
holds on the **insect**, where a single frame reads the command at R2 0.779 against a pair's 0.887.
**It does not hold on the B1**, where a single frame reads 0.161 against a pair's 0.342. The
dissociation itself is unharmed and is in fact sharpest on the insect -- R2 0.887 recoverable, under
3% of forward prediction error -- but **the sentence has to say "on the insect" or it is an
overclaim on the quadruped.** See F145 for the per-family numbers.


Learn a latent action `z_t` from simulation video that **maps to a shared body-motion coordinate**
-- forward, lateral and yaw, the same physical quantities on both robots (F127) -- given no
morphology label and no kinematics.

**`z_t` itself is not body-blind, and saying it is contradicts our own measurement.** Body identity
is decodable from `z` at **0.974** against a chance of 0.5, and at 0.732 with the coordinate term
switched off (F146). **The agnosticism lives in the coordinate `z` maps to, not in `z`.** That is
also the more coherent reading of the architecture: the decoders are per-body (F120), so `z` has to
carry body identity for them to work at all, and removing it adversarially made transfer 1.2x worse
(F21).

Two claims, at two scopes, because they turned out to need different words:

**Within the hexapod family** (Stage 1) the latent transfers to an unseen body **without
retraining**: `m3d_cross` scores 3.44 deg and R² +0.81 on a held-out body. The mechanism is not
what was expected -- the decoder reads the body from the frame and the latent carries the movement
(F42, slide 5).

**Across embodiments** (Stage 2) the claim is **cheap adaptation, not zero-shot transfer**. A
frozen forward model does not survive the change of robot (F44, 0.57-0.71x on the B1, worse than
predicting no motion), and adapting it on target data was always the design -- the source method
itself finetunes on 7,265 target trajectories. Measured: **one B1 clip clears break-even, nine
clear it at every horizon tested, about 7x fewer target clips than starting cold** (F45).

**Closed loop, in physics** (Stage 2, 2026-08-26/27) is now the third scope and the strongest one.
An **unseen hexapod body** is controlled with the world model **completely frozen** -- only the
two-layer action projector refitted -- at survival **15/15**, behaviour **15/15**, median speed
error **19.0%** over fifteen runs (F85). **With the ten warm-start steps removed** -- they replay
the goal clip's own actions -- it is **14/15 and 58.8%**: the result survives, its margin is
partly inherited, and forward is the only behaviour that *improves* without the hint (F101).

A **quadruped** stands through every episode under the same planner after the world model is
adapted on 24 of its clips, at behaviour-family 38-58% against a 28% chance rate, and **hits none of
three speed targets** (F91).

**Cross-embodiment control** is the fourth scope and the narrowest. Goal frames come from a
hexapod clip, candidates stay B1 clips because only those are executable, and the driven robot is
the quadruped -- so only the goal crosses embodiments, which is the form this demonstration can take
without a motion decoder that generalises across bodies.

**Every B1 number was withdrawn on 2026-08-29 and nothing has replaced it yet.** The set those
numbers were measured on had four defects -- the robot clipped by the image edge in 61% of frames,
an unpinned camera giving every clip its own background, a forward clip filed as the weakest turn
level, and turns that ran opposite to the insect's (F104-F106, F108). The data is corrected and all
three sets now turn the same way; **the checkpoints were deleted and stages 1-3 have to be rebuilt.**

**What the withdrawn runs showed, kept because it shapes what to measure next**, and none of it is
quotable until the rebuild:

| | |
|---|---|
| forward crosses | 53-58% of planned steps against 33% chance, under every objective tried |
| **forward does not need the world model** | an arm whose `/mean-z` is 0.977 -- it answers the same given the real action or the mean one -- still selects forward at 53%. A 2% residual sensitivity is enough, so forward selection is not evidence the forward model works |
| turning is where the objectives differ | the collapsed arm sits *below* chance at 22%, the contrastive arm clears it at 43%, and `/mean-z` goes 0.977 to 0.493 |
| the contrastive advantage is not established | MSE at the original budget reaches 36% on turning against 43%, inside that arm's own spread, and the two had been compared at different budgets. Three seeds per arm is the outstanding measurement |
| sideways fails everywhere | 19-20% against 17%, on four independent measurements |

**And the axis that makes any of it interesting is the joint-space one.** The two robots' action
spaces are 18-D and 12-D with nothing commensurable between them, and the correspondence is learned
by the projector rather than defined by a kinematic model. LAC-WM unifies quantities that already
mean the same thing on both bodies; X-Morph retargets through a URDF; proprioceptive
morphology-agnostic control is handed the kinematic graph. **A joint-space target is what makes the
problem hard and what makes solving it worth something** (F99), and F73's conditional is the
measured answer: it works within a robot unsupervised and crosses robots only with the body term.

**So the defensible sentence names the behaviour and the limit**: a quadruped walks forward from a
stick insect's video, and turning is the only behaviour where a better world model measurably beats
a collapsed one. Do not write "behaviours cross".

**The adaptation objective is a claim in its own right.** LAC-WM's three stages are MSE throughout.
Applied across families that fails in a specific way: the forward model improves its predictions
and **discards the action channel entirely** -- its answer given the real action equals its answer
given the mean action to three decimals, at every checkpoint of a 15k-step run. A contrastive term,
which asks for the ranking a planner performs rather than the prediction MSE asks for, lifts
quadruped selection from 30% to 57% with data, robot, architecture and budget unchanged (F88).
**That term is ours and is the second thing this project contributes.**

> Do not write "transfers to a new robot without retraining". It was measured and it is false --
> **except within the hexapod family in closed loop, where it is now true and measured.** Across
> families the defensible sentence is that the model is **cheap to adapt** to a robot it has never
> seen, and that a camera is the only thing it has to be given about that robot.

> **And "a camera is the only thing" is not yet earned.** The planner chooses among twelve recorded
> behaviours of the target robot, so something already made that robot walk, turn and strafe.
> Replacing the recorded library with **random motor babbling** is the untested experiment that
> would make the sentence true.

---

## 1.1 Positioning update (2026-09-07) — current, supersedes the framing above pending a full rewrite

> **Scope note.** Everything above this point in §1 predates the egocentric pivot and the
> candidate-selection-vs-imagination-RL diagnostic arc (F141-F178) — it still references withdrawn
> B1 numbers and deleted checkpoints from 2026-08-29. It has not been rewritten wholesale (that is
> separate, larger work), but this section is the current positioning and should be read as
> overriding it for anything about the thesis's actual claim and evidence status.
>
> **Label discipline, kept explicit on purpose**: every line below is tagged **[CONFIRMED]** (a
> measured result, safe to write into the thesis as established) or **[AIM]** (a hypothesis or
> planned test, not yet a result — do not migrate an [AIM] line into the thesis as if it were
> [CONFIRMED]). When this section moves into `report/proposal.tex`, carry the tags with it, or
> resolve them to prose that preserves the distinction.

### The gap, two axes

- Cross-embodiment locomotion exists, but via **body description** — proprioception, morphology
  parameters, or large-scale embodiment randomization (URMA/URMAv2, Multi-Loco, One-Policy,
  H-Zero, PEAC). *(Citations as supplied; not independently re-verified this session the way
  UWM-JEPA was read in full — confirm each before it goes into the thesis, per this file's own
  standing discipline of not citing a paper without reading it.)*
- Cross-embodiment **from vision** exists, but for **manipulation** — optical-flow or
  end-effector retargeting (TrajSkill, LAC-WM, IEEE latent-space work). Same caveat on citation
  verification applies.
- The intersection — vision-based cross-embodiment **locomotion**, frozen encoder, a
  physically-grounded shared coordinate, no body description handed to the model — is unoccupied.
  **[AIM as a literature claim]**: this is the gap being claimed; it depends on the citation set
  above actually being correct and complete, which has not been independently checked this
  session.

### What's confirmed — safe to write as results now

- **[CONFIRMED]** A shared Froude body-motion coordinate transfers cross-embodiment (F127 and the
  closed-loop measurements built on it).
- **[CONFIRMED]** Egocentric view fixes 1-step action-conditioning that allocentric view could not
  (GATE C, null/real 1.03→1.16) — the frozen video encoder can be made to carry the action, at
  least coarsely, by a viewpoint choice alone.
- **[CONFIRMED]** The forward transition model (FTM) predicts coarse, family-level behaviour
  accurately out to a k=10 horizon on both bodies, and the accuracy does not degrade with horizon
  the way fine-magnitude prediction does: rollout correlation 0.938→0.970 (B1) and 0.898→0.943
  (hexapod) from k=1 to k=10, with prediction error *shrinking* relative to the real outcome
  spread as horizon grows (err/std 0.327→0.198 on B1, 0.341→0.226 on hexapod). Ranking the 12
  behaviour conditions by FTM-predicted outcome matches their real ranking strongly and
  significantly at every horizon on both bodies (Spearman ρ 0.84-0.93, p<0.001 throughout). A
  small gradient-usefulness check on B1 (exact MuJoCo state-resume, not approximate) found the
  gradient of predicted outcome w.r.t. action pointed toward real improvement in 3/3 tested cases.
  **Together this is a viable controller signal at the coarse/family level** — the resolution this
  thesis's actual goal (cross-embodiment *behaviour* transfer) needs, as distinct from the
  fine-within-family magnitude discrimination that six independent, mechanistically distinct fixes
  (F173, F177-F178) all failed to achieve and that this thesis explicitly parks as future work, not
  a claim.
- **[CONFIRMED, as prior art, not as our invention]** Frozen-encoder world model + policy learned
  in imagination is a validated paradigm elsewhere (DINO-WM, DreamerPro) — this thesis instantiates
  that paradigm for cross-embodiment locomotion, it does not invent the paradigm itself. *(Same
  citation-verification caveat as above.)*

### What is hypothesis, not yet result — do not claim this section as done

- **[AIM]** "We investigate whether a policy trained in imagination on this frozen-encoder world
  model achieves cross-embodiment behaviour transfer." This is the aim the RL loop (see below) is
  built to test — not a result until Stage A/B (below) actually run and clear their pre-registered
  bars.
- **[AIM, explicitly parked]** Fine-grained, within-family magnitude discrimination. Named as
  future/stretch work, not attempted as part of the core claim, per the six-null diagnostic arc
  (F173, F177-F178) that closed this thread for this thesis's scope.

### Why the thesis is defensible even if the loop underdelivers

If Stage A/B (below) do not clear their bars, the thesis still stands on: a viable coarse-level
controller signal, validated on both bodies, in a gap the existing literature does not occupy, using
a paradigm validated elsewhere but not yet instantiated for cross-embodiment locomotion. That is
"viable signal + positioning," not nothing. A working loop upgrades it to "demonstrated transfer."
Do not let the loop's outcome retroactively change how the [CONFIRMED] section above is worded —
those results are true regardless of what Stage A/B show.

### Claim (3) does not need imagination-RL, and its controller has a hard scoring-space constraint (2026-09-08)

**Resolved after a self-contradiction was flagged and checked against the actual F116-F127 text
(not memory of it).** Claim (3) — drive a body not in pretraining via the shared coordinate +
motor babble — does not require long-horizon imagination-based value learning (the thing F179's
whole arc, and PPO/GAMMA after it, failed to make work). It needs: babble → an empirically-learned
action→Froude map for the new body → a controller that picks actions toward a Froude goal. That
controller does not need to integrate reward over a 20-100 step imagined horizon; it needs to move
the body in roughly the right direction/magnitude, which is coarse, short-horizon decision-making —
exactly the resolution the FTM is independently confirmed good at (k≤10, rho 0.85+).

**"Action-selection is dead" (F135/F116/F118) does NOT mean this controller is dead — the actual
failure was the scoring metric, not goal-directed FTM control, and this is a controlled, proven
result, not an inference:**

- F116/F118: candidate pool-selection scored by **embedding distance**
  (`score(a) = ||rollout(a) − goal_embedding||`) never conditioned on the goal at all — under a
  mismatch control, picks tracked "what the robot is currently doing" (56-70% agreement with the
  demonstration) and scored *below chance* against the actual goal shown (18-23% vs. 28% chance).
- F119/F122/F127: the **same pool-selection mechanism**, rescored by **Froude/body-motion
  distance** — but the winning version (Mode D) is `score(a) = |body_head(proj(a)) − goal_froude|`,
  with **no FTM rollout at all**. Mode C (the rollout version, `body_head(rollout(a))`) was tested
  in the same pass and is *worse* (33-44% vs. Mode D's 68-70% pooled, and on 3 channels the rollout
  actively *destroys* the turning signal, F127). Goal-conditioning appeared with the coordinate
  change and survived its own mismatch control: 76-86% same-robot, 35-38% cross-embodiment with 1
  channel (F122), 70% cross-embodiment with 3 channels (F127) — **all via the no-rollout Mode D.**

**Important terminological correction, not to be blurred**: `body_head(proj(a))` — the actual F127
winner — **is not a forward/world model.** It is a direct, single-step, stateless action→Froude
regressor: given a recorded action alone, predict the resulting body motion, no rollout, no
multi-step state, no FTM involved in the winning mechanism at all. It works, and it is not "using
the world model to control" in the sense the rest of this plan (Stage A/B, the RL loop) means. Any
description of claim (3)'s controller should say this precisely — "Froude-scored action regression"
or similar — not "FTM-based control," which overstates what F127 actually validated.

**Hard constraint on any future controller for claim (3), not a preference**: score in Froude/
body-motion space, never embedding space, and prefer the no-rollout (Mode D-style) direct
action→Froude scoring over routing through the FTM — the FTM version is the one already measured
to be worse here, not merely untested.

**What was never tested, and remains open**: short-horizon MPC, replanning every step — every prior
attempt (F81-F127) picks from a **fixed, pre-recorded pool** of whole candidate behaviours, never
continuously optimizes/replans a per-step action. Note this can be built either way given what's
now confirmed: an FTM-free version (repeatedly query `body_head(proj(a))` over a small continuous
action search each step, no rollout at all — extending the already-proven-best mechanism) or a
classic FTM-based version (V-JEPA-2-AC style, rolling the world model forward each replanning step)
— and given Mode D beat Mode C here, the FTM-free version is the one with evidence behind it, not
just the untested one. F116/F118's "doesn't read the goal" finding is specific to the
embedding-metric pool-selection setup that was tested; it does not establish that either MPC variant
would fail the same way, and the F119/F122 fix (change the scoring space, keep everything else)
gives good reason to expect a Froude-scored MPC controller to condition on the goal correctly too,
plausibly better than a fixed pool (continuous replanning vs. picking from a finite discrete set).

**So claim (3)'s controller options, in order of how proven they are**: (a) Froude-scored,
no-rollout, direct action→Froude pool-selection (`body_head(proj(a))`) — proven working on known
bodies, modest (35-70% depending on channel width), not a world model; (b) the same direct
scoring extended to short-horizon MPC (replan every step, no FTM) — untried, plausibly stronger,
uses the same validated coordinate and the same no-rollout mechanism; (c) an FTM-based rollout
version of either — already measured worse than the no-rollout version here (Mode C vs. D, F127),
not the default choice. None of these need the long-horizon imagination-RL machinery that F179
onward spent the session on, and none of them are "world-model-based control" in that sense.

### The RL loop plan — tried in full, retired (full ladder in FINDINGS.md F179)

**What was attempted**: a Dreamer-style (Hafner et al. 2020/2021) actor-critic trained entirely in
imagination against the frozen FTM -- reward = distance to a goal in Froude space, backprop through
the differentiable world model. Motivated by the FTM's independently-confirmed coarse/short-horizon
accuracy (k<=10, rho 0.85+) and an explicit, named deviation from Dreamer (the FTM stays frozen
throughout, never re-grounded on fresh data under the current policy -- flagged in advance as the
mechanism that would let a policy exploit off-manifold FTM errors).

**What happened, compressed**: Stage A (frozen FTM, real closed loop) failed against the B1 clone
baseline. Two cheap fixes (action regularization, a joint uncertainty penalty) came back null.
Re-grounding at three budgets didn't close the gap. All of this was then found to be **confounded**
by a separate bug: an isolation test showed the actor-critic optimizer itself doesn't climb even a
static FTM surface (critic value diverged unboundedly, -82 -> -421) -- every FTM-side diagnosis
above assumed a working optimizer that didn't exist. A full critic-stabilization ladder followed
(EMA target, symlog + percentile normalization, hard target updates, a two-hot distributional
critic matching DreamerV3's own mechanism) -- the two-hot critic was the best of four and still
narrowly failed its pre-registered bar (0.272 vs 0.25). Escalating to PPO (which doesn't bootstrap
a value function through imagined rollouts, sidestepping this exact failure) reached the same
~0.27-0.28 gap by a structurally unrelated algorithm -- strong evidence the wall is the FTM's
long-horizon rollout, not the RL algorithm. A shorter-horizon variant (GAMMA=0.95) ruled out the
hopeful reading: the gap didn't shrink, and the policy instead collapsed into a frozen, degenerate
stance (action variation down ~92%). **Conclusion: RL via a TD-learned critic on this FTM is dead
for a genuine, isolated reason -- a value-learning/function-approximation problem, confirmed across
two unrelated algorithms, not measurement error, drift, or rollout realism.**

**Retired by §1.2 below**, which reframes claim (3) away from "improve this controller" entirely.
`body_head(proj(a))` -- proven, cheap, no FTM in the loop -- remains available as the fallback
mechanism if the world-model-in-the-loop constraint is relaxed. Cite Dreamer for the mechanics
(DreamerPro only replaces Dreamer's world-model objective, irrelevant here); Koopman Dreamer for
the documented compounding-error failure mode in locomotion specifically, confirmed empirically
here across two algorithms rather than only by citation.


### 1.2 Where claim (3) actually stands (2026-09-10) — supersedes the Stage A/B/C gating above

The Stage A/B/C plan above is **retired**: it gated the thesis result on an RL/critic fix, and the
claim was subsequently reframed (F182-F189) away from "improve the controller" — which is settled —
toward **"ground a body absent from pretrain."** Current status, tagged the same way as §1.1:

| step | status | evidence |
|---|---|---|
| (1) pretrain WM on known bodies | **[CONFIRMED]** | — |
| (2) babble on an unseen body → fit `a→z` | **[CONFIRMED] for B1** | F183: median ρ 0.264 → **0.572**, forward finally real |
| (3) transfer a Froude goal, source → new body | **[CONFIRMED] for B1** | F184/F187: mode A and D both clear, vision-only goal costs nothing |
| (4) control via direct-Froude selection | **[CONFIRMED], settled** | F188: 3/3 behaviour class, 17.4% median speed error |
| (2)-(4) on gecko (babble-only, no expert library) | **[PARTIAL, and the ceiling is the camera]** | F189: two measurement bugs found and fixed; stage 4 now 0.970 (below 1.0 for the first time), projector-path rho 0.032 -> 0.249 |

**The controller question is closed and should not be reopened.** The 2×2 ablation (F188) showed
goal source explains none of the gap and candidate mechanism explains all of it; rollout fails
same-robot too, which is F126/F118 reconfirmed. **No further rollout debugging is warranted.**

**The one open technical question is Q20** (`OPEN_QUESTION.md`), and F189 moved it from a debugging
question to a scoping one. Gecko's stage-4 failure was two measurement bugs, both now fixed: the
stage-1 "gate" was read with the sign inverted (it is `hold/model`, so higher is better — and it is
anti-correlated with downstream success anyway), and gecko's body frame was built on a body x axis
that points almost straight up, scrambling forward, lateral and yaw together. Correcting the frame
took yaw from dead (−0.042) to the strongest channel (+0.501); giving the projector 20 frames of
action history instead of 1 took stage 4 to 0.970.

What remains is a property of the robot, measured end to end: gecko's **actions** carry ρ 0.736, its
**egocentric video** carries only **0.374**, and the pipeline delivers 0.249 — 67% of what the video
allows. B1's video carries 0.747. Gecko's forward Froude is 0.038–0.048 against B1's 0.126 and is
flat across gait frequencies, so it barely translates between frames while its legs fill the view.
**The binding constraint is what the camera can see, which is a claim about the sensor, not about
the shared coordinate.**

### 1.3 What Week 15's advisor review changes (2026-09-10)

`feedbacks/feedback_ajan_go.md` Week 15 moved the goalposts, and the move is fair. Recorded here
because §1.2 above describes what is *measured*; this describes what is now *required*.

**The hole he found:** every closed-loop number in §1.2 scores candidates drawn from **B1's own
recorded expert clips**. A body that has never had a controller has no such clips. So "control
works on a body absent from pretrain" is true only for a body that arrives with a behaviour
library — which is not the interesting case, and is exactly the gecko case that fails (F189).

**What he asked for:** stop at ranking, or go all the way to a **Controller** — a policy that takes
state in and emits joint actions, no library. He recommends turning the WM's Froude error into a
**reward term for vanilla PPO in the real simulator** (no imagined rollout), and pinning the
contribution to **Vanilla RL vs WM-guided RL** on sample efficiency and cross-embodiment transfer.

**This does not contradict F179.** F179 killed policy learning *inside the FTM's imagination*
(compounding rollout error). The proposal never rolls the FTM — real physics steps the world, the
WM only scores the step. Untested, and not ruled out.

**Order of work, and it is not his order** (full reasoning in `OPEN_QUESTION.md` Q21):

| # | step | why here |
|---|---|---|
| 1 | **B1 motor babble as the candidate source** | cheapest, reuses the whole existing loop, and **B1 is the only body where the answer can be graded** (it has both babble and ground truth) |
| 2 | reward-quality gate | F136 measured this reward ranking local perturbations at 33% vs a 50% coin — and PPO explores by local perturbation |
| 3 | RL controller + WM reward | only if 2 clears |
| 4 | history-state ablation | independent; would improve 3's reward if it works |

**Deliberately not doing:** building the RL controller first. F179's arc was built on an assumption
never checked and cost weeks; step 2 exists so that does not recur.

**What the thesis can claim today, stated at true strength**: grounding and goal-conditioned
control both work on a body genuinely absent from pretrain — using B1's own behaviour library, so
this is the **near-morphology** version of claim (3). The **far** version (gecko, babble-only) is
partially grounded and its remaining gap is measured and attributed: not the mechanism, not the
data's information content, but how much of its own motion the robot's camera can see at Froude
0.04. That is a scoped, measurable limitation — "the method grounds a novel body when that body's
motion is visible to its own camera" — and it bounds which bodies the procedure handles rather than
refuting the coordinate.

---

## 2. Approach
> Full detail in `PROGRESS.md` §12. The Core Claim above is now **Stage 1** of a two-stage plan.

The committee's core push (*"why is vision worth it over proprioception?"*) can't be answered on
same-topology bodies alone: the 3 leg-length variants share an identical **18-D** joint space, so
proprioception could share it too — vision's edge there is *reach*, not a provable advantage. To
**prove** it we add a genuinely different body whose action space is **disjoint** from the hexapod's,
where proprioception can't be shared at all but vision (pixels) can.

- **Stage 1 — cross-morphology** (this doc's Steps -1 … 2): 3 leg lengths, **IK-retargeting** (per-body-
  different `a_t` in the *same* 18-D space). Gets the pipeline working + latent organizes by behavior +
  the decisive latent-vs-raw-joint ablation. Proves the latent is *better*; does **not** prove
  vision > proprioception (same topology).
- **Stage 2 — cross-embodiment / compositional transfer**: train on **6-leg stick insect + Unitree B1
  quadruped (12-D)**. The train set contains two disjoint action spaces (hexapod 18-D and B1 12-D),
  which one camera describes in the same coordinates and a joint-space model has no correspondence
  for. B1 data + render pipeline exist (`data/allocentric/fwd_b1_50hz`, MuJoCo rollout → CoppeliaSim kinematic
  replay, same camera/floor as insect = render-consistent).

  > **The 4-leg stick insect was the intended test body and it does not qualify.** It was built by
  > removing legs from the base scene, so its geometry is a training body's and its commands are
  > that body's corner columns bit-identically; the latent places it **0.578** from the body it was
  > cut from against a chance level of 0.981 (F41). It tests a new *action space*, not a new
  > embodiment. **The test body is now the B1 itself, held out entirely**: backbone trained on
  > insects only, never a quadruped (F43, F45, slide 16). That is the only genuinely different
  > robot in the project, and everything cross-embodiment rests on this single pair.

**Terminology** — "disjoint action space" (Stage 2, B1) **≠** IK-retargeting (Stage 1). IK gives
different *values* in the *same* 18-D space (comparable — proprioception still shares); disjoint =
*different spaces* with no correspondence, so a joint-space model has to be told how the two bodies
map onto each other before it can share anything. That distinction is exactly why the two stages
prove different things. See the caveat under "Why vision rather than proprioception" for how to
word this without overclaiming.

---

## 3. Pipeline

Implemented in `wm/`. Architecture follows LAC-WM; optimisation is scaled to one GPU.
All hyperparameters live in `wm/config.py`.

![Pretraining Pipeline](/doc/images/Pretraining_pipeline.png)


### Encoder — V-JEPA2, frozen

`facebook/vjepa2-vitg-fpc64-256`, ViT-g/16, 1B parameters, weights never updated.

```
frame_t  ∈ ℝ^{256×256×3}          RGB from the sim camera
  → 16×16 grid of 16×16 px patches
  → linear projection + 3D-RoPE positional embedding
  → frozen ViT, self-attention across patches
e_t      ∈ ℝ^{256×1408}           256 patch tokens, 1408 dims each
```

Each frame is encoded **independently** — fed twice into the minimal 2-frame tubelet so the
model acts as an image encoder, not a video encoder. Feeding a real clip would let each frame
see the future through bidirectional attention, so `e_t` would not be independent per timestep
(`scripts/vjepa2_encoder.py`, verified in `scripts/finished/test_vjepa2_frame_isolation.py`).

The encoder **stays inside the training loop**: cross-augmentation needs fresh random views every
epoch, so embeddings cannot be cached. It is the dominant cost — four encoder passes per sample.

Frozen because V-JEPA2 is pretrained on ~1M hours of video and already carries motion-relevant
features; fine-tuning would cost far more compute and risk losing that generality. Step 0 confirms
the features are usable before any training.

### Cross-augmentation

Two independent augmentations `A1`, `A2` are drawn per sample and each is applied to **both**
frames of the pair, so the transition itself carries no augmentation difference:

```
A1 → (x_t¹, x_{t+1}¹)      ITM consumes this pair
A2 → (x_t², x_{t+1}²)      FTM is scored against this one

z_t     = ITM(x_t¹, x_{t+1}¹)
x̂_{t+1} = FTM(x_t², z_t)          L_recon = ‖x̂_{t+1} − x_{t+1}²‖²
```

Without it the ITM can satisfy `L_recon` by smuggling `x_{t+1}`'s content into `z_t` instead of
learning the action; the mismatch between views blocks that shortcut.

Augmentations are **random crop (85–100%) plus brightness/contrast jitter**. Horizontal flip is
excluded: mirroring swaps the robot's left and right legs while the supervised action vector keeps
its original leg order, so the motion target would contradict the image (`wm/data/augment.py`).

### Inverse Transition Model — `wm/models/itm.py`

```
in:  e_t, e_{t+1}  (2 × 256 tokens, projected to width 512)
out: z_t ∈ ℝ^64
```
2 causal self-attention blocks then 2 cross-attention blocks, 16 heads (LAC-WM Table 4 gives 4
blocks total, 512 hidden, action embedding 64). Causal masking means `x_t`'s tokens cannot attend
to `x_{t+1}` — verified: `x_t`'s representation is bit-identical when the future frame changes,
while `z_t` still responds to it. A learned query token then cross-attends to that context and is
projected to `z_t`.

### Forward Transition Model — `wm/models/ftm.py`

```
in:  e_t, z_t   →   out: ê_{t+1} ∈ ℝ^{256×1408}
```
8 blocks, 16 heads, each block: self-attention over visual tokens, self-attention over latent
tokens, then cross-attention from visual to latent.

### Motion Decoder — `wm/models/motion_decoder.py`

```
in:  e_t (visual context), z_t (query)   →   out: â_t      L_motion = ‖â_t − a_t‖²
```
Visual tokens are downsampled by a strided 2D convolution over the patch grid (16×16 → 8×8) to cut
compute, then `z_t` cross-attends to them and an MLP produces the action.

**Shared backbone, one output head per embodiment.** The backbone (4.96M parameters) reads the
behaviour from `z_t` against the visual context and is shared by every body; only the final
projection is embodiment-specific (0.27M each: 18-D hexapod, 12-D quadruped), because action spaces
of different dimensionality have no common coordinates. **95% of the decoder transfers**; adapting
to a new body means fitting a small new head, not retraining the model.

Conditioning on `e_t` is what lets one latent decode to different joint values for different bodies,
and the weights are **kept**, not discarded: the decoder is the only bridge from a latent action back
to executable commands (`policy → z_t → MD → joint targets → robot`). This is the answer to the
week-4 objection *"if the robot needs joint commands anyway, why convert to a latent and back?"* —
the policy learns in the latent space because that part transfers; the decoder does the body-specific
part. The conversion is exactly what separates the transferable from the non-transferable.

### Loss

```
L = λ_recon · L_recon + λ_motion · L_motion          currently λ_recon = λ_motion = 1.0
```
LAC-WM reports no numeric λ. Note the two terms sit on different scales (reconstruction ≈ 1.3,
motion ≈ 0.002 in standardised action units), so equal λ does not mean equal influence.

### Simulator and data

CoppeliaSim 4.10, Bullet 2.78, 20 Hz (50 ms timestep), rendering fixed across every body.

| | |
|---|---|
| Bodies | short 0.5× / medium 0.75× / long 1.0× leg length, built and verified with `sim/scene/make_leg_morphology.py` |
| Action | joint position targets ∈ ℝ^18 (6 legs × 3 joints), radians |
| Clip length | 66 frames (~3.3 s) — one expert episode |
| Episodes | 100 forward-walk episodes per body, from a 1000-episode expert set |
| Behaviours | forward walk only; turn and stop are excluded until they can be collected without a camera/path shortcut |
| Camera | single fixed world-frame side view, 8 m distance, 15° FOV, 40° elevation, 256×256 |
| Framing | `--cam_dx -0.6 --spawn 0 0` — the body stays fully in frame for all 66 frames and the floor edge stays out of view |
| Train / held out | long + short / medium |

The camera is **fixed in the world**, not tracking the robot, so the body visibly travels through a
static frame — that world-frame travel is exactly what a joint encoder cannot report.

### Why `a_t` must differ per body

The decoder is `MD(e_t, z_t) → â_t`. If every body received identical commands, `a_t` would be the
same per behaviour, `L_motion` would trivially force `z_t` to be body-independent, and the decoder
would never need `e_t`. The result would be circular: *"of course the latent is body-independent —
you fed every body the same action."*

IK retargeting gives per-body-different `a_t` for the same Cartesian foot trajectory, so the decoder
must read `e_t` to know which body it is looking at, and a `z_t` carrying pure behaviour becomes an
earned result rather than an artefact of the data.

## 4. Roadmap

> Two stages (see §0/§2). Stage 1 = cross-morphology (3 leg lengths). Stage 2 = cross-embodiment
> (hexapod + B1). **Both are done and settled** -- full dated log in `PROGRESS.md`, measured
> results in `FINDINGS.md` F1-F99. What follows is the status table only; the narrative behind each
> line lives in those two files, not repeated here.

**What stayed open past this era, into the work §0/§1.1-1.3 cover**: Step 1e (an EAC-WM-analogue
baseline, still never run -- §9 still lists it as required, not optional) and Step 2o (a third
embodiment, to make the scaling claim available -- not built, LAC-WM's own headline result needs
3+ embodiments and this project has 2). Everything else below is closed.

**Stage 1 -- Cross-morphology** (3 leg lengths, IK-retargeting, shared 18-D space)

| # | Step | Status |
|---|---|---|
| -1 | Morphology gap check | done -- passed |
| 0 | Visual-encoder sanity | done -- macro-F1 0.886 |
| 1a | Render-lock gate | done, revisit note: passes within the insect family, weaker across insect/B1 |
| 1b | Collect IK dataset | done |
| 1c | Train ITM+FTM+MD | done |
| 1d | Latent validation (two-sided) | done, half-passes: behaviour transfer up, morphology decode stays ~99% (F over-specialisation, not a bug -- see F42) |
| 1e | EAC-WM analogue baseline | **never run** -- still open, §9 |
| 1e' | Invariance ablation | done -- forcing invariance didn't move it, the objective did (F21/F38) |
| 1f | Transfer test, held-out body | done -- F42 |
| 1g | Mechanism (frame vs latent ablation) | done -- F42 |
| 1h | Coverage (femur/tibia) | done -- F42 |

**Stage 2 -- Cross-embodiment** (hexapod + B1, held out entirely as the test body)

| # | Step | Status |
|---|---|---|
| 2a | B1 data (MuJoCo rollout -> CoppeliaSim replay) | done -- superseded several times since by later egocentric collections, see `wm/runs/README.md` for the current live sets |
| 2b | 4-leg insect candidate | dropped -- tests a new action space, not a new embodiment (F41) |
| 2c | Train latent WM across {insect, B1} | done -- F37 |
| 2d | Hold B1 out entirely | done -- F43 |
| 2e | Cross-embodiment validation (shared code vs switch) | done -- produces a switch, not a shared code (F37/F40) |
| 2f | Forward model across robots (frozen) | done -- fails frozen (F44) |
| 2g | Few-shot adaptation of the forward model | done -- F45 |
| 2h | Control: dynamics or manifold | done -- both, and they separate (F47) |
| 2i | Training window (frame_stride/action_chunk) | closed -- wider pair turns z into a clip identifier, kills transfer (F78) |
| 2j-2n | Shared-target widening, behavioural coverage, data balance | done -- F62-F75, superseded by the `beh12_*` matched-condition sets |
| 2o | A third embodiment | **never built** -- still open |
| 2p | Few-shot curve on the current model | done |

> **Two dated corrections from this era, kept only because a pre-2026-08-22 number should not be
> trusted without them**: a frame-rate mismatch made a stored transition 50ms on one robot and 20ms
> on the other until fixed (F65), and a camera-framing bug clipped 67% of all frames until fixed
> (`--cam_dx -0.6 --spawn 0 0`). Both fixed; full detail in `PROGRESS.md`.


## 5. Steps

Every Stage-1/Stage-2 step's full write-up (what it tested, method, numbers, verdict) lives in
`FINDINGS.md` (F1-F99, indexed by the F-numbers in §4's table above) and the dated log in
`PROGRESS.md`. Not duplicated here -- §4's table is the current, compact status; this section used
to hold the long-form version and has been trimmed to stop this file duplicating those two.

## 6. Risks and confounds (Stage-1-era, settled)

Render-style dominance (raw frozen embeddings separating by rendering session rather than
behaviour) and the "why a latent at all, if every body shares one joint space" objection were both
real risks during Stage 1 and are both closed: render-lock was enforced at data-collection time and
verified per-run (`FINDINGS.md` F-numbers throughout Stage 1), and the latent-vs-raw-joint case is
made in §1's Claim section. Full historical discussion in `PROGRESS.md` §5 -- not repeated here.

## 7. Fallbacks (not needed)

Two fallback architectures (HiLAM, UniSkill) were scoped in case Stage 1's latent failed to
organise by behaviour. It didn't fail -- neither fallback was needed. Kept in `PROGRESS.md` as a
record of the decision, not carried here.

## 8. Deployment -- superseded by what was actually built

This section (dated 2026-08-26, pre-egocentric-pivot) scoped the closed-loop demonstration around
the problem "the B1's recorded actions are PPO-policy responses, not replayable sequences" and
asked whether deployment needs a camera at run time. **Both questions were later resolved
differently than scoped here, by the babble/direct-Froude-planner work §0 and §1.2/§1.3 describe**:
motor babble (not recorded expert sequences) is the candidate source being tested (F190-F194), and
the working controller is direct Froude-space action scoring (`body_head(proj(a))`, F119-F127),
not a rollout-based sequence planner. The architecture questions this section worked through (why
not command `(vx,vy,wz)` directly, z-space vs e-space matching, closed- vs open-loop) are still
correct reasoning and are reflected in `wm/policy/planner.py`'s actual design -- read the code and
`doc/SIM_GUIDE.md` for the current implementation rather than this section's now-superseded plan
for reaching it.

## 9. Baselines and references

### What we compare against

- **What we beat**: training from scratch per morphology (no transfer)
- **Metric**: sample efficiency on medium leg — with vs. without pretrained FTM
- No existing locomotion cross-morphology baseline → comparison is transfer vs. no-transfer
- LAC-WM baseline (EAC-WM) is manipulation-only, cannot port directly. **Note the structural
  resemblance**: EAC-WM is defined by per-embodiment action encoders, and our Stage 2 has
  per-embodiment output heads inside pretraining. The difference that matters is the *coordinate*
  those heads decode into, not their existence -- see F60 and step 2j.
- **AMP (RL-trained per-body controller), as a negative-result baseline**: documents that a plausible
  alternative to vision-latent + IK ground truth (train a policy per body against a shared gait prior)
  was tried and produced worse, less coordinated behaviour than the IK route — supports the case for the
  chosen approach rather than being an oversight. Failure-mode videos + gait diagnostics already exist
  (`PROGRESS.md` §13, `results/dataset/amp_failed/`).

---

### The source method, read from the paper (F60)

The full text is `doc/LATENT ACTION ROBOT FOUNDATION WORLD MODELS FOR CROSS-EMBODIMENT ADAPTATION.pdf`.
Read it before writing anything that characterises the method; earlier text in this repo was written
from a summary and got three things wrong.

- **It has an alignment term.** `L = λ_recon·L_recon + λ_motion·L_motion`, where the motion decoder
  is an auxiliary loss whose stated purpose is mitigating shortcuts. Its Figure 2 shows the latent
  space is **disjoint by dataset without it** -- the same experiment as our `lambda_body` control.
- **Motion labels are per-dataset and large**: Droid **10**-D, Agibot **29**-D, EgoDex **147**-D
  (9-D wrist pose plus 60 finger-keypoint dimensions per hand, plus 9-D camera pose). So different
  output widths per embodiment is *their* design too, not our divergence.
- **`z` is split evenly**: first half decodes the end-effector pose, second half the camera pose.
- **Actions are chunked into 5-step sequences** before training.
- **The motion decoder sees the current frame** (`â_t = MD(x_t, z_t)`, `z_t` as a cross-attention
  query over its visual tokens). Safe there because the target is a *delta*; unsafe for a state
  target, which is F57.
- **Per-embodiment mapping happens at finetuning**, via an action projector (raw action → latent),
  trained in three LoRA stages after `z` is already shaped.
- **Cross-augmentation** (two independent augmentations, IDM on one pair, FDM predicting the other)
  -- we already do this.

### Notes from ICLR reviews of LAC-WM
- **V-JEPA2 pixel decoder**: not included in V-JEPA2 — must be trained separately if needed for pixel-space output.
  L_recon = ||ê_{t+1} − e_{t+1}||² is computed in *embedding* space → no pixel decoder required for training.
- **Training scale**: LAC-WM used 64 H200 GPUs × 4 days for 3 manipulation datasets (confirmed App. A.5).
  Our 3-morphology locomotion setting is far smaller — reasonable to train on a single node.
- **Why ICLR rejected**: weak evaluation (1 task, 1 baseline), not because the core method is wrong.
  Professors will likely raise same concern → plan for ≥2 baselines and ≥3 behaviors.
- **EAC-WM degrades with more embodiments**; LAC-WM improves. This is our key supporting evidence.

### Lab resources
- Stick insect model: confirmed — Ajan YuChen's `airl-insect-walking` repo, CoppeliaSim model, migrated to `sim/`
- Data collection policy: ask lab if scripted controller already exists for the model
- P'Beam's work may connect to this in future

## 10. Open decisions

Most items originally in this table are resolved; see §4 for the Stage 1/2 status table and §0 for
current work. What's actually still open, today:

| Block | Status |
|---|---|
| Step 1e -- EAC-WM analogue baseline | **still never run.** §9 calls it required, not optional. Either run it or write down explicitly why it's out of scope for the thesis. |
| Step 2o -- a third embodiment | **still not built.** LAC-WM's scaling claim needs 3+ pretraining embodiments; this project has 2. Named as a limitation, not pursued. |
| Reward-quality gate / RL controller (Q21) | **the current live work** -- see §0. |
| Cross-embodiment stage-3 InfoNCE (F197's opened question) | **designed, not built** -- see §0's Track 2. |
| λ_recon/λ_motion, z_t=64 | never ablated, but no longer load-bearing -- the objective-level question (MSE vs. discriminative) turned out to matter far more (§0, F193-F199) than these specific weights. |
| LAC-WM source code | not released. Rejected ICLR 2026, accepted ICML 2026. |
