# Research Direction — Cross-Morphology Locomotion via Latent Action World Models

> **Role**: The plan as it stands today.
>
> Edited in place, never stacked with updates -- if a step changes, the old text is replaced. For how a step came to be what it is, read `PROGRESS.md`.

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

1. [Claim](#1-claim)
2. [Approach](#2-approach)
3. [Pipeline](#3-pipeline)
4. [Roadmap](#4-roadmap)
5. [Steps](#5-steps)
6. [Risks and confounds](#6-risks-and-confounds)
7. [Fallbacks](#7-fallbacks)
8. [Deployment (out of scope)](#8-deployment-out-of-scope)
9. [Baselines and references](#9-baselines-and-references)
10. [Open decisions](#10-open-decisions)

---

## 1. Claim

Learn a **morphology-agnostic latent action z_t** from simulation video that separates *what
movement is happening* from *which body is doing it*, given no morphology label and no kinematics.

Two claims, at two scopes, because they turned out to need different words:

**Within the hexapod family** (Stage 1) the latent transfers to an unseen body **without
retraining**: `m3d_cross` scores 3.44 deg and R² +0.81 on a held-out body. The mechanism is not
what was expected -- the decoder reads the body from the frame and the latent carries the movement
(F49, slide 5).

**Across embodiments** (Stage 2) the claim is **cheap adaptation, not zero-shot transfer**. A
frozen forward model does not survive the change of robot (F51, 0.57-0.71x on the B1, worse than
predicting no motion), and adapting it on target data was always the design -- the source method
itself finetunes on 7,265 target trajectories. Measured: **one B1 clip clears break-even, nine
clear it at every horizon tested, about 7x fewer target clips than starting cold** (F52).

**Closed loop, in physics** (Stage 2, 2026-08-26/27) is now the third scope and the strongest one.
An **unseen hexapod body** is controlled with the world model **completely frozen** -- only the
two-layer action projector refitted -- at survival **15/15**, behaviour **15/15**, median speed
error **19.0%** over fifteen runs (F95). **With the ten warm-start steps removed** -- they replay
the goal clip's own actions -- it is **14/15 and 58.8%**: the result survives, its margin is
partly inherited, and forward is the only behaviour that *improves* without the hint (F110).

A **quadruped** stands through every episode under the same planner after the world model is
adapted on 24 of its clips, at behaviour-family 38-58% against a 28% chance rate, and **hits none of
three speed targets** (F101).

**Cross-embodiment control** (2026-08-28, refitted on the corrected B1 set 2026-08-29) is the
fourth scope and the narrowest. **Forward and turning both cross; sideways does not.** With the two
stage-3 arms differing only in `--lambda_nce`: forward **53%** under MSE and **54%** with the
contrastive term against 33% chance, turning **22%** and **43%**, sideways 19% and 20% against 17%.
Every run stayed upright and **every turning run turned the right way** -- the first time that could
be measured, since the two robots turned opposite ways until F115.

**Whether the contrastive term helps at all is open again.** Its `/mean-z` goes 0.977 to 0.493, so
it does make the forward model use its action; but MSE given the *original* budget reaches 36% on
turning against the contrastive arm's 43%, inside that arm's own spread, and the two were compared
at different budgets. Both are being retrained matched (F116).

**What the term does, mechanically:** Its `/mean-z` goes 0.977 to 0.493; the collapsed arm still selects forward at 53%, so
**forward can be chosen by a model that ignores the action entirely** and is not evidence the world
model works. F112's 32%-against-74% on forward does not reproduce and was partly a label defect
(F114, F116). The ladder, all arms at
`--warm_start 0 --commit 3`, four goal clips per arm, against a 33% chance rate on the forward
goal: frozen world model with only the projector fitted **5% +/- 0**, adapted separately under MSE
**28% +/- 5**, adapted **jointly** under MSE **32% +/- 7**, jointly with a contrastive term
**74% +/- 3**. **Only the loss term separates the last two** -- same file, same 24 clips, same code
path, and the MSE arm ran 25% more steps. The two do not overlap. **Turning clears chance in no
arm**, the contrastive one included (32% +/- 5). **A world model pretrained on the insect cannot drive the quadruped at
all, and MSE adaptation leaves it at chance** -- the contrastive term is what crosses the gap, which
is F98's mechanism deciding a physics loop rather than a ranking on recorded clips (F112). Goal frames come
from a **hexapod** clip, candidates stay B1 clips because only those are executable, and the B1
walks: **67% / 84% / 71% of planned steps on forward candidates against 33% chance** under three
warm-start settings (a turning clip, a forward clip, and none at all), upright for every step of
every run. **Only forward travel crosses.** Sideways is at or below chance in all three, and turning
straddles chance in all three -- it was reported as crossing and then withdrawn. Over thirteen
cross-embodiment runs the goal's yaw and the robot's yaw correlate at **-0.33 with 46% sign
agreement**: the loop controls *forward or not forward* and does not control direction (F107, F109).
The defensible sentence is **"a quadruped walks forward from an insect's video"**, not "behaviours
cross" -- and narrower still: **the loop transfers the kind of motion, not the amount.** Seven goals
spanning a 1.72x range of commanded Froude produce achieved speeds correlating at **+0.074**, and
the planner does not select faster candidates for faster goals (-0.167) although the library covers
the range. The same loop tracks speed to a median 14.8% when the goal is the same robot (F111).

> **The checkpoint that closes the loop has no shared body target.** `beh12_hexonly` trains on
> `hexapod=data/beh12_c10f10t10_flat` alone, so `lambda_body` supervises forward *within the insect*; stage 3
> trains only the projector and the forward model. A channel screen on that checkpoint transfers
> **nothing**, forward included (-0.112 hex->b1), while forward crosses in the loop at 67-84%.
> **The cross-embodiment result comes from V-JEPA2's features plus stage-3 adaptation, not from
> `lambda_body`.** F83 is not contradicted -- it measured `stage2_*` checkpoints trained on two
> robots, and those sit on pre-F74 data. **No checkpoint is both correct and cross-embodiment**, so
> the loop has never used the mechanism this project identified as what creates transfer. The next
> run is stage 2 on `beh12_hex_flat` + `beh12_b1_flat` against a `lambda_body 0.0` control, then
> stage 3 and the loop on both (F110). Heavy -- fibo7.

**The adaptation objective is a claim in its own right.** LAC-WM's three stages are MSE throughout.
Applied across families that fails in a specific way: the forward model improves its predictions
and **discards the action channel entirely** -- its answer given the real action equals its answer
given the mean action to three decimals, at every checkpoint of a 15k-step run. A contrastive term,
which asks for the ranking a planner performs rather than the prediction MSE asks for, lifts
quadruped selection from 30% to 57% with data, robot, architecture and budget unchanged (F98).
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
  for. B1 data + render pipeline exist (`data/fwd_b1_50hz`, MuJoCo rollout → CoppeliaSim kinematic
  replay, same camera/floor as insect = render-consistent).

  > **The 4-leg stick insect was the intended test body and it does not qualify.** It was built by
  > removing legs from the base scene, so its geometry is a training body's and its commands are
  > that body's corner columns bit-identically; the latent places it **0.578** from the body it was
  > cut from against a chance level of 0.981 (F47). It tests a new *action space*, not a new
  > embodiment. **The test body is now the B1 itself, held out entirely**: backbone trained on
  > insects only, never a quadruped (F50, F52, slide 16). That is the only genuinely different
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

> **Two stages (see Direction update above). Stage 1 = the detailed Steps below; Stage 2 adds the
> cross-embodiment (B1) steps. Target ≈12 weeks, Aug–Oct.** Week numbers are relative from Stage-1 start.

**Where this stands, 2026-08-22.** Stage 1 is finished and retrained clean. Stage 2's two original
goals — a head that fits cheaply on a new robot, and a forward model that predicts on it — are both
met, the second by few-shot adaptation rather than frozen transfer.

**A third goal opened, and is now met at one dimension.** The shared trunk acted as a switch rather
than a common language (F55); no frame-level pairing exists to fix it the way Stage 1 was fixed
(F56). What worked is a **body-motion head shared by both embodiments** (F58) on data where the
insect's speed varies (F57, F60). Cross-robot speed transfer goes from **-7.08 to +0.54 / +0.68 /
+0.75** across three runs, against controls at -7.08 and -2.36, at **no measurable cost** at
`lambda_body 0.1` (F65). Measured as direction agreement rather than R^2, the control reads
**-0.01** and the treated runs **0.85 to 0.92** (F66).

**The behaviour set that was blocking everything now exists.** `data/beh12_*`: twelve conditions per
robot, balanced 4/4/4 across speed, turn and sideways, forward matched to 4% and yaw to 2%, at a
common 20 Hz and 66 frames. Building it exposed four defects in existing code -- a frame-rate
mismatch that made a stored transition mean 20 ms on one robot and 50 ms on the other (F74), a sign
flip that had the robots turning opposite ways under a magnitude-only match (F75), a
proportional-only heading controller leaving a standing yaw bias on the B1 (F78), and a body target
differenced in the world frame, so "forward speed" was partly a rotation measurement once a robot
turned (F79). **Three of the four are fixed; F75's is not** -- measured again on 2026-08-28, the B1's turn
conditions still read yaw +0.0146 / +0.0359 / +0.0760 against the insect's -0.0241 / -0.0372 /
-0.0878, so the two robots still turn opposite ways in `beh12_*` (F115). Every cross-embodiment
turning result compares a left turn with a right turn and could not have succeeded; **every cross-embodiment number computed before 2026-08-22 was
measured across at least the first of them.**

**Untrained, the new channels still do not transfer** -- forward +0.36 +/- 0.10, lateral and yaw at
zero (F76). That is the *before* condition and does not settle the question: forward speed itself
reads 0.31 frozen against 0.85-0.92 trained (F66, F77). **The open question is what training does to
yaw**, and three arms are running to answer it.

**Reading the source paper properly (F67) reframes what to do next, and this is the current plan.**
LAC-WM is not a rival result to beat -- it already showed latent conditioning beats explicit, and
its Figure 2 is our control experiment. Three things follow:

1. **Its motion labels are 10 to 147 dimensions; ours is 1.** Per-embodiment output heads are not
   the divergence -- their label sizes differ per dataset too. **The divergence is the coordinate**:
   they decode into a shared physical space (wrist poses, fingertip positions), we decode into
   body-specific joint angles, which have no common referent across robots.
2. **The obvious richer coordinate is the foot, and our own measurements rule it out (F69).** A
   foot's motion splits into the part that is body speed rewritten -- which adds nothing we do not
   already have -- and the gait itself, which F41b measured transferring at **0.373, below chance**.
   Duty factor fails the other way, nearly identical on both robots (0.533 against 0.515) and so
   carrying nothing to learn. **What a hexapod and a quadruped share is at body level, not leg
   level.** The reason the shared head reached only one axis is that only one body channel *varies*
   in our data: lateral speed is zero in every B1 clip and yaw rate is constant per policy.
3. **Their target is a change between frames; ours was a state** a single frame supplies at R^2
   0.676. That is why their frame-conditioned head fails here (F64) and, on the other side, why the
   alignment left the forward model unmoved at 1.42x against 1.42x -- aligning something the frame
   already gives cannot inform a model that sees the frame. **F68 measured which target form fixes
   this and it is not their chunking**: short windows stay readable (0.670 at W=5), while a
   one-second *forward* displacement drops to 0.246.

**The task-space proposal was raised and withdrawn, and the reason is worth keeping.** LAC-WM's
target is an end-effector pose, so the obvious move is to predict foot positions instead of joint
angles. Two independent objections land on it:

- **Ours, measured.** Foot motion splits into body speed rewritten -- which adds nothing we do not
  already have -- and the gait itself, which transfers at **0.373, below chance** (F41b, F70).
  Everything a foot target adds beyond body speed is the part that does not cross.
- **Ajan Go's, structural (Week 12).** Cartesian targets require knowing the kinematic model and IK
  for every robot. *"หากนำไปใช้กับหุ่นยนต์ชีวภาพหรือสรีระแปลก ๆ ที่หา Kinematic Model ไม่ได้ ไปป์ไลน์นี้จะล้มทันที"* --
  and it weakens this project's own claim, which is that a camera is the only thing a new body has
  to be given.

**The two agree**, and the resolution keeps both: **joint angles for the per-embodiment heads,
body pose delta for the shared head.** The body channel is observed from outside and needs no
kinematic model at all, so widening it costs nothing on either count. What a new robot still needs
is its **joint count** -- a number read off the calibration clips, not a model.

**What is left is therefore behavioural coverage, not a new coordinate**: steps 2j to 2n below. It is
also the condition under which the published frame-conditioned motion decoder becomes runnable here
rather than in the blinded variant F64 forced on us.

**Two scope limits that no amount of retraining fixes.** The B1 side is **14 clips, 5 held out** --
every quadruped number rests on that, and recollecting is the cheapest improvement available. And
step 1e, the EAC-WM analogue baseline, is still never run; §9 lists it as required.

**The one decision left: is a single robot pair enough?** Everything cross-embodiment rests on
insect ↔ B1. A third embodiment would test whether the result generalises, and the natural axis is
leg count — 6 → 4 → **2**, i.e. a biped such as H1 or G1, which the existing MuJoCo-rollout →
CoppeliaSim-replay path could carry. Not Go1: a second quadruped is the same family and tests
nothing new. Cost is days, not hours — new scene, new policy, new render path, and it must be
watched walking before anything is trained on it.


**Stage 1 — Cross-morphology** (3 leg lengths, **IK-retargeting**; shared 18-D space)
*Goal: pipeline works end-to-end + latent organizes by behavior + latent beats raw-joint. Does **not**
prove vision > proprioception (same topology — that's Stage 2).*
*Data-generation route (final, 2026-08-06): **IK retargeting**. An AMP (RL-trained per-body controller)
route was tried in between (full log: `PROGRESS.md` §13) but produced gaits too messy/uncoordinated to
trust as ground truth — kept only as a documented negative-result baseline for the proposal, not the data
source. IK is back to being the primary route; see the Step 0.5 section below for the current setup.*

| # | Step | Tests / produces | Status | ~Week |
|---|---|---|---|---|
| -1 | Morphology gap check | same command → different behavior per leg length | passed | done |
| 0 | Visual-encoder sanity | is the behavior signal present in `e_t`? (foot-contact decodable) | passed, macro-F1 0.886 | done |
| 1a | **Render-lock gate** | domain-UMAP across sessions/bodies must **overlap** (else everything downstream is invalid) | passes between insect bodies, but the test is weak (repeats are re-runs in one session, not independent recordings) and it **fails between insect and B1** — see the framing note below | revisit |
| 1b | Collect IK dataset | walk via IK-retargeting, fixed cam → per-body-different `a_t` | re-collecting as `data/ik_walk_100_framed` (100 episodes, framing fixed) | 1 |
| 1c | Train ITM + FTM + MD | `z_t`=64, fp16, trained on **long + short** only (medium held out) | implemented in `wm/`; 3 runs done | done |
| 1d | Latent validation (two-sided) | behavior transfers **up** across legs **and** morphology decode **down** | **half passes** — behaviour transfer up (+0.11 to +0.22 macro-F1); morphology decode stays **~99%** across all 3 runs | done |
| 1e | **EAC-WM analogue baseline** | latent-conditioned decoder vs one conditioned on the raw joint state | **☐ never run** — see the note below | open |
| 1e′ | Invariance ablation | shrink `z_dim` / adversarial head / centring → does *forcing* invariance change transfer? | done — **three tried, none moved it** (F44); the objective did instead (F24) | done |
| 1f | Transfer test, held-out body | frozen ITM+MD predicts `â_t` from the held-out body's own frames, scored against its real IK actions | done — `m3d_cross` **3.44 deg, R² +0.81**, control 3.67 / +0.79 (F49) | done |
| 1g | **Mechanism, not error** | which pathway carries the answer: ablate the frame, ablate the latent, swap the latent between bodies | done — deleting the frame costs the cross-term run **9.6x** against the control's 0.4x; **swapping the latent moves the answer 0.04 deg** (F49, slide 5) | done |
| 1h | **Coverage** | does filling the femur/tibia gap remove the failure or only soften it? | done — four bodies tying femur to tibia **12.67 deg, R² −0.78**; six decoupled at matched volume **3.27 deg, R² +0.89** (F49, slide 9) | done |

> **Step 1e is the one Stage-1 item never done, and §9 lists it as required, not optional.**
> Nothing here compares the latent against a decoder conditioned on the raw joint state. The frame
> and latent ablations in 1g answer *which pathway inside our model carries the answer*, which is a
> different question from *whether the latent beats the obvious non-latent baseline*. Decide
> explicitly whether to run it or to state in the write-up why it is out of scope; do not leave it
> implicitly covered.

**Stage 2 — Cross-embodiment** (train 6-leg insect + B1; test the B1 itself, held out entirely)
*Goal: answer the committee's "too simple" critique with a real embodiment jump — two disjoint action
spaces, hexapod 18-D and B1 12-D, that one camera describes in the same coordinates.*

| # | Step | Tests / produces | Status | ~Week |
|---|---|---|---|---|
| 2a | B1 data | rollout (MuJoCo) → kinematic replay (CoppeliaSim), render-consistent with the insect | done — `data/fwd_b1_50hz`, 14 clips | done |
| 2b | 4-leg insect candidate | "middle-loss" variant, built by removing ML/MR from the base scene | **abandoned as the test body** — F47: the latent places it 0.578 from the body it was cut from against a chance of 0.981, so it tests a new *action space*, not a new embodiment | dropped |
| 2c | Train latent WM across {insect, B1} | shared ITM/FTM backbone + **per-embodiment Motion Decoder head** (18-D / 12-D) | done — `stage2_clean` and variants; transfer works and the embodiment identity is passive (F43) | done |
| 2d | **Hold the B1 out entirely** | backbone trained on insects only, never a quadruped; fit a 12-D head on a few B1 clips against the same head on a random backbone | done — **1.28x**, velocity-matched, and **all of it travels through `z`**: zeroing the latent gives 0.98x, identical to random weights (F50, slide 16) | done |
| 2e | Cross-embodiment validation | does the shared trunk produce a shared code or a switch? | done — it produces a **switch**: Stage 1's pathology repeats and the adversary narrows but never reverses it (F43, F46) | done |
| 2f | **Forward model across robots** | can a planner's forward model survive the change of robot? | done — **not frozen** (0.57–0.71x, worse than predicting no motion) and coverage does not fix it (5–8% against the decoder's 3.9x). Not architectural: trained on both robots it rolls the B1 at 1.34–1.53x (F51) | done |
| 2g | **Few-shot adaptation of the forward model** | how few target clips does adapting one take, and is insect pretraining worth anything? | done — **one clip clears break-even, nine clear every horizon tested; ~7x fewer clips than from cold**, and the two curves separate rather than converge (F52, slide 16) | done |
| 2h | **Control: dynamics or manifold?** | is the pretraining advantage learned locomotion, or just familiarity with V-JEPA2's feature space? | done — **both, and they separate**. Frozen, real time order beats shuffled on the B1 by 1.38x at one step; after a thousand adaptation steps the two are identical. Dynamics transfer but are overwritten; the foothold in the shared representation is what buys the 7x (F54) | done |
| 2j | **Widen the shared target from 1 DOF to 6** | `lambda_body` supervises forward speed alone. The full shared quantity is a **body pose delta** -- three translation, three rotation, dimensionless. **Joint angles stay as the per-embodiment target** (see the note below on why the task-space version was withdrawn) | open — needs 2k first, since five of the six are constants today |
| 2k | **Behavioural coverage** | **done.** `--gait cpg` gives the hexapod a commandable gait with no IK, ported from the lab's Olaf scene: straight, `--spin` for turning, a sideways gait, and a speed range via `--cycles`, all at `--scale 0.65`. Its drive is a tripod; its contacts are not, and neither are the lab's own (F71, F73) | done (F71) |
| 2k' | **Record body orientation for the hexapod** | **done 2026-08-21** -- `body_quat` per frame, off the abdomen. Head orientation and Euler angles were both tried first and both read straight walking as a large turn (F72) | done, but every clip before this date lacks it |
| 2k'' | **Collect the matched set** | **done** — `data/beh12_c10f10t10_flat` and `data/beh12_b1_flat`, 48 clips each. Hexapod `--spin` 0.05 / 0.15 / 0.29 / 0.56 against B1 `--wz` 0 / 0.055 / 0.139 / 0.294, matched within 3% on dimensionless turn rate with Froude held at 0.12-0.13 on both sides. Speed matched by widening the hexapod's foot path rather than moving either robot's pace (F71, F72) | done |
| 2k''' | **More distinct behaviours** | **partly addressed.** Effective n is the number of behaviours, not clips: within a condition the clips agreed to 2-10% of the between-condition spread. Running the B1's **two** policies at 2.00 and 1.67 Hz gives it genuinely different dynamics inside a condition (F80). The hexapod side still has one gait per condition | open, revisit if the screen stays underpowered |
| 2k''''' | **Why do the channels compete?** | F83: adding yaw costs forward 68% and buys a yaw channel at +0.37 +/- 0.27. **"Yaw carries less signal" was tested and refuted** -- both channels have signal share 0.86 and yaw's between-condition spread is the larger. Capacity and optimisation remain. Cheapest test is widening the body head alone, holding the data fixed. **Separately**, yaw's signal is concentrated in ~6 of 12 conditions, which explains its +/-0.27 to +/-0.56 spread across condition-level splits -- evaluation leverage, not model instability, and more turn levels would tighten it without changing the trade. **And its noise floor cannot be cleaned**: the hexapod's yaw sd is 2.6x the B1's, giving the insect the same PI heading control makes it *worse* at every gain, so the gap is the gait rather than a missing controller (F85) | open |
| 2k'''' | **Train the three arms** | control / body head forward-only / body head forward+yaw, on `data/beh12_*`. Arm 2 re-tests F66's 0.85-0.92 on rate-fixed data (F74); arm 3 asks the question the collection was built for. Commands written; `body_channels` is a config field now. **fibo7, not the 2080 Ti** | **next** |
| 2o | **A third embodiment** | **The single experiment that would make the scaling claim available, and it is not cheap.** LAC-WM's headline is that downstream performance rises with the number of pretraining embodiments while its explicit-action baseline degrades. **Two embodiments cannot produce that curve** (F82), so we currently declare it as a limitation. Pretraining on hexapod + B1 and few-shot adapting a **third** body would convert "our two robots transfer to each other" into "our two robots are pretraining data for a third". Nothing suitable is on disk -- `sim/assets/` holds only the B1 and the hexapod family, and `olaf` is another 18-DOF hexapod, which is within-family and solves the wrong problem. Cost is what the B1 cost: source a model, get a controller that walks, build a render-locked scene on the identical camera, calibrate matched conditions, collect. **Choose for topology, not convenience** -- the thesis argument is *incomparable* topology, so a body sharing topology with neither (a biped, or a different leg count) is worth far more than a second quadruped | open, weeks |
| 2p | **Few-shot curve on the current model** | Slide 16's curve was measured with a Stage 1 backbone. Redoing it needs a **hexapod-only** run with the body head, since `beh12_body_fwd` trained on B1 clips and so cannot hold the B1 out. Smoke-tested: `--sources hexapod=data/beh12_c10f10t10_flat --lambda_body 0.5 --body_dim 1 --body_channels 0` trains, 2779 pairs. Then `finetune_ftm.py --ckpt wm/runs/beh12_hexonly/best.pt --data data/beh12_b1_flat`. **Asks something the original did not**: whether the body term's benefit survives to a robot the model has never seen | ready, 1 run |
| 2l | **Re-test the published motion decoder** | with several futures available from one state, a still stops answering the question, so the frame-conditioned head (F64, −10.5 today) should become runnable and the forward model should move | open, follows 2k |
| 2m | **Balance the data, not just the sampler** | after `clips_per_body hexapod=7` the training set is still **2.43:1 in frames** (2,780 against 1,143), and `balance_embodiments` closes that by repeating the B1 about 2.4 times an epoch. Repetition is not data: it invites memorisation on a validation split too small to detect it, which `config.py` already warns about. **Collect more B1 rather than capping the hexapod further** -- capping throws away bodies, and F13 says bodies are what matters | open, rides with 2l |
| 2n | **Make the measurement read the same data as training** | `body_motion_probe.py` and `plot_z_umap.py` read the whole directory and see **5.9:1**, where training sees 2.43:1. The probe fits each embodiment separately so its numbers are unaffected, but the UMAP layout is dominated by the larger set and every quoted ratio has to say which of the two it means. One data-selection path for both | open — do it *after* the current deck is presented, since it moves the quoted numbers |
| 2i | **Training window** | is one timestep the right pair to train the forward model on? F87 measured the forward model barely reading the latent and ruled out the weighting as the cause | **Yes, and the alternative is worse.** `cfg.frame_stride` widens the ITM's pair and `cfg.action_chunk` widens the command target with it, both implemented and both measured (F88). At stride 10 the forward model's use of `z` **triples** -- 4.257 to 12.279 -- and the joint decoder falls **0.218 to 0.879** while cross-embodiment transfer goes to **zero**. The mechanism is that a wider pair turns `z` into a clip identifier: `probe` rises 0.94 to 0.997. **This confirms F54 and supplies the reason it never had.** The two flags stay in the code, default to off, and are used by nothing | **closed** -- the remaining route to a more useful latent is a shared target the frame does not already supply, not a harder prediction task |

> **Frame rate — read before trusting any cross-embodiment number (F74, found 2026-08-22).**
> The insect records at **20 Hz** and the B1 replay rendered one frame per 50 Hz rollout step, so a
> stored transition was **50 ms on one robot and 20 ms on the other**. The ITM consumes `(e_t,
> e_{t+1})`, so every number computed across the pair — F43/F46 sharing, F51 forward model, F58
> channel AUCs, F45 pairing — compared latents describing 2.5x different durations. Fixed by
> `--fps 20` on the replay (physics untouched at 50 Hz, frames subsampled). `data/fwd_b1_50hz` still
> carries the old rate and is kept only so published results stay reproducible.

> **Data quality — read before trusting any pre-2026-08-07 number.**
> The fixed camera was anchored to the robot's *start* pose with a 0.75 m runway aim, so the
> robot began **outside the right image edge** and walked in: **67% of all frames were clipped**,
> and unequally per body (long 70% / medium 66% / short 58%). Morphology decodability was
> therefore partly measuring *framing*. The floor (only 5×5 m) also intruded into frame.
> Fixed with `--cam_dx -0.6 --spawn 0 0` → 0/66 clipped, no floor edge, and **13,000 usable
> transitions instead of 3,800**. Measured benefit: **3.4× faster learning per gradient step**.
>
> **What this fix does NOT explain.** Held-out-body error tracks **training length**, not framing:
> both good runs sat at ~9.5k steps (clipped 0.166, clean 0.179) while the bad one ran 30.9k
> (clipped 0.422). Clean frames were never trained past 9.5k, so clean-vs-clipped is perfectly
> confounded with short-vs-long training. **Over-specialisation remains the leading explanation for
> the cross-body collapse, and it is untested on clean data** — that run (clean frames to ~31k
> steps) is the decisive experiment. Everything in `PROGRESS.md` §16.
>
> **B1 was never actually render-locked to the insect.** Both cameras anchor to their own
> robot's start, and B1 replays at raw MuJoCo coordinates, so the two stand on different parts
> of the floor: backgrounds differ across **27% of pixels** (insect long-vs-short: 0.29/255).
> Uncorrected, Stage 2's "embodiment decodable from `z`" would have measured background.
> Fixed by giving `render_b1_replay.py` the same `--spawn` / `--cam_dx` / `--travel`.

*Detailed write-ups of each Stage-1 step follow below.*

---

## 5. Steps

Every step follows the same shape: what it tests, how, what came out, and where it stands.
Stage 1 runs Steps -1 through 2; Stage 2 reuses the same numbering on the cross-embodiment data.

### Step -1 — Morphology gap check
**Status** done, passed.
**Goal**: confirm short leg and long leg actually behave differently under same command

| Task | Send identical joint command to short leg and long leg |
|---|---|
| Expected | long leg drags / overshoots, short leg walks normally |
| Pass | visually distinct behavior → morphology gap is real → proceed |
| Fail | identical behavior → leg length difference too small → adjust morphology parameters |

> P'Hap: if same command produces same behavior, the whole experiment is pointless

**RESULT — PASS** (`sim/diagnostics/step_minus1_morphology_gap.py`, 10s run, `step_minus1_comparison.png`).
Controller = the open-loop gait replay baked into `main_script.py`/`ds_loopsm.csv` — morphology-independent by
construction, so no trained policy was needed.

| Metric | short (0.5×) | long / base (1.0×) |
|---|---|---|
| Forward distance | 3.49 m | 4.77 m |
| Body height std | 0.0192 m | 0.0165 m |
| Foot swing clearance (6 legs) | **consistent** ~0.13–0.16 m | **erratic** 0.05–0.38 m |

Front-left foot shows it most clearly: long/base has sharp swing peaks to ~0.39 m; short stays low with a
visibly different rhythm. Identical commands → qualitatively different gait character, not just a scaled
version of the same motion.

> **Step -1 is not merely a sanity check** — it is the evidence base for the whole motivation.
> See "The Motivation Problem" below. These numbers are what justify needing a latent action at all.

**Caveat (honest)**: the PASS is a human read against the plan's own qualitative criterion ("visually distinct
behavior"). The script computes and prints the numbers but asserts no threshold — it is not reproducible as an
automated gate.

---

### Step 0 — Visual encoder sanity check
**Status** done, passed.
**Goal**: confirm frozen V-JEPA2 features carry usable behaviour information before training anything.

**Data**: 3 morphologies × 5 episodes × 200 steps, render-locked. Labels are **6-bit foot contact**
(which feet are planted) — a real body-pose quantity, measured from force sensors. An earlier
`step mod 64` time label was dropped: the gait CSV is a hand-trimmed loop, so it measured an
artefact rather than pose, and it understated the encoder by ~17 points.

| | within one body | across bodies (train 2, test held-out) |
|---|---|---|
| foot-contact decode | **85.1%** | **55.2%** |

Foot contact is highly decodable within a body and **transfers only partially across bodies**
(85% → 55%). That residual is the gap ITM + cross-augmentation exist to close, and it is the
baseline Step 1.5 must beat. Current headline figure: **macro-F1 0.886** (`scripts/finished/step0_macro_f1.py`).

**Morphology is ~100% decodable from raw `e_t`, and that is expected, not a failure.** A 0.5× leg
genuinely looks different from a 1.0× leg; an encoder blind to that would be a worse encoder.
Removing morphology is the ITM's job, not the encoder's. Record it as the **baseline** against
which `z_t` is compared.

> **Report probe *and* silhouette — they answer different questions.**
> Silhouette measures **dominance** (is this the main axis of variation?); a probe measures
> **presence** (is it there and linearly extractable?). They can disagree completely: a signal
> silhouette calls absent (≈0) can be 85–93% decodable, because Euclidean distance in 1408-d is
> swamped by everything else. QWM (App. F-E) reports silhouette only; reporting both is a
> differentiator, and reporting one alone gives the wrong answer.

**Failure condition** (not triggered): `e_t` carries no behaviour information at all → ITM has
nothing to extract. The response would be partial fine-tuning of the last V-JEPA2 blocks, or
revisiting camera framing/resolution (the 16×16-patch caveat vs thin legs). We have seen this for
real on B1 footage, where render style so dominated `e_t` that behaviour signal was undetectable.

**Caveat**: contact labels derive from one animal's replay with a non-smooth loop. Raw forces
are stored, so alternative labels can be tried without re-collecting.

*Abandoned approach, kept only as a warning: per-patch temporal-similarity heatmaps were tested on
three backgrounds and gave no reliable signal (r = −0.16 / −0.20 / −0.006). The problem was the
per-patch method itself, not the background. Blank patches fluctuate **most** — a known ViT
artefact, and the reason the floor must be matte and mildly textured rather than flat.*


### Step 0.5 — Per-body actions
**Status** done. Blocks Step 1: identical commands across bodies make the latent vacuous.

**Verified 2026-07-21**: `a_t` is **bit-identical across all three bodies** in every collected episode.
`np.array_equal` is True for each pair; variance across bodies at fixed `t` is 7.2e-16, i.e. machine
epsilon. Confirmed on `data/step0_v2/{long,medium,short}_ep0.npz`.

**This is correct and intentional for Step -1.** Holding the command constant is exactly what makes the
morphology-gap test valid: identical input, different outcome. The problem is that the Step 0 dataset
inherited it, and **Step 1 cannot run on data with this property.**

**Why identical `a_t` makes the latent action vacuous**

`L_motion = ‖MD(z_t) − a_t‖²` is the only loss that grounds `z_t` to actions. With `a_t` shared:

- MD sees the same target regardless of which body produced the frame, so **nothing pushes it to
  condition on the body at all**. It can satisfy the loss as a function of timestep alone, ignoring `z_t`.
- There is **no retargeting to learn**, because the command was already body-independent before training.

Note what this does *not* break: training still converges, and the factorisation stays self-consistent
(`e_t` carries morphology at 99.9%, `z_t` shared, `a_t` shared). **What breaks is the claim.** Asserting
"`z_t` is a body-independent action representation" invites the immediate reply: *the action was already
body-independent, so what did the model contribute?* There is no good answer.

**What is actually being represented (the question this resolves)**

There is **no joint→joint and no foot→foot correspondence anywhere in the architecture**. `z_t ∈ ℝ^64` is
unconstrained; only the two losses shape it. `contact_8` is the **evaluation label, never a training
signal** — it is the ruler, not the target. The intended factorisation is:

```
e_t  = which body this is        (measured: 99.9% decodable)
z_t  = what movement happened    (hoped to be body-independent; nothing enforces it)
MD   = re-expresses z_t as THIS body's joint command
```

That last line is only meaningful once `a_t` differs per body. **Then** `z_t` means something like
"swing the left-middle leg forward" and MD means "for long legs that is this joint vector, for short legs
another." That is the thesis claim, and it is untestable on the current data.

**Requirement**

Each morphology needs its **own** command sequence for the same locomotion task, so that identical

*behaviour* maps to different *joint values*.
| Route | Cost | Behaviour correspondence | Verdict |
|---|---|---|---|
| **IK retargeting** (Cartesian foot trajectory → `simIK` per body) | zero training | holds by construction | **current route** |
| AMP policy per body | ~0.5–1 GPU-day per body | via a shared discriminator, not by construction | rejected — gaits stayed uncoordinated; kept as a negative-result baseline (`PROGRESS.md` §13) |

IK is not merely the cheaper option, it is the cleaner experiment: it fixes the behaviour and varies
only the joint values, which is precisely the retargeting the Motion Decoder is supposed to discover.
A per-body RL policy gives different `a_t` with no guarantee the behaviours correspond — which is the
failure AMP actually ran into.

**Dataset**: 100 expert episodes × 3 bodies, forward walk only, collected with the framing fix
(`--cam_dx -0.6 --spawn 0 0`; see the data-quality note in the roadmap). Turn and stop are excluded
for now — those clips carried a camera/path shortcut that let a probe separate them too easily to trust.
IK is not merely the cheaper option, it is the cleaner experiment: it fixes the behaviour and varies only
the joint values, which is precisely the retargeting the Motion Decoder is supposed to discover. A
per-body RL policy would give different `a_t` but no guarantee the behaviours correspond, which
reintroduces a confound — which is exactly the failure mode AMP ran into.

**Anticipated objection**: if `a_t` is generated as `IK(trajectory, body)`, is learning MD just learning
IK? Yes, and that is the point worth stating plainly — the claim is that **the model recovers body-specific
retargeting from observation alone, without being given the kinematics**. State it that way rather than
letting a reviewer frame it as circular.

**IK also delivers walk / turn / stop — same task, not a separate one**

The Turn/Stop-behaviours gap (see "Decisions Still Needed") is **not a separate work item; IK resolves it
in the same step.** IK defines a behaviour as a Cartesian foot trajectory, so adding behaviours is adding
trajectories, not building a new system:
- walk = feet cycle forward
- turn = left/right feet cycle at different rates
- stop = feet held stationary

Each is solved per body: `IK(behaviour, body) → a_t`. So a single successful IK pipeline unblocks three
things at once: per-body `a_t` (this section), the K-means(K=3) test in Step 1.5, and the "≥3 behaviours"
answer to the ICLR critique.

Sequencing: **get IK working on walk first** (proves retargeting), then add turn/stop as extra
trajectories. Two cautions when adding them:
- **turn vs drift**: `turn` is *commanded* heading change, but the open-loop gait also *drifts*. The metric
  must separate commanded turning from uncommanded drift (reuse the path-length / net-displacement split
  from Step -1), or `turn` and `walk-that-drifted` will be conflated.
- **stop may be too easy**: a stationary body gives near-static frames the encoder can separate trivially,
  inflating behaviour-decode. Check whether `stop` makes the result look better than it is.

**Gate**: verify `a_t` differs across bodies before any Step 1 training run. Re-run the check above and
require variance well above machine epsilon.

---

### Step 1 — Train the pretraining pipeline
**Status** done. Six runs; per-run metrics in `results/wm/README.md`, analysis in
[FINDINGS.md](FINDINGS.md).
**Goal**: train ITM + FTM + Motion Decoder on short + long leg

| Task | Train on short + long leg data, cross-augmentation on, LoRA off |
|---|---|
| Monitor | L_recon and L_motion both decreasing over training |
| Sanity check mid-training | sample z_t every 10k steps → UMAP should show emerging structure |
| Pass criterion | L_recon converges, L_motion < threshold → proceed to Step 1.5 |
| Fail | loss not converging → check λ weighting, learning rate, data pipeline |

---

### Step 1.5 — Latent validation
**Status** done. Behaviour transfer passes; morphology invariance does not.
**Goal**: prove z_t is morphology-agnostic before testing transfer

| Task | Collect z_t from short + long + medium leg × 3 behaviors |
|---|---|
| Check 1 | UMAP colored by behavior → 3 clusters (walk / turn / stop) visible |
| Check 2 | UMAP colored by morphology → no separation between short / long / medium |
| Check 3 | K-means (K=3) → cluster labels match behavior labels (quantitative) |
| Pass | clusters by behavior, not morphology → proceed to Step 2 |
| Fail | clusters by morphology → fallback to **UniSkill** (see below — *not* HiLAM, which was the wrong fallback) |

> **Result — morphology stays in `z`, and transfer to a new body does not work.**
> Full evidence with reproduction commands in **[FINDINGS.md](FINDINGS.md)**; the short version:
>
> **Body identity is ~99% decodable from `z`** in every run. Its *dominance* falls (silhouette
> 0.034→0.015) but its *presence* never does. This is structural rather than a training failure:
> `L_recon` and `L_motion` both condition on `x_t`, which already carries morphology, so neither
> penalises `z` for carrying it too. Cross-augmentation blocks a *different* shortcut. LAC-WM
> report only UMAP pictures, never a probe, so the same is likely true there.
>
> **What the model does achieve** is body identification from pixels with no morphology label:
> on bodies it trained on it emits the right body's joint offsets to within **0.03–0.06 deg**,
> across bodies whose postures differ by **33.8 deg (CF)** and **50.1 deg (FT)**.
>
> **What it does not achieve** is generalisation. Shown a held-out body, it moves only
> **~45%** of the required distance along the morphology axis (outputs 0.15 where 0.36 is
> correct, on a 0 = `long` to 1 = `short` scale). Tracing the signal: `e_t` places the body at
> **0.465** — the encoder preserves it — and it is lost in ITM (0.301) and the decoder (0.15).
> Cause: the motion loss sees exactly **two** bodies, and two points cannot define a curve.
> Confirmed by correcting the loss weighting (`within_body_std`), after which the model moves
> along the axis but overshoots to 0.61 by epoch 18 instead of converging on 0.36.
>
> **Trivial baselines currently win.** Averaging the two training bodies' commands scores
> **6.68 deg** on held-out `medium` against the model's **10.95 deg**. On fold 2 (`short` held
> out) the model scores 7.00, statistically identical to copying the nearest training body
> (6.96), and loses to linear extrapolation (1.91) by 3.7×. The cause is data design: the shared
> Cartesian foot trajectory makes the three bodies' joint commands **92–99% a constant offset**
> apart. The opening is that the leg-scale to joint-offset map is *nonlinear* — `medium` at leg
> scale 0.75 sits at 0.5 on the scale axis but 0.30–0.36 on the offset axis — so no linear
> baseline can be right, and enough bodies to express that curve would beat all of them.
>
> **Also settled:** standard validation cannot see cross-body failure. Validation motion improves
> **10–11×** over training while the held-out body does not move. And held-out scores need error
> bars: identical config and `seed: 0` on two different GPUs land within 0.3% in-distribution but
> up to **2.1× apart** on the held-out body.

> Note: failure mode = z_t encoding body-shape visual features instead of motion.
> In LAC-WM this was viewpoint clustering. In our work it would be morphology clustering.

---

### Step 2 — Transfer to an unseen morphology
**Status** open, and blocked on collecting more bodies. Zero-shot transfer measured first and
does not currently beat trivial baselines ([FINDINGS.md](FINDINGS.md) F6), so a fine-tuning
sample-efficiency curve measured now would be against a broken starting point. Collect ~30
episodes each across 6–8 bodies before running this.
**Goal**: prove pretrained World Model reduces data needed for medium leg

| Task | Fine-tune ITM + FTM on N medium leg episodes using LoRA rank 2 |
|---|---|
| Condition A | pretrained FTM + N episodes |
| Condition B | scratch FTM + N episodes (baseline) |
| Vary N | 5 / 10 / 20 / 50 / 100 episodes |
| Metric | L_recon on held-out medium leg test set |
| Pass | pretrained reaches same L_recon as scratch with significantly fewer episodes |
| Fail | no gap between pretrained and scratch → z_t did not transfer → revisit Step 1.5 |

> Ajan Go: main claim = "World Model ลด training time อย่างชัดเจน"
> This is interpolation (medium leg is between short and long) — not extrapolation

---

### Step 2.5 — More bodies, more morphology dimensions
**Status** done for the data, one run finished and one running.
**Goal**: test whether coverage is what cross-body transfer was missing

`sim/scene/make_leg_morphology.py` now scales coxa, femur and tibia independently, so morphology is a
volume rather than a line and a held-out body can be a combination no training body has.
`data/fwd_hex8body` holds 7 usable bodies at 30 clips each with 0.0% edge clipping.

> **Coverage is the only intervention that has worked.** Five training bodies instead of two cut
> held-out error from 11.04 to **3.57 deg**, and for the first time the model beats the baseline
> that averages the training bodies' commands (3.57 against 11.48). The latent became far more
> load-bearing: the z-ablation gap went from 3-4x to **10-37x**.
>
> A matched control confirms bracketing rather than data volume is what did it. `m3d_bracketed`
> and `m3d_outside` share everything except which body is held out, and the bracketed body scores
> **10 to 30 times better** at every epoch.
>
> Two other interventions failed: rescaling the motion target, and shrinking the decoder head
> (1.4 to 2.1 times worse over ten epochs).
>
> **The morphology space is 3 parameters but 2 dimensions.** Scaling the coxa moves the joint
> commands by 0.73 deg where the tibia moves them by 28.63, so the family lives in a plane. Two
> numbers place the held-out body to 0.20 deg, which is the ceiling this task allows.

### Step 2.6 — Why it still copies
**Status** done. Mechanism identified and fixed in Step 2.7.

> Crossing the decoder's two inputs shows it takes the body from **`z`, not from the frame**:
> body A's frame with body B's latent yields body B's commands to within **3.48 deg**, where the
> bodies differ by 28.63. `z` itself is 64.1% gait and 11.1% body, so it is doing its job, but a
> probe still recovers the body from it at **0.724** against 0.200 chance, and that small
> component is what the decoder keys off. From the output side, **0.883** of the mixture weight
> sits on one training body.
>
> A lookup over five body codes is cheaper than reading leg geometry from 256x1408 tokens, and it
> has no entry for an unseen body.
>
> **The information is in the frame and the decoder still will not use it.** A ridge probe on
> mean-pooled `e_t`, fitted on the five training bodies, recovers the held-out body's segment
> scales to 0.050, 0.039 and 0.002 — better than the 5.2M-parameter decoder, which implies
> (0.98, 0.98, 0.97) against a true (0.80, 0.90, 0.90).
>
> **Four interventions, none worked.** Rescaling the motion target: no change. Shrinking the
> decoder head: 1.4–2.1x worse. Removing the body code from `z` with gradient reversal: the
> decoder moved onto the frame by 2x and transfer got 1.21x worse. Handing it the pooled view as
> a zero-initialised residual: it used the frame **7.6x less** and transfer stayed level.
>
> Capacity, access and latent content are ruled out. What is left is the objective — `L_motion`
> never asks for the appearance-to-morphology mapping that transfer needs. See OPEN_QUESTION.md
> Q5 and Q6, both answered.

### Step 2.7 — Asking the loss for the mapping
**Status** done. `lambda_cross 0.5` is the fix; the forward model's defect is untouched.
**Goal**: make the objective require what transfer needs, instead of merely permitting it

> **The change**: decode body A's latent against body B's frame, supervised by B's command. Every
> body walks the same expert episodes, so at a given timestep they share the intent and differ
> only in geometry — a lookup in `z` is wrong by construction. No new data. `config.lambda_cross`.
>
> Against the matched control `m3d_bracketed`, held out `c08f09t09`:
>
> | | control | `lambda_cross 0.5` |
> |---|---|---|
> | held-out MSE, epochs 1-10 | 0.0992 | **0.0760** |
> | best checkpoint, degrees | 3.57 | **2.91** |
> | mixture weight on one training body | 0.947 | **0.540** |
> | latent ablation (z-gap) | 21x | 2.2-3.2x |
> | frame ablation (x-gap) | 10.7x | **40-69x** |
>
> First run to beat copy-nearest (3.47 deg) and the first that does not degrade with training. The
> swap test inverts fully: body A's frame with body B's latent now yields **A's** command to 1.18
> deg, against 1.17 when both agree. Stopped at epoch 27, plateaued; best is epoch 8.
>
> **The latent was purified, not emptied.** Contact-pattern decodability from `z` holds at
> 0.744-0.787 against the control's 0.757 (8 patterns, majority 0.144), while the body's share of
> `z`'s variance falls **8.8% to 1.2%** and gait rises 64.5% to 88.7%. The small z-gap is `z`
> shedding the body code. This is what the adversarial head was built for and could not do.
> OPEN_QUESTION.md Q8.
>
> **What is not fixed**: the forward model still does not need `z` (1.03x), because `L_recon`'s
> target is **4.39x more augmentation noise than signal**. Splitting the augmentation shows no
> strength setting recovers it — photometric jitter alone, which moves nothing in the image, is
> still 2.10x the signal, so the frozen encoder is not invariant in the way cross-augmentation
> assumes. Untested options: drop cross-augmentation and rely on the 64-d against 359,000-d
> bottleneck, or augment in embedding space. FINDINGS.md F25.

### Step 2.8 — The target the decoder could already see
**Status** cause found and fixed; the corrected baseline is training.
**Goal**: make the task require the latent it is built around

> `sim/collect/collect_ik.py` applies `cmds[t]`, steps the simulator, and only then captures
> `frames[t]`. So `frames[t]` is the **result** of `actions[t]`, and the command that caused
> `frames[t] -> frames[t+1]` is `actions[t+1]`. Training asked for `actions[t]` from
> `(e_t, e_{t+1})` -- and the Motion Decoder receives `e_t` directly while never seeing
> `e_{t+1}`. **The answer was already in the decoder's own input.**
>
> Measured on the held-out body, replacing the second frame given to the ITM:
>
> | what the ITM is given as `e_{t+1}` | control | cross |
> |---|---|---|
> | the real next frame | 3.57 | 2.91 |
> | **`e_t` again, no transition at all** | **3.96 (1.11x)** | **3.47 (1.19x)** |
> | `e_{t-1}`, the transition backwards | 5.13 (1.44x) | 4.18 (1.44x) |
> | the latent zeroed | 19.24 (5.39x) | 6.04 (2.08x) |
>
> Deleting the transition costs 11-19 percent. Feeding a *wrong* transition costs more than
> feeding none (1.44x for `e_{t-1}`, 2.10-2.70x for a random other frame), so the latent is
> sensitive to the second frame -- having it is simply worth only 11 to 19 percent, because most
> of what the decoder needs is already in `e_t`.
>
> **One fact explains every earlier finding.** The FTM not needing `z` (F23), the decoder doing a
> lookup (F19), `z` being 83-89 percent gait phase (F26), the pose improving while the distance
> did not (F27) -- all follow from the target never having to travel through `z`.
> Confirmed independently: `lambda_recon 0` on real data leaves the reconstruction unchanged
> (0.1025 against 0.0992) while the FTM's loss never moves from its initial value (F30).
>
> **The fix**: `cfg.action_lag`, now 1. The decoder is asked for the command that caused the
> transition, which it cannot see, so the answer can only arrive through `z`. Consecutive
> commands differ by 3.44 deg, which no function of `e_t` recovers. Every run recorded before
> this reads back through `wm.config.from_checkpoint` with `action_lag 0`, so their numbers are
> unchanged -- verified, `m3d_cross` epoch 8 still scores 2.91 deg.
>
> Cross-augmentation **stays on**. The earlier case for dropping it rested on `z` improving
> `L_recon` by only 3-7 percent, measured while `z` had no job. Now compressing `e_{t+1}` into
> `z` is the cheapest way to satisfy `L_motion`, and the same compression makes `L_recon`
> trivial: one shortcut paying into both terms is exactly what the augmentation blocks.
>
> Running: `lag1_ctrl` and `lag1_cross`, matched on everything but `lambda_cross`. The number
> that decides it is the latent ablation, 21x in the old control; it should rise sharply, and
> the held-out error should get **worse**, because the task is genuinely harder now.

### Step 2.9 — Testing the diagnosis instead of asserting it
**Status** done, and it is the strongest Stage 1 result. Four bodies tying femur to tibia score
**12.67 deg, R² −0.78**; six decoupled bodies at matched data volume score **3.27 deg, R² +0.89**.
Filling the named gap **removes** the failure rather than softening it (F49, slide 9).
**Goal**: turn "the failure has the shape of a gap in the data" into a prediction that can fail

> Step 2.8 ends with an explanation. This step makes it falsifiable. In all four bodies the
> tibia-short split trains on, the femur and tibia carry the same scale, and the model answers
> that they are equal for a held-out body where they are not. **If that is the cause, adding
> bodies where they differ should remove the failure.**
>
> Three such bodies now exist and walk: `c10f10t08`, `c10f09t07`, `c10f08t06`, all decoupling the
> two segments while staying inside the reach limit below. Collected at the full 30 episodes with
> the same framing as every other body.
>
> **Registered before the run**, on the same held-out body that failed at 27.68 deg:
>
> | | before | after |
> |---|---|---|
> | interpolation floor | 19.58 deg | recompute on the real clips |
> | threshold for "reads the geometry" | 15.7 deg | 15.7 deg |
> | model | **27.68 deg** | **under 15.7 if the diagnosis holds** |
>
> The run is **volume-matched** -- seven bodies at 17 clips each against four at 30 -- so a
> success cannot be attributed to more data. A second prediction costs minutes and no GPU: the
> encoder probe's error on the held-out body should fall from 0.172 toward the 0.030 it reaches
> on a bracketed body.
>
> If it fails, coverage is not the whole explanation, which is a sharper result than the one we
> have rather than a weaker one.

### Step 2.10 — What the simulator will and will not give us
**Status** measured, and now enforced in the generator.

> Generating morphologies is bounded by a constraint that had not been noticed. A two-link leg
> cannot place its foot closer to its own shoulder than `|femur - tibia|`; below that the knee
> would have to fold past straight. The collector pulls every foot target to half the hip-to-foot
> distance, and the closest target across all 30 episodes sits at **92.5 mm**.
>
> So `|femur - tibia| < 92.5 mm` decides whether a body can walk at all, and it is a step rather
> than a gradient: a body 2 mm past the line misses 0.3% of its targets and walks, one 40 mm past
> misses 24% and returns IK residuals of 350 to 810 mm.
>
> `sim/scene/make_leg_morphology.py` now refuses to generate a body that violates it, printing the
> margin, with `--force` to override. Three of the first six bodies were infeasible; the rule
> would have caught all three before any collection.
>
> **What this bounds**: the decoupling axis is usable in both directions but not arbitrarily far,
> and `c10f10t06` -- the held-out body of the tibia-short split -- is itself 2 mm past the limit,
> so no feasible body can bracket it exactly in segment-scale space. The command-space floor is
> what the experiment turns on, and that is unaffected.

### Step 3 — Extrapolation
**Status** measured once, out of proposal scope for the write-up.
**Goal**: test morphology outside training range

> Fold 2 (`--train_morphs long medium --heldout_morph short`) is extrapolation: leg scale 0.5
> sits outside the 0.75–1.0 training range. Held-out error was **flat at 6.93–7.07 across 41
> epochs** while validation improved 12.8×, and the model scored the same as copying the nearest
> training body.
>
> With five training bodies this changes: `m3d_outside` improves **0.78x** from early to late
> epochs where the two-body run was flat at 1.02x. Coverage helps outside the hull too, just far
> less than inside it (10-30x worse than the bracketed body at every epoch).

---

## 6. Risks and confounds

Things that can make a clean-looking result meaningless. Each has a mitigation that must be in
place *before* the measurement it threatens.

### Render-style dominance

**The finding** (`scripts/_archive/umap_domain_check.py`, logged in `PROGRESS.md §5`): three videos of the **same
behavior** (walking), rendered by three different setups (white bg / IsaacSim grid / MuJoCo checkerboard),
produced **three completely non-overlapping UMAP clusters** of whole-frame `e_t`.

**What it means**: raw frozen V-JEPA2 `e_t` is currently more sensitive to **rendering style** (background,
lighting, engine) than to **behavior**. This is expected for a pretrained encoder with no cross-augmentation —
and the V-JEPA2 paper offers no help here: **VideoMix22M contains zero simulated data**, and the paper never
studies rendering-domain gap. Our finding is unexplained by, but not contradicted by, the paper.

**Why it is dangerous**: if camera / lighting / background differ *at all* between morphology recording
sessions, then **Step 1.5 measures which session a clip came from, not morphology vs. behavior.** The result
would look clean and be meaningless. This confound is invisible unless controlled for up front.

**Mitigation — mandatory, at data-collection time:**
1. **Lock the render environment**: identical camera pose, lighting, and background across **every** morphology
   and **every** behavior session. Vary *only* the robot's legs and its motion. Nothing else.
2. **Background choice** — avoid both empirically-found failure modes: **no checkerboard** (aliasing → fake
   motion signal) and **no blank/flat surface** (ViT register-token noise — blank patches fluctuate *most*).
   Prefer a matte, mildly-textured, non-repeating surface.
3. **Gate before Step 1.5**: encode N frames from each morphology session and run the domain-UMAP. Clusters
   **must now overlap**. If they still separate by session, the environment is not locked → data is invalid.
4. Cross-augmentation is designed to suppress exactly this kind of nuisance — but note the caveat below: body
   shape is *real content* that survives crop/color-jitter/flip, so augmentation is **not** a substitute for
   controlling the environment.

> Corroborated independently by `deep_research.md`'s own construct-validity critique: *"Latent space analysis
> must show locomotion-relevant structure, not visual artifacts."* Written for the abandoned direction, still
> bites this one.

---

**Three independent sources converge on the same objection. It is currently unanswered.**

1. **Ajan Blink, Week 4** (recorded in `feedbacks/feedback_ajan_go.md:21-23`): *"if the policy and robot
   ultimately need joint-space commands, why convert to a latent/frame space at all, adding a converter back?"*
   — **raised, never answered, still open.**
2. **LAC-WM's actual premise**: its embodiments have **genuinely disjoint action spaces** — 10D Franka EE /
   20D bimanual humanoid EE / **138D** human-hand keypoints / 25D BFA. Its EAC-WM baseline is *architecturally
   forced* into per-embodiment action encoders **because of that heterogeneity**. **Our 3 variants share an
   identical 18-dim joint space and identical DOF.** That motivation evaporates — and so does the baseline's
   pathology (our natural EAC-WM analog would just use *one shared* encoder).
3. **`deep_research.md` CP-002**: explicit morphology-conditioning may suffice for **interpolation within a
   family**; implicit/latent approaches are motivated for **extrapolation**. We are interpolation-only
   (medium sits between short and long — Ajan Blink made this exact correction).

**A reviewer will ask: "why do you need a latent action if every body takes the same 18-dim command?"**

### Motivation: why a latent action at all

**The reframing**
Not **action-space heterogeneity** (we have none) but **dynamics heterogeneity**: identical joint commands
produce *materially different motion* depending on leg length. **Step -1 already proved this** — same command,
3.49 m vs 4.77 m, and swing clearance 0.13–0.16 m (consistent) vs 0.05–0.38 m (erratic).

So the latent action's job here is **not** to reconcile differing action dimensionality — it is to unify the
**effective dynamics mapping** from identical joint commands to different resulting motion/appearance across
leg lengths, kept physically grounded by the motion-decoding loss so it can't collapse into a no-op identity.

**Be explicit in the writeup that this is a reframing, not what LAC-WM tested.**

**What counts as evidence**

**Not a raw loss comparison.** Comparing `F(e_t, z_t)` against `F(e_t, a_t)` one-step is unfair to
the baseline: `z_t` is inferred by the ITM from `(e_t, e_{t+1})`, so it has already seen the target
frame, and it is 64-dim against `a_t`'s 18 — a free information and capacity edge that lowers its
loss regardless of any real transfer benefit.

Three things carry the argument instead:

1. **Held-out-body motion prediction with latent ablations.** Predict an unseen body's joint
   commands from its video alone, and compare against zeroed and shuffled `z`. Measured on Stage 1:
   **0.18 with `z` vs 1.67 ablated** — the latent is doing the work, not the frame.
2. **Adaptation efficiency** — episodes needed to reach a target error, pretrained-on-`z` vs
   pretrained-on-raw vs from scratch (N = 5/10/20/50/100).
3. **The availability argument** — the correct `a_t` for a new body requires its kinematics via IK,
   which is privileged information; `z_t` comes from vision alone. A tie already favours the latent.

`F(e_t, z_t)` vs `F(e_t, a_t)` vs `F(e_t, 0)` stays as an early **diagnostic** (Step 1e), with the
observation-only control isolating the action's contribution.

> **On the two-sided probe.** The ideal is that `z_t` raises cross-morphology behaviour transfer
> *and* lowers morphology decodability. On Stage 1 **only the first half holds**: behaviour transfer
> improves by +0.11 to +0.22 macro-F1 while morphology stays ~99% decodable in every run. Report both
> halves; the second is a finding, not a gate. The structural reason is in Step 1.5.

**Extrapolation beyond the training range**
He explicitly corrected short+long→medium as *"just Interpolation."* Currently **no out-of-range morphology
exists**. `sim/scene/make_leg_morphology.py` makes this nearly free — generating a **1.25×** (or 0.35×) 4th variant
answers him with one command. Cheap, high-value.

---

## 7. Fallbacks

### HiLAM — not the right fallback for this failure mode

Original plan was: freeze ITM → dynamic-chunk `z_t` sequences → skill-level `z^h`, hoping `z^h` clusters by
behavior more cleanly than flat `z_t`. **After reading the paper (`doc/2603.05815v1`), this is a mismatch.**

- HiLAM solves **temporal abstraction** ("existing LAMs... focus on short-horizon frame transitions... capture
  low-level motion while overlooking longer-term temporal structure") — **not embodiment invariance**.
- Its chunking operator is a boundary rule over feature dissimilarity between **temporally-adjacent tokens
  within a single video**. There is **no cross-embodiment alignment objective anywhere in it** — no mechanism
  to disentangle nuisance (body) from behavior, no contrastive/alignment term across agents.
- Therefore: if `z_t` already encodes morphology strongly, chunking **pools existing features** and would build
  a *separate* skill hierarchy per body — **inheriting and reifying** the morphology clustering, not fixing it.
- Its experiments are 100% LIBERO tabletop manipulation. **No locomotion. No code released.**

**HiLAM is only the right fallback if the failure mode is "z_t captures only short-horizon kinematics and
misses longer behavioral structure" — a different problem than the one we fear.**

### UniSkill — the correct fallback

**UniSkill** (Kim et al. 2025, CoRL) — *"Imitating Human Videos via Cross-Embodiment Skill Representations"*.
- Explicitly targets **cross-embodiment skill representation** — exactly the failure mode of Step 1.5.
- Telling detail: **HiLAM itself uses UniSkill's IDM/FDM as its frozen submodules.** The cross-embodiment
  property HiLAM borrows comes from UniSkill; HiLAM's own contribution (hierarchical chunking) is orthogonal.
- → If `z_t` clusters by morphology, go to the paper that solves *that*, not the one built on top of it.

**Secondary option worth considering**: **DiLA** (Zhang et al. 2026, *Disentangled Latent Action World Models*)
— content/structure disentanglement, aimed at keeping body-specific visual features out of the behavior latent.

**Action**: read UniSkill before Step 1.5 runs, so the fallback is ready rather than discovered under pressure.

---

## 8. Deployment (in scope, as a demonstration)

Deployment is the eventual use: a controller that makes a new body walk. **Brought into scope
2026-08-11**, on the judgement that there is time to close the loop. Several pretraining decisions
were already only correct or incorrect relative to it, so it was recorded here throughout.

**State, 2026-08-26. The loop is closed on the hexapod and cannot be closed on the B1 as built.**

    hexapod    78% speed within 15%, 100% behaviour class, 100% survival over 56 planned steps,
               three behaviours, nine runs (F92). Requires a warm start: from a standstill there
               is no motion in the frame to read and it picks a turn, 1 in 5.

    B1         blocked. The planner's candidates are recorded action sequences; the B1's gait is
               a PPO policy reading state at 50 Hz, so its recorded actions are responses rather
               than plans and 0 of 8 replay to the end (F93). Not a gap in the method -- ranking
               and selection are both measured on B1 latents and the forward model adapts to it
               in three clips. What fails is re-issuing a recorded sequence.

    held-out    `c08f09t09`, the body every Stage 1 result withholds, with the world model
    hexapod     entirely frozen. Refitting only the action projector -- a two-layer MLP, minutes,
                no gradient through anything else -- gives **6/6 behaviour class, 6/6 survival,
                median speed error 19.2%** against 37.1% with the trained body's projector (F95).
                Speed accuracy does not recover: 2/6 against 7/9. The expensive component
                transfers; the cheap one needs clips of the new robot and nothing else.

**So a cross-embodiment demonstration needs one of two things, and neither is free**: command the
B1's own policy in `(vx, vy, wz)`, which concedes the task-space question this project exists to
avoid; or give the planner closed-loop primitives per robot instead of sequences, which adds the
per-robot component the recorded-sequence design was chosen to avoid. **A held-out *hexapod* body
is the cross-body test that stays inside the design**, and needs `data/beh12_*` collected for a
second body.

**What it can and cannot claim, decided in advance so the result is not over-read.**

The forward model does roll the world forward -- 1.38x better than a frozen world at one step,
1.47x at three, 1.20x at ten (F32) -- so candidate B has something to plan with. That was measured
only after we noticed we had been scoring the module on action reconstruction, which is not its
job (F30, F32).

But **F31 makes planning easy on this data for the wrong reason**: one frame nearly determines the
joint command at every horizon out to 32 frames, because the gait is periodic and a single frame
fixes the phase. A selector choosing between candidate latents therefore faces an almost
deterministic problem. **A working demonstration here shows the loop closes; it does not show that
planning is what made it work.** Distinguishing those needs data whose future is genuinely open --
varying speed, turning, terrain, disturbance -- which is exactly the gap F31 identifies and the
AMP policies in `amp/logs/` are a candidate source for.

Two further constraints carried over: match candidate rollouts in **z-space, not `e`-space**, since
V-JEPA2 encodes morphology strongly and `e`-distances between bodies are confounded by shape; and
the loop must be **closed**, because open-loop replay of a demo latent sequence desynchronises on a
differently-timed body.

So the deliverable is: **the loop runs end to end on a held-out body, and we state plainly what the
data lets that mean.**

### Does this need a camera at run time? No, and the answer has a name

**The objection.** LAC-WM is a manipulation method, and manipulation has a camera by default -- a
wrist or head camera is present at training *and* at deployment. Our camera is worse than that: it
is **fixed and third-person**, chosen for render-lock between the two robots. A legged robot out
doing a task cannot depend on an external observer, so if the pipeline needs frames to act, it is
not deployable as built.

**Resolution, two parts, and they compose.**

**Onboard camera.** Egocentric is realistic for legged robots, and it is what EgoDex and Agibot
already use -- only Droid is third-person, and it zeroes its camera channel because of it. Moving
onboard has a bonus that is not a coincidence: **an egocentric camera's motion *is* the body's
motion**, so their camera-pose channel and our body-pose channel become the same quantity. The one
coordinate we measured to be shared is exactly the one their camera already labels.

**Teacher-student distillation.** Train with vision, act on proprioception: the vision-conditioned
policy is the teacher, and a proprioception-only student is trained to match it. Standard practice
in legged locomotion, and it settles the framing question this project keeps running into:

> **Vision is the medium that lets one model span incomparable bodies *during learning*. Nothing
> requires it to be the sensor at run time.**

That sentence is also the honest answer to "why vision for locomotion at all", which has been asked
at every review: the claim was never that a camera is the right runtime sensor for a walking robot.
It is that a camera is the only thing a *new body* has to be handed, where proprioceptive methods
must be handed a kinematic tree.

**Consequence for the current setup.** The fixed third-person camera stays for now -- it is what
makes the two robots render-comparable, and swapping it invalidates every measurement in the deck.
It should be described as **a research instrument, not a deployment configuration**, and the
egocentric recollection belongs with the behaviour collection in 2k rather than as separate work.

### What the execution loop requires

```
policy --> z_t --> Motion Decoder --> 18 joint targets --> robot
                   ^ the module pretraining was going to throw away
```

### What transfers and what does not

| Component | Role | Transfers across bodies? | Evidence |
|---|---|---|---|
| `z_t` + ITM | "what is being done" (behaviour) | **the thesis bets yes** | to be tested in Step 1.5 |
| FTM | dynamics in embedding space | partially, fine-tune | Step 2 |
| **Motion Decoder** | `z_t` to joint commands, body-specific | **no** | needs `a_t`, which the new body may not supply |
| **reward / physics heads** | latent to reward | **no** | force→velocity is R²=+0.926 within a body but −0.33 to −5.23 across bodies (PROGRESS.md 10.14) |

The shape that falls out: **a shared behaviour latent, with body-specific encoder and decoder heads.**
This is close to what L3P arrived at from a different direction (frozen backbone, per-robot heads),
which is mild evidence the decomposition is the right one. Our version differs in having a world model
and a latent action inferred by an inverse model, neither of which L3P has.

### Three candidate architectures

| | Method | Needs a reward model? | Main risk |
|---|---|---|---|
| **A** | Dreamer-style imagination RL: policy trained purely inside FTM | **yes** | reward is not readable from our latent (PROGRESS.md 10.14), and the tracking camera hides world-frame progress |
| **B** | Planning in embedding space, V-JEPA2-AC style: sample candidate `z`, roll out with FTM, pick the one minimising distance to a goal embedding | **no** | locomotion is cyclic, so "goal state" is awkward to define |
| **C** | Use `z_t` or `e_t` as a pretrained feature space for ordinary RL on the real environment | no, reward comes from the environment | least novel, but most robust |

Current preference: **B is the most interesting and sidesteps the reward problem** (and V-JEPA 2-AC
already demonstrated this exact architecture on real hardware). **C is the safe fallback.** A is the
riskiest given what we now know about reward readability.

### What this requires of pretraining

1. **Keep the Motion Decoder weights.** Corrected above.
2. **Keep logging `a_t` for the held-out morphology.** Already done. Needed to test whether the MD
   generalises rather than assuming it does.
3. **Keep logging world-frame head position.** Already done. It is the only reward label available,
   since the tracking camera removes world-frame progress from the image.
4. **New experiment worth adding to Step 2** (cheap, data already collected):
   *does the Motion Decoder generalise across bodies?* Train MD on short + long, then ask it to decode
   `z_t` into joint commands for the medium body.
   - Worth knowing either way, but **not a blocker**: F81 puts the MD outside the runtime path.
   - What a new body actually costs is the **action projector**, fitted on its own actions. "A new
     body needs only video" overstates it, and overstates LAC-WM too -- the source paper adapts
     "through finetuning". Video is what lets the world model span incomparable bodies; the
     projector still needs target-robot actions, which F52 measured at one clip to break even.

### What has to be built, and in what order (scoped 2026-08-22)

**Nothing of the loop exists yet.** Every component so far consumes two ground-truth frames and
reports what the robot *did*: `predict_actions.py` says so outright. There is no selector anywhere
-- no module that chooses a `z`. The pieces that do exist and are reusable are the trained ITM/FTM/MD,
the sim stepping code inside `collect_ik.py`, and `latent_rollout.py`, which already scores the FTM
on "if I apply this, what happens next" -- the question a planner asks.

**Two constraints set the order, and both are already measured.**

*The encoder cannot run the loop.* F40: the vision path is **94.9 ms, 10.5 Hz** on a 2080 Ti,
against data at 20 Hz and legged control that wants 50. Vision-in-the-loop is therefore a research
instrument, not a runtime configuration -- which is the argument for distillation, but distillation
needs a teacher, and the teacher is the vision loop. **The slow loop has to be built first even
though it is not the deployable artefact.**

*The fixed camera bounds an episode.* It sees about 2.1 m, which is why every collector carries a
`--travel` gate. A closed-loop run is a few strides, not a traverse, until the camera moves onboard.

**Before the loop, one measurement that re-presents what we already have (F82).** LAC-WM's section
5.2 conditions the forward model on observations from one embodiment while feeding action embeddings
derived from **another**, and scores the generated frames in pixel space -- PSNR, LPIPS, FID, FVD.
That is the same quantity as our cross-embodiment readout, in a medium that can be **looked at**:
a B1 frame rolled forward by a hexapod's latent. With the body head at +0.76 against a control at
-28.9 (F83), the pixel version is the slide. The FTM and both datasets already exist, so this is a
script rather than an experiment, and it does not wait on the closed loop.

**Order of work, rebuilt around the action projector (F81):**

1. **Train the action projector.** `a_t -> z_t`, fitted so it matches `ITM(e_t, e_{t+1})` on the
   same transitions. Small network, data already collected, one per embodiment. **This is the piece
   that makes control possible at all** -- the ITM needs the next frame and so can never run in the
   loop.
2. **Close the loop on the body it was trained on.** Sample candidate actions, project, roll the
   FDM, score, execute the winner. Same robot, no transfer. The point is the machinery: if the loop
   cannot hold a gait on the body it learned from, cross-body is not worth attempting. Score in
   **`z`-space, not `e`-space** -- V-JEPA2 encodes morphology almost completely, so matching raw
   embeddings across bodies is confounded by shape.

   **And score at behaviour scale, not per frame.** `z_t` describes a *transition*, so its fast
   component is where the robot is in its gait cycle -- and F56 measured that the two robots have no
   correspondence there at all (phase concentration 0.07-0.24 against 0.99-1.00). A planner matching
   latents frame by frame would be asking a quadruped to be at the same point of a hexapod's stride,
   which is not a thing. F70 measured the consequence directly: forward reads **-1.45 per frame and
   +0.54 stride-averaged**. **What crosses is what the robot is doing, not how it is moving its
   legs** -- so the target robot supplies its own gait, and the claim is "the new body performs the
   same behaviour with its own gait", never "the new body walks like the source". The latter is
   impossible when no dimension of one action space corresponds to any dimension of the other, and
   the former is the stronger claim anyway.
3. **Close the loop across bodies.** Fit a projector on the target robot's actions -- this is
   LAC-WM's finetuning step, and the honest cost of a new body. Only here does the OOD risk below
   apply.
4. **Distil to a proprioception-only student.** The deployable artefact. The vision path runs at
   **10.5 Hz** (F40) against control's 50, so vision-in-the-loop is a research instrument by
   measurement, not by preference.

**Two things this ordering drops**, both from an earlier scoping that assumed the Motion Decoder
would be the controller. *Sampling in latent space* is out: a sampled `z` need not correspond to any
executable behaviour, where a sampled action does by construction. *"Does the MD generalise across
bodies"* is no longer a blocker: in this architecture the MD is an auxiliary loss during pretraining
and is not in the runtime path at all. It remains worth measuring, but nothing waits on it.

**What is deliberately not being done yet:** moving the camera onboard. It would make the
camera-pose channel and the body-pose channel the same quantity, which is the right end state --
but it invalidates every render-locked measurement in the deck, so it waits until the current
results are defended.

### One closed loop, not open-loop replay

Earlier framing had two deploy paths (imitation vs RL). **Collapsed to one closed-loop controller.** Reason:
open-loop replay of a demo `z`-sequence on a differently-timed body **desynchronises**. `z_t` is a *local
transition*, the new body's timing differs, so the demo's `z_t` (e.g. swing) lands on the body's actual `e_t`
(e.g. still stance) — an `(e_t, z_t)` combination the decoder never saw in calibration (**OOD**) → wrong /
unstable command. Gait is cyclic, so the drift compounds. This is the classic imitation distribution-shift
problem; the fix is closed-loop.

**The loop:** `e_t (real state) → z-selector → z_t → Motion Decoder → a_t → body → e_{t+1} → …`. The selector
reads the body's **actual** state every step, so phase cannot drift.

**Reward is optional — it is only the selector's objective** (this unifies the A/B/C candidates above):
- **match a demo in latent space** — *no reward*; pick `z_t` whose FTM rollout best tracks the demo target.
  This **is candidate B (planning)** — already the doc's preferred option. Caveat: matching `e`-embeddings across
  bodies is **confounded by body shape** (V-JEPA2 encodes morphology ~100%), so match in `z`-space, not raw `e`.
- **maximize a task reward** — RL in imagination, adds reward model + Critic. This **is candidate A (Dreamer)**.
- **behaviour-matching RL (2026-07-25, current preferred variant)** — feed the demo's `z_target`, execute on the
  new body, **re-encode the achieved transition with ITM → `z_achieved`**, and train the decoder by RL with
  reward `r = −‖z_achieved − z_target‖²`. Two wins: supervision is in **`z`-space (morphology-invariant → no
  cross-body confound)** and needs **no ground-truth `a_t` labels**. Cost: it is **RL, not backprop** — `a_t →
  e_{t+1}` is real physics (non-differentiable), and the FTM can't substitute (it is `z`-conditioned, not
  `a`-conditioned). Use a **sample-efficient** method (CEM / off-policy SAC), **not PPO** (on-policy, sample-hungry
  → fights few-shot). This is the scheme drawn in `report/pipeline_diagram.tex` and on deck Slide 24. The
  match-demo/reward options above remain valid alternatives — keep all three on the table.

**Decoder calibration is unchanged, still offline and once:**
- a few logged `(e_t, a_t)` rollouts of the *new body*; `e_t`↔`a_t` time-aligned *within* this set. ITM/FTM
  reused frozen; only the decoder is adapted.
- **Independent of, and NOT time-synchronised with, the demo/target.** The demo only needs consecutive frames.
- **Coverage, not duration.** The decoder is reliable only on the `z`-region calibration covered; span the
  deploy behaviours (walk / turn / stop). Minimum count = the adaptation-efficiency sweep (N = 5/10/20/50/100).
- **Do not use the new body's own rollout as the target behaviour** — decoding its own `z` back is a trivial
  round-trip (no transfer). Transfer means something only for a target we have **no** new-body actions for.

**Open question / ablation — cross-behaviour generalisation.** Calibrate the decoder on **forward-walk only**,
then drive a **turn** objective: does the new body turn? Asks the decoder to **extrapolate** to motor patterns
it never saw paired with commands. Plausible if `z` is local / per-step and turning's per-step foot targets
fall within walking's range; fails if turning needs joint coordination walking never produced. Either outcome
is publishable. **Safe default: calibration spans the deploy behaviours; treat walk-only → turn as a measured
ablation, not an assumption.**

---

## 9. Baselines and references

### What we compare against

- **What we beat**: training from scratch per morphology (no transfer)
- **Metric**: sample efficiency on medium leg — with vs. without pretrained FTM
- No existing locomotion cross-morphology baseline → comparison is transfer vs. no-transfer
- LAC-WM baseline (EAC-WM) is manipulation-only, cannot port directly. **Note the structural
  resemblance**: EAC-WM is defined by per-embodiment action encoders, and our Stage 2 has
  per-embodiment output heads inside pretraining. The difference that matters is the *coordinate*
  those heads decode into, not their existence -- see F67 and step 2j.
- **AMP (RL-trained per-body controller), as a negative-result baseline**: documents that a plausible
  alternative to vision-latent + IK ground truth (train a policy per body against a shared gait prior)
  was tried and produced worse, less coordinated behaviour than the IK route — supports the case for the
  chosen approach rather than being an oversight. Failure-mode videos + gait diagnostics already exist
  (`PROGRESS.md` §13, `results/dataset/amp_failed/`).

---

### The source method, read from the paper (F67)

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
  target, which is F64.
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

| Block | Status | Note |
|---|---|---|
| λ_recon, λ_motion | **open** | LAC-WM reports no numeric λ (nor LR, optimiser or schedule). Currently 1.0 / 1.0, but the terms sit on different scales — reconstruction ≈ 1.3, motion ≈ 0.002 — so equal weights do not mean equal influence. Ablate. |
| `z_t` dimension | 64, following LAC-WM §4.2 | Also the lever for the invariance ablation: shrinking it should evict morphology first, since `e_t` already supplies body identity to both losses. |
| Turn / stop behaviours | **collected, and it was not the lever** | `data/beh12_c10f10t10_flat` and `data/beh12_b1_flat` hold twelve matched conditions per robot -- four speeds, four turn rates, two lateral levels each side -- forward matched to 4% and yaw to 2%. **Collecting them changed nothing on its own**: on the frozen encoder the new channels still read zero, and yaw only moves from -5.2 to +0.37 when a loss term supervises it. The variety bought the possibility; the term realises it (F83, slide 14). |
| Step 2 evaluation metric | **settled and measured** | Sample efficiency, pretrained against from-scratch, at N = 1/3/5/7/9 clips rather than episodes -- the B1 set has 14 clips, so 10 is the largest budget that leaves a clean held-out four. Reported for the action head (F50) and the forward model (F52). |
| Baseline exact setup | **open, and it is Step 1e** | Section 9 calls an EAC-WM analogue required rather than optional, and it has never been run. Either run it or write down why it is out of scope. |
| **Pipeline: Dreamer or candidate scoring** | **open, and it is the decision Week 13 asked for** | The advisor's note is that we are sitting between two designs and should commit. *Candidate scoring* needs an answer to "where do the candidates come from that is not itself a controller" -- ours is a recorded library, the source paper's is a pretrained VLA sampled with 500 noise seeds, and **both are circular**. *Dreamer / teacher-student* moves the search into training: the reward is *reach this latent*, so no library and no kinematics are needed, the run-time cost drops from twelve forward-model rolls to one, and the student is trained on the states it actually reaches rather than on recorded ones. **The evidence collected on 2026-08-26/27 points at the second**, with the caveat that it leans on long-horizon rollout accuracy, which is this project's weakest measurement -- so it should be started on the hexapod, where the forward model holds to half a second, not on the B1, where it does not. |
| **Where candidate diversity comes from** | **answered for the source paper, open for us** | The advisor asked how a fixed policy yields several candidates. LAC-WM's answer, from their §6: they inject different noise seeds into a VLA's flow-matching head to produce **N=500** sequences, having found that *"random action sequence sampling... is inefficient for action optimization, especially for a difficult dexterous manipulation task"*. Ours is twelve recorded clips. **Motor babbling is the proposal that removes the dependency**, untested. |
| **The research gap, stated in one place** | **open, and the most important item** | See §1 and F98. Draft: LAC-WM demonstrates cross-embodiment latent actions where a **shared task space already exists** -- end-effector pose is meaningful for every arm in their corpus, so the alignment problem is solved by choosing the coordinate. Locomotion across leg counts has no such space: 18 and 12 joint targets share no dimension. Two things follow that are not in the source work, and both are measured -- **(a)** a joint-space target crosses only when a shared body-motion term is present (-28.9 to +0.61), and **(b)** MSE adaptation is insufficient across families because the forward model discards the action channel; a discriminative term is required (19% to 57%). Confirm the framing with the advisor before building on it. |
| LAC-WM source code | not released | Rejected ICLR 2026, accepted ICML 2026; no public code. |
---

- Fine-tune last 2 V-JEPA2 blocks if Step 0 shows sim gap?
  - Note: V-JEPA2 paper only ablates **fully frozen vs. fully unfrozen** — partial/last-N-block fine-tuning is
    *not* tested anywhere in it. Reasonable extrapolation, but no empirical backing to cite.
- k = 64 (LAC-WM §4.2) or ablate for locomotion?
- λ values? ablate from equal weights (paper gives no numbers — see Decisions table)
- Baseline: an EAC-WM analogue is a required comparison, not an optional one. See section 6.
