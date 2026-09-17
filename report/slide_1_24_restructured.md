# Slides 1–24, restructured against the engineering-report rubric

Current Slides 1–24 are ordered as a debugging narrative: each slide walks through what was tried,
what broke, what got diagnosed, in the order it happened to occur. That's the right order for
`doc/FINDINGS.md`. It is the wrong order for a defense — the committee is scoring against fixed
categories, and the actual experiments (with a clean problem → method → result each) are currently
buried inside that narrative.

This file pulls the same content apart into those categories, and — separately — pulls just the
**experiments** into one timeline so Slide "Experiment Design" and "Result and Analysis" can be two
tables read side by side instead of two slides of prose per finding. Nothing here is new content;
every number is already in Slides 1–24 (or Sections 1–10 in the same file's earlier format). What
changes is which slide it lives on.

---

## 1. Problem Statement

*(Section 1 / Slide 1)*

- RL locomotion policies are **morphology-specific**: a policy trained on one body's joint space
  fails on another; retraining costs hours-to-days per body.
- The analogy this thesis tests: **vicarious learning** (Bandura, 1977) — learning a behaviour by
  watching a different body perform it, with no shared kinematics and no direct instruction.
- The gap, stated as a 2×2 (Slide 1's diagram): everything that crosses leg count needs a kinematic
  body model; everything that needs no body model doesn't cross leg count. The lower-right quadrant
  — crosses leg count, needs no kinematics — is empty. This thesis targets that quadrant.

**One-line problem statement:** *Can a behaviour recorded on one legged robot drive a structurally
different robot, using egocentric video alone, with no kinematic model, no retargeting, and no
controller already existing on the target body?*

---

## 2. Requirements

*(implicit across Section 1, made explicit in proposal.tex §1.2/§1.4 — not yet a slide of its own)*

| # | requirement | why |
|---|---|---|
| R1 | the shared quantity must mean the same physical thing on both bodies without a hand-built correspondence | otherwise it isn't cross-embodiment, it's relabelling |
| R2 | no CAD/URDF, no kinematic tree, no per-robot adapter fitted from that robot's own proprioception | this is the exact thing every existing route (Section 2) supplies and this thesis withholds |
| R3 | readable from **egocentric video alone** at deployment time | proprioception-free is the whole point — an animal or damaged robot has no other channel |
| R4 | the visual encoder stays frozen | isolates the claim to what a frozen, off-the-shelf video model already carries |
| R5 | the representation must be shown to be *used* by the forward model, not just decodable from it | Slide 11 exists because R5 is not automatically satisfied by R1–R4 |

This table doesn't exist yet as a slide — recommend adding it right after the Problem Statement
slide, before Scope, so the committee sees the bar before seeing what clears it.

---

## 3. Scope and Assumptions

*(proposal.tex §1.4, not currently a slide — Slide 1's diagram implies some of this but doesn't state it as scope)*

- Entirely simulation: hexapod in CoppeliaSim/Bullet, Unitree B1 in MuJoCo. No sim-to-real.
- Two-stage design: Stage 1 (hexapod leg-length variants, one joint space) is a **controlled
  prerequisite**, not the claim — it cannot show vision beats proprioception, because all variants
  share one 18-D joint space. Stage 2 (hexapod × B1, disjoint 18-D/12-D spaces) is where the claim
  is actually tested.
- Camera is egocentric by design decision, not accident — and that decision is itself one of the
  measured results (Slide 15), not an assumption taken for granted going in.
- Behaviours are drawn from a **curated library** (forward/turn/strafe at several speeds), not an
  unconstrained babble space, for Stages 1–2; babble is a separate, later extension (Slide 26, out
  of scope for Slides 1–24).
- **Explicitly not claimed:** real-robot transfer, zero-shot cross-embodiment, reliable multi-step
  closed-loop control, a scaling law over many embodiments.

---

## 4. Background Study and Literature Review

*(Section 2, Slides 11–14)*

Two lines, kept separate on purpose:

| line | what it establishes | what it leaves open |
|---|---|---|
| Hu, Chen & Lipson 2025 — egocentric visual self-model | no morphology/kinematics needed, from babble + one camera, per robot | never shares anything *between* robots |
| Huang et al. 2026 (LAC-WM), ICML — latent-action world model | one latent action space unifies several manipulation embodiments | rides on end-effector pose as a task space that's already shared; locomotion has no such space |

Three more, pulled in specifically to diagnose the action-blindness problem (Slides 11–14), not to
motivate the pipeline:

| paper | names |
|---|---|
| ActSWM (2607.26712) | **context collapse** — the predictor extrapolates from context alone and ignores the action channel |
| UWM-JEPA (2605.25313) | same failure from the training side — a teacher-forced target already contains the action's effect |
| AHA-WAM (2606.09811) | adjacent-frame redundancy as a design premise; splits horizon asynchronously to dodge it |

**Positioning statement for the slide:** none of these five bridge locomotion with genuinely
disjoint action spaces using vision only — that gap is Section 5's subject, not repeated here.

---

## 5. Solution Design and System Overview

*(Section 3, Section 7, Section 8 — architecture, not experiments)*

```
   frame ──▶ [frozen V-JEPA2 encoder] ──▶ e_t
                                             │
                       (e_t, e_t+1) ──▶ [ITM] ──▶ z_t  (latent action)
             (e_t, z_t) ──▶ [FTM] ──▶ ê_t+1
             (e_t, z_t) ──▶ [Motion Decoder] ──▶ joint command
                       z_t ──▶ [body_head] ──▶ Froude (forward, lateral, yaw)
```

- **Frozen encoder** (1B params, never trained on robots) + three ~5M-param modules on top.
- **Shared coordinate = Froude number** (`Fr = v / sqrt(g·L)`), dimensionless so a hip-height-0.09m
  hexapod and a hip-height-0.56m B1 land on the same number for "the same walk." This is what makes
  R1 (Section 2) possible without a kinematic model.
- **Two loss terms are the design, not incidental detail:**
  - the cross-body decoder term (Section 4) — forces the *image* to carry body identity and the
    *latent* to carry only movement, fixing the decoder's "recall the nearest training body" failure.
  - `L_body` (Section 8) — forces `z` to be linearly readable into the same Froude coordinate on
    both bodies; without it, cross-robot readout is systematically negative.

---

## 6. Implementation

*(the "how it was built" story — Sections 4–6, 9–10, Slide 21 — kept separate from the experiments that validate it)*

Stage 1 build (hexapod leg-length variants):
1. Data: same foot trajectory, IK-solved per body → different joint numbers, same intent.
2. Baseline decoder found to memorise "nearest training body," not read geometry — diagnosed by a
   probe/decoder comparison and a latent-swap test (Section 4).
3. Fix: add the cross-body decoder loss term (`A's latent, B's frame, B's command`).

Stage 2 build (hexapod × B1):
1. Add `L_body` to force a shared readable coordinate (Section 8).
2. B1 adaptation is a **4-stage pipeline**, assembled from pieces that existed separately and had
   never been run together before (Section 9): `wm.adapt` (ITM/FTM fine-tune) → `fit_projector`
   (action → z, no frame pair) → `wm.adapt3` (optional joint fine-tune, skipped) → `fit_body_head`
   (refit Froude head).
3. **A wrong mechanism was tried first** (Slide 21): `wm.train --init_ckpt` jointly retrains
   everything under the full pretrain loss — B1 got *worse* than zero-shot. This was a tooling
   error (not running LAC-WM's actual adaptation), not a result about the claim. Worth one line on
   an implementation slide, not a full slide — it's a lesson about process, not a finding.
4. `body_head` diagnosed as under-trained, not `z` — fixed with `z.detach()` before `body_head`
   (Section 10), trading a weaker `z` (32–76% of its old signal) for a `body_head` that actually
   converges (~9× better).

---

## 7 / 8. Experiment Design and Result and Analysis — one timeline

Every row below is a real experiment: a stated question, a method, a result — pulled out of the
surrounding debugging narrative. Ordered by when it was run, grouped by what it answers.

### Stage 1 — does the pipeline read geometry, and where does it break?

| # | question | method | result |
|---|---|---|---|
| 1.1 | does the decoder read a held-out body's geometry, or memorise the nearest training body? | probe (4.2k params) vs. motion decoder (5.2M params) readout of coxa/femur/tibia on a held-out body; latent-swap test (give decoder body A's frame + body B's latent) | probe reads geometry accurately (0.84 vs. true 0.80); decoder misreads by 22% and, when swapped, follows the *latent* not the *frame*, matching body B's command to within 6° |
| 1.2 | does one loss term fix it? | add cross-body decoder loss; re-measure command error, image's-worth-to-decoder, and latent composition | error 3.67°→3.44°; image's worth to decoder 0.4×→9.6×; movement share of latent 82%→93% |
| 1.3 | does the fix generalise, or only within the training geometry's span? | same model, held-out body inside vs. outside the training set's geometric span | inside: 90% of ground-truth distance, 5.5° heading error; outside: 13.4° error, R² = −0.34 |
| 1.4 | can this failure be predicted *before* training, from the split alone? | fit a probe on training bodies only, check its error recovering a held-out body's geometry, before training any decoder | error tracks pass/fail exactly — became a pre-training checklist item |
| 1.5 | how much does the actual transition (not just one frame) matter to the latent? | ablate what `e_{t+1}` is fed to the inverse model (true / `e_t` / `e_{t-1}` / random / none) | wrong transition is worse than no transition (1.65× vs 1.34× baseline error) — but one frame alone already recovers most of what a pair does (horizon barely matters, 3° at t vs 2.9° at t+32) |

### Stage 2 — does the coordinate transfer across disjoint bodies?

| # | question | method | result |
|---|---|---|---|
| 2.1 | does `L_body` make `z` cross-body readable? | frozen linear readout, fitted on one body, scored on the other, with/without `L_body` | without: −7.08/−2.36 R² (systematically wrong); with (λ=0.5): +0.54/+0.70 |
| 2.2 | one term, two payoffs — does it cost the robot its own decode quality? | same run, own-joint decode loss vs. cross-robot R² | own-joint loss 0.35→0.22 (38% better) *and* cross-robot transfer positive — not a trade |
| 2.3 | does staged adaptation actually transfer pretrain behaviour to a genuinely different body? | 4-stage pipeline (Implementation §6) vs. zero-shot, on B1 | median ρ 0.264 (zero-shot) → 0.572 (adapted); every channel more than doubles |
| 2.4 | is `z` or `body_head` the bottleneck on Froude readout? | freeze `z`, bolt on a fresh head trained on Froude alone, compare to production `body_head` | fresh head passes the pre-set bar by ~10× — `z` was never the problem; fix is `z.detach()` on the production head, ~9× improvement, real trade (`z` weaker but `body_head` converges) |

### Action-blindness — is the action actually used, and by what mechanism?

| # | question | method | result |
|---|---|---|---|
| 3.1 | is the field's own published fix for action-blindness (context collapse) effective here? | 6 pre-registered interventions: full rebuild, null-action swap, wider frame spacing, residual probing, motion-organised (delta) representation, remove the shared-coordinate term | **all 6 negative** — action worth <3% of next-frame prediction throughout |
| 3.2 | is the mechanism "action is genuinely unrecoverable," or "recoverable but unused"? | linear command recovery from one frame vs. a frame pair | one frame alone: 0.78 insect / 0.16 quadruped — near-total substitution; recoverable ≠ necessary for prediction |
| 3.3 | is the cause gait *rhythm*, or the pose being visible at all? | break the gait two ways: random noise vs. real controller-issued stops/speed-changes/turns | noise: action-sensitivity more than doubles; real commands: **no change** — rules out rhythm as the cause |

### Camera viewpoint — the falsifiable prediction

| # | question | method | result |
|---|---|---|---|
| 4.1 | does removing the agent's own pose from view restore action-sensitivity? | move camera onto the robot's head (with a leak-guard control on room appearance first) | single-frame command readability falls 0.78→0.29; transition's value nearly triples |
| 4.2 | does the fix destroy the one cross-body result already established? | fit shared coordinate on insect egocentric view, apply to B1 with **no refitting** | turn goes from dead (0.07) to strongest channel (0.64); forward/lateral fall but stay usable |
| 4.3 | five-way stress test of what egocentric actually fixed | GATE C (action-dependence), turn readout, order similar actions, order different behaviours, command-from-(frame,z) recovery | fixed: GATE C, turn readout. not fixed: ordering similar actions (47% vs 50% chance). unchanged: ordering different behaviours (already worked). cost: command recovery −14% |

### Selection mechanism — coarse vs. fine, and which half of the loop is broken

| # | question | method | result |
|---|---|---|---|
| 5.1 | does scoring in the shared coordinate (vs. raw frame distance) fix candidate selection? | same-robot and cross-embodiment selection, frame-distance vs. Froude-distance scoring, 1-channel vs. 3-channel | frame distance: 18–23% (28% chance); Froude, no rollout: 76–86% same-robot, 68–70% cross-embodiment (3ch); rollout added back: destroys turning |
| 5.2 | is selection coarse-only, or does it also rank fine magnitude? | family selection (12 conditions) vs. 0.5-sd perturbation ranking | family: works, crosses embodiments. fine magnitude: 47% vs 50% coin — signal is present in the raw embedding (probed directly) but the pipeline can't use it; 6 independent readout fixes, all null |
| 5.3 | 2×2 — is the failure the goal (vision vs. physics) or the scoring mechanism (direct vs. rollout)? | same goal clip, same 12 candidates, 4 modes crossed | goal source is irrelevant (A≡D, both 100%/0.029); **rollout fails regardless of goal quality** (B and C both land on the single farthest candidate) |
| 5.4 | does ITM's transition structure earn its keep, or would any equal-capacity nonlinearity do as well? | linear(e_t,e_t+1)→Froude vs. matched-capacity MLP vs. ITM→z→body_head, same held-out split | linear 0.486, MLP 0.667, ITM 0.736 — most of the gain is "any nonlinearity," a smaller real remainder is ITM-specific, and ITM generalises better than the MLP despite fitting training data worse |

---

## Notes on the last three rubric items

- **Engineering Content in the Presentation / in the Report** and **Presentation Skills** are
  grading criteria about how the material is delivered, not additional content buckets — nothing to
  extract into them. What this restructuring does for those three: every row above already states
  problem → method → result in one line, which is the actual lever for both "Engineering Content"
  scores (a reviewer scores whether the engineering reasoning is visible, and it's currently
  diluted by debugging narrative around it).
- Slide 24 (Gantt timeline) and Slide 20 (controller-vs-selection framing) are **not** experiments —
  keep them as framing/status slides, not folded into the Experiment Design/Result tables above.
- Slides 25–27 (next plan, babble, gecko) are out of this extraction's scope (Slides 1–24 only, per
  your ask) — say if you want the same treatment applied to those.
