# START HERE

**Read this file first, then `direction_plan.md`.** Everything else is reference: `FINDINGS.md` is
the evidence (F1-F117, append-only), `PROGRESS.md` is the narrative history (append-only),
`OPEN_QUESTION.md` holds what is unresolved, `SIM_GUIDE.md` is how to run the simulators and collect
data.

## What the project is trying to do

Learn a **morphology-agnostic latent action** from simulated video, so that a behaviour recorded on
one robot can drive a **different** robot with an incomparable body -- an 18-DOF six-legged stick
insect and a 12-DOF Unitree B1 quadruped. No kinematic model, no retargeting, no shared joint space:
the only thing the two robots share is what a camera sees.

The pipeline is LAC-WM ported to legged locomotion. A frozen V-JEPA2 encoder turns frames into
embeddings; an inverse model reads a latent `z` out of a transition; a forward model predicts the
next embedding from `(e_t, z)`; an action projector maps a robot's raw joint command into that same
`z`, which is what lets the world model be driven at control time when the future frame is not
available. Planning is candidate scoring: roll the forward model on each recorded behaviour's
actions and keep the one whose prediction lands nearest the goal.

## Where it stands, precisely, on 2026-08-29

**Two results are solid and neither is about the quadruped.**

| | |
|---|---|
| the encoder carries morphology | a ridge regression on frozen V-JEPA2 embeddings recovers an unseen body's segment scales; nothing supervises it (F20) |
| the world model trained on top does not use it | body A's frame with body B's latent yields body B's commands; `lambda_cross` is the intervention that fixed it (F18-F24) |

**Every B1 number is withdrawn.** The quadruped's dataset had four defects, all found on 28-29
August by watching preview videos rather than reading tables: the robot clipped by the image edge in
61% of frames, an unpinned camera, a forward clip filed as the weakest turn level, and turns running
opposite to the insect's. The data is now corrected; **all B1 checkpoints were deleted and stages 1,
2 and 3 have to be rebuilt from `data/beh12_b1_flat`.**

**What the withdrawn runs pointed at, worth keeping as hypotheses:** forward selection works even
when the forward model demonstrably ignores its action input, so forward is not evidence the world
model works; turning is the only behaviour where a model that uses the action beats one that does
not; sideways fails on every measurement.

## The immediate next steps

1. Rebuild stage 1 (`wm/adapt.py`), stage 2 (`wm/fit_projector.py`), stage 3 (`wm/adapt3.py`) on
   `data/beh12_b1_flat`. About two and a half hours.
2. Three seeds per stage-3 arm at one budget -- `scripts/com7_stage3_seeds.sh` -- because the
   MSE-vs-contrastive ordering flipped between two budgets on single runs.
3. Close the loop and re-measure. `sim/control/close_loop_b1_physics.py`, defaults already correct.

## Things that will mislead you if nobody says them

| | |
|---|---|
| **name the insect body, never "hexapod"** | `beh12_c10f10t10_flat` is pretrained on, `beh12_c08f09t09_flat` is held out. They turned opposite ways for a week because every table said "hexapod" (F117) |
| **camera settings are part of the data** | `--cam_fov 24 --spawn 0 0 --floor_scale 3`. A loop that differs from its adaptation set in any static way measures that difference |
| **behaviour-family accuracy cannot see direction** | a run that turns the wrong way scores identically to one that turns the right way. Report sign separately (F109) |
| **chance is 33%, not 8%** | the twelve conditions hold unequal families. An unadapted model scores the chance rate exactly |
| **MuJoCo repeats, CoppeliaSim does not** | rerunning a B1 configuration returns the identical number; spread must come from different goal clips. The insect is the other way round (F105) |
| **a finding marked "fixed" may not be** | F75's sign flip was recorded as fixed and was not, and shaped four later findings before anyone re-measured (F115) |
| **watch the videos** | six defects this project found were caught by looking; none by the tables that were passing at the time |

## Where this sits in the literature

| approach | what it needs on the new robot | generates or selects? |
|---|---|---|
| **LAC-WM**, STORM, World Action Planner | a pretrained, competent VLA to propose candidates | **selects** -- it reranks an existing policy's output |
| **Li et al. 2020**, hexapod + quadruped latent planning | separate expert demonstrations per robot | generates, but as two separately trained latent spaces rather than one shared one |
| **X-Morph** | a URDF and a kinematic retargeting stage | generates, by solving correspondence with a body model |
| **QWM**, morphology-conditioned world model | morphology parameters | generates, zero-shot *within* the quadrupedal family; never crosses leg count |
| **CAPE / CD-LAM** | -- | the precedent for this project's contrastive term: removing it makes the predictor ignore the action query |
| **this project** | 24 recorded clips today; unlabelled interaction is the target | selects today, generates is the plan (teacher-student, Q16) |

**The gap being claimed** is that no published method learns a shared latent action space that
transfers across legged robots with **different leg counts** from video alone. Methods that avoid a
kinematic tree stay inside one robot or one leg-count family; methods that span leg counts use
explicit retargeting.

**The contribution is three things, and they are not equally proven.**

| | what it is | status |
|---|---|---|
| **1. joint targets, no kinematics** | the action space is raw joint commands -- 18-D and 12-D, disjoint, nothing commensurable between them -- and the correspondence is *learned* by the action projector rather than defined by a URDF or a shared task-space coordinate. **This is the axis that separates the work from everything else in the table**: X-Morph retargets kinematically, LAC-WM unifies quantities that already mean the same thing on both bodies (a fingertip at `(x,y,z)`), and morphology-agnostic proprioceptive control has to be handed the kinematic graph. A camera is handed nothing | implemented and runs end to end; its cross-embodiment evidence is withdrawn with the rest of the B1 numbers |
| **2. a joint target crosses robots only with a body term** | within one robot the joint decoder works unsupervised (0.35 error); across robots it is -28.9 / -43.1 without the term and +0.61 / +0.57 with it (F82, F83). **The conditional is the finding**, not "joint targets work" | the A/B contrast survives its data's defects, since both arms carry them; **the figures do not** -- forward walking only, and a frame-rate mismatch (F74) |
| **3. the adaptation objective** | MSE adaptation makes the forward model discard the action channel across morphology families; a contrastive term restores it. CAPE reports the same failure mode elsewhere, so what is new is that it appears across morphology families and is invisible in the loss curve -- **not** the technique | the most fragile of the three: the ordering flipped between budgets on single runs, and three seeds per arm is outstanding |

**Claim 1 is the one to lead with** and the one the literature table is built around. `Q17_ANS.md`
works the positioning out in full; it is a draft the user owns.

**Two things are still overclaimed there and are flagged in that file**: "a camera is the only thing
it needs" is untrue while the candidate library is 24 curated clips, and the shared-body-target
result (F83) was measured on forward walking only and on data with a frame-rate defect, so its
numbers cannot be quoted about current work.
