# Scoping teacher-student on the egocentric model — measured, not guessed

**Nothing is built here.** F144 failed and F145 gave the mechanism; the question is whether
egocentric removes *those* causes, and that has to be known before a student exists rather than
after. Three answers below, then four prerequisites the gate needs before it can run at all.

---

## Q1 — the F145 gate: can the teacher now rank local perturbations?

**The measurement is defined and its instrument already exists.**
`scripts/diagnostics/teacher_label_quality.py` runs exactly this: at branch points along a held-out
clip, the student's own action and the teacher's pick out of 32 Gaussian perturbations at 0.5 sd are
each **executed in the simulator** for three steps, and the body motion each produced is compared
against the goal. The teacher never grades its own homework.

The baseline it is read against, from F145 on `c10f10t10`:

| | |
|---|---|
| teacher's pick closer to the goal | **4 of 12 = 33%** |
| a coin | 50% |
| mean distance to goal, student / teacher | **0.1299 / 0.1304** |

**The two candidate causes and what separates them, stated before the run.**

| | cause | does egocentric address it? |
|---|---|---|
| (a) | pose-redundancy -- the action contributed nothing to prediction, so the ranking had nothing to rank with | **yes**, and GATE C is the evidence: 1.03 to 1.16 (F172) |
| (b) | the perturbations are physically near-identical -- 0.1304 against 0.1299, and **a ranker cannot order what the outcome does not distinguish** | **no.** Egocentric changes the camera, not the physics |

**F145 flagged (b) itself**, in the note read alongside F147, so this is not a new worry -- it is the
one the gate is built to separate.

**Pre-registered reading.** Clearly above 50% means (a) was the binding cause, egocentric removed it,
and the teacher works. Still near 33-50% means (b) was binding, egocentric does not touch it, and
**teacher-student stays blocked -- which is a finding about the task and gets reported as one, not a
result to iterate quietly against.**

**One control this gate needs and F145 did not have.** The separation statistic itself
(0.1304 vs 0.1299) must be reported again on the egocentric run. **If physics separates the
candidates no better than before, a ranking score above 50% is measuring noise** and the
pre-registered pass has to be read against that, not on its own. Report both or neither.

---

## Q2 — the F137 bootstrap: does the student move at all?

**Not blocking, and F137 is not the mechanism here.** F137 measured *random babble*, which never
walks -- sampled joint sequences stay upright, travel backwards and never become a gait. **The
existing pipeline does not bootstrap from babble.** `teacher_student_insect.py bc` clones the
student on the insect's **recorded frames and commands** first, and that clone is also F144's
control: it walked 0.2349 m, 36% of `D_real`, upright for the full window.

So the student starts from a policy that already walks a third of the reference distance, and the
teacher's job is refinement from there. **The bootstrap exists, it is behaviour cloning, and F137's
problem is not in this path.**

**What does change egocentrically, and it is not a bootstrap problem.** The clone is fitted on
frames, so cloning on egocentric frames is a different and probably harder regression -- Q1's whole
result is that an egocentric frame states less about the command. **The egocentric clone's own
distance has to be measured as the control before the teacher is credited with anything**, exactly
as F144 did allocentrically. That is one simulator run, not a research problem.

---

## Q3 — the reading frame, locked before anything runs

**Student numbers are read against each body's refitted-decoder reference and never against 1.0:**

| | refit reference (F174) | linear floor (F173) |
|---|---|---|
| insect | **0.847** | 0.608 |
| B1 | **0.778** | 0.334 |

**0.847 and 0.778 are what a head that is handed the true `z` achieves.** A student has to produce
its own `z` from a frame, so **it should not be expected to beat them**, and a student landing near
them has done everything the representation allows. The linear floor stays as the lower marker.

**For the closed-loop arm the bar is F142's and is unchanged**: distance against `D_real` = 0.6566 m
with 50% required, upright for the full 3 s, **and the cloning-only control run in the same session**
-- F144's clone reproduced its own 36% exactly that way, which is what made its comparison
trustworthy.

---

## Prerequisites — four, and the gate cannot run until they are met

**1. There is no egocentric teacher.** `load_teacher` wants one file carrying `itm`, `ftm`, `md` and
`projector`. `wm/runs/beh12_ego/best.pt` is a stage-1 pretrain with no projector. The parts exist
and have to be assembled: `best.pt` + `projector_ego.pt` (GATE D2) + `md_refit.pt` (F174).

**Recommendation: assemble from those and do not run stage 3 first.** GATE C, GATE D and F174 were
all measured on `best.pt`; adapting it first would put the gate on a model nothing else has
measured, and stage 3's own seed ordering is still unresolved (`README` on `s2_*`).

**2. The loop still films allocentrically.** `run_in_sim` calls `drive_and_record(..., cam_dx=-0.6,
cam_dy=0.0)` -- the fixed chase camera. **`drive_and_record` already supports `ego=True`**, so this
is threading arguments rather than new code, but the settings must match the collection exactly or
the gate measures the mismatch: **`ego=True`, `cam_fov=90.0`, `ego_box=8.0`**, default `ego_offset`
and `ego_euler`, and `ego_seed` matching the goal clip's repeat index.

**Camera settings are part of the data and are not stored in the npz** -- the loop has to be told,
and getting this wrong is the failure mode this project has hit repeatedly.

**3. The student pools the frame, and egocentric is the case where that hurts most.** `Student`
takes `pooled(e) = e.mean(-2)`: one vector per frame, **spatial layout discarded**. Its own docstring
flags the trap and accepts it for speed.

**F174's mechanism hypothesis is that egocentrically the command lives precisely in where things sit
in the frame** -- that is why a linear ridge on flattened tokens read 0.334 while cross-attention
over the same tokens reached 0.778, a 2.3x gap against 1.15x allocentrically. **A pooled student is
the architecture least able to exploit what makes egocentric work.** This is a hypothesis about a
component that has not been measured egocentrically, and the cheap check is to refit the *pooled*
readout against F174's numbers before a student is trained on it. **If pooling costs most of the
2.3x, the student needs the token grid and the 20 Hz budget has to be revisited.**

**4. The teacher is scope-limited to `c10f10t10`.** F144 records that the allocentric teacher's state
fidelity on `c08f09t09` is 1.052, worse than a frozen frame, and says **do not use it off
`c10f10t10`.** Whether the egocentric teacher inherits that limit is unmeasured; the gate should run
on the body the teacher is valid for and the limit should be re-checked before any other body.

---

## What to run, in order

| | | blocks the next? |
|---|---|---|
| P1 | assemble the egocentric teacher from `best.pt` + `projector_ego.pt` + `md_refit.pt` | yes |
| P3 | pooled-readout check against 0.847 / 0.778 -- one ridge, no simulator | no, but it decides the student's architecture |
| P2 | thread the egocentric camera through `run_in_sim` and render one clip to confirm the view | yes |
| **Q1** | **the F145 gate: ranking, plus the physics-separation statistic** | **decides everything after** |
| Q2 | egocentric clone-only control, distance against `D_real` | only if Q1 passes |

**Nothing after Q1 is built until Q1 reads out.**

---

# Corrected reading frame, and the projector question P3 opens

**The student's bar is the pooled-alone row, not F174's.** 0.847 and 0.778 are what a head *handed
the true `z`* achieves; `Student` is handed `(pooled(e_t), goal)` and never sees `z`, so it cannot be
asked to beat a number measured on an input it does not receive. Allocentrically the pooled-alone row
reads **0.800 insect / 0.250 B1**, and **0.250 was already the B1 student's architectural bound at
F144 and F145 without being measured** -- a bound no teacher and no label quality can move. The
egocentric version of that row is what P3 ego produces and is the number every student result gets
read against.

## "Feed the student `proj(goal)` instead of `z`" -- the idea is right, the operation is not

**`proj` maps an action to `z`, not a goal to `z`.** `ActionProjector` takes an 18-D or 12-D joint
command; the goal is 3-D dimensionless body motion. **`proj(goal)` is not a defined call** -- the
dimensions do not match and there is no head that maps a body-motion goal into the latent. So the
proposal as stated cannot be run, and the useful part of it survives in two forms:

**(a) `proj(a_{t-1})`, the previous action.** Causal, available at run time, needs no new component,
and it is the projector's actual signature. It gives the student a `z` built from what it just did
rather than from the future. **This is the cheap one and it should be measured first** -- one more
row in `pooled_student_check.py`, no simulator.

**(b) a goal -> z head, trained.** This is a new component, and it is the inverse of the body head
that already exists (`z -> (fwd, lat, yaw)`). Inverting a 64-D-to-3-D map is one-to-many, which is
the same identifiability problem F97 recorded for `a -> z` and F131 had to work around. **Not a free
lunch and not to be attempted before (a) is measured.**

**One trap if (a) is run.** The projector must be fitted **against the checkpoint whose `z` is being
used**. `wm/runs/beh12_hex-b1_body3/projector_b1_adapted.pt` was fitted against an *adapted*
checkpoint, so its latent is not the one 0.778 and 0.250 were measured in; using it would compare two
different `z` spaces and read as a result. The egocentric side already has the right file --
`projector_ego.pt`, fitted against `best.pt` in GATE D2.

## Why 0.250 matters beyond architecture

**It is a candidate explanation for F144's failure that is not about the teacher at all.** F145
concluded the teacher could not rank local perturbations; a student bounded at 0.25 from its own
inputs would produce the same symptom -- labels that cannot be fitted, a policy that wanders -- and
the two have never been separated. **Q1 and P3 ego together are what separate them**, which is the
reason P3 runs first.
