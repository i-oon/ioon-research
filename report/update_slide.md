# Progress Update — Cross-Morphology and Cross-Embodiment Latent Action Models

Stick insect (*Medauroidea extradentata*) and Unitree B1, simulated in CoppeliaSim.

| part | slides | what it covers |
|---|---|---|
| **Motivation** | 0a-0c | why morphology-specific control is expensive, what every existing route needs from the new body, and why answering this requires two bodies whose command spaces share nothing |
| **Part 1 — Stage 1** | 1-8 | one six-legged topology, several leg geometries: the controlled first step, and the three requirements it establishes |
| **Part 2 — the gap and the attempt** | 9-12 | what crossing embodiments requires, the field's own proposed fix rebuilt faithfully, and six independent measurements of why it does not work here |
| **Part 3 — the principle** | 13-15 | pose determines the future, so the action is redundant: what that explains in the literature, and in our own three failed attempts |
| **Part 4 — the prediction, tested** | 16-17 | removing the body from view restores action-sensitivity, the shared coordinate survives the change, and what we claim as contributions |
| **Part 5 — where this stands** | 18-30 | behaviour selection from a recorded library, the diagnostic arc, a controller that now genuinely walks, and a second controller attempt (imitation) that does not yet |

**The arc, in one line.** We could not make the world model use the action, so we measured why; the
answer turned out to be a property of the **viewpoint** rather than of the model; that property
predicted a fix which prior work had already adopted without explaining; and building on the result
eventually produced a controller that walks. **This is not "we tried things until one worked."**

**Terms used throughout.** *Latent action* — a 64-number code inferred from a pair of observations,
standing in for the command that caused the change between them. *Shared coordinate* — forward,
lateral and turning speed made dimensionless by body size, so the same number means the same
behaviour on a 0.09 m insect and a 0.56 m quadruped.

**Citations are separated from contributions throughout.** Each claim slide states what prior work
found, what we measured where they did not, and what is ours.

---

## Slide 0a — Why morphology-specific control is expensive, and what it would mean to fix it

**The problem.** Legged robots are typically controlled by policies learned through RL. A policy
learned for one body does not generalise to another — lengthen or shorten the legs, change the mass
distribution, change the skeleton's topology, and the policy fails. A new one is trained from
scratch, hours to days per body. Biological organisms share locomotion principles across vastly
different body plans; a shared, body-independent representation of *movement* may be possible.

**The precise analogy already has a name.** Psychology calls this **vicarious learning** (Bandura,
1977) — acquiring a behaviour by observing another agent perform it, with no direct instruction, no
first-hand demonstration of the observer's own body doing the task, and no requirement that the
observed model resemble the observer. That is exactly the transfer under test here: a held-out
body's controller, informed by video of a different body's behaviour — one it does not resemble and
has never itself performed — with no kinematic account of either body supplied to bridge them. The
behaviour crosses through what is *observed*, not through anything told to the model about the
bodies involved.

> **บทพูด (TH).** ปัญหาคือ policy ที่เทรนให้หุ่นตัวหนึ่งใช้กับหุ่นตัวอื่นไม่ได้เลย เปลี่ยนความยาวขา
> เปลี่ยนโครงกระดูก ต้องเทรนใหม่ทุกครั้ง เป็นชั่วโมงถึงวัน ในขณะที่สิ่งมีชีวิตใช้หลักการเดินเดียวกัน
> ข้ามร่างกายที่ต่างกันมาก แนวคิดที่ตรงที่สุดมีชื่อในจิตวิทยาอยู่แล้วคือ **vicarious learning** — เรียนรู้
> พฤติกรรมจากการ**ดู**ตัวอื่นทำ ไม่ต้องมีใครสอนตรงๆ ไม่ต้องเคยทำเองมาก่อน และร่างไม่จำเป็นต้องเหมือนกัน
> นี่คือสิ่งที่งานนี้ทดสอบเป๊ะๆ

---

## Slide 0b — Every existing route hands the model privileged information about the new body; one result does not, but stops short

**Every existing route around morphology-specificity assumes access to something about the target
body** that the motivating scenario (an animal, a damaged robot, hardware acquired with no published
kinematics) does not provide:

| approach | what it needs from the new body |
|---|---|
| QWM | a CAD or URDF description read from the robot's design files |
| graph/transformer universal controllers | a kinematic tree — which joint connects to which |
| L3P | an observation encoder and action decoder fitted per robot from its own proprioception and foot force |
| LAC-WM | a task space (e.g. end-effector pose) that already means the same thing on every body |
| Demo-JEPA | demonstrations of the same task, paired across both bodies |

**One result drops the requirement entirely — the starting point for this thesis.** Hu, Chen, and
Lipson (2025) learn a task-agnostic visual self-model for a legged robot from a single egocentric
camera and random motor babbling, with no prior knowledge of morphology, kinematics, or task, and
use it to plan locomotion and recover from physical damage. **But the self-model is fitted to one
robot at a time** — each body is babbled and modelled separately, nothing is shared or transferred
between bodies. **Extending that premise across embodiments is what this thesis attempts, and it is
the step nobody in this literature has taken.**

> **บทพูด (TH).** วิธีที่มีอยู่ทุกวิธีต้องได้ข้อมูลพิเศษเกี่ยวกับหุ่นตัวใหม่ก่อนเสมอ — CAD/URDF, โครงสร้าง
> ข้อต่อ, ข้อมูล proprioception ของหุ่นนั้นเอง, หรือ demonstration ที่จับคู่ไว้แล้ว มีงานเดียวที่ตัด
> ข้อกำหนดนี้ทิ้งได้จริง (Hu et al. 2025) — babble หุ่นตัวเดียว ไม่รู้ kinematics เลย แต่ใช้ควบคุมและ
> ซ่อมแซมตัวเองได้ **แต่ทำทีละตัว ไม่เคยแชร์อะไรข้ามหุ่นเลย** — งานนี้คือการเอาแนวคิดนั้นไปทดสอบข้ามหุ่น
> ซึ่งไม่มีใครในงานวิจัยที่ผ่านมาทำมาก่อน

---

## Slide 0c — Why this needs two genuinely disjoint action spaces, and what is learned

**Why the leg-geometry variants (Stage 1) cannot answer this alone.** Several leg-geometry variants
share one 18-D joint space, so a proprioceptive model *could in principle* be shared across them too
— vision's advantage there is convenience, not necessity. Answering whether a vision-only latent
action can unify locomotion where **proprioception has no shared coordinate to begin with** needs a
second body whose action space is genuinely disjoint from the first.

```
  hexapod (18 actuated joints, CoppeliaSim)     Unitree B1 (12 actuated joints, MuJoCo)
          │                                              │
          └──────────── share no dimension, no correspondence ────────────┘
                                     │
                    a single egocentric camera describes both
                    in the same 256x256x3 pixel array regardless
```

**What is actually learned is not an opaque latent action, but a shared body-motion coordinate**:
dimensionless forward, lateral, and yaw velocity — the same three physical quantities on both
robots, inferred from egocentric video alone, no morphology label, no kinematic model or URDF for
either body, no manually defined joint correspondence. Learning this coordinate is one problem;
*using* it to drive an unseen body is a second, harder one — a substantial part of this thesis is
the diagnostic work separating what the resulting world model can do (coarse, single-step
action-conditioning) from what it cannot yet do reliably (fine-grained ranking, multi-step rollout),
and the measured mechanism behind that gap (Parts 2-4).

**The significance, twofold.** Scientifically: whether a body-independent notion of locomotion
behaviour can be learned from vision alone, across bodies whose action spaces cannot be reconciled
by any coordinate choice without a kinematic model — and, along the way, a diagnostic methodology
for telling whether a video world model is *using* an inferred action at all, not merely able to
decode it. Practically: if such a coordinate transfers cheaply, it cuts the cost of controlling a
new robot by reusing behaviour already observed on a different body, rather than retraining from
zero or requiring that body's engineering specification.

> **บทพูด (TH).** Stage 1 ตอบคำถามนี้เองไม่ได้ เพราะหุ่นทุกตัวใน Stage 1 ใช้ joint space เดียวกัน (18-D)
> ต่อให้ไม่ใช้ภาพก็แชร์กันได้ในหลักการ ภาพเลยแค่สะดวก ไม่ใช่จำเป็น ต้องมีหุ่นสองตัวที่ joint space ไม่มี
> ความสัมพันธ์กันเลยจริงๆ — หกขา 18 ข้อต่อ กับสี่ขา 12 ข้อต่อ — กล้องตัวเดียวอธิบายทั้งคู่ได้ในรูปแบบ
> เดียวกันเป๊ะ **สิ่งที่เรียนรู้จริงๆ ไม่ใช่ latent ลึกลับ แต่คือพิกัดการเคลื่อนที่ร่วม** (เดินหน้า/ไถลข้าง/
> เลี้ยว แบบไม่มีหน่วย) ความสำคัญมีสองด้าน: ทางวิทยาศาสตร์ (พิสูจน์ว่าเรียนรู้พฤติกรรมข้ามร่างได้จากภาพ
> ล้วนๆ ไหม) และทางปฏิบัติ (ถ้าทำได้ ควบคุมหุ่นตัวใหม่ถูกกว่าการเทรนใหม่ทั้งหมด)

---

# Part 1 — Stage 1: the controlled first step

**The question.** Can a model learn, from video alone, a representation of *movement* that is
independent of *which body* is moving — and then convert it back into the correct joint command for
one specific body?

**The scope.** Stage 1 varies leg geometry only: every body has six legs and the same eighteen
joints, so the command spaces already correspond. Stage 2 removes that correspondence entirely
(Slide 0c).

```
  STAGE 1 — the easy case: command spaces already match
      measures three things, each of which Stage 2 then has to satisfy again:

   1  a shared representation must be FORCED by the objective,      Stage 2 needs the same forcing,
      not bought with capacity or architecture        (Slide 4)  ─▶  with no shared command space
                                                                    to apply it in

   2  "morphology" is several INDEPENDENT axes; coverage must be     check coverage per axis,
      checked per axis, and can be checked in advance (Slides 5-6) ─▶ not per body count

   3  a metric can SATURATE and hide what the model is not doing;    test the mechanism you
      test the mechanism, not a proxy for it          (Slides 7-8) ─▶ depend on, directly
```

Stage 1 never closes the loop. It is the controlled experiment that justifies attempting Stage 2.

---

## Slide 1 — The pipeline: one frozen encoder, three small trained modules

```
   frame ──▶ [ frozen visual encoder ] ──▶ observation
                                               │
                            observation pair ──┴──▶ [ inverse model ] ──▶ latent action
                                                                              │
            observation + latent action ──▶ [ forward model  ] ──▶ predicted next observation
            observation + latent action ──▶ [ command decoder] ──▶ joint command
```

| module | the question it answers |
|---|---|
| inverse model | given a transition, what action produced it? |
| forward model | does the latent action let us predict what happens next? |
| command decoder | can the latent action be turned back into an executable joint command? |

- **Encoder:** a one-billion-parameter video model, **frozen throughout**, never trained on robots.
  The three modules on top are about five million parameters each.
- **Training signal:** predict the next observation, and recover the real joint command. Equal
  nominal weight on the two.

**Two structural facts that shape every result below.** The command decoder never sees the second
frame — whatever the transition contributes has to pass through a 64-number latent, and that
bottleneck is the whole design. And the two training terms differ in scale by two orders of
magnitude, so **next-observation prediction takes roughly 99% of the gradient in practice**: the
term meant to ground the latent in real commands runs on the remainder.

> **บทพูด (TH).** encoder เป็นโมเดลวิดีโอขนาดพันล้านพารามิเตอร์ที่ **แช่แข็งไว้ ไม่เคยเทรนกับหุ่นยนต์เลย**
> เราเทรนแค่สามโมดูลเล็ก ๆ ข้างบนมัน: ตัวแรกถามว่า "การเปลี่ยนจากภาพนี้ไปภาพนั้น เกิดจากคำสั่งอะไร"
> ตัวที่สองถามว่า "คำสั่งที่ถอดได้ ทำนายอนาคตได้ไหม" ตัวที่สามถามว่า "แปลงกลับเป็นคำสั่งข้อต่อที่สั่งได้จริงไหม"
> **สองข้อที่ต้องจำไว้**: ตัวถอดคำสั่งไม่เคยเห็นเฟรมที่สอง ข้อมูลต้องลอดผ่าน latent 64 ตัวเท่านั้น
> และ loss สองก้อนต่างสเกลกันร้อยเท่า ทำให้ **การทำนายภาพกินเกรเดียนต์ไปราว 99%**

---

## Slide 2 — Setup, hypothesis, and how every number is read

**The bodies.** Six-legged walkers differing only in segment lengths — hip, thigh, shin. Held-out
bodies are never trained on. Only clips where the body genuinely walked are used (a minimum forward
travel, a maximum sideways drift); bodies that collapse or veer are excluded by name, not by hope.

**Commands are retargeted, not copied.** One foot trajectory in Cartesian space is solved separately
for each body by inverse kinematics: **same intended behaviour, genuinely different joint numbers.**
Without this the transfer question would be empty — every body would receive the same command.

**Behaviour:** forward walking, one speed. This turns out to matter, and Slide 7 is where it returns.

**The hypothesis.** If the latent truly separates movement from body, then a body never seen in
training should receive commands appropriate to *its own* geometry — not the commands of whichever
training body it most resembles.

| how each claim is tested | what it isolates |
|---|---|
| linear probe on the frozen encoder | is the information present and readable at all, before any training |
| swap test — one body's frame, another body's latent | does the decoder take the body from the image or from the latent |
| input ablation — delete one input, re-measure | which input the decoder actually depends on |
| mixture fit | is the model interpolating between training bodies, or copying the nearest one |
| physical replay | do the predicted commands actually walk, not merely score well per joint |

**Reading the numbers.** Command error is RMSE in degrees, pooled over all eighteen joints, on a
body never trained on. **The commands' own spread is 11.7°** — that is the scale every error below
is read against. R² is measured against the held-out body's own mean posture, so a negative value
means *worse than memorising one fixed pose*. A "control" is the identical run with one setting
changed.

**Ratios in this deck point in two different directions, so each one says which.** An *error* ratio
(`error / baseline error` — the held-out body-head fit, for instance) is **better below 1.0**: 1.0
means the model only learned the dataset mean. A *beats-the-baseline* ratio (the forward model's
rollout score, `baseline error / model error`) is **better above 1.0**: 1.0 means the model ties with
predicting that nothing moves, and 1.5 means its error is 1.5x smaller. An earlier version of this
line declared "above 1.0 means worse" without qualification, which is backwards for every rollout
number in Part 3 — the same sign error the project already made once inside a finding and had to
correct.

> **บทพูด (TH).** หุ่นทุกตัวมีหกขาและข้อต่อ 18 ข้อเหมือนกัน **ต่างกันแค่ความยาวขา** คำสั่งของแต่ละตัว
> ได้มาจากการแก้ IK จากรอยเท้าเดียวกัน — **เจตนาเดียวกัน แต่ตัวเลขคำสั่งต่างกันจริง** ถ้าไม่ทำแบบนี้
> คำถามเรื่องการถ่ายโอนจะไม่มีความหมาย เพราะทุกตัวจะได้คำสั่งชุดเดียวกัน
> **สมมติฐาน**: ถ้า latent แยก "การเคลื่อนไหว" ออกจาก "ร่างกาย" ได้จริง หุ่นที่ไม่เคยเห็นควรได้คำสั่งที่
> เหมาะกับขาของตัวเอง ไม่ใช่คำสั่งของหุ่นที่หน้าตาใกล้ที่สุดในชุดเทรน
> **เลขที่ต้องเทียบตลอด**: ค่าความกว้างของคำสั่งเองคือ 11.7 องศา — error ทุกตัวอ่านเทียบกับเลขนี้
> และ R² ติดลบหมายถึง **แย่กว่าการจำท่านิ่งท่าเดียว**

---

## Slide 3 — The geometry is visible in the image; the trained model does not use it

**A four-thousand-parameter probe on the frozen encoder recovers a held-out body's segment lengths
to within 0.04**, fitted on four training bodies and applied to one never seen, with nothing
supervising it. The information is in the image. **This is the premise the whole project rests on.**

**The trained decoder, a thousand times larger, reads the same body wrong:**

| held-out body, segment scales | hip | thigh | shin |
|---|---|---|---|
| the truth | **0.80** | 0.90 | 0.90 |
| probe on the frozen encoder, 4k parameters | **0.84** | 0.91 | 0.91 |
| the trained command decoder, 5M parameters | **0.62** | 0.96 | 0.96 |

The probe lands within 0.04 everywhere; the decoder implies a hip segment **22% shorter than the
body has**. The larger model is the one that misreads it.

**A swap test says why.** Given one body's image together with a *different* body's latent — two
bodies whose commands differ by 21° — the decoder answers with the **latent's** body to within 6°.
It followed the latent and ignored the image.

**Diagnosis: the decoder learned to recognise which training body it is looking at, and recall that
body's commands.** There is no entry in that lookup for a body it has never seen — which is why more
trained capacity buys lower error on bodies it saw and *higher* error on the one it did not.

![the encoder places the unseen body correctly; the decoder does not](../results/wm/stage1/figures/encoder_vs_decoder.png)

> **บทพูด (TH).** สองบรรทัดนี้คือหัวใจของสไลด์: **probe เล็กจิ๋วอ่านความยาวขาของหุ่นที่ไม่เคยเห็นได้แม่น**
> (ข้อมูลอยู่ในภาพจริง ไม่มีใครไปบอกมันเลย) แต่ **decoder ที่ใหญ่กว่าพันเท่าอ่านผิด** — มันเดาขาหน้าสั้นกว่า
> ความจริง 22% และ swap test บอกสาเหตุ: เอาภาพของตัวหนึ่งคู่กับ latent ของอีกตัว **มันตอบตาม latent
> ไม่สนใจภาพเลย** แปลว่ามันไม่ได้อ่านรูปร่างจากภาพ มันแค่ **จำได้ว่านี่คือหุ่นตัวไหนในชุดเทรน แล้วเรียกคำสั่ง
> ของตัวนั้นออกมา** — ซึ่งหุ่นที่ไม่เคยเห็นไม่มีอยู่ในตารางนั้น

---

## Slide 4 — The fix has to be in the objective, and that is the lever

**Capacity, access, and the contents of the latent were each ruled out first.** Rescaling the
target, shrinking the decoder, stripping body identity adversarially, and handing the decoder a
global view of the frame all failed or made transfer worse. **What remained was the objective:
nothing in the loss ever *required* reading geometry from pixels.** Recognising the body was cheaper
and scored just as well.

**So change what the loss asks for.** Every body walks the same episodes, so at a given timestep two
bodies share the intent and differ only in geometry. Add one term: *take body A's latent, show the
decoder body B's frame, and require body B's command.*

```
  BEFORE   the latent can carry the body  ──▶  decoder recalls a training body's commands
                                               (cheap, and the loss never objects)

  AFTER    reading the body out of the latent is WRONG BY CONSTRUCTION
                                          ──▶  the only way to be right is to read
                                               geometry from the image
```

**One term, one extra decoder pass per batch, nothing else changed:**

| matched pair, same held-out body | without the term | with it |
|---|---|---|
| command error | 3.67° | **3.44°** |
| **how much the image is worth to the decoder** | 0.4× | **9.6×** |
| movement's share of the latent | 82% | **93%** |
| body identity's share of the latent | 12% | **3%** |

**The second row is the result: a 22-fold change in what the image is worth, from one term.** The
last two rows are the mechanism, and the check that this is purification rather than destruction —
behaviour decodes out of the latent slightly *better* than before, so nothing was lost. **The latent
stopped carrying a job that was never its own**, and the two inputs ended up with separate jobs:
the image carries *which body*, the latent carries *what movement*.

**After the change, swapping the latent between two bodies moves the decoder's answer by 0.04°.**
It is reading geometry from pixels.

![what the cross-body term does](../results/wm/stage1_correct/figures/cross_loss_effect.png)

> **บทพูด (TH).** ลองแก้ที่ตัวโมเดลมาสี่ทาง (ลดขนาด, เพิ่มการเข้าถึงภาพ, ไล่ body identity ออกจาก latent)
> **ไม่ได้ผลหรือแย่ลง** — เพราะปัญหาไม่ได้อยู่ที่ความสามารถ แต่อยู่ที่ **loss ไม่เคยบังคับให้มันต้องอ่านรูปร่าง
> จากภาพเลย** วิธีจำตัวหุ่นมันถูกกว่าและได้คะแนนเท่ากัน
> **เราจึงเปลี่ยนสิ่งที่ loss ขอ**: เอา latent ของหุ่น A ไปคู่กับภาพของหุ่น B แล้วบังคับให้ตอบคำสั่งของ B
> → การอ่าน "ตัวไหน" ออกจาก latent กลายเป็น **คำตอบที่ผิดโดยโครงสร้าง** ทางเดียวที่จะถูกคือต้องอ่านจากภาพ
> **ผลคือภาพมีค่าต่อ decoder เพิ่มขึ้น 22 เท่า จาก term เดียว** และ latent สะอาดขึ้น (การเคลื่อนไหว 82→93%,
> ตัวตนของร่าง 12→3%) โดยที่ข้อมูลพฤติกรรมไม่ได้หายไปเลย

---

## Slide 5 — Where it works, and where it stops: the held-out body has to sit inside the geometry the data spans

**Inside that span, the predicted commands actually walk.** Driven open-loop through the same physics
used to collect the data, on a body never trained on:

| | inverse kinematics | control | with the cross-body term |
|---|---|---|---|
| forward distance | 100% | 85% | **90%** |
| heading deviation | 0° | 11.8° | **5.5°** |
| worst joint-limit excursion | 0° | 8.2° | **3.9°** |

Both walk. The cross-body run is **steadier** rather than dramatically better: it stays in a narrow
band everywhere and is closer to the reference heading on every clip, where the control matches it
twice and then fails badly on the third. Gait structure matches the reference — tripod alternation
and stance durations line up, with 3.02 feet on the ground on average against the reference's 3.08.
Stated plainly: on one clip the reference walks almost straight and both models veer. **Neither
reproduces a straight walk on demand.**

**Outside that span, the same weights fail — and fail at one specific joint.**

| the same model, asked about | error | R² |
|---|---|---|
| a body inside the geometry spanned by training | **3.4°** | **+0.81** |
| a body whose shin is short while its thigh is not | 13.4° | **−0.34** |

Every training body that walked happened to have thigh and shin the same length — **not by design:
the bodies where they differ are the ones that fall over.** So those two segments never moved apart
in anything the model saw.

| asked what that body's thigh and shin are | thigh | shin |
|---|---|---|
| the truth | **1.00** | **0.80** |
| the trained decoder | 0.68 | 0.68 |
| the probe on the frozen encoder | 0.84 | 0.84 |
| the best any mixture of training bodies could say | 0.60 | 0.60 |

**All three give the two segments the same number** — a large trained decoder, a tiny readout of the
raw encoder, and a calculation with no learning in it at all, making the identical mistake. The last
row is why: **no combination of bodies in which two segments always move together can pull them
apart.** And the joint that fails worst is the knee, **the joint between exactly those two
segments**, while the joint that does not involve the shin still works. That localisation is the
fingerprint of a data gap, not of a model that merely got worse — and it repeats on every unseen
thigh/shin ratio available, all negative.

![gait structure matches the reference](../results/wm/stage1_correct/gait/gait_stage1_m3d_cross_clip0.png)

![per-joint traces, inside the span](../results/wm/stage1_correct/figures/action_trace_m3d_cross_c08f09t09.png)

![per-joint traces, outside it — the damage is at the knee](../results/wm/stage1_correct/figures/action_trace_m3d_cross_c10f10t08.png)

> **บทพูด (TH).** สไลด์นี้เทียบ **"ได้" กับ "ไม่ได้" ในแง่ของร่างกายที่ป้อนเข้าไป ไม่ใช่ในแง่ loss**
> **ได้**: ถ้าหุ่นที่ไม่เคยเห็นอยู่ในช่วงรูปร่างที่ข้อมูลครอบคลุม คำสั่งที่ทำนายออกมา **เดินได้จริง** ในฟิสิกส์
> (ระยะ 90% ของ IK, เลี้ยวเพี้ยนน้อยกว่าครึ่ง, จังหวะขาตรงกับอ้างอิง)
> **ไม่ได้**: ถ้าอยู่นอกช่วงนั้น พังทันที (13.4 องศา, R² ติดลบ = แย่กว่าจำท่านิ่ง)
> **สาเหตุไม่ใช่โมเดล**: หุ่นทุกตัวที่เดินได้ในชุดเทรนมีท่อนขาสองท่อนยาวเท่ากันหมด (ตัวที่ไม่เท่ากันมันล้ม)
> ดังนั้น **ทั้ง decoder ใหญ่, probe จิ๋ว, และการคำนวณที่ไม่มีการเรียนรู้เลย ตอบผิดเหมือนกันเป๊ะ**
> และข้อต่อที่พังที่สุดคือ **หัวเข่า ซึ่งอยู่ระหว่างสองท่อนนั้นพอดี** — นี่คือลายนิ้วมือของช่องว่างในข้อมูล

---

## Slide 6 — Testing that explanation instead of asserting it

**An explanation makes a prediction.** If the two segments are tied because every training body ties
them, then adding bodies where they differ should untie them. Two such bodies were generated and
checked to walk.

**A matched pair, same clip budget, same held-out body, same clips.** The only thing that differs is
whether the training set contains bodies whose segments move apart:

| | four bodies, segments tied | six bodies, segments decoupled |
|---|---|---|
| command error | 12.7° | **3.3°** |
| R² against the body's own mean posture | **−0.78** | **+0.89** |

**A 3.9× improvement, and it crosses zero:** from worse than memorising one fixed pose, to
explaining 89% of the held-out body's variance. **Filling the gap the diagnosis named does not
merely improve extrapolation — it removes the failure.**

**The same thing shows up before any decoder is trained, at no GPU cost.** Refit only the probe on
the enlarged set, and the two segments come apart: the gap between them opens to 0.18 against a true
0.20, from 0.00 before. **The tying broke in the frozen encoder's readout first.**

**Two qualifications, because they bound the number.** Adding coverage converts an extrapolation
problem into an interpolation one — that is exactly what coverage is *for*, but it means the two runs
do not face equally hard tasks, so the 3.9× measures the conversion rather than the same task done
better. And both runs were still improving when the budget ended, so both figures are lower bounds.

![the coverage experiment](../results/wm/stage1_correct/figures/coverage_experiment.png)

> **บทพูด (TH).** สไลด์ก่อนจบด้วย *คำอธิบาย* — สไลด์นี้คือ **การทดสอบคำอธิบายนั้น ไม่ใช่แค่เชื่อมัน**
> ถ้าสาเหตุคือ "ข้อมูลไม่เคยมีหุ่นที่สองท่อนนี้ยาวไม่เท่ากัน" การเพิ่มหุ่นแบบนั้นเข้าไปต้องแก้ได้ — **และแก้ได้จริง**
> **คุมจำนวนคลิปให้เท่ากัน** เพื่อไม่ให้เถียงได้ว่าเพราะข้อมูลเยอะขึ้น: 12.7 องศา → 3.3 องศา, R² −0.78 → +0.89
> และที่สำคัญ **เห็นได้ก่อนเทรนด้วยซ้ำ** แค่ refit probe ตัวเล็กบนชุดใหม่ สองท่อนก็แยกออกจากกันทันที
> **ข้อจำกัดที่ต้องพูดเอง**: การเพิ่ม coverage เปลี่ยนโจทย์จาก "ทำนายนอกช่วง" เป็น "ทำนายในช่วง" ด้วย
> เลข 3.9 เท่าจึงวัดการเปลี่ยนโจทย์ด้วย ไม่ใช่วัดโจทย์เดิมที่ทำได้ดีขึ้นล้วน ๆ

---

## Slide 7 — Why this task hides what the model is not learning: the gait is a cycle

**Scope first, because it decides how far the claim reaches.** Everything here is forward walking at
one speed.

```
   a gait is a limit cycle
          │
          ▼   one frame shows the pose   ──▶   the pose fixes the phase
          │
          ▼   the phase fixes what comes next
   the command is largely readable from a SINGLE frame — and so is the future
```

**Three measurements, all pointing the same way:**

| | |
|---|---|
| predicting the command **32 frames ahead** | as accurate as predicting the present (2.9° vs 3.0°) |
| removing the transition entirely — show the same frame twice | costs only about **30%** |
| which feet are swinging, from one frame alone | **82%** correct, against 50% by chance |

**And the consequence, measured directly:** replace the real action with a null one and re-predict.
**The prediction changes by under 3%.** A model with nothing to gain from the action channel will
not use it, and no reweighting can create a signal the task does not contain.

> **This was first noticed here as an oddity at one speed.** It is not confined to one speed — the
> same measurement, repeated later across twelve behaviour conditions including four speeds, gives
> the same answer. **Parts 2 and 3 are what happened when that oddity was taken seriously.**

> **บทพูด (TH).** อันนี้สำคัญมากและเป็นต้นทางของทุกอย่างหลังจากนี้ **การเดินเป็นวงรอบ (limit cycle)**
> เฟรมเดียวบอกท่าทาง ท่าทางบอกเฟสของการเดิน และเฟสบอกว่าอะไรจะเกิดต่อไป
> เลขสามตัวที่พูดพอ: (1) **ทำนายคำสั่งล่วงหน้า 32 เฟรม แม่นเท่าทำนายปัจจุบัน** (2) ตัดข้อมูลการเปลี่ยน
> ระหว่างเฟรมออกทั้งหมด เสียความแม่นแค่ราว 30% (3) ดูเฟรมเดียวเดาได้ว่าขาไหนกำลังยก ถูก 82% เทียบเหรียญ 50%
> **ผลที่ตามมาคือ**: สลับคำสั่งจริงเป็นคำสั่งว่าง ๆ แล้วทำนายใหม่ — **ผลเปลี่ยนไม่ถึง 3%**
> ถ้าโมเดลไม่ได้อะไรจากช่องคำสั่งเลย มันก็ไม่ใช้ และการไปปรับน้ำหนัก loss ก็สร้างสัญญาณที่โจทย์ไม่มีขึ้นมาไม่ได้

---

## Slide 8 — The forward model is not broken; it was being judged on the wrong task

Every measurement so far asks whether forward prediction helps **reconstruct the command**. It does
not — but that is not what a forward model is for.

**Judged on its own job — rolling the world forward, with true latents supplied so the module is
isolated:**

| steps ahead | how much it beats holding the frame still |
|---|---|
| 1 | 1.5× |
| 5 | 1.7× |
| 10 | 1.5× |

It beats a frozen world at **every horizon out to ten steps**, and beats a constant-velocity
baseline by two orders of magnitude. **And the latent action does real work:** delete it and the
command decoder loses roughly a factor of three.

**So both parts function, and the ceiling is the task rather than the parts.** The margin over a
frozen world is real but modest and decays with horizon; the latent matters for decoding commands;
and yet **the action contributes under 3% of what next-observation prediction needs**, because the
pose already says what comes next. That gap — a module that works, a latent that carries real
information, and an action channel worth almost nothing to prediction — **is the quantity the rest
of this deck chases.**

> Part of the mistake was ours, and reading the source method confirmed it: there, the command
> decoder is an **auxiliary regulariser**, and the deployed system plans by comparing predicted
> futures against a goal image. **We had made the auxiliary term the whole evaluation.**

> **บทพูด (TH).** ทุกการวัดก่อนหน้านี้ถาม forward model ว่า "ช่วยถอดคำสั่งได้ไหม" — **ซึ่งไม่ใช่หน้าที่มัน**
> พอวัดด้วยงานของมันเองคือ **ม้วนโลกไปข้างหน้า มันทำได้จริง** ชนะการ "หยุดภาพไว้เฉย ๆ" ทุกช่วง 1-10 สเต็ป
> และ **latent action ก็มีผลจริง** ลบออกแล้ว decoder แย่ลงราวสามเท่า
> **แปลว่าชิ้นส่วนไม่ได้พัง แต่เพดานอยู่ที่ตัวโจทย์** — margin มีจริงแต่ไม่มาก และลดลงตามระยะ
> ขณะที่ **คำสั่งมีค่าต่อการทำนายไม่ถึง 3%** เพราะท่าทางบอกอนาคตไปแล้ว ช่องว่างนี้คือสิ่งที่เดคที่เหลือไล่ตาม
> และต้องยอมรับว่า **ส่วนหนึ่งเราวัดผิดเอง**: ในงานต้นฉบับ ตัวถอดคำสั่งเป็นแค่ตัวช่วย regularise
> ระบบจริงเขาวางแผนด้วยการเทียบภาพอนาคตกับภาพเป้าหมาย เราเอาตัวช่วยมาเป็นการประเมินทั้งหมด

---

# Part 2 — Crossing embodiments: the gap, and the attempt

## Slide 9 — What crossing embodiments requires, and what the field already has

**The target.** Plan toward a goal defined in a coordinate **shared across bodies whose command
spaces have nothing in common** — an eighteen-joint six-legged insect and a twelve-joint quadruped —
with no kinematic model, no retargeting, and no controller already running on the target body.
**The only thing the two robots share is what a camera sees.**

```
                          needs a kinematic model
                                    ▲
          joint-tokenised universal  │   retargeting-based cross-morphology
          controllers (graph /       │   controllers (body description + IK)
          transformer over the tree) │
      ────────────────────────────────┼────────────────────────────────▶  crosses leg count
                                    │
          morphology-parameter       │   ███ THIS WORK ███
          quadruped world models     │   video only, eighteen joints ↔ twelve
                                    │
                          needs no kinematics
```

**Everything that crosses leg count is handed a body model; everything that needs no body model
stays inside one leg count.** The lower-right quadrant is empty. Manipulation escapes the problem by
sharing end-effector pose; locomotion has no such fallback, because eighteen and twelve joint
targets share no dimension at all.

**Vision-based world models for locomotion, and what each leaves open:**

| | what it establishes | what it does not do |
|---|---|---|
| egocentric visual self-model, one legged robot | morphology and kinematics are **not needed** — babble plus a single camera suffice to plan, and even to detect and recover from damage | fits **one robot at a time**; no action representation is shared, nothing is transferred between bodies |
| latent-action world models (manipulation) | an action inferred from video alone can drive a policy | end-effector pose already means the same thing on every body |
| cross-embodiment latent-goal planning | a shared latent goal space across embodiments is achievable | built from **retargeted, temporally aligned paired demonstrations** |
| action-conditioned video world models | name the failure when prediction ignores the action | diagnosed on manipulation or single-body locomotion, never across embodiments |

**The gap this thesis occupies:** no body description, no paired data, no retargeting — in
locomotion, where the action is precisely what has to cross.

**And the three requirements Stage 1 hands this part.** A shared representation has to be forced by
the objective; coverage has to be checked per independent axis; and a saturating metric will hide
what the model is not doing. All three return in what follows.

> **บทพูด (TH).** เป้าคือ **วางแผนผ่านพิกัดที่ใช้ร่วมกันได้ ระหว่างหุ่นที่ space ของคำสั่งไม่มีอะไรตรงกันเลย**
> (แมลงหกขา 18 ข้อต่อ กับสี่ขา 12 ข้อต่อ) โดยไม่ใช้ kinematic model ไม่ retarget และไม่ต้องมี controller
> ที่ใช้ได้อยู่แล้วบนหุ่นเป้าหมาย — **สิ่งเดียวที่สองตัวนี้แชร์กันคือสิ่งที่กล้องเห็น**
> ดูแผนภาพ: **งานที่ข้ามจำนวนขาได้ ทุกงานถูกป้อนข้อมูลร่างกายให้ / งานที่ไม่ต้องใช้ข้อมูลร่างกาย ก็อยู่ในจำนวนขาเดียว**
> ช่องขวาล่างว่างอยู่ งาน manipulation หนีปัญหานี้ได้เพราะใช้ตำแหน่งปลายมือร่วมกัน **แต่การเดินไม่มีอะไรให้หนีไป**
> งานที่ใกล้เราที่สุดคือ self-model จากกล้องบนหัว — **พิสูจน์ว่าไม่ต้องรู้ kinematics จริง แต่ทำทีละตัว ไม่เคยข้ามร่าง**

---

## Slide 10 — We rebuilt the field's own fix for this, faithfully

**The failure already has names in the literature.** A world model conditioned on an action should
predict a different future for a different action; when it does not, one line of work calls it
*context collapse* — the predictor extrapolates from the observation and becomes insensitive to the
action channel. A second names the same thing from the training side: a teacher-forced target
already contains the action's effect, so an **action-invariant solution fits the loss perfectly**. A
third takes adjacent-frame redundancy as a design premise and splits its prediction horizon to avoid
it.

**So neither the diagnosis nor the proposed fix is ours.** The published remedy is a term that
forces the rollout under the real action apart from the rollout under a null action, plus a frozen
readout that must recover the action from the prediction. We rebuilt our pretraining to match it,
using their settings where we had evidence for them and our own where we did not:

| setting | theirs | ours | why the difference |
|---|---|---|---|
| separation margin | 0.3 | **0.1** | at 0.3 the term overshoots, switches itself off, and its gradient dies; at 0.1 it rises and holds on both bodies |
| rollout length the term acts over | 12 steps | **3** | our rolled prediction becomes worse than a frozen frame by five steps — acting past that trains on noise |
| context frames | 32 | **1** | our forward model conditions on one frame; 32 frames is a different architecture, not a hyperparameter |
| an extra regulariser from their backbone | used | **not used** | specific to their encoder; not guessed in here |

**Every run that follows was pre-registered:** what would count as success was written down before
the run started. **Six came out negative, and none of the criteria moved afterwards.**

> **บทพูด (TH).** ขอย้ำว่า **ทั้งการวินิจฉัยและวิธีแก้ ไม่ใช่ของเราเอง** — วงการตั้งชื่อปัญหานี้ไว้แล้ว
> (โมเดลไม่สนใจช่องคำสั่ง เพราะเดาจากภาพได้อยู่แล้ว) และเสนอวิธีแก้ไว้แล้วด้วย
> เราจึง **สร้างวิธีของเขาขึ้นมาใหม่ให้ตรงที่สุด** ที่ต่างออกไปมีสี่จุด และทุกจุดมีเหตุผลจากการวัดของเราเอง
> ไม่ใช่การเดา (เช่น margin 0.3 ของเขาทำให้ term ดับไปเลยในระบบเรา เราจึงใช้ 0.1)
> และ **ทุกการทดลองเขียนเกณฑ์ตัดสินไว้ก่อนรัน** หกครั้งออกมาเป็นลบ และไม่มีการขยับเกณฑ์ย้อนหลังเลย

---

## Slide 11 — Six independent measurements, one answer

**Read this as a chain of eliminations, not a list of failures.** Each row closes one hypothesis
about *where* the missing action-sensitivity lives, and each returns a number.

| the hypothesis | the measurement | the result |
|---|---|---|
| the objective is wrong; the published fix will repair it | full rebuild, both bodies | one-step prediction fine; **3-4× worse than a frozen frame at two steps**, no sensitivity gained |
| the weighting was off | swap the real action for a null one, re-predict | **under 3%** change — the true action is worth almost nothing |
| one frame is too short a step | retrain at three-frame spacing | no change |
| the action lives in what an action-blind model *misses* | probe that residual for the command | adds about **1%** |
| a motion-organised representation will expose it | difference consecutive observations, same probes | the redundancy survives, and **cross-body transfer is destroyed** |
| our own shared-coordinate term caused it | remove it entirely, controlled | **it got worse** — the term was a small *positive* contributor |

**The last row is the control a committee asks for, and it is the one that makes the rest mean
something.** "Did your own objective cause this?" is answerable: no, and removing it hurts.

**The sharpest of the six is worth watching rather than reading.** From a bit-identical reset, branch
into two genuinely different behaviours — walk on, or turn away. They finish **42° apart in the
world.** In the encoder's representation, that separation sits at **1.1× the noise floor between two
runs of the same command.**

![two futures a human separates instantly; the encoder does not](../results/cf_confirm/insect_forward-vs-turn.mp4)

**The gap between what you see in that clip and what the encoder encodes is the finding.**

> **บทพูด (TH).** ให้อ่านตารางนี้เป็น **การตัดสาเหตุออกทีละข้อ ไม่ใช่รายการความล้มเหลว** แต่ละแถวปิดสมมติฐาน
> หนึ่งข้อว่า "สัญญาณที่หายไปอยู่ที่ไหน" และคืนค่าเป็นตัวเลขทุกแถว
> **แถวสุดท้ายคือ control ที่กรรมการต้องถาม**: "แล้ว objective ของคุณเองทำให้มันพังหรือเปล่า" — ตอบได้ว่าไม่
> เพราะเอาออกแล้วแย่ลง
> **อันที่ชัดที่สุดคือคลิปนี้**: รีเซ็ตให้เหมือนกันเป๊ะ แล้วแยกเป็นเดินตรงกับเลี้ยว ปลายทางห่างกัน 42 องศา
> ซึ่งคนดูแยกออกทันที **แต่ในพื้นที่ของ encoder ระยะห่างนั้นเท่ากับ 1.1 เท่าของ noise ระหว่างการรันคำสั่งเดียวกันสองครั้ง**

---

## Slide 12 — The number underneath all six

**One quantity explains every row above.** In third-person locomotion video, the joint command is
readable from a **single frame**:

| command recoverable from | one frame | a frame pair |
|---|---|---|
| six-legged insect | **0.78** | 0.89 |
| — turning only | **0.93** | 0.96 |
| quadruped | 0.16 | 0.34 |

**Prior work found the ingredient; we measured how far it goes.** One group showed a frozen video
encoder carries action structure recoverable by an inverse model, and noted in passing that a static
tabletop scene lets per-frame appearance stand in for temporal context. **In periodic locomotion
that substitution is nearly total:** one frame recovers 88% of what a pair recovers on the insect,
and 97% on turning.

**And this is the distinction that makes it a result rather than a restatement:**

```
   RECOVERABLE FROM the observation     ≠     NECESSARY FOR predicting the next observation
        the command reads out at 0.89              the same command is worth under 3%
                              │
                              ▼
        in locomotion the ACTION-INVARIANT SOLUTION IS NEAR-OPTIMAL,
        so a fix applied to the training target has nothing better to converge to
```

Honest scope: measured on two robots, in simulation, across twelve behaviour conditions. Not
measured on manipulation.

> **บทพูด (TH).** ทั้งหกแถวในสไลด์ก่อนอธิบายได้ด้วยเลขตัวเดียว: **ในวิดีโอมุมที่สามของการเดิน
> คำสั่งข้อต่ออ่านออกได้จากเฟรมเดียว** (0.78 จากเฟรมเดียว เทียบ 0.89 จากสองเฟรม — คือ 88% ของกันและกัน
> และถ้าเป็นการเลี้ยวคือ 97%)
> **งานก่อนหน้าเจอวัตถุดิบนี้แล้ว** (ว่า encoder มีข้อมูล action อยู่ และฉากนิ่ง ๆ ทำให้เฟรมเดียวแทนบริบทเวลาได้)
> **แต่ไม่มีใครวัดว่าในการเดินที่เป็นวงรอบ มันแทนได้เกือบทั้งหมด**
> และประโยคที่ทำให้มันเป็นผลงานไม่ใช่การพูดซ้ำคือ: **"ถอดออกมาได้" ไม่เท่ากับ "จำเป็นต่อการทำนาย"**
> คำสั่งถอดออกมาได้ 0.89 แต่มีค่าต่อการทำนายไม่ถึง 3% → **คำตอบที่ไม่สนใจ action จึงเกือบดีที่สุดอยู่แล้ว**
> การไปแก้ที่เป้าของการเทรนจึงไม่มีอะไรดีกว่านั้นให้ลู่เข้าหา

---

# Part 3 — The principle

## Slide 13 — Pose determines the future, so the action is redundant

> **When the agent's own configuration is visible and determines what happens next, the action
> carries no information the observation lacks — and a model trained to predict the next observation
> will ignore it.**

**It is not simply "the agent is in frame."** Manipulation puts the arm in frame and its world models
work. The condition is stronger: the visible configuration must determine the **future**, not merely
reveal the current command.

```
  THIRD-PERSON LOCOMOTION           MANIPULATION                  EGOCENTRIC LOCOMOTION
  ┌─────────────────────┐          ┌─────────────────────┐       ┌─────────────────────┐
  │  whole body visible │          │  arm ──▶ object     │       │      the world      │
  │                     │          │       (independent) │       │    (body unseen)    │
  └─────────────────────┘          └─────────────────────┘       └─────────────────────┘
   pose ⇒ the command               pose ≠ the outcome             pose is invisible
   pose ⇒ the next pose (cycle)     object state is free           future depends on action
   ──────────────────────           ──────────────────────         ──────────────────────
   ACTION REDUNDANT                 action needed                  action needed
   the world model collapses        world models work              ◀── the prediction
```

**Periodicity is what makes locomotion the severe case.** A gait is a limit cycle: the pose fixes the
phase and the phase fixes the next pose, so the pose determines not just what the robot is doing but
what it is *about to* do — which is exactly the quantity a forward model is trained on.

**Two interventions separate rhythm from redundancy:**

| break the gait with | does the action gain value? |
|---|---|
| random command noise | **yes** — action-sensitivity more than doubles |
| real stops, speed changes and turn onsets, verified to reach the robot | **no** |

**So the cause is not rhythm as such.** Any command a controller would actually issue is still
visible in the pose.

![the same behaviour under both viewpoints](../results/deck/principle_allo_vs_ego.mp4)

> **บทพูด (TH).** ประโยคในกรอบคือหลักการของงานนี้: **ถ้าท่าทางของตัวเองมองเห็นได้ และท่าทางนั้นกำหนดอนาคต
> คำสั่งก็ไม่ได้เพิ่มข้อมูลอะไรจากที่ภาพบอกอยู่แล้ว** โมเดลที่ถูกเทรนให้ทำนายภาพถัดไปจึงเมินมันทิ้ง
> **ไม่ใช่แค่ "เห็นตัวเองในภาพ"** — งาน manipulation ก็เห็นแขนตัวเองและยังทำงานได้ เพราะท่าแขนไม่ได้บอกว่า
> **ของ** จะไปอยู่ไหน เงื่อนไขจริงแรงกว่านั้น: ท่าทางต้องกำหนด **อนาคต** ไม่ใช่แค่บอกคำสั่งปัจจุบัน
> **การเดินเป็นกรณีที่หนักที่สุดเพราะมันเป็นวงรอบ** ท่าบอกเฟส เฟสบอกท่าถัดไป
> และ **สาเหตุไม่ใช่ "จังหวะ"**: ใส่ noise มั่ว ๆ ให้จังหวะเสีย คำสั่งมีค่าขึ้นจริง แต่ใส่การหยุด/เปลี่ยนความเร็ว/
> เริ่มเลี้ยว แบบที่ controller จริงจะสั่ง — **ไม่ช่วยเลย** เพราะคำสั่งแบบนั้นยังมองเห็นได้จากท่าทางอยู่

---

## Slide 14 — The principle explains results that are already published

**The pieces are not ours; the connection is.** Sort existing systems by the single question the
principle says matters — *does the visible configuration determine the future?* — and their
published outcomes line up without exception:

| system | agent visible? | pose determines the future? | does it work? | what the principle adds |
|---|---|---|---|---|
| cross-embodiment latent-goal planning (manipulation) | yes, and the object | **no** — object state is independent of arm pose | **yes** | why it *can* work: the arm's pose says nothing about where the object ends up, so the action stays informative |
| egocentric locomotion self-model | **no** | **no** — the body is unseen | **yes** | **why egocentric is necessary** — which that paper does not claim |
| static tabletop manipulation | yes | partly | mixed | names the exception its own authors noted only in passing |
| **ours — third-person locomotion** | yes | **yes** — the gait is a limit cycle | **no** | this is the collapse, measured end to end |

```
  FOUR separate observations in the literature, none connected to the others
    context collapse named · the action-invariant solution named · adjacent-frame
    redundancy assumed as a design premise · static scenes noted as an exception
                                │
                                ▼   WE MEASURED WHERE THEY DID NOT
                         how far the substitution goes in PERIODIC locomotion
                                │
                                ▼   recoverable  ≠  necessary
        ⇒ ONE mechanism joins all four — and it is a VIEWPOINT property,
          not an objective and not a target, that decides whether an
          action-conditioned world model can exist for a given task at all
```

**Two limits, stated because a reader will look for them.** We closed the residual-target version of
one proposed route; **we did not train the counterfactual-target version, and do not claim it
fails.** And the rows above read published results *through* the principle — they are not
re-measurements of those systems.

> **บทพูด (TH).** ชิ้นส่วนทั้งหมดในตารางนี้ไม่ใช่ของเรา **สิ่งที่เป็นของเราคือเส้นที่ลากเชื่อมมัน**
> เรียงงานที่มีอยู่ด้วยคำถามเดียวคือ "ท่าทางที่มองเห็นได้ กำหนดอนาคตไหม" — **ผลที่เขาตีพิมพ์เรียงตามนั้นพอดีทุกงาน**
> งานที่สำเร็จคืองานที่ท่าทางไม่กำหนดอนาคต (ของวางอยู่บนโต๊ะ / มองไม่เห็นตัวเอง)
> งานที่ล้มคือของเรา ซึ่งท่าทางกำหนดอนาคตเต็มที่เพราะการเดินเป็นวงรอบ
> **ข้อสรุปคือมันเป็นเรื่องของ "มุมกล้อง" ไม่ใช่เรื่อง objective หรือเป้าของการเทรน**
> และขอบเขตที่ต้องพูดเอง: เราปิดเส้นทางแก้แบบหนึ่งไปแล้ว **แต่ยังไม่ได้ลองอีกแบบ จึงไม่เคลมว่ามันแก้ไม่ได้**

---

## Slide 15 — Three attempts built on this model, and how each was tested

Three different things were built on top of this world model. All three failed — and the tests show
that **two of them failed for the same reason, and the third for a different one.**

| what was built | what it does | how it was tested | what came back |
|---|---|---|---|
| **behaviour selection** | score a library of recorded clips against a goal, replay the best one | swap the goal for a different behaviour and see whether the choice changes; delete the world-model rollout and see whether the score changes | the goal swap costs only **3-7 points**; deleting the rollout costs **nothing at all** (41% vs 42%) |
| **teacher-graded policy** | the same scoring, applied instead to **small variations of one behaviour**, to grade a policy being trained | check the grades against what physics actually produced for those same variations | the variations it was asked to rank were **physically indistinguishable** (0.1304 against 0.1299) |
| **action-conditioned prediction** | make next-observation prediction depend on the action at all | the six measurements of Part 2 | the action is worth **under 3%** of prediction |

**The first two differ only in granularity, and that is exactly what separates their failures.**
Choosing between *different behaviours* is a coarse judgement, and it works. Choosing between *small
variations of one behaviour* requires resolving outcomes that the physics itself barely separates —
so no representation, however good, can order them.

```
  behaviour selection   ─┐
                         ├─▶ both need the action to be VISIBLE IN PREDICTION
  action-conditioning   ─┘      ──▶  POSE DETERMINES THE FUTURE
                                     ──▶ removed by the egocentric view (Slide 16):
                                         single-frame readability 0.78 → 0.29

  teacher-graded policy ───────▶ a SECOND obstacle the viewpoint does not touch:
                                 the outcomes being ranked are effectively the same outcome
```

![the teacher subtracts](../results/wm/closed_loop/f142_video/f144_labelled.mp4)

**The clip is the teacher-graded row, and it is blunt:** the recorded walk travels 100%, imitation
alone gets 36%, and imitation *plus* the world-model teacher gets **31%**. The teacher subtracts.

**What this licenses:** two of the three share one mechanism, and that mechanism is removed by the
view change. **What it does not:** that any of the three automatically revive — they were tested
again afterwards, and Part 5 is that record.

> **บทพูด (TH).** เราสร้างของสามอย่างบนโมเดลนี้ และพังทั้งสามอย่าง **แต่การทดสอบบอกว่าสองอันพังเพราะเหตุเดียวกัน
> และอีกอันพังเพราะเหตุอื่น**
> **สองอันแรกคือการให้คะแนนเหมือนกัน ต่างกันแค่ความละเอียด**: เลือกระหว่าง "ท่าที่ต่างกันคนละท่า" = หยาบ → **ทำได้**
> เลือกระหว่าง "ท่าเดียวกันที่เปลี่ยนไปเล็กน้อย" = ละเอียด → **ทำไม่ได้ และไม่ใช่ความผิดของโมเดล**
> เพราะ **ฟิสิกส์จริงแยกสองตัวเลือกนั้นออกจากกันแค่ 0.1304 กับ 0.1299** ไม่มี representation ไหนเรียงอันดับ
> สิ่งที่เหมือนกันได้
> **อันที่สามคือ action-conditioning** ซึ่งกลับไปที่เหตุเดียวกับอันแรก: ท่าทางบอกอนาคตไปแล้ว
> **คลิปนี้คือแถวที่สอง**: เดินจริง 100% / โคลนนิ่งเฉย ๆ 36% / โคลนนิ่ง + ครูที่เป็น world model **31%** — **ครูหักคะแนน**

---

# Part 4 — The prediction, tested

The principle makes a falsifiable prediction: if *pose determines the future* is what kills
action-conditioning, then removing the agent's own pose from view should restore it. This part tests
that directly, and reports which half held.

## Slide 16 — Moving the camera onto the body removes the redundancy — and the shared coordinate survives it

**The cheapest test that could answer it, built to be discarded:** four textured walls and a ceiling
around the spawn point, camera moved onto the robot's head. **Not an environment.**

**A leak guard ran first, and the result was not read until it passed:** room appearance predicts
heading **below chance** on held-out clips, on both bodies. Nothing is being read off the wallpaper.

| command recoverable from, six-legged insect | third-person | egocentric |
|---|---|---|
| one frame | **0.78** | **0.29** |
| a frame pair | 0.89 | 0.58 |
| **what the transition adds** | +0.11 | **+0.29** |

**Single-frame readability falls by two thirds, and the transition's value nearly triples.** Sideways
motion reads **nothing at all** from one egocentric frame, against 0.61 third-person. **This is the
first intervention in the entire chain to move the quantity all six measurements were trying to
move.**

**And the risk it had to clear.** A head camera could break the redundancy and simultaneously destroy
the one cross-body result the project has. Fitted on the insect's egocentric observations and applied
to the quadruped **with no refitting at all**:

| the shared coordinate, quadruped unrefitted | forward | lateral | turn |
|---|---|---|---|
| third-person | 0.63 | 0.43 | **0.07** |
| **egocentric** | 0.50 | 0.39 | **0.64** |

**Turning goes from dead to the strongest channel** — the channel this project has fought longest,
and the one the view change helps most, which the physics predicts: a head camera sees rotation as
global image flow whatever body is underneath. **Forward and lateral fall.** The coordinate is harder
to read from a head view and it still crosses; that trade is the honest summary.

![the world turns the same way under either robot](../results/deck/q1_turn_both_bodies.mp4)

**What this does not say.** It shows the action is no longer redundant with the pose. It does **not**
show that a trained world model then uses the transition — this project's own record is of signals
that existed and were ignored, so that measurement needs the trained model.

> **บทพูด (TH).** การทดสอบที่ถูกที่สุดที่ตอบคำถามนี้ได้: **ย้ายกล้องจากข้างสนามไปไว้บนหัวหุ่น** กับห้องสี่ผนัง
> ที่สร้างมาเพื่อทิ้ง ไม่ใช่ environment จริงจัง
> **เช็คการรั่วก่อนอ่านผล**: สีผนังทำนายทิศทางได้ **แย่กว่าการเดาสุ่ม** ทั้งสองตัว → ไม่ได้แอบอ่านจากวอลเปเปอร์
> **ผลแรก**: อ่านคำสั่งจากเฟรมเดียวได้ 0.78 → **0.29** และค่าของ "การเปลี่ยนระหว่างเฟรม" เพิ่มเกือบสามเท่า
> นี่คือ **ครั้งแรกในทั้งเคสที่ตัวเลขที่เราพยายามขยับมาหกครั้ง ขยับจริง**
> **ผลที่สองคือความเสี่ยงที่ต้องผ่าน**: กล้องบนหัวอาจทำลายผลข้ามร่างที่เรามีอยู่อันเดียว — **ไม่ทำลาย**
> fit บนแมลงแล้วเอาไปใช้กับสี่ขา **โดยไม่ fit ใหม่เลย**: ช่องเลี้ยวจาก 0.07 (ตาย) → **0.64 (แข็งแรงที่สุด)**
> ส่วนเดินหน้า/ไถลข้างลดลง — **อ่านยากขึ้นจากมุมนี้ แต่ยังข้ามร่างได้ นี่คือสรุปที่ซื่อสัตย์**

---

## Slide 17 — What we claim as contributions

### Contribution 1 — a cross-embodiment coordinate that needs no correspondence

| | prior cross-embodiment latent-goal work | ours |
|---|---|---|
| paired data across bodies | **retargeting** | **none** |
| temporal alignment | **required** | **none** |
| hand labels | — | **none** |
| transfer | — | fitted on the insect, applied to the quadruped **with no refit** |

**Why it is possible: the target is a quantity both bodies already have, not a correspondence that
has to be built.** Dividing speed by body size makes the two dynamically comparable — the insect and
the quadruped average the **same dimensionless walking speed across a fourfold size difference.**

**Scope limits, stated plainly.** The coordinate is regressed onto a *measured* physical quantity —
body motion differenced from recorded pose, which on hardware means odometry or motion capture — so
it is **not learned from pixels alone**, and anyone claiming vision-only here would be overclaiming.
And it buys exactly three channels: forward, lateral, turn. Anything the two bodies do not share —
joint spaces, gaits, contact patterns — is not carried by it.

### Contribution 2 — separating where the body is going from how the body shakes

An egocentric camera carries both at once: **the trajectory, which both robots share**, and **gait
oscillation, which is a six-legged tripod on one and a trot on the other** (15.6° of turn sway
against 6.8°). They are separable — the gait sits at a fixed number of cycles per clip on both
bodies while the net turn does not — and removing the gait component helps **across bodies
specifically**:

| quadruped, unrefitted | forward | lateral | turn |
|---|---|---|---|
| egocentric | 0.45 | 0.38 | 0.57 |
| **gait oscillation removed** | **0.47** | **0.46** | **0.61** |

**The within-body fit does not improve and the cross-body fit does** — that asymmetry is the
hypothesis's own signature, since generic denoising would move both. Lateral comes back **past** its
third-person value.

**Status, precisely: proven feasible, not done.** The removal here is three harmonics of one
frequency, estimated per clip and subtracted linearly. **That a projection this blunt already works
is the argument for a learned version; it is not evidence that a learned version will be better.**
Forward does not recover, so part of that drop is something other than gait shake, and remains
unexplained.

**No existing work has this combination:** egocentric locomotion work does not separate gait from
trajectory, and cross-embodiment alignment work builds correspondence by retargeting instead of
removing body-specific motion.

> **บทพูด (TH).** สไลด์นี้คือ **สิ่งที่เราเคลมว่าเป็นผลงานของเรา มีสองข้อ**
> **ข้อแรก — พิกัดที่ข้ามร่างได้โดยไม่ต้องจับคู่อะไรเลย**: ไม่ต้อง retarget ไม่ต้องจัดเวลาให้ตรงกัน ไม่ต้องมี
> label มือ และ **fit บนแมลงแล้วใช้กับสี่ขาได้เลยโดยไม่ fit ใหม่** ที่ทำได้เพราะ **เป้าหมายเป็นปริมาณที่หุ่นทั้งสอง
> มีอยู่แล้ว ไม่ใช่ความสัมพันธ์ที่เราต้องไปสร้าง** — หารความเร็วด้วยขนาดตัว แล้วหุ่นสองตัวที่ขนาดต่างกันสี่เท่า
> ได้ความเร็วไร้หน่วยเท่ากัน
> **ข้อจำกัดที่ต้องพูดเอง**: พิกัดนี้ถอยกลับไปอ้างการวัดทางกายภาพ (บนฮาร์ดแวร์คือ odometry หรือ motion capture)
> **ไม่ใช่เรียนจากพิกเซลล้วน ๆ** ถ้าเคลมว่า vision-only คือเคลมเกิน และมันให้แค่สามช่อง
> **ข้อสอง — แยก "ไปทางไหน" ออกจาก "ตัวสั่นยังไง"**: กล้องบนหัวมีทั้งสองอย่างปนกัน อันแรกแชร์กันได้
> อันหลังเป็นของเฉพาะร่าง (หกขาเดินสามขาสลับ vs สี่ขาวิ่งทรอต) **พอเอาส่วนสั่นออก ค่าข้ามร่างดีขึ้น แต่ค่าในร่างเดิมไม่ดีขึ้น**
> — ความไม่สมมาตรนี้คือลายเซ็นของสมมติฐานเราเอง **สถานะ: พิสูจน์แล้วว่าเป็นไปได้ ยังไม่ได้ทำให้สมบูรณ์**

---
# Part 5 — Where this stands

> **How to read Part 5.** Everything here is about **motor babble** — the standard way a robot that
> nobody has a controller for gets bootstrapped. B1 results and gecko results are kept apart: B1 was
> in pretrain, gecko never was. **All gecko work is on Slide 26 alone** and appears in no flow
> diagram or result table above it.
>
> **บทพูด (TH).** ตั้งแต่ part นี้ไปคือเรื่อง **motor babble** — วิธีมาตรฐานที่ใช้ตั้งต้นหุ่นที่ยังไม่มี controller
> ขอแยก B1 กับ gecko ให้ชัด: **B1 เคยอยู่ใน pretrain แต่ gecko ไม่เคย**
> **เรื่อง gecko อยู่ที่สไลด์ 32 หน้าเดียว** ไม่ปนกับแผนภาพหรือตารางไหนข้างบนเลย

---

---

## Slide 18 — Imagination-RL: the wall is the rollout


```
  looked like: the actor exploits the frozen FTM's blind spots
        │
        ▼  isolation test: 5,000 iterations, frozen FTM, no re-grounding at all
  actually was: the critic never converges, even against a STATIC target
        │
        ▼  four real stabilisation fixes, in order (table below)
  best Dreamer variant still fails the pre-registered bar
        │
        ▼  switch algorithm entirely (PPO -- no bootstrapped imagined-rollout value)
  lands on the SAME wall as the best Dreamer variant
        │
        ▼  shorten the horizon instead (GAMMA 0.99 → 0.95)
  gap does not shrink -- policy freezes into a static stance instead
        │
        ▼  every one of these iterates the SAME single-step predictor to build a return
  failure localises to the FTM's rollout, not the algorithm
```

| fix | MC-check relative difference (bar: < 0.25) |
|---|---|
| EMA target critic alone | fails outright (unbounded value drift) |
| + symlog critic, + return normalisation | 0.464 |
| hard/periodic target updates | 0.624 — worse |
| + two-hot distributional critic (DreamerV3's own fix) | **0.272** — best Dreamer variant, still fails |
| **PPO** (different algorithm, no imagined-rollout bootstrap) | **0.281** |
| PPO, GAMMA 0.99 → 0.95 (horizon ~100 → ~20 steps) | 0.306 — worse, policy went static (action variation −92%) |

**Two unrelated algorithms, same ~0.27-0.28 wall; shortening the horizon doesn't move it.** Not an
algorithm problem — every one of these returns iterates the same single-step predictor. The exact
failure mode Koopman Dreamer (2607.19719) names.

---

> **บทพูด (TH).** สไลด์นี้คือความพยายามใช้ RL บนโลกจำลอง (imagination) แล้วมันไม่ผ่าน
> ตอนแรกคิดว่า actor ไปหาช่องโหว่ของ forward model แต่จริง ๆ คือ **critic ไม่ลู่เข้าเลย** แม้เป้านิ่ง ๆ
> แก้ 4 อย่างตามตำรา Dreamer แล้วก็ยังไม่ผ่านเกณฑ์ เปลี่ยนไปใช้ PPO ซึ่งคนละอัลกอริทึมกันเลย ก็ไปชนกำแพงเดียวกัน
> แปลว่าปัญหาไม่ใช่อัลกอริทึม แต่เป็นการเอา forward model แบบ 1 สเต็ปมาม้วนต่อกันยาว ๆ

---

## Slide 19 — Sequence context does not fix it


```
  FTM(frame_t, action_t) → frame_t+1        one observation, one action
        │
        ▼  same action means different things at different points in a gait cycle
  one-action-many-outcomes, no phase context to disambiguate them
        │
        ▼  Yu / WMP: locomotion world models are recurrent-over-HISTORY
              frame-SEQUENCE + command-SEQUENCE → next state
        │
        ▼  premise checked on GROUND TRUTH first, no model in the loop
  signal exists in the data (table below) → not a task property, a model gap
        │
        ▼  every combination of {pooled, spatial-preserved} x {non-recurrent, recurrent}
             that is cheaply testable — command-sequence + delta-target held fixed throughout
  ALL FOUR fail the bar (table below), including the one that is genuinely
  spatial + recurrent at once — the widest miss of the four
```

Slides 18 and 25 both improved something real and both hit a wall, tracing to one place: the FTM is
**stateless, single-step** (confirmed — no hidden state between calls, not an RSSM). The standard
locomotion fix (Yu, WMP) conditions on frame-*history* and command-*sequence* instead.

| ground-truth check (Test 1, no model) | value |
|---|---|
| corr(\|Δaction\|, \|ΔFroude\|), within one fixed behaviour | +0.432 |
| permutation p-value (n = 20,000) | 0.002 |
| ridge, leave-one-out R² | 0.204 |

| kill-gate (command-sequence + delta-target throughout, bar: gap > 0.110) | spatial preserved? | recurrent? | gap |
|---|---|---|---|
| pooled mean-vector per frame → GRU | no | yes | +0.036 |
| learned attention-pool per frame → GRU | no — still pools every frame | yes | +0.058 |
| full token grid, self-attention across frames | yes | no | +0.048 |
| **ConvGRU: spatial hidden state, pooled only at final read-out** | **yes** | **yes** | **+0.069** |
| stateless single-step FTM (reference) | — | — | +0.042 |

**The genuine test — spatial detail flowing through a real recurrent state — fails by the widest
margin, not the narrowest.** So the sequence-context hypothesis is **ruled out, not unconfirmed**.
Test 1's signal still stands (the flatness is real, not a task property); no context-based fix
recovers it. Open next: capacity/optimisation of an end-to-end-trained model, not input
representation.

---

> **บทพูด (TH).** สไลด์นี้รวบว่าทำไมทั้งสองทางข้างบนถึงชนกำแพงที่เดียวกัน
> คำตอบคือ forward model ของเราเป็นแบบ **สเต็ปเดียว ไม่มีความจำ** ซึ่งงาน locomotion เขาไม่ทำกัน เขาใช้ประวัติ
> เราเลยเช็คก่อนว่าสัญญาณมีจริงไหมใน ground truth — **มีจริง** แล้วค่อยลองใส่ context 4 แบบ — **ตกหมดทั้ง 4 แบบ**
> อันที่ควรดีที่สุด (spatial + recurrent พร้อมกัน) กลับพลาดมากที่สุด สมมติฐานนี้เลยถูก **ตัดทิ้ง ไม่ใช่แค่ยังไม่ยืนยัน**

---

## Slide 20 — Stop-gradient clears the lever ~9×


```
  eliminated, each ruling out one cause (Slides 18-20 + offline probes):
    loss target · gradient share · sequence-context architecture (4 variants) · the encoder itself
        │
        ▼  what's left: the trained z-only head STILL fails the lever (+0.045 / +0.099)
             even though z demonstrably HAS the signal (offline rho 0.535, survives to proj(a))
        │
        ▼  isolate: freeze z, train a clean head on Froude ALONE (no competing L_recon/L_motion)
  PASSES -- matches/exceeds offline rho          → cause localised: joint-training competition for z
        │
        ▼  fix: z.detach() before L_body (L_state retired, redundant+worse)
             real 50-epoch retrain, BIAS-2, per-channel reading pre-registered before running
  PASS -- gap +1.008, ~9x the 0.110 bar
```

| action-lever, real retrain (`beh12_body_stopgrad`) | value |
|---|---|
| real z, median cos | 0.693 |
| mean z, median cos | -0.315 (anti-correlated, not just flat) |
| **gap (bar: 0.110)** | **+1.008** |

| channel | sign-agreement gap (real − mean) |
|---|---|
| forward | +0.238 |
| lateral | +0.031 (weakest, still positive) |
| yaw | +0.243 |

**Eight nulls were the diagnosis, not the failure** — each eliminated a cause, which is what
localised the real one to joint-training gradient competition rather than guessing it.

**Mechanism, precisely:** `L_body`'s gradient was not *pure* competition — it does real positive
work too (recon+motion alone develops only 32-76% of a jointly-shaped `z`'s signal). The fix trades
that contribution away to remove the competition, and still wins by more than the proxy predicted.

**Not yet shown: that it improves control.** The lever measures directional sensitivity, not
ranking or closed-loop behaviour — a gap this project has been burned by before (F116-F127).

---

> **บทพูด (TH).** สไลด์นี้คือผลที่คลี่คลายเรื่องทั้งหมด
> null ทั้ง 8 อันก่อนหน้าไม่ใช่ความล้มเหลว แต่คือการ **ตัดสาเหตุออกทีละอัน** จนเหลือสาเหตุจริง
> สาเหตุคือตอนเทรนรวมกัน loss หลายตัวแย่ง gradient ใน z กัน พอใส่ stop-gradient ให้ L_body ผลดีขึ้น ~9 เท่าของเกณฑ์
> **ข้อควรระวัง**: อันนี้วัดแค่ "ทิศทางไว" ยังไม่ได้แปลว่าคุมหุ่นได้จริง ซึ่งโปรเจกต์นี้เคยพลาดตรงนี้มาแล้ว

---

## Slide 21 — What the coordinate fixed


**Scoring in the right space is what turned selection on.** Frame distance reads the current frame,
not the goal. Rescoring by body-motion (Froude) distance fixes that immediately.

| selection rule | same-robot | cross-embod., 1ch | cross-embod., 3ch |
|---|---|---|---|
| frame/embedding distance | 18-23% (28% chance) | — | — |
| **Froude distance, no rollout** | **76-86%** | 35-38% | **68-70%** |
| + FTM rollout added back | — | — | 33-44%, turning destroyed |

| strafing | 1-channel | 3-channel |
|---|---|---|
| | 13-25% (17% chance) | **86-100%** |

### Limitation: coarse yes, fine no

Picking the right *kind* of motion works. Picking the right *amount* does not — and it did not
improve when the representation changed. Images could not separate fine magnitudes; Froude could
not either.

| question | result |
|---|---|
| which family? (walk / turn / strafe) | **works, crosses embodiments** |
| how much? rank 12 conditions | 28% exact, mean rank 2.33/12 |
| how much? 0.5-sd perturbations of one behaviour | **47% vs a 50% coin** |
| direction of correction | 0.867 — right way |
| extent of correction | 0.71 sd — wrong amount |

**The information is not missing.** Probed straight off the frozen embedding delta — no ITM, no FTM,
no trained head — the fine speed signal reads out clearly. The pipeline cannot use it; the encoder
did not discard it.

Six independent readout fixes, all null. **Not focused — the claim is the shared coordinate, which
is coarse and works.**

> **บทพูด (TH).** สไลด์นี้มีสองส่วน: **ส่วนที่แก้ได้** กับ **ข้อจำกัด**
> แก้ได้: เดิมให้คะแนนด้วยระยะห่างของภาพ ซึ่งไปอ่านเฟรมปัจจุบัน ไม่ได้อ่านเป้า พอเปลี่ยนเป็น Froude การทำตามเป้าโผล่มาทันที
> ขยายจาก 1 เป็น 3 ช่อง การเลือกข้ามหุ่นดีขึ้นเกือบเท่าตัว การไถลข้างจากมองไม่เห็นเลยเป็นเกือบสมบูรณ์
>
> ข้อจำกัด: เลือก**ชนิดท่า**ได้ เลือก**ความแรง**ไม่ได้ และ**ไม่ดีขึ้นเลยตอนเปลี่ยนวิธี** — ใช้ภาพก็ไม่ได้ ใช้ Froude ก็ไม่ได้ (47% เทียบเหรียญ 50%)
> **แต่ข้อมูลไม่ได้หาย** — probe ตรง ๆ บน embedding ดิบยังอ่านออกชัด แปลว่า pipeline ใช้มันไม่เป็น ไม่ใช่ encoder ทิ้ง
> ลองแก้มา 6 วิธี null หมด **เราไม่โฟกัสตรงนี้** เพราะ claim คือพิกัดกลางที่ข้ามหุ่นได้ ซึ่งเป็นงานหยาบ ๆ และอันนั้นทำได้แล้ว

---

## Slide 22 — Controller vs. what we test, and what Froude is

```
  A CONTROLLER / POLICY                 WHAT WE TEST (a "closed loop")
  ────────────────────                  ──────────────────────────────
  state ──▶ network ──▶ torques         goal ──▶ score 12 RECORDED clips ──▶ replay winner
  invents the motion                    picks from motions that already exist
  can fall over                         cannot fall — the body is posed directly
```

| | controller/policy | our closed loop |
|---|---|---|
| motion comes from | invented by the network | **a library of 12 recorded clips** |
| what "survival" proves | the policy is stable | **nothing — it cannot fall by construction** |
| what it DOES prove | — | **was the goal read, and the right motion chosen** |

**Everything on this slide and through Slide 26 is about selection, not control, and was true when
measured.** A real controller exists now — Slide 27, dated, separate, and bounded — but it does not
retroactively apply to any result below: those numbers are about picking the right recorded clip
from a library, and remain exactly what they say they are.

### Froude, and how a goal is made

`Fr = v / sqrt(g · L)` — speed made dimensionless by body size (`L` ≈ hip height). Two robots of
different size walking "the same way" get the **same Froude**. That is the whole reason a hexapod's
goal can mean anything to a B1. Three channels: **forward**, **lateral**, **yaw**.

| | hexapod | B1 | gecko |
|---|---|---|---|
| hip height `L` | ~0.09 m | ~0.56 m | ~0.06 m |
| a normal walk, m/s | 0.13 | 0.29 | 0.013 |
| **same walk, in Froude** | **~0.13** | **~0.13** | **~0.04** |

```
  GOAL — two sources, kept separate
    physics :  read the recorded NUMBER off the source clip   (privileged ceiling)
    vision  :  source clip's video → ITM → body_head          (what deployment does)

  SCORING — two mechanisms, kept separate
    DIRECT   a ──▶ projector ──▶ z ──▶ body_head ──▶ Froude          (no world model)
    ROLLOUT  e_t + z ──▶ FTM ──▶ imagined ──▶ ITM ──▶ body_head      (world model in)

  score = | candidate Froude − goal Froude |   → pick the smallest
```

Crossing the two gives the 2×2 (A/B/C/D) on Slide 24 — separated **on purpose**, since bundling
them confounded an earlier version of this test.

> **บทพูด (TH).** ผมเองก็สับสนบ่อย ขอแยกให้ชัด
> **Controller จริง ๆ** = เน็ตเวิร์ก**คิดท่าเองจาก state** ส่งแรงให้ข้อต่อ และ **ล้มได้**
> **สิ่งที่เราทดสอบ** = มีคลิปอัดไว้ 12 คลิป ระบบแค่**เลือกว่าจะเล่นอันไหน** แล้ว replay — **ล้มไม่ได้เลยโดยโครงสร้าง**
> ตัวเลข "รอด 3/3" จึงไม่ได้แปลว่าเก่ง มันแปลว่าไม่มีอะไรให้ล้ม สิ่งที่พิสูจน์จริงคือ**อ่านเป้าถูกไหม เลือกท่าถูกไหม**
>
> **Froude** คือความเร็วหารด้วยขนาดตัว `v/sqrt(g·L)` — หุ่นคนละขนาดที่เดินเหมือนกันจะได้ค่าเท่ากัน
> **นี่คือเหตุผลเดียวที่เป้าจากแมลงมีความหมายกับ B1** ใช้ 3 ช่อง: เดินหน้า/ไถลข้าง/เลี้ยว
> **เป้าผลิตได้ 2 แบบ** (อ่านตัวเลขที่อัดไว้ = มีข้อมูลพิเศษ / ถอดจากวิดีโอ = แบบที่ใช้จริง)
> **ให้คะแนนได้ 2 แบบ** (Direct ไม่ใช้ world model / Rollout ใช้) — ที่ต้องแยกสองแกนนี้เพราะเวอร์ชันก่อนเรามัดรวมกัน เลยสรุปไม่ได้ว่าตัวไหนพัง

## Slide 23 — The correct adaptation pipeline


```
  wrong mechanism, tried first: wm.train --init_ckpt
    jointly retrains ITM+FTM+decoder+body_head+probe under the FULL pretrain loss
    → B1 got WORSE than zero-shot (0.264 → 0.231), forward went negative
        │
        ▼  the actual question: "are we even running LAC-WM's adaptation?" — NO
  the real staged procedure, never assembled into one pipeline before:
    stage 1  wm.adapt          — fine-tune ONLY ITM+FTM on the new body's own clips
    stage 2  fit_projector     — refit a→z against the ADAPTED itm
    stage 3  wm.adapt3         — optional joint fine-tune (skipped)
    stage 4  fit_body_head     — refit the shared Froude head against the projector's own z
```

| B1, z=proj | forward ρ | lateral ρ | yaw ρ | median ρ |
|---|---|---|---|---|
| zero-shot (no adaptation) | 0.057 | 0.264 | 0.526 | 0.264 |
| `wm.train` joint retrain (wrong mechanism) | −0.068 | 0.231 | 0.326 | 0.231 |
| **correct staged adaptation** | **0.572** | **0.449** | **0.670** | **0.572** |

**Every channel more than doubles, forward included — never once positive under the wrong
mechanism.** The whole "B1 got worse" episode was a tooling error, not a result about the claim.

> **บทพูด (TH).** สไลด์นี้คือการแก้ที่ต้นเหตุ
> ตอนแรกเรา "fine-tune" ด้วยวิธีที่ผิดสนิท — มันไป retrain ทุกอย่างพร้อมกันด้วย loss ของการ pretrain
> ผลคือ B1 **แย่ลงกว่าไม่ทำอะไรเลย** (0.264 → 0.231) แล้วเราก็เสียเวลาไล่หาว่าทำไมมันพัง ทั้งที่คำถามที่ถูกคือ "เราใช้วิธี adaptation จริง ๆ หรือยัง" — **คำตอบคือยัง**
> พอใช้ขั้นตอน 4 stage ที่ถูกต้อง **ทุกช่องดีขึ้นเกินเท่าตัว** โดยเฉพาะช่อง forward ที่ไม่เคยเป็นบวกเลยในวิธีเดิม
> บทเรียน: **มันเป็นบั๊กเครื่องมือ ไม่ใช่ข้อสรุปว่าวิธีเราไม่เวิร์ก**

---

## Slide 24 — The 2×2: which half is broken


```
  goal source (physics / vision)  ×  candidate scoring (direct / rollout)
```

Same goal clip, same 12 candidates, scored per step (not dominant-pick):

| mode | candidates | goal | goal read err | top pick | % steps right family | dist. to true goal |
|---|---|---|---|---|---|---|
| **A** | direct | physics (privileged) | 0.0000 | turn_w0.008 | **78%** | **0.038** |
| **D** | direct | **vision only** | **0.0293** | turn_w0.008 | **80%** | **0.038** |
| B | rollout | physics (privileged) | 0.0000 | side_R_lvl1 | 47% | 0.290 |
| C | rollout | **vision only** | 0.0293 | side_R_lvl1 | 40% | 0.290 |

**Reading the goal from video cost nothing here.** D is handed no recorded number and misreads the
goal by 0.029 — yet matches A exactly: same top pick, same distance, 80% vs 78%.

> **The explanation this slide used to give was wrong, and is replaced by a stronger measurement
> (2026-09-14).** It read: *"because that misreading is smaller than the gap between the two closest
> candidates (0.033 vs 0.038)."* Those are the two candidates' distances **to the goal**, not the gap
> **between** them — the gap is **0.0079**, so a 0.029 error is nearly 4x too large. Perturbing the
> goal by that magnitude in random directions changes the top pick **65% of the time**; this clip was
> simply lucky in the direction its error pointed. **No threshold is claimed any more.** What carries
> the claim instead: across **12 goal conditions and 96 planning decisions**, the vision goal selects
> the right behaviour family in **92%** of steps at a median distance of **0.043**, against **90–92%**
> and **0.062** for the privileged recorded goal (random pick: 0.143). The vision goal matches or
> beats the privileged one over the whole goal set, not one clip.

> **Both halves of this table were measured through a flaw of our own, and correcting it moves one
> of them.** Every stage that fits the video-to-Froude reading path — the inverse model, the stage-1
> adaptation, the projector, the shared head — trains on **adjacent** frame pairs. The goal was read
> at a spacing of **five**, because one flag set both the planner's rollout depth (where five is
> deliberate) and the goal-read spacing (which rolls nothing). A second defect compounded it: the
> shared head's fit validated only the adapted body, never the body whose goal is read, so the
> reading quality on that side had never been measured at all.
>
> **Corrected — read at the trained spacing, with the head fitted and validated on both bodies:**
>
> | | goal read error | top pick | % steps right family | distance |
> |---|---|---|---|---|
> | measured goal (privileged) | 0.0000 | `turn_w0.008` | **100%** | **0.029** |
> | **vision goal only** | **0.0164** | `turn_w0.008` | **100%** | **0.029** |
>
> **A goal read from the other body's video is now indistinguishable from the recorded number** —
> same pick, same distance, both perfect, and 0.029 is the best distance any of the twelve candidates
> achieves. Both arms beat the 78%/80% above. The figures in the table are superseded: the 0.0293 was
> this bug plus a favourable clip, and across all 48 source clips the corrected read has a median
> error of 0.017. (The "77% of clips under the candidate-spacing threshold" figure that stood here
> is withdrawn — see the box above: that threshold was never a real quantity and no code computed
> it.)
>
> **The rollout half does not move.** Handed a goal read at 0.0097 error it still selects at 40% and
> still prefers a candidate three times farther from the goal than the achievable optimum. That
> verdict has now survived a checkpoint change, a goal-quality change, and a spacing correction.

**Rollout does not merely fail — it prefers the worst candidate.** `side_R_lvl1` is the *farthest*
of all 12 from the goal (0.290 against the best 0.033). Handing it a perfect goal (mode B, error
0.0000) does not help. The goal is not the problem; the world model in the scoring loop is.

**Reminder (Slide 22): selections from a library, not a controller.** Nothing here can fall.

**📹 VIDEO — C vs D.** Three panels, aligned by elapsed time (the two bodies record at 20 Hz and
50 Hz, so matching by frame index puts the goal 2.5× ahead). Goal panel shows both what the system
read and the true value; footer shows both errors.

| panel | shows |
|---|---|
| left | source body ego — the goal |
| middle | new body ego — what the loop sees |
| right | new body allocentric — **replayed ground truth, not control** |

> **บทพูด (TH).** สไลด์นี้แยกว่า**ครึ่งไหนของลูปพัง** โดยไขว้สองแกน: เป้ามาจากไหน × ให้คะแนนยังไง
> **A กับ D เท่ากัน** — D อ่านเป้าจากวิดีโอผิดไป 0.029 แต่**เลือกคลิปเดียวกัน ระยะห่างเท่ากัน** (80% เทียบ 78%)
> **คำอธิบายเดิมที่ว่า "ผิดน้อยกว่าช่องว่างระหว่างผู้สมัคร (0.033 กับ 0.038)" นั้นผิดและถอนแล้ว** — สองค่านั้นคือ
> *ระยะห่างจากเป้า* ของผู้สมัครสองตัว ไม่ใช่ช่องว่างระหว่างกัน ช่องว่างจริงคือ 0.0079 และวัดแล้วว่าความผิดพลาด 0.029
> ทำให้เลือกผิดตัวถึง 65% ของทิศทางที่สุ่ม คลิปนี้แค่โชคดี
> **สิ่งที่ค้ำข้อเคลมแทนคือการวัดทั้งชุด: 12 เงื่อนไขเป้า 96 การตัดสินใจ — D ได้ 92% ระยะ 0.043 ส่วน A ได้ 90–92% ระยะ 0.062**
> **นี่คือข้อเคลม: เป้าจากภาพทำงานได้เท่ากับการวัดด้วย proprioception ที่แอบดูตัวเลขจริง**
> **ส่วน rollout ไม่ใช่แค่พลาด แต่เลือกตัวที่แย่ที่สุด** — `side_R_lvl1` ห่างจากเป้าที่สุดใน 12 ตัว (0.290)
> และ**ต่อให้แจกเป้าที่ถูกต้อง 100% ให้ (mode B) ก็ยังพัง** แปลว่าปัญหาไม่ใช่เป้าหมาย แต่คือ world model ตอนให้คะแนน
> **ย้ำ:** นี่คือการเลือกคลิปจากคลัง ไม่ใช่ controller — มันล้มไม่ได้อยู่แล้ว

---

## Slide 25 — Motor babble, and what we added


```
  a robot nobody has a controller for
        │  flail semi-randomly, record (action, video, body motion)
        ▼
  clips that are NOT demonstrations — nobody is doing the task well
        │  fit: action ──▶ latent ──▶ physical body motion
        ▼
  the robot's own commands become readable in a coordinate shared with other robots
```

**What we contributed is calibration and convention, not a controller.** Babble gives a *map* from a
new body's commands to a shared coordinate. Turning that map into behaviour still needs our pipeline
to construct the motion — **that part is not working yet.** The method is widely used and has real
potential; this deck reports how far the calibration gets and where it stops.

### Zero-shot vs fine-tune (median ρ, `z = proj`, held-out)

| | clips needed | hexapod (in pretrain) | B1 (in pretrain) | gecko (never) |
|---|---|---|---|---|
| zero-shot, frozen head | **0** | 0.454 | 0.427 | 0.113 ⚠ |
| wrong "fine-tune" (`wm.train`) | 48 | — | 0.231 | not run |
| **correct staged adaptation** | **9** | — | **0.572** | 0.249 ⚠ |

**Nine clips.** Stage 1 adapts ITM+FTM on **9 babble clips, 1000 steps** — that is the whole cost of
a new body. Stages 2 and 4 refit two small heads (projector, and an 8.8k-parameter Froude head) on
the same babble set; no new recording. Zero-shot needs none of it and gets 0.427 on B1.

⚠ **The gecko column is not comparable and must not be read as one.** Its 0.113 was measured through
a body-frame bug found later and is withdrawn; its 0.249 comes from a later mechanism. Gecko is a
separate experiment — below.

**B1 is the defensible claim:** zero-shot to a body the world model saw in pretrain already
half-works (0.427); adapting on that body's own babble takes it to 0.572.

> **บทพูด (TH).** สไลด์นี้แนะนำวิธีที่งานเราอยู่ในนั้น: **motor babble**
> หุ่นที่ยังไม่มีใครทำ controller ให้ ก็ปล่อยให้มัน**ขยับมั่ว ๆ** แล้วอัดว่า สั่งอะไร → ภาพเป็นยังไง → ตัวเคลื่อนที่ยังไง
> คลิปพวกนี้**ไม่ใช่การสาธิตท่าที่ดี** มันแค่ทำให้เราแปลคำสั่งของหุ่นตัวใหม่เป็นพิกัดกลางได้
> **สิ่งที่เราทำคือ calibrate ไม่ใช่สร้าง controller** — สุดท้ายยังต้องให้ pipeline ไปประกอบท่าให้ **และส่วนนั้นยังไม่สำเร็จ**
> ตาราง: **B1 ดีขึ้นจริง 0.427 → 0.572** โดยใช้แค่ **9 คลิป / 1000 steps** — นี่คือต้นทุนทั้งหมดของหุ่นตัวใหม่
> ไม่ต้องอัดข้อมูลเพิ่ม stage อื่นแค่ fit หัวเล็ก ๆ สองอันบน babble ชุดเดิม ส่วนช่อง gecko **ห้ามอ่านรวมกัน** เดี๋ยวอธิบายแยก

---

## Slide 26 — Gecko: the actual unseen body


**Separate from everything above.** B1 was in pretrain. Gecko was not — no URDF, no kinematics, no
expert clips, only babble. This is early work, kept apart on purpose.

### Two measurement bugs, found and fixed

| | |
|---|---|
| the stage-1 "gate" | sign inverted — it is anti-correlated with success, not a gate |
| gecko's body frame | built on an axis pointing straight up; all three channels scrambled |
| after both fixes | yaw went from **dead (−0.04) to strongest (+0.50)** |

### Where it stands

| stage-4 held-out (below 1.0 = the head learned something) | |
|---|---|
| broken frame, projector sees 1 action frame | 1.010 — dataset mean only |
| corrected frame, 1 action frame | 1.004 — still the mean |
| **corrected frame + 20 action frames** | **0.970** — first time below 1.0 |
| B1, same metric | **0.751** |

**Why 20 frames:** one action frame predicts gecko's motion at ρ 0.215; twenty predict it at
**0.736** ≈ B1's 0.770. Gecko's speed is set by gait *frequency*, invisible in one snapshot.

### The remaining gap is the camera

| what carries the signal | ρ | |
|---|---|---|
| actions (20-frame window) | **0.736** | ≈ B1's 0.770 — data is fine |
| **egocentric video** | **0.374** | B1's 0.747 — **the ceiling** |
| pipeline delivers | 0.249 | 67% of what the video allows |

Gecko's forward Froude is 0.038–0.048 vs B1's 0.126, flat across gait frequency. A robot that slow
barely moves between frames while its legs fill the view. **A limit of the sensor, not the
coordinate** — it bounds which bodies this method reaches.

**The 2×2 has not been run on gecko and should not be yet.**

> **บทพูด (TH).** หน้านี้แยกออกมาจากทุกอย่างข้างบน เพราะ **B1 เคยอยู่ใน pretrain แต่ gecko ไม่เคย**
> ไม่มี URDF ไม่มี kinematics ไม่มีคลิปครู มีแต่ babble — เป็นงานที่เพิ่งเริ่ม เลยจงใจแยกไว้
> **เจอบั๊กการวัดสองตัว**: อ่านเกตกลับด้าน กับ แกน forward ชี้ขึ้นฟ้า พอแก้แล้ว yaw ที่คิดว่าตายกลายเป็นดีที่สุด
> **สถานะตอนนี้**: ต้องแก้ทั้งสองอย่างถึงจะลงต่ำกว่า 1.0 ได้ (0.970) — B1 อยู่ที่ 0.751
> ที่ต้องให้ดู action 20 เฟรม เพราะ **ความเร็ว gecko มาจากความถี่การก้าว** ซึ่งดูเฟรมเดียวไม่มีทางรู้
> **ที่เหลือคือกล้อง**: action มีข้อมูลพอ (0.736) แต่วิดีโอ ego มีแค่ 0.374 — gecko เดินช้ามาก ระหว่างเฟรมแทบไม่ขยับ
> แต่ขาตัวเองบังเต็มจอ **เป็นข้อจำกัดของเซนเซอร์ ไม่ใช่ของพิกัดกลาง** และยัง**ไม่ควรรัน 2×2 กับมันตอนนี้**

---


## Slide 27 — A real controller now walks, on B1, bounded

**Referenced from Slide 22.** Everything above this slide is selection from a library of recorded
clips. This is a network that invents its own motion from state, trained by RL, that can fall —
and does not, and moves.

**Two real bugs, not six failed mechanisms, explain a whole prior arc of null results.** Building a
real-physics RL controller (Q21 step 3: real MuJoCo physics, the world model only scoring the action
just taken, never rolled forward — deliberately not the mechanism that killed Slide 18's
imagination-RL attempt) produced a policy frozen at a single pose across six independently-tested
fixes (action-space reachability, reward-scale calibration, no-fall-termination, correlated
exploration noise, a command curriculum, a feet-air-time reward). Two bugs, found by testing a
known-working gait through the pipeline rather than trusting the training curve:

| bug | real gait, measured directly | through the bug |
|---|---|---|
| an action-space remap built on the wrong premise (calibrated from an *unbounded* expert-policy action range, not a real requirement) | Froude 0.196 | ~0.005 |
| per-step velocity read with 2 samples through a function built to smooth over ~50 | Froude 0.196 | 0.0025 |

Both reverted/fixed. The same real gait, through the corrected environment end to end: Froude 0.155,
tracking reward 0.336 — against the ~0.09 ceiling every prior configuration hit with zero exception.

**Retrained from scratch on the corrected environment.** 300 updates + 300 more resumed (real PPO
instability along the way: tracking peaked ~0.20-0.22, dropped to ~0.09-0.11, partially recovered —
periodic checkpointing is a named gap, not yet built). Evaluated deterministically, not on the
training-time proxy:

| | value |
|---|---|
| forward Froude, deterministic policy | 0.113, against a goal of 0.105 |
| accumulated forward travel, 4 s | 2.25 m (**0.56 m/s — about twice a normal B1 walk**) |
| **falls per 4-second episode** | **2** |
| **mean body height** | **0.405** (nominal stand 0.56) |

> **This is a fast, unstable lunge — not walking.** Watching the render is what caught it; the
> numbers alone read as success. Two reporting faults did the hiding: displacement was measured
> across the teleport a fall-reset causes (giving a plausible-looking 0.45 m), and "survival" was
> read off the fall flag at the *final* step, which is meaningless once falls stop ending episodes —
> a policy that falls every two seconds still finishes upright. **The cause is a change made earlier
> in this same arc**: removing episode-termination-on-fall (to defeat the "standing still is safe"
> optimum) made falling cheap, and a forward dive is the fastest way to earn forward Froude. This is
> Slide 22's own warning, walked into: *what survival proves — nothing.*

**Bounded, stated precisely.** This is `reward_mode="true_froude"` — ground truth, a diagnostic
never available on a genuinely novel body — not yet the WM-only reward
(`body_head(proj(action))`, Slides 21-24's own subject) that a deployed system would actually have;
that test is the immediate next step, on this same now-corrected environment. Lateral and yaw
tracking remain weak (goal 0.302/0.254, achieved 0.009/-0.048) — this controller tracks forward
speed, not the full three-channel goal.

> **บทพูด (TH).** สไลด์นี้ต่อจาก 28 — ทุกอย่างก่อนหน้าคือการเลือกจากคลังคลิป อันนี้คือ network
> ที่คิดท่าเดินเองจาก state ผ่าน RL จริง ล้มได้ — และไม่ล้ม แล้วก็เดินได้จริง
> สาเหตุที่ผลเป็น null มาตลอด 6 วิธีที่ลอง ไม่ใช่กลไก RL ไหนเลย คือบั๊ก 2 ตัวใน environment เอง
> (1) map action ผิดหลักการ (2) วัดความเร็วต่อ step ผิด ใช้ sample แค่ 2 จุดกับฟังก์ชันที่ต้องการ ~50
> แก้แล้ว วัดด้วยท่าเดินจริงที่รู้อยู่แล้วว่าเดินได้ Froude กลับมาที่ 0.155 จากเดิม 0.0025
> เทรนใหม่แล้วเดินได้จริง: เดินหน้า 0.446 เมตร ใน 4 วินาที ไม่ล้มเลย
> ยังไม่จบ: ผลนี้ใช้ reward แบบรู้ความเร็วจริง (ground truth) ยังไม่ได้ลองกับ reward จาก world model
> ตัวจริงที่ใช้ได้กับหุ่นที่ไม่เคยเห็น และเลี้ยว/ไถลข้างยังทำไม่ได้ดี เดินหน้าเก่งอย่างเดียว

---

## Slide 28 — A checkpoint bug reversed two other results; corrected, the shared coordinate reads better than a trivial baseline

**Two diagnostics run the same night had defaulted to the wrong checkpoint** — one whose shared
Froude head was never actually fit for B1, still carrying whatever the original hexapod-only
pretrain left it at. Corrected to the checkpoint the candidate-selection result (Slide 24) itself
used, both results reverse:

| | wrong checkpoint | correct checkpoint |
|---|---|---|
| a linear function of a raw frame pair vs. the shared head's Froude read, held out | **linear wins** (R² 0.292) | **shared head wins** (R² 0.736) |
| same-behaviour clustering across the two bodies, Froude output, held out | 25% of within-body signal survives | **89% survives, three-fold cross-validated** |

**The clustering result is the more direct evidence for this project's actual claim** — that what
crosses bodies is a shared body-motion coordinate, not a shared latent — measured geometrically
rather than only through candidate-selection accuracy.

**One remaining question, asked and answered in full:** is the shared head's advantage over a
linear baseline just "any nonlinear function would do," or does its specific transition structure
matter?

| | held-out R² |
|---|---|
| linear function of the raw frame pair | 0.486 |
| a generic nonlinear network of matching size, same raw pair | 0.667 |
| **the shared head, reading the inferred transition instead of the raw pair** | **0.736** |

**Most of the gain over linear is just "any nonlinearity" — a real, smaller remainder is specific to
reading the transition rather than the raw pair**, and it holds up better out of sample than the
generic network does (which fits training data almost perfectly and generalises worse). Both
alternatives still lose to the shared coordinate; neither replaces it.

> **บทพูด (TH).** สองผลก่อนหน้านี้ในคืนเดียวกันใช้ checkpoint ผิด (ตัวที่ shared head ไม่เคย fit จริงสำหรับ B1)
> พอแก้เป็นตัวที่ถูกต้อง **ผลกลับด้านทั้งคู่**: จากที่ linear ชนะ กลายเป็น shared head ชนะ (R² 0.736)
> จากที่ cluster ข้ามร่างได้แค่ 25% กลายเป็น **89%** ยืนยันด้วย cross-validation
> **คำถามที่เหลือ**: ที่ชนะ linear เพราะเป็น nonlinear เฉย ๆ หรือเพราะโครงสร้างเฉพาะตัว — ทดสอบแล้วทั้งสองส่วนจริง:
> ส่วนใหญ่มาจาก "ไม่ใช่ linear" (0.486→0.667) ส่วนที่เหลือมาจากโครงสร้างเฉพาะของมันจริง ๆ (0.667→0.736)
> และ generalize ดีกว่าโครงข่ายทั่วไปที่ความจุเท่ากันด้วย

---

## Slide 29 — A second, independent controller attempt: imitation instead of RL, same discipline, same result

**Same rigor as Slide 27, a different route and a different failure mode.** Slide 27's controller is
trained by RL and invents its own motion from state. This asks the same question a different way:
clone a policy directly from B1's own recorded expert clips (34-d proprioceptive state → 12-d joint
target), pre-register the same kind of bar in advance (upright the whole window **and** ≥50% of the
distance the expert itself covers, replayed under the same physics the student is judged in), and
report whichever way it comes out.

**Two real bugs caught by watching the video, not by trusting a number — the same lesson this
project has already learned once.** A first pass reported "246% of D_real, PASS"; the clip showed
the robot walking backward.

| bug | effect once fixed |
|---|---|
| the distance bar (`D_real`) was measured from the wrong starting state — a clip's first recorded action assumes a body already mid-stride, not one freshly reset | replaying the expert's own actions now reproduces its recorded distance at 95.5% fidelity |
| the environment's action clip (`[-1, 1]`, correct for a bounded RL actor) silently truncated a third of this dataset's unbounded expert commands | fixed to reproduce the collection script's own convention exactly |

**With both fixed, no condition clears the pre-registered bar** — and the way it fails depends on
which physics evaluates it, not on the student:

| student | trained-on physics | judged-on physics | result |
|---|---|---|---|
| forward-only | placeholder joint damping | **system-identified (the eval default)** | 52% of D_real, **falls** |
| forward-only | placeholder joint damping | placeholder (matched to training) | 35%, **stays upright** |
| all 3 behaviours | placeholder joint damping | system-identified | 30%, upright |
| all 3 behaviours | placeholder joint damping | placeholder | 7%, upright, barely moves |

**The same weights produce a qualitatively different failure depending only on which physics
executes them** — fast-and-falling under one, stable-and-stationary under the other. That rules out
reading either failure as evidence about the cloning mechanism itself; the physics mismatch between
training and evaluation was never controlled going in.

> **บทพูด (TH).** สไลด์นี้คือ **ความพยายามที่สองที่แยกจาก RL ของสไลด์ก่อน** — รอบนี้ clone นโยบายตรงจาก
> คลิปจริงของ B1 (state 34 มิติ → คำสั่งข้อต่อ 12 มิติ) ตั้งเกณฑ์ผ่าน/ไม่ผ่านไว้ก่อนเทรนเหมือนเดิม
> **เจอบั๊กสองตัวจากการดูวิดีโอ ไม่ใช่จากตัวเลข** (บทเรียนเดิมของโปรเจกต์นี้) แก้แล้วยังไม่ผ่านเกณฑ์ในทุกเงื่อนไข
> **ที่สำคัญกว่านั้นคือ น้ำหนักโมเดลชุดเดียวกัน พังคนละแบบขึ้นอยู่กับฟิสิกส์ที่ใช้ตัดสิน** — เร็วแต่ล้ม กับ นิ่งแต่ไม่ล้ม
> แปลว่ายังสรุปอะไรเกี่ยวกับตัว mechanism การ clone เองไม่ได้ เพราะ confound เรื่องฟิสิกส์ยังไม่ได้ควบคุม

---

## Slide 30 — Grading the clone with the world model: the same mechanism Slide 15 found on the insect, now measured directly on B1 instead of assumed

**Slide 15 already showed this failure once, on the insect, and the reasoning was carried to B1
rather than re-tested:** grading small variations of one behaviour asks the model to rank outcomes
physics itself barely separates (0.1304 against 0.1299), so no representation can order them. Built
the grading stage for B1 anyway, using the properly-fit shared head (Slide 28), to check that
reasoning directly rather than keep assuming it.

**Same physics, same clip, same bar as Slide 29; the only addition is 30 rounds of grading small
perturbations of the cloned policy's own action against the shared head's Froude prediction, and
refitting on the winner.**

| policy | travelled | stays upright | verdict |
|---|---|---|---|
| clone only | 52% of D_real | falls near the end of the window | FAIL |
| **clone + 30 rounds of grading** | **31%** | **falls at roughly the halfway point** | FAIL |

**Grading made it worse, not better — confirmed by watching both videos, not by the number alone:**
the graded policy visibly collapses onto its back well before the clone-only one's later, milder
tip-over.

**Why, measured rather than assumed.** The expert's own recorded actions, replayed through the exact
physics the student is judged in, complete all 66 steps, cover the full recorded distance, and never
once command a joint past its physical limit. The cloned policy's own actions do — climbing from the
expert's typical command size to nearly triple it by the point the body starts to sink, and
beginning to hit joint limits the expert itself never touches. That is a **compounding-error
signature**: a small early deviation from the expert's trajectory reaches a state the policy was
never shown, so it answers increasingly badly, which pushes it further off course. Grading is
supposed to correct exactly that — but if the grader cannot tell nearby actions apart (precisely what
Slide 15 measured), the "correction" it hands back is close to a random label, added on top of an
otherwise-clean training set.

```
  what breaks the clone            what grading was supposed to fix it with
  ────────────────────             ─────────────────────────────────────────
  small drift → unseen state       ask the model: which nearby action recovers best?
  → the policy answers worse       → the model can't tell nearby actions apart (Slide 15)
  → drift compounds                → the "best" pick is close to random
                                    → refitting on it teaches the wrong lesson
```

**Open, not yet tested: whether the model helps through a different mechanism entirely** — not by
ranking discrete nearby actions, but as a training-time signal computed once, directly, by gradient,
at the policy's own action, using the same frozen model. That sidesteps the specific failure measured
here (comparing noisy nearby candidates), but could still fail if the shared head's local sensitivity
is genuinely flat rather than merely noisily estimated — a question this test does not answer either
way.

> **บทพูด (TH).** สไลด์ 15 เจอปัญหานี้บนแมลงแล้ว: การให้คะแนน "ท่าเดียวกันที่เปลี่ยนไปนิดเดียว" คือถามคำถามที่
> **ฟิสิกส์จริงเองก็แยกไม่ออก** (0.1304 กับ 0.1299) ตอนนั้นสรุปว่าไม่คุ้มลองซ้ำที่อื่น — **รอบนี้ลองจริงกับ B1**
> ผล: การให้คะแนนทำให้แย่ลง ไม่ใช่ดีขึ้น (52% → 31%) ยืนยันด้วยวิดีโอทั้งคู่ ไม่ใช่แค่ตัวเลข
> **สาเหตุที่วัดได้จริง**: คำสั่งจริงของ expert เดินครบ 66 สเต็ป ไม่เคยชนขีดจำกัดข้อต่อเลย ส่วนนโยบายที่ clone มา
> ยิ่งเบี่ยงจาก expert คำสั่งก็ยิ่งแรงขึ้นเรื่อย ๆ จนชนขีดจำกัด — คือ **ความผิดพลาดที่สะสมตัวเอง** การให้คะแนนควรจะ
> แก้จุดนี้ได้ แต่ถ้าตัวให้คะแนนเองแยกท่าใกล้เคียงกันไม่ออก มันก็สอนบทเรียนผิด ๆ ทับเข้าไปแทน
> **ที่ยังไม่ได้ลอง**: ใช้โมเดลแบบ backprop ตรง ๆ แทนการเทียบตัวเลือก — อาจเลี่ยงปัญหานี้ได้ หรืออาจล้มด้วยเหตุผล
> เดียวกันก็ได้ ยังไม่รู้

---
