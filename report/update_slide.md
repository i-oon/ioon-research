# Progress Update — Cross-Morphology and Cross-Embodiment Latent Action Models

Stick insect (*Medauroidea extradentata*) and Unitree B1, simulated in CoppeliaSim.

> **Note on format.** Sections 1-10 below are written the way I actually talk through them with the
> committee — plain background/methodology notes, not a numbered slide deck. From Part 2 onward
> (Slide 11 on) the file is still in the earlier, more formal slide-by-slide format; that part
> hasn't been reshaped yet.

## 1. Background — A locomotion controller is fitted to one body

The analogy this thesis attempts is a real one in psychology: **Social Learning Theory**, vicarious
learning (Albert Bandura, 1977) — the acquisition of a behaviour by observing another agent perform
it, with no direct instruction. A held-out body's controller, with no prior knowledge of the
behaviour required of it, is informed only by video of a different body's behaviour. It does not
resemble that body and has never itself performed the behaviour, with no direct instruction and no
kinematic account of either body supplied to bridge them.

- The behaviour crosses through what is *observed*, not by telling the model about the bodies
  involved.
- **The problem.** A locomotion controller maps state to joint commands. With a morphology change,
  the same numerical command can produce a different physical result: the robot stumbles, stands at
  a different height, or does not move at all. Every body needs its own commands for the same
  behaviour.
- **The contribution.** A world model that plans toward a goal defined in a coordinate shared across
  bodies whose action spaces have nothing in common — no kinematic model, no retargeting, no
  existing controller on the target robot. The only thing the two robots share is what a camera
  sees.
- **The gap, stated against what exists:**

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

Everything that crosses leg count needs a body model. Everything that needs no body model doesn't
cross leg count. The lower-right quadrant is empty, and locomotion has no end-effector pose to
retreat to the way manipulation does: eighteen and twelve joint targets share no dimension.

> **บทพูด (TH).** ปัญหาคือ policy ที่เทรนให้หุ่นตัวหนึ่งใช้กับหุ่นตัวอื่นไม่ได้เลย เปลี่ยนความยาวขา
> เปลี่ยนโครงกระดูก ต้องเทรนใหม่ทุกครั้ง แนวคิดที่ตรงที่สุดมีชื่อในจิตวิทยาอยู่แล้วคือ **vicarious
> learning** — เรียนรู้พฤติกรรมจากการ**ดู**ตัวอื่นทำ ไม่ต้องมีใครสอนตรง ๆ ไม่ต้องเคยทำเองมาก่อน และร่างไม่
> จำเป็นต้องเหมือนกัน **โจทย์**: หุ่นคนละร่างสั่งคำสั่งตัวเลขเดียวกันแล้วได้ผลจริงต่างกัน ต้องมีของตัวเอง
> **สิ่งที่เราเสนอ**: world model ที่วางแผนผ่านพิกัดร่วม ไม่ต้อง kinematic model ไม่ retarget ไม่ต้องมี
> controller อยู่แล้วบนหุ่นเป้าหมาย — สิ่งเดียวที่สองตัวแชร์กันคือสิ่งที่กล้องเห็น ดูแผนภาพ: ช่องขวาล่าง
> (ข้ามขาได้ + ไม่ต้อง kinematics) ว่างอยู่ เพราะการเดินไม่มีตำแหน่งปลายมือให้หนีไปแบบ manipulation

---

## 2. Background — Idea forming from the literature

- **Hu, Chen, and Lipson (2025)**, *Egocentric Visual Self-Modeling for Autonomous Robot Dynamics
  Prediction and Adaptation*. They learn a task-agnostic visual self-model for a legged robot from a
  single egocentric camera and random motor babbling, with no prior knowledge of the robot's
  morphology, kinematics, or task, and use it to plan locomotion and even to detect and recover from
  physical damage. Their claim is that the body description was never necessary. Their self-model is
  fitted to one robot at a time: each body is babbled and modelled separately, there is no action
  representation shared between bodies, and nothing is ever transferred from one body to another.
  **Extending that premise across embodiments is what this thesis attempts.**
- **Huang et al., *Cross-Embodiment Robot Foundation World Models with Latent Actions*, ICML 2026.**
  Different robots have different action formats, so they discard explicit action labels as the
  conditioning signal and define a latent action instead — the world model is conditioned on `z`
  rather than on any robot's native command vector. One latent space covers several embodiments, and
  adding embodiments *improves* it rather than fragmenting it. **This is the piece we adapt into
  locomotion.** Their setting is manipulation, where every embodiment already shares one task space
  (end-effector pose); locomotion has no such shared space to condition on, which is exactly the gap
  Section 1 draws.

> **บทพูด (TH).** สองงานนี้เป็นจุดเริ่ม **Hu et al. (2025)** พิสูจน์ว่าไม่ต้องรู้ kinematics เลยก็ควบคุม
> ได้ ด้วยแค่ babble กับกล้องตัวเดียว แต่ทำทีละตัว ไม่เคยแชร์อะไรข้ามหุ่น — งานนี้เอาแนวคิดนั้นไปทดสอบข้ามหุ่น
> **Huang et al. (ICML 2026)** เสนอ latent action แทนคำสั่งดิบของแต่ละหุ่น เพราะหุ่นแต่ละตัวมี action
> format ไม่เหมือนกัน ใช้ latent เดียวครอบหลายร่างได้ และยิ่งเพิ่มร่างยิ่งดีขึ้น — แต่ของเขาคือ manipulation
> ที่มี task space ร่วมอยู่แล้ว (ตำแหน่งปลายมือ) การเดินไม่มีของแบบนั้นให้ยืม นี่คือช่องว่างที่ section 1 พูดถึง

---

## 3. Methodology — Pretraining pipeline

```
   frame ──▶ [ frozen visual encoder ] ──▶ observation
                                               │
                            observation pair ──┴──▶ [ inverse model ] ──▶ latent action
                                                                              │
            observation + latent action ──▶ [ forward model  ] ──▶ predicted next observation
            observation + latent action ──▶ [ command decoder] ──▶ joint command
```

- **Encoder:** a one-billion-parameter video model, frozen throughout, never trained on robots. The
  three modules on top of it are about five million parameters each.
- **Inverse model:** given a transition, what action produced it? — this is where the latent action
  comes from.
- **Forward model:** does the latent action let us predict what happens next?
- **Command decoder:** can the latent action be turned back into an executable joint command? It
  never sees the second frame — whatever the transition contributes has to pass through the 64-number
  latent, and that bottleneck is the whole design.
- **Training signal:** predict the next observation, and recover the real joint command, nominally
  equal weight — but the two losses differ in scale by two orders of magnitude, so next-observation
  prediction takes roughly 99% of the gradient in practice. The term meant to ground the latent in
  real commands runs on the remainder.

> **บทพูด (TH).** encoder เป็นโมเดลวิดีโอพันล้านพารามิเตอร์ **แช่แข็งไว้ ไม่เคยเทรนกับหุ่นยนต์เลย** เทรนแค่
> สามโมดูลเล็ก ๆ บนมัน: inverse model ถามว่า "เปลี่ยนจากภาพนี้ไปภาพนั้น เกิดจากคำสั่งอะไร", forward model
> ถามว่า "คำสั่งที่ถอดได้ ทำนายอนาคตได้ไหม", command decoder ถามว่า "แปลงกลับเป็นคำสั่งข้อต่อที่สั่งได้จริง
> ไหม" — decoder ไม่เคยเห็นเฟรมที่สอง ต้องลอดผ่าน latent 64 ตัวเท่านั้น และ loss สองก้อนต่างสเกลกันร้อยเท่า
> ทำให้การทำนายภาพกินเกรเดียนต์ไปราว 99%

---

## 4. Methodology — Stage 1: cheap test and hypothesis from simple cross-morphology transfer

**The bodies.** Training data uses six-legged walkers differing only in segment lengths. Held-out
bodies are what the model is never trained on, and become the unseen-body test.

**Commands are retargeted.** Data used in training takes one foot trajectory in Cartesian space,
solved separately for each body by inverse kinematics: same intended behaviour, genuinely different
joint numbers.

**Behaviour:** forward walking, one speed.

**The hypothesis.** If the latent truly separates movement from body, then a body never seen in
training should receive commands appropriate to its own geometry — not the commands of whichever
training body it most resembles.

| held-out c08f09t09 | coxa | femur | tibia |
|---|---|---|---|
| the truth | 0.80 | 0.90 | 0.90 |
| probe on the visual encoder (4,227 params) | 0.84 | 0.91 | 0.91 |
| the motion decoder (5.2M params) | 0.62 | 0.96 | 0.96 |

The probe lands close everywhere; the decoder implies a coxa **22% shorter than the body actually
has**. The bigger model is the one that misreads it.

**Swap test:** give the motion decoder body A's frame together with body B's latent. The two
bodies' commands differ by 21°. It answers with body B's command, to within 6°. It followed the
latent and ignored the frame it was holding.

**Diagnosis.** The decoder learned to recognise the closest body in its training set and recall that
body's commands. With this, the whole idea of "make the new body observe the old body and behave
like it" would just fail outright, because the model doesn't understand the frame-action
relationship at all — it's doing lookup, not reading geometry.

**The fix has to be in the objective.** Capacity, access, and the contents of the latent were each
ruled out first: rescaling the target, shrinking the decoder, stripping body identity adversarially,
and handing the decoder a global view of the frame all failed or made transfer worse. What remained
was the objective — no loss function ever forces reading geometry from the frame. Every body shares
the intent and differs only in geometry, so add one loss term: take body A's latent, show the
decoder body B's frame, and require body B's command (`A's latent, B's frame, B's command`).

| held-out body test | without the term | with it |
|---|---|---|
| command error | 3.67° | 3.44° |
| image's worth to the decoder | 0.4× | 9.6× |
| movement's share of the latent | 82% | 93% |
| body identity's share of the latent | 12% | 3% |

The latent stopped carrying a job that was never its own, and the two inputs ended up with separate
jobs: the image carries which body, the latent carries what movement.

> **บทพูด (TH).** หุ่นทุกตัวมีขา 6 ขา 18 ข้อต่อเหมือนกัน ต่างกันแค่ความยาวขา คำสั่งได้จากแก้ IK จาก
> รอยเท้าเดียวกัน — เจตนาเดียวกัน ตัวเลขคำสั่งต่างกันจริง **probe เล็กจิ๋วอ่านความยาวขาของหุ่นที่ไม่เคยเห็นได้
> แม่น แต่ decoder ใหญ่กว่าพันเท่าอ่านผิด** (coxa สั้นกว่าจริง 22%) swap test บอกสาเหตุ: สลับ latent คนละตัว
> มันตอบตาม latent ไม่สนใจภาพเลย — **มันจำหุ่นที่ใกล้ที่สุดในชุดเทรนได้ ไม่ได้อ่านรูปร่างจากภาพ** ลองแก้ที่
> โมเดลมาสี่ทางแล้วไม่ได้ผล เพราะปัญหาไม่ใช่ความสามารถ แต่ loss ไม่เคยบังคับให้อ่านรูปร่างจากภาพเลย — เพิ่ม
> loss term เดียว (latent ของ A คู่กับภาพของ B ต้องตอบคำสั่งของ B) ภาพมีค่าต่อ decoder เพิ่มขึ้น 22 เท่า
> และ latent สะอาดขึ้น (การเคลื่อนไหว 82→93%, ตัวตนของร่าง 12→3%)

---

## 5. Methodology — Stage 1: to work, the held-out body has to sit inside the geometry the data spans

| | ground truth (IK) | control | with the cross-body loss |
|---|---|---|---|
| forward distance | 100% | 85% | 90% |
| heading deviation | 0° | 11.8° | 5.5° |
| worst joint-limit excursion | 0° | 8.2° | 3.9° |

Inside the geometry the training set spans, the predicted commands actually walk, driven open-loop
through the same physics used to collect the data, on a body never trained on.

| same model | error | R² |
|---|---|---|
| a body with geometry inside the training span | 3.4° | 0.81 |
| a body with femur/tibia ratio outside the training span | 13.4° | −0.34 |

Outside that span, the same weights fail — proved, not assumed.

**Test:** same budget, same held-out body, but the training set now contains bodies whose segments
don't move together the way every other training body's do.

| same model | coxa | femur | tibia |
|---|---|---|---|
| ground truth | 1.00 | 1.00 | 0.80 |
| probe fitted on the original dataset | 0.955 | 0.819 | 0.819 |
| probe fitted on the original dataset + the added bodies | 0.973 | 0.954 | 0.772 |

The distance from the held-out body's segment scales to the nearest mixture of the training bodies
predicts this failure on its own — no encoder, no model, no learning at all. Every body that cannot
be mixed from the training set fails; the one that can, succeeds.

**A tool this produced:** before committing to a train/held-out split, fit the probe on the training
bodies and read off how well it recovers the held-out one. A large error there says the split is
asking for a direction the data does not span, and the run will not answer the question meant to be
asked — check this before spending the training run, not after.

> **บทพูด (TH).** ในช่วงรูปร่างที่ข้อมูลครอบคลุม คำสั่งที่ทำนายเดินได้จริง (ระยะ 90%, เลี้ยวเพี้ยนน้อยกว่า
> ครึ่ง) นอกช่วงนั้นพังทันที (13.4°, R² ติดลบ) **สาเหตุพิสูจน์ได้ ไม่ใช่แค่เดา**: ลองเพิ่มหุ่นที่สองท่อนขา
> ไม่เท่ากันเข้าไปในชุดเทรน แค่ probe อย่างเดียว (ไม่ต้องเทรน decoder ใหม่เลย) ก็บอกได้แล้วว่าจะพังไหม —
> **เครื่องมือที่ได้จากตรงนี้**: ก่อนจะ split train/held-out ให้ fit probe บนชุดเทรนแล้วดูว่า probe อ่านตัว
> held-out ได้แม่นแค่ไหนก่อน ถ้าคลาดมากคือ split นั้นถามคำถามที่ข้อมูลตอบไม่ได้ตั้งแต่ต้น

---

## 6. Methodology — Stage 1: one frame nearly determines the command — the gait is a cycle

| what is `e_{t+1}` in the ITM | change |
|---|---|
| ground truth `e_{t+1}` | 3.37° (1×) |
| `e_t` (no transition) | 1.34× |
| `e_{t-1}` (wrong transition) | 1.65× |
| `e_random` (other time) | 3.44× |
| `ITM(None)` | 3.48× |

Without the transition, the motion decoder `MD(e_t, z_t)` cannot work — `z` is carrying the movement
it needs. A wrong transition hurts *more* than no transition; removing the transition entirely costs
31% accuracy. The latent action is genuinely sensitive to what the second frame contains.

```
   a gait is a limit cycle
     one frame shows the pose
       the pose fixes the phase
         the phase fixes what comes next
```

| predict from 1 frame | t | t+8 | t+32 |
|---|---|---|---|
| error (signal spread 11.3°) | 3° | 3.4° | 2.9° |

| steps ahead | `e_{t+h}` | `e_0` | beats `e_0` by |
|---|---|---|---|
| 1 | 1.39 | 2.11 | 1.52× |
| 3 | 1.78 | 3.05 | 1.72× |
| 5 | 2.12 | 3.57 | 1.69× |
| 10 | 2.98 | 4.36 | 1.46× |

But the horizon barely matters — predicting 32 frames ahead is about as accurate as predicting the
present. One frame already identifies which feet are swinging (81.5% accuracy against 50% by
chance), and a second frame is worth only 1.11× on the step-to-step change. There is barely anything
left in the transition for `z` to carry.

Rolling the forward model `FTM(e_t, z_t)` forward with the *true* `z`: the action contributes under
3% of what prediction needs, because the pose in a single frame already says what comes next. That
gap — a module that works, a latent that carries real information, and `z` still worth almost
nothing to prediction — is what Stage 1 leaves open.

**Stage 1 can only prove the pipeline is promising and that the remaining problem is mostly the
structural periodicity of locomotion.** It cannot yet prove that vision helps share behaviour where
proprioception could not, because this setup doesn't have the variety needed for that — it needs a
second body whose action space is genuinely disjoint from the first.

> **บทพูด (TH).** ไม่มี transition, MD ทำงานไม่ได้เลย — z แบกข้อมูลการเคลื่อนไหวไว้จริง transition ที่ผิด
> ยิ่งแย่กว่าไม่มี transition ตัด transition ออกเสียแม่นยำแค่ 31% **แต่ horizon แทบไม่มีผล**: ทำนาย 32
> เฟรมล่วงหน้า ≈ ทำนายปัจจุบัน เฟรมเดียวบอกได้แล้วว่าขาไหนกำลังยก (81.5% เทียบเหรียญ 50%) เฟรมที่สองมีค่า
> เพิ่มแค่ 1.11 เท่า **เหลือให้ z แบกน้อยมาก** — ม้วน FTM ด้วย z จริง action มีค่าต่อการทำนายไม่ถึง 3% เพราะ
> ท่าทางเฟรมเดียวบอกอนาคตไปแล้ว **Stage 1 พิสูจน์ได้แค่ว่า pipeline มีทางไปได้ และปัญหาที่เหลือเป็นเรื่อง
> โครงสร้างวงรอบของการเดิน** ยังพิสูจน์ไม่ได้ว่าภาพช่วยแชร์พฤติกรรมที่ proprioception ทำไม่ได้ เพราะ setup
> นี้ยังไม่มีหุ่นตัวที่สองที่ action space แยกจากตัวแรกจริง ๆ

---

## 7. Methodology — Stage 2: what crossing embodiments with vision requires

Vision-based world models for locomotion and manipulation, and what each leaves open — separated on
purpose so the search across both literatures is visible:

| | what it establishes | what it does not do |
|---|---|---|
| egocentric visual self-model (locomotion) — Hu et al. 2025 | morphology and kinematics are not needed; babble + a camera suffice to plan, and can detect and recover from damage | no cross-embodiment, no transferring between bodies |
| latent-action world models (manipulation) — Huang et al. 2026 | a shared action between bodies, inferred from video alone | no need to map anything, because all bodies already share the same task space |
| cross-embodiment latent-goal planning (manipulation) | a shared latent goal space across different bodies is achievable | built from retargeted, temporally aligned paired demonstrations |
| action-conditioned video world models | name the failure when prediction ignores the action | diagnosed on manipulation or single-body locomotion, never across embodiments — this is Section 6's own problem, still open, and Section 11 is where it gets confronted directly |

**The key thing none of these three hand us: a shared GOAL.** Every one of them either stays inside
one body (Hu et al.) or gets its cross-embodiment coordinate for free, because manipulation already
has one — end-effector pose already means the same thing on every arm. Locomotion has no such
task space. So before anything about *control* can even be asked, there has to be a coordinate that
means the same thing on an 18-joint hexapod and a 12-joint quadruped, built with no kinematic model,
no retargeting, and no paired demonstrations across the two bodies.

**That coordinate is the Froude number, and the claim rides on it entirely:**

```
   Fr = v / sqrt(g · L)
```

speed made dimensionless by body size (`L` ≈ hip height). Two robots of very different size, walking
"the same way," land on the *same* Froude number:

| | hexapod | B1 |
|---|---|---|
| hip height `L` | ~0.09 m | ~0.56 m |
| a normal walk, m/s | 0.13 | 0.29 |
| **same walk, in Froude** | **~0.13** | **~0.13** |

Three channels: forward, lateral, yaw. This single equation is what makes a goal read off one
body's video mean anything at all to a body that shares no joint, no size, and no kinematics with
it — without it there is no shared target to plan toward, and the whole cross-embodiment claim has
nothing to stand on.

**And this is also where the "no prior" claim actually gets tested.** The goal can be produced two
ways: read the true, privileged number directly off the source body's recorded motion, or read it
off nothing but the source body's *video* — no proprioception, no state, no kinematics, exactly what
a genuinely novel body would have to work with. The claim this thesis needs is that the second way is
**good enough to replace the first**: that vision alone recovers the same shared coordinate a
privileged read would, so that nothing about crossing bodies secretly depends on information the
motivating scenario (an animal, a damaged robot, unknown hardware) could never actually supply.

> **บทพูด (TH).** สามงานนี้ไม่มีตัวไหนให้ **เป้าหมายร่วม** มาเปล่า ๆ เลย — Hu et al. ไม่ข้ามร่างเลย, Huang
> et al. ได้พิกัดร่วมมาฟรีเพราะ manipulation มี task space ร่วมอยู่แล้ว (ตำแหน่งปลายมือ) การเดินไม่มีของแบบ
> นั้นให้ยืม **สิ่งที่ต้องมีก่อนจะถามเรื่องควบคุมด้วยซ้ำ คือพิกัดที่ความหมายเดียวกันบนหุ่น 18 ข้อต่อกับ 12
> ข้อต่อ** โดยไม่ต้อง kinematic model ไม่ retarget ไม่ต้องจับคู่ข้อมูล
> **พิกัดนั้นคือ Froude number และข้อเคลมทั้งหมดตั้งอยู่บนสมการนี้**: `Fr = v / sqrt(g·L)` — ความเร็วหาร
> ด้วยขนาดตัว หุ่นคนละขนาดที่เดินแบบเดียวกันได้ค่าเท่ากันเป๊ะ ถ้าไม่มีสมการนี้ก็ไม่มีเป้าหมายร่วมให้วางแผนไปหา
> เลย ข้อเคลมเรื่องข้ามร่างทั้งหมดจะไม่มีอะไรค้ำ
> **และตรงนี้คือจุดที่ทดสอบข้อเคลม "ไม่ใช้ข้อมูลพิเศษ" จริง ๆ**: เป้าหมายทำได้สองแบบ — อ่านตัวเลขจริงที่อัดไว้
> (มีข้อมูลพิเศษ) กับอ่านจาก**วิดีโอ**ล้วน ๆ ไม่มี proprioception ไม่มี kinematics เลย ซึ่งคือสิ่งเดียวที่หุ่น
> ตัวใหม่จริง ๆ จะมี **ข้อเคลมที่ต้องพิสูจน์คือวิธีที่สองต้องดีพอที่จะแทนวิธีแรกได้** — ว่าภาพเพียงอย่างเดียว
> อ่านพิกัดร่วมได้เท่ากับการอ่านแบบมีสิทธิพิเศษ ไม่งั้นข้อเคลมเรื่องข้ามร่างจะแอบพึ่งข้อมูลที่หุ่นตัวใหม่จริง ๆ
> ไม่มีทางมีได้

---

## 8. Methodology — Stage 2: it transfers, but not by sharing a latent

Same setup style as Stage 1: a hypothesis, a term, a measurement. Froude is the shared target
(Section 7). To force it to actually be shared across the two bodies' latents, add one loss term:

```
   L_body = || b_hat_t − b_t ||        b_hat_t = body_head(z_t)
```

Both robots are walked at matched Froude speed, so a readout fitted on one body should work on the
other — a bad score is the representation's fault, not the question's.

| | insect→insect | b1→b1 | insect→b1 | b1→insect |
|---|---|---|---|---|
| frozen encoder | 0.676 | 0.753 | −0.046 | 0.131 |
| control, no term | 0.664 | 0.167 | −7.083 | −2.357 |
| + shared head, λ=0.5 (2 seeds) | 0.798 / 0.815 | 0.879 / 0.881 | +0.544 / +0.749 | +0.435 / +0.704 |
| + shared head, λ=0.1 | 0.809 | 0.868 | 0.675 | 0.624 |

Without the term, cross-robot readout is systematically wrong. With it, both directions go
positive — and the model is not preserving structure V-JEPA2 already supplied (the frozen-encoder
row is itself negative on `insect→b1`); it is creating structure the encoder did not have.

**The same thing, checked one level down, at the raw joint command instead of the Froude scalar.**
This is a different measurement — not a second version of the number above, a decode-level one:
does forcing the shared term also help a body still recover its *own* joint targets, and does it
survive being read out on the other body's joints?

| | within-robot joint error | cross-robot transfer |
|---|---|---|
| joint target, no body term | 0.3517 | −28.9 / −43.1 |
| joint target + shared body term | **0.2183** | **+0.610 / +0.573** |

The term does not just fix the cross-robot number — it improves the robot's own joint decoding by
38%, at both levels of the pipeline.

**But "shared" is not the same claim as "transferred," and this is the open question the term does
not settle.** The readout improving on both bodies is consistent with two different mechanisms: the
latent genuinely learning what a shared body-motion coordinate should look like across robots, or
the term simply making the latent more normalized/well-scaled in a way that happens to help a linear
readout regardless of whether the network anywhere *uses* that shared structure for anything.
Section 9 is what happens when that question gets asked directly.

> **บทพูด (TH).** ไอเดีย: Froude คือเป้าหมายร่วม (section 7) เพิ่ม loss term บังคับให้ latent สองหุ่น
> ถูกอ่านออกมาตรงกันได้จริง (`L_body = ||b_hat_t − b_t||`) — สองหุ่นเดินที่ Froude เท่ากัน ถ้า readout ที่
> fit จากหุ่นหนึ่งเอาไปใช้กับอีกหุ่นแล้วแย่ ก็เป็นความผิดของ representation ไม่ใช่คำถาม
> **ผล**: ไม่มี term การอ่านข้ามหุ่นผิดเพี้ยนสิ้นเชิง (ติดลบหนัก) มี term แล้วทั้งสองทิศเป็นบวก และเช็คอีกชั้น
> ที่ระดับคำสั่งข้อต่อดิบ (คนละตัวเลขกับด้านบน) ก็ดีขึ้นเหมือนกัน — แถมช่วยให้หุ่นถอดคำสั่งของตัวเองแม่นขึ้น
> ด้วย (38%)
> **แต่ "แชร์กันได้" ไม่เท่ากับ "เอาไปใช้จริง"** — readout ดีขึ้นอาจเป็นเพราะ latent แค่ normalize เนียนขึ้น
> ไม่ได้แปลว่า network เข้าใจพิกัดร่วมจริง ๆ **นี่คือคำถามที่ค้างไว้ ให้ section 9 ไปตอบต่อ**

---

## 9. Methodology — Stage 2: fine-tune to adapt to a genuinely different robot

**The setup.** Backbone: the hexapod, pretrained across several behaviours and speeds. Question: can
that pretrain transfer its behaviour understanding to a genuinely different robot — B1 — by adapting
on only a few of B1's own clips, rather than retraining from nothing?

**The staged procedure this uses, assembled from pieces that already existed separately but had
never been run as one pipeline before:**

```
  stage 1  wm.adapt        — fine-tune ONLY the inverse/forward model on B1's own clips
  stage 2  fit_projector   — fit a separate network, the PROJECTOR: action → z, no frame pair needed
  stage 3  wm.adapt3       — optional joint fine-tune (skipped here)
  stage 4  fit_body_head   — refit the shared Froude head against the projector's own latent
```

**Why a projector exists at all.** The `z` used everywhere so far comes from the inverse model
reading a *pair* of frames — it only exists once the next frame has already happened. At the moment
a controller has to pick an action, that frame doesn't exist yet. The projector is a small separate
network trained to guess what `z` an action *would* produce, from the action alone
(`z = proj(action)`), so a control loop has something to plan with before the outcome is known.
Section 10 uses both `z` sources side by side for exactly this reason.

| B1, `z = proj` | forward ρ | lateral ρ | yaw ρ | median ρ |
|---|---|---|---|---|
| zero-shot, frozen head, no adaptation | 0.057 | 0.264 | 0.526 | 0.264 |
| **correct staged adaptation** | **0.572** | **0.449** | **0.670** | **0.572** |

Every channel more than doubles, forward included. **This is the actual cost of bringing up a new
body**, stated as a number: a handful of B1's own clips through stages 1, 2 and 4, not full expert
demonstrations and not training from scratch.

> **บทพูด (TH).** Backbone คือแมลงหกขาที่ pretrain ไว้หลายพฤติกรรม/ความเร็ว คำถาม: เอาความเข้าใจนั้นไปใช้กับ
> หุ่นที่ต่างกันจริง (B1) ได้ไหม โดย fine-tune แค่คลิปไม่กี่คลิปของ B1 เอง ไม่ต้องเทรนใหม่ทั้งหมด
> **ขั้นตอน 4 stage**: (1) fine-tune inverse/forward model บนคลิป B1 (2) fit **projector** — เน็ตเวิร์ก
> แยกอีกตัวที่เดา z จาก action อย่างเดียว ไม่ต้องรอเฟรมถัดไป (เพราะตอนควบคุมจริง ต้องเลือก action ก่อนรู้ผล)
> (3) fine-tune รวมอีกที (ข้ามในรอบนี้) (4) refit shared Froude head **ผล**: ทุกช่องดีขึ้นเกินเท่าตัว
> (median 0.264 → 0.572) — **นี่คือต้นทุนจริงของการเอาหุ่นตัวใหม่เข้ามา** ใช้แค่คลิปไม่กี่คลิปของมันเอง
> ไม่ต้องมี demonstration เต็มรูปแบบ ไม่ต้องเทรนจากศูนย์

---

## 10. Methodology — Stage 2: z had the signal all along; body_head just never learned to read it

**Why `body_head` has to exist at all.** `z` is an opaque 64-number code — nothing in it is
inherently "forward speed," and nothing guarantees coordinate 17 of the insect's `z` means the same
thing as coordinate 17 of B1's `z`. Somebody has to know how to read it — a trained function mapping
`z` onto the one space that *is* shared: the three-channel Froude coordinate. That's `body_head`, and
it's not a training-time extra — it's what every real use calls: reading a goal off video, scoring an
action at control time (`score(a) = |body_head(proj(a)) − goal|`). No `body_head`, no cross-embodiment
claim to test.

**"z already contains Froude" only ever means: some function fitted on z can read it out.** Not that
the raw numbers in z are Froude, or line up the same way across bodies for free. Even the test below
uses a small trained network to do that reading — that network *is* `body_head`'s job, just done
separately. The question is whether the real one is trained well.

**Test: is the signal there regardless of whether the real head found it?** Freeze the same `z`
Section 8 already used, bolt on a fresh head, train it on Froude alone, nothing else competing:

| z source | action-lever gap (bar 0.110) |
|---|---|
| `z = ITM(e_t, e_next)` | +1.048 to +1.226 |
| `z = proj(action)` | +1.049 to +1.092 |

**Passes by ~10×.** So `z` was never the problem — the original `body_head`, co-trained alongside
it, simply never learned to read it this well.

**Why not.** `L_recon`, `L_motion`, `L_body` all push gradient into the same `z`, every step.
`body_head` is trying to learn `z → Froude` while `z` itself keeps getting reshaped by two losses
that have nothing to do with Froude — a moving target it never converges against.

**Fix: `z.detach()` before `body_head`.** `body_head` still trains, but can no longer push back and
reshape `z`. `z` is now shaped by `L_recon`+`L_motion` only. One line, one full retrain, same hex+B1
recipe as Section 8:

| action-lever, real retrain | value |
|---|---|
| real z, median cos | 0.693 |
| mean z, median cos (baseline) | −0.315 |
| **gap (bar: 0.110)** | **+1.008 (~9× the bar)** |

| channel | gap (real − mean) |
|---|---|
| forward | +0.238 |
| lateral | +0.031 (weakest, still positive) |
| yaw | +0.243 |

**Not free — `z` gets measurably worse.** `L_body`'s gradient wasn't pure competition, it was also
doing real work shaping `z`. Cut it, and `z` develops only **32-76%** of its old signal. A real trade:
`z` comes out weaker, `body_head` trains against a stable target instead and actually learns to use
it — net result still 9× better despite the weaker `z`.

**Not yet shown: this improves control** — only that `body_head` is now correctly direction-sensitive.
Whether that becomes working closed-loop behaviour is separate, unproven.

> **บทพูด (TH).** ทำไมต้องมี body_head: z คือเลข 64 ตัวที่ไม่มีความหมายในตัวเอง ต้องมี "ใครสักคนอ่านมันเป็น"
> — คือฟังก์ชันที่เทรนมาแมป z ไปยังพื้นที่ร่วม (Froude 3 ช่อง) ไม่ใช่ของช่วยตอนเทรน แต่คือสิ่งที่ทุกการใช้งาน
> จริงเรียกใช้ (อ่านเป้าจากวิดีโอ, ให้คะแนน action ตอนควบคุม) ไม่มี body_head ก็ไม่มีข้อเคลมข้ามร่างเหลือทดสอบ
> **"z มีสัญญาณอยู่แล้ว" แปลว่าแค่ "มีฟังก์ชันอ่านออกได้"** ไม่ใช่ตัวเลขดิบเป็น Froude เอง — แม้แต่ตอนทดสอบก็
> ยังต้องมีเน็ตเวิร์กเล็ก ๆ อ่านมันอยู่ดี (นั่นคืองานของ body_head) คำถามจริงคือ head ตัวจริงเทรนมาดีพอไหม
> **ทดสอบ**: freeze z ตัวเดิม ต่อหัวใหม่ เทรนอ่าน Froude อย่างเดียว **ผ่าน 10 เท่า** → z ไม่ใช่ปัญหา
> **ปัญหาคือ**: body_head ตัวเดิมเทรนพร้อม loss อื่น (recon, motion) ที่แย่ง gradient เข้า z ตลอด z เลย
> ขยับตลอดเวลา body_head ไล่จับเป้าที่วิ่งหนีไม่ทัน
> **แก้ด้วย** `z.detach()` — body_head เทรนได้ แต่บีบ z ไม่ได้อีก เทรนใหม่รอบเดียว **ดีขึ้น ~9 เท่า**
> **แต่ไม่ฟรี**: z เองอ่อนลงจริง (เหลือ 32-76% เพราะ L_body เคยช่วยสร้างสัญญาณด้วย) — trade คือ z อ่อนลง
> แลกกับ body_head นิ่งพอเรียนได้จริง **ยังไม่ได้พิสูจน์ว่าคุมหุ่นได้จริง** แค่ไวต่อทิศทางที่ควรไวแล้วเท่านั้น

---

# Part 2 — Crossing embodiments: the gap, and the attempt

*(from here down, the file is still in the earlier slide-by-slide format — not yet reshaped)*

**With Stage 2's setup in place, cross-embodiment is ready on the readout side: `body_head` can now
correctly read the shared coordinate out of `z`.** That does not touch the other open problem. Section
6 already found that the action contributes under 3% to next-frame prediction — the forward model
itself barely cares about the action at all, because the pose alone already determines what happens
next. Nothing in Section 9 or 10 changes that; `body_head` and the forward model are different parts
of the network, and fixing one says nothing about the other.

**So this is where that problem gets mapped onto what the field already calls it.** Slide 11 picks
the action-blindness problem back up and asks whether it has a name and a published fix elsewhere —
it does, and rebuilding that fix faithfully is the next step.

## Slide 11 — We rebuilt the field's own fix for this, faithfully

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

## Slide 12 — Six independent measurements, one answer

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

## Slide 13 — The number underneath all six

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

## Slide 14 — Pose determines the future, so the action is redundant

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

## Slide 15 — The principle explains results that are already published

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

## Slide 16 — Three attempts built on this model, and how each was tested

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
                                     ──▶ removed by the egocentric view (Slide 17):
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

## Slide 17 — Moving the camera onto the body removes the redundancy — and the shared coordinate survives it

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

## Slide 18 — What we claim as contributions

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
> in pretrain, gecko never was. **All gecko work is on Slide 27 alone** and appears in no flow
> diagram or result table above it.
>
> **บทพูด (TH).** ตั้งแต่ part นี้ไปคือเรื่อง **motor babble** — วิธีมาตรฐานที่ใช้ตั้งต้นหุ่นที่ยังไม่มี controller
> ขอแยก B1 กับ gecko ให้ชัด: **B1 เคยอยู่ใน pretrain แต่ gecko ไม่เคย**
> **เรื่อง gecko อยู่ที่สไลด์ 27 หน้าเดียว** ไม่ปนกับแผนภาพหรือตารางไหนข้างบนเลย

---

---

## Slide 19 — Imagination-RL: the wall is the rollout


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

## Slide 20 — Sequence context does not fix it


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

## Slide 21 — Stop-gradient clears the lever ~9×


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

## Slide 22 — What the coordinate fixed


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

## Slide 23 — Controller vs. what we test, and what Froude is

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

**Everything on this slide and through Slide 27 is about selection, not control, and was true when
measured.** A real controller exists now — Slide 28, dated, separate, and bounded — but it does not
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

Crossing the two gives the 2×2 (A/B/C/D) on Slide 25 — separated **on purpose**, since bundling
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

## Slide 24 — The correct adaptation pipeline


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

## Slide 25 — The 2×2: which half is broken


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

**Reminder (Slide 23): selections from a library, not a controller.** Nothing here can fall.

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

## Slide 26 — Motor babble, and what we added


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

## Slide 27 — Gecko: the actual unseen body


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


## Slide 28 — A real controller now walks, on B1, bounded

**Referenced from Slide 23.** Everything above this slide is selection from a library of recorded
clips. This is a network that invents its own motion from state, trained by RL, that can fall —
and does not, and moves.

**Two real bugs, not six failed mechanisms, explain a whole prior arc of null results.** Building a
real-physics RL controller (Q21 step 3: real MuJoCo physics, the world model only scoring the action
just taken, never rolled forward — deliberately not the mechanism that killed Slide 19's
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
> Slide 23's own warning, walked into: *what survival proves — nothing.*

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

## Slide 29 — A checkpoint bug reversed two other results; corrected, the shared coordinate reads better than a trivial baseline

**Two diagnostics run the same night had defaulted to the wrong checkpoint** — one whose shared
Froude head was never actually fit for B1, still carrying whatever the original hexapod-only
pretrain left it at. Corrected to the checkpoint the candidate-selection result (Slide 25) itself
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

## Slide 30 — A second, independent controller attempt: imitation instead of RL, same discipline, same result

**Same rigor as Slide 28, a different route and a different failure mode.** Slide 28's controller is
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

## Slide 31 — Grading the clone with the world model: the same mechanism Slide 16 found on the insect, now measured directly on B1 instead of assumed

**Slide 16 already showed this failure once, on the insect, and the reasoning was carried to B1
rather than re-tested:** grading small variations of one behaviour asks the model to rank outcomes
physics itself barely separates (0.1304 against 0.1299), so no representation can order them. Built
the grading stage for B1 anyway, using the properly-fit shared head (Slide 29), to check that
reasoning directly rather than keep assuming it.

**Same physics, same clip, same bar as Slide 30; the only addition is 30 rounds of grading small
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
Slide 16 measured), the "correction" it hands back is close to a random label, added on top of an
otherwise-clean training set.

```
  what breaks the clone            what grading was supposed to fix it with
  ────────────────────             ─────────────────────────────────────────
  small drift → unseen state       ask the model: which nearby action recovers best?
  → the policy answers worse       → the model can't tell nearby actions apart (Slide 16)
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

## Slide 32 — Slide 30's own baseline was buggy in four separate ways; fixed, plain cloning passes

**Same student, same physics family, four independent bugs found in Slide 30/31's own setup —
not a new mechanism, the measurement underneath it.** Each was found and fixed one at a time, each
verified before moving to the next, on the same held-out clip throughout:

| # | bug | fix |
|---|---|---|
| 1 | trained on placeholder-physics data, evaluated on the system-identified model — the exact confound Slide 30 already names, never actually checked | re-collected training data directly on the system-identified model |
| 2 | recorded actions are 20 Hz, replayed one row per 50 Hz physics step — every clip played back 2.5× too fast, at 40% of its recorded resolution | hold each action for its real 0.05 s, not the environment's 0.02 s substep |
| 3 | the goal fed to the student is the whole-clip mean Froude, dragged 12% below cruising speed by every clip's accel/decel edges | average only the steady-state middle window |
| 4 | the expert's straight-line behaviour depends on a live external heading-correction signal the student's state never included | append live heading error as a 35th state input |

```
  bug 1 fixed → falling stops, travel 20% (stable but slow — a DIFFERENT failure)
       │
  bug 2 fixed → 20% → 29%
       │
  bug 3 fixed → 29% → 37%
       │
  bug 4 fixed → drift halved (53.8° → 25.7°), but distance drops again (37% → 23%)
       │            a real trade-off, not a regression: same total action effort,
       │            redirected toward correction instead of peak speed
       ▼
  still short of the bar — one thing never in any of these fixes: an example of RECOVERING
```

**The fifth fix closed it: the training data never demonstrated recovery.** Every clip starts at
zero heading error by construction, so bug 4's fix gave the student the right *input* with no
example of what to *do* with a large value of it. Added spawns rotated 10/20/30° off target, target
pinned at the canonical heading — six new clips, confirmed to show real, scaled recovery (e.g.
+25.6° → +1.8° within one clip).

| student | travelled / expert distance | upright | verdict |
|---|---|---|---|
| plain BC, Slide 30's original buggy baseline | 52% | falls | FAIL |
| + all four bugs fixed, no recovery data | 23-37% (varies by fix) | stable | FAIL |
| **+ recovery data** | **65%** | **stable** | **PASS** |

**What this settles, and what it doesn't.** A properly-measured, properly-informed plain clone
*can* clear the bar for this one goal — none of Slide 31's grading/gradient mechanisms were
necessary, the blocker was measurement and a missing input, not a need for a teacher. It does not
settle whether a teacher earns its keep *on top of* a baseline that already works, since this one
needed no such mechanism to start working. And this result is forward-walking only, in isolation —
Slide 33 is what happened when the same goal-conditioned student was asked to handle more than one
behaviour at once.

> **บทพูด (TH).** นักเรียนตัวเดิม ฟิสิกส์ตระกูลเดิมจากสไลด์ 29/30 — **เจอบั๊กสี่ตัวในสิ่งที่วัดผล ไม่ใช่กลไกใหม่**
> แก้ทีละตัว: (1) ฟิสิกส์เทรน/ประเมินไม่ตรงกัน — confound ที่สไลด์ 29 เอ่ยไว้แต่ไม่เคยเช็คจริง (2) เล่นคลิปเร็วเกิน 2.5
> เท่า (3) เป้าหมายที่ป้อนถูกดึงต่ำลงจากช่วงเร่ง/ชะลอความเร็ว (4) ไม่มีสัญญาณแก้ทิศทางที่ expert ใช้จริง
> แก้ครบสี่ข้อ **ยังไม่ผ่านเกณฑ์** (23-37%) เพราะไม่มีข้อมูลตัวอย่าง "หลุดแล้วกลับมา" เลยสักคลิป — เพิ่มคลิปที่ปล่อย
> ให้หันเบี่ยงจากเป้า 10/20/30 องศาแล้วบังคับให้กลับมา **ผ่านทันที 65%** ยืนได้ตลอด
> **สรุป**: การ clone เฉย ๆ ที่วัดถูกต้องก็ผ่านได้ ไม่ต้องมีตัวช่วยให้คะแนนแบบสไลด์ 30 เลย — ปัญหาคือการวัดกับ
> input ที่ขาดไป ไม่ใช่ตัวโมเดล **แต่นี่คือเดินหน้าอย่างเดียว** สไลด์ 32 คือตอนให้ทำหลายพฤติกรรมพร้อมกัน

---

## Slide 33 — Reframed, and where the multi-behaviour gate still fails

**A methodological correction came before this result, and governs how to read it.** Slide 32's
student is trained and tested on B1's *own full expert data* — every goal it is asked about has a
labelled demonstration behind it. That is a fair, cheap **gate**: does a goal-conditioned clone even
change behaviour correctly when given a different goal, before anything cross-embodiment is
attempted? It is not yet the thesis's actual claim, which needs a genuinely novel body with *no*
full expert demonstrations, only babble.

```
  STEP 1 — the gate (this slide)          STEP 2 — the real claim (not started)
  B1's own labelled expert data      ──▶  babble instead of expert data
  cheap, fast to iterate                  the actual thing a novel body would have
  privileged, NOT the thesis claim        gated on step 1 passing first
```

**The gate is not yet passed.** Slide 32 only ever tested one goal family (forward) in isolation.
Asked to condition on speed, turn, *and* side goals from the same 54-clip mixed set, using the exact
same recipe that passed alone, a new failure appeared:

| goal family | what happened |
|---|---|
| speed, turn | **static-pose collapse** — action converges to a fixed, non-cyclic vector; body freezes within 10-20 steps and never moves again |
| side | reaches 133% of expert distance, but **falls** |

**Ruled out one at a time, each checked directly rather than assumed:** fall-contaminated training
data (checked every clip's own height trace — clean), action-normalisation skew across the three
behaviours (per-behaviour action statistics nearly identical), undertraining (5000 vs. 2000 epochs:
no offline or closed-loop improvement). Removing the heading-error input eliminates the freeze
specifically — all three goals move again — but then all three **fall** instead (75-120% of
distance, min body height ~0.09 m vs. ~0.58 m settled). The heading-error input is a confirmed
contributing cause of the freeze, not the whole story.

**The decisive check: even a training goal, fit to R² 0.99 offline, still falls in closed loop.**
This rules out "can't generalise to unseen goals" entirely — it is the same compounding closed-loop
drift Slide 30 already diagnosed for forward-only, before recovery data fixed that one case. Turn
and side simply never received the equivalent treatment. Built the natural extension — recovery
clips for turn, spawned off-heading with the target advancing at the real turn rate instead of
sitting still — and retrained on the enlarged 60-clip set:

| behaviour | result |
|---|---|
| speed | 13% of expert distance, stable (slow, not frozen — a third distinct failure) |
| turn | 132%, falls |
| side | not yet given recovery data at all |

**Neither the original collapse nor a clean pass — a still-unresolved failure mode.** Whether it
needs more/better recovery coverage, a different state representation, or something else entirely is
open. **Paused here** to redirect effort to writing.

**One further avenue considered and also paused, for the same reason.** The one existing real
controller (Slide 28) has no goal input at all — its network only ever tracks one fixed goal it was
trained against, the Froude channel is used to *shape the reward*, never fed to the policy. Making
it genuinely goal-conditioned (so a goal read from another body's video could later drive it) needs
a real architecture change and a full retrain, not a rerun — and it is the identical open question
above, moved into a slower, noisier setting to debug. Held for after this gate resolves.

> **บทพูด (TH).** ก่อนอ่านผลนี้ต้องปรับกรอบก่อน: **สไลด์ 31 เทรนและทดสอบบนข้อมูล expert เต็มของ B1 เอง** —
> ทุกเป้าหมายมีตัวอย่างสาธิตรองรับ นี่คือ **ด่านทดสอบราคาถูก** ว่ากลไก goal-conditioning ทำงานถูกไหม ก่อนจะไป
> ข้ามร่างกายจริง **ยังไม่ใช่ข้อเคลมของวิทยานิพนธ์** ซึ่งต้องใช้หุ่นใหม่ที่ไม่มี expert data เต็ม มีแค่ babble
> **ด่านนี้ยังไม่ผ่าน**: ให้เทรนพร้อมกันสามพฤติกรรม (เดินหน้า/เลี้ยว/ไถลข้าง) เจอ **การแช่แข็งท่ายืนนิ่ง** ในเดินหน้า/
> เลี้ยว และไถลข้าง **ล้ม** ทั้งที่ไปได้ไกลพอ
> ตัดสาเหตุออกทีละอัน: ข้อมูลปนเปื้อน (ไม่ใช่), สเกลคำสั่งเพี้ยน (ไม่ใช่), เทรนไม่พอ (ไม่ใช่) — เอาสัญญาณแก้ทิศทาง
> ออก อาการแช่แข็งหายแต่กลับไปล้มแทน **แม้แต่เป้าหมายที่เคยเห็นตอนเทรนก็ยังล้ม** — พิสูจน์ว่าไม่ใช่เรื่อง generalize
> แต่คือ compounding drift แบบเดียวกับที่เจอตอนเดินหน้า เพิ่มข้อมูล recovery สำหรับการเลี้ยวแล้วเทรนใหม่ — **ยังไม่ผ่าน
> อีก เป็นความล้มเหลวแบบที่สาม** (เดินหน้าช้าแต่นิ่ง, เลี้ยวยังล้ม) **หยุดตรงนี้** เพื่อไปโฟกัสงานเขียน
> **อีกทางที่พักไว้เหมือนกัน**: PPO controller ตัวเดียวที่เดินได้จริง (สไลด์ 27) ไม่มีช่องรับเป้าหมายเลย ถ้าจะทำให้
> รับเป้าได้ต้องแก้สถาปัตยกรรมและเทรนใหม่ทั้งหมด ซึ่งเป็นคำถามเดียวกับด้านบน แค่ย้ายไปอยู่ใน RL ที่ช้าและวุ่นกว่า

---
