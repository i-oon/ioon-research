## Summary — Q17: What is the research gap, in one sentence

**Final answer:**

> No published method learns a controller for legged locomotion, from video alone, that transfers across robots with disjoint leg counts. Methods that avoid a kinematic tree/URDF stay within one robot (Li et al. 2020) or one leg-count family (QWM); methods that span leg counts use kinematic retargeting (X-Morph) or a shared task-space coordinate that only exists for manipulation (LAC-WM) — and LAC-WM, on inspection, doesn't even generate the controller itself, it *selects* among a pretrained VLA's proposals via image-closeness reranking. This project's gap is producing the controller directly, from video and joint count alone, in a setting with no shared coordinate and no existing policy to select from — but only by adding a body-motion term, since MSE alone was found to silently discard the action channel across morphology families.

---

## What the three novelty checks found

| Check | Result | Confidence |
|---|---|---|
| **1. Do cross-morphology locomotion methods need a kinematic tree/URDF?** | Yes, consistently (URMA/GenBot-1K, MS-PPO, DexGrasp-Zero, canonical URDF work, X-Morph's retargeting, QWM's morphology encoder). One nuance: non-privileged latent spaces *within* a single robot already exist (Li et al. 2020) — the gap has to stay specifically about *cross-body* transfer, not "no kinematics at all." | Solid |
| **2. Has anyone learned a shared latent action space across leg counts from video?** | No. Closest candidates (Li et al., X-Morph, QWM) all either use one robot, use kinematics, or stay within one leg-count family. This is the load-bearing check — it held up under two rounds of digging. | Solid, stress-tested twice |
| **3. Is a contrastive term in an action-conditioned world model already published?** | Yes (CAPE, CD-LAM, VITA). As anticipated, this narrows contribution (b) from "the fix" to "the finding" — the diagnosis that MSE silently discards the action channel across morphology families is the actual contribution, not InfoNCE itself. | Solid, low stakes either way |

---

## Papers to explicitly pre-empt in OPEN_QUESTION.md

Cite these four rather than risk a committee member surfacing them cold — each sharpens the gap rather than closing it:

1. **Li et al. 2020** — hexapod + quadruped, but separate per-robot latent spaces, proprioceptive not video, no cross-body transfer claim. *Useful to steal:* their baseline taxonomy (learned latent vs. expert library vs. model-free) as an ablation template; their damaged-leg stress test as a cheap extra result.
2. **X-Morph (2026)** — quadruped/hexapod/quadruped+arm, but explicit kinematic retargeting. *Useful to steal:* as a fallback data-augmentation idea if you're ever clip-starved.
3. **QWM (Aug 2026)** — confirmed generator, zero-shot cross-embodiment, but same-leg-count quadrupedal family only, morphology-conditioned (not kinematics-free). *Useful to steal:* the "train inside imagination" recipe as a path to scale beyond 2 embodiments later.
4. **LAC-WM** — shares a task-space coordinate (end-effector), but confirmed via the actual paper text to be a **selector**, not a generator: a VLA proposes candidates, LAC-WM reranks by predicted-image closeness to a subgoal. This turned out to be part of a broader pattern (STORM, World Action Planner do the same move) — worth naming as a family, not a one-off.

---

## The generator-vs-selector axis (new, worth adding to the table)

This was the most valuable thing that came out of the deeper digging — a cleaner cut than the original kinematics-only axis:

| | generates the controller, or selects from an existing one's proposals? |
|---|---|
| LAC-WM, STORM, World Action Planner | **Selects** — needs a pretrained VLA to propose from |
| Li et al., X-Morph, QWM | **Generates** — but each still needs a kinematic tree, URDF, or same-leg-count family |
| this project | **Generates**, from video + joint count alone, across disjoint leg counts |

---

## What's still open

- Check 2 remains a hypothesis with two rounds of search behind it, not a certainty — worth one more pass if time allows, but nothing in three searches came close.
- The three-caveat list in your original draft still stands unchanged: circular candidate library (Q16), speed uncontrolled on both robots, and the LAC-WM scaling-with-embodiments result unreproducible with two robots.
- Not yet checked: whether any of the other near-miss papers (Li et al., X-Morph, QWM) are *also* secretly selectors once read past the abstract — this pass confirmed they're not, so that thread is closed for now.


- "สุ่มคำสั่ง/สุ่มรบกวนรอบท่าเดิน" — สุ่มรอบ ท่าเดินที่มีอยู่แล้ว (perturb a known gait) กับสุ่มแบบไม่มี prior เกี่ยวกับการเดินเลย (pure random torque/joint babble) เป็นคนละระดับของ "ไม่ต้องมีท่าที่ถูก" ถ้าใช้แบบแรก กรรมการอาจถามกลับว่า "ก็ยังต้องมีท่าเดินเริ่มต้นอยู่ดี ไม่ใช่หรือ" ควรตัดสินใจล่วงหน้าว่าจะทำแบบไหน แล้วเขียนเหตุผลไว้ในเอกสารว่าทำไมถึงยังนับเป็น "ไม่ต้องมีท่าที่ถูก"
- teacher = planner ที่ดูวิดีโอ ต้อง freeze ก่อน distill หรือ distill ระหว่าง training พร้อมกัน — ถ้า freeze แยกสองสเต็ปจะตรวจสอบง่ายกว่าและตัด confound ได้ชัดกว่า
ควรทำข้อ 1 ก่อนข้อ 2 เพราะถ้าข้อ 1 ไม่ผ่าน (babble ไม่พอ) ข้อ 2 ก็จะพึ่ง curated data อยู่ดี — ข้อ 1 คือ gate ของข้อ 2
ตารางที่เสนอ (เลือก/สร้าง+URDF/เรา) ตอนนี้ควรมีคอลัมน์สถานะกำกับไว้ด้วย เพราะแถวสุดท้ายยังเป็น "เป้าหมายที่ยังไม่ปิด" ไม่ใช่ "ผลที่ได้แล้ว" — ถ้าเอาไปให้ advisor ดูตอนนี้แบบไม่กำกับสถานะ จะเจอคำถาม "24 คลิปมาจากไหน" แน่นอนตามที่กังวลไว้ ทางที่ปลอดภัยที่สุดคือใส่แถวนี้เป็นสองเวอร์ชัน — "as claimed" (เป้าหมาย) กับ "as it actually stands" (ตอนนี้) แบบเดียวกับที่ทำไว้ในโครงร่าง Q17 เดิม ซึ่งตรงกับ pattern ที่มีอยู่แล้วในเอกสาร ("ours, as claimed" vs "ours, as it actually stands") — ไม่ต้องคิดใหม่ ใช้โครงเดิมได้เลย

สรุปสิ่งที่ควรทำต่อ

ล็อกเกณฑ์ของข้อทดลอง 1 ไว้ล่วงหน้า (นิยาม babble แบบไหน, threshold เท่าไหร่ถึงนับว่าผ่าน) แล้วค่อยรัน
ถ้าผ่าน → รันข้อ 2
อัปเดตแถวสุดท้ายของตารางเป็นสองเวอร์ชันตามสถานะจริง พร้อม flag ว่าข้อ 3 (video+joint count) ยังไม่ปิดจนกว่าข้อทดลอง 1-2 จะเสร็จ
เชื่อม claim นี้เข้ากับ Q17_ANS เวอร์ชันล่าสุด (objective-at-adapt-time) — สองอันนี้เป็นคนละแกนแต่ต้องไม่ขัดกัน ตัวที่เป็นแกนหลักคือ objective, ตัวนี้เป็น secondary claim ที่ยังอยู่ระหว่างพิสูจน์


## Where "ours" stands vs. every alternative — full positioning summary

**The one-line version:**

> Every alternative either needs a human to define correct behavior on the target robot, needs a kinematic model to define correspondence, or (like LAC-WM) needs an existing policy to select from. Ours needs none of the three — it needs unlabelled interaction and the transferred latent action space.

---

### The full landscape, by what each approach requires

| Approach | Requires on the new robot | Generates or selects? | Why it's not a real alternative |
|---|---|---|---|
| **Vanilla RL from scratch** | a hand-designed reward function | Generates | Reward engineering is the same cost as kinematic engineering, just relocated — throws away everything the video-trained latent space already knows |
| **BC on curated B1 clips** | 24 human-picked "correct" demonstrations | Generates | Circular — a human decided what counts as correct walking/turning/strafing before training starts (Q16) |
| **this project, as currently built** | the same 24 curated clips | Selects | **The same circularity, deliberately, and it is the measuring instrument rather than the claim.** "Did the world model pick the right action" is only answerable against a set of actions whose correct answer is already known; a library with labels is what makes selection scorable at all. It bounds one claim and not the others: it is why "a camera is the only thing it needs" is listed below as unproven, and it does not touch whether the latent picks correctly, which is what the library exists to test |
| **BC on bio (insect) video** | *not well-defined* — no `(obs, action)` pairing exists without... | — | ...either hand-labeling the correspondence, or a kinematic retargeting step (X-Morph's move), or a learned cross-embodiment map — which *is* the pipeline itself. Not a baseline; it's excluded, not beaten. |
| **LAC-WM / STORM / World Action Planner** | a pretrained, competent VLA to propose candidates | **Selects**, doesn't generate | Presupposes the hard part (a working policy) already exists; only reranks its outputs |
| **Li et al. 2020 (hexapod+quadruped)** | separate expert demonstrations per robot | Generates | No shared latent across bodies — same method applied twice, not one transferred representation |
| **X-Morph** | a URDF / kinematic retargeting stage | Generates | Solves correspondence with a body model, exactly the cost this project avoids |
| **QWM** | morphology parameters (scale-invariant descriptor) | Generates | Zero-shot only *within* the quadrupedal family — never crosses leg count |
| **Dreamer-style RL in imagination** | a reward, and a world model accurate over the training horizon | Generates | Moves the cost to reward design and to long-horizon accuracy, this project's weakest measurement; QWM is this within one leg-count family |
| **BC on the same 24 clips we use** | nothing we do not already need | Generates | **The comparison that has not been run and is the sharpest "so what".** If behaviour cloning on the identical clips matches the planner, the world model is buying nothing on this data |
| **Ours (target state)** | unlabelled interaction on the target robot, plus a goal video from another robot | Generates | No hand-designed reward, no curated demo, no kinematic model, no pretrained policy to select from — the latent action space does the correspondence work. **The supervision is not nothing: it is a video of another robot doing the thing** |

---

### The three claims, and where each actually stands right now

Since this was already flagged precisely: **"ours" is three separate claims bundled into one row, and they're not equally proven.**

1. **"Across disjoint leg counts"** — ⏳ demonstrated in a controller, not yet stable. **Do not cite
   F82 or F83 for it**: F82 is a positioning argument rather than a measurement, and F83's transfer
   numbers come from the `stage2_*` checkpoints on pre-F74 data, while the checkpoint that closes
   the loop has no cross-embodiment shared target at all (F110). The load-bearing evidence is the
   control result -- a quadruped walks forward and turns the correct way from a stick insect's
   video, above chance on both (F116) -- and its caveat is that each arm has been trained once, with
   the ordering moving between budgets, which is what the three-seed run is for.
2. **"Generates"** — ⏳ well-defined, testable, not yet run (distillation experiment).
3. **"From video + [unlabelled interaction], not curated demonstrations"** — ⏳ the one that's currently overclaimed as "video + joint count alone"; needs experiment 1 (babble-fit projector) to hold before it's honest.

The correct sentence for the doc, combining everything from this conversation:

> No method in the surveyed landscape generates a legged locomotion controller for an unseen robot from unlabelled interaction alone: reward-based RL needs a hand-designed reward, imitation needs curated or kinematically-retargeted demonstrations, and the closest cross-embodiment world models (LAC-WM and its family) only rerank an already-competent policy's proposals rather than generating one. This project's claim to occupy that gap is currently proven for cross-leg-count transfer and well-defined for the generation step, but not yet proven for the "no curated demonstration" step — that depends on the babble-fit projector experiment currently pending.

That's the accurate, defensible version — strong where the evidence is strong, explicit about the one leg that's still standing on a plan rather than a result.