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