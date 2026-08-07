# โน้ตอธิบายงานวิจัย — Cross-Morphology Locomotion via Latent Action World Models

---

## 1. ปัญหาคืออะไร

### ปัญหาพื้นฐาน

ในงานด้าน robot locomotion ปัจจุบัน เวลาจะสอน robot ให้เดินได้ เราต้องเทรน policy ซึ่งเป็น neural network ที่รับ state ของ robot เป็น input แล้ว output ออกมาเป็น action เช่น จะหมุน joint ไหน เท่าไหร่ ปัญหาคือ policy นี้ specific กับ body ของ robot ตัวนั้นมากๆ ถ้า body เปลี่ยนไป เช่น

- ขาสั้นลงหรือยาวขึ้น
- เปลี่ยน mass distribution
- robot ได้รับความเสียหาย ขาหัก

policy เดิมใช้ไม่ได้เลย ต้องเริ่มเทรนใหม่ตั้งแต่ต้น ซึ่งแต่ละครั้งใช้เวลาหลายชั่วโมงถึงหลายวัน

### สโคปที่เราทำ (หลังพูดคุยกับ Ajan Go — Week 4, Week 5)

เราไม่ได้ทำ biological video แล้ว เปลี่ยนมาทำใน **simulation ทั้งหมด** โดยใช้ **stick insect 3 morphologies** คือขาสั้น (0.5×), ขากลาง (0.75×), และขายาว (1.0× = base model) ใน **CoppeliaSim v4.10**

เป้าหมายหลัก: เทรน World Model บนขาสั้น + ขายาว แล้วพิสูจน์ว่า World Model ช่วยให้ขากลาง (ที่ไม่เคยเห็นมาก่อน) เรียนรู้ได้เร็วขึ้น โดยใช้ข้อมูลน้อยกว่า

---

### 🔄 อัปเดตทิศทาง (2026-08) — staged: cross-morphology → cross-embodiment
> รายละเอียดเต็มที่ `PROGRESS.md` §12. เป้าหมายด้านบน (cross-morphology) กลายเป็น **Stage 1** ของแผน 2 ชั้น

คำถามหลักของกรรมการ ("ทำไม vision ถึงคุ้มกว่า proprioception?") ตอบด้วยร่าง topology เดียวกันไม่ได้ — 3 ความยาวขาใช้ **18-D เหมือนกัน** proprioception ก็แชร์ได้ → vision ได้เปรียบแค่ **reach** ไม่ใช่ข้อพิสูจน์ เพื่อ *พิสูจน์* ต้องเพิ่มร่างที่ action space **disjoint** (แชร์ proprioception ไม่ได้เลย แต่ vision แชร์ได้)

- **Stage 1 — cross-morphology** (Step -1 … 2 ในไฟล์นี้): 3 ความยาวขา, **IK-retargeting** (a_t ต่างต่อร่างใน space 18-D เดียวกัน). ทำ pipeline ให้เดินได้ + latent จัดกลุ่มตาม behavior + ablation latent vs raw-joint. พิสูจน์ "latent ดีกว่า" **แต่ยังไม่พิสูจน์ vision>proprio** (topology เดียวกัน)
- **Stage 2 — cross-embodiment / compositional transfer**: เทรนบน **6-leg stick insect + Unitree B1
  quadruped (12-D)** แล้ว test บน **4-leg stick insect**. นี่คือการตอบ feedback กรรมการว่า
  3 ความยาวขาง่ายเกินไป: train set มี action space คนละชนิด (hexapod 18-D vs B1 12-D) ที่ proprioception
  แชร์ตรงๆ ไม่ได้ แต่ vision แชร์ได้; 4-leg test body แชร์รูปร่างกับ insect และแชร์จำนวนขากับ quadruped
  จึงเป็น held-out body ที่ทดสอบการ compose ความรู้สองด้าน. ต้องหา/เทรน 4-leg walker เอง; policy cutlegs
  ของแล็บใช้ไม่ได้ (obs config drift)

**คำศัพท์**: disjoint action space (Stage 2) **≠** IK-retargeting (Stage 1). IK = ค่า a_t ต่างกันใน **space เดียวกัน** (comparable — proprioception ยังแชร์ได้); disjoint = **คนละ space** ไม่มี correspondence (proprioception แชร์ไม่ได้) — นี่คือเหตุผลที่ 2 stage พิสูจน์คนละอย่าง

---

## 2. Intuition — ทำไมถึงทำแบบนี้

### ทำไมต้องการ morphology-agnostic representation

ถ้าเราสั่งขาสั้น "ยกขาสูง 20 องศา" แล้วมันเดินได้ แต่ถ้าเอา command เดียวกันไปใส่ขายาว ขาอาจลากพื้นเพราะยกสูงไม่พอ Ajan Go ยกตัวอย่างนี้เพื่อแสดงว่า **morphology gap มีอยู่จริง** และนั่นคือ Step -1 ที่เราต้องเช็คก่อนเลยว่าขาสั้น/ยาวทำให้ behavior ต่างกันจริงๆ

(need close up/ slow down video to prove this)

สิ่งที่เราต้องการคือ latent variable **z_t** ที่ encode "พฤติกรรม" (เช่น เดินตรง, เลี้ยว, หยุด) โดยไม่ encode "รูปร่างร่างกาย" ถ้า z_t เป็นแบบนั้น มันก็ transfer ข้าม morphology ได้

### ทำไมถึงใช้ video + visual encoder แทน joint state โดยตรง

แรงบันดาลใจมาจาก **LAC-WM** (ICML 2026) ที่แสดงว่าการ extract latent action จาก visual observation ทำได้และ scale ได้ข้าม embodiments หลายตัว นอกจากนี้ visual encoder ที่ pretrain บน internet video จำนวนมาก เช่น V-JEPA2 มี feature ที่ rich และ generalizable กว่าการใช้ joint state ดิบๆ


---

## 3. Stack ทางเทคนิค

### 3.1 Visual Encoder — V-JEPA2

**V-JEPA2** พัฒนาโดย Meta AI (2025) เป็น ViT-g/16 ขนาด 1B parameters ที่ pretrain บน internet video กว่า 1 ล้านชั่วโมง (VM22M, 22 ล้านวิดีโอ) ด้วย mask-denoising objective ใน representation space

- **Frozen** ตลอด Phase 1 — ไม่มี gradient ไหลผ่าน
- Input: frame ∈ ℝ^{256×256×3}
- Output: 256 patch tokens ∈ ℝ^{1408} ต่อ frame
- ใช้ 3D-RoPE positional embedding

เหตุผลที่ใช้แม้ pretrain บน general video ไม่ใช่ locomotion โดยตรง: V-JEPA2 เรียนรู้ motion-relevant features จาก video ทั่วไป feature เหล่านี้ (เช่น การเคลื่อนไหว, spatial structure, temporal change) transferable ไปยัง locomotion ใน simulation ได้ Step 0 จะ verify ก่อนว่า feature จาก V-JEPA2 มี locomotion signal หรือเปล่า

### 3.2 Cross-Augmentation

ก่อน encode เราทำ augmentation กับ frame pair (O_t, O_{t+1}) สองครั้งด้วย independent random parameters A1, A2 ได้ embedding pair สองชุด:

```
(x_t¹, x_{t+1}¹) = encode(A1(O_t), A1(O_{t+1}))   → ส่งเข้า ITM
(x_t², x_{t+1}²) = encode(A2(O_t), A2(O_{t+1}))   → ส่งเข้า FTM
```

- ITM ใช้ pair 1: z_t¹ = ITM(x_t¹, x_{t+1}¹)
- FTM ใช้ pair 2 + z_t¹: ê_{t+1}² = FTM(x_t², z_t¹) แล้วเทียบกับ x_{t+1}² จริง (L_recon)

จุดประสงค์ (ตามที่ LAC-WM paper ระบุตรงๆ ใน section "Cross-Augmentation Inputs"): เพราะ z_t ถูก supervise บางส่วนด้วย L_recon ITM มี incentive จะ **cheat โดยยัด x_{t+1} เข้าไปใน z_t ตรงๆ** แทนที่จะเรียนรู้ action จริง เพราะแบบนั้นก็ทำให้ predict แม่นได้เหมือนกัน (ผิดจุดประสงค์) cross-augmentation ตัด shortcut นี้ทิ้งเพราะ x_{t+1}¹ ที่ ITM เห็น (จาก aug1) ไม่ตรงกับ x_{t+1}² ที่ FTM ต้อง predict (จาก aug2) — ถ้า z_t แค่ copy x_{t+1}¹ มาตรงๆ จะ predict x_{t+1}² ผิด

**หมายเหตุสำคัญ**: cross-augmentation ป้องกัน shortcut แบบ "copy future frame ตรงๆ" ได้ แต่**ไม่ได้การันตีว่า z_t จะไม่ encode morphology** เพราะรูปร่างร่างกาย (body shape) เป็น content จริงที่ยังอยู่ไม่ว่าจะ crop/color-jitter/flip ยังไง (ต่างจาก texture/color ที่เป็น nuisance ที่ augmentation ทำลายได้) นี่คือเหตุผลที่ **Step 1.5 ต้องมี empirical check** (UMAP + K-means) และมี **UniSkill** เป็น fallback ถ้า z_t ดัน cluster ตาม morphology จริงๆ (แก้จากเดิมที่เขียนว่า HiLAM — ดูข้อ 5 Fallback)

### 3.3 Inverse State-Transition Model (ITM)

- **4 causal self-attention blocks, 16 heads**
- Input: [e_t, e_{t+1}] รวม 512 tokens (256 + 256 — ตัวเลขนี้ถูกแล้ว)
- Learned query token q_t (trained parameter ไม่ใช่ input)
- Output: **z_t ∈ ℝ^{64}**

> **แก้ไขสำคัญ**: เดิมเขียนว่า z_t ∈ ℝ^{512} — อ่าน LAC-WM Table 4 ผิด
> ตัวเลข "Latent Dimension = 512" ใน Table 4 คือ **hidden width ภายใน** ของ ITM/FTM ไม่ใช่ขนาดของ latent action
> LAC-WM §4.2 เขียนแยกไว้ว่า: *"Both models employ an action embedding dimension of 64"* → **z_t ∈ ℝ^{64}**
> (ข้อดี: latent เล็กลง 8 เท่า ช่วยเรื่อง compute บน RTX 2080 Ti ด้วย)

Causal mask: e_t เห็นแค่ตัวเอง, e_{t+1} เห็น e_t และตัวเอง ทำให้ ITM ถามว่า "มีอะไรเกิดขึ้นระหว่าง t กับ t+1?"

### 3.4 Forward State-Transition Model (FTM)

- **8 transformer blocks, 16 heads**
- แต่ละ block: self-attn(e_t) + self-attn(z_t) + cross-attn(e_t queries z_t)
- Input: [e_t, z_t] → Output: ê_{t+1} ∈ ℝ^{1408}

FTM ถาม "ถ้า state ตอนนี้เป็น e_t และ latent action เป็น z_t สถานะถัดไปจะเป็นยังไง?"

### 3.5 Motion Decoder

- cross-attn(z_t queries e_t) + MLP → â_t ∈ ℝ^{18}
- z_t เป็น query, e_t เป็น visual context (keys/values)
- Output: joint position targets 18 มิติ (6 ขา × 3 joints)
- **ไม่ใช้ตอนวัดผล Phase 1 แต่ต้องเก็บ weight ไว้** (แก้ 2026-07-19 จากเดิมที่เขียนว่า "ทิ้ง")
  - หน้าที่ใน Phase 1: anchor z_t ให้ ground กับ action จริง
  - หน้าที่ใน Phase 2: **มันคือสะพานเดียวจาก z_t กลับไปเป็นคำสั่งข้อต่อ** `policy → z_t → MD → 18 joint → หุ่น` ถ้าทิ้ง policy สั่งหุ่นไม่ได้เลย
  - **และนี่คือคำตอบคำถาม Ajan Blink** (*"แปลงเป็น latent แล้วแปลงกลับทำไม"*): policy เรียนใน latent เพราะ**ส่วนนั้น transfer ข้ามร่างได้** ส่วน MD ทำหน้าที่ถอดรหัส**เฉพาะร่าง** — การแปลงคือการ**แยกส่วนที่ transfer ได้ออกจากส่วนที่ transfer ไม่ได้**

### 3.6 Loss Functions

```
L_recon  = ||ê_{t+1} − e_{t+1}||²     ← self-supervised (embedding space)
L_motion = ||â_t − a_t||²              ← supervised (sim auto-logs a_t)
L_total  = λ_recon · L_recon + λ_motion · L_motion
```

L_recon คำนวณใน **embedding space** ไม่ใช่ pixel space ดังนั้นไม่ต้องการ pixel decoder

---

## 4. Data Setup

| รายการ | ค่า |
|---|---|
| Simulator | **CoppeliaSim v4.10** (แก้จาก IsaacSim 5.0 ที่เขียนผิด) |
| Robot | Stick Insect *Medauroidea extradentata* — 3 morphologies: **short 0.5× / medium 0.75× / long 1.0× (base)** — สร้างและ verify แล้ว (`sim/env/*.ttt`) |
| Action type | Joint position targets ℝ^{18} (6 legs × 3 joints) |
| Episode length | ~1,000 steps (~16s at 60Hz) |
| Episodes | ~100 per morphology per behavior |
| Behaviors | Walk / Turn / Stop |
| Camera | Fixed, side view, ~30° elevated — **ยังไม่มีใน scene ต้องสร้างเอง** (ดูข้อ 6) |
| Train morphologies | Short + Long leg |
| Transfer test | Medium leg (interpolation) |
| Data collection policy | **IK retargeting** — นิยาม behavior เป็น Cartesian foot trajectory แล้วใช้ `simIK` แก้ per morphology (ดูข้อ 4.1) |

Model stick insect ที่ใช้: ✅ ได้แล้วจาก repo `airl-insect-walking` ของ Ajan YuChen — migrate มาที่ `sim/` (ดู `sim/SOURCES.md`)

### 4.1 Data Collection Policy — ทำไมถึงเลือก IK Retargeting

**ปัญหา**: เราต้องการ walk/turn/stop × 3 morphologies แต่ของที่มีอยู่ทำไม่ได้:
- `ds_loopsm.csv` มีแค่ **67 rows = 1 gait cycle เดินหน้าอย่างเดียว** (loop rows 2–64)
- AIRL reward = `discriminator_logit + vx*100` → **forward velocity อย่างเดียว** ไม่มี turn/stop
- Expert data = **สัตว์ตัวเดียว, gait เดียว, trial เดียว** (Animal06) — 30 ไฟล์ใน `expert/trails/` คือ replay อันเดิม 30 รอบ ไม่ใช่ 30 recording

**ทำไมไม่ retrain AIRL ต่อ morphology**: normalization bounds ใน `normalized_env*.py` เป็น **ค่า literal ที่วัดมือมาจากร่างเดิม** ไม่มีอะไร parameterize ตามความยาวขา ถ้าขาสั้นลง 50% ทุก bound ผิดหมด (joint range, foot z, force, body height, standing pose) และ **ไม่มีการ clip** → ผิดแบบเงียบๆ + ต้องเทรน ~1 วัน/run + expert data ใช้ไม่ได้กับร่างที่ scale แล้ว

**วิธีที่เลือก — IK Retargeting**:
1. นิยาม behavior เป็น **Cartesian foot trajectory** (walk = เดินหน้า, turn = ซ้าย/ขวาไม่เท่ากัน, stop = ยืนนิ่ง)
2. ใช้ `simIK` (มีใน CoppeliaSim อยู่แล้ว) แก้ IK → ได้ joint angles ต่อ morphology
3. ได้ **a_t ต่างกันต่อ morphology** แต่ behavior เทียบกันได้ ไม่ต้องเทรนเลย

**ทำไม a_t ต้องต่างกันต่อ morphology (สำคัญมาก)**: Motion Decoder คือ `MD(x_t, z_t) → â_t` — มันดู visual context `x_t` ด้วย
- ถ้าส่ง **command เดียวกัน** ให้ทุกร่าง → `a_t` เหมือนกันหมด → `L_motion` บังคับให้ `z_t` morphology-agnostic **แบบ trivial** และ MD ไม่ต้องใช้ `x_t` เลย → reviewer บอกได้ทันทีว่า "ก็แน่ล่ะ คุณส่ง action เดียวกันให้ทุกร่าง" → **circular**
- ถ้า `a_t` ต่างกันต่อร่าง → MD **ต้อง** ใช้ `x_t` เพื่อรู้ว่า "นี่ร่างไหน" → `z_t` ที่เก็บแต่ behavior จึงเป็นผลลัพธ์ที่ **ได้มาจริง** ไม่ใช่ของแถม

---

## 5. Execution Plan (Milestone-based)

### Step -1 — Morphology Gap Check
ส่ง joint command เดียวกันไปให้ขาสั้นและขายาว ถ้าได้ behavior ต่างกันจริง (เช่น ขายาวลาก) → morphology gap มีอยู่จริง → ดำเนินการต่อ

### Step 0 — Visual Encoder Sanity Check
รัน V-JEPA2 บน frame จากทั้ง 3 morphologies × 3 behaviors → ดู UMAP ว่า e_t มี structure ที่ดีหรือเปล่า behavior ต้องแยกได้บ้าง morphology ต้องไม่ dominate

### Step 1 — Train Phase 1 Pipeline
เทรน ITM + FTM + Motion Decoder บน short + long leg ดู L_recon และ L_motion converge ทั้งคู่

### Step 1.5 — Latent Space Validation
เก็บ z_t จากทุก morphology × behavior → UMAP ดูว่า cluster ตาม behavior (ผ่าน) ไม่ใช่ morphology (ล้มเหลว)

**Evaluation metrics:**
- UMAP colored by behavior → 3 clusters ชัดเจน (primary visualization)
- UMAP colored by morphology → ไม่มี separation
- K-means (K=3) → cluster labels match behavior labels (quantitative check)

### Step 2 — Transfer to Unseen Morphology
Fine-tune ITM + FTM บน N medium leg episodes ด้วย **LoRA rank 2**

| Condition | ความหมาย |
|---|---|
| Pretrained FTM + N episodes | ใช้ World Model ที่เทรนแล้ว |
| Scratch FTM + N episodes | baseline — เทรนใหม่ตั้งแต่ต้น |

Vary N = 5 / 10 / 20 / 50 / 100 episodes

**Metric หลัก:** training time reduction — pretrained ต้องถึง L_recon เดิมด้วย episodes น้อยกว่า scratch อย่างชัดเจน (Ajan Go: "นี่คือตัววัดผลหลัก")

### Fallback (ถ้า Step 1.5 ล้มเหลว) — **แก้แล้ว: UniSkill ไม่ใช่ HiLAM**

~~เดิม: ใช้ HiLAM ทำ dynamic chunking → z^h~~ ❌ **HiLAM เป็น fallback ที่ผิด**

อ่าน paper จริงแล้ว (`doc/2603.05815v1`) พบว่า HiLAM แก้ปัญหา **temporal abstraction** (z_t มองแค่ช่วงสั้นๆ ไม่เห็น structure ระยะยาว) — **ไม่ใช่ปัญหา embodiment invariance**:
- chunking mechanism ของมันดู feature dissimilarity **ระหว่าง token ที่ติดกันใน video เดียว** เท่านั้น
- **ไม่มี objective ใดๆ ที่ align ข้าม embodiment เลย** ไม่มีการแยก nuisance (รูปร่าง) ออกจาก behavior
- ถ้า z_t encode morphology อยู่แล้ว → chunking แค่ **pool feature เดิม** → จะได้ skill hierarchy แยกต่อร่าง = **ยิ่งตอกย้ำ morphology clustering** ไม่ได้แก้
- การทดลองเป็น LIBERO manipulation 100% ไม่มี locomotion เลย และไม่มี code ปล่อย

✅ **ตัวที่ถูกคือ UniSkill** (Kim et al. 2025, CoRL) — *"Imitating Human Videos via Cross-Embodiment Skill Representations"* — แก้ **cross-embodiment** โดยตรง ซึ่งคือ failure mode ที่เรากลัวจริงๆ

> จุดสังเกตสำคัญ: **HiLAM เอา IDM/FDM ของ UniSkill มาใช้เป็น frozen submodule** — คุณสมบัติ cross-embodiment ที่ HiLAM มี จริงๆ มาจาก UniSkill ส่วน contribution ของ HiLAM เอง (hierarchical chunking) เป็นคนละเรื่องกัน
> → ถ้า z_t cluster ตาม morphology ต้องไปที่ paper ที่แก้เรื่องนั้นโดยตรง ไม่ใช่ paper ที่สร้างทับมันอีกที

ตัวสำรองอีกตัวที่น่าสนใจ: **DiLA** (Zhang et al. 2026) — disentangle content/structure เพื่อกันไม่ให้ feature รูปร่างเข้าไปปนใน behavior latent

---

## 5.5 🔴 Confound ที่อันตรายที่สุด — Render Style ครอบงำ e_t

**สิ่งที่เจอ** (`scripts/umap_domain_check.py`, บันทึกใน `PROGRESS.md §5`): วิดีโอ 3 อันที่ **behavior เหมือนกัน** (เดินหน้าเหมือนกันหมด) แต่ render คนละแบบ (พื้นขาว / IsaacSim grid / MuJoCo checkerboard) → whole-frame `e_t` แยกเป็น **3 cluster ที่ไม่ทับกันเลย**

**แปลว่า**: raw frozen V-JEPA2 `e_t` ตอนนี้ sensitive กับ **สไตล์การ render** (พื้นหลัง/แสง/engine) มากกว่า **behavior**
(V-JEPA2 paper ช่วยอะไรไม่ได้ตรงนี้ — VideoMix22M **ไม่มี simulated data เลยสักนิด** และ paper ไม่เคยศึกษาเรื่อง rendering domain gap → ผลของเราไม่ขัดกับ paper แต่ paper ก็อธิบายมันไม่ได้)

**ทำไมอันตราย**: ถ้ากล้อง/แสง/พื้นหลัง ต่างกันแม้แต่นิดเดียวระหว่าง session ที่ถ่ายแต่ละ morphology → **Step 1.5 จะวัด "คลิปนี้ถ่ายจาก session ไหน" ไม่ใช่ morphology vs behavior** ผลจะออกมาสวยงามและ**ไม่มีความหมาย** — และมองไม่เห็นเลยถ้าไม่ได้คุมไว้ตั้งแต่แรก

**วิธีป้องกัน — บังคับ ทำตอนเก็บข้อมูล:**
1. **ล็อค render environment**: กล้อง (ตำแหน่ง/มุม), แสง, พื้นหลัง ต้อง**เหมือนกันเป๊ะ**ทุก morphology และทุก behavior เปลี่ยนแค่ **ขาหุ่น** กับ **การเคลื่อนไหว** เท่านั้น ห้ามเปลี่ยนอย่างอื่น
2. **เลือกพื้นหลังให้ถูก** — เลี่ยงทั้ง 2 สุดขั้วที่ทดลองแล้วพัง: **ห้าม checkerboard** (aliasing → motion ปลอม) และ **ห้ามพื้นเรียบ/ว่างเปล่า** (ViT register-token noise — patch ว่างแกว่งมากที่สุด) → ใช้พื้นผิว **matte, texture อ่อนๆ, ไม่ซ้ำลาย**
3. **Gate ก่อนเข้า Step 1.5**: encode frame จากแต่ละ morphology session แล้วรัน domain-UMAP → cluster **ต้องทับกันแล้ว** ถ้ายังแยกตาม session = ยังคุมไม่ได้ = ข้อมูลใช้ไม่ได้
4. cross-augmentation ออกแบบมาลด nuisance แบบนี้อยู่แล้ว — **แต่ไม่ใช่ตัวแทนของการคุม environment** เพราะ body shape เป็น real content ที่ crop/color/flip ทำลายไม่ได้

> ยืนยันซ้ำจาก `deep_research.md` (เอกสารเก่า): *"Latent space analysis must show locomotion-relevant structure, not visual artifacts"* — เขียนไว้ตอนยังทำ direction เก่า แต่ยังใช้กับตอนนี้ได้เป๊ะ

---

## 6. งานที่เกี่ยวข้องและช่องว่าง

| | Latent Action WM | Locomotion | Cross-Morphology |
|---|---|---|---|
| LAC-WM — Stanford/Meta (ICML 2026) | ✓ | ✗ manipulation | ✓ |
| RLWAV — LAAS-CNRS (2024) | ✗ classifier reward | ✓ | ✗ single robot |
| SLoMo — CMU (RA-L 2023) | ✗ explicit trajectory | ✓ | ✗ single robot |
| **งานเรา** | **✓** | **✓** | **✓** |

LAC-WM คืองานหลักที่เรา adapt มา แต่ LAC-WM ทำ manipulation เราเป็น**รายแรก**ที่นำ pipeline นี้มาใช้กับ locomotion

หมายเหตุ: LAC-WM ถูก ICLR 2026 reject (weak evaluation: 1 task, 1 baseline) แต่ได้รับ ICML 2026 ข้อวิจารณ์นี้บอกว่าเราต้องการ **≥2 baselines และ ≥3 behaviors** เพื่อ evaluation ที่แข็งแกร่งกว่า

---

## 7. สรุปประโยคเดียวสำหรับบอกอาจารย์

> "เราจะเทรน Latent Action World Model บน simulation ของ stick insect 3 รูปแบบ (ขาสั้น/กลาง/ยาว) โดยใช้ V-JEPA2 เป็น visual encoder แบบ frozen และ LAC-WM pipeline (ITM + FTM + Motion Decoder) เพื่อสกัด z_t ที่ cluster ตาม behavior ไม่ใช่ morphology และพิสูจน์ว่า World Model ที่เทรนจากขาสั้น+ยาวช่วยให้ขากลางเรียนรู้ได้เร็วขึ้นอย่างชัดเจน"

---

## 8. คำถามที่อาจารย์น่าจะถาม

**"ทำไม V-JEPA2 ที่ pretrain บน internet video ถึงใช้กับ simulation locomotion ได้?"**

V-JEPA2 เรียนรู้ motion-relevant features จาก video ทั่วไป เช่น object movement, temporal change, spatial structure สิ่งเหล่านี้ไม่ specific กับ domain ใดๆ และ locomotion ก็ต้องการ feature เหล่านี้ — ขาเคลื่อนไหวยังไง body เอียงยังไง Step 0 จะเป็น empirical check ว่า feature จาก frozen V-JEPA2 มี locomotion signal จริงหรือเปล่าก่อนเดินหน้าต่อ

**"ทำไม z_t ถึงไม่ encode morphology?"**

เหตุผลตรงจาก LAC-WM paper: cross-augmentation บังคับให้ ITM cheat ไม่ได้ด้วยการยัด x_{t+1} (raw future embedding) เข้าไปใน z_t ตรงๆ เพราะ ITM เห็น x_{t+1} จาก aug1 แต่ FTM ต้อง predict x_{t+1} จาก aug2 (คนละ augmentation) ถ้า z_t แค่ copy ข้อมูลดิบมาจะ predict ผิด — z_t จึงถูกบีบให้ capture เฉพาะสิ่งที่ generalize ข้าม augmentation ได้ ซึ่งควรเป็น motion/action มากกว่า raw appearance

**แต่ต้องระวัง**: นี่ไม่ได้การันตี 100% ว่า z_t จะไม่ encode morphology เพราะ body shape เป็น real content ที่ crop/color/flip ทำลายไม่ได้ (ต่างจาก texture/color ที่เป็นแค่ nuisance) ดังนั้นต้องพิสูจน์ด้วย empirical evidence ใน **Step 1.5** (UMAP + K-means บน z_t ข้าม morphology) จริงๆ ถ้าล้มเหลว → fallback ไปใช้ **UniSkill**

**"ทำไมต้อง LoRA ใน Step 2?"**

LoRA rank 2 ให้เพิ่มน้อย parameters มากสำหรับ fine-tune บน medium leg (~0.1% ของ total params) ทำให้โมเดล adapt ได้โดยไม่ทำลาย latent structure ที่เรียนรู้มาจาก short+long leg ถ้า fine-tune แบบ full model เสี่ยงที่จะ overwrite knowledge ที่ pretrain ไว้

**"Baseline คืออะไร?"**

เทรน FTM จาก scratch บน medium leg N episodes เดียวกัน เปรียบเทียบว่าต้องการกี่ episodes ถึงจะได้ L_recon เท่ากัน ถ้า pretrained ใช้ episodes น้อยกว่าอย่างชัดเจน → World Model มีประโยชน์จริง

**"ถ้า z_t cluster ตาม morphology ทำยังไง?"**

ใช้ **UniSkill** (Kim et al. 2025, CoRL) เป็น fallback — เป็น paper ที่แก้ **cross-embodiment skill representation** โดยตรง ตรงกับ failure mode ที่เจอพอดี

(เดิมเคยตอบว่าใช้ HiLAM — **ผิด** HiLAM แก้ปัญหา temporal abstraction ไม่ใช่ embodiment invariance ถ้า z_t encode morphology อยู่แล้ว การ chunk มันเป็น skill จะยิ่งตอกย้ำ cluster เดิม ไม่ได้แก้ — ดูข้อ 5 Fallback)

**"ทำไมต้องมี latent action? ในเมื่อทุก morphology ใช้ joint command 18 มิติเหมือนกันหมด?"** ⚠️ **คำถามที่อันตรายที่สุด — Ajan Blink ถามตั้งแต่ Week 4 ยังไม่เคยตอบ**

คำถามเต็มของ Ajan Blink: *"ถ้าสุดท้าย policy กับหุ่นก็ต้องใช้ joint command อยู่ดี แล้วจะแปลงเป็น latent/frame space ทำไม แถมต้องมี converter แปลงกลับอีก?"*

ทำไมมันแรง: LAC-WM มี embodiment ที่ **action space ต่างกันจริงๆ** (Franka EE 10 มิติ / humanoid 2 แขน 20 มิติ / มือคน **138 มิติ** / BFA 25 มิติ) — latent action ของเขาจำเป็นเพราะต้องรวม action space ที่คนละขนาดกันให้เป็นภาษาเดียว **แต่ของเรา 3 ร่างใช้ 18 มิติเหมือนกันเป๊ะ** เหตุผลนั้นหายไปเลย

**คำตอบที่ควรใช้ — เปลี่ยน framing**: เหตุผลของเราไม่ใช่ *action-space heterogeneity* (เราไม่มี) แต่เป็น **dynamics heterogeneity** — command เดียวกันให้ผลการเคลื่อนที่ต่างกันชัดเจนตามความยาวขา **ซึ่ง Step -1 พิสูจน์ไปแล้ว** (3.49 m vs 4.77 m, swing clearance 0.13–0.16 สม่ำเสมอ vs 0.05–0.38 กระจาย)

→ หน้าที่ของ latent action ที่นี่คือรวม **effective dynamics mapping** จาก command เดียวกัน → การเคลื่อนที่/ภาพที่ต่างกันข้ามความยาวขา โดยมี motion-decoding loss คอย ground ไม่ให้มัน degenerate เป็น identity function

**การทดลองที่ต้องทำเพื่อตอบให้ได้จริง**: **latent-conditioned FDM vs raw-joint-conditioned FDM (shared encoder)** — ถ้า latent ชนะ = ตอบ Ajan Blink ได้ด้วยหลักฐาน ถ้าแพ้ = thesis กลวง **ต้องรู้ตั้งแต่เดือนแรก ไม่ใช่เดือนที่ 3**

> **อัปเดต (2026-07-25) — ยังเก็บการทดลองนี้ไว้ แต่ปรับ framing: ห้ามตัดสินด้วยการเทียบ loss ตรงๆ และมันไม่ใช่ตัวตัดสินเดียว**
> การเทียบ loss ของ `F(e_t, z_t)` กับ `F(e_t, a_t)` **ไม่แฟร์**: `z_t` อนุมานจาก `(e_t, e_{t+1})` แปลว่ามันเห็นเฟรมอนาคตแล้ว
> และเป็น 64 มิติ เทียบกับ `a_t` 18 มิติ — ได้เปรียบข้อมูล/ความจุฟรีๆ ตำแหน่งปัจจุบัน (ตรงกับ proposal §3.6.3 และสไลด์):
> - **หลักฐานหลัก/ตัวตัดสินจริง = two-sided probe** — `z_t` ทำให้ behaviour transfer ดีขึ้น **และ** morphology decode ได้น้อยลง
>   พร้อมกัน เทียบกับ `e_t` ดิบ (ใช้แค่วิดีโอของร่างที่ hold out)
> - **คุณค่าเหนือคำสั่งดิบ (value-over-raw)** วัดด้วย (1) **adaptation efficiency** (จำนวน episode ถึง target error:
>   pretrained-on-`z` vs pretrained-on-raw vs from-scratch) และ (2) **availability argument** (จะได้ `a_t` ที่ถูกของร่างใหม่
>   ต้องรู้ kinematics ผ่าน IK = privileged ส่วน `z_t` มาจากภาพ เพราะงั้นแค่**เสมอ**ก็ชนะแล้ว)
> - การเทียบ `F(e_t,z_t)` vs `F(e_t,a_t)` vs `F(e_t,0)` ยังอยู่เป็น **Step E** (รันเร็ว) ในฐานะ *diagnostic* (มี observation-only
>   control `F(e_t,0)` แยกส่วนที่ action ช่วย) — ไม่ใช่ "ถ้า latent แพ้ = thesis กลวง" ตัวที่ load-bearing คือ probe

---

## Phase 2 — Deployment (closed-loop) — บันทึก 2026-07-25

Phase 2 อยู่นอก scope thesis แต่บันทึกไว้ให้ตรงกับ direction_plan.md (มีรายละเอียดเต็มที่นั่น) และ deck Slide 24 + `report/pipeline_diagram.tex`

- **Deploy เป็นลูปปิดอันเดียว ไม่ใช่ open-loop replay** — เล่นซ้ำลำดับ `z` ของ demo บนร่างที่จังหวะต่างกันจะ **phase หลุด** (`z_t` เป็น transition เฉพาะที่; demo swing แต่ร่างจริงยัง stance → คู่ `(e_t, z_t)` ที่ decoder ไม่เคยเห็น = OOD) ลูปปิดอ่านสภาพจริงทุก step เลยไม่หลุด
- **reward เป็นแค่ objective ของ selector — มีหรือไม่มีก็ได้** เก็บหลายไอเดียไว้:
  1. **match a demo ใน z-space** (ไม่ต้อง reward) — planning; match ใน `z` ไม่ใช่ `e` ดิบ (เพราะ `e` มี body shape ปน)
  2. **maximize reward** (RL/Dreamer) — ได้ควบคุมอัตโนมัติ แต่ต้องเพิ่ม reward model + Critic
  3. **behaviour-matching RL (ตัวที่ชอบตอนนี้)** — ป้อน `z_target` → ทำจริง → เอา transition ที่ได้เข้า ITM → `z_achieved` →
     เทรน decoder ด้วย RL, reward = `−‖z_achieved − z_target‖²` **ข้อดี: อยู่ใน z-space (ไม่ผูกรูปร่าง) + ไม่ต้องมี label `a_t`**
     **ข้อควรรู้: เป็น RL ไม่ใช่ backprop** เพราะ `a_t → e_{t+1}` เป็นฟิสิกส์จริง diff ไม่ได้ (FTM แทนไม่ได้ เพราะมันรับ `z` ไม่ใช่ `a`)
     → ใช้วิธีประหยัด sample (CEM / off-policy SAC) **ไม่ใช่ PPO**
- decoder ยังต้อง **calibrate offline นิดหน่อย** ด้วยข้อมูล `(e_t, a_t)` ของร่างใหม่ (coverage สำคัญกว่า duration; ไม่ต้อง sync กับ demo)

---

## 9. สิ่งที่ยังต้องทำ / open questions

| รายการ | สถานะ |
|---|---|
| ~~Stick insect model~~ | ✅ **เสร็จแล้ว** — ได้จาก repo `airl-insect-walking` ของ Ajan YuChen, migrate มาที่ `sim/` แล้ว |
| ~~Data collection policy~~ | ✅ **ตัดสินใจแล้ว: IK retargeting** (ดูข้อ 4.1) |
| 🔴 **Camera ใน CoppeliaSim scene** | **ยังไม่มี** — scene เป็น state-only ทั้งหมด ไม่มี vision sensor เลย **นี่คือ blocker อันดับ 1** ทุกอย่างตั้งแต่ Step 0 เป็นต้นไปติดตรงนี้ |
| 🔴 **Turn / Stop behavior** | **ยังไม่มี** — gait CSV มีแค่ walk (67 rows), AIRL reward เป็น forward-velocity อย่างเดียว → ต้องสร้างด้วย IK retargeting |
| 🔴 **Motivation: ทำไมต้องมี latent action?** | **ยังไม่ตอบ** — Ajan Blink ถามตั้งแต่ Week 4 ยังไม่มีคำตอบ (ดูข้อ 10) |
| λ_recon, λ_motion weights | เริ่มจาก equal แล้ว ablate — **หมายเหตุ: paper ไม่ได้บอกค่า λ ไว้เลย** (รวมถึง learning rate / optimizer ด้วย) |
| k = 64 (LAC-WM §4.2) หรือ ablate | แก้จากเดิมที่เขียนว่า 512 (อ่าน Table 4 ผิด — 512 คือ hidden width ไม่ใช่ latent action) |
| LAC-WM source code | ยังไม่มี public code (accepted ICML 2026) |
| GPU สำหรับ training | ใช้เครื่องแล็บ (RTX 2080 Ti 11GB) — **ต้องระวัง**: LAC-WM ใช้ 64× H200 นาน 4 วัน (≈256 H200-GPU-days), batch 512 ต้องลด scale ลงมากและใช้ gradient accumulation + fp16 |
