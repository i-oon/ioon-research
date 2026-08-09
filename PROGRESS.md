# Progress Log — Cross-Morphology Locomotion Project

## สารบัญ

1. [Environment Setup](#1-environment-setup)
2. [ปัญหา Temporal Mixing และวิธีแก้](#2-ปัญหา-temporal-mixing-และวิธีแก้)
3. [ตรวจสอบ Cross-Augmentation กับ Paper ต้นฉบับ](#3-ตรวจสอบ-cross-augmentation-กับ-paper-ต้นฉบับ)
4. [Step 0 — ลอง Per-Patch Temporal Heatmap (3 backgrounds)](#4-step-0--ลอง-per-patch-temporal-heatmap-3-backgrounds)
5. [Whole-Frame UMAP — ข้อค้นพบสำคัญ](#5-whole-frame-umap--ข้อค้นพบสำคัญ)
6. [ผลกระทบต่อการเก็บข้อมูลจริง](#6-ผลกระทบต่อการเก็บข้อมูลจริง)
7. [CoppeliaSim Setup + Migration จาก airl-insect-walking](#7-coppeliasim-setup--migration-จาก-airl-insect-walking)
8. [สร้าง Morphology Variants (short/medium/long)](#8-สร้าง-morphology-variants-shortmediumlong)
9. [Step -1 — Morphology Gap Check: PASS](#9-step--1--morphology-gap-check-pass)
10. [Full Project Audit (6 agents) — ข้อค้นพบที่เปลี่ยนทิศทาง](#10-full-project-audit-6-agents--ข้อค้นพบที่เปลี่ยนทิศทาง)
11. [สถานะปัจจุบัน / ต้องทำต่อ](#11-สถานะปัจจุบัน--ต้องทำต่อ)
12. [อัปเดตใหญ่ 2026-08 — Cross-Embodiment pivot](#12--อัปเดตใหญ่-2026-08--cross-embodiment-pivot--แผน-staged)
13. [AMP — เทรน controller ต่อร่าง เพื่อสร้าง behavior dataset](#13--amp--เทรน-controller-ต่อร่าง-เพื่อสร้าง-behavior-dataset-2026-08-in-progress)
14. [อัปเดต 2026-08-06 — กลับมาใช้ IK forward-only + 4-leg preview](#14--อัปเดต-2026-08-06--กลับมาใช้-ik-forward-only--4-leg-preview)
15. [อัปเดต 2026-08-06 — ขยาย render-lock check + validate train(long+short)→test(medium)](#15--อัปเดต-2026-08-06--ขยาย-render-lock-check-เป็น-6-episodes--validate-trainlongshorttestmedium)
16. [อัปเดต 2026-08-07 — เทรน world model + เจอ data bug (framing)](#16--อัปเดต-2026-08-07--เทรน-world-model-จริง-3-รอบ--เจอ-data-bug-ใหญ่-framing)

---

## 1. Environment Setup

- สร้าง `.venv` (Python venv, `--system-site-packages` เพื่อใช้ torch+CUDA ที่มีอยู่แล้ว)
- ติดตั้ง `transformers`, `huggingface_hub`, `torchvision`, `opencv`, `scipy`, `umap-learn`
- โหลดโมเดล `facebook/vjepa2-vitg-fpc64-256` (ViT-g/16, 1B params) รันบน GPU (RTX 2080 Ti, 11GB) ได้สำเร็จ ไม่ gated ไม่ต้อง login

---

## 2. ปัญหา Temporal Mixing และวิธีแก้

**พบว่า**: checkpoint ที่โหลดมาเป็น **video encoder** จริง (`frames_per_clip=64`, `tubelet_size=2`) ไม่ใช่ per-frame image encoder ตามที่ direction_plan.md สมมติไว้ตอนแรก ถ้าป้อน clip 64 เฟรมจริงเข้าไป แต่ละ frame จะเห็นข้อมูลจาก frame อื่นในอนาคตผ่าน self-attention แบบ bidirectional → `e_t` ไม่บริสุทธิ์ ไม่ independent ต่อ timestep

**วิธีแก้**: ป้อนแต่ละ frame ซ้ำ 2 ครั้ง (duplicate) เข้า tubelet แทนการป้อน clip จริง (`scripts/vjepa2_encoder.py` → `VJEPA2FrameEncoder`)

**ยืนยันด้วยการทดลอง** (`scripts/test_vjepa2_frame_isolation.py`):
- Duplicated-frame encoding เป็น deterministic 100% (diff = 0.000000)
- Frame เดียวกัน เข้ารหัสเดี่ยว vs. ฝังอยู่ใน clip จริง 64 เฟรม → cosine similarity เฉลี่ยแค่ 0.52 (ควรจะ = 1.0 ถ้าไม่มี contamination) → ยืนยันว่า clip-mode encoding รั่วข้อมูลข้าม timestep จริง

---

## 3. ตรวจสอบ Cross-Augmentation กับ Paper ต้นฉบับ

อ่าน LAC-WM paper ตัวจริง (`doc/LATENT ACTION ROBOT FOUNDATION WORLD MODELS...pdf`) เพื่อยืนยัน mechanism:

- Encoder ใช้แบบ **image mode** จริง — paper เขียนว่า "V-JEPA2 RGB tokenizer **for image encoding**" และ output คือ 256 tokens/frame (ตรงกับที่เราคาดไว้)
- Cross-augmentation: augment frame pair `(O_t, O_{t+1})` สองครั้งแยกกัน (A1, A2) → ITM ใช้ pair 1, FTM ใช้ pair 2
- **เหตุผลที่แท้จริง** (ต่างจาก note เดิมเล็กน้อย): ป้องกัน ITM แอบยัด `x_{t+1}` (raw future embedding) เข้า `z_t` โดยตรง ไม่ใช่แค่เรื่อง texture/color เฉยๆ
- แก้ไข `direction_plan.md` และ `note.md` (รวม Q&A สำหรับอาจารย์) ให้ตรงกับ paper — พร้อมเพิ่มหมายเหตุว่า cross-augmentation **ไม่การันตี** ว่า z_t จะไม่ encode morphology (เพราะ body shape ไม่ใช่ nuisance ที่ augmentation ทำลายได้) → นี่คือเหตุผลที่ Step 1.5 ยังต้อง empirical check อยู่ดี

---

## 4. Step 0 — ลอง Per-Patch Temporal Heatmap (3 backgrounds)

**แนวคิด**: เทียบ embedding ของแต่ละ patch (16×16 grid) ระหว่าง frame t กับ t+1 คาดหวังว่า background (นิ่ง) → similarity สูง, ขาหุ่นยนต์ (เคลื่อนที่) → similarity ต่ำ — ใช้ตอบคำถาม Ajan Go เรื่อง "High Relation vs Low Relation"

ทดสอบกับ background 3 แบบ ทุกแบบมีปัญหา:

| Background | วิดีโอ | ผลลัพธ์ | ปัญหา |
|---|---|---|---|
| Checkerboard | `forward_walk.mp4` | correlation ติดลบชัดเจน (r=-0.16, p=7.6e-24) | pixel เปลี่ยนเพราะ aliasing ไม่ใช่ motion จริง |
| พื้นขาวล้วน | `removebg_forward_walk.mp4` | correlation ติดลบชัดเจน (r=-0.20, p=4.7e-37) | patch ที่ไม่มีข้อมูลเลย embedding กลับ "แกว่ง" มากที่สุด (ปรากฏการณ์ ViT ที่รู้จัก — blank patch ถูกใช้เป็นพื้นที่คำนวณภายใน) |
| Grid บาง | `play-step-0_realtime.mp4` | ไม่มี correlation ชัดเจน (r=-0.006, p=0.70) | ตัด bias ลบออกได้ แต่ noise floor เดิมยังอยู่ ไม่ได้ positive signal |

**สรุป**: ปัญหาไม่ได้อยู่ที่ background — เป็นข้อจำกัดของวิธี **per-patch comparison เอง** (ดูทีละ patch เล็กๆ) ไม่ว่าจะลอง background แบบไหนก็ไม่ให้สัญญาณที่เชื่อถือได้

Scripts: `scripts/temporal_similarity_heatmap.py`, `scripts/temporal_similarity_quantified.py`, `scripts/temporal_similarity_correlation.py`

---

## 5. Whole-Frame UMAP — ข้อค้นพบสำคัญ

เปลี่ยนวิธี: แทนที่จะดูทีละ patch เอา 256 patch embeddings มา **average รวมเป็นเวกเตอร์เดียวต่อ frame** แล้วทำ UMAP เทียบ 3 วิดีโอที่ "เดินเหมือนกัน" (behavior เดียวกัน) แต่ render มาคนละแบบ:

- `removebg_forward_walk.mp4` (พื้นขาว)
- `play-step-0_realtime.mp4` (IsaacSim, grid บาง)
- `light_ood_mujoco.mp4` (MuJoCo, checkerboard)

**ผลลัพธ์**: ทั้ง 3 domain แยกเป็น **3 กลุ่มที่ไม่ทับกันเลย** ทั้งที่ behavior เดินเหมือนกันหมด (ดู `domain_umap.png`)

**การตีความ**: encoder ไม่ได้ "ห่วย" — ทำงานสะอาดมากในระดับ whole-frame เพียงแต่ตอนนี้ raw `e_t` (ไม่ผ่าน cross-augmentation) ยัง sensitive กับ **สไตล์การ render** (แสง/พื้นหลัง/engine) มากกว่า behavior จริง ซึ่งเป็นเรื่องปกติของ pretrained encoder — และเป็นเหตุผลที่ยืนยันว่าทำไม ITM + cross-augmentation (ข้อ 3) ถึงจำเป็น เพื่อบีบให้ z_t ทิ้ง style แล้วเก็บแต่ motion

Script: `scripts/umap_domain_check.py`

---

## 6. ผลกระทบต่อการเก็บข้อมูลจริง

ตอนถ่ายข้อมูลจริง 3 morphology (สั้น/กลาง/ยาว) **กล้อง, แสง, พื้นหลัง ต้องเหมือนกันเป๊ะทุก session** ไม่งั้นมีความเสี่ยงว่าผล Step 1.5 จะไปเจอว่า `z_t` แยกกลุ่มตาม "session ที่ถ่าย" (rendering style) แทนที่จะแยกตาม morphology หรือ behavior จริง — เป็น confound ที่ควบคุมได้ตั้งแต่ตอนเก็บข้อมูล

---

## 7. CoppeliaSim Setup + Migration จาก airl-insect-walking

**พบว่า**: `airl-insect-walking/` (repo ของ Ajan YuChen ที่มีอยู่แล้วใน `/home/aria/ioon-research/`) มี stick insect model จริง (*Medauroidea extradentata*) รันบน **CoppeliaSim v4.10** — repo นี้มีทั้ง base model, trained AIRL/PPO policy, real expert motion-capture data, และ cross-dynamic transfer experiment (Stick Insect → RedMirror) ที่เป็น pattern เดียวกับที่เราต้องการทำกับ short/medium leg

**ติดตั้ง**:
- ดาวน์โหลด CoppeliaSim v4.10.0 Edu จาก `downloads.coppeliarobotics.com` → `/home/aria/CoppeliaSim`
- ติดตั้ง Python connector (`coppeliasim_zmqremoteapi_client`, `pyzmq`, `msgpack`, `cbor2`, `pandas`) ใน `.venv`
- **สำคัญ**: ต้อง activate venv **ก่อน** รัน `coppeliaSim.sh` เสมอ เพราะ ZMQ remote API server เรียก python3 subprocess ของตัวเอง ถ้าไม่ activate venv จะหา `zmq`/`cbor2` ไม่เจอ แล้ว server จะ fail เงียบๆ
- **Headless mode (`-h`/`-H`) ใช้ไม่ได้ตอนนี้** — เปิดได้ ZMQ port เปิดสั้นๆ แล้ว segfault ระหว่าง cleanup Python subprocess (bug ที่ไม่ root-cause แล้ว) → **ใช้ GUI mode แทนไปก่อน** เรื่องนี้สำคัญถ้าจะรัน data collection แบบ unattended/บน server ในอนาคต

**Migrate มาไว้ที่ `sim/`** (copy ไม่ symlink กันพัง ถ้า airl-insect-walking เปลี่ยน):
- `env/medauroidea_stick_insect.ttt` — base model
- `env/main_script.py` — gait replay script ที่ scene ต้องใช้ (แก้ hardcoded path `/home/yuchen/...` → local path แล้ว)
- `env/ds_loopsm.csv` — gait trajectory data ที่ main_script.py อ่าน
- `coppeliasim_env.py` — `CoppeliaSimEnv` class (Gym-style wrapper, normalize/denormalize obs-action, `reset()`/`step()`)

รายละเอียดเต็มดูที่ `sim/SOURCES.md`

---

## 8. สร้าง Morphology Variants (short/medium/long)

**เป้าหมาย**: สร้าง short-leg และ medium-leg variant จาก base model (= "long" leg ตาม plan) เพื่อใช้ทำ Step -1 และ Step 1 data collection

**โครงสร้างขา** (ต่อ 1 ขา, สำรวจผ่าน object hierarchy): `m1 (joint/ThC) → coxa (segment) → m2 (joint) → femur (segment) → m3 (joint/FTi) → tibia (segment) → forceSensor → foot`

**พบ bug ระหว่างทำ**: segment แต่ละชิ้น origin อยู่ที่ "กึ่งกลาง" ไม่ใช่ปลายด้านใดด้านหนึ่ง ดังนั้นความยาวของ segment ถูกแบ่งเป็น 2 offset เท่าๆ กัน (parent-joint→center, center→child-joint) — ตอนแรก scale แค่ offset ที่สอง (child reposition) ทำให้ scale factor ที่ตั้งใจ 0.7 ออกมาจริงเป็น 0.85 (เฉลี่ยของ 1.0 กับ 0.7) ทุกขา แก้โดย scale ทั้ง 2 offset ของทุก segment (ดู `sim/make_leg_morphology.py`)

**พบ typo ใน base scene**: `/tibia_HR` จริงๆ ชื่อ `/tibial_HR` (มี l เกิน) — handle ใน script แล้ว

**ผลลัพธ์สุดท้าย** (หลังปรับจาก 0.7/0.85 เป็น 0.5/0.75 ตามที่เห็นว่า noticeable กว่า):

| Variant | Scale factor | ยืนยันด้วย |
|---|---|---|
| long (base) | 1.0 | unmodified |
| medium | 0.75 | reload + วัด reach ratio ทั้ง 6 ขา = 0.75 พอดี |
| short | 0.5 | reload + วัด reach ratio ทั้ง 6 ขา = 0.5 พอดี |

**หมายเหตุ**: script (`sim/make_leg_morphology.py`) ต้องมี CoppeliaSim instance รันอยู่จริง (ต่อผ่าน ZMQ port 23000) ถึงจะทำงานได้ ไม่ใช่ standalone file transform — เจอ confusion เรื่องนี้ตอนแรกเพราะรัน script ตอนไม่มี CoppeliaSim รันอยู่ ทำให้ terminal ปริ้นท์ผลลัพธ์ที่ถูกต้องแต่ไฟล์ไม่ถูกเขียนจริง (ยัง unclear ว่า mismatch เกิดจากอะไรกันแน่ แต่ regenerate ใหม่ตอนมี sim รันแล้ว verify ผ่าน)

---

## 9. Step -1 — Morphology Gap Check: PASS

**Task**: ส่ง joint command **เดียวกัน** (gait replay จาก `main_script.py`/`ds_loopsm.csv` — เป็น open-loop, ไม่ขึ้นกับ morphology) ไปที่ short-leg (0.5x) กับ long-leg (base) แล้วดูว่า behavior ต่างกันชัดเจนไหม

**ข้อสังเกตสำคัญ**: ไม่ต้องมี trained policy — main_script.py's gait replay ถูก migrate มาพร้อมกับทุก morphology variant อยู่แล้ว (เพราะสร้างจาก base scene เดียวกัน) ใช้เป็น "controller" สำหรับ Step -1 ได้ทันที ส่วน trained AIRL/PPO policy (closed-loop, ปรับตาม morphology) เก็บไว้ใช้ตอน Phase 2 (real data collection) แทน

**ผลลัพธ์** (`sim/step_minus1_morphology_gap.py`, 10 วินาที):

| Metric | short (0.5x) | long/base (1.0x) |
|---|---|---|
| Forward distance | 3.49 m | 4.77 m |
| Body height std | 0.0192 m | 0.0165 m |
| Foot swing clearance (6 ขา) | สม่ำเสมอ ~0.13–0.16 m | ไม่สม่ำเสมอมาก 0.05–0.38 m |

Foot height ของ front-left leg เห็นความต่างชัดเจนที่สุด: long/base มี peak แหลมสูงถึง ~0.39m ระหว่าง swing phase ส่วน short นิ่งกว่ามากและ rhythm ต่างกัน (ดู `step_minus1_comparison.png`)

**สรุป**: **PASS** — morphology gap เป็นจริงภายใต้ scale factor ปัจจุบัน (0.5/0.75/1.0) ผ่านเกณฑ์ที่ direction_plan.md ตั้งไว้ ("visually distinct behavior → morphology gap is real → proceed")

---

## 10. Full Project Audit (6 agents) — ข้อค้นพบที่เปลี่ยนทิศทาง

ทำ audit เต็มรูปแบบด้วย Sonnet agent 6 ตัว 2 รอบ — อ่านทุก doc, ทุก script, repo ของ Ajan YuChen, paper ทั้ง 5 เล่ม, และ feedback ของอาจารย์ทั้ง 2 ท่าน เจอเรื่องสำคัญที่ต้องแก้ทันที

### 10.1 🔴 Blocker #1: ไม่มีกล้องใน CoppeliaSim — งาน 2 ฝั่งไม่เชื่อมกัน

**scene ไม่มี vision sensor เลย** ไม่มีโค้ดไหน capture RGB จาก CoppeliaSim ทั้งสิ้น (ยืนยัน: ไม่มี `getVisionSensorImg` ใน `sim/`, ไฟล์ `.ttt` ไม่มี vision-sensor object, repo แล็บก็ไม่มี)

→ **งาน V-JEPA2 ทั้งหมดที่ผ่านมารันบนวิดีโอ Unitree B1 (หุ่น 4 ขา) ที่ render จาก IsaacSim/MuJoCo — ไม่เคยรันบน stick insect เลย** แปลว่า gate ของ Ajan Go ("ทดสอบ Visual Encoder ก่อน") **ยังไม่ผ่านจริง** ทุกอย่างตั้งแต่ Step 0 เป็นต้นไปติดตรงนี้

### 10.2 🔴 z_t = 64 ไม่ใช่ 512 — อ่าน paper ผิด

`direction_plan.md` และ `note.md` เขียนว่า z_t ∈ ℝ^512 "confirmed LAC-WM Table 4" — **ผิด**
Table 4 คอลัมน์ "Latent Dimension = 512" คือ **hidden width ภายใน** ของ ITM/FTM ส่วน LAC-WM §4.2 เขียนแยกไว้ว่า *"Both models employ an action embedding dimension of **64**"* → **z_t ∈ ℝ^64** (แก้ทั้ง 2 ไฟล์แล้ว — ข้อดี: latent เล็กลง 8 เท่า ช่วยเรื่อง compute ด้วย)

### 10.3 🔴 HiLAM เป็น fallback ที่ผิด → เปลี่ยนเป็น UniSkill

HiLAM แก้ **temporal abstraction** ไม่ใช่ **embodiment invariance** — chunking ของมันดู feature dissimilarity ระหว่าง token ที่ติดกัน**ใน video เดียว** ไม่มี cross-embodiment objective เลย ถ้า z_t encode morphology อยู่แล้ว การ chunk จะ **ตอกย้ำ cluster เดิม** ไม่ได้แก้ (การทดลองเป็น LIBERO manipulation 100%, ไม่มี code)

→ ตัวที่ถูกคือ **UniSkill** (Kim et al. 2025, CoRL) — *"Imitating Human Videos via Cross-Embodiment Skill Representations"* **จุดสังเกต: HiLAM เอา IDM/FDM ของ UniSkill มาใช้เป็น frozen submodule** คุณสมบัติ cross-embodiment มาจาก UniSkill ตั้งแต่แรก

### 10.4 🎯 คำถามที่ยังไม่ตอบ และเป็นตัวตัดสินโปรเจกต์

**3 แหล่งอิสระชี้ไปที่ปัญหาเดียวกัน**: (1) Ajan Blink ถามตั้งแต่ Week 4 ว่า *"ถ้าสุดท้ายก็ต้องใช้ joint command แล้วแปลงเป็น latent ทำไม?"* — **ยังไม่เคยตอบ**; (2) LAC-WM มี embodiment ที่ action space **ต่างกันจริง** (10D/20D/**138D**/25D) แต่ของเรา 3 ร่างใช้ **18 มิติเหมือนกันเป๊ะ** → motivation ของ paper หายไป; (3) `deep_research.md` CP-002 บอกว่า explicit conditioning อาจพอสำหรับ interpolation

**คำตอบ**: เปลี่ยน framing จาก *action-space heterogeneity* (เราไม่มี) → **dynamics heterogeneity** (command เดียวกันให้ผลต่างกันตามความยาวขา) **ซึ่ง Step -1 พิสูจน์ไปแล้ว** → Step -1 ไม่ใช่แค่ sanity check แต่เป็น **ฐานหลักฐานของ motivation ทั้งหมด**

**การทดลองที่ต้องทำ**: latent-conditioned FDM **vs** raw-joint-conditioned FDM (shared encoder) — ถ้า latent ไม่ชนะ thesis กลวง **ต้องรู้เดือนแรก**

> **อัปเดต (2026-07-25) — เก็บการทดลองไว้ แต่ปรับ framing (ตรงกับ proposal §3.6.3 + deck):** ห้ามตัดสินด้วยการเทียบ **loss** ตรงๆ
> — `F(e_t,z_t)` vs `F(e_t,a_t)` ไม่แฟร์ เพราะ `z_t` เห็นเฟรมอนาคตแล้ว (อนุมานจาก `e_t,e_{t+1}`) และเป็น 64 มิติ vs 18 มิติ
> **ตัวตัดสินจริง = two-sided probe** (`z_t` ทำ behaviour transfer ขึ้น + morphology decode ลง พร้อมกัน) ส่วน **value-over-raw**
> วัดด้วย **adaptation efficiency** (episode ถึง target: pretrained-z vs pretrained-raw vs scratch) + **availability argument**
> (`a_t` ที่ถูกต้องของร่างใหม่ต้องใช้ kinematics/IK = privileged; `z_t` มาจากภาพ → แค่เสมอก็ชนะ) การเทียบ FDM latent/raw/obs-only
> ยังอยู่เป็น **diagnostic (Step E, รันเร็ว)** ไม่ใช่ตัวตัดสินเดียว

### 10.5 Phase 2 policy plan เดิมใช้ไม่ได้ → เปลี่ยนเป็น IK Retargeting

repo ของ Ajan YuChen ให้ไม่ได้อย่างที่คิด: checkpoint ตัวดี (`66k_aug3c`) **ไม่มีในสำเนานี้**; ตัวที่มีเป็น **34-dim obs** แต่**ไม่มี `normalized_env*.py` ตัวไหนใน repo ที่ผลิต 34 dim** (base = 36, module ที่ทำได้ถูกลบไปแล้ว); normalization bounds เป็น **ค่า literal วัดมือต่อร่าง** ไม่มี tooling คำนวณใหม่ และ **ไม่มีการ clip** → ขาสั้นลงจะพังแบบเงียบๆ; expert data = **สัตว์ตัวเดียว gait เดียว trial เดียว** ใช้กับร่างที่ scale แล้วไม่ได้; ~1 วัน/run; **ไม่มี precedent เรื่องความยาวขาเลย** (ทุก variant คือ *ตัดขา* หรือ *เปลี่ยนพื้น*)

→ **IK retargeting** (ตัดสินใจแล้ว): นิยาม behavior เป็น Cartesian foot trajectory → `simIK` ต่อ morphology → ได้ **a_t ต่างกันต่อร่าง** โดยไม่ต้องเทรน และได้ turn/stop มาฟรี

### 10.5b Phase 2 — Deployment scheme (closed-loop) — 2026-07-25

นอก scope thesis แต่บันทึกให้ตรงกับ direction_plan.md (เต็มที่นั่น) + deck Slide 24 + `report/pipeline_diagram.tex`

- **ลูปปิดอันเดียว ไม่ใช่ open-loop replay** — เล่นซ้ำ `z` ของ demo บนร่างจังหวะต่างกัน → phase หลุด (`z_t` เป็น transition; คู่ `(e_t,z_t)` กลายเป็น OOD)
- **reward optional** เก็บหลายไอเดีย: (1) match-demo ใน z-space (ไม่ต้อง reward), (2) reward RL (Dreamer), (3) **behaviour-matching RL (preferred)** — `z_target` → ทำจริง → ITM re-encode → `z_achieved` → RL reward `−‖z_achieved−z_target‖²`
- **behaviour-matching:** อยู่ใน z-space (ไม่ผูกรูปร่าง) + ไม่ต้องมี label `a_t` **แต่เป็น RL ไม่ใช่ backprop** (`a_t→e_{t+1}` = ฟิสิกส์ diff ไม่ได้; FTM แทนไม่ได้เพราะรับ `z`) → ใช้ CEM / off-policy SAC **ไม่ใช่ PPO**
- decoder ยังต้อง calibrate offline นิดหน่อยด้วย `(e_t,a_t)` ของร่างใหม่ (coverage > duration)

### 10.6 ข้อมูลจาก paper ที่ยืนยัน/แก้ความเข้าใจ

- ✅ **V-JEPA2 paper ยืนยัน frame-duplication trick ของเรา**: V-JEPA2-AC §3.1 — *"We use V-JEPA 2 encoder as an **image encoder and encode each frame independently**… the encoder is kept **frozen**"* และตอน pretrain เองก็ *"duplicate an image temporally and treat it as a 16-frame video"*
- ⚠️ **V-JEPA2 ไม่เคยเห็น simulated data เลย** (VideoMix22M = วิดีโอจริงล้วน) และ paper ไม่เคยศึกษา rendering domain gap → ผล §5 ของเราไม่ขัดกับ paper แต่ paper ก็อธิบายไม่ได้
- ⚠️ **LAC-WM fine-tune ไม่ใช่ few-shot**: ใช้ BFA ทั้ง 7,265 trajectories × 60k iterations — Step 2 ของเรา (N=5/10/20/50/100 episodes) เป็นคนละการทดลอง (อาจดีกว่าด้วย) แต่**อย่าอ้างว่าทำตาม protocol เขา**
- ⚠️ **Compute**: LAC-WM = 64× H200 × 4 วัน ≈ **256 H200-GPU-days**, batch 512 — เรามี 2080 Ti 11GB ตัวเดียว ต้องลด scale มาก
- ⚠️ **paper ไม่บอกค่า λ_recon/λ_motion เลย** รวมถึง learning rate, optimizer, schedule
- ℹ️ **`s44182` คืออะไร**: *"Learning aggressive animal locomotion skills for quadrupedal robots solely from monocular videos"* (npj Robotics 2025, HKU) — ใช้ pose→3D→**retarget ด้วยสูตรมือ**→AMP ไม่มี latent action ไม่มี world model → เป็น **related work / ตัวเปรียบเทียบ** (วิธี "สูตรมือ" vs วิธี "เรียนรู้ invariance" ของเรา) ไม่ใช่ component

### 10.7 Baseline ที่ควรใช้ (ต้องมี ≥2 ตามที่ ICLR วิจารณ์ LAC-WM)

1. **raw-joint-conditioned FDM** (ข้อ 10.4) — **บังคับ ไม่ใช่ทางเลือก**
2. **Danesh et al. 2026** — *"hardware-agnostic quadrupedal world models via morphology conditioning"* (arXiv:2604.08780) — explicit conditioning vs implicit latent ของเรา = คู่เทียบที่ตรงที่สุด
3. scratch FTM (มีในแผนอยู่แล้ว)

paper อื่นที่ควรเพิ่มใน literature review: **CLAM** (continuous latent action — architecture ใกล้เราที่สุด), **"What Do LAMs Actually Learn?"** (Zhang et al., NeurIPS 2025 — ใช้ป้องกัน clustering claim), **H-Zero** (locomotion + cross-embodiment few-shot), **DiLA**, **AnyMorph** — และ **ต้อง cite V-JEPA2 เอง** (`deep_research.md` ไม่เคย cite ทั้งที่เป็น backbone เรา)

### 10.9 Phase 1 — สร้าง vision pipeline สำเร็จ + เจอปัญหาใหม่ 3 อย่าง (2026-07-17)

**สร้างเสร็จแล้ว** (`sim/set_floor_texture.py` → `sim/add_camera.py` → `sim/record_episode.py`):
CoppeliaSim ต่อกับ V-JEPA2 ได้แล้ว → `frames.npy (N,256,256,3)` + `actions.npy (N,18)` ที่ **align กันเป๊ะภายใน step เดียวกัน** (จำเป็นสำหรับ L_motion — screen record ทำแบบนี้ไม่ได้)

**ยืนยันแล้ว**: void 0.00%, camera offset `[0, 1.532, 1.286]` **เหมือนกันเป๊ะทั้ง 3 variant**, brightness 128.3/129.0/129.4 → **render lock ใช้ได้**
morphology gap เห็นในภาพจริง: **1.372 / 1.080 / 0.831 m** (long/medium/short) จาก command เดียวกัน

**bug ที่เจอ (บันทึกใน `sim/SOURCES.md` แล้ว)**: (1) vision sensor มองตาม **+Z ไม่ใช่ -Z** → ภาพดำสนิท; (2) `createVisionSensor` default layer = 8 แต่หุ่นอยู่ layer 1 พื้น 32768 → **ไม่ render อะไรเลยแบบเงียบๆ**; (3) พื้น default เป็น **checkerboard** = สิ่งที่ audit บอกว่าอันตราย (r=-0.16) → เปลี่ยนเป็น matte texture; (4) elev 30° + FOV 60° → ขอบบนภาพเป็น **void ดำ ~15%** = blank-patch problem → แก้เป็น elev 40° FOV 45° → void 0.00%

**🔴 ปัญหาใหม่ที่สำคัญกว่า episode length**:

1. **หุ่นไม่ได้เดินตรง — มันเดินโค้ง** และโค้งคนละทางต่อ morphology!

| | straightness | heading drift / 400 steps |
|---|---|---|
| long 1.0× | 0.68 | **+83.7°** (โค้งซ้าย) |
| medium 0.75× | 0.85 | **−19.9°** (โค้งขวา) |
| short 0.5× | 0.83 | **−18.5°** (โค้งขวา) |

→ **"walk" ตอนนี้ไม่ใช่ behavior ที่นิยามได้** มันคือ "เดินโค้ง 84° ถ้าขายาว, โค้ง 20° อีกทางถ้าขาสั้น" → **walk กับ turn แยกกันไม่ออก** ซึ่งพังตรง K-means(K=3) ของ Step 1.5 พอดี

2. **Sim เป็น chaotic** (ไม่ใช่ bug): reload scene ทำให้ต่างกันระดับ **4.4e-16 (machine epsilon)** ที่ step 0 → ขยายเป็น 1e-3 ที่ step 10 → **1.8 m ที่ step 200** เป็นฟิสิกส์จริงของ legged contact dynamics แก้ไม่ได้
   - **โหลด scene ครั้งเดียวแล้วรันหลาย episode = deterministic 100%** (spread 0.0000 m)
   - engine: Bullet 2.78, timestep 0.05s (**20Hz ไม่ใช่ 60Hz ตามที่ plan เขียน**)

3. **แต่ morphology gap รอด** — 5 episodes × 200 steps: long **4.125 ± 0.434** / medium **3.562 ± 0.015** / short **2.646 ± 0.002** m → `long_min 3.593 > short_max 2.648` **ไม่ทับกันเลย → Step -1 PASS ยืนยันได้**
   - แต่ variance ขึ้นกับความยาวขา (σ 0.434/0.015/0.002) และ **long leg เป็น bimodal** (ลงที่ 4.479 หรือ 3.593 เท่านั้น = 2 basins)
   - → **Step -1 ควรรายงานเป็น mean ± std หลาย episode ไม่ใช่ค่าเดียว**

**ทางแก้ที่ตัดสินใจแล้ว**: **IK + yaw feedback** — IK อย่างเดียวไม่พอ เพราะ IK คุมเท้าเทียบกับลำตัว ไม่ได้คุม heading ในโลก → ยังโค้งอยู่ดี ต้องมี P-controller คุม heading:
- **walk** = hold heading 0 | **turn** = hold heading rate ω | **stop** = ไม่ก้าว
→ behavior นิยามชัดโดยโครงสร้าง, ตัด drift, **damp chaos ด้วย** (closed-loop ลดการขยายของ perturbation), และ a_t ยังต่างกันต่อ morphology

### 10.10 🎓 Novelty audit + literature — ข้อค้นพบที่เปลี่ยนทิศทาง (2026-07-17)

> รายละเอียดเต็ม (อังกฤษ) อยู่ที่ `report/audit_2026-07.md` §6.5–6.7

**เรื่องใหญ่ที่สุด: Ajan Go = Poramate Manoonpong** → **Larsen et al. 2023, Chuthong et al. 2026, Sun/Dai/Manoonpong 2023 = งานแล็บเราเอง ไม่ใช่คู่แข่ง** paper ที่ใกล้ที่สุดในโลก (สปีชีส์เดียวกัน simulator เดียวกัน) อยู่ห้องข้างๆ → **CPG params, ข้อมูล Medauroidea จริง, คนสร้าง = เข้าถึงได้หมด ถามเอาไม่ต้อง reimplement**

**Novelty: claim เดิมตาย แต่ claim ใหม่แข็ง**

| | ประเมินรอบแรก | **หลังอ่าน PDF จริง** |
|---|---|---|
| Li et al. RA-L 2021 | ชนตรงๆ | ❌ **ไม่ชน** — "generalize to multiple robots" = เอาสูตรไปรันใหม่ทีละหุ่น, Daisy กับ A1 **ต่างคนต่างเทรน ไม่แชร์ latent เลย**, ไม่เคย transfer ข้าม morphology, ไม่เคยเปลี่ยนความยาวขา, **ไม่มี vision**, ไม่วิเคราะห์ latent |
| QWM 2026 | ชนตรงๆ | ❌ **ไม่ชน — เป็นงานคู่ขนาน** — **ไม่มี latent action เลย** (action = joint target ดิบ 12 มิติ, ไม่มี inverse dynamics), **ต้องใช้ CAD ตลอด**, proprioception ล้วนไม่มี pixel, **ไม่มี controlled sweep** (8 หุ่นต่างกันทุกอย่างพร้อมกัน — เขาเขียนเองว่า *"not merely a collection of scaled variants"*), domain randomization **ไม่เคยเปลี่ยนความยาวขา** |

**claim ที่ป้องกันได้**: *"first to test whether a **vision-only, spec-free** latent **action** space organizes by **behavior** rather than **morphology**, under a **controlled single-axis** leg-length sweep"* — ไม่ต้องไปสู้คำว่า "first WM for cross-morphology locomotion" (ปล่อยให้ QWM ในขอบเขตเขา)

**🎁 ของที่ได้จาก paper ที่กลัวว่าจะฆ่าเรา**
1. **QWM รัน ablation ที่พิสูจน์เดิมพันเราแล้ว** — "w/o PME (Implicit Identification)" → *"successfully learns to maximize episode length"* แค่ reward ต่ำกว่า → **implicit เวิร์ค** และ **เขาไม่เคยทดสอบ zero-shot = ช่องว่างเรา**
2. **QWM แจกวิธีวัดให้** (App. F-E): **silhouette + variance decomposition** — เขาวัดแค่ด้าน morphology (z_t silhouette 0.033) **เราวัดทั้ง 2 ด้าน = contribution**
3. **Li et al. แจก baseline design** — 4 ตัว รวม **IK oracle** ที่เขาเรียกเองว่า *"rarely addressed in hierarchical learning literature"*
4. **Li et al. แจก stress-test framing** — ขาพัง: IK ล่ม (0.91) learned ทน (0.62) = **leg-length transfer ของเราคือรูปแบบเดียวกัน**
5. **เราเข้มงวดกว่า Li et al. ได้** — เขา**ไม่รายงาน seed/trial** ของ Fig.7 เลย, "sample efficiency" เป็นแค่ 3 จุด ไม่ใช่ loss curve

**⚠️ 3 คำเตือน**
1. 🔴 **QWM เขียน future work ว่า "integrate visual observations" + "disentangle task representations"** = ทางเรา (paper เม.ย. 2026) → **ต้องรีบ**
2. เขา**โจมตีแนวคิดเราตรงๆ**: *"implicit system identification... **unsafe** for zero-shot"* (adaptation lag) → **เตรียมตอบ**: เราดูจากกล้อง**ภายนอก** หุ่นไม่ต้องขยับเพื่อระบุตัวเอง
3. **pitfall ที่จะเจอ**: *"All configurations lacking ARN **fail entirely to learn**"* — ขายาวเดินเร็วกว่า → signal scale ต่างกัน → **ต้องเช็ค**

**📚 Literature ล้มแผน C (FK→scale→IK→yaw PID) ทั้ง 3 จุด**
1. **foot trajectory เป็น invariant ที่ผิด** — วงการใช้ **AEP/PEP + duty factor + phase** (Cruse Rule 4 นิยามบน AEP) และ**รูปเส้นทางระหว่าง AEP-PEP แทบไม่ถูกควบคุม** → **ยืนยันจาก notebook Ajan YuChen เอง**: `foot_traj` ทั้ง codebase = **แค่พิกัด Z** ใช้เป็น contact detector, metric ที่เขาวัดคือ duty factor/phase/tripod index/cyclogram (joint space)
2. **timing เดิม + geometric scale = ถูกปฏิเสธ** — Alexander: `f ∝ l^(−1/2)` (ขา 0.5× ควรเร็วขึ้น 1.41×) แต่ **caveat**: Froude มาจากสัตว์วิ่งมีช่วงลอย ตั๊กแตน duty factor≈1 = regime ที่ Froude อ่อนสุด **และมดจริง (Cataglyphis, JEB 2021) ขาสั้น 32% แต่ stride ต่าง 40-50% = ไม่ scale เชิงเส้นอยู่ดี**
3. **yaw PID ไม่มีในสัตว์จริง** — *"Straight walking on a slippery surface"* (JEB 2009): **ตัดขา 5 ขา ขาที่เหลือยัง generate pattern ถูก** → เดินตรงเป็น emergent จากกฎ local ไม่ใช่ตัวคุมกลาง; เลี้ยวจริง = **เลื่อน AEP/PEP ไม่สมมาตร** (Dürr 2009)
4. **สมมติฐาน nymph ของผมผิด** — Büscher 2022: สัดส่วนขา:ตัว **isometric ทุก instar**
5. **α=0.85/β=1.2 ของ s44182 เป็นสมการกำพร้า** — โผล่ครั้งเดียว ไม่เคยถูกใช้ ไม่มีที่มา **ห้ามอ้าง**

**🔴 และข้อเท็จจริงที่กระทบ behavior taxonomy โดยตรง**
Larsen et al. เขียนเอง: *"Only sequences in which the insects **walked without stopping or turning** were analyzed"*
→ **ข้อมูลชีวภาพของสปีชีส์นี้ ไม่มี turn ไม่มี stop เลย** → **turn/stop จะไม่มีฐานชีวภาพไม่ว่าทำวิธีไหน**

**Larsen et al. 2023 — CPG ของแล็บ (คำตอบสำคัญ)**
- **ไม่มี scaling law เลย** — *"no explicit scaling law is applied to any control parameter"* จัดการขายาวไม่เท่ากันด้วยการ **เทรน RBF แยกต่อชนิดขา ด้วยข้อมูลจริงของขานั้น** → **ไม่แก้ปัญหาเรา** (เราไม่มีข้อมูลจริงของขา 0.5×)
- แต่แถมให้ฟรี: **timing ปรับเอง** (dual-rate learner) → ปัญหา Froude ละลาย; **เลี้ยวด้วย bilateral TC gain** (Sun et al. 2023); **bounded output** → ไม่มีปัญหา reachability; **coordination emergent** (CPG 6 ตัวไม่มี coupling เลย)
- ขาจริง: หน้า **84.94** / กลาง **56.61** / หลัง **71.08** mm → **FL:ML:HL = 1.50:1.00:1.26** → **สัตว์จริงมีช่วง 1.5× อยู่แล้ว, 0.5× ของเราอยู่นอกช่วงธรรมชาติ**

### 10.11 ✅ Step 0 — PASS (2026-07-17) รันบนตั๊กแตนจริงเป็นครั้งแรก

**แก้เกณฑ์ก่อนรัน** — เกณฑ์เดิม (*"morphology ต้องไม่ dominate e_t"*) **ผิดตรรกะ**: ขา 0.5× กับ 1.0× **หน้าตาต่างกันจริง** encoder ที่ดี*ต้อง*แยกออก และถ้า `e_t` morphology-agnostic อยู่แล้ว **เราไม่ต้องมี ITM เลย** → **Step 0 ต้องถามเรื่อง "ข้อมูลมีอยู่ไหม" ไม่ใช่ "invariance เกิดหรือยัง"** (เหตุผลเต็มใน `direction_plan.md`)

**ข้อมูล**: 3 morphology × 3 episode × 200 step = **1800 เฟรม**, render ล็อกหมด
`sim/collect_step0.py` → `scripts/step0_encode.py` → `scripts/step0_analyze.py`

**คุณสมบัติสำคัญของการทดลอง**: gait period = **64 step เป๊ะ** (`a_t` ซ้ำ bit-exact ที่ 64/128/192) → phase label แม่นยำ ไม่ใช่ประมาณ · **`a_t` เหมือนกัน bit-exact ทุก morphology ทุก episode** → **สัญญาณ morphology ใน `e_t` เป็นภาพล้วนๆ**

**✅ Check 1 (ด่าน) — phase decode ได้ → ผ่าน**
| | accuracy | chance |
|---|---|---|
| linear probe (random 5-fold) | **85.1% ± 5.6** | 12.5% |
| linear probe (**grouped CV** — ตัดทั้ง episode ไม่มี leak) | **92.7% ± 1.8** | 12.5% |
| **shuffle control** | **12.3%** ≈ chance | 12.5% |

shuffle ตกที่ chance พอดี → **สัญญาณจริง ไม่ใช่ overfit** (จำเป็นต้องเช็ค: 1408 มิติ vs ~1440 sample) → **ITM มีของให้ดึง**

**📊 Check 2 (baseline ไม่ใช่ด่าน)** — morphology probe = **99.9%** · silhouette(morphology) = **+0.0835** (between-var 22.4%) · silhouette(phase) = **−0.0222**

**🔑 ข้อค้นพบสำคัญที่สุด — phase code พันกับ morphology**
| | phase accuracy |
|---|---|
| **ภายใน** ร่างเดียว | long **97.3%** · medium **96.3%** · short **93.8%** |
| **ข้าม** ร่าง (เทรน 2 ทดสอบตัวที่เหลือ) | long **39.0%** · medium **34.8%** · short **27.0%** |

→ phase อ่านได้เกือบสมบูรณ์*ในร่างเดียว* แต่ **ย้ายข้ามร่างไม่ได้** (93-97% → 27-39%)
→ **แต่ละ morphology มี phase manifold ของตัวเอง** — เห็นในภาพ UMAP: 3 เกาะแยกขาด phase กระจายอยู่*ในแต่ละเกาะ* ไม่มีโครงสร้าง phase ระดับโลก (`step0_umap.png`)

> **นี่คือช่องว่างที่ ITM + cross-augmentation มีไว้เพื่อปิด — และตอนนี้*วัดได้แล้ว***
> เป้าของ Step 1.5 ไม่คลุมเครืออีกต่อไป: **ดัน cross-morphology phase transfer ให้เกิน 27-39% พร้อมกับกด morphology decodability ให้ต่ำกว่า 99.9%**

**⚠️ ข้อค้นพบเชิงวิธีการ — ถ้าใช้ silhouette อย่างเดียวจะสรุปผิด**
silhouette(phase) = **−0.0222** บอกว่า *"ไม่มีสัญญาณ phase"* แต่ probe บอก **85-93%** — **ถูกทั้งคู่**:
- **silhouette วัด "ครอบงำไหม"** → phase ไม่ครอบงำ
- **probe วัด "มีอยู่ไหม"** → phase มีอยู่

ระยะทางแบบ Euclidean ใน 1408 มิติถูกกลบด้วย variance ที่ไม่ใช่ phase (Check 3: same-phase 40.23 vs diff-phase 44.93 = **1.12× เท่านั้น**)
→ **QWM (App. F-E) รายงานแค่ silhouette** — **เราต้องรายงานทั้ง probe และ silhouette** ไม่งั้นจะสรุปว่าสัญญาณที่มีอยู่ "ไม่มี"

### 10.12 🔴 Step 0 label อ่อน — ต้องเก็บใหม่ด้วย foot contact (2026-07-18)

**ผู้ใช้จับจุดอ่อนของ label "phase = step mod 64" ได้ 3 ข้อ — ทั้งหมดถูก:**

1. **64 ไม่ใช่จังหวะเดินธรรมชาติ — มันคือความยาว loop ที่ทีมแล็บตัดมา** `ds_loopsm.csv` มี 67 แถว, `main_script.py` วน rows 2→64. ยืนยัน: `a_t` ซ้ำเป๊ะทุก 64 step **เพราะโปรแกรมสั่งวน** ไม่ใช่เพราะขาเดินครบรอบ
2. **loop ไม่เนียน** — แถวต้น loop vs ปลาย loop มุมข้อต่อต่างกัน **14.75°** → มีรอยสะดุด (discontinuity) ปลอมทุก 64 step
3. **step-mod ไม่ตรงข้ามร่าง** — คำสั่งเหมือนกัน ≠ เท้าแตะพื้นจังหวะเดียวกัน (ขาสั้นแตะพื้นเร็วกว่า) → **นี่อาจเป็นสาเหตุที่ cross-morphology transfer ได้แค่ 27-39% — เป็นความผิด label ไม่ใช่ encoder**

**ผลกระทบต่อผล Step 0 เดิม (§10.11):**
- ✅ "phase decode ได้ 85%" — ยังจริง (label เทียมก็ correlate กับท่าจริงพอควร)
- ⚠️ **"cross-morphology 27-39%" เชื่อไม่ได้เต็มร้อย** — อาจต่ำเพราะ label หยาบ ไม่ใช่ encoder แย่ → **ต้องเก็บใหม่เพื่อแยกให้ออก**

**สิ่งที่ยืนยันจากข้อมูล (แก้ไข 2026-07-21)**: การเดิน **ไม่ใช่ tripod สะอาด** — เคยเขียนว่าเป็น tripod โดยอ้าง cross-set corr −0.31 ถึง −0.42 แต่นั่นคือการหยิบ **คู่ขาที่ลบสุด** มารายงาน ไม่ใช่ค่าเฉลี่ย วัดใหม่ด้วยวิธีเดียวกันทั้ง pilot (@0.5N) และ **expert 66k (binary contact ของ sim เอง)**: ชุด tripod A=FL,HL,MR / B=FR,HR,ML ให้ within-set corr เฉลี่ย ~**−0.05** (tripod ต้องเป็น +แรง เพราะขาในทีมต้องขึ้นลงพร้อมกัน) และ clean-tripod frame แค่ **~4%** → **ไม่ใช่ tripod** เป็น gait แบบคลื่น/สลับเฟส (staggered/metachronal-ish) ที่ขาทยอยลงไล่กัน (lag ~7-9 step/ขา) สอดคล้องกับ **Medauroidea เดินช้า** ที่ paper แล็บ (Larsen/Grabowska) บอกว่า tetrapod-ish. **สำคัญ: expert 66k ที่เทรนเต็มก็เป็นแบบเดียวกัน (FL duty 29% เทียบ pilot 24%)** → ไม่ใช่ bug จากการ replay/scale ของเรา แต่เป็นลักษณะ gait ของโมเดลตัวนี้เอง

**force sensor ใช้ได้**: ทุกขาแกว่ง ~0.05N (ยกขา) ↔ 7-21N (แตะพื้น) — สัญญาณ stance/swing ชัด

**ตัดสินใจ (2026-07-18)**: เก็บ Step 0 ใหม่ + บันทึก **force ดิบทั้ง 6 ขา** → label ด้วย **6-bit contact** (ขาไหนแตะพื้นบ้าง — ตรงที่สุด ไม่ตีความ)
> 🔴 **LIMITATION ที่ต้องกลับมาแก้**: 6-bit contact มาจาก **gait replay ของสัตว์ตัวเดียว (Animal06) ที่ loop ไม่เนียน** — ไม่ใช่ท่าเดินที่ optimize สำหรับแต่ละร่าง **เมื่อมี proper expert (เช่น CPG ของแล็บ Larsen et al. ที่ปรับ α ต่อร่าง หรือ AIRL retrain ต่อร่าง) ต้องเก็บใหม่และ label ใหม่** — 6-bit ตอนนี้คือ pilot ให้ pipeline เดินได้ ไม่ใช่ผลสุดท้าย
> เก็บ force ดิบไว้ → ลอง label แบบอื่น (contact pattern, จำนวนขาที่แตะ) ตอนวิเคราะห์ได้โดยไม่ต้องเก็บใหม่ — **หมายเหตุ: "2-phase tripod" ใช้ไม่ได้ เพราะ gait ไม่ใช่ tripod (ดูด้านบน)**

### 10.13 ✅ Step 0 v2 — 6-bit contact label ดีกว่าจริง (2026-07-18)

เก็บชุดใหม่ `data/step0_v2` (3 ร่าง × **5 ep** × 200 step = **3000 เฟรม**) + บันทึก **force ดิบ 6 ขา**
`sim/collect_step0.py` (เพิ่ม force) → `scripts/step0_encode.py` (เพิ่ม 6-bit contact) → `scripts/step0_analyze_v2.py`

**เทียบ label 3 แบบ — ตัววัดคือ cross-morphology transfer (เทรน 2 ร่าง ทดสอบร่างที่เหลือ):**

| label | within ร่างเดียว | **across ข้ามร่าง** | transfer ratio |
|---|---|---|---|
| time_phase (step mod 64 — เดิม) | 92.5% | 38.4% | 0.42 |
| **contact_6 (6-bit foot — ใหม่)** | 85.1% | **55.2%** | **0.65** ⬆️ |
| n_support (นับขาที่แตะ) | 70.0% | 29.6% | 0.42 |

**ผลสำคัญ**: เปลี่ยนไป label ที่ตรงกับ**ท่าจริง** (เท้าไหนแตะพื้น) → cross-morphology transfer **38% → 55%** (+17 จุด)
→ **ยืนยันว่า 27-39% เดิม ส่วนหนึ่งเป็นความผิดของ label ไม่ใช่ encoder** — "เท้าไหนแตะพื้น" เป็นภาษากลางข้ามร่างที่แท้จริง (ท่าเดียวกัน → contact pattern เดียวกัน ไม่ว่าขายาวแค่ไหน)
→ แต่ **55% ยังไม่ 100%** = ยังมีส่วนที่ encoder ผูก phase กับ morphology จริง = **ช่องว่างที่แท้จริงที่ ITM ต้องปิด** (หลังหักความผิด label ออกแล้ว)

- morphology probe = **99.0%** — baseline ไม่เปลี่ยน ✅
- n_support หยาบเกินไป (นับ 2 vs 3 ขา) → ทิ้ง
- ดู `step0_v2_labels.png` (bar chart within vs across)

**เป้า Step 1.5 อัปเดต**: ดัน cross-morphology contact-transfer จาก **55% → สูงขึ้น** พร้อมกด morphology < 99%
(ใช้ **contact_6 เป็น label หลัก** ตั้งแต่นี้ ไม่ใช่ step-mod-64)

🔴 **LIMITATION คงเดิม** (§10.12): 6-bit contact ยังมาจาก replay สัตว์ตัวเดียว loop ไม่เนียน — **ต้องเก็บ+label ใหม่เมื่อมี proper expert** (CPG แล็บ / AIRL ต่อร่าง) · เก็บ force ดิบไว้แล้ว → ลอง label แบบอื่นได้โดยไม่ต้อง re-collect

### 10.14 🔬 ทดสอบ reward signal — และยืนยันคำพูด Ajan Blink ด้วยข้อมูลจริง (2026-07-19)

**ที่มา**: ถ้าจะทำ Phase 2 (เทรน policy ในจินตนาการแบบ Dreamer) ต้องมี reward ที่อ่านจาก latent ได้
**แก้ความเข้าใจก่อน**: DreamerV3 **ไม่ได้**ให้ reward จาก "ทำนายถูก" — มี **reward predictor แยกต่างหาก** `r̂_t ~ p(r̂_t|h_t,z_t)` เรียนจาก reward จริงของ env (paper บรรทัด 328) ส่วน "ทำนายผิด = reward สูง" คือ curiosity-driven exploration ซึ่งเป็นคนละเรื่องและกลับด้าน

**ผลทดสอบ (ทำนาย forward velocity, grouped CV):**

| สัญญาณ | R² |
|---|---|
| time phase (step mod 64) | +0.249 |
| **foot contact 6-bit** | **+0.313** ← ดีกว่า time phase อีกครั้ง |
| ภาพ `e_t` (1408 มิติ) | +0.456 |
| **แรงที่เท้าดิบ (6 ค่า)** | **+0.926** |

**ข้อวินิจฉัยสำคัญ**: ทำนายความเร็วจาก**ป้ายท่าทางอย่างเดียว** ได้ **+0.612** ซึ่ง**ดีกว่าภาพทั้งภาพ (+0.456)** → `e_t` ไม่ได้เห็นการเคลื่อนที่จริง มัน**อ่านท่าทางแล้วเดา** · และหน้าต่างเวลายาวขึ้นไม่ช่วย (2 step −0.34 / 20 step +0.28) → **กล้องที่ track หุ่นทำให้ความก้าวหน้าในโลกจริงหายไปจากภาพ**

### 🔴 ผลชี้ขาด: ความสัมพันธ์ทางฟิสิกส์ **ไม่ transfer ข้ามร่าง**

| | ภายในร่างเดียว | ข้ามร่าง (เทรน 2 ทดสอบตัวที่เหลือ) |
|---|---|---|
| แรง → ความเร็ว | **+0.926** | **−0.33 / +0.02 / −1.66** |
| ภาพ → ความเร็ว | +0.456 | −0.36 / −1.11 / −2.42 |

fit ขายาว → ทดสอบขาสั้น: **R² = −5.23** (คู่ที่ต่างกันมากสุด) · short→medium: −0.19 (คู่ที่ใกล้กันสุด) → **ไล่ตามระดับความต่างของร่าง เป็นกลไก ไม่ใช่ noise**

**🎯 และ Ajan Blink พูดคำตอบไว้แล้ว** (`feedbacks/feedback_ajan_blink.md:25`, บริบทรอยเท้าไดโนเสาร์):
> *"ความลึกของรอยเท้า... ทำให้คำนวณย้อนกลับไปหา **แรง (Force), มวล (Mass), และความเร็ว (Velocity)**"*

เขาพูดถึง **มวล** ด้วย ไม่ใช่แค่แรงกับความเร็ว — เพราะ `F = ma` แรงเท่ากันบนมวล/แขนโมเมนต์ต่างกันให้ความเร็วต่างกัน
→ **ห่วงโซ่ของเขาต้องการพารามิเตอร์ร่างกาย ซึ่งคือสิ่งที่เราตั้งใจไม่รู้** → ข้อมูลของเรายืนยันข้อต่อ "แรง→ความเร็ว" ภายในร่าง (0.926) และยืนยัน**ความจำเป็นของเทอมมวล** โดยการที่มันพังข้ามร่าง

**ข้อเสนอที่ถูกฆ่าโดยผลนี้**: แนวคิด "ให้ FTM ทำนายแรง แล้วคำนวณ reward จากแรง" **ใช้ไม่ได้** เพราะแรงทำนายความเร็วบนร่างใหม่ไม่ได้
**สิ่งที่รอด**: reward head **ต้อง fine-tune ต่อร่าง** (ทำได้ เพราะ Step 2 fine-tune บนร่างใหม่อยู่แล้ว และมี `head` position เป็น label)

**💡 ความหมายเชิงวิทยาศาสตร์ (สำคัญกว่าเรื่อง reward)**: ผลลบนี้เป็น**หลักฐานสนับสนุน thesis** — ถ้าความสัมพันธ์ระดับฟิสิกส์ไม่ transfer แปลว่าวิธี "วัดฟิสิกส์แล้วคำนวณ" พังเมื่อเจอร่างใหม่ จึงต้องมี abstraction ระดับสูงกว่า **นั่นคือหน้าที่ของ latent action** และเป็นคำตอบต่อคำถาม *"ทำไมต้องมี latent"* ของ Ajan Blink เอง

**เข้าชุดกับที่เจอก่อนหน้า** — contact pattern ตรงกันข้ามร่างแค่ 16-36% · ความเร็วไม่ scale เชิงเส้นตามความยาวขา (exponent ~0.7-0.8) · force→velocity ไม่ transfer → **ทุกความสัมพันธ์ระดับฟิสิกส์ล้วนผูกกับร่าง**

**⚠️ ข้อจำกัดการทดสอบ**: ใช้ ridge/GBR บนเฟรมเดียวหรือคู่เฟรม ส่วน Dreamer ใช้ recurrent state `h_t` ที่สะสมประวัติ → **นี่คือขอบล่าง ไม่ใช่ขอบบน** แต่ผลที่หน้าต่างเวลายาวขึ้นแล้วไม่ดีขึ้น ชี้ว่าปัญหาอยู่ที่**ข้อมูลไม่มีในภาพ** ไม่ใช่โมเดลอ่อน

**การตัดสินใจที่ตามมา**: ตัด time-phase label ออกจาก analysis pipeline ถาวร (`step0_encode.py`, `step0_analyze_v2.py`) — รู้แล้วว่าเครื่องมือวัดพัง ไม่ต้องวัดซ้ำ ผลเปรียบเทียบครั้งเดียวบันทึกไว้ที่ §10.13 พอ

### 10.8 สิ่งที่แก้ไปแล้วใน doc (Phase 0)

| แก้อะไร | ไฟล์ |
|---|---|
| z_t 512 → **64** | `direction_plan.md`, `note.md` |
| Simulator IsaacSim → **CoppeliaSim v4.10** | `note.md` (2 จุด) |
| Morphology variants "ยังไม่มี" → **สร้างแล้ว verify แล้ว** (0.5/0.75/1.0) | `direction_plan.md` |
| Data collection policy TBD → **IK retargeting** + เหตุผลเต็ม | ทั้ง 2 ไฟล์ |
| Step -1 PASS + ตัวเลข | `direction_plan.md` |
| Step 0 Check 3 → **ABANDONED** (ทดสอบ 3 background ไม่ผ่านสักอัน) | `direction_plan.md` |
| **เพิ่ม render-confound risk** (ข้อ 5-6) — คุกคาม Step 1.5 | ทั้ง 2 ไฟล์ |
| Fallback HiLAM → **UniSkill** | ทั้ง 2 ไฟล์ (รวม Q&A อาจารย์) |
| **เพิ่ม motivation reframe + decisive ablation** | ทั้ง 2 ไฟล์ |
| เพิ่ม camera + turn/stop เป็น blocker ชัดเจน | ทั้ง 2 ไฟล์ |

---

## 11. สถานะปัจจุบัน / ต้องทำต่อ

> 🔄 **อัปเดต 2026-08 — ดู §12 ท้ายไฟล์สำหรับสถานะล่าสุด** (pivot ไป cross-embodiment + แผน staged)
> §11 ด้านล่างนี้คือสถานะ ณ ก.ค. 2026 (cross-morphology อย่างเดียว) เก็บไว้เป็นประวัติ

**Timeline: Aug–Nov ตั้งเป้าจบ ตุลาคม ≈ 12 สัปดาห์**

### ✅ เสร็จแล้ว
- [x] Encoder ทำงานถูกต้อง (single-frame mode) — **และ V-JEPA2 paper ยืนยัน trick นี้ด้วย** (ข้อ 10.6)
- [x] Cross-augmentation mechanism ยืนยันตรงกับ paper
- [x] Per-patch heatmap — ทดสอบ 3 background ไม่ผ่านสักอัน → ยกเลิก
- [x] Whole-frame UMAP — เจอ **render-style dominance** (เป็น confound ที่ต้องคุม ไม่ใช่ผลบวก)
- [x] CoppeliaSim ติดตั้ง + เชื่อมต่อได้ (GUI); headless ยัง segfault
- [x] Migrate model + gait script + wrapper จาก `airl-insect-walking` → `sim/`
- [x] สร้าง morphology variants 0.5× / 0.75× / 1.0× ยืนยันตัวเลขครบ 6 ขา
- [x] Step -1 Morphology Gap Check — **PASS**
- [x] **Full audit ด้วย 6 agents** (ข้อ 10)
- [x] **Phase 0: แก้ doc ให้ตรงกันหมด** (ข้อ 10.8) — z_t=64, CoppeliaSim, IK retargeting, UniSkill, confound, motivation reframe

### 🔴 Phase 1 — สร้างสิ่งที่ขาด (สัปดาห์ 2–3) ← **ตัวปลดล็อกจริง**
- [ ] **เพิ่ม vision sensor** เข้า `.ttt` ทุก variant (fixed, side view, ~30° elevated)
- [ ] **Recorder**: dump RGB frame + `a_t` (joint targets) ที่ sync กัน (`main_script.py:10` ตั้ง `self.logging = False` อยู่ — ต้องเปิด หรือ log ผ่าน ZMQ)
- [ ] **IK retargeting**: นิยาม walk/turn/stop เป็น Cartesian foot trajectory → `simIK` ต่อ morphology
- [ ] **ล็อค render environment** — กล้อง/แสง/พื้นหลัง เหมือนกันเป๊ะทุก session (ข้อ 5.5 ใน note.md) เลี่ยง checkerboard **และ** พื้นเรียบว่าง
- [ ] **Gate**: รัน domain-UMAP ข้าม session → cluster **ต้องทับกัน** ถ้าแยก = ข้อมูลใช้ไม่ได้

### Phase 2 — Step 0 ของจริง (สัปดาห์ 4)
- [ ] รัน encoder บน **stick insect จริง** (ที่ผ่านมาเป็น B1 ทั้งหมด)
- [ ] **implement Check 1** (UMAP by behavior) — `direction_plan.md` เขียนไว้แต่ **ไม่เคยมี script**
- [ ] Check 2 (UMAP by morphology)
- [ ] **เพิ่ม metric เชิงปริมาณ** (silhouette / k-NN purity) — ตอนนี้ `umap_domain_check.py` ไม่มีเลย ใช้ตาดูล้วนๆ
- [ ] **implement negative control** (stop-clip) — ไม่เคยทำ
- [ ] ยืนยัน short/medium variant ด้วยตาใน GUI (geometry ไม่ clip)

### Phase 3 — Step 1 + 1.5 (สัปดาห์ 5–9)
- [ ] เทรน ITM+FTM+MD บน short+long — **z_t=64**, fp16, batch เล็ก + gradient accumulation
- [ ] Step 1.5 validation (UMAP + K-means เชิงปริมาณ)
- [ ] อ่าน **UniSkill** ให้จบ**ก่อน** Step 1.5 รัน เพื่อให้ fallback พร้อม ไม่ใช่ไปหาตอนพัง

### Phase 4 — Step 2 + baselines (สัปดาห์ 10–12)
- [ ] Transfer: pretrained vs scratch FTM, N=5/10/20/50/100, LoRA rank 2
- [ ] 🎯 **Baseline 1 (บังคับ)**: raw-joint-conditioned FDM — diagnostic (Step E, รันเร็ว) **ไม่ใช่ตัวตัดสินเดียว** ตัวตัดสินจริงคือ two-sided probe + adaptation efficiency (ดู update 10.4)
- [ ] **Baseline 2**: Danesh et al. 2026 explicit morphology conditioning
- [ ] **Bonus**: สร้าง morphology 1.25× → ตอบคำถาม extrapolation ของ Ajan Blink (ใช้ `make_leg_morphology.py` คำสั่งเดียว)

### ค้างไว้ / ต้องถาม
- [ ] ⏰ **Proposal deadline**: `feedback_ajan_blink.md:36` เขียนว่า "ส่ง proposal วันที่ 7 เดือนหน้า" แต่ไฟล์ไม่มีวันที่ → **ต้องเช็คว่าคือวันไหน ยัง live อยู่ไหม**
- [ ] คุยกับ Ajan Go เรื่อง **motivation reframe** (ข้อ 10.4) ก่อนลงแรง 12 สัปดาห์
- [ ] แก้ headless segfault ถ้าต้อง collect แบบ unattended
- [ ] Rewrite `report/pre_proposal.md` (PCA→UMAP, forward-only→3 behaviors, simulator, ชื่ออาจารย์/นักศึกษาที่ยังว่าง)

---

## 12. 🔄 อัปเดตใหญ่ 2026-08 — Cross-Embodiment pivot + แผน staged

### 12.1 ทำไมต้อง pivot ไป cross-embodiment
คำวิจารณ์หลักของกรรมการ (Preaw/Hap): *"ทำไม vision ถึงคุ้มกว่า proprioception?"* หลัง vision
pipeline เสร็จ (fixed camera, render-lock, foot-contact macro-F1 = **0.886**) สรุปได้ว่า **บนร่าง
topology เดียวกัน (3 ความยาวขา, 18-D เหมือนกัน) พิสูจน์ "vision > proprioception" ไม่ได้** เพราะ
proprioception ก็แชร์ 18-D ข้ามร่างได้ → vision ได้เปรียบแค่ **reach** ไม่ใช่ **accuracy**

**การตัดสินใจ**: เพื่อ *พิสูจน์* (ไม่ใช่ argue) ต้องมีร่างที่ **action space ไม่ comparable (disjoint)** —
proprioception แชร์ไม่ได้เลย แต่ vision (pixel) เป็นพื้นที่ร่วม → เพิ่ม **Unitree B1 quadruped
(12-DOF)** เทียบกับ hexapod (18-DOF)
> **disjoint action space ≠ IK-retargeting**: IK = ค่า a_t ต่างกันใน **space เดียวกัน (18-D)** →
> comparable (proprioception ยังแชร์ได้); disjoint = **คนละ space** (18-D vs 12-D, ไม่มี correspondence)
> → proprioception แชร์ไม่ได้. นี่คือเหตุผลที่ 2 stage พิสูจน์คนละอย่าง (ดู 12.2)

### 12.2 แผน staged
- **Stage 1 — Cross-morphology** (3 ความยาวขา, **IK-retargeting**, 18-D เหมือนกัน): ทำให้ pipeline
  ทั้งเส้นเดินได้ในกรณีควบคุม + latent จัดกลุ่มตาม behavior ไม่ใช่ body + transfer. พิสูจน์ **"latent ดีกว่า
  raw-joint"** (แต่ยัง *ไม่* พิสูจน์ vision>proprio เพราะ topology เดียวกัน) — ใช้ IK เพื่อให้ a_t ต่างต่อร่าง
  → Motion Decoder ต้องอ่าน x_t (ไม่ vacuous)
- **Stage 2 — Cross-embodiment / compositional transfer**: เทรนบน **6-leg stick insect + Unitree B1
  quadruped** แล้ว **ทดสอบ transfer ไป 4-leg stick insect**. นี่คือคำตอบตรงต่อฟีดแบคกรรมการว่า
  "3 ความยาวขาง่ายเกินไปเมื่อเทียบกับศักยภาพ pipeline": train set มี action space คนละชนิด
  (hexapod 18-D vs B1 12-D) → proprioception แชร์ไม่ได้ แต่ vision เป็นพื้นที่ร่วม; test body 4-leg
  แชร์ appearance กับ insect และแชร์ leg-count/topology idea กับ quadruped → เป็น compositional held-out
  body ไม่ใช่แค่ interpolation.

### 12.3 B1 pipeline (สร้างเสร็จ session นี้)
`sim/rollout_b1_mujoco.py` → `sim/render_b1_replay.py`:
- **ที่พบ**: B1 policy (PPO, `base_gait3/model_600.pt`) เดินได้ใน MuJoCo แต่ CoppeliaSim รัน policy เองไม่ได้
- **วิธีทำ**: rollout ใน MuJoCo → **replay kinematic ใน CoppeliaSim** (set base pose + joint แล้ว capture)
  → `sim/build_b1_scene.py` สร้าง scene B1 จาก scene insect (พื้น/แสง/ค่ากล้องเดียวกัน)
- ⚠️ **ฉาก+ค่ากล้องเดียวกันยังไม่พอ** — ต้องให้ทั้งสองร่าง spawn ที่ตำแหน่งโลกเดียวกันด้วย ไม่งั้นเห็นพื้น
  คนละบริเวณ (รายละเอียด §16.8) ใช้ `--spawn` / `--cam_dx` / `--travel` ให้ตรงกันทั้งสองฝั่ง
- ข้อมูล: `data/b1_v1/` (8 clips: fwd 0.2–0.5, turn, spin, strafe), trajectory `data/b1_traj/`

### 12.4 ข้อมูล hexapod
- `data/hexapod_v1/` (CSV gait, long/medium/short × 8, 2699 เฟรม)
- `data/ik_v1/` (IK-retargeted, long/medium/short × 3) — **Stage 1 ใช้ IK** (บางไป ต้องเก็บเพิ่ม)

### 12.5 4-leg insect — policy ของแล็บใช้ไม่ได้ → เทรนเอง
- **ที่พบ**: AIRL cutlegs policy ของแล็บรันไม่ได้ (obs config หาย, กู้ไม่ครบ) — เช่นเดียวกับ AIRL 6-leg
- **สรุป**: world model ต้องการแค่ **เฟรม+command** ของ 4-leg ไม่ใช่ obs ของ policy → ถ้าจะทำ 4-leg **เทรนเอง
  (PPO, ไม่ต้อง expert demo)** เลือกขาที่ตัดเอง (น่าจะตัดคู่หน้า = quadruped-like). hexapod ใช้ CSV gait (เดินดีอยู่แล้ว)

### 12.6 Cleanup (session นี้)
จัด sim/scripts/data: ย้ายของเลิกใช้ (terrain, deprecated B1 direct-physics `collect_b1`, old recorder) →
`sim/_archive/`; merge `temporal_similarity_*` 3 ไฟล์ → 1 (`--mode`); ข้อมูลเก่า (step0_v2, terrain_all,
step0_fixedcam) → `data/_archive/`. **เก็บ**: report-figure generators ทั้งหมดใน scripts/, IK tooling (คืนมาแล้ว)

### 12.7 สถานะ / ต้องทำต่อ (staged)
**✅ เสร็จ**: vision pipeline (macro-F1 0.886) · morphology variants + Step -1 · audit + Phase 0 · **B1 pipeline+data** · **hexapod data (CSV+IK)** · render-lock ข้ามร่าง · cleanup

**🔴 Stage 1 (cross-morphology — current working route = IK forward-only, ดู §14; AMP log kept in §13)**
- [x] render-lock/encoder check บน `data/ik_walk`: `results/render_lock_ik_walk/` — body decode 0.951, behavior/repeat skipped เพราะ walk-only/no repeats
- [🔄] ใช้ `data/ik_walk` เป็น forward-only clean pilot ก่อน; postpone turn/strafe เพราะ fixed-camera path-in-frame confound ชัดเกินไป
- [ ] เทรน ITM+FTM+MD (z_t=64, fp16)
- [ ] latent validation two-sided แบบ forward-only ก่อน: morphology signal ใน raw `e_t` มีจริง → หลัง train ต้องดูว่า `z_t` ลด morphology และยังรักษา gait/contact signal ได้ไหม
- [ ] decisive ablation latent vs raw-joint FDM (diagnostic)
- [ ] transfer / sample-efficiency curve

**🔴 Stage 2 (cross-embodiment / compositional held-out)**
- [ ] สร้าง/เก็บข้อมูล 4-leg stick insect สำหรับ held-out test (น่าจะต้องเทรน walker เอง; policy cutlegs เดิมใช้ไม่ได้)
- [ ] เทรน latent WM บน {6-leg insect 18-D, B1 12-D} (per-embodiment Motion Decoder head)
- [ ] ทดสอบ transfer ไป 4-leg insect; เทียบ scratch / proprio baseline
- [ ] proof framing: proprioception แชร์ไม่ได้ข้าม 18/12-D; vision-latent แชร์ได้ และ generalize ไป body ใหม่ที่ compose จากสอง embodiment

**ค้าง/ต้องถาม**: คุย Ajan Go เรื่อง reframe (dynamics heterogeneity + cross-embodiment) · วิธีสร้าง/เก็บ 4-leg
held-out ให้เร็วและน่าเชื่อ · proposal deadline

---

## 13. 🦿 AMP — เทรน controller ต่อร่าง เพื่อสร้าง behavior dataset (2026-08, in progress)

**ทำไมเปลี่ยนจาก IK มาเป็น AMP**: Stage 1 เดิมวางไว้ใช้ **IK-retargeting** สร้าง `a_t` ต่อร่าง (§12.4,
direction_plan Step 0.5) แต่ `ik_v1` **บางไป** และ IK ให้ trajectory ที่ scripted/แข็ง — บังคับขาสั้นเดินตาม
foot path ของขายาว **ไม่ปรับ gait ตาม morphology**. เปลี่ยนมาเทรน **AMP (Adversarial Motion Priors)**
policy ต่อร่างแทน → ได้ gait ที่ **natural + ปรับตามความยาวขาเอง** และแต่ละร่างเดินด้วยจังหวะ/สปีดของตัวเอง.
งานอยู่ใน `amp/` เท่านั้น (**ไม่แตะ `airl-insect-walking/`** = reference อย่างเดียว).

### 13.1 reward = frozen gait prior + leg-scaled command
`reward = g(s') + λ(step) · command_reward`
- **`g(s')`** = gait prior จาก discriminator ของเพื่อนในแล็บ (`amp/discriminator.pth`, `AIRLDiscrim` 28-D,
  เทรนบน expert ขายาว *Medauroidea*) — **frozen + shared ทั้ง 3 ร่าง**. นี่คือกลไก **behavior correspondence**:
  ทุกร่างถูกดึงเข้าหา gait distribution อ้างอิงเดียวกัน → "ท่าเดียวกัน" นิยามด้วย gait prior ร่วม ไม่ใช่ Cartesian
  path ร่วม (อ่อนกว่า IK by-construction แต่แลกกับ gait ที่เป็นธรรมชาติ). ตอบข้อกังขาเดิม "RL อาจไม่ align
  ข้ามร่าง" — align ถูก anchor ด้วย shared frozen prior + command ที่ leg-scaled ไม่ใช่หวัง emergent luck.
- **obs 28-D**: body_z(1) + orientation(3, IMU-relative) + joint_angles(18, leg-major) + contacts(6).
- **command_reward = `track` mode**: `exp(−(vx−vx_target)²/σ²)` ∈ [0,1] — bounded, reward การ **hit** สปีด
  เป้า ไม่ให้รางวัลกับการวิ่งเกิน (กัน reared sprint). เดิมใช้ `vx*coef` (unbounded) → vx โดม g ~20:1 → เดินเชิดหัว.

### 13.2 morphology scaling (task-space เท่านั้น — ดู memo `morphology-scaling`)
วัด leg_length + standing height ตอน env init แล้ว scale อัตโนมัติ:
- **vx_target ∝ leg**: long 0.45 → medium 0.337 → short 0.225 m/s (ขาสั้นไม่ถูกบังคับวิ่ง 2× จังหวะตัวเอง)
- **track σ ∝ leg** (คง relative precision), **body_z obs bounds ∝ standing height** (0.254/0.195/0.135 m —
  ไม่งั้น stance ปกติของขาสั้นถูก normalize เป็น −2.49 = discriminator เห็นเป็น "ล้ม")
- **ไม่ scale**: joint-angle obs/action, orientation/contact, contact-force threshold (มวลเท่ากันทุกร่าง)

### 13.3 design fixes ที่ทำระหว่างทาง (กัน retrain ซ้ำ)
- **`g_clip=3.0`** — policy game frozen discriminator ได้ถึง g~7-8 (expert ~2.8) ด้วยการยืนนิ่งท่า adversarial
  → cap g ใกล้ช่วง expert ให้ command reward สู้ได้ = ต้องเดินจริง
- **`g_center`** — ลบ g_clip ออกจาก training reward ให้ค่าใกล้ 0 (critic converge เร็ว; policy ไม่เปลี่ยน
  เพราะ constant offset หายใน GAE advantage). critic loss 602 → ~4
- **λ gait-first schedule** — λ แบน `lam_min` ช่วง warmup (ให้ gait ก่อ) แล้ว ramp แบบ convex (quadratic,
  slow-start) ขึ้น `lam_max` = "เก่ง gait ก่อน แล้วค่อยตามคำสั่ง". env_reward bounded → lam_max เป็นเพดาน
  แข็ง command กลบ gait prior ไม่ได้อีก
- **windowed avg-velocity reward (anti-rocking, 2026-08-04)** — `track` เดิมอ่าน **instantaneous** head vx →
  ขาสั้น game ได้ด้วยการ **โยกตัวไปมา** (หัวแตะสปีดเป้าชั่วขณะทุกรอบ แต่ net displacement ~0, return/test ~80
  ทั้งที่ x_dist ~0). แก้เป็น **เฉลี่ย vx จาก net head displacement ต่อ window (25 step)** → การโยกหักล้างกันใน
  window ได้ ~0, เดินจริงเท่านั้นถึงได้รางวัล (`--track_window`, default 25). ยืนยันหลัง restart: short
  return/test 80 → 0.15 = exploit ตายแล้ว, ทั้ง 3 ร่างเริ่มที่ baseline เป็นธรรม
- **root-cause bug ที่เคยทำ sim ค้าง**: CoppeliaSim ใช้ system python3 (ไม่มี zmq) → scene script error →
  pause-on-error → sim ค้าง state=8 เทรนบนฟิสิกส์แช่แข็ง. แก้ด้วย `defaultPython` ใน `~/.CoppeliaSim/usrset.txt`
  + ลบ auto-runner `/script` (TARGET_RUNS=1) ออกจาก scene ทั้ง 3

### 13.4 setup การรัน
- 3 ร่าง เทรนขนานผ่าน 3 CoppeliaSim GUI (port 23060/61/62; render copy 23063). launch จาก venv เสมอ
- `amp/amp_train.py --port P --scene <body>.ttt --name insect_{long,medium,short}` +
  `--lam_warmup_frac --lam_ramp_frac` (schedule สั้น 0.03/0.15 → full λ ~350k, feedback ~4-5h)
- diagnostics: TensorBoard (`return/test`, `eval/x_dist`, `gait/g_eval`, `eval/pitch_abs`, `eval/ep_len`,
  loss/entropy/ratio) · `scripts/gait_report.py` (rate-normalized, ไม่ assume tripod) · `scripts/render_rollout.py`

### 13.5 สถานะ (2026-08-04)
- ✅ pipeline healthy: g_eval ~2.9 (≈expert), ep_len 1000 (ไม่ล้ม), pitch ต่ำ, PPO stable
- ✅ long/medium **เดินหน้าจริง** ตั้งแต่ warmup; anti-rocking fix ทำให้ทั้ง 3 เริ่มที่ baseline เป็นธรรม
- 🔄 กำลังดู: `eval/x_dist` climb หลัง λ ramp (โดยเฉพาะ short ที่ตอนนี้ game ไม่ได้แล้ว) — เช็ค ~100-150k
- ☐ ถัดไป: forward validate → command-conditioned fwd/turn/strafe (actor รับ command, disc คง 28-D,
  warm-start จาก forward policy) → เก็บ behavior dataset ต่อร่าง สำหรับ world model Stage 1
- IK ยังเก็บไว้เป็น clean-correspondence sanity baseline ถ้าต้องใช้

---

## 14. 🧭 อัปเดต 2026-08-06 — กลับมาใช้ IK forward-only + 4-leg preview

### 14.1 ทำไมพัก AMP/turn แล้วกลับมา forward-only ก่อน
หลัง train/evaluate AMP หลาย checkpoint พบว่า policy เดินได้ช่วงสั้น ๆ แต่ gait pattern ยังไม่คล้าย expert
พอจะใช้เป็น evidence ที่ convincing; frozen discriminator `g(s')` เป็น state-only จึงให้คะแนน posture/phase
รายเฟรมได้ แต่ไม่บังคับ temporal wave-gait transition ชัดพอ. สรุปตอนนี้: **AMP ยังเก็บไว้เป็น engineering
log/ทางเลือก แต่ไม่ใช่ main path สำหรับ proposal evidence ตอนนี้**.

ด้าน IK: audit ยืนยันว่า old forward IK logic ถูกกว่าและสะอาดกว่า:
- `data/ik_walk` = 9 clips (long/medium/short × ep521/625/926), 66 frames, forward-only, มีวิดีโอแล้ว
- `data/ik_v2` = 90 clips แต่ปน walk/old fake turn/stop; walk-only มี 54 clips (6 episodes × 3 repeats × 3 morphs)
- forward IK มี property สำคัญ: task-space foot path เดียวกัน แต่ joint action ต่างกันต่อ morphology
  (action RMS ≈ 0.407 rad, contact mismatch ข้าม morphology ≈ 0.09)

การทดลอง turn ล่าสุด:
- ใช้ expert curvy episode 472 แล้ว loop 3 รอบได้ turn ที่เห็นจริง (`data/ik_turn`, `data/ik_fair_96`)
- แต่ fixed side camera ทำให้ behavior แยกจาก **position/path-in-frame** ง่ายเกินไป: walk วิ่งขวา→ซ้าย,
  turn โค้งลง/ออกอีกตำแหน่ง. Position/path-only probe แยก walk vs turn ได้ 100%.
- ดังนั้น turn ยังเหมาะเป็น video evidence/debug แต่ **ไม่เหมาะเป็น training/evaluation หลักตอนนี้**.

**Decision:** Stage 1 เดินด้วย **forward-only IK** ก่อน เพื่อให้ pipeline เรียบและ thesis story ไม่โดน shortcut
จากกล้อง/trajectory.

### 14.2 Render-lock / encoder check บน `data/ik_walk`
รัน `scripts/render_lock_check.py --data data/ik_walk --out results/render_lock_ik_walk`

ผล:
- 9 clips, 594 frames, behavior = walk only, episodes = 521/625/926
- `silhouette(body) = +0.033`
- body decode = **0.951** (chance 0.333)
- behavior metric skipped เพราะมี behavior เดียว
- repeat-lock gate skipped เพราะ `data/ik_walk` ไม่มี repeats

Artifacts:
- `results/render_lock_ik_walk/emb.npz`
- `results/render_lock_ik_walk/umap.png`

Interpretation: raw V-JEPA2 `e_t` เห็น morphology ชัดจริง แต่ UMAP ไม่ใช่สามเกาะแยกแข็ง ๆ; เป็น walking
manifold เดียวที่ long/medium/short occupy คนละ band. นี่เหมาะเป็น baseline ก่อน train ITM: ต่อไป `z_t`
ควรลด morphology decodability ลง แต่ยังรักษา gait/contact/action information.

### 14.3 4-leg stick insect preview สำหรับ Stage 2
เพิ่ม preview scripts:
- `sim/render_leg_loss_preview.py` — static render/contact sheet
- `sim/render_leg_loss_walk.py` — rough walking video โดยใช้ six-leg open-loop gait เดิม แล้วทำ selected legs
  เป็น ghost/disabled เพื่อไม่ให้ scene script พัง

Artifacts:
- `results/leg_loss_preview_headcam/leg_loss_contact_sheet.png`
- `results/leg_loss_walk/grid_leg_loss_walk.mp4`
- `results/leg_loss_walk/six_leg_base.mp4`
- `results/leg_loss_walk/front_loss.mp4`
- `results/leg_loss_walk/middle_loss.mp4`
- `results/leg_loss_walk/hind_loss.mp4`

Rough walking preview (ยัง **ไม่ใช่** controller ที่ train สำหรับ 4-leg):
| variant | dx | dy | final_z | read |
|---|---:|---:|---:|---|
| six_leg_base | +3.080 | +0.607 | 0.234 | normal baseline |
| front_loss (remove FL/FR) | -0.543 | -0.359 | 0.025 | fails/falls |
| **middle_loss (remove ML/MR)** | **+2.237** | -1.353 | 0.155 | ugly but moves forward |
| hind_loss (remove HL/HR) | -0.223 | +1.943 | 0.145 | spins/drifts |

**Stage 2 candidate:** `middle_loss` is the best held-out 4-leg insect because it leaves front+hind legs,
which is quadruped-like, while retaining stick-insect appearance. This supports the slide framing:
train on **6-leg insect + B1 quadruped**, test on **4-leg insect** as compositional transfer.

### 14.4 Immediate next plan
1. Use `data/ik_walk` as the first forward-only Stage-1 dataset.
2. Train/validate the minimal ITM+FTM+MD pipeline on short+long, evaluate medium held-out.
3. Report raw `e_t` morphology baseline (`results/render_lock_ik_walk`) vs learned `z_t`.
4. Keep `data/ik_v2` walk-only as scale-up data after the small pipeline works.
5. Keep turn/4-leg as proposal/Stage-2 evidence, not the first training target.

---

## 15. 🔎 อัปเดต 2026-08-06 — ขยาย render-lock check เป็น 6 episodes + validate train(long+short)→test(medium)

### 15.1 ทำไมขยายจาก 3 เป็น 6 episodes
เอา UMAP จาก `data/ik_walk_3sec` (3 ep × 3 ร่าง, ไม่มี repeat = 594 เฟรม) ไปถาม AI ตัวอื่นดู — สรุปว่า
"เล็กเกินไปและสะอาดเกินไป" (3 episode อาจ overfit เป็น manifold เปราะบาง) แนะนำให้ใช้ทั้ง 6 forward-walk
episodes ที่มีอยู่ (144, 285, 521, 625, 926, 997) พร้อม repeat เพื่อเช็ค render-lock

**พบว่าข้อมูลที่แนะนำมีอยู่แล้ว** ใน `data/ik_all` — 6 episodes × 3 ร่าง × 3 repeats × 66 เฟรม = **3564 เฟรม
forward-walk** (บวก turn/stop ที่เก็บไว้ด้วยแต่ยังไม่ใช้ตามมติเดิม "forward-only ปลอดภัยกว่า") ไม่ต้องเก็บ
ข้อมูลใหม่ — สร้าง symlink dir `data/ik_walk_all6` (54 ไฟล์ forward-only) แล้วรัน
`scripts/render_lock_check.py --data data/ik_walk_all6` ใหม่

### 15.2 ผลลัพธ์ — PASS ที่ scale ใหญ่ขึ้น 6 เท่า
```
3564 เฟรม | body decode = 0.995 (chance 0.333) | silhouette(body) = +0.034
RENDER-LOCK GATE: mean repeat-decode = 0.393  vs chance 0.333  ->  PASS (threshold fail คือ >1.5x chance = 0.50)
```
- ทุกกลุ่ม (body×episode) ใกล้ chance ยกเว้น 3 กลุ่มที่สูงกว่านิดหน่อย: `long_521` (0.505), `medium_521`
  (0.566), `medium_144` (0.485) — episode 521 หลุดสูงสุดในทั้งสองร่าง น่าจะเป็นจุดที่ session การอัดมี
  variation มากกว่าอันอื่นเล็กน้อย ไม่ถึงกับ fail แต่บันทึกไว้เผื่อกลับมาเช็ค
- Artifacts: `results/ik/render_lock_ik_walk_all6/{emb.npz,umap.png,sample_frames.png}`

**สรุป**: การขยายข้อมูล 3→6 episodes (594→3564 เฟรม) เป็นการอัปเกรดจริง ไม่ใช่แค่ "ข้อมูลเยอะขึ้นเฉยๆ" —
ผ่าน render-lock gate เดิมได้สบาย ควรใช้ **`data/ik_walk_all6` เป็น canonical Stage-1 dataset** แทน
`data/ik_walk_3sec` ต่อจากนี้

### 15.3 🔑 พบว่า medium ไม่ได้อยู่ "ระหว่างกลาง" long กับ short ใน embedding space
เช็คด้วย centroid ของ raw `e_t` (1408-D, ไม่ใช่ 2D UMAP projection ที่บิดเบือนง่าย):
```
long <-> medium: 4.01     medium <-> short: 4.99     long <-> short: 6.51
medium project ไปที่ 40% ของเส้น long->short (สมเหตุสมผล คร่าวๆ)
แต่ perpendicular distance จากเส้นนั้น = 3.08  (ใกล้เคียงกับระยะ long<->short เองที่ 6.51!)
```
**ความหมาย**: ถ้า medium เป็นแค่ "ส่วนผสม" ของ long กับ short จริงๆ มันควรอยู่ใกล้เส้นตรงระหว่างสองจุดนั้น
(perpendicular distance ≈ 0) แต่มันไม่ใช่ — medium มีทิศทางของตัวเองใน embedding space ที่ชัดเจน

**ผลต่อแผน train(long+short)→test(medium)**: การทดสอบนี้ไม่ใช่ interpolation ง่ายๆ (กรณีที่ง่าย) แต่ใกล้เคียง
**mild extrapolation** มากกว่า (โมเดลไม่เคยเห็นทิศทางที่ medium อยู่) — ไม่ใช่เหตุผลที่จะไม่ทำการทดลองนี้ ถ้า
มันสำเร็จได้ทั้งที่เป็น extrapolation จะเป็นผลลัพธ์ที่ **แข็งแกร่งกว่า** interpolation ธรรมดา แต่ถ้าล้มเหลว ก็เป็น
finding จริง ไม่ใช่ bug — ควรตั้งความคาดหวังให้ตรงกับความยากของ task นี้

### 15.4 Protocol ที่ชัดเจนสำหรับ train(long+short) → test(medium) held-out
คลี่ความสับสนที่เกิดขึ้น (สำคัญพอที่จะบันทึกไว้กันงงซ้ำ):

**"ต้องใช้ controller ไหนขับ medium เพื่อป้อน e_t?"** — ไม่ต้องมี controller ใหม่ frame ของ medium ที่ต้องใช้
มีอยู่แล้วใน `data/ik_walk_all6` (สร้างจาก IK controller เดียวกับ long/short) "held-out" หมายถึงไม่เอา
`(frame, action)` คู่ของ medium ไปเทรนเท่านั้น ไม่ได้แปลว่าไม่มี frame ให้ใช้ตอน evaluate

**"medium ต้องเดินดีหรือเดินยุ่งๆ?"** — ต้องเป็น IK walk ที่สะอาด (คุณภาพเดียวกับ long/short) เพราะ IK
เป็น scripted/deterministic — ถ้า input เดินยุ่งเราจะแยกไม่ออกว่า `â_t` แย่เพราะ generalize ไม่ได้ หรือเพราะ
input เองกำกวม การทดสอบนี้ต้อง isolate เฉพาะความสามารถ generalize จริงๆ

**"เป็นการโกงไหม ถ้า frame มาจาก IK ตัวเดียวกับที่สร้าง ground truth action?"** — ไม่โกง เพราะ **โมเดลไม่เคย
เห็นตัวเลข action ของ medium เป็น input เลยไม่ว่าจุดไหน**:
- ITM รับแค่ `(e_t, e_{t+1})` = พิกเซลล้วนๆ
- Motion Decoder รับแค่ `z_t` + `x_t` (พิกเซลอีกเช่นกัน) → ทาย `â_t` ออกมา
- `a_t` จริง (ที่ IK สั่งจริง) ใช้แค่ "หลังจบ" เพื่อเทียบคะแนน `L_motion = ‖â_t − a_t‖²` เท่านั้น ไม่เคยเป็น input
- Video เป็นแค่พิกเซล ไม่รั่วตัวเลข joint angle ออกมา — ระบบ sim ให้ทั้งภาพและ ground-truth ตัวเลขพร้อมกัน
  เป็นเรื่องปกติของ synthetic data ไม่ใช่ circular reasoning
- สิ่งที่จะโกงจริงๆ คือถ้าป้อนตัวเลข joint angle เข้าโมเดลเป็น extra input หรือถ้า training เคยเห็นคู่
  `(frame, action)` ของ medium แม้แต่ครั้งเดียว — ทั้งสองอย่างนี้ไม่เกิดขึ้นในแผนนี้

**Metric แนะนำ**: `L_motion` (MSE joint-angle) เป็นตัวเลขหลัก + เสริมด้วย duty-factor/phase/forward-reach
comparison แบบเดียวกับที่ `gait_report.py` ใช้เทียบ AMP กับ expert (ให้ภาพว่า "ดูเหมือนเดินจริงไหม" ซึ่ง
ให้ความหมายมากกว่า raw joint-MSE อย่างเดียว) + ครึ่งที่สองของ two-sided validation (`direction_plan.md`
Step 1.5): morphology probe บน `z_t` ของ medium ควรใกล้ chance ด้วย ไม่งั้น `z_t` แอบจำร่างกายอยู่

---

## 16. 🧪 อัปเดต 2026-08-07 — เทรน world model จริง 3 รอบ + เจอ data bug ใหญ่ (framing)

โค้ดอยู่ที่ `wm/` (ITM + FTM + Motion Decoder ตาม LAC-WM) · ผลทั้งหมด + README อยู่ที่ `results/wm/`

### 16.1 สามรอบที่เทรน

ตั้งชื่อแบบ `stage1_<episodes>ep_<frames>` เทรนบน long+short กัน medium ไว้ทดสอบ

| run | episodes | frames | steps |
|---|---|---|---|
| `stage1_6ep_clipped` | 6 | ทั้งหมด 0-65 | 9,750 |
| `stage1_100ep_clipped` | 100 | ทั้งหมด 0-65 | 30,880 |
| `stage1_100ep_clean` | 100 | 45-65 เท่านั้น | 9,500 |

### 16.2 🔴 พบ data bug: หุ่นถูกตัดขอบภาพ 67% ของทุก clip

กล้อง fixed ถูกวางอ้างอิงกับ **ตำแหน่งเริ่มต้น**ของหุ่น + `RUNWAY_AIM=0.75` → หุ่นเริ่มที่**ขอบขวาแบบโดนตัด**
แล้วค่อยเดินเข้ามาในเฟรม วัดได้:

| ร่าง | เฟรมที่โดนตัด |
|---|---|
| long | 47/66 (70%) |
| medium | 44/66 (66%) |
| short | 36/66 (58%) |

**อันตรายเพราะไม่เท่ากันต่อร่าง** → morphology decodability ที่วัดได้ 99% อาจกำลังอ่าน "การจัดเฟรม" ไม่ใช่รูปร่างจริง
และหุ่นกินพื้นที่แค่ ~1% ของพิกเซล (patch 16×16 ของ V-JEPA2 แตะหุ่นแค่ ~10% ของ patch ทั้งหมด)

### 16.3 ผลการทดลอง (motion MSE บน medium ที่ไม่เคยเห็น)

**หน่วย:** action ถูก standardise ต่อข้อต่อ (หารด้วย std ของชุดเทรน, std จริง 0.174-0.595 rad
เฉลี่ย 0.389 rad) → MSE ไม่มีหน่วย · **1.0 = เดาค่าเฉลี่ย (ไม่มีทักษะ)** · คอลัมน์ deg คือ
RMSE แปลงกลับเป็นองศาต่อข้อต่อ

| run | frames | steps | with z | zero z | shuffled z | **องศา/ข้อต่อ** |
|---|---|---|---|---|---|---|
| `stage1_6ep_clipped` | ทั้งหมด | 9,750 | **0.166** | 0.848 | 0.975 | **9.6°** |
| `stage1_100ep_clipped` | ทั้งหมด | 30,880 | **0.422** | 0.470 | 1.197 | **15.3°** |
| `stage1_100ep_clean` | 45-65 | 9,500 | **0.179** | 1.675 | 1.071 | **10.0°** |

**การถ่ายทอดไปร่างใหม่ขึ้นกับ "จำนวน steps" ไม่ใช่ "การจัดเฟรม"**

- clipping แยกดี/แย่ไม่ได้ — `stage1_6ep_clipped` เป็น clipped แต่ผลดี (0.166 = 9.6°)
- steps แยกได้เป๊ะ — สอง run ที่ผลดีอยู่ที่ ~9.5k steps เท่ากัน, run ที่ผลแย่อยู่ที่ 30.9k

⚠️ **ยังไม่เคยเทรน clean frames เกิน 9,500 steps** → `clean vs clipped` ถูก confound กับ
`9.5k vs 30.9k steps` แบบสมบูรณ์ ยังสรุปไม่ได้ว่า framing มีผลต่อ transfer หรือไม่

**สิ่งที่ framing มีผลจริง (วัดที่ steps เท่ากัน):** `val_motion` = 0.0068 (clean @1,425 steps)
เทียบ 0.023-0.027 (clipped @~1,550 steps) → **เรียนเร็วกว่า 3.4 เท่าต่อ step** เป็นเรื่อง
learning efficiency ไม่ใช่ cross-body transfer

**การทดลองที่ชี้ขาด (ยังไม่ได้ทำ):** เทรน clean frames ถึง ~31k steps (~65 epochs) แล้วดู
`heldout/motion` — ถ้าเสื่อมลงเป็น ~0.42 (15°) คือ over-specialization จากการเทรนนาน
ถ้าไม่เสื่อมคือ framing มีส่วนจริง

### 16.4 🔑 ผลที่ยังยืนอยู่: validation แบบมาตรฐานมองไม่เห็นการล้มเหลวข้ามร่าง

ใน `stage1_100ep_clean` จาก epoch 2 → 20:
```
val_motion (episode ที่ไม่เคยเห็น, ร่างเดิม):  0.0122 → 0.0013   ดีขึ้น 9.3 เท่า
heldout    (ร่างที่ไม่เคยเห็น):                0.1447 → 0.1806   ไม่ดีขึ้นเลย
```
ดีขึ้น 9.3 เท่าในกลุ่มเดิม แต่**ร่างใหม่ไม่ได้อะไรเลย** — เพราะ validation แบ่งด้วย episode ของ**ร่างเดิม**
ร่างใหม่คือ **คนละ distribution** ไม่ใช่ held-out sample ของอันเดิม
→ แก้แล้ว: `wm/train.py` วัด `heldout/motion` ทุก epoch + เซฟ checkpoint เป็นระยะ (`wm/sweep_checkpoints.py` re-score ทีหลังด้วย sample เยอะกว่า 10 เท่า)

### 16.5 morphology ยังอ่านออก ~99% ทุกรอบ (ผลที่ทนทาน)

| run | decode จาก e | decode จาก z | silhouette e → z |
|---|---|---|---|
| `stage1_6ep_clipped` | 0.9969 | 0.9855 | 0.0335 → 0.0148 |
| `stage1_100ep_clipped` | 0.9989 | 0.9963 | 0.0220 → 0.0283 |
| `stage1_100ep_clean` | 0.9987 | 0.9997 | 0.0640 → 0.0403 |

**ข้าม 3 รอบที่ต่างกันหมด (6 vs 100 episodes, clipped vs clean, 9.7k vs 31k steps) → z ยังบอกได้ ~99% ว่าเป็นร่างไหน**
"ความครอบงำ" (silhouette/variance) ลดลง แต่ "การมีอยู่" ไม่ลด

**สาเหตุเชิงโครงสร้าง:** ไม่มีเทอมไหนใน loss ลบ morphology ออกจาก `z` — ทั้ง `L_recon` และ `L_motion`
ต่างก็รับ `x_t` ซึ่งมีข้อมูลร่างกายอยู่แล้ว จึงไม่มีอะไรลงโทษ `z` ที่พกมันมาด้วย
(cross-augmentation แก้ shortcut คนละเรื่อง) **น่าจะเป็นจริงกับ LAC-WM ด้วย เขาแค่ไม่เคยวัด**

### 16.6 💡 invariance ≠ transferability (ข้อค้นพบที่ใหม่ที่สุด)

`z` **รู้ว่าเป็นร่างไหน 99%** แต่ก็ **ถ่ายทอดไปร่างใหม่ได้ดี** (0.18 เทียบ 1.67 ตอนตัด z ออก) พร้อมกัน

วงการสมมติว่า *ต้อง* invariant ก่อนถึงจะ transfer ได้ — จึงไล่ทำ "unified/embodiment-agnostic latent space"
และใช้ภาพ UMAP ที่ cluster ทับกันเป็นหลักฐาน **แต่ข้อมูลเราบอกว่าไม่จำเป็น**

⚠️ **ยังเป็นสมมติฐาน ไม่ใช่ข้อสรุป** — ทดสอบแค่ 3 ร่างที่ใช้ 18-D ร่วมกัน และ**ยังไม่เคยลองบังคับให้ invariant**
การทดลองชี้ขาดคือ `--z_dim` เล็กลง / adversarial head แล้วดูว่า transfer เปลี่ยนไหม (ทุกผลลัพธ์ตีพิมพ์ได้)

### 16.7 🔧 แก้ framing แล้ว (2 flag ใหม่ใน `sim/collect_ik.py`)

- **`--cam_dx -0.6`** — เลื่อนกล้อง ลด runway aim 0.75 → 0.15
- **`--spawn 0 0`** — respawn หุ่นที่**กลางพื้น** (พื้นแค่ 5×5 m, เดิมหุ่นอยู่ห่างขอบแค่ 0.95 m)

| | ก่อน | หลัง |
|---|---|---|
| เฟรมโดนตัด | 47/44/36 จาก 66 | **0/66 ทุกร่าง** |
| ขอบพื้นในเฟรม | 2.4% | **0%** |
| margin ซ้าย/ขวา | 0.055 / 0.152 | **0.102 / 0.109** |
| transition ใช้ได้ | 3,800 (เฉพาะเฟรม 45-65) | **13,000** |

**ลำดับสำคัญ:** ต้องอ่าน `off_xy` (offset กล้องเทียบหุ่นตามที่ฉากออกแบบไว้) **ก่อน** respawn เสมอ
ถ้าอ่านหลังย้ายหุ่น จะได้ offset ที่วัดจากตำแหน่งใหม่ → กล้องไม่ตามหุ่นไป หุ่นเดินออกนอกเฟรม

### 16.8 🔴 B1 ไม่ได้ render-lock กับตั๊กแตนจริง (สำคัญมากกับ Stage 2)

การใช้ฉาก พื้น และค่ากล้องชุดเดียวกัน **ไม่เพียงพอ** ที่จะเรียกว่า render-lock

กล้องถูกวางอ้างอิงกับตำแหน่งเริ่มของ**แต่ละหุ่นเอง** และ B1 ถูก replay ที่พิกัดดิบจาก MuJoCo
→ ทั้งสองร่าง **ยืนคนละที่บนพื้น 5×5 m** จึงเห็นพื้นคนละบริเวณ

| เทียบ | ความต่างของพื้นหลัง |
|---|---|
| ตั๊กแตน long vs short | **0.29** / 255 |
| **ตั๊กแตน vs B1** | **8.3 เฉลี่ย, 33 สูงสุด — 27% ของพิกเซลต่างกันเกิน 10 ระดับ** |

→ ถ้าไม่แก้ "embodiment decodable จาก z" จะกำลังวัด**พื้นหลัง** ไม่ใช่ embodiment (และ `e_t` อ่าน morphology
ได้ 99% แม้ระหว่าง render ที่แทบเหมือนกัน — 27% นี่อ่านง่ายมาก) **Stage 2 จะได้ผลที่ดูเหมือนสำเร็จแต่ผิด**

**ยังพบอีก 2 อย่าง:**
- B1 เดินไกล 1.31–3.06 m แต่กล้องเห็นแค่ 2.11 m → **เดินออกนอกเฟรม** (ที่ vx0.4 ขึ้นไป)
- ความเร็วจริงต่ำกว่าที่ label ไว้มาก → **`fwd_vx0.4` = 0.164 m/s ใกล้ตั๊กแตน long (0.174 m/s) ที่สุด**

แก้แล้วใน `sim/render_b1_replay.py`: เพิ่ม `--spawn`, `--cam_dx/dy`, `--travel` และเปลี่ยนกล้องให้ยึด
**ตำแหน่งเริ่ม** แบบเดียวกับตั๊กแตน (เดิมเล็งที่จุดกึ่งกลางเส้นทาง)

### 16.9 คำสั่งที่ใช้เก็บข้อมูลรอบใหม่

```
python3 sim/collect_ik.py --port 23000 --episodes <100 eps> --repeats 1 --scale 0.5 \
    --travel 0.8 --cam_dx -0.6 --spawn 0 0 --out data/ik_walk_100_framed

python3 sim/render_b1_replay.py --scene sim/env/b1_flat.ttt --traj data/b1_traj/fwd_vx0.4.npz \
    --spawn 0 0 --cam_dx -0.6 --travel 0.8 --out data/b1_framed
```

### 16.10 สถานะ / ต้องทำต่อ
- ☐ เก็บ `data/ik_walk_100_framed` (~55 นาที) + re-render B1 ให้ตรงกัน
- ☐ เทรนใหม่บนข้อมูลที่ framing ถูก (ครั้งแรกที่ได้ทั้ง**เฟรมสะอาด**และ**gait ครบรอบ**)
- ☐ ทดลอง `--z_dim` / adversarial → ชี้ขาดเรื่อง invariance vs transferability
- ☐ Stage 2: `wm/train.py --sources hexapod=... b1=...` (โค้ดพร้อมแล้ว, `wm/evaluate.py` ยังไม่มี metric ข้าม embodiment)

---

## 17. ผลการทดลอง Stage 1 บนข้อมูล framing ถูก (2026-08-08)

เทรนบน `data/ik_walk_100_framed` (100 episodes x 3 ขา, 19,800 เฟรม, 0 เฟรมถูกตัดขอบ) ผลทั้งหมดพร้อม
คำสั่ง reproduce อยู่ใน **[FINDINGS.md](FINDINGS.md)** ส่วนนี้บันทึกเฉพาะลำดับเหตุการณ์

### 17.1 สองรันคู่ขนาน แล้วพบว่า variance สูงกว่าที่วัดได้

`stage1_100ep_framed_runA` (2080 Ti) กับ `runB` (4070 Ti Super) ใช้ config เดียวกันทุกตัวอักษร รวมทั้ง
`seed: 0` ต่างกันแค่ GPU ผล: reconstruction ตรงกัน 0.3% แต่**ขาที่ไม่เคยเห็นต่างกันได้ถึง 2.1 เท่า**
(FINDINGS F12) → ตัวเลข held-out จาก run เดียวตีความไม่ได้ ต้องมี error bar

### 17.2 per-joint breakdown เปลี่ยนข้อสรุปทั้งหมด

ตัวเลขรวม 0.208 ดูดี แต่แยกตามชนิดข้อต่อแล้ว TC ได้ 0.006 ส่วน CF ได้ 0.382 ซึ่ง**แย่กว่าการทายค่าคงที่
3 เท่า** (F8) และ TC ของสามขาต่างกันแค่ 0.58–1.17 องศา คือแทบไม่มีอะไรให้ถ่ายทอด → เพิ่ม
`motion_mse_per_joint_type` เข้า `wm/evaluate.py`

### 17.3 พบว่า standardisation กดน้ำหนักข้อต่อที่สำคัญที่สุด

หาร std จากข้อมูลรวมทุก body ทำให้ระยะห่างของท่าทางระหว่าง body ถูกนับเป็นแอมพลิจูด CF/FT จึงได้
gradient แค่ 0.12/0.11 เท่าของ TC (F9) → เพิ่ม `within_body_std` ใน `wm/config.py` เปิดเป็นค่าเริ่มต้น
แก้แล้ว **RMSE ไม่ดีขึ้น** แต่พฤติกรรมบนแกน morphology เปลี่ยนชัด (F5)

### 17.4 วินิจฉัยสาเหตุจนถึงราก

ไล่สัญญาณทีละขั้น: `e_t` วางขา medium ไว้ที่ตำแหน่ง 0.465 บนแกน long→short (ใกล้ 0.5 ที่ควรเป็นตาม
สเกลขา) → `z` เหลือ 0.301 → output เหลือ 0.15 ขณะที่คำตอบถูกคือ 0.30–0.36

**ข้อมูลอยู่ครบในภาพ แต่หายที่ decoder** เพราะ motion loss เห็น body แค่ 2 ตัว = 2 จุด ซึ่งนิยามเส้นโค้ง
ไม่ได้ (F5) ตัดสมมติฐานอื่นทิ้งหมด: ไม่ใช่ภาพไม่พอ ไม่ใช่น้ำหนัก loss ไม่ใช่ architecture

### 17.5 baseline ที่ไม่ต้องเรียนรู้ชนะโมเดล

เพราะ IK ใช้รอยเท้า Cartesian ร่วมกันทุกขา คำสั่งข้อต่อของสามขาจึงต่างกัน **92–99% เป็น offset ล้วน**
(F7) การเฉลี่ยสอง training body จึงได้ 6.68 องศา ขณะที่โมเดลได้ 10.95 (F6)

fold 2 (`fold_short`, กัน `short` ไว้) ยืนยันซ้ำ: โมเดลได้ 7.00 เท่ากับ "ก๊อป ground truth ของขาที่ใกล้
ที่สุด" (6.96) และแพ้ linear extrapolation (1.91) 3.7 เท่า heldout นิ่งตลอด 28 epochs ขณะ val ดีขึ้น 8 เท่า

### 17.6 ผลบวกที่ได้จริง

บนขาที่เคยเห็น โมเดลระบุ body จากพิกเซลแล้วออก offset ผิดแค่ 0.03–0.06 องศา ทั้งที่สองขามีท่าทาง
ต่างกัน 33.8–50.1 องศา โดยไม่เคยถูกบอก morphology เลย (F1)

### 17.7 สถานะ / ต้องทำต่อ
- ☑ เก็บ `data/ik_walk_100_framed`
- ☑ เทรนบนข้อมูล framing ถูก (runA, runB, fix_norm, fold_short)
- ☑ วินิจฉัยสาเหตุที่ transfer ไม่สำเร็จ → [FINDINGS.md](FINDINGS.md)
- ☐ ขยาย `sim/make_leg_morphology.py` ให้สเกล coxa/femur/tibia แยกกัน
- ☐ เก็บข้อมูลใหม่: ~30 episodes x 6–8 bodies (แทน 100 x 3)
- ☐ re-render B1 ให้ framing ตรงกับตั๊กแตน
- ☐ Stage 2: `wm/evaluate.py` ยังไม่มี metric ข้าม embodiment

---

## 18. Stage 1 รอบสอง: morphology หลายมิติ และการหาสาเหตุเชิงกลไก (2026-08-08)

ผลทั้งหมดพร้อมตัวเลขและคำสั่ง reproduce อยู่ใน **[FINDINGS.md](FINDINGS.md)** F15–F19 ส่วนนี้บันทึกลำดับเหตุการณ์

### 18.1 ขยาย morphology จาก 1 มิติเป็น 3

`sim/make_leg_morphology.py` เดิมรับ `--factor` ตัวเดียว สเกลทั้ง coxa/femur/tibia เท่ากัน → หุ่นเรียงกันเป็นเส้นตรง
หุ่นที่กันไว้จึงเป็นแค่จุดกึ่งกลางของสองตัว ซึ่งทดสอบได้แค่การ interpolate บนเส้น

แก้ให้รับ `--coxa --femur --tibia` แยกกัน และ `sim/collect_ik.py` รับ `--morphs NAME=SCENE` เก็บหุ่นกี่ตัวก็ได้

สร้าง 9 หุ่น เก็บ `data/ik_walk_8body` 30 clips ต่อหุ่น หุ่นทดสอบ `c08f09t09` (0.8, 0.9, 0.9) ตรวจด้วย
linear programming ว่าอยู่ใน convex hull ของหุ่นเทรน และห่างจากเส้นตรงระหว่างคู่หุ่นทุกคู่ 0.082
จึงเป็นการทดสอบ **composition** ไม่ใช่ interpolation

### 18.2 ตัดหุ่น 2 ตัวที่เดินไม่ได้

`c06f06t10` และ `c10f06t10` (femur 0.6 + tibia 1.0 ทั้งคู่) เดินสำเร็จแค่ 20–21 จาก 30 clips
หัวทรุดถึง 0.027 m (ปกติ 0.111 m) และชนขอบภาพ 3–11%

ไม่ใช่เพราะ femur สั้น — `c10f06t06` มี femur 0.6 เหมือนกันแต่ tibia 0.6 ด้วย และเดินดีที่สุดในชุด (0.669 m)
**ปัญหาคือสัดส่วนที่ไม่สมดุลระหว่างสองท่อน**

ตรวจแล้วว่าตัดสองตัวนี้ออกได้โดย `c08f09t09` ยังถูกขนาบด้วย 5 ตัวที่เหลือ (น้ำหนัก 0.25/0.50/0.25)
เก็บ `c06f06t06` เพิ่มเป็นหุ่นทดสอบนอกช่วง สุดท้ายได้ 7 หุ่นสะอาด **ชนขอบ 0.0% ทุกตัว**

### 18.3 พบว่า morphology space เป็น 2 มิติ ไม่ใช่ 3

ย่อ coxa 40% เปลี่ยนคำสั่งข้อต่อแค่ **0.73 องศา** ส่วน tibia เปลี่ยน **28.63 องศา**
SVD ให้ 82.4% (tibia) + 17.5% (femur) + 0.0% ที่เหลือ

coxa เป็นท่อนสั้นที่สุดและติดลำตัว ย่อแล้วตำแหน่งเท้าแทบไม่ขยับ IK จึงไม่ต้องชดเชย
**ตัดสินใจยอมรับว่าเป็น 2 มิติและเขียนกำกับตามจริง** แทนการเพิ่มแกนที่ 3

ผลข้างเคียงที่สำคัญ: การผสมเชิงเส้น 2 มิติทายหุ่นที่ไม่เคยเห็นได้ **0.20 องศา** ซึ่งเป็นเพดานบนของงานนี้

### 18.4 ผลการเทรน — เพิ่มหุ่นช่วยจริง

| | 2 หุ่น | 5 หุ่น |
|---|---|---|
| heldout | 11.04 องศา | **3.57** |
| ชนะ baseline เฉลี่ยหุ่นเทรน | แพ้ (6.7) | **ชนะ (11.48)** |
| z-ablation gap | 3–4× | **10–37×** |

`m3d_outside` (กัน `c06f06t06` ที่อยู่นอกช่วง) เป็น control ที่จับคู่สมบูรณ์ — ต่างจาก `m3d_bracketed`
แค่หุ่นที่กันไว้ ผลห่างกัน **10–30 เท่า** ยืนยัน coverage ด้วยข้อมูลชุดใหม่ทั้งหมด

และการเพิ่มหุ่นช่วยแม้กับหุ่นนอกช่วง: `fold_short` (2 หุ่น) นิ่งที่ 1.02× ตลอด 41 epochs
ส่วน `m3d_outside` (5 หุ่น) ดีขึ้น 0.78×

### 18.5 หาสาเหตุจนถึงกลไก

สร้าง `scripts/swap_pathway.py` สลับ input ของ decoder ระหว่างสองหุ่น (ทำได้เพราะทุกหุ่นเดินจาก
episode เดียวกัน จังหวะจึงตรงกัน)

**ให้ภาพหุ่น A + `z` หุ่น B → decoder ตอบเป็นคำสั่งของหุ่น B ผิดแค่ 3.48 องศา** ทั้งที่สองหุ่นต่างกัน 28.63
ภาพที่เห็นตรงหน้าไม่มีผลเลย และแนวโน้มนี้แรงขึ้นเมื่อเทรนนาน (ห่าง 4.6 → 12.1 องศา ระหว่าง epoch 6 → 8)

แยก variance ของ `z`: จังหวะ 64.1% / ร่างกาย 11.1% / ที่เหลือ 24.8%
`z` ทำหน้าที่ถูกตามดีไซน์ แต่ probe ยังถอดร่างกายได้ 0.724 (สุ่ม 0.200)

**decoder เลือกใช้ร่างกาย 11% ที่อยู่ใน `z` แทนที่จะอ่านจากภาพซึ่งมีข้อมูลครบ**
เพราะ lookup 5 โค้ดง่ายกว่าการอ่านเรขาคณิตจาก 256×1408 tokens — และ lookup ใช้กับหุ่นที่ไม่มีโค้ดไม่ได้

`scripts/morphology_mix.py` ยืนยันจากฝั่ง output: น้ำหนัก **0.883 กระจุกที่หุ่นเดียว**
และความยาวขาที่โมเดลอนุมานคือ (0.980, 0.975, 0.973) ทั้งที่ของจริง (0.80, 0.90, 0.90)

### 18.6 สองทางแก้ที่ทดสอบแล้วไม่ได้ผล

| ลอง | ผล |
|---|---|
| `within_body_std` แก้การถ่วงน้ำหนัก loss | ไม่ช่วย |
| `--md_head linear` ลดขนาดหัวท้าย decoder | **แย่กว่า mlp 1.4–2.1 เท่า ทั้ง 10 epochs** |

`md_head linear` ตัดออกแค่ 5% ของ decoder และไม่ได้แตะ cross-attention (3.15M) ซึ่งเป็นตัวที่สงสัย

### 18.7 การทดลองที่กำลังรัน: ตัดโค้ดร่างกายออกจาก `z`

เพิ่ม `wm/models/adversary.py` — gradient reversal head ที่ผลักร่างกายออกจาก `z` (`--lambda_adv`)
พร้อม `MorphProbe` ที่อ่าน `z.detach()` เป็นเครื่องวัดที่ไม่แทรกแซง (ตรวจแล้ว gradient ต่างกัน 0.00e+00)

Smoke test บนชุดจิ๋ว 975 pairs เทียบกับ control:

| ep 8 | control | adversary |
|---|---|---|
| probe | 0.592 | **0.002** |
| heldout | 0.1206 | 0.1086 |
| val motion | 0.0936 | 0.0997 |

กลไกทำงาน แต่ probe ต่ำกว่าสุ่ม (0.200) มาก แปลว่ายัง "หมุน" ไม่ได้ "ลบ" — เพิ่ม
`adv_warmup_epochs = 5` และ `heldout/motion_zero_x` เพื่อวัดว่า decoder หันไปพึ่งภาพจริงไหม

### 18.8 สถานะ / ต้องทำต่อ
- ☑ `data/ik_walk_8body` 7 หุ่นสะอาด
- ☑ `data/b1_framed` render-locked (แก้ FOV กล้อง 24° → 15°, พื้นหลังต่าง 5.03 → 0.52/255)
- ☑ B1 trajectory 2 policy × 7 ความเร็ว ครอบคลุมช่วงความเร็วตั๊กแตน
- ☐ `m3d_adv01` — ผลตัดสินว่าปิดทางลัดแล้ว decoder หันไปอ่านภาพไหม
- ☐ Stage 2 — รอผล `m3d` ตัดสินว่าจับคู่ B1 กับตั๊กแตนกี่ตัว

### 18.9 ผลการทดลอง adversarial และการตรวจ encoder (สรุปรอบสุดท้ายของ Stage 1)

**adversarial ทำงานตามที่ออกแบบ แต่ไม่ช่วย** (`m3d_adv01` เทียบ `m3d_bracketed` เฉลี่ย 10 epochs)

| | control | adversarial |
|---|---|---|
| heldout | **0.097** | 0.118 (แย่ลง 1.21×) |
| z-gap | 23.4× | 5.1× |
| **x-gap** | 10.7× | **21.8×** |

decoder ย้ายจาก `z` ไปพึ่งภาพจริง และย้ายมากขึ้นเรื่อย ๆ (x-gap 11.3 → 31.9 ตลอด 10 epochs
5 epochs สุดท้ายห่างจาก control 2.46 เท่า) **แต่ transfer ไม่เคยดีขึ้นเลย**
probe นิ่งที่ 0.440 ตั้งแต่ epoch 7 (สุ่ม 0.200) คือ ITM กับ adversary เข้าสมดุล รันต่อไม่เปลี่ยนอะไร

**การตรวจ encoder ตอบว่าทำไม** — fit regression จาก mean-pooled `e_t` ไปหาสเกลของขาทั้งสามท่อน
บนหุ่นเทรน 5 ตัว แล้วทดสอบกับหุ่นที่กันไว้:

| | coxa | femur | tibia |
|---|---|---|---|
| `c08f09t09` ทำนายได้ | **0.850** | **0.939** | **0.898** |
| ค่าจริง | 0.80 | 0.90 | 0.90 |

**ridge regression 4,227 พารามิเตอร์ ทำนายความยาวขาของหุ่นที่ไม่เคยเห็นได้ผิดแค่ 0.05 / 0.04 / 0.002**
ขณะที่ Motion Decoder 5.2 ล้านพารามิเตอร์บอกว่าหุ่นตัวนี้คือ (0.98, 0.98, 0.97)

**ข้อมูลอยู่ในภาพ ถอดได้ด้วยเส้นตรง generalize ได้ แต่ decoder ไม่ใช้**

สาเหตุเชิงสถาปัตยกรรม: probe เห็นค่าเฉลี่ยของ 256 patch tokens ตรง ๆ ส่วน decoder เห็นผ่าน
cross-attention ที่มี `z` เป็น query จึงดึงมาได้เฉพาะสิ่งที่ `z` ถามหา — และ `z` เป็นจังหวะการเดิน 64%
**ข้อมูลรูปร่างอยู่ใน token แต่ไม่เคยถูกถาม**

อธิบายได้ว่าทำไม adversarial ถึงไม่ช่วย: เราบังคับให้ decoder ไปใช้ช่องทางที่มันไม่มีกลไกจะอ่าน

### 18.10 สรุป Stage 1 ทั้งหมด

| สมมติฐาน | สถานะ |
|---|---|
| ข้อมูลไม่อยู่ในภาพ | ตัดออก (F20) |
| `z` มีร่างกายให้ lookup | ตัดออก (F21 เอาออกแล้วไม่ช่วย) |
| น้ำหนัก loss ผิด | ตัดออก (F9) |
| decoder ใหญ่เกินไป | ตัดออก (F4b ลดแล้วแย่ลง) |
| **coverage ไม่พอ** | **ยืนยัน — ทางเดียวที่ได้ผล (F16, F17)** |
| **decoder เข้าถึงภาพผ่านช่องทางที่ผิด** | **เหลืออันนี้ → Q6** |

ผลทั้งหมดพร้อมตัวเลขอยู่ใน [FINDINGS.md](FINDINGS.md) F20–F21 และ [OPEN_QUESTION.md](OPEN_QUESTION.md) Q5 (ปิดแล้ว) / Q6 (เปิดใหม่)

### 18.11 Q6 — ให้ decoder เข้าถึงภาพโดยตรง (2026-08-09)

เพิ่ม `--md_head pooled` ใน `wm/models/motion_decoder.py`: เอา mean ของ 256 patch tokens
ผ่าน `pooled_proj` แล้วบวกเข้า action เป็น residual โดย `offset` init เป็นศูนย์
(ตรวจแล้วว่า output เริ่มต้นเท่ากับ `mlp` ทุกทศนิยม ความต่างใด ๆ คือทาง pooled ถูกใช้จริง)

**ดีไซน์แรกใช้ concat แล้วล้มเหลว** — `fuse` กดทาง pooled ทิ้ง x-gap ตกจาก 12.5 เหลือ 7.1
เปลี่ยนเป็น residual บน output เพราะกดทิ้งไม่ได้

**ผล 11 epochs เทียบ `m3d_bracketed`:**

| | control | pooled |
|---|---|---|
| heldout | 0.098 | 0.099 (เท่ากัน) |
| z-gap | 21.1× | 29.6× |
| **x-gap** | **10.9×** | **1.4×** |

**ให้ทางเข้าถึงภาพที่ตรงที่สุดแล้ว โมเดลใช้ภาพน้อยลง 7.6 เท่า** และไปพึ่ง `z` แทน ผลเท่าเดิม

วัด residual ตรง ๆ บน checkpoint ชุดเต็ม epoch 6: ขนาดเหลือ **0.24–0.28 องศา**
(smoke ได้ 1.5–1.9) between/within = **0.32** (smoke 0.67) — ยิ่งมีข้อมูลมาก
โมเดลยิ่งปิดทางที่เราเปิดให้ และ residual คิดเป็น **0.9%** ของ 28.6 องศาที่ต้องอธิบาย

swap test ที่ epoch 6 ตอกย้ำ: ให้ภาพหุ่น B + `z` หุ่น A → pooled ตอบเป็นของ A ผิดแค่ **3.42 องศา**
(control ผิด 16.52) คือ **pooled ตาม `z` แม่นกว่า control** ภาพที่ต่างกัน 28.63 องศา
เปลี่ยนคำตอบแค่ 12%

### 18.12 สรุป: ตัดสมมติฐานครบแล้ว เหลือ objective

| ทดลอง | ผล |
|---|---|
| แก้ normalization (F9) | ไม่ช่วย |
| ลดขนาด decoder head (F4b) | แย่ลง 1.4–2.1× |
| ตัดโค้ดร่างกายออกจาก `z` (F21) | ใช้ภาพมากขึ้น 2× แต่ transfer แย่ลง 1.21× |
| **ให้ทางเข้าถึงภาพตรง ๆ (F22)** | **ใช้ภาพน้อยลง 7.6× transfer เท่าเดิม** |
| **เพิ่มจำนวนหุ่น (F16, F17)** | **ดีขึ้น 3.1× — ทางเดียวที่ได้ผล** |

capacity, การเข้าถึง, และเนื้อหาใน `z` ถูกตัดออกหมด **เหลือ objective**

`L_motion` ขอแค่ทายคำสั่งข้อต่อให้ถูกบนหุ่นที่เห็นตอนเทรน การจำ 5 โค้ดใน `z` ทำได้ถูกกว่า
การอ่านเรขาคณิตจากพิกเซลเสมอ ไม่ว่าจะเปิดทางให้สะดวกแค่ไหน → **Q7** ใน OPEN_QUESTION.md

### 18.13 ตรวจ FTM — พบรากที่ลึกกว่าทุกอย่าง (2026-08-09)

เราไม่เคยตรวจ FTM เลยตลอดโปรเจกต์ `L_recon` เป็น MSE บน embedding ที่ไม่ normalize
จึงไม่มีใครรู้ว่า 1.6 ดีหรือแย่ ตรวจด้วย baseline สองตัว (ทายว่าภาพไม่เปลี่ยน / ทายค่าเฉลี่ย)
และ ablate `z` ที่ horizon 1, 2, 5, 10 บนทั้งรัน 5 หุ่นและ 2 หุ่น

| horizon | FTM | copy e_t | z zeroed | **z ช่วย** |
|---|---|---|---|---|
| 1 | 1.452 | 2.116 | 1.549 | **1.07×** |
| 2 | 1.778 | 2.756 | 1.910 | 1.07× |
| 5 | 2.494 | 3.646 | 2.620 | 1.05× |
| 10 | 3.187 | 4.431 | 3.294 | **1.03×** |

**FTM ทำงานได้** (ดีกว่า copy 39–55%) **แต่ `z` ช่วยแค่ 3–7%** เทียบกับ Motion Decoder ที่ z-gap 20–37×

- เป็นแบบนี้ทั้งสองรัน (2 หุ่นได้ 1.04×) ไม่ใช่ผลของการเพิ่มหุ่น
- horizon ไกลขึ้น**ไม่ช่วย** กลับลดเหลือ 1.03× เพราะ FTM ถอยไปทายค่ากลาง
- `lambda_recon = lambda_motion = 1.0` แต่ recon = 1.6 ส่วน motion = 0.01
  → **99% ของ gradient ไปที่ loss ที่ไม่ต้องการ `z`**

**ทฤษฎี LAC-WM คือ `z` เป็น action เพราะ `L_recon` บังคับ — ในระบบเรา `L_recon` ไม่ได้บังคับอะไรเลย**
`z` จึงถูกกำหนดรูปร่างโดย `L_motion` ด้วยงบ gradient 1% และ `L_motion` ยอมให้จำได้

นี่อยู่ใต้ทุกอย่างที่เจอมา: decoder หยิบโค้ดร่างกายจาก `z` เพราะไม่มีอะไรหล่อหลอม `z` ให้เป็นอย่างอื่น
และการแก้ decoder ทุกวิธีจึงไม่ช่วย

**ทำได้ทันทีไม่ต้องแก้โค้ด:** `--lambda_motion 100` หรือ `--lambda_recon 0`

### 18.14 `lambda_cross` — การทดลองแรกที่แก้ปัญหาได้จริง (2026-08-09)

เพิ่ม 1 term ใน loss: **ถอดคำสั่งของหุ่น B จาก latent ของหุ่น A โดยให้ decoder เห็นภาพหุ่น B**
ทำได้เพราะทุกหุ่นเดิน expert episode เดียวกัน ที่ timestep เดียวกันจึงมีเจตนาเดียวกัน ต่างแค่เรขาคณิต
**การ lookup จาก `z` ตอบผิดโดยโครงสร้าง**

**`m3d_cross` เทียบ `m3d_bracketed` 25 epochs:**

| | control | cross |
|---|---|---|
| heldout เฉลี่ย ep1–10 | 0.0992 | **0.0760** (ดีขึ้น 23%) |
| heldout เฉลี่ย ep11+ | 0.0965 | **0.0715** (ดีขึ้น 26%) |
| ดีที่สุด (องศา) | 3.57 | **2.91** |
| **x-gap** | 10.7× | **40–69×** |
| z-gap | 21× | 2.2–3.2× |

**swap test พลิกสมบูรณ์** — ให้ภาพหุ่น A + `z` หุ่น B → ตอบเป็นคำสั่งของ **A ผิด 1.18 องศา**
เทียบกับตอน input ถูกทั้งคู่ที่ 1.17 (ต่างกัน 0.01) **`z` ไม่ได้ตัดสินร่างกายอีกต่อไป ภาพเป็นตัวตัดสิน**

**เลิกก๊อป** — น้ำหนักกระจุกลดจาก 0.883/0.947 เหลือ **0.540** ต่ำกว่าส่วนผสมที่ดีที่สุด (0.697)
และ **ชนะ copy-nearest เป็นครั้งแรก** (2.91 เทียบ 3.47)

**ไม่เสื่อมเมื่อเทรนนาน** — ทุกรันก่อนหน้าพีคที่ ep8 แล้วนิ่งหรือถอย รันนี้ ep11–25 **ดีกว่า** ep1–10

**สองอย่างที่ต้องรายงานตามตรง:**
- z-gap เหลือ 2.2–3.2× (control 21×) — `z` แทบไม่ถูกใช้แล้ว
- FTM ไม่เปลี่ยน (`z` ช่วย 1.03× เทียบ control 1.07×) — **F23 ยังยืน** `L_recon` ยังไม่ได้หล่อหลอม `z`

**เหตุผลที่ได้ผลต่างจากสี่วิธีก่อน:** สี่วิธีนั้นเปลี่ยน**วิธีเข้าถึง**ภาพโดยไม่เปลี่ยนคำถาม
และ lookup ตอบคำถามเดิมได้ถูกกว่าเสมอ ส่วนอันนี้**เปลี่ยนคำถาม** — lookup ผิดโดยโครงสร้าง

### 18.15 จัดระเบียบ `results/wm`

แยกเป็น `figures/` `analysis/` `predictions/` `replay/` `cache/` `eval/` และเขียน README ใหม่
พร้อมตารางเทียบทุกรัน baseline เป็นองศา และคำสั่ง regenerate ทุกอย่าง
ลิงก์ในเอกสารทั้งหมดอัปเดตและตรวจแล้วว่าชี้ไปไฟล์ที่มีจริง

### 18.16 พบว่า cross-augmentation คือสาเหตุที่ `L_recon` ไม่ทำงาน (2026-08-09)

ตามข้อสงสัยของผู้ใช้ วัดขนาด noise จาก cross-augmentation เทียบสัญญาณจริง (40 เฟรม หน่วยเดียวกับ `L_recon`)

| | ค่า |
|---|---|
| **noise จาก augmentation** (เฟรมเดียวกัน สอง view) | **8.51** |
| **สัญญาณ** (เฟรมติดกัน ไม่ augment) | **1.97** |
| เป้าหมายที่ FTM ต้องปิด | 8.43 |

**noise / signal = 4.33** และ augmentation คิดเป็น **101%** ของเป้าหมายที่ FTM ถูกเทรน

`z` ช่วยได้อย่างมาก 23% ของ `L_recon` วัดจริงได้ 3–7% (F23) — **`z` ไม่ได้ไร้ประโยชน์ มันถูกกลบ**

รวมกับ F23: 99% ของ gradient ไปที่ term ที่เป้าหมายเป็น noise ล้วน เหลือ 1% ให้ `L_motion`
ซึ่งยอมให้จำ → **`z` ไม่เคยอยู่ภายใต้แรงกดดันให้เป็น action เลย**

cross-augmentation มีเหตุผลที่ต้องมี (กัน ITM ลักลอบส่ง `x_{t+1}` เข้า `z`) แต่ไม่มีใครเคยวัดราคาของมัน
บันทึกเป็น FINDINGS F25

### 18.17 แยก crop กับ jitter — ไม่มีระดับความแรงไหนกู้สัญญาณได้ (2026-08-09)

ก่อนจะลดความแรง augmentation ต้องรู้ว่า noise มาจากส่วนไหน (สัญญาณ = 1.92)

| augmentation | noise | noise / signal |
|---|---|---|
| crop 85-100% + jitter (ปัจจุบัน) | 8.42 | 4.39 |
| crop 85-100% อย่างเดียว | 8.56 | 4.47 |
| crop 95-100% อย่างเดียว | 6.76 | 3.53 |
| **jitter อย่างเดียว ไม่ crop** | **4.02** | **2.10** |
| crop 95-100% + jitter | 7.02 | 3.66 |

crop เป็นตัวใหญ่กว่า แต่บีบจาก 85 เป็น 95 ลด noise ได้แค่ 21% และ **jitter อย่างเดียว —
ซึ่งไม่ขยับอะไรในภาพเลย — ยังให้ 2.10 เท่าของสัญญาณ** แปลว่า encoder ที่แช่แข็งไว้ไม่ invariant
ต่อความสว่าง/คอนทราสต์ ซึ่งเป็นข้อสมมติที่ cross-augmentation ทั้งกลไกวางอยู่บน

**การลดความแรงแก้ไม่ได้** ทางที่เหลือคือตัด cross-augmentation ทิ้งแล้วพึ่ง bottleneck
(`z` 64 มิติ เทียบ `e_{t+1}` 359,000 มิติ — ต่างกัน 5,600 เท่า คัดลอกตรง ๆ เป็นไปไม่ได้อยู่แล้ว)
หรือย้ายไป augment ใน embedding space ยังไม่ได้ทดสอบทั้งคู่

ข้อที่เคยเสนอว่า "ปล่อยเป้าหมายเป็นเฟรมสะอาด" **ผิด** — การป้องกันมาจากการที่เป้าหมาย *สุ่ม*
เป้าหมายที่แน่นอนคือสิ่งที่การคัดลอกยิงโดนพอดี ตัดทิ้งแล้ว

### 18.18 `z` ไม่ได้กลวง มันบริสุทธิ์ขึ้น (2026-08-09)

ผู้ใช้กังวลว่า z-gap ที่ตกจาก 21× เหลือ 2.2× แปลว่า `z` ว่างเปล่า ซึ่งจะทำให้ Stage 2 พังทั้งหมด
วัดโดยถอด foot-contact pattern (8 แบบ majority 0.144) และ body (5 แบบ สุ่ม 0.200) ออกจาก `z`
พร้อมแยก variance ของ `z` ว่าอะไรอธิบายมัน

| | control ep20 | cross ep8 | cross ep27 |
|---|---|---|---|
| **contact pattern จาก `z`** | 0.757 | 0.744 | **0.787** |
| body จาก `z` | 0.707 | 0.638 | 0.665 |
| **variance: จังหวะ** | 64.5% | **88.7%** | **83.4%** |
| **variance: ร่างกาย** | 8.8% | **1.2%** | **1.2%** |
| variance: ส่วนที่เหลือ | 26.8% | 10.1% | 15.4% |

พฤติกรรมยังถอดได้เท่าเดิมหรือดีกว่า ส่วนของร่างกายลดลง **7 เท่า** และจังหวะขึ้นไป 83-89%

`lambda_cross` ทำสำเร็จในสิ่งที่ adversarial head (Q5) ถูกสร้างมาทำแล้วล้มเหลว — และทำได้โดยเป็น
ผลพลอยได้จากงานที่ตั้งถูก ไม่ใช่การไปฝืนกับ latent จังหวะจึงไม่เสียหาย

ที่ z-gap ตกลงก็อธิบายได้: **21× ของ control ส่วนใหญ่คือการสูญเสียโค้ดร่างกาย ไม่ใช่จังหวะ**
พอ decoder อ่านร่างกายจากภาพแล้ว ตัด `z` ออกก็เสียแค่จังหวะ ซึ่งภาพเดียวก็บอกได้เยอะ
→ บันทึกเป็น F26 และปิด Q8

**ผลต่อ Stage 2:** ข้อสมมติหลักยืนอยู่ได้ `z` เป็นตัวแทนจังหวะที่ไม่ขึ้นกับร่างกาย และตอนนี้
วัดได้แล้วไม่ใช่แค่สมมติ สิ่งที่ยังไม่รู้คือมันจะไม่ขึ้นกับร่างกายข้าม *topology* ด้วยหรือไม่
(6 ขา → 4 ขา contact pattern คนละจำนวน) ซึ่งเป็น claim B ของ Q0 ต้องรันจริงเท่านั้นถึงจะรู้

เครื่องมือวัดถูกย้ายเข้าโปรเจกต์แล้ว: `scripts/z_content.py` และ `scripts/aug_noise.py`
