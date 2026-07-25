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
