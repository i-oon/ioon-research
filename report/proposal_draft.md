# PROPOSAL DRAFT: written sections
### Cross-Morphology Locomotion via Latent Action World Models

> This file contains the written content for the sections your advisor asked you to write now
> (Abstract TH/EN, Ch.1, Ch.2, Ch.3, References). Section numbers match the template so you can
> copy each block straight into Word. Sections marked "ยังไม่ต้องเขียน" are omitted.
>
> Two facts baked into the text (from our decisions):
> - **University advisor**: Mr. Bawornsak Sakulkueakulsuk (KMUTT / IFR).
>   **Lab advisor (internship)**: Prof. Poramate Manoonpong (VISTEC).
> - We are **vision-grounded but NOT vision-only**: the latent action is grounded on auto-logged
>   joint commands via a motion-decoding loss (inherited from LAC-WM). The proposal states this honestly.
> - `[[DECIDE]]` marks a spot where you must choose/confirm before final submission.

---

# 3. บทคัดย่อ (ภาษาไทย)

การพัฒนาหุ่นยนต์เดินด้วยการเรียนรู้แบบเสริมกำลัง (Reinforcement Learning) ในปัจจุบันมีข้อจำกัดสำคัญคือ นโยบายควบคุม
(policy) ที่ฝึกได้จะผูกติดกับสัณฐานวิทยา (morphology) ของหุ่นยนต์ตัวนั้นโดยตรง เมื่อรูปร่างของหุ่นยนต์เปลี่ยนไป เช่น
ความยาวขาเปลี่ยน นโยบายเดิมจะใช้ไม่ได้และต้องฝึกใหม่ทั้งหมด ซึ่งใช้เวลาและทรัพยากรสูง งานวิจัยนี้จึงศึกษาการเรียนรู้
"การกระทำแฝงที่ไม่ขึ้นกับสัณฐานวิทยา" (morphology-agnostic latent action) จากวิดีโอในสภาพแวดล้อมจำลอง โดยประยุกต์
แนวคิดของแบบจำลองโลกเชิงการกระทำแฝง (Latent Action World Model) ซึ่งเดิมใช้ในงานหยิบจับวัตถุ (manipulation)
มาสู่โดเมนการเคลื่อนที่ (locomotion) เป็นครั้งแรก

งานวิจัยใช้แมลงกิ่งไม้ (stick insect, *Medauroidea extradentata*) ในโปรแกรมจำลอง CoppeliaSim จำนวนสามสัณฐาน
ที่มีความยาวขาต่างกัน (1.0, 0.75 และ 0.5 เท่า) โดยใช้ตัวเข้ารหัสภาพ V-JEPA2 แบบตรึงค่า (frozen) ร่วมกับแบบจำลอง
การเปลี่ยนผ่านเชิงผกผัน (Inverse Transition Model) และเชิงไปข้างหน้า (Forward Transition Model) เพื่อสกัดการกระทำ
แฝง z_t วัตถุประสงค์หลักคือเพื่อพิสูจน์ว่า z_t เกาะกลุ่มตาม "พฤติกรรม" (เช่น การเดิน) แทนที่จะเกาะกลุ่มตาม "รูปร่างขา"
และเพื่อแสดงว่าแบบจำลองโลกที่ผ่านการฝึกล่วงหน้าช่วยให้สัณฐานที่ไม่เคยเห็นมาก่อนเรียนรู้ได้เร็วขึ้นโดยใช้ข้อมูลน้อยลง

ความแตกต่างสำคัญจากงานก่อนหน้าคือ แบบจำลองรับรู้สภาพหุ่นยนต์ผ่าน "ภาพ" เท่านั้น และ **ไม่เคยถูกบอกข้อมูล
ความยาวขา** (spec-free / implicit) ต่างจากงานที่ต้องอ่านข้อมูลโครงสร้างจากไฟล์ CAD ทั้งนี้การฝึกใช้ป้ายกำกับการกระทำ
(joint command) ที่บันทึกอัตโนมัติจากตัวจำลองเพื่อยึดโยงความหมายของ z_t และป้ายกำกับดังกล่าวจะถูกละทิ้งในขั้นใช้งานจริง

*(ผลการทดลองจะเพิ่มเติมเมื่อดำเนินการเสร็จ)*

**คำสำคัญ:** การเคลื่อนที่ข้ามสัณฐาน / การกระทำแฝง / แบบจำลองโลก / การเรียนรู้เชิงถ่ายโอน / V-JEPA2

---

# 4. Abstract (English)

Reinforcement-learning controllers for legged locomotion have a fundamental limitation: a trained policy is
tightly coupled to the specific body (morphology) it was trained on. When the body changes, for example when
leg length changes, the policy becomes unusable and must be retrained from scratch, at high cost in time and
compute. This research studies the learning of a **morphology-agnostic latent action** from simulation video,
adapting the Latent Action World Model paradigm, previously demonstrated only for robotic manipulation, to the
domain of **legged locomotion** for the first time.

The study uses a simulated stick insect (*Medauroidea extradentata*) in CoppeliaSim, instantiated as three
morphologies differing only in leg length (1.0×, 0.75×, and 0.5×). A frozen V-JEPA2 visual encoder is combined
with an Inverse Transition Model and a Forward Transition Model to extract a latent action z_t. The primary
objectives are (i) to demonstrate that z_t organizes by *behaviour* (e.g. walking) rather than by *body shape*,
and (ii) to show that a pretrained world model enables an unseen morphology to be learned with substantially
fewer episodes than training from scratch.

The key distinction from prior work is that the model perceives the robot **only through vision** and is
**never given its leg length** (spec-free / implicit morphology), in contrast to methods that read structural
parameters from CAD files. Training uses joint-command labels that the simulator logs automatically to ground
z_t through a motion-decoding loss; this supervision is discarded at inference.

*(Experimental results to be added upon completion.)*

**Keywords:** cross-morphology locomotion / latent action / world model / transfer learning / V-JEPA2

---

# Chapter 1  Introduction

## 1.1  Background and Significance

Legged robots are typically controlled by policies learned through reinforcement learning (RL). A persistent
and expensive limitation of this approach is **morphology specificity**: a policy learned for one body does
not generalize to another. If a robot's legs are lengthened or shortened, its mass distribution changes, or a
limb is damaged, the previously trained policy fails and a new one must be learned from scratch, a process
that can take hours to days of training per body. In contrast, biological organisms share locomotion principles
across vastly different body plans, suggesting that a shared, body-independent representation of *movement* is
possible.

One promising direction is the **World Model**, an internal learned model of environment dynamics that predicts
future states from actions and lets an agent "imagine" rollouts instead of always acting in the real
environment (e.g. DreamerV3). A recent advance, the **Latent Action World Model** (LAC-WM, Huang et al., 2026),
learns a *unified latent action space* shared across multiple robot embodiments, and shows that a learned
latent action generalizes across embodiments better than raw, body-specific action commands. However, LAC-WM
was demonstrated only for **manipulation** (robot arms and hands), and, like other cross-embodiment methods,
the embodiments it unifies have *structurally different action spaces*.

This research asks whether the same idea transfers to **legged locomotion**, and specifically to a setting the
prior work does not address: bodies that share an *identical* action space and differ only in **limb length**.
Here the challenge is not to reconcile different action dimensionalities, but to learn a latent action that
captures the **behaviour** (the intent of a movement, such as "step forward") independently of how a given body
realizes that behaviour, because identical joint commands produce *different resulting motion* on a short leg
versus a long leg. We ground this study in a simulated stick insect (*Medauroidea extradentata*), a
well-studied model organism for insect walking, instantiated at three leg lengths.

The significance is twofold. Scientifically, it tests whether a morphology-invariant notion of behaviour can
**emerge from vision alone**, without the model ever being told the body's dimensions. Practically, if such a
representation transfers, it would reduce the cost of deploying new robot morphologies by reusing accumulated
locomotion "skill" rather than retraining from zero.

## 1.2  Objectives

1. To learn a morphology-agnostic latent action z_t from simulation video of a legged robot, by adapting the
   Latent Action World Model pipeline (frozen visual encoder → Inverse Transition Model → Forward Transition
   Model → Motion Decoder) from manipulation to locomotion.
2. To demonstrate quantitatively that z_t organizes by **behaviour** rather than by **morphology**, using
   linear-probe accuracy and cluster-separation (silhouette / variance-decomposition) metrics across three
   leg-length variants of the same body.
3. To demonstrate that a world model pretrained on two morphologies enables a third, unseen morphology to reach
   the same reconstruction quality with **significantly fewer training episodes** than a model trained from
   scratch (i.e. improved sample efficiency of transfer).

## 1.3  Expected Benefits

1. A demonstration that latent-action world models extend beyond manipulation to legged locomotion.
2. Evidence on whether morphology-invariant behaviour structure can be learned **without** privileged
   morphology information (no CAD/spec input), using only visual observation and auto-logged action labels.
3. A reusable evaluation methodology (behaviour-vs-morphology latent probing) and a controlled, single-axis
   leg-length benchmark for future cross-morphology locomotion studies.
4. A foundation for reducing the retraining cost of deploying legged robots of new body proportions.

## 1.4  Scope of the Research

1. The study is conducted **entirely in simulation** (CoppeliaSim v4.10 with the Bullet physics engine). No
   physical-robot or sim-to-real deployment is in scope.
2. The robot is a single species, the stick insect *Medauroidea extradentata*, instantiated as **three
   morphologies** that differ **only in leg length** (1.0× baseline, 0.75×, 0.5×); all other properties
   (topology, degrees of freedom, 18-dimensional joint-position action space) are held identical. Training
   uses the 1.0× and 0.5× variants; the 0.75× variant is held out for the transfer test. This is an
   **interpolation** study (the unseen body lies between the trained ones), not extrapolation.
3. The primary behaviour studied is **forward walking**, driven by a fixed reference gait. Turning and stopping
   are planned extensions. `[[DECIDE]]`: confirm with advisor whether the proposal commits to walk/turn/stop
   or to forward-walking only for this phase (the biological reference data contains only straight walking).
4. The visual encoder (V-JEPA2) is used **frozen**; training it is out of scope except for a possible
   fine-tuning of its final blocks if a domain gap is found.

## 1.5  Research Procedure

The project follows a staged, milestone-gated procedure in which each stage must pass before the next begins
(this component-by-component validation strategy was set by the lab advisor):

1. **Morphology Gap Check:** confirm that the three leg-length variants produce measurably different behaviour
   under identical commands (otherwise the transfer question is vacuous).
2. **Visual Encoder Sanity Check:** confirm that the frozen V-JEPA2 features carry the behaviour-relevant
   signal (gait/contact state) needed downstream, before any training.
3. **World-Model Training:** train the Inverse/Forward Transition Models and Motion Decoder on the two
   training morphologies.
4. **Latent-Space Validation:** verify that the learned latent action z_t clusters by behaviour, not
   morphology.
5. **Transfer Evaluation:** measure the sample-efficiency gain of the pretrained world model on the unseen
   morphology, against a from-scratch baseline and a raw-action baseline.

---

# CHAPTER 2  Literature Review

This project draws on four bodies of work: world models, latent action models, cross-morphology legged
control, and the biomechanics of insect walking (which motivates the model organism and the evaluation). Each
is reviewed below, and the review closes with the specific gap this work addresses.

## 2.1  World Models and Video Representation Learning

**2.1.1  World models for control.**
A world model learns environment dynamics so that an agent can plan or learn "in imagination." DreamerV3
(Hafner et al., 2023) established that a single model-based agent with a recurrent latent-state world model can
master a large range of domains. Such models predict compact latent states rather than raw pixels, which makes
long-horizon imagined rollouts tractable.

**2.1.2  Video foundation models.**
V-JEPA 2 (Assran et al., 2025) is a self-supervised video model trained by masked prediction in representation
space over ~1M hours of internet video. It learns motion-relevant features that transfer to understanding,
prediction, and planning. Importantly for this work, the accompanying V-JEPA 2-AC variant uses the encoder as a
**frozen per-frame image encoder** feeding a lightweight downstream predictor, which is precedent for the way
we use V-JEPA2 here. A known caveat is that V-JEPA2 was trained only on real-world video (no rendered/simulated data),
so its behaviour under a rendering domain gap is not characterized by the original work.

## 2.2  Latent Action Models and Latent Action World Models

**2.2.1  Learning actions from unlabeled video.**
A line of work learns a *latent action*, an inferred low-dimensional code for the change between consecutive
observations, without ground-truth action labels. Genie (Bruce et al., 2024) demonstrated
controllable environments learned from unlabeled video via an inverse-dynamics-style latent action; LAPA
(Ye et al., 2024) and CLAM (Liang et al., 2025) extended latent-action pretraining toward robot control, with
CLAM using continuous latent actions closest to the continuous joint control needed here. Zhang et al. (2025)
analyze what latent action models actually learn, and provide theoretical grounding for interpreting the latent.

**2.2.2  Latent Action World Models (the base method).**
LAC-WM (Huang et al., 2026) unifies a latent action space across heterogeneous manipulation embodiments. Its
architecture is the pipeline this project adapts: a frozen V-JEPA2 encoder, an Inverse Transition/Dynamics
Model that produces the latent action, a Forward Transition/Dynamics Model that predicts the next visual
embedding, and a Motion Decoder that grounds the latent against ground-truth motion. LAC-WM reports that its
unified latent action improves cross-embodiment transfer over an explicit-action baseline (EAC-WM), and that
performance scales positively with embodiment diversity. Its key premise, however, is heterogeneity of *action
spaces* across embodiments. That premise does not hold in our single-species, leg-length-only setting, which
motivates the reframing in Section 2.5.

## 2.3  Cross-Morphology and Cross-Embodiment Legged Locomotion

**2.3.1  Latent-action / world-model approaches.**
Li et al. (2021) plan in a learned latent action space for legged locomotion and demonstrate the method on both
a hexapod and a quadruped. Crucially, their framework is re-instantiated and trained *separately per robot*:
no single latent space or dynamics model is transferred across bodies, so it does not perform cross-morphology
transfer in the sense studied here. QWM (Danesh et al., 2026) introduces a morphology-conditioned world model
that generalizes zero-shot to new quadruped morphologies, but it does so by **explicitly conditioning** on
morphology parameters (limb lengths, mass) read from each robot's CAD/USD file, uses proprioception (not
vision), and does not learn a latent *action*. These two works are the closest prior art and are contrasted
directly in Section 2.5.

**2.3.2  Policy-transfer approaches.**
A broad literature learns morphology-agnostic control policies via graph/transformer architectures (e.g.
MetaMorph, Gupta et al., 2022) or shared latent-to-latent policies across robots (L3P, Zheng et al., 2025).
Ai et al. (2025) study embodiment scaling laws and show that training over many procedurally varied bodies
improves generalization. These establish cross-morphology locomotion as an active problem but approach it
through policy transfer rather than a vision-based latent-action world model.

## 2.4  Biomechanics of Insect Walking and Prior Lab Work

**2.4.1  Stick insect locomotion.**
The stick insect (*Carausius morosus*, *Medauroidea extradentata*) is a classical model for legged locomotion
(Cruse, Bässler, Büschges, Dürr). Coordination arises from decentralized local rules (the "Cruse rules") acting
between neighbouring legs. The standard descriptors of gait are the anterior/posterior extreme positions
(AEP/PEP), the duty factor, and the inter-leg phase. Notably, straight walking is largely an *emergent* property
of local rules and bilateral symmetry rather than a dedicated heading controller, as shown by experiments in
which a single remaining leg still produces the correct stepping pattern. This informs both the choice of
foot-contact state as a behaviour label and the interpretation of gait in this study.

**2.4.2  Prior work in the host lab.**
Directly relevant, Larsen et al. (2023, Manoonpong group) model the *same species* in CoppeliaSim under a
decentralized CPG controller with foot-contact feedback, explicitly handling heterogeneous leg lengths, and
Chuthong et al. (2026) study resilient stick-insect-inspired control. These provide the biological data, the
simulation model, and a controller lineage that this project builds upon, while pursuing a different
(learned-latent-representation) research question.

## 2.5  Research Gap

Prior cross-morphology world models either (a) require **explicit** morphology specifications from CAD
(QWM), (b) are trained **separately per body** without transfer (Li et al.), or (c) are demonstrated only for
**manipulation** (LAC-WM). None studies whether a **vision-grounded latent action**, with morphology **never
provided to the model**, organizes by *behaviour* rather than *body shape*, and none isolates the effect of a
single morphological axis (leg length) on that latent. This project addresses exactly that gap: an
implicit/spec-free, vision-based latent-action world model for locomotion, evaluated with a controlled
single-axis leg-length sweep and an explicit behaviour-vs-morphology latent probe. We are careful **not** to
claim vision-only learning: the latent is grounded during training on auto-logged joint commands (a by-product
of simulation, discarded at inference), following LAC-WM.

---

# CHAPTER 3  Research Methodology

## 3.1  System Overview

The system comprises four learned/parameterized components on top of a fixed perceptual front-end, trained
end-to-end except for the frozen encoder:

```
 RGB frame o_t ─▶ [ V-JEPA2 encoder (FROZEN) ] ─▶ visual embedding e_t (256×1408)
                                                      │
              e_t, e_{t+1} ─▶ [ Inverse Transition Model (ITM) ] ─▶ latent action z_t (64-d)
                                                      │
                     e_t, z_t ─▶ [ Forward Transition Model (FTM) ] ─▶ ê_{t+1}   (L_recon)
                                                      │
                     x_t, z_t ─▶ [ Motion Decoder (MD) ] ─▶ â_t  vs  a_t (auto-logged)  (L_motion)
```

Input/output specification of each block (units stated, per the advisors' requirement):

| Block | Input | Output | Trained? |
|---|---|---|---|
| V-JEPA2 encoder | RGB frame, 256×256×3 (uint8) | e_t ∈ ℝ^{256×1408} patch tokens | No (frozen) |
| Inverse Transition Model | [e_t, e_{t+1}] (512 tokens) | z_t ∈ ℝ^{64} (latent action) | Yes |
| Forward Transition Model | [e_t, z_t] | ê_{t+1} ∈ ℝ^{1408} | Yes |
| Motion Decoder | (e_t as context, z_t as query) | â_t ∈ ℝ^{18} (joint targets, rad) | Yes (discarded at inference) |

Losses: **L = λ_recon · L_recon + λ_motion · L_motion**, where L_recon = ‖ê_{t+1} − e_{t+1}‖² is computed in
embedding space (no pixel decoder required) and L_motion = ‖â_t − a_t‖². **Cross-augmentation** is applied:
two independent augmentations of each frame pair are encoded, one feeding the ITM and the other the FTM target,
which prevents the ITM from smuggling raw future-frame content into z_t as a shortcut.

## 3.2  Simulation Environment and Morphology Variants

**3.2.1  Simulator and model.** CoppeliaSim v4.10 (Bullet 2.78 engine, 20 Hz timestep). The robot is the
*Medauroidea extradentata* model (six legs, three actuated joints each: ThC/CTr/FTi), borrowed from the host
lab, with an 18-dimensional joint-position-target action space.

**3.2.2  Leg-length variants.** Three scenes are generated by scaling the coxa/femur/tibia segments of all six
legs by a uniform factor and repositioning downstream joints accordingly: **long 1.0× (base), medium 0.75×,
short 0.5×**. Scaling is verified numerically (measured foot-reach ratio matches the target factor for all six
legs).

**3.2.3  Observation capture.** A single fixed vision sensor (256×256, side view, ~40° elevation) is added
programmatically so that its pose relative to the robot is *identical across all morphologies by construction*.
The floor uses a matte, mildly-textured, non-repeating surface (a controlled choice: both a stark checkerboard
and a blank surface were found to corrupt the visual features). The camera tracks the robot's planar position
at fixed height and orientation, keeping apparent size constant. `[[DECIDE]]`: single-camera is used; a
multi-camera setup was considered and rejected (see project notes).

## 3.3  Visual Encoder (Frozen V-JEPA2)

The encoder is `facebook/vjepa2-vitg-fpc64-256` (ViT-g/16, ~1B parameters), used frozen. Because this
checkpoint is natively a 64-frame *video* encoder with 2-frame tubelets, feeding a real clip would let each
frame attend to other timesteps and contaminate its embedding. To obtain an independent per-frame embedding,
each frame is **duplicated into the minimal 2-frame tubelet** and encoded alone, yielding 256 patch tokens of
dimension 1408 per frame. This usage has direct precedent in V-JEPA 2-AC.

## 3.4  Latent Action Model

**3.4.1  Inverse Transition Model (ITM).** Attention-based (causal self-attention blocks with a learned query
token) mapping the pair [e_t, e_{t+1}] to the latent action z_t ∈ ℝ^{64}. The causal structure frames z_t as
"what happened between t and t+1."

**3.4.2  Forward Transition Model (FTM).** Attention-based, predicting the next visual embedding ê_{t+1} from
[e_t, z_t]; supervised by L_recon in embedding space.

**3.4.3  Motion Decoder (MD).** Cross-attention with z_t as query over the current visual context, producing
â_t; supervised by L_motion against the auto-logged joint command a_t. This is the only place ground-truth
action is used, and the decoder is **discarded after pretraining**. Its role is to anchor z_t to real motion so
the latent cannot collapse into a trivial identity.

**3.4.4  Dimensions and hyperparameters.** z_t is 64-dimensional (per LAC-WM §4.2). λ weights, learning rate,
and optimizer are to be determined by ablation (the base paper does not report them). Training uses mixed
precision on a single GPU (RTX 2080 Ti, 11 GB), with small batch size and gradient accumulation to fit the
frozen 1B-parameter encoder in the training loop (cross-augmentation prevents offline embedding caching).

## 3.5  Data Collection

Episodes are recorded by driving each morphology with a fixed reference gait (replayed real-animal joint
trajectories) and capturing, per timestep, the RGB frame, the 18-D joint command a_t, the body pose, and the
per-foot contact force. Reloading the scene between episodes provides genuine variation (the contact-rich
dynamics are chaotic, so each reload yields a different trajectory), while within an episode the simulation is
deterministic. Behaviour is labelled by **6-bit foot-contact state** (which feet are planted), a body-pose
label preferable to a time-based phase label. `[[DECIDE / LIMITATION]]`: the reference gait is a single
animal's replay applied to all bodies; a per-morphology "working" controller (e.g. the lab CPG) would give a
cleaner behaviour target and is noted as future work.

## 3.6  Evaluation Protocol

**3.6.1  Metrics.** Two complementary measures are reported for every latent-structure claim, since reporting
only one can mislead. The first is **linear-probe accuracy**: can a simple classifier decode a label
(behaviour or morphology) from the representation? This measures the *presence* of information. The second is
the **silhouette score together with between-class variance**, which measure *dominance*, that is, whether
that information is the main axis of variation. Cross-morphology **transfer** is measured by training a probe
on two morphologies and testing on the held-out one.

**3.6.2  Milestone gates.**
- *Morphology gap:* pass if the three variants travel measurably different distances under identical commands.
- *Encoder sanity:* pass if behaviour (foot-contact state) is decodable from e_t above a noise floor.
- *Latent validation:* pass if z_t lowers morphology decodability and raises cross-morphology behaviour
  transfer relative to the raw encoder baseline.
- *Transfer:* pass if the pretrained world model reaches the same L_recon on the unseen morphology with
  significantly fewer episodes than a from-scratch model.

**3.6.3  Baselines.** (1) A from-scratch world model on the unseen morphology (the sample-efficiency
reference). (2) A **raw-joint-conditioned** forward model with a shared encoder (an "explicit action" analog):
if the latent action does not beat raw joint conditioning, the latent bottleneck adds no value, which makes
this the decisive ablation. (3) Where applicable, an explicit morphology-conditioned model in the spirit of QWM, as
a contrast between explicit and implicit morphology handling.

---

# REFERENCES  *(preliminary; refine formatting to the required style)*

[1] D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap. "Mastering Diverse Domains through World Models" (DreamerV3).
    arXiv:2301.04104, 2023.
[2] A. Assran, et al. "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning."
    arXiv:2506.09985, 2025.
[3] H. Huang, et al. "Latent Action Robot Foundation World Models for Cross-Embodiment Adaptation" (LAC-WM).
    ICML 2026.
[4] M. H. Danesh, et al. "Toward Hardware-Agnostic Quadrupedal World Models via Morphology Conditioning" (QWM).
    arXiv:2604.08780, 2026.
[5] T. Li, R. Calandra, D. Pathak, Y. Tian, F. Meier, A. Rai. "Planning in Learned Latent Action Spaces for
    Generalizable Legged Locomotion." IEEE RA-L, 2021 (arXiv:2008.11867).
[6] J. Bruce, et al. "Genie: Generative Interactive Environments." ICML 2024 (arXiv:2402.15391).
[7] S. Ye, et al. "Latent Action Pretraining from Videos" (LAPA). ICLR 2025 (arXiv:2410.11758).
[8] A. Liang, et al. "CLAM: Continuous Latent Action Models for Robot Learning from Unlabeled Demonstrations."
    arXiv:2505.04999, 2025.
[9] C. Zhang, et al. "What Do Latent Action Models Actually Learn?" NeurIPS 2025 (arXiv:2506.15691).
[10] A. D. Larsen, T. H. Büscher, T. Chuthong, T. Pairam, H. Bethge, S. N. Gorb, P. Manoonpong.
    "Self-Organized Stick Insect-Like Locomotion under Decentralized Adaptive Neural Control." Advanced Theory
    and Simulations 6, 2300228, 2023.
[11] T. Chuthong, T. H. Büscher, S. N. Gorb, P. Manoonpong. "Insect-Inspired Resilient Machines." Advanced
    Intelligent Systems, 2026.
[12] H. Cruse. "What mechanisms coordinate leg movement in walking arthropods?" Trends in Neurosciences 13, 1990.
[13] R. McN. Alexander, A. S. Jayes. "A dynamic similarity hypothesis for the gaits of quadrupedal mammals."
    Journal of Zoology 201, 1983.
[14] A. Gupta, et al. "MetaMorph: Learning Universal Controllers with Transformers." ICLR 2022.
[15] B. Ai, et al. "Towards Embodiment Scaling Laws in Robot Locomotion." CoRL 2025 (arXiv:2505.05753).

---

# APPENDIX  *(optional candidates)*
- A. Morphology scaling procedure and numerical verification of leg-reach ratios.
- B. Camera/render configuration and the render-environment control (avoiding checkerboard / blank surfaces).
- C. Milestone results obtained so far (morphology-gap check; encoder sanity check). Move these to Ch.4 once
  the proposal becomes the full thesis.
