# Prior Work: Cross-Body Goals and Shared Signals for Latent Actions

Written 2026-09-28. The question: has anyone
- taken a goal or demonstration from a **different body** and reproduced it on this body, or
- used **Froude or dimensionless body velocity** as the shared signal that aligns latent actions or world models across bodies?

Sources:
- **Read in full:** two local PDFs in `doc/ref/`, Sections 1 and 2.
- **Abstracts and snippets only:** a web search of 7 queries, Section 3. Verify every 2026 arXiv id before citing it.
- **Read in full via arXiv HTML:** AnyWorld and 2609.19846, Section 4.

## Our framing (for comparison)
- **Model:** a latent-action world model on egocentric video, with a frozen V-JEPA2 encoder.
  - z = ITM(frame_t, frame_t+k), stride 5.
  - The FTM predicts the next frame embedding from (frame_t, z).
  - A per-body projector maps joint commands to z.
- **Planned grounding:** one decoder, shared by every body, reads z alone and predicts Froude-normalised body velocity:
  - fwd / lat speed ÷ √(g·h);
  - yaw rate × √(h/g).
- **Task:** the goal is the body-motion sequence of a hexapod; a quadruped (B1) reproduces it by selecting actions, evaluated in a physics closed loop.

## 1. QWM: "Toward Hardware-Agnostic Quadrupedal World Models via Morphology Conditioning"
Danesh et al., arXiv 2604.08780, April 2026. Read in full.

- **Input:** proprioception only, no vision. The world model is a DreamerV3 RSSM, and its z is a **state** latent, not an action latent.
- **Shared across bodies:** the velocity-tracking objective.
  - Commands are in raw m/s and rad/s, with the same ranges for every robot (App. Table IX).
  - **No Froude and no normalised velocity.**
  - Rewards use per-robot adaptive normalisation of the return quantiles (ARN).
- **Embodiment:** an explicit morphology descriptor µ, taken from the robot's USD file (link lengths, mass ratios, torque density = τ_max / (M·g)). It is injected into the encoder and into the recurrence (§IV-B).
- **New body:** zero-shot from µ. Trained on 7 quadrupeds, 1 held out.
  - Sim episode length out of about 1000: ANYmal-D 949, Go1 974, B2 405 (B2 is the extrapolation case, and it fails).
  - Real robots: velocity error 0.30 vs a specialist's 0.28 (ANYmal-D), 0.34 vs 0.31 (Go1).
  - Quadrupeds only.
- **Goal:** the same velocity command for every body. No goal comes from another body.
- **Relation to us:** a baseline to contrast with. It injects morphology explicitly from the robot description; we would ground z in normalised motion from vision.

## 2. LAT: "Planning in Learned Latent Action Spaces for Generalizable Legged Locomotion"
Li, Calandra, Pathak, Tian, Meier, Rai. RA-L 2021, arXiv 2008.11867. Read in full.

- **Input:** state only. A 2–4-D latent z indexes a low-level policy imitated from 50 IK experts. MPC by random shooting (8000 samples, H = 1) runs on a learned CoM-motion model f(ẋ, ẏ, z).
- **Bodies:** a hexapod (Daisy) and a quadruped (A1), **trained separately**. Nothing is shared. Costs use raw CoM velocity.
- **New body:** full retraining. Sim to hardware refits only the dynamics model, from 2000 samples (§IV-D).
- **Goal:** CoM velocity or position of the same body.
- **Relation to us:** precedent that planning in a latent space whose output is body motion works on both a hexapod and a quadruped. Not cross-body.

## 3. Web search (abstracts only)

| paper | goal from another body | shared space | ego vision | domain |
|---|---|---|---|---|
| AnyWorld, arXiv 2608.29242 (2026) | yes, human → robot | latent split into action / camera / embodiment | yes | manipulation |
| UniSkill, 2505.08787 (2025) | yes, human or robot video prompt | skill latent from frame pairs | no | manipulation |
| Action-similarity supervision for LAMs, 2609.19846 (2026) | partly | latent action aligned by supervised similarity | unclear | manipulation |
| XIRL, 2106.03911 (CoRL 2021) | yes, different manipulator morphologies | task-progress reward embedding | no | manipulation |
| CrossLoco, 2309.17046 (ICLR 2024) | yes, human → quadruped | cycle-consistent motion correspondence | no (mocap) | locomotion |
| Animal video → quadruped: 2412.04273; 2203.05973; npj Robotics s44182-025-00048-x | yes, animal → quadruped | reward classifier / keypoints / retargeted joints | no | locomotion |
| AdaWorld (ICML 2025), UniVLA, LAPA | partly (human video for pretraining only) | latent action | mixed | manipulation |
| ManyQuadrupeds 2310.10486, Multi-Loco 2506.11470, 50-robot policy 2509.02815 | no | raw velocity command / shared action space | no | locomotion |

**What the search found:**
- **Goal from another body:** it exists in manipulation (AnyWorld, UniSkill, XIRL). In locomotion it exists for human/animal → quadruple, through retargeting or reward learning, **not through a world model**.
- **Not found:** legged → legged across different leg counts from an egocentric view.
- **Froude:** not found as a shared signal for latent actions or world models. Multi-body locomotion uses raw velocity commands. Froude appears in biomechanics as dynamic similarity, and in gait-design work (e.g. 2412.09440).
  - This is absence of evidence from a short search.
  - Still to search: "dynamic similarity" + cross-embodiment, and animal-to-robot retargeting that scales root motion by leg length.

## 4. AnyWorld and 2609.19846, read in full
Read through the arXiv HTML using WebFetch, which returns a summary, not the raw text. Details that could not be recovered are marked as missing.

### AnyWorld: "Factorized Egocentric World Models for Cross-Embodiment Generalization", arXiv 2608.29242
- **Setting:** manipulation only.
  - Bodies: human (EgoDex), RoboCasa GR1 (sim) and the IRON real humanoid.
  - Egocentric video. Pretrained on 200K human clips, then fine-tuned on 5K human + 5K robot clips per body.
- **Model:** a pixel-space video diffusion model (from WAN Fun-Control 14B). **No latent action.**
  - The action is a rendered skeleton / wrist-trajectory video.
  - Camera: Plücker rays.
  - Embodiment: a text tag plus the first frame.
- **"Same action" across bodies:** defined by hand as the skeleton / wrist trajectory. There is no alignment loss. Human → robot uses an action-calibration module (retargeting; details in the supplement, which could not be retrieved).
- **Measurements:**
  - ActionAlign 0.659, CameraAlign 0.789, EmbodAcc 0.886 (Table 1).
  - Generated rollouts used to train a VLA: GR1 49.8 → 54.6%, IRON 20 → 55% (Table 4).
- **Relevant ablations:**
  - Relabelling actions alone does not work; it needs visual recomposition as well (Table 5).
  - There is no ablation of a frame-seeing head vs a latent-only head.
- **Relevance:** low. It is a generative data-augmentation model with an explicit action channel. It shares with us the idea of separating a body-agnostic action signal from an embodiment context.

### "Improving Cross-embodiment Transfer in Latent Action Models with Action-Similarity Supervision", arXiv 2609.19846
- **Setting:** RoboTwin 2.0 sim, bimanual manipulation.
  - Two robots: aloha-agilex and franka. Head camera, fixed, not truly egocentric.
  - Each robot has 3 tasks the other robot never sees (50 demos each). The test is each robot running the other's tasks, 50 episodes per pair.
- **Model:**
  - Latent action models trained from scratch: AdaWorld (d=64) and LAOM (d=1024, forward model in embedding space, closest to ours).
  - A π₀ VLA predicts the latents.
  - A small per-body decoder maps the latent to that robot's actions.
- **Alignment signal (§III-B, Eqs. 4–7):**
  - S^z = cosine similarity between latents in the batch.
  - S^a = cosine similarity between action sequences, where each sequence is per-step deltas, standardised per dimension, max over time shifts of ±2.
  - L_sim = ||M ⊙ (S^z − S^a)||²_F, with the diagonal masked out. Total loss L = L_recon + 0.05·L_sim.
  - Best variant: S^a computed on **end-effector deltas** and **including cross-robot pairs** (sim-EE-X).
- **Measurement:** closed-loop transfer success only. No probe or retrieval test.

  | variant | transfer success |
  |---|---|
  | baseline | 24.0% |
  | + sim on joint deltas | 46.0% |
  | + sim on end-effector deltas | 53.0% |
  | + sim on end-effector deltas, cross-robot pairs | **61.3%** |

  - Similarity supervision beats an auxiliary action-prediction head given the same labels (decoder MSE 0.003 vs 0.015).
  - Adding wrist cameras: own-task success 69.3 → 71.3%, **transfer 61.3 → 46.0%**.
  - With 10% of the labels, transfer is 65.7%.
- **Relevance: high, and the method is directly adoptable.**
  - Replace S^a with the similarity of Froude-normalised body-velocity sequences, including hexapod × B1 pairs.
  - The loss acts on z alone and never sees the frame. It is a relational form of the z-only shared head (F57), and it constrains the geometry of z instead of forcing a regression.
  - The wrist-camera result matches our measurement that egocentric frames identify the body (99.8% hexapod vs B1): views that carry more body-specific information transfer worse.

## What remains ours
Relative to all papers checked. Neither Section 4 paper does locomotion, planning over a world model, a frozen V-JEPA2 encoder, or a goal sequence from another body:
- a latent-action world model from egocentric video, for **locomotion across different leg counts**;
- z grounded by a **dimensionless physical signal (Froude)** through **one decoder shared across bodies**, instead of keypoints, rewards or a learned similarity;
- reproducing **another body's motion** in a closed loop;
- Froude (dynamic similarity) as the definition of "the same action" across bodies. 2609.19846 uses end-effector deltas for manipulation.

**Scope:** Froude matches body motion (speed and turning), not gait style or contact pattern. The claim is "reproduce the body motion", not "reproduce the gait".
