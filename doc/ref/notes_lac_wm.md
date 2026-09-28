# LAC-WM: Latent Action Robot Foundation World Models for Cross-Embodiment Adaptation

Source: "Latent Action Robot Foundation World Models for Cross-Embodiment Adaptation" — anonymous, under double-blind review at ICLR 2026. PDF: `doc/ref/LATENT ACTION ROBOT FOUNDATION WORLD MODELS FOR CROSS-EMBODIMENT ADAPTATION.pdf` (19 pages incl. appendix/refs).

## 1. Core Claim / Research Question

Problem: robot world models (action-conditioned video prediction models) need to generalize across the huge diversity of robot embodiments (different morphologies, action spaces, sensors). Building a **unified latent action space** shared across embodiments (including humans) is proposed as the solution, instead of conditioning directly on embodiment-specific explicit action labels (raw motion/joint values), which creates **disjoint** action spaces per embodiment.

Two research questions:
1. Do latent action models better unify action representations across diverse embodiments compared to directly using action labels?
2. Does conditioning a world model on a unified latent action space enhance cross-embodiment learning and downstream robot task planning performance compared to a disjoint latent space?

Headline results:
- LAC-WM (latent-action-conditioned) vs. EAC-WM (Explicit Action Conditioned World Model, baseline): LAC-WM gets up to **46.7% improvement in downstream task success rate** over EAC-WM on a dexterous manipulation task.
- LAC-WM's downstream planning performance **scales positively** with number of pretraining embodiments; EAC-WM's **scales negatively** (disjoint action space gets worse as more embodiments are added) — this is the paper's central empirical finding.
- Unified latent space also improves image/video generation quality (PSNR/FID/FVD) over EAC-WM.

## 2. Full Architecture (LAC-WM)

Four components: **Inverse Dynamics Model (IDM)**, **Forward Dynamics Model (FDM)**, **Motion Decoder (MD)**, **Action Projector** (finetuning-time only). All trained jointly end-to-end during pretraining except the Action Projector, which is added afterward.

### 2.1 Visual encoder / tokenizer
- Pretrained **V-JEPA2 (1B params)**, used as an RGB image encoder producing visual embeddings `x_t` from raw observations `O_t`. **Frozen** during both pretraining and finetuning of LAC-WM/EAC-WM.
- Input embeddings: 16×16 tokens, each of dimension 1408, for a 256×256 image.

### 2.2 Inverse Dynamics Model (IDM)
- Input: two consecutive visual embeddings `x_t`, `x_{t+1}`.
- Output: latent action `z_t = IDM(x_t, x_{t+1})`.
- Architecture: `M` **causal attention layers** producing contextualized representations of `x_t`/`x_{t+1}`, followed by `N` **cross-attention layers** that transform a learned query token `q_t` into the latent action `z_t` by cross-attending to the contextualized representations.
- Hyperparameters (Table 4): **4 attention blocks**, **16 attention heads**, **latent dimension 512**, **46,939,968 parameters** (~46.9M).
- Latent action embedding dimension used for conditioning downstream (action embedding dim): **64**.

### 2.3 Forward Dynamics Model (FDM)
- Input: current visual embedding `x_t` and latent action `z_t`.
- Output: predicted next visual embedding `x̂_{t+1} = FDM(x_t, z_t)`.
- Architecture: **L blocks** composed of self-attention on `x_t`, self-attention on `z_t`, and cross-attention from `x_t` to `z_t`, to predict `x̂_{t+1}`.
- Hyperparameters (Table 4): **8 attention blocks**, **16 attention heads**, **latent dimension 512**, **93,844,480 parameters** (~93.8M).
- During pretraining, temporal attention in the FDM is **removed for the first 60,000 of 80,000 iterations** (to encourage strong conditioning on actions, following Chen et al. 2024), then the final 20,000 iterations **add temporal attention with a temporal horizon of 8**.

### 2.4 Motion Decoder (MD)
- Purpose: an **auxiliary loss** that decodes latent actions back into explicit motion labels, to mitigate the tendency of continuous, unconstrained latent action spaces to learn "shortcut" solutions (e.g., just encoding perceptual/optical-flow-like features) — follows Nikulin et al. (2025) / Liang et al. (2025). Unlike prior work that uses quantization or low-dimensional bottlenecks to limit representational capacity, LAC-WM keeps latent actions continuous and instead regularizes via this decoding loss.
- Mechanism: `z_t` serves as a **query in a cross-attention module** over visual tokens extracted from the **current frame** (conditioned on current observation, to enable viewpoint-dependent predictions). Visual tokens are first **downsampled via a 2D convolutional layer** to reduce compute. Cross-attended features → MLP → motion output: `â_t = MD(x_t, z_t)`.
- Loss: MSE, `L_motion = ||â_t - a_t||²_2`.
- Ground-truth motion targets: delta human hand poses, robot end-effector actions, and camera motion (see Section 5, data details below).
- For LAC-WM's motion decoder output, the latent action vector is split evenly in half: **first half decodes end-effector pose, second half decodes camera pose**.

### 2.5 Action Projector (finetuning-only component)
This is the mechanism the project cares about most — see Section 5 below for full detail.

### 2.6 Overall pretraining loss
```
L = λ_recon * L_recon + λ_motion * L_motion
```
- `L_recon = ||x̂²_{t+1} - x²_{t+1}||²_2` — image latent (self-supervised) prediction loss, computed via the **cross-augmentation scheme** (below), not a plain reconstruction of the same embeddings the IDM saw.
- `L_motion` — motion decoding loss (Section 2.4).
- The paper does not give exact numeric values of λ_recon / λ_motion in the main text or appendix (not specified in Table 4 or elsewhere read).
- IDM, FDM, MD are **trained jointly end-to-end** during pretraining. After pretraining, only IDM+FDM (FDM is kept) are retained; an Action Projector is added for finetuning to adapt to an unseen embodiment.

### 2.7 Cross-Augmentation Input scheme (regularization against IDM shortcut)
Because the IDM is only partially supervised by the image latent prediction loss, it could encode future information into `z_t` as a shortcut. To mitigate:
- Given observations `O_t`, `O_{t+1}`, apply two **independent augmentations** `A_1`, `A_2` to get two augmented embedding pairs `(x_t^1, x_{t+1}^1)` and `(x_t^2, x_{t+1}^2)`.
- IDM uses pair 1 to produce `z_t^1 = IDM(x_t^1, x_{t+1}^1)`.
- FDM uses the *other* augmented pair's current-frame embedding plus that latent action to predict: `x̂_{t+1}^2 = FDM(x_t^2, z_t^1)`.
- `L_recon = ||x̂_{t+1}^2 - x_{t+1}^2||²_2`.
- This decouples the IDM's z-encoding from simply copying/leaking the exact target embedding the FDM must predict.

### 2.8 Image decoder (for visualization only, not part of core world model loss)
- Custom **Vision Transformer-based decoder** to reconstruct pixels from V-JEPA2 embeddings.
- Pretrained on DROID, then finetuned on the BFA (finetuning) dataset.
- Projects input embeddings (16×16 tokens, dim 1408 each, for 256×256 image) into a **1024-dimensional decoder space**, followed by **12 self-attention layers, 32 attention heads each**, then linearly mapped back to pixel space.
- Trained with standard MSE loss.
- Parameter count (Table 4): **153,845,952** (~153.8M).
- This decoder is NOT used for control/planning — only to visualize predicted embeddings as RGB frames for qualitative figures.

### 2.9 Parameter summary table (from Table 4, Appendix A.5)
| Component | Key hyperparams | Params |
|---|---|---|
| Image Tokenizer | V-JEPA 2 (1B) | — |
| Action Projector | — | 1,550,528 (~1.55M) |
| Image Decoder | 1024-d decoder space, 12 layers, 32 heads | 153,845,952 |
| IDM | 4 attn blocks, 16 heads, latent dim 512 | 46,939,968 |
| FDM | 8 attn blocks, 16 heads, latent dim 512 | 93,844,480 |

## 3. Baseline: EAC-WM (Explicit Action Conditioned World Model)

- Same FDM architecture as LAC-WM, but action conditioning comes from directly encoding explicit motion/action labels (robot end-effector actions, delta human hand poses) rather than a learned latent action.
- Because different embodiments have distinct (disjoint-dimensional) action spaces, EAC-WM uses **separate action encoders per embodiment** (following common multi-embodiment practice, cites NVIDIA GR00T N1, 2025).
- Each action encoder shares the same architecture as LAC-WM's action projector: takes only the explicit action as input, outputs the corresponding action embedding.
- To adapt EAC-WM to an unseen embodiment: train a **new action encoder from scratch** and fine-tune the FDM using **LoRA rank 2**.
- Also compared: **EAC-WM-S** — EAC-WM trained **from scratch** directly on the finetuning (BFA) dataset, i.e. no cross-embodiment pretraining at all (ablates the value of pretraining).

## 4. Data / Pretraining Datasets

Three pretraining datasets, cross-embodiment:
1. **EgoDex** (Hoque et al., 2025) — humans manipulating different objects in a tabletop environment, egocentric head camera. Human hand pose annotations used as motion labels.
2. **Agibot** (Bu et al., 2025a) — bimanual humanoid robot ("Agibot World Colosseo"), egocentric head camera. Robot end-effector action labels.
3. **Droid** (Khazatsky et al., 2025) — single Franka arm with Robotiq gripper, third-person view camera. Robot end-effector action labels.

- **50,000 trajectories sampled from each dataset** for pretraining.
- Actions are **chunked into 5-step sequences** to down-sample observation frequency (found to improve world model learning in practice).
- Action space dims (Appendix A.2):
  - Wrist pose: 9-dim (3 translation + 6 rotation) baseline unit for all.
  - **Agibot**: end-effector pose = 20-dim total (10 per arm: 9-dim wrist pose + 1 gripper-state dim), bimanual.
  - **Droid**: end-effector pose = 10-dim (single arm: 9-dim wrist pose + 1 gripper dim).
  - **EgoDex**: end-effector pose = 9-dim wrist pose + 60-dim finger positions (20 keypoints × 3D) = 69-dim per hand → **138-dim total for both hands**.
  - Camera motion: 9-dim change-of-pose for EgoDex and Agibot; **set to zero for Droid** (third-person camera is fixed per episode).
  - End-effector pose and camera-motion vectors are concatenated as EAC-WM's explicit action encoder input.

### Pretraining hyperparameters
- Both EAC-WM and LAC-WM: **80,000 iterations**, **batch size 512**.
- Action embedding dimension: **64** (both models).
- V-JEPA2 RGB tokenizer: **frozen** during pretraining and finetuning.
- FDM temporal attention: removed for first 60k iters, added back (temporal horizon 8) for final 20k iters.
- Training cost: **~4 days on 64 H200 GPUs**.
- Inference time: **~0.1 sec per batch of 10, predicting 8 future frames at 256×256**, on an NVIDIA L40S GPU.

## 5. Finetuning / Adaptation to an Unseen Embodiment — THE ACTION PROJECTOR (core mechanism)

### 5.1 The problem being solved
The IDM requires a **future frame** (`x_{t+1}`) to compute a latent action, which is unavailable at control/inference time (you only have the current observation, not the future one you're trying to reach). Prior approaches address this by training a **policy conditioned on current observations to directly generate latent actions**, but this does not support planning with a raw-action-conditioned world model (i.e., you can't feed it an explicit candidate action to roll out). Instead, LAC-WM learns an **action projector** that maps explicit raw actions into the latent action space, following Gao et al. (2025) — this is the exact idea the project's own docs cite.

### 5.2 Action Projector architecture
- A **two-layer MLP**.
- Input: raw explicit action `a_t`.
- Output: corresponding latent action `z_t` (same 64-dim latent action space the IDM produces).
- Parameter count: **1,550,528 (~1.55M)** — small, per Table 4.
- Architecturally the **same shape as EAC-WM's per-embodiment explicit action encoder** (for fair comparison) — the difference is what target space it maps into (unified latent z space for LAC-WM vs. a disjoint per-embodiment embedding space for EAC-WM).

### 5.3 Adaptation training procedure — THREE STAGES (explicit staged pipeline)
To adapt pretrained LAC-WM to an unseen embodiment, initialize a fresh action projector and finetune in three stages:
1. **Stage 1**: Fine-tune the **IDM and FDM** of pretrained LAC-WM **end-to-end using LoRA rank 2**. (Action projector not yet involved — this stage adapts the core latent-action world model to the new embodiment's visual/dynamics distribution, still using IDM to generate z from frame pairs.)
2. **Stage 2**: **Freeze the FDM**, train the **action projector from scratch** to map explicit actions into the (now-adapted) latent space.
3. **Stage 3**: **Jointly fine-tune the projector and FDM end-to-end using LoRA rank 2**.
- At **inference time**, only the **action projector + FDM** are used to perform action-conditioned imagined rollouts (IDM is not needed at control time — this is the whole point).
- The IDM, FDM, MD are trained jointly end-to-end during original pretraining; only after that pretraining is the action projector introduced for this 3-stage finetuning.

### 5.4 Finetuning dataset and iteration counts (BFA task)
- Target unseen embodiment: **BFA** = bimanual Franka robot setup equipped with **Allegro hands**.
- Task: pick-and-place — lifting an object from a kitchen counter and placing it back down.
- Dataset: **7,265 trajectories** collected in the **RoboCasa** simulator (via MimicGen, 10 teleoperation trajectories used to generate synthetic data), covering **22 object categories** (e.g., apple, bell pepper, ketchup), including both successful and failed attempts (**20% success rate** trajectories, i.e., ~1,453 successful / ~5,812 failed).
- Grasping/placement rule: pick up object, place it left or right of its initial position with minimum 0.1 m movement.
- Only the **right robot arm** is used for actions. Action space is **25-dimensional**: 16 joint dims (robot hand) + 3 dims end-effector position + 6 dims rotation.
- Egocentric visual observations from a **head camera**; actions again chunked into 5-step sequences.
- Finetuning: **batch size 256, total 60,000 iterations**.
  - LAC-WM 3-stage split: **Stage 1 = 20k iters, Stage 2 = 5k iters, Stage 3 = 35k iters** (20k+5k+35k = 60k).
  - EAC-WM: finetuned for **60k iterations in a single stage** (new action encoder + FDM LoRA rank-2, done together) — the paper explicitly notes they found **directly fine-tuning EAC-WM in one stage yields better downstream performance than a multi-stage approach** for EAC-WM (Section A.6 ablation, described further below).

### 5.5 Data efficiency for adapting to a new embodiment — explicit check against "one clip clears break-even"
**The paper does NOT state a "one clip" or single-example data-efficiency figure anywhere in the text I read (full 19 pages, main text + appendix).** There is no sentence about minimum number of examples/clips needed to "break even" or any cost-per-example economics discussion. The paper's adaptation data efficiency evidence is instead indirect, via:
- The **finetuning dataset size is 7,265 trajectories** (not few-shot) for adapting to the BFA embodiment — this is a full finetuning run, not a few-shot/one-clip experiment.
- The **scaling law study** (Section "Scaling Law with Number of Embodiments and Data", Figures 4–5) is the closest thing to a data-efficiency result: it shows how *pretraining* embodiment/data count affects *downstream* adaptation performance (see Section 8 below), not how many finetuning examples of the new embodiment are needed.
- **Conclusion for the project**: the "one B1 clip clears break-even" cost figure referenced in this project's own docs is **not sourced from this LAC-WM paper** — it must come from a different analysis (likely the project's own internal cost/compute estimate, or another paper). This paper does not support or contradict that specific number; it simply doesn't address it. Do not attribute that figure to LAC-WM.

## 6. Evaluation Methodology

Two main evaluation axes, both using held-out BFA splits:
- **Unseen Instances**: novel object instances within object categories seen during BFA finetuning.
- **Unseen Categories**: **3 entirely unseen object categories** — donut, lime, banana — out of the 22 total BFA categories (so 19 seen + 3 unseen; text also phrases it as "22 unseen instances from 22 seen categories and 3 unseen categories" for the planning eval in Section 6.2).

### 6.1 Action-conditioned imagined roll-out (video/image quality, Table 1)
- Predicts **8 future frames** given an initial observation + action sequence.
- Metrics: **PSNR↑, LPIPS↓, FID↓** (image quality), **FVD↓** (video quality), computed over **512 randomly sampled windows per split**.
- Results (Table 1):

| Model | Unseen Instances PSNR / LPIPS / FID / FVD | Unseen Categories PSNR / LPIPS / FID / FVD |
|---|---|---|
| EAC-WM-S | 22.212 / 0.077 / 10.648 / 42.257 | 22.062 / 0.081 / 10.642 / 40.477 |
| EAC-WM | 24.252 / 0.057 / 9.312 / 30.349 | 24.091 / 0.060 / 9.144 / 29.702 |
| **LAC-WM** | **27.484 / 0.037 / 8.439 / 23.213** | **27.333 / 0.040 / 8.509 / 24.456** |

LAC-WM best on every metric, both splits. Ordering: LAC-WM > EAC-WM > EAC-WM-S, confirming pretraining + unified action space both help.

### 6.2 Robot planning with action selection (Table 2/3/5, Figs 7–9)
- Setup: lifting an object from kitchen counter, placing back, using right arm of BFA setup.
- Planning procedure: given a demo trajectory, sample a **subgoal image every p timesteps**. For each subgoal, sample **N candidate action sequences of length p**; roll out each with the world model from the current observation; select the sequence whose final predicted image is closest (L2 in embedding space) to the subgoal image; execute; repeat until episode ends.
- In practice, to reduce noise: select the **top-k** action sequences by final-image closeness and **execute the average of those k sequences**.
- Metrics:
  - **δf** ↓ — average embedding distance between each subgoal image and the achieved image at each rollout step (subgoal-following accuracy).
  - **δf_g** ↓ — embedding distance for the *final* goal specifically.
  - **S.R.C.** ↑ — Success Rate of Contact (robot touched the object).
  - **S.R.L.** ↑ — Success Rate of Lifting (object lifted without touching counter).
  - **S.R.** ↑ — overall Task Success Rate (object successfully lifted and placed back by end of episode).

**Action selection over VLA-generated candidates (Table 2, headline numbers)**: an off-the-shelf VLA trained on BFA (diffusion action head, architecture in A.7) generates candidate actions; world models (EAC-WM-S / EAC-WM / LAC-WM) select among **N=500** candidate sequences (length p=100, ~4–5 subgoals per episode, 20-step rollout per world-model evaluation window), using **k=3** for averaging. Compared against VLA-only baselines **VLA-mean** (average of all N candidates) and **VLA-random** (single random candidate). 100 episodes per split, 3 random seeds.
- **LAC-WM achieves 29.0% higher S.R. than the strongest VLA baseline, 22.2% improvement over EAC-WM-S, and 46.7% improvement over EAC-WM**, averaged across splits (this 46.7% figure matches the abstract's headline number).
- Table 2 numbers (S.R., averaged over splits, "Average" row): VLA-mean 0.17±0.03, VLA-random 0.15±0.02, EAC-WM-S 0.18±0.02, EAC-WM 0.15±0.04, **LAC-WM 0.22±0.02** (best). δf/δf_g also best for LAC-WM.
- Note: **overall absolute success rates remain low** (~0.10–0.25 range) — task is a genuinely hard dexterous manipulation problem; the paper is explicit that absolute performance is still poor even for the best model (see Limitations, Section 7).
- EAC-WM **underperforms even the VLA-only baselines** in this setting, attributed to its disjoint action space limiting reliable long-horizon future prediction.

**Action selection over training-set action sequences (Table 5, Appendix A.6)**: same idea but candidate actions sampled from the training dataset (not VLA) to avoid VLA-sampling bias. N=500, p=30, k=5 → 12–15 subgoals per episode, 6-frame rollout per subgoal chunk (5-step action chunking). 50 episodes/split × 3 seeds × 3 repeats.
- Compares finetuning-strategy variants: EAC-WM-S, **EAC-WM-TFT** (two-stage: freeze FDM & finetune action encoder, then LoRA end-to-end), EAC-WM, **LAC-WM-DFT** (train action projector from scratch + LoRA finetune FDM from scratch, no staged freeze), **LAC-WM-(2,3)** (skip stage 1: train projector from scratch with FDM frozen, then jointly finetune projector+FDM with LoRA), **LAC-WM-(1,2)** (do stages 1+2 only: end-to-end LoRA finetune of IDM+FDM, then train projector from scratch+finetune FDM with LoRA — no final joint stage), and full **LAC-WM** (all 3 stages).
- Full 3-stage **LAC-WM performs best overall** (S.R. 0.10–0.12 in unseen categories/instances, vs. lower for the ablated variants), confirming the value of the full 3-stage recipe over any 2-stage subset. For EAC-WM, however, the earlier text (Section 4.3) says the single-stage direct finetune outperformed a multi-stage approach — i.e., **staging helps LAC-WM but not EAC-WM**.

## 7. Latent Action Analysis (Section 5 of paper) — the "shared latent action space" evidence

This is the section most relevant to the shared-coordinate-space problem.

### 7.1 UMAP visualization (Figure 2)
- 7,000 action embeddings sampled from validation splits of the 3 pretraining datasets (Droid, Agibot, Egodex), visualized with UMAP, comparing:
  - **IDM (full LAC-WM)**: embeddings from all 3 datasets **cluster closely together, occupying a unified action space** — strong cross-dataset alignment.
  - **IDM w.o. MD** (IDM trained *without* the motion-decoder auxiliary loss): Agibot and Egodex cluster together but **separate from Droid**. Hypothesis: Agibot and Egodex are both egocentric-view datasets → similar visual distributions, so a purely visually-grounded (unsupervised) latent action ends up encoding perceptual/viewpoint features rather than task-relevant motion, producing disjoint spaces when the visual distribution changes (Droid is third-person). This is direct evidence that **the motion-decoding loss is what makes the space embodiment-invariant, not the IDM/continuous-latent design alone**.
  - **EAE (Explicit Action Encoder, the EAC-WM baseline)**: embeddings are **clearly separated by dataset** — no shared representation at all.

### 7.2 Action Latent Transfer (Figure 3, qualitative)
- Test: take an 8-frame reference video from one embodiment, extract latent action embeddings via IDM, then condition the FDM on a *different* embodiment's current observation using that latent action (cross-embodiment transfer, e.g., robot-to-human and human-to-robot, tested with Agibot ↔ human).
- Result: **the same latent action moves the robot's right end-effector and the human's right hand in the same direction** (e.g., both move rightward and upward) — i.e., the latent action encodes the **end-effector motion direction in image space**, transferring semantically across embodiments.
- EAE-based transfer **fails** — produces nearly static human hands / robot grippers (explicit actions don't transfer because they're embodiment-specific dimensionalities/semantics).
- IDM-without-MD transfer produces artifacts (duplicated hands, floating objects) — motion decoding loss is again shown to be necessary to avoid encoding embodiment-specific perceptual shortcuts (e.g., optical flow) instead of true motion.

### 7.3 Takeaway stated by the paper
"Latent action models do better unify action representations across diverse embodiments than directly using action labels on heterogeneous datasets. Moreover, incorporating a motion decoding loss further enhances the coherence of the latent action space."

## 8. Scaling Law: Number of Embodiments / Data (Figures 4–5)

- Pretrain both LAC-WM and EAC-WM on 3 nested subsets: (1) EgoDex only, (2) EgoDex+Agibot, (3) EgoDex+Agibot+Droid — each dataset always contributing 50,000 samples, so total pretraining data scales with number of embodiments (1×50k, 2×100k, 3×150k). Then finetune each on BFA and evaluate via VLA action selection (same setup as Table 2).
- **Figure 4 results** (averaged over 3 seeds):
  - S.R.: LAC-WM improves monotonically as embodiments increase — 16.5% (1 embodiment) → 17.2% (2) → 21.5% (3).
  - EAC-WM **degrades**: 16.5% → 15.5% → 15.0%.
  - δf and δf_g show the same pattern (LAC-WM improves/holds, EAC-WM worsens) as embodiments increase.
- **Hypothesis given**: as embodiment count increases, EAC-WM's action embedding space becomes **more disjoint**, making cross-embodiment pretraining harder to leverage and less effective for adapting to the new BFA embodiment. LAC-WM's unified space instead benefits from more diverse pretraining data.
- **Figure 5** — disentangling "more embodiments" vs. "more data": two settings — (a) fixed total sample budget while increasing embodiment count (first two points: 2/24k, 3/24k), (b) fixed embodiment count while increasing dataset size (last two points: 3/24k, 3/150k).
  - In **both** scenarios, LAC-WM shows **consistent performance improvement** as either embodiments or samples increase (S.R.: 16.08% → 17.67% → 18.33%).
  - EAC-WM **degrades when adding more embodiments** (17.67% → 14.00% at 3/24k) and **only improves when the additional data comes from the same embodiment** (14.00% → 18.00%? — paper states EAC-WM improves only when data increase is same-embodiment, i.e., disjoint-space penalty is specifically about *embodiment diversity*, not data volume per se).
- **Overall conclusion**: "EAC-WM cannot efficiently leverage heterogeneous action spaces, further highlighting the advantage of learning a unified latent action representation." This is the paper's strongest evidence that a shared/unified latent action space is what enables beneficial cross-embodiment pretraining scaling, whereas explicit/disjoint per-embodiment action spaces actively hurt as you add more embodiments.

## 9. Ablations

### 9.1 Motion Decoding loss + Cross-Augmentation Inputs ablation (Table 3)
- Compare full LAC-WM vs. **LAC-WM-MD-CA**: a variant trained **without the motion decoding loss or cross-augmentation** inputs, on the Unseen Categories validation split (3 seeds).
- Table 3 results: LAC-WM-MD-CA — δf 26.98, δf_g 27.80, S.R.C. 0.83, S.R.L. 0.46, S.R. 0.13. LAC-WM (full) — δf 26.73, δf_g 27.58, S.R.C. 0.85, S.R.L. 0.56, **S.R. 0.18**.
- **LAC-WM achieves a 38% higher success rate** than the ablated variant → both motion decoding and cross-augmentation are "crucial for learning latent actions that encode physically meaningful control information."

### 9.2 Finetuning-strategy ablation (Table 5 / Appendix A.6, described in Section 6.2 above)
- Confirms the full 3-stage adaptation recipe is best for LAC-WM; single-stage direct finetuning is best for EAC-WM (staging isn't universally beneficial — it specifically helps the unified-latent-space model).

## 10. Vision-Language-Action (VLA) model used for candidate-action generation (Appendix A.7)

Used only to generate candidate action sequences for the action-selection experiments (Table 2, Section 6.2) — not itself a contribution, but relevant if the project needs a comparable VLA baseline spec:
- **Visual encoder**: pretrained **DINOv2** ViT, frozen during policy training.
- **Language encoder**: pretrained language encoder, frozen, embeds task instructions.
- **Proprioception and action encoders**: transformer-based, trained from scratch; encode end-effector positions + finger joint angles, input dim **25** (9-dim EE pose + 16 finger joint angles).
- **Multi-modal transformer trunk**: fuses vision/language/proprioception/previous-action embeddings, **3 layers**, trained from scratch.
- **Action decoder**: **diffusion head**, action space = future EE pose + finger joint angles (25-dim), predicted over an **action horizon of 100** steps, trained with diffusion loss for diverse high-quality action sequences.
- Deployment: predicts **100-step action chunks at 15 Hz control frequency**, executes the whole chunk before replanning.
- Training data: **2,000 successful demonstrations** of the robot picking and placing objects (on BFA).

## 11. Latent Action PCA Analysis (Appendix A.8, Figure 10)

- PCA on latent action sequences (length 4) extracted from all 3 pretraining datasets (Agibot, Droid, EgoDex).
- Project onto first 2 principal components; **K-means clustering with K=4** to find distinct action clusters.
- For each cluster, pick representative samples per dataset closest to the centroid; visualize start/end frames.
- Result: clusters correspond to **semantically meaningful actions consistent across embodiments** — labeled clusters: "Move Left," "Move Right," "Open Gripper," "Close Gripper." This is further qualitative confirmation that the unified latent space captures embodiment-invariant, human-interpretable motion primitives.

## 12. Limitations (stated explicitly by the paper, Section 7)

- **Absolute task success rate remains low** across all models/settings (roughly 0.10–0.25 range even for best LAC-WM configuration) — the dexterous manipulation task (bimanual Franka + Allegro hands, pick-and-place) is genuinely hard, and the paper is explicit that this indicates significant room for future improvement (better VLA policies, more robust planning) rather than a solved problem.
- **No real-robot experiments** — all evaluation is in the RoboCasa simulator. The paper explicitly flags the **sim-to-real gap** as a limitation for practical deployment; future work will incorporate real-world data collection and hardware validation.
- The choice to pretrain on datasets that *do* have motion labels (rather than fully label-free video) is called out as a deliberate design choice for controlled comparison against the explicit-action baseline (EAC-WM needs labels too) — not a fundamental limitation of the latent-action approach itself, but the paper notes that combining **label-free video data with labeled cross-embodiment data** during pretraining is a promising future direction, since the motion-decoding loss is compatible with (and likely would benefit from) unlabeled video.
- Exact λ_recon / λ_motion loss-weighting values are not disclosed in the paper text I read.

## 13. Key citations relevant to the project's own methodology

- **Action Projector idea**: Gao et al., 2025 — "AdaWorld: Learning adaptable world models with latent actions" (arXiv:2503.18938). This is the paper LAC-WM cites for the action-projector mechanism itself (LAC-WM applies/extends it to the cross-embodiment pretraining + unified latent space setting; Gao et al. focus on action-free video pretraining and lack comparison with explicit-action-label baselines, per LAC-WM's related-work section).
- **Motion decoding auxiliary loss**: Nikulin et al., 2025 — "Latent action learning requires supervision in the presence of distractors" (arXiv:2502.00379); also Liang et al., 2025 (CLAM, arXiv:2505.04999).
- **Cross-augmentation training scheme**: adapted from Chen et al., 2024 ("Igor: Image-goal representations are the atomic control units for foundation models in embodied AI," arXiv:2411.00785) — same source used for the FDM's staged temporal-attention removal/addition schedule.
- **V-JEPA2**: Assran et al., 2025 (arXiv:2506.09985) — the frozen visual encoder/tokenizer (1B params) used throughout.
- **LoRA**: Hu et al., 2021 (arXiv:2106.09685) — used at rank 2 for all embodiment-adaptation finetuning of FDM (and IDM in Stage 1).
- **RoboCasa / MimicGen**: Nasiriany et al., 2024 (RoboCasa, arXiv:2406.02523); Mandlekar et al., 2023 (MimicGen) — used to generate the BFA finetuning dataset.
- Pretraining datasets: EgoDex (Hoque et al., 2025, arXiv:2505.11709), AgiBot World Colosseo (Bu et al., 2025a, arXiv:2503.06669), DROID (Khazatsky et al., 2025, arXiv:2403.12945).
