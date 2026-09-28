# Egocentric Visual Self-Modeling for Autonomous Robot Dynamics Prediction and Adaptation

Source paper: Hu, Chen & Lipson, *npj Robotics* 3:14 (2025), https://doi.org/10.1038/s44182-025-00031-6
Reference code: `/home/fibo07/ioon/ioon-research/doc/ref/Egocentric_VSM/` (public repo: github.com/H-Y-H-Y-H/Egocentric_VSM)
This note is a self-contained reference — cross-checked against the actual code, not just the paper prose.

---

## 1. Core research question / claim

**Claim:** A legged robot can learn a *self-model* (a forward dynamics model of its own body) using **only a single first-person-view (egocentric) RGB camera plus motor commands** — no IMU, no external mocap/optical tracking, no CAD/kinematic model, no proprioceptive pose sensors. This is called "egocentric visual self-modeling" (EVSM), as opposed to visual odometry (which estimates camera/world pose, not action-conditioned future dynamics).

**Problem it solves:** Traditional self-models are either analytical (CAD/kinematics-based) or data-driven from IMUs/external cameras — both need prior knowledge of morphology or external infrastructure. EVSM removes both dependencies, using only onboard vision, enabling: (a) zero-shot sim-to-real transfer via domain randomization, (b) generalization/transfer to different robot morphologies by reusing a pretrained visual encoder, (c) autonomous anomaly (damage) detection and self-model retraining/recovery after damage, without human intervention.

**Key contributions (paper's own list):**
1. Legged robot performs locomotion using *purely* egocentric visual observation (no proprioception) via a network fusing image sequences + actions to predict future robot state.
2. A domain-randomization + data-augmentation pipeline enabling zero-shot sim-to-real transfer.
3. Generalization across robots of different morphology/complexity (transfer learning of the visual encoder).
4. Autonomous detection of, and re-training-based recovery from, physical damage (broken leg link), purely from vision.

---

## 2. Architecture

### 2.1 High-level pipeline (Fig. 1 in paper)
```
Motor Babbling Generator ──► Past Actions A_t (12-dim) ──► Robot Execution (sim/real)
                         └─► Next Actions A_{t+1} (12-dim)
   A_t ⧺ A_{t+1} = 24-dim action vector
                                                │
                                    Egocentric Camera
                                                │
                                    Image Sequence: 5 × 128 × 128 (grayscale)
                                                │
                              ┌─────────────────┴─────────────────┐
                              │        Egocentric Visual Self-Model│
                              │                                    │
                 Action Encoder (MLP)                 Visual Encoder (CNN → "LSTM")
                 24 → 64 → 128 → 256                   5×128×128 → CNN(ResNet-style) →256
                 (Leaky ReLU each layer)                → replicate ×5 → LSTM(256) → 256
                              │                                    │
                              └─────────► concat (512) ──► MLP (512→128→32→6) ─────┘
                                                │
                                    Next state ŝ_{t+1}: (Δx, Δy, Δz, Δroll, Δpitch, Δyaw)
```
Fig. 1a labels the visual encoder output as "5×50×256" and shows CNN → LSTM → 256-dim; Fig.1a action-encoder MLP dims labelled 24→64→128→256 (paper's Methods text: "Action Encoder maps the action vectors into a high-dimensional latent space... three fully connected layers with Leaky ReLU").

### 2.2 Exact module definitions (from `ResNet_RNN.py`, class `ResNet`, which is the actual `EgocentricVisualSelfModel` used by `main.py`)

- **Visual encoder stem:** `conv1`: Conv2d(in=5, out=64, kernel=7, stride=2, pad=3, bias=False) → BN → ReLU → MaxPool2d(k=3, s=2, pad=1). Input channel count = **5** (the 5 grayscale frames are stacked as 5 input *channels*, not processed as a temporal sequence through the CNN — see §7).
- **Backbone:** standard ResNet-50 bottleneck stages (`layers=[3,4,6,3]`, bottleneck `block` with expansion=4): layer1 (64→256, stride1), layer2 (128→512, stride2), layer3 (256→1024, stride2), layer4 (512→2048, stride2). `ResNet18/ResNet50/ResNet101/ResNet152` constructors exist; **the actual training run in `main.py` (mode 2, "Train VSM") instantiates `ResNet50`** (the `__main__` speed-test block in `ResNet_RNN.py` uses `ResNet18` — that's a standalone timing script, not the trained model).
- **Global pool + FC compress:** AdaptiveAvgPool2d(1,1) → flatten (2048-dim, since ResNet50 layer4 outputs 512*4) → `fc0`: 2048→1024 → `fc1`: 1024→512 → `fc2`: 512→256, each followed by activation (`LeakyReLU` if `ACTIVATED_F="L"`, else `Tanh`; actual training run uses `ACTIVATED_F="L"`).
- **Fake-sequence replication + LSTM (see §7 for the mismatch):** the 256-dim vector `x` is replicated 5× via `torch.cat(5*[x.unsqueeze(0)])` → shape `[5, batch, 256]`, then `torch.transpose(x, 0, 2)` → shape `[256, batch, 5]`. This is fed into `nn.LSTM(input_size=5, hidden_size=256, num_layers=1)`; the **last** output `x[-1]` (shape `[batch, 256]`) is taken as the visual encoder output.
- **Action encoder:** `fc_a0`: 24→64 (or 12→64 if `input_pre_a=False`; actual run has `input_pre_a=True`, so input is 24 = 12 previous joint angles ⧺ 12 next joint angles) → `fc_a1`: 64→128 → `fc_a2`: 128→256, each with the same activation function.
- **Fusion head:** `torch.concat((visual_256, action_256), dim=1)` → 512-dim → `fc3`: 512→128 (+activation) → `fc4`: 128→32 (no activation) → `fc5`: 32→6 (no activation, linear output). Output = 6-dim next-state delta `(Δx, Δy, Δz, Δroll, Δpitch, Δyaw)`.
- **Loss:** `nn.MSELoss`-equivalent, hand-written as `torch.mean((pred - target) ** 2)` — plain MSE over all 6 output dims, matches paper Eq. 4: `L = MSE(f_p(I_t, A_t), S_{t+1})`.
- **Image normalization option:** if `normalization=True` (used in the real training run), input images are rescaled `x_IMG = x_IMG*2-1` (from [0,1] to [-1,1]) before conv1.

### 2.3 Baseline / auxiliary models in the same file
- **`IMU_bl`** (IMU-only baseline used for Fig. 3 "Our Method (Only input IMU)"): no CNN at all — takes a 3-dim state input (fc0: 3→64→128→256) concatenated with the same 24→64→128→256 action encoder → fc3 (512→128) → fc4 (128→num_classes). Pure MLP, no vision.
- **`ov_model`** ("OV_Net", the **Visual Odometry** model used to auto-label anomaly detection / damage recovery, §"Autonomous anomaly identification" in paper): same ResNet-bottleneck backbone but **7-channel input** (7 consecutive frames, no action encoder at all — VO is not action-conditioned, matches paper: "This information is not predictive and does not factor in motor commands"). Same replicate-then-LSTM pattern: `torch.cat(7*[x.unsqueeze(0)])`, `nn.LSTM(input_size=7, hidden_size=256)`. Output: 6-dim `(Δx, Δy, Δz, Δroll, Δpitch, Δyaw)` state-change between the two endpoints of the 7-frame window (used as pseudo-ground-truth pose-change label from vision alone).
- **`RCNN`** (in `model.py`): an alternative from-scratch CNN+LSTM (not ResNet-based, 7 conv layers, in_channels=7, LSTM input_size=7 hidden=256) — appears to be an earlier/alternative visual-odometry architecture, not the one used for the headline EVSM results.

### 2.4 Shapes summary
| Tensor | Shape |
|---|---|
| Image sequence input | `[batch, 5, 128, 128]` (grayscale, 5 frames stacked as channels) |
| Action input (with `input_pre_a=True`) | `[batch, 24]` (12 prev joint angles + 12 next joint angles) |
| Visual encoder output | `[batch, 256]` |
| Action encoder output | `[batch, 256]` |
| Fused | `[batch, 512]` |
| Model output (next state) | `[batch, 6]` = `(Δx, Δy, Δz, Δroll, Δpitch, Δyaw)` |
| VO model image input | `[batch, 7, 128, 128]` |

---

## 3. Hyperparameters (Table 6 in paper + confirmed/contrasted in code)

| Hyperparameter | Paper (Table 6) | Code (`main.py`, `train_in_sim`) |
|---|---|---|
| Optimizer | Adam | `torch.optim.Adam(model.parameters(), lr=lr)` ✓ |
| Initial learning rate | 1e-4 | `lr=1e-4` passed into `start_train_model` (RUN_PROGRAM==2) ✓ |
| Batch size | 128 | **16** in the actual VSM fine-tuning call (`batch_size = 16`, RUN_PROGRAM==2, line ~1079); batch size **128** is instead used for the separate VO-model training call (RUN_PROGRAM==3, line ~1119). Table 6 says it's for "training visual self-model" — code shows 128 is actually the VO/OV_Net batch size, not the VSM's in this default script. |
| Input size | [5×128×128, 24] | matches `ResNet50(... img_channel=5 ..., input_pre_a=True)` |
| Output size | 6 | matches `fc5: 32→6` |
| Hardware | 16 cores CPU + 1 GPU | matches "16-core CPU + GPU" mentioned for Hill Climbing optimization (2h on 16-core CPU) and NVIDIA RTX 3090 mentioned for real-robot deployment/desktop |
| LR decay | "scaled by 0.1 if validation loss does not decrease over **twenty** epochs" (paper prose) | `ReduceLROnPlateau(optimizer, factor=0.1, patience=5)` — **patience=5, not 20**. The "20" in the paper prose actually matches the *early-stopping* counter in code: `if abort_learning > 20: break` (training-loop early stop), which is a **different** mechanism than the LR scheduler. This looks like the paper conflated the two. |
| Epochs | Not explicitly given a max in main text | code caps at `epochs=5000` but early-stops after 20 epochs without validation improvement (`abort_learning`) |
| Loss | `L = MSE(f_p(I_t, A_t), S_{t+1})` (paper Eq. 4) | `torch.mean((pred - target) ** 2)` ✓ matches |
| Normalization of GT | paper: "we normalized the ground-truth value to the same standard deviation" | `normalize_output_data(all_NS, scale=scale)` — per-dimension mean-subtract + scale so each of 6 output dims has similar spread (see `data_enginering.py`: computes per-dim mean and a scale factor so that centered data has Euclidean norm ~10) |
| Sequence length (visual) | 5 frames chosen (ablation Table 4: loss keeps dropping to 4 frames, diminishing returns beyond) | 5 frames hard-coded in `ImgData2` (`for i in range(1,6)`) and in ResNet_RNN `torch.cat(5*[...])` — not actually varied per-experiment in the *main* code path; Table 4 ablation was apparently a separate side-experiment (not directly visible as a flag in `main.py`) |

### Domain randomization / augmentation parameters (paper Table 5, not independently verified line-by-line in code but matches design described in `env.py`/`main.py`):
- Crop range: uniform [0.1, 1] (ratio of original image)
- Rotation range: uniform [−π, π]
- Translation range: uniform [−3, 3]
- Brightness range: uniform [0.1, 10] — matches `ColorJitter(brightness=[0.1, 10])` in `env_agent.py`/`main.py`'s `transform_img`
- Gaussian blur kernel size: uniform [3, 41]; blur std: uniform [0.1, 5] — matches `cv2.GaussianBlur(img, (win_r, win_r), sigmaX=sig_r_xy, sigmaY=sig_r_xy)`, `win_r = 2*randint(1,20)+1` (odd, 3–41), `sig_r_xy = uniform(0.1, 5)` in `main.py`'s `transform_img`
- Motor torque range: uniform [1.5, 2] — matches `action_tq = random.uniform(1.5, 2)` in `env.py.act()`
- Motor position range: N(0, 0.05) — matches `a = np.random.normal(loc=a, scale=0.05)` in `env.py.act()` (`pos_rand_flag`)
- Friction range: uniform [0.5, 0.99] — matches `self.friction = random.uniform(0.5, 0.99)` in `env.py.__init__`

---

## 4. Data

### 4.1 Robot hardware
- 4-legged, **12-DoF** robot (3 motors/leg), 3D-printed + off-the-shelf electronics.
- Actuators: Hiwonder LX-224 serial bus servos, 0.3° accuracy, 20 kg·m torque @ 7.4V.
- Position control from onboard Raspberry Pi 4.
- Front RGB camera: 87°×58° FOV, 60 FPS, aimed 41° down toward the ground.
- Onboard controller + camera connected to a desktop with **1× NVIDIA RTX 3090 GPU** for real-time inference.

### 4.2 Observation / action representation
- Image input: crop 320×240×3 → 240×240×3 → resize to **128×128×3** → converted to **grayscale** (dimensionality reduction for real-time control).
- 5 most recent frames used per timestep (≈83 ms span: 5 frames at 60 FPS ≈ 16.7 ms/frame apart), captured during the previous action's execution window.
- Action vector: 24-dim = 12 previous-step joint angles ⧺ 12 next-step joint angles (position-control targets, normalized to [−1,1] representing joint angles from −90° to 90°).
- State/label: robot-frame delta pose `(Δx, Δy, Δz, Δroll, Δpitch, Δyaw)` — **not** absolute global pose (no GPS-like localization used); defined as the change of position/orientation in the robot's own coordinate frame between consecutive states.
- Sim timestep: each control step runs for `n_sim_steps = 60` PyBullet substeps (**code**, `env.py`), i.e. one action executes over 60 simulation steps.

### 4.3 Data collection
- **Simulation:** PyBullet physics engine, `pybullet` Python module. Initial random-uniform action policy found insufficient (too far from real-motion distribution); switched to CPG-based gait generator + Gaussian noise for data collection (see §5).
- Collected **100,000 steps per texture (A–D)** in simulation to train the visual self-model (paper "Environments details" section). For real-to-sim generalization transfer to new robot configs (robots 1–3), **200k steps** used vs. **400k steps** for the original robot 0.
- Real-world anomaly-recovery retraining: ~30 min motor babbling on the damaged robot → **7000 steps** of data collected to retrain/adapt the self-model to the new (damaged) body configuration.
- Real-world evaluation: 4 tasks (forward, backward, turn left, turn right), each episode = **56 steps**, repeated **N=10** trials per method per task.

### 4.4 Environments / domains (Fig. 8, 7 total: 4 sim + 3 real)
- Simulation ground textures: **A) rug, B) grid/checkerboard lines, C) color dots, D) grass** (rug texture augmented via random crop/rotate/translate of 1 base rug photo into 1000 cropped variants, since only one real rug photo was available).
- Real-world terrains: **E) rug, F) checkerboard, G) rug with scraps of paper** (perturbation/clutter test).
- Sim uses simplified robot CAD (main-body CAD kept identical to physical robot; other parts simplified for compute) and matched simulation timestep count to real motor trajectories to reduce sim-to-real gap.

---

## 5. Gait generator / training data policy (CPG)

- **CPG (Central Pattern Generator)**, one sinusoidal function per motor: `θ_i(t) = a_i·sin(t/T·2π + b_i) + c_i` (paper Eq. 1).
- Parameters `(a_i, b_i, c_i)` optimized via **Hill Climbing**: population of 16 candidate parameter sets/copies, initial random search over 200 steps, then iterative epochs where one of the 16 copies is fully re-randomized each epoch; optimizes forward-locomotion reward `R(Δx,Δy) = Σ(100·Δy − 50·|Δx|)` (paper Eq. 2) over ~2h of parallel compute on a 16-core CPU.
- Training-data action noise: `a_n(i,t) = Gaussian(μ = a_o(i,t), σ)` (paper Eq. 3), with **σ=0.2** used ("providing good performance... robot can reach various states without falling easily").
- Paper explicitly notes: purely-random motor babbling (uniform actions in [−1,1] over ±90°) was tried first and found *ineffective* — didn't match the state distribution needed at deployment; using the CPG-optimized gait + Gaussian noise (rather than pure random exploration) was found to be more sample-efficient for self-supervision. The CPG itself is only used at **training-data-collection time**, not at deployment.

---

## 6. Deployment / control (Fig. 1b) and reward functions

- At deployment, a set of candidate next-actions `A_0...A_i` is proposed; each is combined with the same 5 most recent camera frames and passed through the (frozen) egocentric self-model in parallel on GPU to predict `Ŝ_0...Ŝ_i`; a task-specific reward function scores each predicted next state; the action with max reward `R_m` is executed (classic MPC / model-predictive control with a learned forward model — this is what the paper calls "self-modeling" decoupled from the environment/task model).
- Batch size for candidate action evaluation: tested up to **100 parallel candidate motor commands per model run** on an NVIDIA RTX 3090; system pipeline runs at **~5 Hz**.
- Reward functions per task (paper Eqs. 5–8, simulation reward defs; Eqs. 9–12, real-world reward/score metric with a=b=10):
  - Forward: `maximize f(Δx,Δy) = 2·Δy − |Δx|` (sim, Eq.5); real-world score `r_forward = a·y − b·|yaw|` (Eq.9)
  - Turn left: `maximize f(Δyaw) = Δyaw` (Eq.6); real `r_left = a + b·yaw` (Eq.10)
  - Turn right: `maximize f(Δyaw) = −Δyaw` (Eq.7); real `r_right = a − b·yaw` (Eq.11)
  - Backward: `maximize f(Δx,Δy) = −2·Δy − |Δx|` (Eq.8); real `r_backward = −a·y − b·|yaw|` (Eq.12)

---

## 7. CODE vs PAPER — the most important section

### 7.1 The "LSTM over 5 frames" is NOT real temporal processing (confirmed, biggest mismatch)
Paper Fig. 1a visually depicts: `Image Sequence (5×128×128) → CNN → 5×50×256 → LSTM → 256`, implying a per-frame CNN pass producing a feature per frame, and an LSTM that aggregates temporal information across the 5 frame-features. The Methods text similarly says the visual encoder "leverages LSTM units to process the features and embed the concept of time, allowing the model to learn the temporal dynamics within the sequential pictures."

**What the code (`ResNet_RNN.py`, class `ResNet.forward`, lines ~105–129) actually does:**
1. The 5 grayscale frames are stacked as **5 input channels** of a single `Conv2d(5, 64, ...)` — i.e., **early channel-fusion**, not per-frame processing. `conv1`→...→`layer4`→`avgpool`→`fc0,fc1,fc2` runs **once** on this 5-channel image stack, producing a **single** 256-dim vector `x` (shape `[batch, 256]`). There is no per-frame feature — the 5 frames are already collapsed into one feature vector before any recurrence.
2. That single 256-dim vector is then replicated 5 times: `x = torch.cat(5*[x.unsqueeze(0)])` → `[5, batch, 256]`.
3. `l_x = torch.transpose(x, 0, 2)` swaps dims 0 and 2 → shape `[256, batch, 5]`. This means the tensor fed to the LSTM has **sequence length = 256** (the feature dimensions, treated as "time steps") and **input_size = 5** (which, because of the replication in step 2, is 5 *identical copies* of the same scalar at every position — no new information across that axis).
4. `self.rnn = nn.LSTM(input_size=5, hidden_size=256, num_layers=1)` processes this fake 256-long "sequence" of constant 5-vectors, and `x = x[-1]` takes the final LSTM hidden output as the 256-dim visual-encoder output.

**Consequence:** The LSTM here does *not* aggregate information across the 5 camera frames — that fusion already happened via 2D-conv channel-stacking in step 1, a purely feed-forward (non-recurrent) operation with no explicit temporal ordering beyond "5 channels of one conv." The LSTM instead runs over the 256 *feature dimensions* of a single already-fused vector, with a redundant/degenerate 5-wide input at each step (5 copies of the same scalar). This is exactly the same "5 copies of one frame's feature, not real temporal history" pattern noted before this task started, now precisely localized: `ResNet_RNN.py:124` (`torch.cat(5*[x.unsqueeze(0)])`) and `ResNet_RNN.py:127` (`torch.transpose(x, 0, 2)`).
- The same bug/pattern is replicated in the **VO model** (`ov_model.forward`, `ResNet_RNN.py:285-290`, using `torch.cat(7*[x.unsqueeze(0)])` and `nn.LSTM(input_size=7, hidden_size=256)`), and in `model.py`'s standalone `RCNN` (`torch.cat(7*[out.unsqueeze(0)])`, line 101-104 of `model.py`).
- Net effect: whatever "temporal dynamics" claim the paper makes for the LSTM is not literally what the LSTM computes; the real temporal/motion information the network can exploit comes entirely from the 5-channel early fusion in the first conv layer (optical-flow-like channel differencing learned implicitly by the 7×7 stride-2 conv), not from any recurrent processing across frames.

### 7.2 Table 4 (sequence-length ablation) is not reproducible from the shown code path
The paper's Table 4 (loss vs. sequence length 1–6 frames) implies a configurable frame count that was swept. The `main.py`/`ResNet_RNN.py` code path hard-codes 5 frames (dataset loader `ImgData2` always loads `range(1,6)` frames; `ResNet.forward` hard-codes `5*[...]` and `LSTM(input_size=5,...)`). No flag/config to vary sequence length was found in the reviewed files — the ablation was presumably run with a separately modified copy of the model/dataset code not included in this snapshot of the repo.

### 7.3 Default trained backbone is ResNet-50, not ResNet-18 shown in the standalone demo
`ResNet_RNN.py`'s `if __name__ == "__main__":` block (a speed/summary test script) instantiates `ResNet18(... img_channel=5, input_pre_a=True)`. The actual training entry point used for real results, `main.py` `RUN_PROGRAM == 2` (line ~1089), instantiates **`ResNet50`** with the same `img_channel=5, input_pre_a=True, normalization=True`. Readers grabbing the `__main__` block as "the model" would get the wrong backbone depth.

### 7.4 Batch size discrepancy vs. Table 6
Table 6 lists batch size **128** for "training visual self-model." In the actual default VSM-training call path (`main.py`, `RUN_PROGRAM==2`), `batch_size = 16` is what's passed to `start_train_model`/`train_in_sim`. Batch size 128 is instead what's used for training the **separate VO model** (`RUN_PROGRAM==3`). Likely the paper's table describes a different/tuned run than the batch size hard-coded in this snapshot's default script, or the two batch-size settings for VSM vs VO got merged into one table row.

### 7.5 LR-decay patience vs. paper prose
Paper prose ("Training egocentric visual self-model" section): "scaled by 0.1 if the validation loss does not decrease over twenty epochs." Code: `ReduceLROnPlateau(optimizer, factor=0.1, patience=5)` — decay patience is **5**, not 20. The number "20" in the paper text actually matches a *different* code-level constant, the **early-stopping** counter `abort_learning`: `if abort_learning > 20: break` (`main.py` ~line 595), which stops training entirely after 20 non-improving epochs. The paper appears to have conflated the LR-decay patience with the early-stopping patience — they are two different hyperparameters in the code (5 vs 20).

### 7.6 Visual odometry (VO) model is architecturally identical to the self-model, minus the action encoder
Paper describes VO model as: "same architecture as our egocentric visual self-model, consisting of the visual encoder and MLPs" but explicitly *not* predictive (no action input). Code confirms: `ov_model` in `ResNet_RNN.py` (lines 224-337) reuses the exact same ResNet-bottleneck + replicate-then-LSTM visual pipeline (just 7 channels instead of 5, and `fc3`(256→128, note: **not** 512→128 since there's no action-vector concatenation) → `fc4`(128→32) → `fc5`(32→6)), with no `fc_a*` action-encoder branch at all — this matches the paper's description faithfully.

### 7.7 Freezing behavior during cross-robot transfer matches the paper claim
Paper states: for generalizing the pretrained visual encoder to new robot configs (robots 1–3), "the model did not learn any new visual features... it relied on the visual features learned from the initial robot" (visual encoder weights kept frozen). Code confirms in `main.py` (`train_in_sim`, lines ~518-528): when `freeze_weights == True`, gradients are disabled for `model.conv1`, `model.layer1`, `model.layer2`, `model.layer3`, `model.layer4` — i.e., the entire convolutional backbone (but *not* `fc0/fc1/fc2`, the action encoder, or the fusion head, which presumably remain trainable) — matches paper's description of fine-tuning with the visual feature extractor frozen.

### 7.8 Grayscale-only justified as latency optimization (matches paper)
Paper explicitly justifies using grayscale over RGB as a real-time-control latency/compute optimization ("smaller network to process the reduced visual inputs... save the cost of the decision-making process during inference time"). Code (`camera_pybullet.py`) confirms: `cv2.cvtColor(img[2][:,:,:3], cv2.COLOR_BGR2GRAY)` is applied in both `robo_camera()` and `atlas_camera()` — grayscale conversion happens at the simulated-camera level itself, before any storage/training, matching the paper's stated rationale.

---

## 8. Evaluation methodology & headline results

### 8.1 Real-world experiments (Fig. 2, 3; four tasks: forward, backward, turn left, turn right)
- Baselines compared: **Sinusoidal Gait** (fixed, no adaptation), **Gait Generator** (CPG, dynamic but not learned-vision-based), **Ours (Only input IMU)** (IMU-only ablation, uses `IMU_bl` model), **Ours (Rug)**, **Ours (Checkerboard)** — trained on rug texture only in sim, tested on unseen real checkerboard texture too.
- Each experiment: robot starts from the same pose, episode = 56 steps, repeated N=10 trials.
- Result (Fig. 3, qualitative from bar chart): EVSM (rug/checkerboard variants) scores highest across all 4 tasks; move-backward is the task where the Gait Generator baseline actually goes strongly *negative* (~−25) while EVSM stays modestly positive (~5-8), and EVSM (rug & checkerboard) are nearly identical, showing robustness to the unseen checkerboard texture.
- Statistical test: one-way ANOVA + Tukey HSD post-hoc per task; **all p-values < 0.05** for "Ours" vs. either baseline ("Action Only" and "IMU Only"); no significant difference among the four "Ours" variants themselves (rug/grid/color-dots/grass all comparable, i.e., texture-robust given domain randomization).

### 8.2 Simulation quantitative eval — Table 1 (Mean/Std of prediction loss, per task, per terrain)
| Method | Move fwd (mean) | Move bwd | Turn right | Turn left |
|---|---|---|---|---|
| Action Only (no vision) | 3.10E-03 | 1.90E-03 | 2.80E-03 | 2.30E-03 |
| IMU Only | 2.40E-03 | 1.80E-03 | 2.10E-03 | 2.20E-03 |
| Ours (rug) | 1.90E-03 | 1.40E-03 | 1.60E-03 | 1.50E-03 |
| Ours (grid) | 1.90E-03 | 1.40E-03 | 1.60E-03 | 1.50E-03 |
| Ours (Color Dots) | 1.40E-03 | 1.30E-03 | 1.60E-03 | 1.60E-03 |
| Ours (grass) | 1.40E-03 | 1.30E-03 | 1.60E-03 | 1.30E-03 |
"Action Only" = commands-only / open-loop baseline (no vision at all, isolates the contribution of visual input). Ours consistently lowest prediction loss across all terrains/tasks.

### 8.3 Ablation: contribution of the visual encoder
Three-way baseline comparison to isolate contribution of vision:
1. **Commands only — no vision** ("Action Only"): open-loop, tests whether vision matters at all vs. blind proprioception-free control.
2. **IMU-only**: same action-encoder branch, IMU-derived pose input instead of image; isolates high-dim vision vs. low-dim proprioception (uses the same 24-dim action encoder as EVSM, per `IMU_bl` code).
3. **Full EVSM**: vision + actions.
Finding: vision > IMU > commands-only, and this gap is statistically significant (ANOVA/Tukey, §8.1).

### 8.4 Sequence-length ablation — Table 4
| Sequence length (frames) | Mean loss | Std |
|---|---|---|
| 1 | 0.042308 | 0.001156 |
| 2 | 0.028885 | 0.000137 |
| 3 | 0.027686 | 0.000281 |
| 4 | 0.027272 | 0.000508 |
| 5 | 0.027072 | 0.000764 |
| 6 | 0.027008 | 0.000166 |
Trained/tested on rug-textured data, 50,000 steps per configuration, 3 repeats each. Finding: loss drops sharply 1→2→3 frames, then flattens (diminishing returns) from 4 frames onward — motivates the paper's choice of **5 frames** as a practical trade-off (captures optical-flow-like local motion cues, bounded compute for real-time inference). Given §7.1/7.2, this table's temporal-benefit interpretation should be read cautiously, since (a) the deployed 5-frame model doesn't process frames temporally via the LSTM (early channel-fusion instead) and (b) the ablation's implementation wasn't found in the reviewed code snapshot.

### 8.5 Generalization to new robot morphologies — Table 1(new)/Fig. 5
- 3 new robot configs (Robot 1–3) created by altering leg-body connection angles/orientations from Robot 0 (Robot 3: 2nd/3rd leg joints each rotated 90°, "significant kinematic changes").
- Pretrained visual encoder from Robot 0 reused (frozen), fine-tuned with **less data (200k steps)** vs Robot 0's original 400k steps.
- Two baselines: **NV** (non-visual, motor commands + proprioception only) and **IM** (initial model of Robot 0, applied zero-shot with no fine-tuning to new robots).
- Result (Fig. 5 bar charts across 4 ground textures × 4 tasks): **OM (ours, fine-tuned)** consistently lowest prediction error across all robots/terrains/tasks, "highlighting the substantial reduction in prediction errors achieved by fine-tuning... with a small batch of data."

### 8.6 Applicability to a humanoid robot (Atlas, simulated) — Table 2, 3, Fig. 6
- Atlas: upper-body joints frozen, only 6 DoF used (hip, knee, ankle joints) to focus on basic locomotion.
- Table 2 (state prediction error, mean±std): Our method vs. No-Images baseline — **forward**: 5.99E-04 vs 3.38E-01; **right**: 7.05E-04 vs 4.61E-01; **left**: 5.77E-04 vs 4.71E-01. Our method is ~2-3 orders of magnitude lower error.
- Table 3 (locomotion performance / net displacement score, mean±std): Our method vs No-Images-Input Baseline vs Gait Generator — **forward**: 1.27E+00 vs 4.06E-01 vs 4.50E-01; **right**: 2.01E+00 vs −6.96E-01 vs 3.46E-01; **left**: 1.67E+00 vs 1.53E-01(std 1.89E-01, baseline text shows 1.53E-01 mean) vs 4.10E-01. Our method achieves the highest displacement in the intended direction for all 3 tasks; No-Images baseline can even go *negative* on "right" (−6.96E-01), indicating it turns the wrong way without vision.

### 8.7 Anomaly detection / damage recovery — Fig. 7, Table (Detect Error), Discussion
- Damage simulated by cutting a leg end-link in half ("broken leg module").
- **Anomaly detection mechanism:** a separate **Visual Odometry (VO) model** (7 consecutive frames in, same backbone as EVSM minus action encoder, outputs `(Δx,Δy,Δz,Δroll,Δpitch,Δyaw)` state-change, non-predictive/non-action-conditioned) provides an independent visual-only ground-truth-ish estimate of actual motion. If the EVSM's *predicted* motion (from actions) diverges from the VO's *measured* actual motion beyond some threshold, this discrepancy triggers an anomaly alarm.
- Fig. 7D bar chart "Mean Square Error" of detection: Normal robot ≈0.03, Damaged robot ≈0.045 (highest), Damaged-robot-after-recovery ≈0.02 (lowest, even below normal — after retraining the self-model matches the new damaged body well).
- **Recovery procedure:** ~30 min motor babbling on the damaged robot → 7000 steps collected → re-train/update EVSM with this new data → once EVSM's predictions match VO-measured motion again, robot regains ability to walk forward despite the still-broken leg (adapts control policy to the new, damaged body dynamics rather than fixing the hardware).
- This is demonstrated qualitatively via video (Supplementary Movies 3-5) and Fig. 7E image sequences (Normal / Damaged / Damaged-after-recovery, each moving forward over 10s), not a large quantitative benchmark table.

---

## 9. Known limitations (stated explicitly by the paper, Discussion section)

1. **Static-ground assumption:** predictive accuracy depends on the ground being static; dynamic environments (moving surfaces, significant terrain deformation) introduce visual-cue inconsistencies that would reduce model reliability. This is called "the primary limitation."
2. **Short temporal window / no long-horizon prediction:** the model relies on short sequences of visual data (5 frames, ≈83ms), providing limited temporal information — it cannot predict long-term future states.
3. **Extreme visual conditions:** heavy occlusions and rapid lighting changes can disrupt extraction of motion cues from vision, degrading performance (not empirically tested, stated as a concern for future work).
4. **No confidence estimation:** the paper suggests future work should add a confidence-estimation mechanism to the self-model (uncertainty quantification is absent in the current design).
5. Paper also notes real-world reliance is only on **ground-plane** visual reference (front camera pointed down at ground, not forward), reasoning that this mimics how humans/animals use ground-texture optical flow, but explicitly flags that combining ground-facing + forward-facing cameras for both short- and long-term planning is future work (not done here).

---

## 10. Practical run-modes of the released code (`main.py`, README)

`main.py`'s `RUN_PROGRAM` flag selects mode (all in one file, no CLI args — edit the constant and rerun):
1. **Collect Data** in PyBullet simulation (`OpticalEnv`, ground textures cycled through `["grass_rand","rug_rand","color_dots","grid"]`).
2. **Train the Egocentric Visual Self-Model** (`start_train_model` → `train_in_sim`, ResNet50 backbone, optionally loads a pretrained model and freezes the CNN backbone for transfer to new robots).
3. **Train the Visual Odometry (VO) Model** (`start_train_ov_model` → `train_ov`, `OV_Net`, batch_size=128, 7-channel input, no action encoder).
4. **Test the Egocentric Visual Self-Model** (`test_model`, runs 4 ground textures × 4 tasks, GUI-connected PyBullet).
5. **Recovery Test** (anomaly detection + retraining after simulated damage).
6. **Use VO to collect data** (`use_vo_collect_data`, uses the trained VO model's predictions as pseudo-labels/state estimate during data collection, supports `frozen_joint` list to simulate locked/broken joints).

Related standalone scripts in the repo (not covered in depth above, noted for completeness):
- `env.py` / `env_agent.py`: PyBullet Gym environments for the quadruped (`env.py`) and Atlas humanoid (`env_agent.py`); both implement `World2Local` (converts world-frame position delta into robot's local/yaw-relative frame) — this is how the Δx,Δy,Δz,Δroll,Δpitch,Δyaw labels are computed relative to the robot, not the world.
- `traj_optim.py` / `traj_optim_agent.py`: trajectory/gait optimization (likely the Hill Climbing CPG optimizer described in the paper's gait-generator section).
- `data_enginering.py`: computes the per-dimension normalization scale/offset (`norm_dataset_*.csv` files) used to standardize the 6 output dims before training (paper: "normalized the ground-truth value to the same standard deviation").
- `resnet_test_speed.py`: standalone inference-speed benchmarking script for the ResNet variants (not part of the training/eval pipeline).

---

## 11. Quick-reference formula sheet

- Loss: `L = MSE(f_p(I_t, A_t), S_{t+1})`
- CPG motor trajectory: `θ_i(t) = a_i·sin(t/T·2π + b_i) + c_i`
- CPG optimization reward: `R(Δx,Δy) = Σ_{t=1}^{n} (100·Δy − 50·|Δx|)`
- Training-data action noise: `a_n(i,t) = Gaussian(μ=a_o(i,t), σ=0.2)`
- Sim reward functions: forward `2Δy − |Δx|`; backward `−2Δy − |Δx|`; turn-left `Δyaw`; turn-right `−Δyaw`
- Real-world score functions (a=b=10): forward `a·y − b·|yaw|`; left `a + b·yaw`; right `a − b·yaw`; backward `−a·y − b·|yaw|`
- Image size: 128×128 grayscale × 5 frames; action: 24-dim (12 prev + 12 next joint angles, normalized ±90°→[−1,1])
- Visual encoder output: 256-dim; action encoder output: 256-dim; fused: 512-dim; final output: 6-dim delta pose
