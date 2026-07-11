# Research Direction — Cross-Morphology Locomotion via Latent Action World Models

---

## Core Claim
Learn a **morphology-agnostic latent action z_t** from simulation video
that clusters by **behavior** (walk/turn/stop), not by body shape.
Prove it transfers to an unseen morphology without retraining.

---

## Pipeline (confirmed)

- **Encoder**: V-JEPA2 RGB tokenizer — frozen ✓ (confirmed: ViT-g/16, 1B params)
  - per-frame, no temporal mixing inside encoder
  - output: 16×16 = 256 patch tokens per frame, each ∈ ℝ^1408
  - pre-encode offline → save e_t to disk (keep encoder in loop only if using cross-augmentation)

  **V-JEPA2 Encoder — Input / Flow / Output**

  ```
  INPUT
    frame_t  ∈ ℝ^{256×256×3}     raw RGB from sim camera

  PREPROCESSING (before encoder, your code)
    aug1 = transform(frame_t)     random crop + color jitter + flip  → ℝ^{256×256×3}
    aug2 = transform(frame_t)     different random params             → ℝ^{256×256×3}

  PATCH SPLITTING
    256×256 ÷ 16 = 16×16 grid → 256 patches of 16×16×3 pixels each

  LINEAR PROJECTION
    each patch → Linear → ℝ^{1408}
    + positional embedding (3D-RoPE)

  VIT TRANSFORMER (frozen, 1B params)
    256 tokens attend to each other (self-attention across patches)
    each token becomes spatially context-aware

  OUTPUT
    e_t¹  ∈ ℝ^{256×1408}   from aug1  → to ITM + Motion Decoder
    e_t²  ∈ ℝ^{256×1408}   from aug2  → to FTM
    e_{t+1}¹ ∈ ℝ^{256×1408} from aug1 → to ITM + L_recon target
  ```

  **Load code**
  ```python
  from transformers import AutoModel
  encoder = AutoModel.from_pretrained("facebook/vjepa2-vitg-fpc64-256")
  encoder.eval()
  for p in encoder.parameters():
      p.requires_grad = False   # frozen — no gradient flows through
  ```

  **Why frozen**: V-JEPA2 pretrained on 1M hours of internet video (VM22M, 22M videos).
  Already learned motion-relevant features. Fine-tuning would require massive compute and risk losing generality.
  Step 0 verifies these features are useful before committing to training.

- **ITM**: attention-based ✓ (confirmed from LAC-WM Table 4)
  - 4 attention blocks, 16 heads, learned query token q_t
  - in: [e_t, e_{t+1}] (512 tokens total) → out: z_t ∈ ℝ^512
  - NOT a simple MLP

- **FTM**: attention-based ✓ (confirmed from LAC-WM Table 4)
  - 8 blocks: self-attn(e_t) + self-attn(z_t) + cross-attn(e_t→z_t), 16 heads
  - in: [e_t, z_t] → out: ê_{t+1} ∈ ℝ^1408
  - NOT a simple MLP

- **Motion Decoder**: cross-attn(z_t, x_t) + MLP ✓ (confirmed from LAC-WM)
  - z_t = query, x_t = visual context (keys/values)
  - out: â_t ; L_motion = ||â_t − a_t||²
  - discarded after pretraining

- **Cross-augmentation**: YES ✓ (following LAC-WM)
  - two independent augmentations per frame pair before encoding
  - prevents ITM from taking shortcuts via texture/color instead of motion
  - encoder must stay in training loop (not pre-encoded offline)

- **Total loss**: L = λ_recon · L_recon + λ_motion · L_motion

- **Simulator**: IsaacSim 5.0 ✓

- **Data**: stick insect, 3 morphologies (short / medium / long), third-person RGB + auto-logged joint commands
  - **Action type**: joint position targets ℝ^18 (6 legs × 3 joints)
  - **Episode length**: 1000 steps (~16s at 60Hz physics)
  - **Episodes**: ~100 per morphology per behavior (walk/turn/stop) → ~200,000 training pairs for short+long
  - **Camera**: fixed, side view, ~30° elevated, sees all 6 legs clearly
  - **Data collection policy**: TBD
  - **Behaviors**: walk / turn / stop
  - **Train morphologies**: short + long leg
  - **Held-out morphology**: medium leg (Step 2 transfer test)

---

## Decisions Still Needed

| Block | Status | Note |
|---|---|---|
| z_t dimension k | **512 (confirmed LAC-WM Table 4)** | ablate smaller k for locomotion (simpler domain) |
| λ_recon, λ_motion | undecided | start equal, ablate |
| Stick insect DoF | **use lab model** | borrow from Ajan YuChen or P'Nai — already exists in lab |
| Data collection policy | **TBD** | ask lab — may already have scripted controller for existing model |
| Step 2 evaluation metric | **training time reduction** (confirmed Ajan Go) | pretrained FTM reaches same L_recon with fewer medium leg episodes than scratch |
| Baseline exact setup | undecided | separate pipeline per morphology vs. scratch RL |
| LAC-WM source code | not found yet | rejected ICLR 2026, accepted ICML 2026 — no public code yet |

---

## Execution Plan

---

### Step -1 — Morphology Gap Check
**Goal**: confirm short leg and long leg actually behave differently under same command

| Task | Send identical joint command to short leg and long leg |
|---|---|
| Expected | long leg drags / overshoots, short leg walks normally |
| Pass | visually distinct behavior → morphology gap is real → proceed |
| Fail | identical behavior → leg length difference too small → adjust morphology parameters |

> P'Hap: if same command produces same behavior, the whole experiment is pointless

---

### Step 0 — Visual Encoder Sanity Check
**Goal**: confirm V-JEPA2 frozen features are meaningful for your sim domain before any training

| Task | Run V-JEPA2 on frames from all 3 morphologies × 3 behaviors → collect e_t |
|---|---|
| Check 1 | UMAP of e_t colored by behavior → should show loose separation |
| Check 2 | UMAP of e_t colored by morphology → should NOT dominate |
| Check 3 | Attention map → encoder attends to robot legs, not background |
| Pass | behavior signal visible in raw e_t → ITM has clean foundation → proceed to Step 1 |
| Fail | no structure / morphology dominates → sim domain gap severe → partial fine-tune last 2 V-JEPA2 blocks |

> Ajan Go: test Visual Encoder first, do not proceed to Step 1 until this passes

---

### Step 1 — Train Phase 1 Pipeline
**Goal**: train ITM + FTM + Motion Decoder on short + long leg

| Task | Train on short + long leg data, cross-augmentation on, LoRA off |
|---|---|
| Monitor | L_recon and L_motion both decreasing over training |
| Sanity check mid-training | sample z_t every 10k steps → UMAP should show emerging structure |
| Pass criterion | L_recon converges, L_motion < threshold → proceed to Step 1.5 |
| Fail | loss not converging → check λ weighting, learning rate, data pipeline |

---

### Step 1.5 — Latent Space Validation
**Goal**: prove z_t is morphology-agnostic before testing transfer

| Task | Collect z_t from short + long + medium leg × 3 behaviors |
|---|---|
| Check 1 | UMAP colored by behavior → 3 clusters (walk / turn / stop) visible |
| Check 2 | UMAP colored by morphology → no separation between short / long / medium |
| Check 3 | K-means (K=3) → cluster labels match behavior labels (quantitative) |
| Pass | clusters by behavior, not morphology → proceed to Step 2 |
| Fail | clusters by morphology → fallback to HiLAM (see below) |

> Note: failure mode = z_t encoding body-shape visual features instead of motion.
> In LAC-WM this was viewpoint clustering. In our work it would be morphology clustering.

---

### Step 2 — Transfer to Unseen Morphology
**Goal**: prove pretrained World Model reduces data needed for medium leg

| Task | Fine-tune ITM + FTM on N medium leg episodes using LoRA rank 2 |
|---|---|
| Condition A | pretrained FTM + N episodes |
| Condition B | scratch FTM + N episodes (baseline) |
| Vary N | 5 / 10 / 20 / 50 / 100 episodes |
| Metric | L_recon on held-out medium leg test set |
| Pass | pretrained reaches same L_recon as scratch with significantly fewer episodes |
| Fail | no gap between pretrained and scratch → z_t did not transfer → revisit Step 1.5 |

> Ajan Go: main claim = "World Model ลด training time อย่างชัดเจน"
> This is interpolation (medium leg is between short and long) — not extrapolation

---

### Step 3 (Future) — Extrapolation
**Goal**: test morphology outside training range

> Out of pre-proposal scope. Good future direction if Steps 1-2 succeed.

---

## If Step 1.5 Fails — Fallback (HiLAM)
- Phase A (same): train ITM+FTM → low-level z_t, freeze ITM
- Phase B (new): z_t sequences → dynamic chunking → z^h (skill-level)
- z^h should cluster walk/turn/stop more cleanly than flat z_t
- V-JEPA2 encoding unchanged — hierarchy is a second pass on z_t

---

## Baseline
- **What we beat**: training from scratch per morphology (no transfer)
- **Metric**: sample efficiency on medium leg — with vs. without pretrained FTM
- No existing locomotion cross-morphology baseline → comparison is transfer vs. no-transfer
- LAC-WM baseline (EAC-WM) is manipulation-only, cannot port directly

---

## Notes from ICLR Reviews
- **V-JEPA2 pixel decoder**: not included in V-JEPA2 — must be trained separately if needed for pixel-space output.
  L_recon = ||ê_{t+1} − e_{t+1}||² is computed in *embedding* space → no pixel decoder required for training.
- **Training scale**: LAC-WM used 64 H200 GPUs × 4 days for 3 manipulation datasets (confirmed App. A.5).
  Our 3-morphology locomotion setting is far smaller — reasonable to train on a single node.
- **Why ICLR rejected**: weak evaluation (1 task, 1 baseline), not because the core method is wrong.
  Professors will likely raise same concern → plan for ≥2 baselines and ≥3 behaviors.
- **EAC-WM degrades with more embodiments**; LAC-WM improves. This is our key supporting evidence.

## Lab Resources
- Stick insect model: ask Ajan YuChen or P'Nai for existing IsaacSim model
- Data collection policy: ask lab if scripted controller already exists for the model
- P'Beam's work may connect to this in future

## Open Questions
- Fine-tune last 2 V-JEPA2 blocks if Step 0 shows sim gap?
- k = 512 (LAC-WM) or ablate smaller for locomotion?
- λ values? ablate from equal weights
- Baseline: EAC-WM analog (train FTM per-morphology with explicit joint commands, no ITM)
