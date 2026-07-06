# Research Direction — Cross-Morphology Locomotion via Latent Action World Models

---

## Core Claim
Learn a **morphology-agnostic latent action z_t** from simulation video
that clusters by **behavior** (walk/turn/stop), not by body shape.
Prove it transfers to an unseen morphology without retraining.

---

## Pipeline (confirmed)

- **Encoder**: V-JEPA2 RGB tokenizer — frozen ✓
  - per-frame, no temporal mixing inside encoder
  - pre-encode offline → save e_t to disk (keep encoder in loop only if using cross-augmentation)

- **ITM**: attention-based ✓ (confirmed from LAC-WM paper)
  - M causal attention layers + N cross-attention layers + learned query token q_t
  - in: [e_t, e_{t+1}] → out: z_t ∈ ℝ^k
  - NOT a simple MLP

- **FTM**: attention-based ✓ (confirmed from LAC-WM paper)
  - L transformer blocks: self-attn(x_t) + self-attn(z_t) + cross-attn(x_t→z_t)
  - in: [e_t, z_t] → out: ê_{t+1} ∈ ℝ^768
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

---

## Decisions Still Needed

| Block | Status | Note |
|---|---|---|
| z_t dimension k | probably 64 (LAC-WM uses 64) | ablate 32/64/128 |
| λ_recon, λ_motion | undecided | start equal, ablate |
| Stick insect DoF | **user to research** | search existing sim models |
| Data collection policy | **user to research** | scripted / RL / random — borrow from existing work |
| Step 2 evaluation metric | undecided | L_recon drop? PCA? policy rollout? |
| Baseline exact setup | undecided | separate pipeline per morphology vs. scratch RL |
| LAC-WM source code | not found yet | rejected ICLR 2026, accepted ICML 2026 — no public code yet |

---

## Execution Plan

- **Step 0 — Sanity check (free, before training)**
  - PCA raw V-JEPA2 embeddings e_t from all 3 morphologies doing walk/turn/stop
  - Pass → encoder already separates behaviors, ITM has a clean foundation
  - Fail → sim domain gap is severe → consider partial fine-tune of last 2 V-JEPA2 blocks

- **Step 1 — Train Phase 1 pipeline**
  - Train ITM + FTM + Motion Decoder on short + long leg
  - Flat z_t, short-horizon (frame t → t+1)
  - Use cross-augmentation to prevent shortcuts

- **Step 1.5 — PCA + K-means validation on z_t**
  - Collect z_t from short + long + medium leg (all 3 behaviors)
  - Plot 2D PCA colored by {behavior} and {morphology}
  - Apply K-means (K=3) to latent sequences → check if clusters match walk/turn/stop
  - **Pass**: clusters by behavior, not morphology → proceed to Step 2
  - **Fail**: clusters by morphology → shortcut = body-shape visual encoding → fallback to HiLAM
  - Note: in LAC-WM, shortcut without MD was viewpoint clustering (egocentric vs 3rd-person).
    In our work, all views are 3rd-person so shortcut would be morphology clustering instead.

- **Step 2 — Transfer to unseen morphology**
  - Fine-tune ITM + FTM on medium leg (unseen in pretraining) using LoRA rank 2
    (LoRA preserves unified latent structure while adapting to new morphology — LAC-WM uses this)
  - Metric: TBD (L_recon / PCA / policy rollout)
  - This is **interpolation**, not extrapolation

- **Step 3 (extended) — Extrapolation**
  - Morphology outside training range
  - Out of pre-proposal scope, good future direction

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
- **Training scale**: LAC-WM used 128 GPUs × 4 days for 3 manipulation datasets. Our 3-morphology
  locomotion setting is far smaller — reasonable to train on a single node.
- **Why ICLR rejected**: weak evaluation (1 task, 1 baseline), not because the core method is wrong.
  Professors will likely raise same concern → plan for ≥2 baselines and ≥3 behaviors.
- **EAC-WM degrades with more embodiments**; LAC-WM improves. This is our key supporting evidence.

## Open Questions
- Fine-tune last 2 V-JEPA2 blocks if Step 0 shows sim gap?
- k = 64? ablate
- λ values? ablate from equal weights
- Evaluation metric for Step 2?
- Baseline: at minimum EAC-WM analog (train per-morphology with explicit joint commands) vs. LAC-WM style
