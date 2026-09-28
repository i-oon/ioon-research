# Architecture Comparison: Egocentric VSM, LAC-WM and Ours

Updated 2026-09-28. Sources:
- `doc/ref/notes_egocentric_vsm.md`, checked against the paper and its released code.
- `doc/ref/notes_lac_wm.md`, checked against the paper PDF.
- Our code (`wm/`) and run configs (`wm/runs/*/config.yaml`).

## Side-by-side

Status legend: = same as LAC-WM · ≠ differs.

| component | Egocentric VSM (Hu et al. 2025) | LAC-WM (ICLR 2026 submission) | Ours (current: stride 5) | status vs LAC-WM |
|---|---|---|---|---|
| Visual encoder | ResNet-50, trained end to end | V-JEPA2 1B, frozen | V-JEPA2, frozen | = |
| Latent action | none; the action is an input | z = IDM(x_t, x_t+k), 64-d | z = ITM(e_t, e_t+k), 64-d | = |
| Forward model | MLP fusion (image 256 + action 256 → 6) | FDM, 8 attention blocks, 512 / 16 heads | FTM, 8 attention blocks, 512 / 16 heads | = |
| Temporal attention in forward model | none (single fused frame) | added for the last 20k of 80k iterations, horizon 8 | none; single step FTM(e_t, z) | ≠ |
| Action chunk / step | 1 step (prev + next joint angles) | 5-step chunks | 5-step chunks (stride 5) | = |
| Cross-augmentation | n/a | yes | yes | = |
| Motion decoder target | n/a | end-effector pose + camera motion, z split in half, one per target | joint commands, one head per body (18-d hexapod, 12-d B1), whole z | ≠ |
| Body-level motion loss | predicts Δ body pose (6-d) directly | via end-effector / camera pose above | Froude head shared across bodies; reads `z.detach()` (no gradient to z) since 2026-09-08 | ≠ |
| Extra losses | none | none | hinge (real vs null z), frozen read-out, 2-step rollout | ≠ |
| Pretraining bodies | 1 robot | 3 (human hands, bimanual humanoid, Franka arm) | 1 (hexapod c10f10t10) | ≠ |
| Pretraining data | motor babbling, sim + real | 150k trajectories (50k per dataset) | 48 clips of 66 frames, ≈2.7k transitions | ≠ |
| Action variation within a trajectory | random (motor babbling) | teleoperation / human manipulation | one behaviour per clip (beh24) | ≠ |
| Batch × iterations | not critical (small model) | 512 × 80k ≈ 41M samples | 8 × ≈16.7k ≈ 134k samples | ≠ |
| Embodiment conditioning | n/a | not described | built (`ftm_embodiment_channel`), never enabled | ≠ |
| Adaptation | retrain after damage | 3 stages: LoRA r2 IDM+FDM → projector → joint LoRA r2 | same 3 stages, plus Stage 4 (body head fit) | = (plus one stage) |
| Adaptation data | n/a | one dataset for every stage (7,265 trajectories) | Stage 1: 3 clips; Stages 2 and 4: ≈38 clips | ≠ |
| Goal given to the planner | body-pose target | **subgoal image** from a demonstration of the same body | **Froude** sequence from another body | ≠ |
| Candidate scoring | predicted Δ pose vs target | **L2 in embedding space** between predicted frame and subgoal image | Froude read by ITM + body head from predicted frame (rollout), or body(proj(a)) (direct) | ≠ |
| Re-grounding | every step (real frame) | every subgoal: 6-frame (Table 5) or 20-step (Table 2) rollout | every 2 steps, w = 11 | similar |

## Differences with a measured consequence in our system

Metrics are as defined in `report/weekly_update.md`: M2 normalised score, M3 Pearson r, M5 η², M10 cross-body R².

| difference | measurement | source |
|---|---|---|
| Scoring reads Froude back out of the predicted frame (LAC-WM compares embeddings, no read-out) | Reading the **real** counterfactual outcome through ITM + body head: M3 r 0.24 / 0.52 / 0.24. Direct: 0.83 / 0.91 / 0.84 | F266 |
| One behaviour per clip | M5 on the pretrained model, hexapod val: η² state 0.26–0.66, η² action 0.15–0.23. Pose-matched cross-behaviour pairs: 4 / 765 transitions | F252, FINDINGS_old note before F154 |
| Action variation added (switch babble only, 1 seed, in progress) | η² state 0.37 / 0.32 / 0.22, η² action 0.31 / 0.49 / 0.47 (A2: 0.63 / 0.41 / 0.43 and 0.23 / 0.33 / 0.20) | babble arms, 2026-09-28 |
| Action chunk 1 → 5 (now matches LAC-WM) | B1 NS (w=11): direct +0.66 → +0.82, rollout (current start) −0.25 → +0.40. Stride 5 over 4 pretrains: rollout +0.21 to +0.40 | F264, F273 |
| Motion decoder on per-body joint commands; Froude head detached | z not shared with the B1: M10 c10 → B1 −0.91 / −0.29 / −0.18 (stride 5), body-ID probe 0.67 (chance 0.33) | F272 |
| Before 2026-09-08: Froude head not detached, hexapod + B1 pretrained together | Cross-body R² +0.544 / +0.435 (co-trained head, allocentric data); after detach, per-body heads −0.397 / −0.594 | F232; the controlled detach on/off comparison is queued (`scripts/run/detach_*`) |
| Adaptation, LoRA r2 vs full fine-tune | Forgetting on the hexapod, relative MSE vs mean-latent baseline: 0.918 → 0.786 | F248 |
| Stage 3 (joint projector + FTM) | Prediction top-1 retrieval 0.233 → 0.276; B1 NS rollout +0.40 → +0.44 (within seed spread) | F265 |

Not yet measured in our system:
- temporal attention in the forward model;
- batch size / iteration scale;
- multi-body pretraining;
- embodiment conditioning;
- image-goal scoring in embedding space.
