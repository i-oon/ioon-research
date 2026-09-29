# Architecture Comparison: Egocentric VSM, LAC-WM and Ours

Sources: the Egocentric VSM paper and its released code; the LAC-WM paper; our code and run configurations.

## Side by side

Status legend: = same as LAC-WM · ≠ differs.

| component | Egocentric VSM (Hu et al. 2025) | LAC-WM (ICLR 2026 submission) | Ours (current) | vs LAC-WM |
|---|---|---|---|---|
| Visual encoder | ResNet-50, trained end to end | V-JEPA2 1B, frozen | V-JEPA2, frozen | = |
| Latent action | none; the action is an input | z = IDM(x_t, x_t+k), 64-d | z = ITM(e_t, e_t+k), 64-d | = |
| Forward model | MLP fusion (image 256 + action 256 → 6) | FDM, 8 attention blocks, 512 / 16 heads | FTM, 8 attention blocks, 512 / 16 heads | = |
| Temporal attention in the forward model | none | last 20k of 80k iterations, horizon 8 | none (single step) | ≠ |
| Action chunk per step | 1 step | 5 steps | 5 steps | = |
| Cross-augmentation | n/a | yes | yes | = |
| Motion target that shapes z | Δ body pose (6-d), predicted directly | end-effector pose + camera motion; z split in half, one half per target | **Froude body motion (3-d), one head shared by all bodies, reads z only** | ≠ (a task-space target, as in LAC-WM) |
| Per-body action level | action is the model input | action projector at adaptation | action projector (joint commands → z) | = |
| Extra losses | none | none | hinge (real vs null z), frozen read-out, 2-step rollout | ≠ |
| Pretraining bodies | 1 robot | 3 (human hands, bimanual humanoid, Franka arm) | 1 hexapod; hexapod + B1 jointly under test | ≠ |
| Pretraining data | motor babbling | 150k trajectories | 48 clips of 66 frames per body | ≠ |
| Batch × iterations | small model | 512 × 80k ≈ 41M samples | 8 × ≈16.7k ≈ 134k samples | ≠ |
| Adaptation | retrain after damage | 3 stages: LoRA r2 IDM+FDM → projector → joint LoRA r2 | same 3 stages, plus a Froude-head fit | = (plus one stage) |
| Goal given to the planner | body-pose target | subgoal image of the same body | Froude sequence of another body | ≠ |
| Candidate scoring | predicted Δ pose vs target | L2 in embedding space, predicted frame vs subgoal image | Froude read from z (direct) or from the predicted frame (rollout) | ≠ |
| Re-grounding | every step | every subgoal (6- or 20-step rollout) | every 2 steps, read-out window 11 | similar |

## Differences with a measured consequence

| difference | measurement (metric named in each cell) |
|---|---|
| Action chunk 1 → 5, as in LAC-WM | Normalised score on the B1 (0 = random, 1 = oracle): direct +0.66 → +0.82, rollout (current start) −0.25 → +0.40 |
| Motion target: Froude head allowed to shape z, vs reading z only | Normalised score, c08: direct +0.55–0.61 → +0.79–0.80. Physics closed loop, c08 direct, mean L2 Froude error (random 0.126): 0.072–0.078 → 0.050–0.053 |
| Motion target: Froude only, vs Froude + per-body joint commands | Normalised score, c08 rollout (library start): +0.47–0.54 → +0.56–0.63; B1 direct +0.82 → +0.85; other cells equal |
| Rollout scores by reading Froude back out of a predicted frame (LAC-WM compares embeddings) | Pearson r across 24 actions, Froude read from the **real** future frame: 0.24 / 0.52 / 0.24 (fwd / lat / yaw), vs 0.83 / 0.91 / 0.84 for direct |
| One behaviour per clip (LAC-WM: action varies within a trajectory) | Share of prediction variance explained by the start frame 0.26–0.66, by the action 0.15–0.23. Transitions matching another behaviour's pose: 4 / 765 |
| Random actions held 15–25 frames (babble), vs one behaviour per clip | Pearson r of Froude read from the rollout's prediction, yaw: 0.63–0.68 vs 0.04–0.21. Normalised score not higher |
| Single-body vs joint hexapod + B1 pretraining | Cross-body retrieval c10 → B1 (0 = random, 1 = identical motion): 0.04–0.05 single, 0.21 joint (stride 1); between hexapods 0.32–0.44 |
| LoRA rank 2 vs full fine-tune in adaptation | Forgetting on the hexapod, relative MSE vs a mean-latent baseline (lower is better): 0.918 → 0.786 |

Not yet measured in our system: temporal attention in the forward model; batch and iteration scale; embodiment conditioning; image-goal scoring in embedding space.
