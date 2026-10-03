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
| Image randomisation | crop 10–100%, brightness ×0.1–10, blur kernel 3–41, 4 ground textures | not reported | crop 85–100%, brightness / contrast ±20%; stronger settings (up to Egocentric VSM's ranges, 25% of samples kept clean) implemented, pilot pending | ≠ |
| Rendering across bodies | one robot | real video | every body's room, floor and texture scaled with its camera height (frame-level body-ID probe on matched camera poses 0.552, chance 0.5) | n/a |
| Motion target that shapes z | Δ body pose (6-d), predicted directly | end-effector pose + camera motion; z split in half, one half per target | **Froude body motion (3-d), one head shared by all bodies, reads z only** | ≠ (a task-space target, as in LAC-WM) |
| Per-body action level | action is the model input | action projector at adaptation | action projector (joint commands → z) | = |
| Extra losses | none | none | hinge (real vs null z), frozen read-out, 2-step rollout | ≠ |
| Pretraining bodies | 1 robot | 3 (human hands, bimanual humanoid, Franka arm) | 1 hexapod, or hexapod + B1 jointly | ≠ |
| Pretraining data | motor babbling | 150k trajectories | per body 48 clips of 66 frames (24 behaviours, windows of one long walk each, one room per clip shared by both bodies) + counterfactual branches 3,456 (3 branch points x 24 commands per clip; B1 exact MuJoCo restore, hexapod deterministic replay); Froude at the centre of mass | ≠ |
| Batch × iterations | batch 16, early stop after 20 epochs without validation gain | 512 × 80k ≈ 41M samples | 8 × ≈30k ≈ 244k samples (round 1, both arms; augmentation on the fly) | ≠ |
| Adaptation | retrain after damage | 3 stages: LoRA r2 IDM+FDM → projector → joint LoRA r2 | body in pretraining: projector only. New body: LoRA r8 on ITM / FTM (3000 steps) with the pretraining losses incl. the Froude loss through the frozen head → projector. No head refit, no joint stage | ≠ (rank 8, no stage 3) |
| Goal given to the planner | body-pose target | subgoal image of the same body | Froude sequence of another body: recorded, or read from the goal clip's frames through ITM + Froude head | ≠ |
| Candidate scoring | predicted Δ pose vs target | L2 in embedding space, predicted frame vs subgoal image | Froude read from z (direct) or from the predicted frame (rollout) | ≠ |
| Re-grounding | every step | every subgoal (6- or 20-step rollout) | every 2 steps, read-out window 11 | similar |

## Differences with a measured consequence

Current models unless marked. Normalised score: 0 = random, 1 = oracle. Pearson r across candidate actions from one start state, fwd / lat / yaw.

| difference | measurement (metric named in each cell) |
|---|---|
| Action chunk 1 → 5, as in LAC-WM | Normalised score on the B1 (earlier pipeline): direct +0.66 → +0.82, rollout (current start) −0.25 → +0.40 |
| Motion target: Froude head allowed to shape z, vs reading z only | Normalised score, c08: direct +0.55–0.61 → +0.79–0.80. Physics closed loop, c08 direct, mean L2 Froude error (random 0.126): 0.072–0.078 → 0.050–0.053 |
| Motion target: Froude only, vs Froude + per-body joint commands | Normalised score, B1 (current adaptation, S0 / S1): direct +0.83 / +0.74 vs +0.80 / +0.85; rollout (recorded frame) +0.39 / +0.29 vs +0.54 / +0.34; rollout (current frame) +0.39 / +0.29 vs +0.46 / +0.38. c10 / c08 equal |
| Rollout scores by reading Froude back out of a future frame (LAC-WM compares embeddings) | Pearson r, read of the **real** future of another action from the same start state: c10 0.18 / 0.40 / 0.28, c08 0.40 / 0.42 / 0.38, B1 0.15 / 0.46 / 0.41; own recorded transition c10 0.95 / 0.74 / 0.74; direct 0.88–0.99 |
| One command per clip (LAC-WM: action varies within a trajectory) | The Froude head does not read same-start alternatives (row above), on the pretrained body too. A linear probe fitted on alternatives reads the same z′ at 0.58 / 0.60 / 0.56 (earlier checkpoint), but replacing the head with it did not improve rollout selection (+0.40 → +0.33) |
| Random actions held 15–25 frames (babble), vs one behaviour per clip | Pearson r of Froude read from the rollout's prediction, yaw: 0.63–0.68 vs 0.04–0.21. Normalised score not higher |
| Single-body vs joint hexapod + B1 pretraining (matched rendering) | B1 selection, normalised score: direct +0.74–0.83 → +0.92–0.94; rollout (current frame) +0.29–0.39 → +0.55–0.57. Cross-body retrieval c10 → B1 (0 = random, 1 = identical motion): 0.24–0.25 → 0.22–0.24 (not shared); between hexapods 0.38–0.44 |
| Froude-similarity loss in joint pretraining | No measurable change: retrieval c10 → B1 0.24 vs 0.22, B1 direct +0.94 vs +0.92 |
| LoRA rank 2 vs full fine-tune in adaptation | Forgetting on the hexapod, relative MSE vs a mean-latent baseline (lower is better): 0.918 → 0.786 |
| Adaptation: current (LoRA r8, Froude loss, no refit) vs earlier (LoRA r2, no Froude loss, head refit) | B1 selection, normalised score, S0 (before the label fix): +0.73 / +0.29 / +0.14 → +0.81 / +0.49 / +0.46 (direct / rollout recorded / current frame); re-measuring on corrected data |
| Goal read from vision instead of the recorded Froude | Normalised score, direct: B1 −0.08 (joint) to −0.28 (hexapod-only), c10 −0.03 to −0.09 |

B1 numbers are on the corrected B1 data (FINDINGS F293 / F294) unless marked; the joint models were pretrained before the correction (projectors refit on corrected data).

Not yet measured in our system: temporal attention in the forward model; batch and iteration scale; embodiment conditioning; image-goal scoring in embedding space; training on counterfactual transitions.
