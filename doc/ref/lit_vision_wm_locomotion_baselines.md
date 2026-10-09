# Vision world models for locomotion/navigation: tasks, baselines, metrics (survey 2026-10-08)

Legend: [V] = checked by web search (title, authors, venue, link). [M] = detail from memory of the paper, not re-checked this session; check the PDF before citing it in the paper.

## 1. Vision-based world models / predictive planners

| Work | Venue | What it is | Eval task | Baselines | Metrics | Code |
|---|---|---|---|---|---|---|
| Navigation World Models (NWM). Bar, Zhou, Tran, Darrell, LeCun [V] | CVPR 2025 (oral) [arXiv 2412.03572](https://arxiv.org/abs/2412.03572) | Conditional diffusion transformer (CDiT, 1B params) predicts future egocentric frames from past frames and nav actions. Plans by simulating trajectories and scoring them against a goal image [V] | Image-goal navigation planning on RECON/SCAND/TartanDrive/HuRoN; planning with constraints; standalone planning and ranking of NoMaD samples [M] | GNM, NoMaD; DIAMOND for video prediction [M] | ATE, RPE (trajectory error to the goal); DreamSim/LPIPS/FVD for prediction [M] | Yes: models and training code ([HF](https://huggingface.co/facebook/nwm)) [V] |
| DINO-WM. Zhou, Pan, LeCun, Pinto [V; authors M] | ICML 2025 [PMLR](https://proceedings.mlr.press/v267/zhou25t.html), [arXiv 2411.04983](https://arxiv.org/html/2411.04983v2) | Frozen DINOv2 patch features, predicts the next features, CEM/MPC toward the goal features [V] | PointMaze, Push-T, Wall, Rope/Granular, Reacher: reach a goal observation [V/M] | DreamerV3, TD-MPC2, IRIS, AVDC [M] | Success rate, Chamfer distance (deformables) [M] | Yes (github gaoyuezhou/dino_wm) [M] |
| V-JEPA 2 / V-JEPA 2-AC. Assran et al. [V] | arXiv 2506.09985, 2025 [link](https://arxiv.org/abs/2506.09985) | Frozen V-JEPA 2 encoder plus an action-conditioned predictor trained on <62 h of DROID video; CEM planning toward a goal-image embedding [V] | Zero-shot Franka reach/grasp/pick-and-place in two labs [V] | Octo (BC policy), Cosmos (video-generation WM) [M] | Success rate per task; time per plan step [M] | Yes ([vjepa2 repo](https://cdn.jsdelivr.net/gh/facebookresearch/vjepa2@main/README.md)) [V] |
| DayDreamer. Wu, Escontrela, Hafner, Goldberg, Abbeel [V] | CoRL 2022 [PMLR](https://proceedings.mlr.press/v205/wu23c.html) | Dreamer trained online on real robots, including an A1 quadruped learning to walk in about 1 h [V/M] | Walking, pick-and-place, visual navigation [M] | SAC, PPO (model-free), same data budget [M] | Reward vs. wall-clock time [M] | Yes [project](https://danijar.com/project/daydreamer) [V] |
| World Model-based Perception (WMP) for visual legged locomotion [V] | arXiv 2409.16784 [link](https://arxiv.org/abs/2409.16784v1) | Dreamer-style WM over depth as the perception module of a legged policy [V] | Parkour/terrain traversal (A1) [M] | Teacher-student, model-free [M] | Success rate, traversal rate [M] | Yes (M) |
| DreamerNav [V] | Frontiers Robotics & AI 2025 [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12510832/) | DreamerV3 with depth plus a local occupancy map; robot-agnostic, transferred to two quadrupeds [V] | Dynamic indoor obstacle navigation [V] | Not checked | Success / collision (M) | Unknown |
| BADGR. Kahn, Abbeel, Levine [V] | RA-L 2021 [arXiv 2002.05700](https://arxiv.org/pdf/2002.05700) | Image + candidate action sequence -> predicted collision / bumpiness / position; planner picks the best sequence [V] | Real off-road/urban goal reaching with collision avoidance [V] | LiDAR geometric planner (SLAM-style) [M] | Success %, collision, bumpiness [M] | Yes [M] |
| ViNT / GNM / NoMaD. Shah, Sridhar et al. [V for NoMaD] | NoMaD ICRA 2024 [arXiv 2310.07896](https://arxiv.org/abs/2310.07896v1) | Image-goal navigation foundation policies; no future-frame imagination (ViNT can use a diffusion subgoal generator [M]) [V] | Exploration and image-goal navigation on real robots [V] | ViNT, subgoal diffusion, random-subgoal, VIB, Masked-ViNT (5 methods) [V/M] | Success rate, collisions per run; NoMaD reports 43% fewer collisions than ViNT on RECON (taken from a search summary, check before citing) [V*] | Yes (visualnav-transformer) [M] |
| Genie / LAPA (latent actions from video). [V] | [LAPA arXiv 2410.11758](https://arxiv.org/pdf/2410.11758) | Latent actions inferred from consecutive frames, then mapped to real actions [V] | Manipulation (LAPA) | VLAs trained on action labels (OpenVLA) [V/M] | Success rate | Yes (LAPA) [M] |

Not verified this session: iVideoGPT (NeurIPS 2024), UniSim (ICLR 2024), PlaNet (ICML 2019), Egocentric visual self-modelling, LAC-WM, X-Mobility ([arXiv 2410.17491](https://arxiv.org/html/2410.17491v1), shows up in search: a WM-based generalisable navigation model). Cite these only after checking them.

## 2. Predictive planning vs. reactive / model-free: how fair comparisons are set up

| Paper | Prediction arm | Non-predictive arm | Fairness control |
|---|---|---|---|
| BADGR [V] | MPC over action sequences scored by predicted collision | Geometric LiDAR planner [M] | Same robot and goals; the learned model uses only a camera [M] |
| NWM [V] | Simulate candidate trajectories, score them against the goal | NoMaD/GNM policies trained on the same nav datasets [M] | Same datasets; NWM also ranks NoMaD's own samples, so the only difference is the prediction step [M] |
| DINO-WM [V] | MPC in frozen-feature space | Model-free / other WMs with the same offline data [M] | Same offline trajectories and no reward [V] |
| V-JEPA 2-AC [V] | CEM in embedding space | Octo BC policy fine-tuned on the same DROID data [M] | Same data source and zero-shot labs [M] |
| DayDreamer [M] | Dreamer | SAC/PPO | Same interaction budget |

Reusable rule from these papers: keep the sensor, the action library and the data fixed, and remove only the imagination step. NWM's "rank NoMaD samples" setup matches our direct vs. rollout pair: same candidate set, different scorer.

## 3. Cross-embodiment locomotion / navigation

| Work | Venue | Bodies | Baselines | Metrics | Code |
|---|---|---|---|---|---|
| CrossFormer. Doshi et al. [V] | CoRL 2024 [PMLR](https://proceedings.mlr.press/v270/doshi25a.html) | 900K trajectories, 20 embodiments: arms, wheeled nav, quadruped, quadcopter [V] | Single-robot specialists, prior cross-embodiment methods (e.g. Yang et al. 2024, Octo) [V/M] | Success rate per robot [M] | Yes [M] |
| URMA, "One Policy to Run Them All". Bohlinger et al. [V] | CoRL 2024 [PMLR](https://proceedings.mlr.press/v270/bohlinger25a.html) | 16 embodiments from 3 legged morphologies; zero-shot to an unseen quadruped [V] | Multi-task RL baselines with padding/fixed I/O [M] | Return, robustness to observation dropout, zero-shot transfer [V] | Yes [M] |
| Body Transformer. Sferrazza et al. [V] | CoRL 2024 [PMLR](https://proceedings.mlr.press/v270/sferrazza25a.html) | Graph of the body's sensors and actuators [V] | Vanilla transformer, MLP [V] | Task completion, scaling, compute [V] | Yes [V] |
| X-Nav. Wang, Tan, Fung, Nejat [V] | RA-L 2025 [arXiv 2507.14731](https://arxiv.org/abs/2507.14731) | Random wheeled and quadruped bodies; RL experts distilled into Nav-ACT [V] | Not checked | Success rate, zero-shot transfer to new bodies [V/M] | Unknown |
| NaVILA [V] | RSS 2025 [M] ([overview](https://alphaxiv.org/overview/2412.04453v2)) | VLA plus a low-level legged locomotion policy [V] | VLN methods on R2R-CE [M] | SR, SPL, nav error, OSR [M] | Yes [M] |
| ViNL [V] | ICRA 2023 [M] [arXiv 2210.14791](https://arxiv.org/pdf/2210.14791) | Quadruped navigating and stepping over obstacles from egocentric vision [V] | Blind/no-obstacle-aware locomotion [M] | Success, SPL, collisions [M] | Yes [M] |

## 4. Recommended baseline set for the wall/maze test

| ID | Baseline | Purpose | Public code |
|---|---|---|---|
| a | Blind direct scorer: picks an action from predicted velocity / Froude z only, no future views | Same library and model, prediction step removed (the "direct" arm). This is the main ablation and mirrors NWM ranking NoMaD samples | Ours |
| b | Reactive proximity controller: turns when an obstacle signal (range or image-based nearness) passes a threshold | Shows what a hand-made reactive rule with the same sensors achieves. Report its threshold sweep | Trivial to write |
| c | Current-frame vision policy (no prediction): scorer on the current embedding, e.g. a classifier from the V-JEPA2 embedding to the action, or NoMaD/ViNT run with the same camera | Separates "uses vision" from "imagines the future" | NoMaD/ViNT/GNM public [M] |
| d | Other world models: NWM (rollout plus scoring), DINO-WM (frozen DINOv2 WM + CEM) | External WM comparison under the same candidate library | NWM [V], DINO-WM [M], V-JEPA 2-AC [V] |
| e | Explicit-action WM (conditioned on the commanded gait/velocity) vs. latent-action z | Tests what the latent action adds, with the same predictor size and data | Ours (V-JEPA 2-AC is the explicit-action reference) |

Keep across all arms: same start poses, wall distances, headings, number of seeds, candidate library and horizon. Put compute per decision in a column.

## 5. Metrics

| Metric | Used by | Note |
|---|---|---|
| Collision rate (% episodes with contact) | NoMaD, BADGR [V] | Primary for wall avoidance |
| Success rate (goal reached without collision) | DINO-WM, V-JEPA 2-AC, CrossFormer [V/M] | Maze stage |
| SPL (success weighted by path length) | NaVILA, ViNL, Habitat-style [M] | Maze stage |
| Minimum distance to the obstacle / clearance | Common in avoidance work [M] | Continuous, so it separates "turned early" from "barely missed" |
| Time to collision at the first turn command / turn-onset distance | Not found as a standard metric (unverified) | Direct measure of "turns early"; define it explicitly |
| ATE/RPE vs. the goal trajectory | NWM [M] | Planning quality |
| Normalised score (vs. random = 0 and oracle = 1) | DINO-WM-style [M] | For comparing bodies with different speeds; Froude normalisation fits here |
