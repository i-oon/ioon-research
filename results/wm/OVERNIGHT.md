# Stage 2 on clean data — 2026-08-12

Two runs, identical but for the seed. First Stage 2 whose data is defensible: every training body
walks, the embodiments are balanced by data rather than by repetition, validation is stratified,
and a body is held out.

    hexapod   4 bodies x 4 clips x 65   = 1,040 pairs
    b1        2 policies x 6 clips      = 1,003 pairs      ratio 1.04:1
    held out  c08f09t09                                    never trained on
    excluded  c06f06t10, c10f06t10                         these collapse and rotate (F42)
    withheld  c06f10t06, c10f10t06                         these veer 0.35-0.40 m off course

60 epochs, converged: val moved 0.0001 per epoch over the last six, train 1.5556 against val
1.5708 with no divergence.

## Held-out body — the new measurement

Stage 2 has never had a generalisation test before; with two embodiments neither can hold the
other out, so a hexapod body was withheld instead. `scripts/diagnostics/score_body.py`, same weights, one
unseen body.

| | seed 0 | seed 1 |
|---|---|---|
| deg per joint | 3.85 | 3.43 |
| **R^2 against the body's own mean** | **+0.87** | **+0.90** |
| latent zeroed | 0.365 | 0.444 |
| frame zeroed | 0.193 | 0.395 |

**Positive R^2 on both seeds.** Every Stage 1 held-out body scored negative, -0.42 to -3.16. Both
inputs are used: zeroing the latent costs 5-7x, zeroing the frame 2.6-6.7x.

Stage 1's `m3d_cross` scores 2.91 deg on the same body, so learning a quadruped alongside costs
about 30% of hexapod accuracy and does not break it.

**Caveat**: `c08f09t09` is coxa 0.8, femur 0.9, tibia 0.9 -- inside the training range on all
three axes. This is interpolation, not the extrapolation slide 8 fails at. It was chosen because
Stage 1 held out the same body, which makes the two stages comparable.

**Caveat**: the `zero_x` figures differ 2x between seeds. Every input-ablation ratio we report,
including the older z-gap and x-gap, deserves that scepticism.

## Identity ablation — F39 is reversed

Does anything downstream *use* the embodiment identity in `z`, or is it only present? Delete the
directions carrying it, and compare against deleting the same number of random directions.

| | identity removed | random control | verdict |
|---|---|---|---|
| contaminated run (F39) | **1.69x** | 1.16x | load-bearing |
| **clean, seed 0** | **1.03x** | 1.18x | **passive** |
| **clean, seed 1** | **1.04x** | 1.14x | **passive** |

**Removing the identity costs less than removing arbitrary directions, on both seeds.** F39 was an
artefact of two collapsing robots and a 10:1 imbalance. It should be withdrawn.

`z` itself is heavily used -- zeroing it costs 7.6-8.3x. The latent does real work; its identity
component does not.

Three consequences:

- **The side channel's null result is explained.** It relieved a pressure that does not exist.
- **The adversary becomes the right next experiment, with a prediction**: if identity is passive,
  stripping it should cost nothing -- unlike Stage 1, where a single shared head genuinely needed
  body identity and the adversary made transfer 1.21x worse.
- **Question 2 for the professor sharpens.** Not "the shared trunk built a switch" but: the
  identity is leakage from the frozen encoder, unused by any module and unpenalised by any loss,
  and whether that harms transfer to a genuinely new embodiment cannot be tested with two
  embodiments.

## Noise floor

Two seeds, same config, same machine class.

| | seed 0 | seed 1 | spread |
|---|---|---|---|
| val total | 1.5708 | 1.5926 | 1.4% |
| val motion | 0.0203 | 0.0247 | **22%** |
| held-out R^2 | +0.87 | +0.90 | 0.03 |
| identity ablation | 1.03x | 1.04x | 0.01 |
| random control | 1.18x | 1.14x | 0.04 |

The motion term is five times noisier than the total. The ablation ratios are stable. Same-seed
reruns differ by 0.14% at epoch 1, so PyTorch is not deterministic here even with a fixed seed --
cuDNN kernel choice and non-associative float addition, not the seed.

## The variance share is not a usable number

Re-measured on the bodies each checkpoint actually trained on:

| | seed 0 | seed 1 | spread |
|---|---|---|---|
| gait phase | 44.9% | 61.2% | 16.3 pts |
| **which embodiment** | **12.0%** | **6.7%** | **5.3 pts, nearly 2x** |
| interaction | 43.2% | 32.1% | 11.1 pts |
| probe | 0.994 | 0.992 | 0.002 |
| cluster separation | 0.39x | 0.24x | 0.15 |

The decomposition balances its grid by subsampling every cell to the smallest, and the smallest
holds **six latents**. The whole measurement rests on 2 embodiments x 6 phase bins x 6 latents =
**72 points**, which is why two seeds of the same config disagree by a factor of two.

**F38's headline "33.0% of the latent is which robot" rested on the same 72 points.** It is in the
deck. The claim should move to the probe and the ablation, which reproduce to three decimals:

> The embodiment is fully decodable from the latent at 0.99, and nothing uses it -- removing it
> costs less than removing random directions.

Untested cheap fix: `--bins 3` doubles the latents per cell. Worth checking whether the seeds
converge at that resolution before calling the measurement unusable rather than under-sampled.

Also withdrawn: the 23.0% and 20.7% reported earlier used a stale five-body list that included two
bodies these runs hold out. Every diagnostic carried its own hardcoded `INSECT_BODIES`, in three
different versions; `wm/evaluate.py:training_bodies(cfg)` now derives it from the checkpoint.

## Bugs found and fixed

- **The encoder cache has no locking.** Three scripts wrote `results/wm/cache/stage2_embeddings.pt`
  concurrently; last-writer-wins meant each clobbered the others, so every later run re-encoded
  all 29 clips on CPU. This is what made overnight jobs appear to hang for nine hours. Needs a
  per-script path or a lock before any parallel run.
- **Cached GPU tensors.** `--encode_device cuda` cached tensors on the GPU while the trained
  modules stayed on the CPU. Fixed in four scripts.
- **`checkpoint_every 2` at 60 epochs** writes 30 snapshots, 11 GB, which filled the disk to 100%.
  Use `--checkpoint_every 10` for long runs.
- **`pkill -f`** matches its own command line.

## Centring, measured 2026-08-12 — it does nothing

The last unevaluated survivor. Now scored on all three instruments:

| | 4-leg few-shot split A | held-out hexapod R^2 | probe after 8 dirs | `z` zeroed |
|---|---|---|---|---|
| `stage2_clean` | 1.86 deg | +0.87 | 0.738 | 7.63x |
| **`stage2_clean_centered`** | **1.88** | **+0.89** | 0.697 | **9.96x** |
| `stage2_clean_adv_warm10` | **1.66** | +0.88 | **0.598** | **4.44x** |

Identity stays fully decodable -- the online probe hits **1.000** during training, having started
at 0.594 and climbed back over 25 epochs with the offset already removed. Centring takes out the
first moment; the robots differ in shape, silhouette and leg count, which vary frame to frame.

**A bug of mine cost a wrong headline first.** `score_body.py` did not apply the stored
`embedding_offsets`, so the centred checkpoint was scored on raw embeddings and read **15.10 deg,
R^2 -0.95** -- reported as "centring breaks transfer" before I found it. Corrected: 3.61 deg,
R^2 +0.89. `fit_4leg_head` is immune to the same omission because its new head is fitted on the
target data and absorbs a constant offset into its bias.

All three levers are now measured and written up as F44.

## Not done

- clean UMAP regenerated by Codex as `cross_embodiment_umap_stage2_clean.png`; the older
  `cross_embodiment_umap.png` is stale
- per-leg contact probe -- `LogisticRegression(max_iter=3000)` on 5,632 features will not
  converge; needs a different solver or fewer features before it is rerun
- `--bins 3` check on the variance decomposition (Q13)
- the forward-model rollout comparison for `stage2_clean_adv_warm10`, which is what would decide
  whether the adversary replaces the clean baseline

## Codex handoff — 4-leg middle-loss probe and small code edits

Added after the Stage 2 clean/centred/adversary discussion, to record the local changes and the
4-leg evidence collected for the cross-embodiment pivot.

### Run-directory accident and what remains locally

At one point `results/wm/runs` / `wm/runs` content was accidentally removed before the Stage 2
handoff was complete. These runs were not in git, so the deleted checkpoint directories were not
recoverable from version control. Some empty/name-only directories remain under `wm/runs/_gone`,
but they do not contain usable model files.

What is currently left locally:

```text
wm/runs/stage2_clean/best.pt
wm/runs/stage2_clean/best_motion.pt
wm/runs/stage2_clean/summary/

wm/runs/stage2_clean_centered/best.pt
wm/runs/stage2_clean_centered/config.yaml
wm/runs/stage2_clean_centered/summary/

results/wm/stage2/logs/stage2_clean_adv.log
```

What is **not** left locally:

```text
stage2_clean_adv checkpoint/model files
stage2_clean_s1 checkpoint/model files
older Stage 1/Stage 2 run checkpoints, except for any unrelated artefacts already copied under results/
```

The adversarial run therefore needs retraining before it can be evaluated or used. The old
`results/wm/stage2/logs/stage2_clean_adv.log` is still useful as evidence of the previous trend and as the
source for the warmup setting below, but it is not itself a resumable model.

### Stage 2 adversary reproducibility note

The old `stage2_clean_adv` log in `results/wm/stage2/logs/stage2_clean_adv.log` is not reproduced by the
quick retrain command unless `--adv_warmup_epochs 10` is passed explicitly. The old log ramped
the adversary as `x0.10, x0.20, ..., x1.00` over the first ten epochs; the current config default
is five epochs, so a command without that flag ramps `x0.20, x0.40, ..., x1.00`.

Consequence: a new run launched without `--adv_warmup_epochs 10` is a valid warmup-5 run, but it
is not directly comparable to the old crash log. If the goal is to reproduce the old promising
trend, use a new name and set the warmup explicitly:

```bash
.venv/bin/python3 -m wm.train \
  --name stage2_clean_adv_warm10 \
  --sources hexapod=data/ik_walk_8body b1=data/b1_framed \
  --heldout_bodies c06f10t06 c08f09t09 c10f10t06 \
  --clips_per_body hexapod=5 \
  --action_lag 1 \
  --lambda_adv 0.1 \
  --adv_warmup_epochs 10 \
  --lambda_cross 0.0 \
  --cross_augment true \
  --within_body_std true \
  --md_head mlp \
  --balance_embodiments true \
  --ftm_embodiment_channel false \
  --center_embeddings false \
  --epochs 60 \
  --batch_size 8 \
  --lr 0.0001 \
  --seed 0 \
  --checkpoint_every 10 \
  2>&1 | tee results/wm/stage2/logs/stage2_clean_adv_warm10.log
```

### 4-leg middle-loss dataset

The candidate 4-leg embodiment is the original stick insect with the middle legs removed
(`ML,MR`). This is the cleanest 4-leg test among the leg-loss variants: the previous leg-loss
strip showed middle-loss walking upright, while front-loss and hind-loss were dominated by
rotation/collapse.

Sanity collection used the normal stick-insect scene, ghost-removing the middle legs at runtime:

```bash
.venv/bin/python3 sim/collect/collect_ik.py \
  --port 23063 \
  --episodes 6,20,22 \
  --morphs middleloss=medauroidea_stick_insect.ttt \
  --remove_legs ML,MR \
  --active_legs FL,HL,FR,HR \
  --scale 0.5 \
  --travel 0.8 \
  --warmup 20 \
  --cam_dx -0.6 \
  --cam_dy 0.0 \
  --spawn 0 0 \
  --out data/ik_4leg_middleloss_sanity
```

It produced 3 clips / 198 frames. Shapes were correct:

```text
frames  (66, 256, 256, 3)
actions (66, 12)
forces  (66, 4)
legs    FL,HL,FR,HR
```

Then a larger candidate sweep was collected into `data/ik_4leg_middleloss_candidates` with 30
expert episodes. Many clips veered laterally after the middle legs were removed. The clips marked
`ok` by the collector were copied into:

```text
data/ik_4leg_middleloss_clean9
```

Selected clips:

```text
middleloss_ep6
middleloss_ep22
middleloss_ep28
middleloss_ep69
middleloss_ep93
middleloss_ep101
middleloss_ep130
middleloss_ep144
middleloss_ep198
```

Best-looking by the simple displacement numbers:

```text
ep144  forward +0.71 m, lateral 0.15 m
ep28   forward +0.63 m, lateral 0.16 m
ep198  forward +0.68 m, lateral 0.17 m
ep93   forward +0.69 m, lateral 0.18 m
```

Rendered previews were exported from the `.npz` frames, no simulator needed:

```text
results/dataset/ik_4leg_middleloss_clean9_preview/
  grid_overview.mp4
  middleloss_ep6.mp4
  middleloss_ep22.mp4
  middleloss_ep28.mp4
  middleloss_ep69.mp4
  middleloss_ep93.mp4
  middleloss_ep101.mp4
  middleloss_ep130.mp4
  middleloss_ep144.mp4
  middleloss_ep198.mp4
```

Status: usable as a small 4-leg probe set, not yet a full training dataset. It is good enough to
check whether Stage 2 features help a new embodiment; it is not perfect evidence of straight
4-leg walking because several accepted clips still have ~0.19--0.20 m lateral drift.

### Code changes made for the 4-leg probe

`sim/collect/collect_ik.py`

- Added `CHAIN_NAMES` to enumerate the named objects in each leg chain.
- Changed `read_forces` from fixed six feet to `len(force_h)`, so 4-foot datasets save `(T,4)`
  contact/force arrays instead of a hard-coded `(T,6)`.
- Added `get_optional`, `leg_subtree`, and `ghost_remove_legs`.
- `ghost_remove_legs` hides selected leg shapes, makes them non-respondable when possible, and
  zeros joint target force without deleting handles that scene scripts may still expect.
- Added `active_legs` and `remove_legs` arguments to `drive_and_record`.
- Saved actions now use only active leg columns, e.g. `FL,HL,FR,HR` gives 12-D actions.
- Saved `foot_order` now records the active legs instead of always recording all six legs.
- During warmup and stepping, removed-leg joints have their target force repeatedly zeroed.
- Added CLI flags:
  - `--active_legs`, default `FL,ML,HL,FR,MR,HR`
  - `--remove_legs`, default empty
- Added validation that requested legs are in the known six-leg set.

`sim/render/npz_to_video.py`

- Added `--out`, so inspection videos can be written outside the dataset directory.
- Made the grid overview work for arbitrary clip names, not only `long|medium|short`.
- Added an empty-match guard.
- Changed the grid print from `grid (long|medium|short)` to `grid overview`, because middle-loss
  clips are not morphology triplets.

`scripts/diagnostics/fit_4leg_head.py`

- Added a few-shot 4-leg held-out test script.
- It loads a Stage 2 checkpoint, freezes the trained ITM and Motion Decoder backbone, adds a new
  `middleloss` 12-D output head, fits only that head on a few 4-leg clips, and scores held-out
  clips.
- It also runs a `random_backbone` control with the same architecture and same new-head fitting,
  so the useful comparison is pretrained Stage 2 vs random backbone under the same data budget.
- Default split:
  - train: `ep144, ep28, ep198, ep93, ep22`
  - test: the remaining clean9 clips (`ep101, ep130, ep6, ep69`)
- `py_compile` passes.
- Local smoke run reached V-JEPA encoding, but was stopped: CPU encoding was too slow, and this
  local session reports no usable NVIDIA driver via `nvidia-smi`. Run it on the GPU workstation.
- Bugfix after first GPU run: the cached features returned to CPU while the new head stayed on
  CUDA, causing a LayerNorm device mismatch. `fit_head` and `metrics` now move feature/target
  tensors to the head's device before running the head, and metrics no longer force the head to
  CPU.
- Added `--save_pred`, which saves the pretrained head's held-out predictions and ground truth
  actions as replay-ready flat sequences with per-clip lengths.
- Added `--test_data`, allowing the 4-leg head to be calibrated on clean clips and tested on a
  separate bad/veering clip directory without copying files around.
- Added `--z_ablation`, which fits separate new heads using:
  - `real_z`: normal ITM transition latent
  - `zero_z`: all-zero latent, testing how much the current frame/backbone alone can do
  - `shuffled_z`: real latents permuted within each clip, preserving z distribution but breaking
    frame-transition alignment

`sim/render/render_wm_prediction.py`

- Added `--active_legs` and `--remove_legs` so replay can drive a reduced-action embodiment such
  as the 4-leg middle-loss insect.
- The replay output now stores `active_legs` and `remove_legs`.
- The printed contact summary now reports `of {len(active_legs)}` instead of always `of 6`.

`scripts/diagnostics/wm_gait_report.py`

- Reads `active_legs` from replay files and draws the correct number/order of contact rows.
- Skips the 6-leg tripod score for 4-leg replays, reporting `nan` under a generic `coord`
  column instead of pretending a hexapod tripod exists.
- Allows joint-range checks to use `data/ik_4leg_middleloss_clean9` for the middle-loss body.

`scripts/diagnostics/sweep_4leg_fewshot.py`

- Added a few-shot curve sweep for the 4-leg new-head test.
- It encodes the 9 clean4-leg clips once, then fits new heads for multiple clip budgets and
  random train/test splits.
- Outputs:
  - `results/wm/stage2/4leg_head/fewshot_curve.csv`
  - `results/wm/stage2/figures/4leg_fewshot_curve.png`

### Baseline Stage 2 test plan for 4-leg

The correct test is **few-shot new-head**, not zero-shot. The Stage 2 baseline checkpoint has
two trained output heads:

```text
hexapod head -> 18-D stick-insect commands
b1 head      -> 12-D B1 commands
```

The 4-leg stick insect also has 12 actions, but those 12 coordinates are not the B1 action space.
Using the B1 head directly would only match dimensionality, not semantics.

Recommended test:

1. Load `wm/runs/stage2_clean/best.pt`.
2. Freeze V-JEPA, ITM, FTM, and the Motion Decoder backbone.
3. Add a new `middleloss` output head with `MotionDecoder.add_head(..., action_dim=12)`.
4. Fit only that new head on a few clips from `data/ik_4leg_middleloss_clean9`.
5. Score on held-out middle-loss clips.
6. Compare against a scratch/random-backbone control trained with the same number of clips.

Command now available:

```bash
.venv/bin/python3 scripts/diagnostics/fit_4leg_head.py \
  --ckpt wm/runs/stage2_clean/best.pt \
  --data data/ik_4leg_middleloss_clean9 \
  --train_eps 144,28,198,93,22 \
  --epochs 300 \
  --encode_device cuda \
  --device cuda \
  --chunk 4
```

First result on the clean9 split:

```text
train clips: ep144, ep28, ep198, ep93, ep22
test clips : ep101, ep130, ep6, ep69

model                best_ep  train_deg  test_deg  test_mse  own_mean      R2
pretrained_stage2        300       1.00      1.86    0.0369    0.9808   +0.96
random_backbone          275       3.46      5.06    0.2531    0.9808   +0.74
```

Interpretation: the pretrained Stage 2 backbone gives a strong few-shot advantage on the unseen
4-leg embodiment. Test error is 2.7x lower in degrees (1.86 vs 5.06) and 6.9x lower in
standardised MSE (0.0369 vs 0.2531) than the random-backbone control under the same new-head
training budget. This is evidence for **few-shot transfer via a new embodiment-specific head**,
not zero-shot action transfer.

Repeated over three 5-train / 4-test splits:

| split | train episodes | pretrained test deg | random test deg | ratio | pretrained R2 | random R2 |
|---|---|---:|---:|---:|---:|---:|
| A | 144,28,198,93,22 | 1.86 | 5.06 | 2.72x | +0.96 | +0.74 |
| B | 144,28,198,101,130 | 1.67 | 4.72 | 2.83x | +0.97 | +0.77 |
| C | 144,93,101,6,69 | 1.71 | 5.18 | 3.03x | +0.97 | +0.72 |

Mean over splits:

```text
pretrained_stage2  test_deg 1.75 +/- 0.10, R2 +0.967
random_backbone    test_deg 4.99 +/- 0.24, R2 +0.743
degree-error gain  2.86x
```

This makes the 4-leg result much less likely to be a lucky held-out split. The claim should stay
precise: Stage 2 supports **few-shot calibration** of a new 4-leg embodiment, not direct
zero-shot control.

Few-shot curve over budgets 1, 3, 5, 7 clips, three random splits each:

| clips used for new head | pretrained Stage 2 | random backbone | gain |
|---:|---:|---:|---:|
| 1 | 2.56 +/- 0.18 deg | 6.68 +/- 0.39 deg | 2.61x |
| 3 | 1.97 +/- 0.04 deg | 5.35 +/- 0.17 deg | 2.72x |
| 5 | 1.75 +/- 0.05 deg | 5.09 +/- 0.35 deg | 2.91x |
| 7 | 1.71 +/- 0.08 deg | 4.78 +/- 0.08 deg | 2.80x |

This strengthens the interpretation from "better accuracy" to **better sample efficiency**: the
pretrained Stage 2 backbone needs far fewer 4-leg clips to calibrate a usable action head than a
random backbone.

Artefacts:

```text
results/wm/stage2/4leg_head/fewshot_curve.csv
results/wm/stage2/figures/4leg_fewshot_curve.png
```

Predictions were exported for split A:

```text
results/wm/stage2/4leg_head/splitA_predictions.npz
```

Open-loop replay in CoppeliaSim was then rendered for all four held-out split-A clips, using the
normal stick-insect scene with `ML,MR` ghost-removed and active legs `FL,HL,FR,HR`.

| clip | source ep | predicted forward/lateral | IK forward/lateral | mean feet down pred/IK | out-of-range pred |
|---|---|---:|---:|---:|---:|
| 0 | ep101 | +0.660 / -0.233 m | +0.701 / -0.239 m | 2.55 / 2.55 of 4 | 1.8%, worst 3.0 deg |
| 1 | ep130 | +0.665 / -0.188 m | +0.694 / -0.181 m | 2.58 / 2.60 of 4 | 2.4%, worst 2.7 deg |
| 2 | ep6 | +0.713 / -0.168 m | +0.692 / -0.277 m | 2.51 / 2.55 of 4 | 1.4%, worst 1.5 deg |
| 3 | ep69 | +0.650 / -0.167 m | +0.655 / -0.273 m | 2.58 / 2.65 of 4 | 2.4%, worst 2.0 deg |

Rendered artefacts:

```text
results/wm/4leg_head/
  replay_splitA_clip0.npz
  replay_splitA_clip1.npz
  replay_splitA_clip2.npz
  replay_splitA_clip3.npz
  replay_replay_splitA_clip0.mp4
  replay_replay_splitA_clip1.mp4
  replay_replay_splitA_clip2.mp4
  replay_replay_splitA_clip3.mp4
  gait_replay_splitA_clip0.png
  gait_replay_splitA_clip1.png
  gait_replay_splitA_clip2.png
  gait_replay_splitA_clip3.png
```

Replay verdict: predicted actions physically replay as stable 4-leg walking and closely match the
IK reference over the 65-step held-out clips. In two held-out clips the predicted pass actually
drifts less laterally than the IK reference. This is still open-loop replay of action
reconstruction, not autonomous closed-loop control, but it is much stronger evidence than the
numeric action error alone.

Bad-gait / veering stress test:

`data/ik_4leg_middleloss_badtest` was re-collected after the original candidates directory was
deleted. It contains 8 clips / 528 frames:

```text
ep20, ep34, ep70, ep138, ep167, ep210, ep246, ep262
```

All are forward-moving but laterally veering (`|dy|` roughly 0.22--0.31 m over 66 frames). The
same clean-head setup was used: train/calibrate on clean9 split-A training clips and test on this
separate badtest directory.

```text
model                best_ep  train_deg  test_deg  test_mse  own_mean      R2
pretrained_stage2        300       1.00      2.31    0.0582    0.9779   +0.94
random_backbone          100       5.27      6.75    0.4599    0.9779   +0.53
```

Interpretation: the model still decodes the bad/veering 4-leg actions far better than the random
control. It does **not** repair the demonstrations; when the IK gait veers, the replay generally
veers too. That is the expected behaviour for an action-reconstruction model.

Rendered bad-gait clips:

```text
results/wm/4leg_head/
  badtest_predictions.npz
  badtest_replay_clip2.npz        # ep20
  badtest_replay_clip7.npz        # ep70
  replay_badtest_replay_clip2.mp4
  replay_badtest_replay_clip7.mp4
  gait_badtest_replay_clip2.png
  gait_badtest_replay_clip7.png
```

Bad-gait replay metrics:

| clip | source ep | predicted forward/lateral | IK forward/lateral | mean feet down pred/IK | out-of-range pred |
|---|---|---:|---:|---:|---:|
| 2 | ep20 | +0.617 / -0.319 m | +0.677 / -0.348 m | 2.60 / 2.52 of 4 | 1.7%, worst 4.4 deg |
| 7 | ep70 | +0.622 / -0.315 m | +0.718 / -0.245 m | 2.55 / 2.55 of 4 | 1.5%, worst 1.4 deg |

This answers the "bad gait" concern: matching a bad IK clip means the decoder understood the
visual/action correspondence even when the behaviour is not desirable. It should not be framed as
gait correction.

Z-ablation on clean split A, checking whether the 4-leg result is only the old "one frame already
determines the command" issue:

```text
model                 best_ep  train_deg  test_deg  test_mse  own_mean      R2
pretrained_stage2         300       1.00      1.86    0.0369    0.9808   +0.96
pretrained_zero_z         300       1.65      2.49    0.0635    0.9808   +0.94
pretrained_shuffled_z     200       1.55      3.35    0.1189    0.9808   +0.88
random_backbone           275       3.46      5.06    0.2531    0.9808   +0.74
```

Interpretation: the old structural fact is partly present. A zero latent still works better than
random because one frame plus the pretrained decoder backbone carries strong gait-phase/action
information. But the aligned transition latent is not redundant: `real_z` improves over `zero_z`
(1.86 vs 2.49 deg) and strongly over `shuffled_z` (1.86 vs 3.35 deg). So the cautious claim is:
the 4-leg result uses both the current-frame representation and an aligned latent transition
signal; it is not pure zero-shot latent-action control, and it is not merely random frame-to-action
fitting either.

The same z-ablation on the bad/veering test set strengthens the same conclusion:

```text
model                 best_ep  train_deg  test_deg  test_mse  own_mean      R2
pretrained_stage2         300       1.00      2.31    0.0582    0.9779   +0.94
pretrained_zero_z         300       1.65      3.43    0.1297    0.9779   +0.87
pretrained_shuffled_z     175       1.77      4.33    0.1905    0.9779   +0.81
random_backbone           100       5.27      6.75    0.4599    0.9779   +0.53
```

Summary figures for slides:

```text
results/wm/stage2/figures/4leg_fewshot_and_z_ablation.png
results/wm/stage2/figures/4leg_fewshot_curve.png
results/wm/stage2/figures/4leg_replay_stills.png
```

The stale cross-embodiment UMAP was regenerated from the local `stage2_clean` checkpoint:

```bash
.venv/bin/python3 scripts/diagnostics/cross_embodiment_umap.py \
  --ckpt wm/runs/stage2_clean/best.pt \
  --encode_device cuda \
  --chunk 4 \
  --out results/wm/stage2/figures/cross_embodiment_umap_stage2_clean.png
```

Output:

```text
1909 frames: 780 hexapod, 1129 b1
frozen encoder  e_t    probe 1.000  silhouette +0.638  separation 3.41x
learned latent  z      probe 0.994  silhouette +0.051  separation 0.39x
```

Slide 14 now points to `cross_embodiment_umap_stage2_clean.png` instead of the stale
`cross_embodiment_umap.png`.

The pasted `stage2_clean_adv_warm10` run on `fibo7` completed 60 epochs. It used the corrected
10-epoch adversary warmup (`x0.10` through `x1.00`) and ended at:

```text
epoch 60 | train 1.6325 (recon 1.5449 motion 0.0181)
         | val   1.6465 (recon 1.5544 motion 0.0224)
         | adv 0.456 probe 0.529
best val total 1.6465 | best val motion 0.0223
```

Interpretation from the log only: training is healthy and converged; the adversary lowers the
online probe compared with the clean run, but not to chance in a stable way. It still needs the
same post-run evaluation as `stage2_clean`: held-out hexapod score, identity ablation, and 4-leg
few-shot head test before it can replace the clean baseline.

Post-run evaluation after copying the checkpoint locally to
`wm/runs/stage2_clean_adv_warm10/best.pt`:

Held-out hexapod body `c08f09t09`:

| checkpoint | deg/joint | model MSE | R2 | zero_z | zero_x |
|---|---:|---:|---:|---:|---:|
| `stage2_clean` | 3.84 | 0.073 | +0.87 | 0.365 | 0.193 |
| `stage2_clean_adv_warm10` | **3.64** | **0.066** | **+0.88** | 0.140 | 0.266 |

Identity ablation:

| checkpoint | residual identity probe after 8 dirs removed | identity removed | random 8d removed | z zeroed |
|---|---:|---:|---:|---:|
| `stage2_clean` | 0.738 | 1.03x | 1.18x | 7.63x |
| `stage2_clean_adv_warm10` | **0.598** | **1.01x** | 1.12x | 4.44x |

The adversary does what it was supposed to do in representation space: the removable identity
subspace is weaker, and deleting it is even more free. But it also makes `z` less load-bearing
overall (`z zeroed` cost drops 7.63x -> 4.44x), so this is not a simple win.

4-leg split-A new-head + z-ablation:

| checkpoint | real z | zero z | shuffled z | random backbone |
|---|---:|---:|---:|---:|
| `stage2_clean` | 1.86 | 2.49 | 3.35 | 5.06 |
| `stage2_clean_adv_warm10` | **1.66** | **2.08** | **2.66** | 5.06 |

4-leg few-shot curve, mean held-out deg/joint over three random splits:

| clips | clean | adv warm10 | random backbone |
|---:|---:|---:|---:|
| 1 | 2.56 +/- 0.18 | 2.60 +/- 0.21 | 6.68 +/- 0.39 |
| 3 | 1.97 +/- 0.04 | 1.92 +/- 0.06 | 5.35 +/- 0.17 |
| 5 | 1.75 +/- 0.05 | 1.66 +/- 0.04 | 5.09 +/- 0.35 |
| 7 | 1.71 +/- 0.08 | 1.51 +/- 0.03 | 4.78 +/- 0.08 |

Artifacts:

```text
results/wm/stage2/4leg_head/fewshot_curve_adv_warm10.csv
results/wm/stage2/figures/4leg_fewshot_curve_adv_warm10.png
results/wm/stage2/figures/cross_embodiment_umap_stage2_clean_adv_warm10.png
results/wm/stage2/figures/stage2_clean_vs_adv_summary.png
results/wm/stage2/4leg_head/stage2_clean_vs_adv_summary.csv
```

Working conclusion: `stage2_clean_adv_warm10` is a useful candidate, not an automatic replacement.
It slightly improves held-out hexapod and 4-leg few-shot performance, and it reduces usable
embodiment identity. But because it also reduces the cost of zeroing `z`, the clean baseline should
remain the main reference until the forward-model/rollout side is checked.

Adv UMAP:

```text
frozen encoder  e_t    probe 1.000  silhouette +0.638  separation 3.41x
learned latent  z      probe 1.000  silhouette +0.038  separation 0.31x
```

Compared with `stage2_clean` (`z` silhouette +0.051, separation 0.39x), adv compresses the visual
clusters a bit more in the UMAP/full-space separation metrics. The probe in this script remains
1.000, while the identity-ablation script sees the removable identity subspace weaken to residual
0.598; report the ablation as the more diagnostic measurement.

`report/update_slide.md` was updated with a new Slide 15:

```text
Slide 15 — New held-out embodiment: 4-leg insect with a new head
```

The deck intro and scope text were also updated so Stage 2 is no longer described as appearing
only in the closing questions. The old questions slide is now Slide 16, and the old "three
questions left open" slide is now Slide 17.

Interpretation:

- If pretrained Stage 2 + new head beats scratch, Stage 2 learned useful cross-embodiment
  behaviour/geometry features.
- If it does not beat scratch, the 4-leg test is not yet evidence that the Stage 2 latent
  transfers beyond the trained embodiments.
