# World model results

What each number means and the baselines it has to be read against are in
[FINDINGS.md](../../FINDINGS.md). This file is the index and the per-run metrics.

## Layout

Organised **by stage first, artifact second**. A result belongs to the run that produced it, and a
run belongs to exactly one stage, so that is the outer level.

```
results/wm/
  stage1_correct/  the retrain on clean data -- cite this for current Stage 1 numbers
  stage1/          the original runs -- kept for early findings and the contaminated-cost pair
  stage2/          figures/ gait/ logs/ measurements/ 4leg_head/ ablation_replay/
  dataset/         setup and data-quality checks that belong to neither stage
  cache/           encoded embeddings and latents, shared and regenerable
  README.md        RUNS.md   OVERNIGHT.md
```

**`stage1_correct/` is separate because the retrained runs reuse the same names** --
`m3d_cross`, `m3d_bracketed`, `tib_cross`, `tib_ctrl`, `bracket_cross` -- so their outputs
collide filename-for-filename with `stage1/` and would overwrite it. The two are also not
comparable run-for-run: the retrain corrects `action_lag` **and** drops the non-walking bodies,
so it differs in two things at once. `stage1/` is kept deliberately, because old against new is
the only measurement of what the contaminated data cost. Details in
[stage1_correct/README.md](stage1_correct/README.md).

| | |
|---|---|
| `figures/` | plots referenced from FINDINGS.md or the deck |
| `gait/` | gait diagrams and side-by-side predicted-vs-IK videos |
| `predictions/` | reconstructed joint commands in radians, from `wm.predict_actions` |
| `replay/` | CoppeliaSim replays: frames, foot forces, head trajectories |
| `analysis/` | JSON from `morphology_axis.py`, `morphology_mix.py` |
| `measurements/` | raw stdout dumps from diagnostics, the numbers behind the findings |
| `logs/` | training logs |
| `eval/` | `evaluation.json` from the early two-body runs |
| `4leg_head/` | the held-out 4-leg embodiment: few-shot curves, replays, summaries |
| `ablation_replay/` | identity/frame/latent ablations driven through physics (F43b) |

**`dataset/` is the category that is easy to forget.** Thirteen files -- `morphology_bodies`,
`ratio_gaits_ep6`, `foot_force_threshold`, `dead_zone_comparison`, `leg_loss_strips`,
`frame_clipping_comparison`, `spawn_test` and friends -- are checks on the data and the rendering
setup. They were true before either stage and stay true after, so filing them under a stage would
misattribute them.

### The scripts still write to the old flat paths, on purpose

Stage is a property of the run, not of the script: `plot_action_trace.py` runs on a Stage 1 or a
Stage 2 checkpoint equally, and teaching every script to infer which one is more machinery than
this is worth. So the diagnostics still write to `results/wm/figures/`, `.../predictions/`,
`.../gait/` and `.../replay/`, and those directories **will reappear**.

**Treat them as an inbox, not as the archive.** After a batch of measurements, move what you keep
into the right `stage1/` or `stage2/` subfolder. If a flat directory exists here, it holds
unfiled output.

### Telling a file's stage from its name

Needed when filing the inbox. Everything follows `{script}_{run}_{epoch}_{body}.ext`.

| Stage | Run names that appear in filenames |
|---|---|
| **Stage 1** — cross-morphology, hexapod only, one 18-D head | `m3d_cross`, `m3d_bracketed`, `tib_cross`, `tib_ctrl`, `tib_sound`, `bracket_cross`, `stage1_*`, `fix_norm`, `fold_short`, `lag1_*`, `m3d_noaug`, `m3d_pooled`, `probe_ridge_*` |
| **Stage 2** — cross-embodiment, hexapod + B1, per-embodiment heads | `stage2_clean`, `stage2_clean_s1`, `stage2_clean_centered`, `stage2_clean_adv_warm10`, `stage2_balanced`, `sidechannel_*`, `4leg_*`, `middleloss*`, `cross_embodiment_*`, `embodiment_axis`, `leg_contact_*`, `occlusion_*` |
| **Neither** — dataset and setup checks, valid for both | `morphology_bodies`, `frame_clipping_comparison`, `framing_current`, `robot_framing_final`, `spawn_test`, `replay_frames`, `foot_force_threshold`, `gait_diagram_quality`, `dead_zone_comparison`, `ratio_gaits_*`, `leg_loss_strips`, `encoder_probe_matrix`, `encoder_vs_decoder` |

Body codes disambiguate further: `c08f09t09` and friends are hexapod bodies (Stage 1 and 2 both
use them), `middleloss` is the 4-leg built on the base geometry, `middleloss08` the one built on
held-out `c08f09t09` (F48).

## Where the 2026-08-16 numbers live

The cross-embodiment few-shot result and the re-measured horizon table, so a claim in the deck can
be traced to the file that produced it.

| file | backs |
|---|---|
| *(none in this repo)* | **F52** itself -- the 1/3/5/7/9-clip sweep ran on com7 and its log stayed there. The table is transcribed in FINDINGS.md F52; regenerate with the command in that finding if it is ever needed here. |
| `stage2/logs/preflight_real.log` | the gate that checked the control arms were trained enough to compare |
| `stage2/logs/ft_pretrain_real.log` | F52's control: real temporal pairs |
| `stage2/logs/ft_pretrain_shuffled.log` | F52's control: shuffled pairs, adjacency removed |
| `stage2/measurements/pretrain_control_frozen.txt` | **F54** -- both arms with no adaptation at all, on the B1 and on held-out insect clips. The 1.38x-to-0.54x embodiment gap. |
| `stage2/logs/pretrain_control.log` | how both control arms were trained (see RUNS.md) |
| `stage1_correct/logs/horizon_c10f10t10.log` | **F53** -- F31 re-measured, the table on slide 11 |
| `stage1_correct/logs/horizon_m3d_clean.log` | F53's robustness check, all five bodies pooled |

> An 11-clip row was measured for F52 and is **not reportable**: with 14 clips and 4 held out,
> `train = order[:11]` and `test = order[-4:]` overlap by one clip, so a quarter of the test set
> was in training. The script now refuses that budget outright. Read budgets 1-9 only, and note
> that 10 is the largest clean budget the B1 set allows.

## Which target a run was trained on

`cfg.action_lag` decides which command the Motion Decoder is asked for, counted from frame `t`.
Every run listed below except `lag1_*` was trained with **`action_lag 0`**, where the target is
already visible in the decoder's own input and the latent has no job (FINDINGS.md F29). Their
numbers are still valid against each other and are read back correctly by
`wm.config.from_checkpoint`, which restores the behaviour each checkpoint was trained with.

**MSE is not comparable across the two settings.** `action_lag 1` asks a harder question, so a
larger number there is not a worse model.

## Runs

Two datasets. The MSE scales are **not** comparable between them, because the action
standardisation is computed from each one's own training bodies. Degrees are.

### `data/ik_walk_100_framed` — two training bodies, one morphology parameter

| Run | Training | Held out | Note |
|---|---|---|---|
| `stage1_6ep_clipped` | long, short | medium | 6 episodes, first working run |
| `stage1_100ep_clipped` | long, short | medium | 100 episodes, transfer got worse |
| `stage1_100ep_clean` | long, short | medium | frames 45-65 only |
| `stage1_100ep_framed_runA/runB` | long, short | medium | clean framing; A and B differ only by GPU |
| `fix_norm` | long, short | medium | `within_body_std` |
| `head_linear` | long, short | medium | `md_head linear` |
| `fold_short` | long, medium | short | extrapolation |

### `data/ik_walk_8body` — five training bodies, three morphology parameters

Segment scales are in the name: `c08f09t09` is coxa 0.8, femur 0.9, tibia 0.9.
Training bodies `c10f10t10 c06f10t10 c10f10t06 c06f10t06 c10f06t06` throughout.

| Run | Held out | Flag | Held-out error |
|---|---|---|---|
| `m3d_bracketed` | `c08f09t09` (inside hull) | — the control | 0.0992 |
| `m3d_outside` | `c06f06t06` (uniform 0.6 scale) | — | 1.71-2.61 |
| `m3d_adv01` | `c08f09t09` | `lambda_adv 0.1` | 0.118 |
| `m3d_pooled` | `c08f09t09` | `md_head pooled` | 0.099 |
| **`m3d_cross`** | `c08f09t09` | **`lambda_cross 0.5`** | **0.0760** |
| `m3d_norecon` | `c08f09t09` | `lambda_recon 0` | 0.1025 |

`m3d_cross` was stopped at epoch 27: held-out sat at 0.0760 (epochs 1-10), 0.0705 (11-20) and
0.0715 (21-27), so it had plateaued. Best checkpoint is **epoch 8**, 2.91 deg.

Smoke runs carry a `_smoke_` prefix and existed only to compare interventions cheaply; their
held-out numbers swing by 5x and cannot be read. **`data/_smoke` no longer exists** -- it was 975
pairs and was deleted at some point without a note, so the historical `_smoke_*` numbers cannot be
re-run. Nothing depends on them.

## Held-out body, the numbers that matter

Five bodies, held-out `c08f09t09`, averaged over epochs 1-10 unless stated.

| Run | held-out MSE | best, in degrees | z-gap | x-gap | probe on `z` |
|---|---|---|---|---|---|
| control `m3d_bracketed` | 0.0992 | 3.57 | 21x | 10.7x | 0.724 |
| `lambda_adv 0.1` | 0.118 | — | 5.9x | 19.1x | 0.44 |
| `md_head pooled` | 0.099 | — | 29.6x | **1.4x** | — |
| **`lambda_cross 0.5`** | **0.0760** | **2.91** | 2.4x | **54x** | 0.31-0.66 |

## Does the latent carry the transition

Produced by `scripts/diagnostics/z_dynamics.py`, which substitutes the second frame the ITM is given.
Held-out `c08f09t09`, 195 transitions, both runs `action_lag 0`.

| what the ITM is given as `e_{t+1}` | control ep 6 | cross ep 8 |
|---|---|---|
| the real next frame | 3.57 | 2.91 |
| **`e_t` again, no transition at all** | **3.96 (1.11x)** | **3.47 (1.19x)** |
| a real frame from a random other time | 9.65 (2.70x) | 6.10 (2.10x) |
| `e_{t-1}`, the transition backwards | 5.13 (1.44x) | 4.18 (1.44x) |
| the latent zeroed entirely | 19.24 (5.39x) | 6.04 (2.08x) |

Removing the transition costs 11-19 percent, and reversing it costs more than removing it. `z`
was a pose code, not a latent action. FINDINGS.md F29 -- this is why `action_lag` exists.

## What the latent contains

Foot-contact pattern and body identity decoded from `z` on the same clips, with `z`'s variance
split by what explains it. Eight contact patterns, majority class 0.144; five bodies, chance
0.200. Produced by `scripts/diagnostics/z_content.py`.

| | control ep 20 | cross ep 8 | cross ep 27 |
|---|---|---|---|
| **contact pattern from `z`** | 0.757 | 0.744 | **0.787** |
| body from `z` | 0.707 | 0.638 | 0.665 |
| **variance: gait phase** | 64.5% | **88.7%** | **83.4%** |
| **variance: body** | 8.8% | **1.2%** | **1.2%** |
| variance: interaction | 26.8% | 10.1% | 15.4% |

Behaviour survives; the body's share of the variance falls sevenfold. The small z-gap above is
`z` shedding the body code, not `z` going empty. FINDINGS.md F26.

## The other held-out body, `c06f06t06`

Every segment scaled by 0.6 from `c10f10t10`. The collector scales the IK foot targets by leg
length, so the two bodies are geometrically similar and **their joint commands are identical to
0.07 deg** -- the correct answer is to copy a training body. Both models were evaluated on it
with no retraining, since both had it held out already.

| predictor | RMSE deg | mean R2 |
|---|---|---|
| copy `c10f10t10` -- the correct answer | **0.07** | -- |
| predict this body's own mean | 12.73 | 0.00 |
| control `m3d_bracketed` ep 6 | 13.92 | -2.01 |
| **cross `m3d_cross` ep 8** | **18.82** | **-4.63** |

Both lose to the trivial predictor, and `lambda_cross` is 1.35x worse than the control. Implied
segment scales, where the correct answer in command space is (1.0, 1.0, 1.0):

| | coxa | femur | tibia |
|---|---|---|---|
| control implies | 0.794 | 0.806 | 0.793 |
| cross implies | 0.909 | **0.691** | **0.671** |
| true geometry | 0.60 | 0.60 | 0.60 |

The cross model reads apparent segment size off the image accurately and applies the command
change that *relative* shortening would need; here the shrink was uniform and needed none.
FINDINGS.md F28.

## Physical replay on the held-out body

`c08f09t09`, three clips, open loop, same scene and physics for both passes. FINDINGS.md F27.

| | control ep 6 | cross ep 8 |
|---|---|---|
| mean R2 over 18 joints | 0.832 | **0.868** |
| mean RMSE | 3.40 deg | **2.75 deg** |
| duty-factor error against IK | 0.076 | **0.044** |
| commands outside this body's own range | 7.7% | **5.4%** |
| **worst excursion outside it** | **20.2 deg** | **5.5 deg** |
| forward distance as a fraction of IK | **93%** | 89% |

`lambda_cross` improves the pose, not the distance. The distance was fixed by going from two
training bodies to five: the two-body run covered less than half, both five-body runs cover
84-96 percent.

## Augmentation noise in the reconstruction target

Mean squared distance in embedding space, against a signal of **1.92** (consecutive frames, no
augmentation). Produced by `scripts/diagnostics/aug_noise.py`.

| augmentation | noise | noise / signal |
|---|---|---|
| crop 85-100% + jitter (current) | 8.42 | 4.39 |
| crop 85-100% only | 8.56 | 4.47 |
| crop 95-100% only | 6.76 | 3.53 |
| **jitter only, no crop** | **4.02** | **2.10** |
| crop 95-100% + jitter | 7.02 | 3.66 |

No setting recovers the signal. FINDINGS.md F25.

`z-gap` and `x-gap` are `zero_z` and `zero_x` divided by the held-out error, so they say how much
the decoder needs the latent and the frame. They compare **between runs only**; zeroing an input
is out of distribution.

Baselines on this held-out body, in degrees:

| Predictor | RMSE deg |
|---|---|
| best possible linear mixture of the training bodies | **0.18** |
| `m3d_cross` epoch 8 | **2.91** |
| copy the nearest training body | 3.47 |
| `m3d_bracketed` epoch 6 | 3.57 |
| mean of the five training bodies | 11.48 |
| predict this body's own mean | 13.07 |

## Two-body runs, for the record

| Run | steps | with z | zero z | shuffled z | degrees |
|---|---|---|---|---|---|
| `stage1_6ep_clipped` | 9,750 | 0.166 | 0.848 | 0.975 | 9.6 |
| `stage1_100ep_clipped` | 30,880 | 0.422 | 0.470 | 1.197 | 15.3 |
| `stage1_100ep_clean` | 9,500 | 0.179 | 1.675 | 1.071 | 10.0 |

Read `with z` against 0.444, not 1.0: the standardisation uses the training bodies' statistics,
so predicting their mean costs 1.0 only on them. On held-out `medium` the trivial predictor costs
0.495 and that body's own mean costs 0.444.

## Regenerating anything here

```
.venv/bin/python3 -m wm.predict_actions --ckpt wm/runs/<run>/epoch008.pt --clips 3
.venv/bin/python3 scripts/diagnostics/plot_action_trace.py --pred results/wm/predictions/<name>.npz
.venv/bin/python3 scripts/diagnostics/morphology_axis.py --ckpt wm/runs/<run>/epoch008.pt
.venv/bin/python3 scripts/diagnostics/morphology_mix.py --pred results/wm/predictions/<name>.npz
.venv/bin/python3 scripts/diagnostics/swap_pathway.py --ckpt wm/runs/<run>/epoch008.pt --bodies A B
.venv/bin/python3 wm/sweep_checkpoints.py --run wm/runs/<run>
.venv/bin/python3 sim/render/render_wm_prediction.py --port 23000 --pred results/wm/predictions/<name>.npz --clip 0
.venv/bin/python3 scripts/diagnostics/wm_gait_report.py --replay results/wm/replay/<name>_clip0.npz
.venv/bin/python3 scripts/diagnostics/z_content.py
.venv/bin/python3 scripts/diagnostics/z_dynamics.py --ckpt wm/runs/<run>/epoch008.pt
.venv/bin/python3 scripts/diagnostics/aug_noise.py
```

Add `--encode_device cpu` to any of them when a training run holds the GPU.
`wm/README.md` covers the training log itself.
