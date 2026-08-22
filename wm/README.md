# `wm/` — the world model

Everything that learns from recorded video. Recording lives in `sim/`, measurement in
`scripts/diagnostics/`, and making a robot *act* will live in `control/` when it exists.

**Run convention:** `.venv/bin/python3 -m wm.<name>` from the repository root. Heavy runs go to
fibo7, never the local 2080 Ti.

## The lifecycle, and which file is which stage

The stage boundary runs through the entry points, not through the modules -- `models/`, `config.py`,
`data/` and `losses.py` are shared by both, which is why this package is flat.

| stage | entry point | what it trains | on what data |
|---|---|---|---|
| **pretrain** | `train.py` | ITM, FTM, Motion Decoder, shared body head | every embodiment at once |
| **finetune** | `fit_projector.py` | Action Projector, one per embodiment | the target robot's own actions |

**Why there are two stages at all.** `z_t = ITM(e_t, e_{t+1})` needs the *next* frame, which at
control time is the thing being decided, so **the ITM can never run in the loop** (F81). The
projector is the module that replaces it: `a_t -> z_t`, no future required. Everything measured in
this project so far reads `z` off two ground-truth frames -- reconstruction, not control, as
`predict_actions.py` says in its own docstring.

## Modules

| | |
|---|---|
| `models/itm.py` | `(e_t, e_{t+1}) -> z`. Training and analysis only; cannot run at control time |
| `models/ftm.py` | `(e_t, z) -> e_{t+1}`. The world model proper; what a planner rolls forward |
| `models/motion_decoder.py` | `(e_t, z) -> joint commands`, per-embodiment heads plus a shared body head. An auxiliary loss that stops the latent shortcutting -- **not** a runtime controller |
| `models/action_projector.py` | `a -> z`, per embodiment. What makes control possible |
| `models/adversary.py` | reversed-gradient head on `z`; measured not to reverse the embodiment split (F44) |

## The objective

    L = lambda_recon * L_recon + lambda_motion * L_motion + lambda_body * L_body

`L_recon` and `L_motion` are **per-embodiment** by construction -- 18-D hexapod and 12-D quadruped
commands share no correspondence, so each gets its own head. **`L_body` is the only term that asks
the same `z` to decode the same way on both robots**, which is why matched behaviour data matters to
it specifically. `cfg.body_channels` selects which columns it supervises: `(0,)` forward speed only,
`(0, 2)` forward and yaw. Lateral is column 1 and is excluded -- it fails the embodiment gate, and
half the B1 clips carry a per-policy artefact in it (F79, F80).

## Reading a training log

**Stage 1 and Stage 2 runs print different lines.** Stage 1 trains several hexapod bodies and holds
one out, so it reports `heldout <body>` and an adversary over bodies. Stage 2 trains two
embodiments with nothing held out, so those columns are absent and **`probe` chance is 0.5, not the
0.200 the five-body examples below use**. Everything else reads the same.


One epoch line, annotated:

```
epoch   4 | train 1.9036 (recon 1.7027 motion 0.0326) | val 1.8759 (recon 1.6826 motion 0.0291) | adv 0.070 (x0.80) probe 0.337 | heldout c08f09t09 0.0874 (zero_z 0.4237 zero_x 2.3794)
            └─ bodies it trains on ─┘               └─ unseen episodes, same bodies ─┘   └─ adversary ─┘   └────── a body it has never seen ──────┘
```

### What each number is

| Field | Unit | What it means |
|---|---|---|
| `recon` | MSE on V-JEPA2 embeddings, unnormalised | how well the FTM predicts the next frame's embedding. No absolute meaning; only compare between runs |
| `motion` | MSE on standardised actions | joint-command error. **1.0 means predicting the training-set mean** |
| `train` / `val` | weighted sum | `lambda_recon * recon + lambda_motion * motion`. Dominated by recon, so read the parts, not the total |
| `heldout <body>` | same units as `motion` | the number the whole project is about |
| `zero_z` | same | heldout error with the latent replaced by zeros |
| `zero_x` | same | heldout error with the frame replaced by zeros |
| `adv` | accuracy | the gradient-reversal classifier guessing which body, **chance = 1 / number of training bodies** |
| `(x0.80)` | scale | current reversal strength during warmup |
| `probe` | accuracy | a classifier on a detached `z` guessing which body or embodiment. **Chance is 1/number of classes -- 0.200 with five bodies, 0.5 with two embodiments.** Measures only, never affects training |

Two numbers matter more than anything printed, and you compute them yourself:

```
z-gap = zero_z / heldout      how much the decoder needs the latent
x-gap = zero_x / heldout      how much the decoder needs the frame
```

### Read it in this order

**1. Is it learning at all?** `train recon` and `train motion` should both fall every epoch. If
`recon` rises for two epochs running, something is diverging; stop and look at the learning rate.

**2. Is it overfitting the training bodies?** Compare `train motion` against `val motion`. They
track each other closely in every healthy run here. A widening gap means it is memorising
episodes, which is a different problem from memorising bodies.

**3. Is it transferring?** `heldout`. **Never read a single epoch.** It swings by a factor of two
epoch to epoch in every run measured, and two runs of identical config on different GPUs landed
2.1x apart. Take the mean of five epochs and compare that.

**4. Is the latent doing work?** `z-gap`. Healthy runs on the five-body set sit at 20-39x. Falling
toward 1x means the latent has become useless and the decoder is reading everything off the frame.

**5. Is the frame doing work?** `x-gap`. The five-body control sits at 7-15x.

**6. Is the latent leaking identity?** In a cross-embodiment run this is the tripwire: a `probe`
climbing toward 0.8 in the first epochs means the shared body head is learning *which robot it is
looking at* rather than what the robot is doing. F58 measured exactly that -- 0.824 by epoch 1 with
lateral in the target, against a control at 0.537.

**In adversarial runs only,** `probe` is the one to trust; `adv` fights a reversed gradient and
goes wild. Target is chance, not zero: with five bodies that is **0.200**. Accuracy far below
chance is a failure, not a success -- being wrong 99.8 percent of the time with five classes needs
information, so it means the latent is moving the body code faster than the classifier tracks it.
A smoke run reached 0.002 that way.

### Reference values

Five training bodies, `data/ik_walk_8body`, held-out `c08f09t09`:

| | control (`m3d_bracketed`) |
|---|---|
| `val motion`, epoch 10 | 0.015 |
| `heldout` | 0.08-0.11 |
| `z-gap` | 20-39x |
| `x-gap` | 7-15x |
| probe on frozen `z`, post hoc | 0.724 |
| held-out RMSE in degrees | 3.57 |

Two training bodies, `data/ik_walk_100_framed`, held-out `medium`: `heldout` 0.14-0.36, `z-gap`
3-4x, held-out RMSE 11.04 deg. Different normalisation, so the MSE is **not** comparable across
the two datasets; degrees are.

### Warning signs

| What you see | What it means | What to do |
|---|---|---|
| `recon` rising two epochs running | diverging | stop, lower the learning rate |
| `val motion` improving 10x while `heldout` is flat | the classic failure this project is about | not a bug; it is the finding (F11) |
| `z-gap` falling toward 1x | latent has gone useless | if adversarial, lower `lambda_adv` |
| `x-gap` and `z-gap` both falling | decoder is ignoring both inputs, predicting a constant | check that `motion` is actually falling |
| `probe` below 0.10 with five bodies | adversarial oscillation, not removal | lower `lambda_adv`, lengthen `adv_warmup_epochs` |
| `adv` at 0.000 | expected under reversal | ignore it, read `probe` |
| `heldout` jumps 3x for one epoch then returns | evaluation is a point measurement at an epoch boundary while `train` is an epoch average | ignore single epochs |

### Converting to degrees

MSE in the log is on standardised actions, so it is only comparable within one dataset. For a
number that can be compared anywhere, and against the joint ranges of a real robot:

```
.venv/bin/python3 -m wm.predict_actions --ckpt wm/runs/<run>/epoch010.pt --clips 3
.venv/bin/python3 scripts/diagnostics/plot_action_trace.py --pred results/wm/predictions/<name>.npz
```

Add `--encode_device cpu` when the GPU is busy with a training run.

### Going deeper than the log


| Question | Script |
|---|---|
| which joints failed | `scripts/diagnostics/plot_action_trace.py` |
| does it read the body from the frame or the latent | `scripts/diagnostics/swap_pathway.py` |
| is it copying one training body | `scripts/diagnostics/morphology_mix.py` |
| where does each stage place the held-out body | `scripts/diagnostics/morphology_axis.py` |
| how does transfer change with training compute | `wm/sweep_checkpoints.py` |
| does the command actually walk | `sim/render/render_wm_prediction.py` then `scripts/diagnostics/wm_gait_report.py` |

### Live view

```
tensorboard --logdir wm/runs
```

Everything printed is also written there, plus `heldout/motion_zero_x` and the per-part losses.
