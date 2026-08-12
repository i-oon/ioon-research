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
other out, so a hexapod body was withheld instead. `scripts/score_body.py`, same weights, one
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

## Not done

- clean UMAP (killed to free cores; quick to redo)
- per-leg contact probe -- `LogisticRegression(max_iter=3000)` on 5,632 features will not
  converge; needs a different solver or fewer features before it is rerun
- `stage2_clean_centered` -- the centring experiment, still the untried lever
