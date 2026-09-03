# Model & Training Config

Quick reference. Two columns throughout where they differ: the **default** in `wm/config.py`, and
what the runs **actually use** — several defaults are 0.0 that no real run leaves at 0.0.
Per-run values live in `wm/runs/<name>/config.yaml` and inside each checkpoint
(`from_checkpoint`, `wm/config.py:19`).

Reference run for the "in use" column: `wm/runs/beh12_ego/teacher_ego.pt` (current teacher,
egocentric, hexapod + B1).

---

## 1. Architecture

| Component | What it does | Shape / dims | Params |
|---|---|---|---|
| **Encoder** (frozen) | frame → patch tokens | `256×256×3` → `256 × 1408` | 1B, never updated |
| **ITM** | `(e_t, e_{t+1})` → `z_t` | 2 self + 2 cross blocks, hidden 512, 16 heads → `z ∈ ℝ⁶⁴` | 13,368,384 |
| **FTM** | `(e_t, z_t)` → `ê_{t+1}` | 8 blocks, hidden 512, 16 heads → `256 × 1408` | 77,143,424 |
| **Motion decoder** | `(e_t, z_t)` → `â_t` | cross-attn backbone, 16×16 grid pooled ×2 | 5,499,934 total |
| ├ shared backbone | reads behaviour from `z` vs frame | hidden 512 | 4,957,184 |
| └ per-embodiment head | latent → this body's joints | linear, 18-D / 12-D | 272,914 / 269,836 |
| **Body head** | `z` → body motion, **frame-blind**, one head all embodiments | `LN → 64→128 → body_dim` | 8,577 |
| **Action projector** | `a` → `z` (control time; ITM needs a future frame) | per-embodiment MLP, depth 2, width 512 | 305,216 / 302,144 |
| **LDAD** *(optional)* | action from `FTM(e_t,z) − e_t` | 3 transformer layers, 8 heads | off by default |
| **Adversary** *(optional)* | GRL body classifier on `z` | hidden 128 | off by default |

Encoder: `facebook/vjepa2-vitg-fpc64-256` (`scripts/vjepa2_encoder.py:10`). Each frame encoded
independently via the 2-frame tubelet, so `e_t` carries no future.

---

## 2. Pretraining objective

**As actually trained** (current teacher `wm/runs/beh12_ego/teacher_ego.pt`, and every
`lambda_body` run before it):

```
L = 1.0·L_recon + 1.0·L_motion + 0.5·L_body
```

The config *default* for `lambda_body` is 0.0, but **the runs that matter set it to 0.5** — 0.0 is
the control arm (`s2_fwd_hex8-b1_ctrl`), not normal practice.

| Term | Scores | λ field | Default | **In use** | Defined at |
|---|---|---|---|---|---|
| `recon` | `MSE(ê_{t+1}, e_{t+1})`, scored on view 2 | `lambda_recon` | 1.0 | **1.0** | `wm/losses.py:15,22` · `wm/config.py:177` |
| `motion` | `MSE(â, a)` through per-embodiment head | `lambda_motion` | 1.0 | **1.0** | `wm/losses.py:21,22` · `wm/config.py:178` |
| `body` | shared head `z` → body motion (both robots) | `lambda_body` | 0.0 | **0.5** | `wm/losses.py:39` · `wm/config.py:242` |
| `cross` | body A's `z` on body B's frame → B's command | `lambda_cross` | 0.0 | 0.0 (0.5 in `*_cross` Stage-1 runs) | `wm/losses.py:32` · `wm/config.py:185` |
| `adv` | GRL body classifier (loss ↑ = working) | `lambda_adv` | 0.0 | off | `wm/losses.py:44` · `wm/config.py:235` |
| `hinge` | real vs null rollout separation over K steps | `lambda_hinge` | 0.0 | off | `wm/train.py:172,182,215` · `wm/config.py:249` |
| `readout` | frozen readout recovers `a` from prediction | `lambda_readout` | 0.0 | off | `wm/train.py:183,220` · `wm/config.py:250` |
| `ldad` | action decoded from predicted state difference | `lambda_ldad` | 0.0 | off | `wm/train.py:194,224` · `wm/config.py:244` |

**`L_body` is the term that carries the cross-embodiment claim** — one head, shared by both robots,
`z` → body motion, blind to the frame. `L_motion` cannot do this: it supervises through
per-embodiment heads onto 18-D and 12-D joint commands that have no correspondence.

**Null action**: `z_null = ITM(e_t, e_t)` — the latent of "nothing happened" (`wm/train.py:172`).

**Checkpoint selection** uses `selection = λ_recon·recon + λ_motion·motion` **only** — the two terms
every run has, so matched arms are selected on the same quantity (`wm/losses.py:63` → `best.pt`,
`wm/train.py:624`). `total` includes whichever experimental terms a run enabled and is not used.

---

## 3. Hyperparameters

**Optimisation** — `wm/config.py:272-281`, optimiser at `wm/train.py:464`

| Field | Default | | Field | Default |
|---|---|---|---|---|
| `batch_size` | 8 | | `grad_clip` | 1.0 |
| `lr` | 1e-4 | | `seed` | 0 |
| `weight_decay` | 0.01 (AdamW) | | `num_workers` | 4 |
| `epochs` | 50 | | schedule | cosine, `T_max=epochs` |
| precision | fp16 autocast (`wm/train.py:239`) | | `device` | cuda |

**Dimensions** — `wm/config.py:118-131`

| Field | Default | | Field | Default |
|---|---|---|---|---|
| `token_dim` | 1408 | | `z_dim` | 64 |
| `grid` | 16 | | `hidden` | 512 |
| `action_dim` | 18 | | `heads` | 16 |
| `mlp_ratio` | 4.0 | | `dropout` | 0.0 |
| `itm_self_blocks` | 2 | | `ftm_blocks` | 8 |
| `itm_cross_blocks` | 2 | | `z_tokens` | 1 |

**Targets & time** — `wm/config.py:206-223`

| Field | Default | Meaning |
|---|---|---|
| `action_lag` | **1** | ask for the action that *caused* the transition; at 0 the target is already in `e_t` |
| `frame_stride` | 1 | spacing of the ITM's two frames (1 = 50 ms at 20 Hz) |
| `action_chunk` | 0 | 0 = follow `frame_stride`; keeps both halves of the objective on the same interval |

**Shared body target** — `wm/config.py:254-264`

| Field | Default | In use | Meaning |
|---|---|---|---|
| `lambda_body` | 0.0 | **0.5** | weight of the shared-head term |
| `body_dim` | 1 | **3** | must equal `len(body_channels)` |
| `body_channels` | `(0,)` | **`(0,1,2)`** | 0 = forward, 1 = lateral, 2 = yaw |
| `body_hidden` | 128 | 128 | |
| `body_sees_frame` | False | False | True reproduces the negative result where the head identifies the robot |

Per-run history: `body_dim 1` / `(0,)` = forward only (`beh12_hexonly`, `s1_fwd_m3d_body0.5`,
`s2_fwd_hex7-b1_body0.5`); `body_dim 3` / `(0,1,2)` = all three channels (`beh12_hex-b1_body3`,
`beh12_ego`).

**Data & routing**

| Field | Default | Line |
|---|---|---|
| `within_body_std` | True | `wm/config.py:82` |
| `cross_augment` | True | `wm/config.py:230` |
| `balance_embodiments` | True | `wm/config.py:116` |
| `val_fraction` / `heldout_pairs` | 0.1 / 400 | `wm/config.py:88,70` |
| `md_head` / `md_pool` | "mlp" / 2 | `wm/config.py:174,175` |
| `center_embeddings` / `center_frames` | False / 300 | `wm/config.py:165,168` |
| `ftm_embodiment_channel` | False | `wm/config.py:144` |

**Experimental term settings** — `wm/config.py:248-253`

| Field | Default | | Field | Default |
|---|---|---|---|---|
| `hinge_margin` | 0.1 | | `ldad_layers` | 3 |
| `hinge_K` | 3 | | `readout_hidden` | 512 |
| `adv_hidden` | 128 | | `adv_warmup_epochs` | 5 |

---

## 4. Where each piece lives

| Thing | Path |
|---|---|
| Objective (core terms) | `wm/losses.py:11-72` (`compute_losses`) |
| Objective (hinge / readout / LDAD) | `wm/train.py:163-225` |
| Checkpoint selection criterion | `wm/losses.py:63` → `wm/train.py:624` |
| All hyperparameters | `wm/config.py:61-281` (`Config`) |
| Rebuild config from a checkpoint | `wm/config.py:19-31`, `LEGACY_DEFAULTS` at `:35-58` |
| Chunk rule | `wm/config.py:9-16` (`chunk_of`) |
| Models | `wm/models/{itm,ftm,motion_decoder,action_projector,ldad,adversary,blocks}.py` |
| Optimiser & schedule | `wm/train.py:464-465` |
| Encoder wrapper | `scripts/vjepa2_encoder.py` |
