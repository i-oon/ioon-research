# Stage 1 retrain and re-measure

> **Role**: the run sheet for rebuilding the five lost Stage 1 checkpoints and re-deriving every
> number in `report/update_slide.md` that depends on them. Delete this file once slides 4-12 are
> re-measured; the conclusions belong in `FINDINGS.md`, not here.

## Why this is a retrain and not a restore

The checkpoints were lost with `wm/runs`. That is the occasion, not the reason. The reason is that
the runs were not controlled: `ik_walk_8body` contained bodies that veer 0.36-0.43 m off course
and bodies that collapse and rotate in place, and they entered training because the walk check
used unsigned displacement (`FINDINGS.md` F42). Every m3d number was measured on a training set
that was 40% a robot failing to walk.

So the sequence of experiments in `update_slide.md` stays exactly as it is. What changes is that
each one now runs on data where every clip walks, which is what those experiments were always
supposed to have been.

## What the runs train on

Built by `scripts/build_stage1_dirs.py`, which links only clips where the head travelled at least
0.30 m forward (signed) and drifted less than 0.20 m sideways.

| directory | runs | bodies | train clips | held out |
|---|---|---|---|---|
| `data/ik_walk_m3d_clean` | `m3d_cross`, `m3d_bracketed` | 4 | 111 | `c08f09t09`, 29 clips |
| `data/ik_walk_cov_narrow` | `tib_cross`, `tib_ctrl` | 4 x 24 | 96 | `c10f10t08`, 20 clips |
| `data/ik_walk_cov_wide` | `bracket_cross` | 6 x 16 | 96 | `c10f10t08`, the same 20 clips |

Zero failing clips in any of them.

## Running

**com7 trains and does nothing else.** Its only job is to produce the five checkpoints. It does not
edit code, does not fix a script that misbehaves, does not update a document, does not run a
measurement, and does not commit. If a run fails, stop and report the log — the fix is made here
and pushed, not patched there. Every analysis in the rest of this file happens on the local machine
after the checkpoints come back.

The reason is the one that made this retrain necessary: an experiment is only controlled if the
code that produced it is the code in the repository. A change made on the training machine is a
change nothing here can see.

On com7, from the repository root:

```
.venv/bin/python3 scripts/build_stage1_dirs.py
```

```
bash scripts/retrain_stage1.sh
```

Sequential and in slide order: `m3d_cross`, `m3d_bracketed`, `tib_cross`, `tib_ctrl`,
`bracket_cross`. Roughly 4 h + 4 h + 50 min x 3. A finished run leaves a `COMPLETE` marker and is
skipped on a rerun; a run that died part-way refuses to be skipped, because `best.pt` is rewritten
at every improvement and a half-trained one would otherwise be measured as final.

Then hand back two files per run and nothing else — `best.pt` and the `config.yaml` the run wrote
at startup, which records the exact configuration it trained under:

```
scp com7:~/ioon-research/wm/runs/<run>/{best.pt,config.yaml} wm/runs/<run>/
```

Work resumes here from those.

---

## Two changes that alter what the slides can claim

### 1. m3d no longer tests femur/tibia decoupling

Inside `ik_walk_8body`, every body with femur != tibia is one of the broken ones. So the clean m3d
training set is entirely femur == tibia, and so is the held-out `c08f09t09`.

`c08f09t09` is still an **exact** non-negative mixture of the four training bodies — distance
0.000, weights 0.25 `c10f10t10`, 0.50 `c06f10t10`, 0.25 `c10f06t06` — drawing on three bodies
rather than lying on any pairwise line, so **it remains a composition test, not an interpolation**.
What it no longer probes is femur/tibia independence. That claim now rests solely on `tib_*` and
`bracket_*`.

### 2. Slide 10 loses three of its four rows

The table that shows the probe predicting transfer before training used `c06f06t06`, `c10f10t06`
and `c06f10t06` as its failure cases. `c06f06t06` is now a **training** body — it was the
replacement for the two that veer — and the other two do not walk, so they cannot be test bodies at
all.

Replacements come from `ik_walk_decoupled`: `c10f10t08`, `c10f09t07`, `c10f08t06`. All three walk,
and all three sit outside the m3d training hull, since every m3d body has femur == tibia:

| body | distance to the best mixture of the training bodies |
|---|---|
| `c08f09t09` — held out, succeeds | **0.000** |
| `c10f10t08` | 0.141 |
| `c10f09t07` | 0.141 |
| `c10f08t06` | 0.141 |

Cleaner than the version it replaces, where the three failure cases sat at 0.283 and the split was
0.000 against 0.283. Here all three are the same distance out, so the probe error and the model
error can be read against a single boundary rather than a scatter. Score them with
`score_body.py --data_dir data/ik_walk_decoupled` against the `m3d_cross` checkpoint — no
retraining, because changing a *test* body does not change the weights.

---

## Re-measure, in slide order

Each row is: what the slide claims, which checkpoint, which script. Nothing here needs a new
training run.

### Slides 4, 5, 6, 7, 10, 11, 12 — `m3d_cross` vs `m3d_bracketed`

| slide | claim | script | what changes |
|---|---|---|---|
| 3 | dataset and control tables | — | body counts 5 -> 4, `ik_walk_8body` -> `ik_walk_m3d_clean` |
| 4 | probe recovers held-out geometry | `plot_morphology_evidence.py` | refit on 4 bodies; the per-body table loses two rows and gains `c06f06t06` |
| 4 | decoder ignores the frame | `swap_pathway.py`, `morphology_mix.py` | rerun both |
| 5 | cross-body loss beats copy-nearest | `score_body.py` | 3.57 / 2.91 deg both re-measured |
| 6 | latent purification | `z_content.py`, `z_body_share.py` | the 88.7% / 1.2% split |
| 6 | held-out column | `z_body_share.py` | **needs a decision**: the decomposition needs two held-out bodies and m3d now holds out one. Pair `c08f09t09` with a `ik_walk_decoupled` body, and the slide's existing caveat about mixing an in-range and out-of-range body stands unchanged |
| 7 | the commands walk | `wm_gait_report.py` | distance, heading, out-of-range fraction, duty factor, and the replay video |
| 10 | probe predicts transfer | `score_body.py` + `plot_morphology_evidence.py` | rebuilt on the three `ik_walk_decoupled` bodies, per above |
| 11 | one frame determines the command | `z_dynamics.py` | the substitution table |
| 12 | forward model rolls forward | `latent_rollout.py` | the four horizons against both baselines |

`action_lag` is now 1 for these runs, where the originals ran at the legacy 0. Slide 11's own table
already reports both columns, so it is the one slide that gains rather than loses from this.

### Slide 8 — `tib_cross` vs `tib_ctrl`

Held-out body changes from `c10f10t06` to `c10f10t08`.

The slide's headline row — 27.76 deg, R² −3.16 — **was the broken body** and disappears. What
remains is the table the slide already called the honest version: `c10f10t08`, `c10f09t07`,
`c10f08t06` at 11-13 deg and R² −0.4 to −1.1. Neither of the last two is in `cov_narrow`, so both
stay valid extra test bodies for this checkpoint.

`score_body.py`, `plot_action_trace.py`, `morphology_mix.py`. `tib_ctrl` exists for the first time,
so slide 3's control table stops naming a run that was never trained.

### Slide 9 — `tib_cross` vs `bracket_cross`

Both now hold out `c10f10t08` on the same 20 clips, volume-matched at 96 training clips each rather
than the accidental 7,540 / 7,735 pair counts. Every number in the five-row table is re-measured.

One row shrinks: `c10f09t07` and `c10f08t06` are in `bracket_cross`'s training set, so they cannot
serve as extra test bodies for it the way they can for `tib_cross`.

The no-GPU half of the slide — refit the encoder probe on the wider body set and watch the
femur/tibia gap open — reruns unchanged with `plot_morphology_evidence.py`.

### Slides 14-16 — Stage 2

Untouched. Stage 2 trains from `data/ik_walk_stage2_*` and was already rebuilt on clean data. The
one Stage 1 number quoted there is "Stage 1 scores 2.91 deg on this same body", which moves once
`m3d_cross` finishes.

---

## Order of work

1. `build_stage1_dirs.py`, then `retrain_stage1.sh` on com7.
2. `score_body.py` on all five checkpoints first — it is fast and it says immediately whether the
   clean data reproduces the shape of the old result or changes it.
3. Slides 4-7 from `m3d_cross`, then 8 from `tib_cross`, then 9 from the pair.
4. 10, 11, 12 last; they depend on the same checkpoint as 4-7 and none of them gate the others.
5. Fold the confirmed numbers into `FINDINGS.md`, then rewrite `update_slide.md` in place.
