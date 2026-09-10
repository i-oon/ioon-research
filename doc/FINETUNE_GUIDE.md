# FINETUNE_GUIDE

How to ground a body absent from pretrain into the shared coordinate. This is the manual for
`wm/finetune_new_body.py` and the reasoning behind it -- read this before running anything with
"fine-tune" in the name.

Related documents:

| Document | Contents |
|---|---|
| [FINDINGS.md](FINDINGS.md) | F131 (why stage 4 exists), F200/F200a (the wrong mechanism this guide replaces, and why) |
| [SIM_GUIDE.md](SIM_GUIDE.md) | Setup, data collection |

---

## 1. The mistake this guide exists to prevent

`wm.train --init_ckpt` looks like a fine-tuning tool -- it warm-starts from an existing checkpoint
and keeps training. **It is the wrong tool for grounding a new body.** It jointly retrains
ITM + FTM + the full MotionDecoder (every action head) + the shared `body_head` + the adversarial
probe, all together, under the full multi-task pretrain loss (`L_recon` + `L_motion` +
`lambda_body * L_body` + probe). That is not adaptation, it is closer to "resume pretraining with a
new source mixed in" -- and this project already tried it (F200, F200a) and it made B1's transfer
correlation *worse*, not better, with no clean explanation why.

**This project already has, and had previously validated, a much narrower staged procedure**,
taken from LAC-WM and extended once for a reason specific to this project's control mechanism
(F130/F131). It achieved real, above-chance results the `wm.train`-based approach never has:
B1 calibrated to +0.79 correlation (F131), cross-embodiment selection cleared chance at 35-38%
(F132). Use it.

---

## 2. The four stages

```
stage 1  wm.adapt            fine-tune ONLY the ITM and FTM on the new body's own clips
stage 2  (inside             fit the action projector (a -> z) against the NOW-ADAPTED itm's z
         finetune_new_body)
stage 3  wm.adapt3           OPTIONAL: jointly fine-tune projector + FTM together
stage 4  wm.fit_body_head    refit ONLY the shared body_head against the latent it will
                             actually be shown at control time
```

Run all of it with one command:

```bash
.venv/bin/python3 -m wm.finetune_new_body \
    --base_ckpt wm/runs/beh12_hexonly_stopgrad/best.pt \
    --embodiment b1 --data data/egocentric/beh12_b1_ego_flat \
    --out_dir wm/runs/b1_adapt
```

### Stage 1 -- `wm.adapt`

Fine-tunes ITM and FTM only. **Encoder, decoder, and `body_head` all stay frozen.** "The frozen
model is worse than assuming the frame does not move" is the problem this solves -- the forward
model has never seen this body's dynamics or appearance, and needs some exposure before anything
downstream can be asked to consume its output.

> **Both halves of that premise are wrong as stated, measured (F189). Do not gate on this stage.**
> `rollout()` returns `hold_error / model_error`, so **higher is better**, and the frozen model
> already BEATS holding the frame still on a body it has never seen -- 1.55x on B1, 1.57x on gecko.
> Worse, the direction of stage 1's change does not predict what happens downstream: B1's stage 1
> made the forward model measurably worse at every horizon (h=1 raw error 3.627 -> 3.698) and B1
> succeeded at stage 4; gecko's improved (3.994 -> 3.878) and gecko failed. **Run stage 1, but read
> its ratio as diagnostic colour, never as a gate.**

### Stage 2 -- fit the action projector

`a -> z` has to be refit after stage 1, because stage 1 moved what `z` means for this body.
Refitting against the stale, pre-adaptation `z` would be fitting the wrong target. "The inverse
model cannot run in the loop, so something must turn an action into z" is why a projector exists
at all -- see `wm/models/action_projector.py`'s own docstring.

`wm/fit_projector.py` is hardcoded to exactly two embodiment names (`hexapod`, `b1`) and cannot fit
a third. `wm/finetune_new_body.py` carries its own generic, single-embodiment version of the same
fitting loop, so this works for gecko, or any future body, without editing anything.

### Stage 3 -- `wm.adapt3` (optional, off by default)

Only needed if stage 2's plain MSE regression fails. `wm/adapt3.py`'s own docstring traces the
failure to a specific cause (F97): if the same action recurs in different states and is followed
by different transitions -- true of B1's PPO-policy actions, a response to state -- then `a -> z`
is one-to-many, and no amount of stage-2 data fixes a target like that. "Freezing the forward model
and making the projector chase z exactly is not enough -- the forward model has to move to meet it"
is stage 3's whole point: it changes the loss from `MSE(proj(a), z_ITM)` to
`MSE(FDM(e_t, proj(a)), e_t+1)`, a real function instead of a one-to-many target.

**Open-loop CPG/babble actions (gecko, or any body whose babble is a scripted gait rather than a
trained policy) likely do not have this problem** -- the same 16 numbers mean the same thing
regardless of state, so `a -> z` should already be a function. Turn stage 3 on
(`--stage3`) only if stage 2's reported rollout-gap ratio looks bad (near or above 1.0).

### Stage 4 -- `wm.fit_body_head`

Refits only the shared body-motion head (`z_dim -> body_hidden -> body_dim`, ~8k parameters,
everything else frozen) against whichever latent it will actually be shown at control time -- the
projector's `z`, not the ITM's ground-truth `z`, which is only ever available with the future frame
already in hand. This step is not in vanilla LAC-WM; it was added here because F128 put
`body_head(proj(a))` on the actual control path, so a head fit once on the original pretrain bodies
and never revisited is exactly the stale-target mismatch stage 2 already exists to avoid, one level
further up the pipeline. `wm/fit_body_head.py`'s own docstring names three outcomes in advance
(F130) -- worth reading before judging a result from this stage.

---

## 3. What data to use

**For a genuinely held-out body (never in ANY pretrain), `--data` must be BABBLE clips, not a
scripted/expert demonstration set.** A real deployment on an unseen body has motor babbling and
nothing else -- that is the whole premise of claim (3) (see `doc/FINDINGS.md`'s F199). B1's own
adaptation history in this project (F97, F130-F136) used its PPO-policy clips because that is what
existed for B1 at the time it was first tested this way; a body with no policy at all (gecko, or
any future third body) uses babble at every stage here, the same way `wm/fit_gecko_projector.py`
already does for the projector alone.

---

## 4. If comparing to someone else's cross-embodiment method

If another paper's method claims a new body's video is enough on its own, with no adaptation of
its dynamics model and no adaptation of its inverse map, ask what stands in for stages 1-2 here.
LAC-WM's own ablations, and this project's own F97, found `a -> z` is not always simply invertible
for a new body -- it may be that a different architecture sidesteps the need (e.g. one that never
encodes a per-embodiment inverse map at all), but "we didn't adapt anything and it still worked" is
worth checking for a hidden equivalent of stage 2/3/4 before taking it at face value.
