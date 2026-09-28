# Architecture comparison: Egocentric VSM vs. LAC-WM vs. Ours

Written 2026-09-24, mid-investigation into why `z` fails to separate behavior conditions
(the whole-session finding: 83% of `z`'s raw variance is gait phase, 17% is condition —
see `doc/FINDINGS.md` and this session's transcript for the measurement). This doc exists
to pin down, with verified facts (not guesses), where our architecture matches, deliberately
diverges from, or accidentally diverges from the two reference systems, and to give one
coherent answer for what's actually wrong.

Sources: `doc/ref/notes_egocentric_vsm.md`, `doc/ref/notes_lac_wm.md` (both cross-checked
against the papers' actual PDFs and, for Egocentric VSM, its released code), and this
project's own `wm/config.py`, `wm/models/*.py`, `wm/runs/*/config.yaml`.

## Full comparison table

| | Egocentric VSM | LAC-WM | Ours |
|---|---|---|---|
| Visual encoder | ResNet-50, trained end-to-end | V-JEPA2 1B, **frozen** | V-JEPA2, **frozen** |
| z-generation | none — action always given directly | `IDM(x_t,x_{t+1})` → z, inferred | `ITM(e_t,e_{t+1})` → z, inferred |
| z_dim | n/a (256-d fused feature, no shared latent concept) | 64 | 64 |
| Forward model | single MLP fusion, 512→128→32→6 | FDM, 8 attn blocks | FTM, 8 attn blocks |
| hidden dim / heads | n/a | 512 / 16 | 512 / 16 |
| IDM/ITM blocks | n/a | 4 total | 4 total (2 self + 2 cross) |
| Auxiliary grounding loss | n/a (only loss is the state-prediction loss) | Motion Decoder, **z split in half**: end-effector / camera | MotionDecoder, **whole z used**, no split |
| Explicit separation/contrastive loss | none | **none** | **hinge/margin loss (ours only)** |
| Cross-augmentation (two independent views, IDM sees one, FDM predicts the other) | n/a | yes | yes — confirmed active in every checkpoint this session diagnosed |
| Embodiment-identity handling | n/a (single robot) | not discussed in paper | 2 mechanisms built (`ftm_embodiment_channel`, `center_embeddings`) — **neither ever enabled in any run** |
| Pretraining diversity | 1 robot, 4 ground textures | **3 embodiments, 3 visually unrelated domains** (human hand / bimanual humanoid / robot arm), 150k trajectories | **1 embodiment**, 1 visual domain, ~24-96 clips |
| Batch size | small model, not critical | 512 | 8 |
| Adaptation recipe | n/a | 3-stage, **LoRA rank-2** | 2-of-3-stage built (stage 3 "still not built" per `wm/adapt.py` docstring, though a one-off stage-3 was run for gecko), **full finetune**, no LoRA |
| Action at train time | direct MLP input (24-d: prev+next joint angles) | used only as Motion Decoder's loss *target*, not IDM input | used only as MotionDecoder's loss *target*, not ITM input |
| Real temporal-history mechanism | on the **action** side (12 prev + 12 next joint angles), NOT the claimed LSTM (confirmed via code: LSTM runs on 5 copies of one already-fused frame, not real per-frame history) | n/a within one step; FDM adds true temporal attention (horizon 8) only in the last 20k/80k pretrain iterations | none currently — FTM is single-step, stateless (`FTM(e_t, z_t)`, no history). Two untested hypotheses in flight this session: PhaseHead reading `(e_{t-1}, e_t)`, and FTM reading `(z_{t-1}, z_t)` |

## What's actually wrong — the one row that explains the others

**Pretraining diversity.** LAC-WM's "unified latent space" claim was proven across domains
so visually different (a human hand, a bimanual humanoid, a Franka arm) that literally
nothing survives shared across all three *except* genuine motion semantics — that's why
their `z` ends up motion-only: there's no common visual junk for it to cheat with. Every
clip we train on is the same body doing the same gait, just modulated. Gait phase isn't
noise contaminating our data — it's the single most common, lowest-cost, always-true
feature available to the loss, present in 100% of training examples regardless of
condition. Nothing in our setup forces `z` away from it, because nothing about our data
makes it a shortcut worth avoiding — it's real, shared, exploitable signal every time.

This explains every other divergence in the table, not just adds to the list:

- **Why cross-augmentation (already on) didn't save us**: cross-aug kills shortcuts tied to
  *specific pixels/exact-frame-matching*. Gait phase survives independent augmentation
  fine — it's a real property of both augmented views equally, not a rendering artifact.
- **Why margin/hinge loss backfired**: we're the only one of the three with this mechanism
  at all — built because our data doesn't naturally force separation the way LAC-WM's
  cross-domain diversity does. Forcing separation on a distribution still 83%-dominated by
  one shared factor pushes against the data's own geometry — consistent with what was
  measured: raw ceiling ratio improved (0.983→0.276) but held-out probe R² got *worse*
  (0.486→0.415).
- **Why LAC-WM never needed the kind of z-split we're now planning**: their half/half split
  serves two motion targets that already lived on separate physical channels (hand vs.
  camera) — a structural label, known in advance. Ours needs a split for a different
  reason: to carve a shared, always-present *nuisance* factor out of a low-diversity
  dataset — an emergent pattern discovered via variance decomposition, not a given label.
  Same mechanism (partition `z`'s capacity by function), harder problem.
- **Why beh24 (more conditions, training in progress as of this writing) is the most
  architecturally honest fix attempted so far**: it's the only thing tried this session
  that adds pretraining diversity rather than adding a new loss term on top of the same
  narrow data — a smaller, cheaper step in the same direction LAC-WM took to the extreme
  (3 domains). More conditions within one body, not more domains, but the same lever.

**The architecture itself is not wrong.** It's a verified, deliberate, near-exact clone of
LAC-WM's validated design (same block counts, same hidden dim, same heads, same z_dim —
confirmed via `wm/models/ftm.py`'s own docstring and every checkpoint's `config.yaml`).
What's wrong is running that design on a data regime (one embodiment, one domain, dozens
of clips) it was never proven on, and responding to the resulting failure with bigger loss
terms (reweight, margin loss) instead of more real diversity — backwards, given LAC-WM's
own evidence that diversity, not loss engineering, is what makes a unified latent space
actually unify.

## Open items this doc doesn't resolve yet

- `ftm_embodiment_channel` / `center_embeddings`: built, documented, never tested. Cheap to
  test (flip a boolean, retrain) — queued behind the current beh24 checks.
- PhaseHead validity: unverified whether phase is actually linearly readable from
  `(e_{t-1}, e_t)` — flagged "UNVALIDATED" in `report/pipeline_proposed_zsplit.tex`.
- FTM action-history lever (`z_{t-1}, z_t}` instead of frame-history): test built
  (`scripts/diagnostics/objective_experiments/ftm_action_history_lever_check.py`), not yet
  completed — interrupted by a machine reboot, queued to rerun once the current beh24
  training finishes and GPU is free.
- LoRA-based adaptation: not yet tried anywhere in this codebase; plausible for reducing
  overfitting risk on small adaptation sets (e.g. gecko's 36 babble clips), but the actual
  measured gecko gap (F189) traces mostly to a coordinate-frame bug and a too-narrow
  projector input window, both already fixed/understood — LoRA would be attacking whatever
  residual gap is left after those, not a confirmed live problem.

## Resolved: FTM rollout's usable horizon, and why it doesn't block anything

Direct-vs-rollout candidate-scoring test on the held-out leg-length body (c08f09t09,
`turn_s0.29`, beh24 checkpoint + a freshly-fit hexapod-only projector): rollout tracks the
goal competitively with `direct` early in a trajectory (0.0576 vs 0.0636 mean error,
`t<35`), then degrades (0.0765, `t>=35`, a 1.33x rise) while `direct` does not (0.90x,
actually slightly better late). Confirmed this is genuine closed-loop error accumulation,
not the variable-clip-length clamp fix in `true_local_froude` (the split point, t=35, sits
well before the shortest candidate clip (57 frames) could ever trigger it).

**This is not a live blocker.** Neither reference system rolls out anywhere near 35 steps
before re-grounding in a real observation: Egocentric VSM re-grounds every single step
(effectively a 1-step imagined lookahead, replanned at ~5Hz, real camera frame every
cycle — this is exactly why it never faces this problem at all); LAC-WM re-grounds every
6-20 steps depending on the setup (Table 2: 20-step rollout per window; Table 5: 6-frame
rollout per subgoal chunk; FDM's own pretraining temporal horizon caps at 8). Both are
comfortably inside the region where our own rollout still matches or beats `direct`.

**Design principle to carry forward, not an open question**: the planner must re-ground in
a real observation on a cadence well under ~35 steps — 10-20 steps, matching what both
references actually do — rather than trying to extend rollout accuracy indefinitely. This
was a real design decision that needed measuring, not something to assume from either
paper's numbers alone, since our own architecture/data regime could plausibly have
degraded on a different schedule.

Scripts: `scripts/figures/plot_direct_vs_rollout_c08f09t09.py` (adapted from the B1
version), `scripts/diagnostics/objective_experiments/zero_shot_rollout_c08f09t09.py`
(the earlier, coarser MSE-vs-hold-still check, 1.75-2.01x across horizons 1-10, still
useful as a sanity check but not the metric that settled the horizon question).
Checkpoint: `wm/runs/beh24_hinge_cleansplit/full_c08f09t09.pt` (best.pt + a freshly-fit
hexapod-only projector merged in). Bug fixed along the way:
`final_2x2x2_test.py`'s `true_local_froude` returned NaN for a candidate shorter than the
caller's offset (variable clip lengths, 57-66 frames) — now clamps to the candidate's own
last valid index instead of slicing past the end.
