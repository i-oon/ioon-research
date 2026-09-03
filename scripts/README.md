# scripts

Every file here has a docstring stating the question it answers and, where it matters, the
measurement trap it exists to avoid. Read that before running one — several of these exist because
an earlier, more obvious version of the same measurement gave a confidently wrong answer.

Run everything from the repository root with `.venv/bin/python3`, never bare `python3`.
Add `--encode_device cpu` to anything that encodes frames when a training run holds the GPU.

```
scripts/
  vjepa2_encoder.py   the frozen encoder wrapper, imported by almost everything
  run/                run sheets: retrain_stage1, com7_train, retrain_stage2_clean_adv, f183_ldad, ...
  diagnostics/        what the model learned, and whether it works — grouped into subfolders below
  tools/              CoppeliaSim scene/robot calibration utilities, not research-question diagnostics
  dataset/            build, audit and inspect the recorded data
  figures/            figures for the report that are not measurements
  finished/           answered questions, kept because the findings cite them
  amp/                the parked reinforcement-learning branch
  _archive/           umap_domain_check.py only — still cited by direction_plan.md §6
                       (render-style-dominance risk); everything else superseded was removed
```

`diagnostics/` is a Python package (`diagnostics/__init__.py`) so that scripts can import from each
other (e.g. `from diagnostics.planning.score_closed_loop import channel_for`); each subfolder below
carries its own empty `__init__.py` for the same reason. `tools/` is not a package — nothing imports
from it, it is invoked by file path only.

## Do not redefine these

`wm/bodies.py` is the single source of truth for which bodies and clips are valid, and for what
counts as contact. It depends on nothing heavier than numpy, so `sim/`, `wm/` and `scripts/` all
import from it.

| | |
|---|---|
| `CONTACT_THRESHOLD` | 0.27 N. Was defined in four places. |
| `EXCLUDED_BODIES` | the four bodies that collapse or veer, with the reason for each |
| `walk_check`, `walks` | signed forward travel and lateral drift, separately |
| `bodies_in(data_dir)` | the bodies present in a directory, non-walking ones dropped |
| `training_bodies(cfg)` | the bodies a checkpoint actually trained on, read off its own config |
| `evenly`, `usable_clips`, `contact_labels` | |

**Never write a body list into a script.** Nine did, and four of those lists still held bodies that
do not walk long after that was known — including `z_content.py`, which produced the variance split
quoted on slide 6. A literal list is correct only until the next run changes its split, and nothing
warns you when it stops being. `scripts/dataset/compare_ratio_gaits.py` is the one deliberate
exception, and says so in the file: the broken bodies are its subject.

---

## diagnostics/

**`decoder/` — what the decoder uses**

| | |
|---|---|
| `swap_pathway.py` | Give the decoder body A's frame with body B's latent. Which input tells it what body it is looking at? |
| `morphology_mix.py` | What mixture of training bodies does the model's answer look like — is it interpolating or copying one? |
| `morphology_axis.py` | Where a held-out body lands between two training bodies, at each stage of the pipeline. |
| `plot_action_trace.py` | Predicted against ground-truth joint commands, per joint. Aggregate error hides which joints failed. |
| `score_body.py` | One checkpoint against several held-out bodies, with both constant baselines and R². Use instead of retraining per test body. |
| `motion_decoder_ceiling.py` | The decoder's own ceiling, so a teacher-student number is read against what the decoder can supply, not against 1.0. |

**`latent/` — what the latent contains**

| | |
|---|---|
| `identity_linearity.py` | Is embodiment identity actually removed from `z`, or only hidden from a straight line? |
| `z_content.py` | Is the latent still a behaviour representation, or was it hollowed out? |
| `z_dynamics.py` | Does the latent carry the transition, or only the pose at time t? |
| `z_body_share.py` | How much of the latent is "which body is this", on trained and on held-out bodies. |
| `z_embodiment_share.py` | The same across a hexapod and a quadruped, using stance fraction as a shared phase label. |
| `z_identity_ablation.py` | Is the embodiment in the latent load-bearing or only present? Project it out and compare against deleting random directions. |
| `leg_contact_probe.py` | Does "is this leg loaded" read across embodiments? Needs no shared gait phase. |
| `wm_umap.py`, `cross_embodiment_umap.py` | The encoder embedding beside the learned latent. |

**`forward_model/` — whether the forward model works**

| | |
|---|---|
| `latent_rollout.py` | Closes the forward model on its own output and rolls it forward against two no-learning baselines. The only script that scores it on the task it is actually for. |
| `cross_latent_rollout.py` | Drive one robot's forward model with the other robot's latent, and see if the prediction holds. LAC-WM section 5.2, adapted. |
| `ftm_uses_z.py` | Does the forward model read the latent, or is the next frame guessable without it? |
| `aug_noise.py` | How much of the reconstruction target is augmentation noise rather than motion. |
| `target_window_sweep.py` | Does shortening the window turn the motion target from a state into a change? |
| `loss_gradient_balance.py` | Which loss term is actually pulling on the latent, measured as a gradient rather than a loss. |
| `rollout_fidelity.py` | How far a rollout can be trusted before it drifts off what the model was fitted on. |

**`setting/` — whether the setting supports the claim**

| | |
|---|---|
| `cross_embodiment_probe.py` | Does the frozen encoder place a hexapod and a quadruped in a shared space? Run before spending GPU on Stage 2. |
| `encoder_scale_probe.py` | Can a linear readout of the frozen encoder recover a body's segment scales, and does its error predict whether a split will transfer? Also reports the mixture gap, which is pure geometry and needs no encoder at all. |
| `pairing_feasibility.py` | Can a cross-embodiment `L_cross` be defined at all? Scores each candidate pairing label on coverage *and* on whether it pins down the command within one robot. A label that fails the second gives the decoder a wrong target, not a noisy one. |
| `swap_embodiment.py` | Does a latent inferred from an unseen embodiment drive the decoder, or is that embodiment simply read as a training body? The cross-embodiment form of `swap_pathway.py`, definable only for the 4-leg, which shares expert episodes with the hexapod. |
| `b1_horizon.py` | Is the B1's command as determined by a single frame as the insect's is? |
| `occlusion_dynamics.py` | Does the second frame matter more when one frame cannot fix the gait phase? |
| `pretrain_control.py` | Does insect pretraining transfer locomotion, or only familiarity with the feature space? |
| `speed_variance_split.py` | Is body speed readable from one frame because of physics, or because we built the data badly? |
| `pairing_taskspace.py` | Can a cross-embodiment pairing be built in task space, where the contact labels failed? |

**`shared_body_target/` — the shared body target**

| | |
|---|---|
| `screen_behaviour_channels.py` | Which body-motion channels carry shared meaning, now that both robots vary in them. **Supersedes `channel_screen.py`** (removed in the 2026-09-03 cleanup), which predated the corrected data. Run it against the checkpoint you are actually claiming about: on `beh12_hexonly` nothing transfers, because that run has one robot in it (F110). |
| `body_head_ablation.py` | Does the body-motion term constrain the latent, or does the head read the frame instead? |
| `body_motion_probe.py` | Does a body-level motion readout transfer between the two robots, where a leg-level one does not? |
| `show_body_motion_edges.py` | See the edge artefact in the `lambda_body` target, next to the video it was computed from. |
| `body_head_calibration.py` | Calibrates the body-motion head before a run is trusted, so the head's own scale isn't mistaken for a modelling result. |

**`cross_embodiment/` — cross-embodiment transfer**

| | |
|---|---|
| `fit_4leg_head.py` | Fit a new output head on a held-out embodiment with the backbone frozen. |
| `fit_b1_head.py` | The same on the B1, using a Stage 1 checkpoint so the quadruped is genuinely held out. `--stratify` matches velocities across the split; `--z_modes` ablates the latent, which is what showed the transfer travels entirely through `z` (F50). |
| `sweep_4leg_fewshot.py` | The same, swept over how many clips the new head gets. |
| `finetune_ftm.py` | Adapts the forward model to a target robot with a few clips; the source of the few-shot adaptation curve (F52). |

**`objective_experiments/`** — is the action channel necessary, and can the model recover it at all,
independent of the specific fix under test.

| | |
|---|---|
| `action_necessity.py` | Is the one-step prediction task easy enough to solve without reading the action at all? |
| `check_actswm_wiring.py` | Three wiring checks before the ActSWM rebuild burns five hours of pretraining. |
| `delta_action_decoding.py` | Delta-JEPA's LDAD, measured on what we already have before anything is retrained. |
| `dreamer_gradient.py` | Does the gradient through the world model's imagined rollout point the right way in the real world? |
| `intent_recoverability.py` | Is the signal a broken rhythm exposes intent, or only which behaviour this clip is? |
| `inverse_dynamics_r2.py` | How much does the transition add over a single frame, for reading the action? |
| `motion_rep_check.py` | De-risking Direction B: can a motion-organised representation be action-necessary and body-shared? |
| `null_action.py` | Which "zero action" means do not move rather than fall over, on each robot. |
| `null_separability.py` | Is `ITM(e_t, e_t)` still a different thing from a real transition, on this checkpoint's own data? |
| `residual_structure.py` | Is what the null-action prediction misses structured by the action, or is it noise? |

**`egocentric_view/`** — the Q1/Q2 thread: does the egocentric view fix action-conditioning, and
what does it cost.

| | |
|---|---|
| `branch_divergence.py` | After a shared prefix, do two actions produce separable futures — in position and heading? |
| `check_appearance_leak.py` | Can heading be read off the room's colour? If so, Q1 measures a leak, not egocentric. |
| `degait_coordinate.py` | Does removing the gait-locked component of an egocentric view improve the cross-body coordinate? |
| `embedding_divergence.py` | Does the counterfactual divergence survive the encoder the world model actually sees through? |
| `pooled_student_check.py` | Does the student's pooling throw away the thing that makes egocentric work? |
| `student_head_arch.py` | Attention or convolution over the token grid — and does either fit the 20 Hz budget? |
| `texture_for_vjepa.py` | Which wall texture does V-JEPA2 actually read motion from? |

**`planning/`** — everything below scores *selection*, not prediction. A model that predicts the
next frame well and ranks candidates badly passes every other table on this page.

| | |
|---|---|
| `plan_open_loop.py` | Run the planner over recorded frames, before any simulator is involved. Start here: if selection fails offline it will not appear in a loop. |
| `plan_discriminates.py` | Can the forward model tell a good candidate action from a bad one? |
| `does_rollout_matter.py` | Does rolling the forward model change which action the planner picks? Scores `rollout` against `direct` (no roll) and `blind` (no action) — deleting the rollout costs 24-30 points on the B1 (F100). |
| `action_identifies_behaviour.py` | Does the action, on its own, say which behaviour is being performed? The control for "this robot's actions carry nothing plannable", which was asserted and then refuted (F97). |
| `score_closed_loop.py` | Score a closed-loop run against the criteria fixed in advance. **Read its `channel_for` first**: grading a turn on forward speed is what it exists to prevent (F108). Survival is `n/a`, never 0%, on a loop that cannot fall. |
| `does_score_see_speed.py` | Does the planner's score separate *how fast*, or only *which behaviour*? |
| `why_speed_misses.py` | Is the candidate library too coarse to hit the commanded speed, or is the planner missing? |
| `what_stitching_costs.py` | Does switching between recorded clips cost travel, or was that an artefact of a residual? |
| `loop_frames_are_off_manifold.py` | Are the frames the loop drives itself into outside what the model was fitted on? |
| `hexapod_replay_fidelity.py` | Does replaying a recorded hexapod clip reproduce its speed, as F93 implies it must? |
| `b1_replay_stability.py` | Does a recorded B1 action sequence keep the robot upright when replayed open loop? Answers whether a fall is the planner's fault or the clip's. |
| `z_crosses_bodies.py` | Is the latent organised by behaviour or by body? **Its verdict has been backwards about the loop twice** — it ranks turning first and forward last, and the loop does the opposite (F107, F112). A pooled linear readout is not what the planner computes. |
| `wm_gait_report.py` | Gait diagram and side-by-side video, predicted commands against IK ground truth, driven through the same physics. |
| `score_by_body_motion.py` | Scores a closed-loop run by the shared body-motion coordinate rather than joint error. |
| `summarise_stage3_seeds.py` | Averages a stage-3 seed sweep over its late window, per arm, from the training log — `wm/adapt3.py` writes only the final `top1`/`family`. |
| `teacher_label_quality.py` | Can the teacher rank actions, and at which scale? The characterisation of F144's failure. |
| `clone_walk_test.py` | Does the egocentric clone walk further than the allocentric one? The behavioural row F184 filled in. |

---

## tools/

CoppeliaSim scene/robot calibration utilities — not diagnostics answering a research question, so
kept out of the `diagnostics/` package (no cross-imports touch them).

| | |
|---|---|
| `inspect_scene.py` | List what is inside a CoppeliaSim scene: joints, their order, and any attached scripts. |
| `run_olaf_reference.py` | Run the lab's Olaf scene as it ships and record its gait, as the reference to compare against. |
| `tune_legs.py` | Per-leg offsets and gains that make six unequal legs trace the same stroke. |

---

## dataset/

| | |
|---|---|
| `build_stage1_dirs.py` | Build the three clip directories the Stage 1 retrains read from. Links only clips that walk. |
| `audit_ik_dataset.py` | Audit an IK dataset for cross-morphology correspondence. |
| `make_ik_equal_windows.py` | Build equal-length clips from longer sources. |
| `plot_gait_quality.py` | Contact raster and duty factor per body, plus the raw forces against the 0.27 N cut. Use before trusting any stance-derived label. |
| `compare_ratio_gaits.py` | Side-by-side video and contact diagram across the femur/tibia boundary, from recorded frames. |
| `render_lock_check.py` | Is the camera and scene identical across bodies? |
| `write_run_log.py`, `recover_config_yaml.py` | Reconstruct a run's record after the fact. |

## figures/

Every one of these reads its numbers from disk. **None of them holds a measured value as a
literal**, so a figure cannot drift away from the run it describes.

| | |
|---|---|
| `plot_adapt_objective.py` | The adaptation-objective result: MSE on chance, InfoNCE well above it, one point per recorded goal clip (F112). |
| `plot_ftm_fewshot.py` | How many clips of a new robot adapt the forward model, and to what horizon. |
| `plot_4leg_fewshot_and_ablation.py` | The 4-leg head fit against clip count, with the latent ablated. |
| `plot_body_head_design.py`, `plot_cross_loss_effect.py` | What each added loss term buys and costs. |
| `plot_morphology_evidence.py`, `plot_z_umap.py` | The encoder's morphology readout, and the latent's layout. |
| `plot_obs_format.py`, `plot_ik_intuition.py` | Explanatory, not measurements. |
| `make_track_figures.py`, `make_stage2_diagram.py` | Pipeline diagrams. |

## dataset/, continued

| | |
|---|---|
| `collect_beh12.py` | Collect the twelve matched behaviours for one body. `--separability` is not optional: it checks that `side_L` travels positive lateral and `side_R` negative, and that each level exceeds the one below **on its own channel**. Two conditions shipped reversed before that check existed (F106). |
| `merge_behaviour_dirs.py`, `merge_speed_dirs.py` | Combine collected directories into one training set. |
| `preview_clips.py`, `plot_gait_compare.py` | Look at a new body before training on it. Always do this. |

## Traps this directory exists because of

| | |
|---|---|
| **a verdict is not a measurement** | Several scripts used to print "reachable" / "not reachable" from a threshold comparison of two overlapping means. They were removed. Report the number and the chance rate; let the reader conclude. |
| **`pgrep -f` matches the shell running it** | A wait-loop keyed on `pgrep -f "<script name>"` never exits, and the job it was guarding silently never starts. Check for output files, not for processes. |
| **CoppeliaSim fails silently** | No error, no exit, ~0% CPU. It needs a GUI instance (`DISPLAY=:0`), exactly one, and headless exits on its own. |
| **`--demo` is an intervention** | In the closed loop it supplies the starting state *and* the first `--warm_start` actions, which for a same-robot run are the goal clip's own. Hold it fixed and neutral, or vary it as a control (F109, F110). |
| **MuJoCo repeats, CoppeliaSim does not** | Rerunning a B1 configuration returns the identical number and carries no information; a hexapod configuration spreads 37-71%. Get spread from different goal clips on the B1, from repeats on the insect (F105). |

## finished/

Answered, kept because the findings cite them. Not part of any current workflow.
`test_vjepa2_encoder.py`, `test_vjepa2_frame_isolation.py`, `step0_macro_f1.py`,
`temporal_similarity.py`, `plot_sanity_check.py`, `plot_step_minus1.py`.

## amp/

The reinforcement-learning branch, not part of the world-model pipeline. Kept because the trained
policies are a candidate source of behavioural diversity, which the IK data lacks.
`render_rollout.py`, `gait_report.py`, `measure_g_range.py`.

---

## Traps these exist because of

Each was hit at least once and cost real time.

- **Never read a single epoch of held-out error.** It swings 2x within a run, and two runs of
  identical config and seed on different GPUs landed 2.1x apart. Average five epochs.
- **MSE is not comparable across datasets or across `action_lag` settings** — the standardisation
  differs. Degrees are.
- **Ratios like z-gap and x-gap compare between runs only**, since zeroing an input is
  out-of-distribution. The swap test feeds real embeddings and does not have this problem.
- **Below-chance probe accuracy is a failure, not a success.** Reaching 0.002 against a 0.200
  chance level means the latent is rotating the code faster than the classifier tracks it.
- **Fit mixtures with `scipy.optimize.nnls`**, not projected gradient — a hand-rolled version
  returned a solution 39x worse than optimal and briefly inverted a conclusion.
- **A small z-gap does not mean an empty latent.** Decode behaviour from `z` directly and split its
  variance before concluding anything about content.
- **Always run a matched control**, differing in exactly one flag.
- **"Present" and "used" are different questions.** A quantity can occupy a third of a
  representation's variance and still be read by nothing. Ablate it and compare against deleting
  the same number of random directions before calling it a mechanism.
- **One direction is not a subspace.** Delete the probe's weight vector and the probe refits onto
  correlated axes. Peel directions off one at a time until accuracy reaches chance, and if it
  never does, the signal is distributed and no adversary can excise it.
- **Check what your held-out body is actually held out *on*.** The 4-leg embodiment was built by
  removing legs from the *base* scene, so its geometry was a training body's and its commands were
  that body's corner columns bit-identically. It reads as a new embodiment and scores like one,
  but the only novel axis is leg count -- the latent lands 0.578 from the base body against a
  chance of 0.981. List the axes a test body is meant to be new on and verify each separately
  (F47).
- **A label two datasets share is not yet a label that means the same thing.** Checking that both
  robots visit the same contact patterns says the pairing is *defined*, not that it is *right*.
  A training pair carries the partner's command as its target, so a label that does not pin down
  the command within a single robot yields a wrong target rather than a noisy one, and wrong
  targets do not average out with more data. Measure coverage and meaning separately: coarsening
  a label until both sides overlap is the same operation that destroys what it meant (F45).
- **Do not retrain to change a *test* body — but you must retrain to change a *training* set.**
  A held-out body is never trained on, so any body absent from `train_morphs` already tests an
  existing checkpoint, and retraining for it changes the weights as well as the test: doing that
  once produced a "the frame is actively harmful, 1.34x" result the original checkpoint does not
  show. **This only holds when the training set is already sound.**
- **Say which constant baseline.** Beating the *training* mean pose only says the model noticed
  this is not an average body. R² is defined against the *held-out body's own* mean, and a model
  can beat the first while losing badly to the second — here, by 16.16 against 2.45.
- **A generated body is not a valid body until you watch it walk.** Two in `ik_walk_8body`
  collapse and rotate on the spot; they passed a walk check that used unsigned displacement, which
  reads a healthy 0.46 m for something tumbling. Check forward displacement and lateral drift
  separately, signed, and look at the frames. `wm.bodies.walk_check` is that check.
- **Never write a body list into a script.** See above; this is the same trap one level up, and it
  is why `wm/bodies.py` exists.
- **Do not score an animal's gait against a textbook template.** The expert here is a real stick
  insect walking a variable wave, so tripod separation sits at chance for every body and that is
  correct. Judge the labels instead: is the contact threshold in a gap.
- **How you pool the patch tokens is part of the measurement.** Mean-pooling suits quantities
  spread across the frame and buries ones confined to a few patches, and it also preserves a large
  constant offset between datasets that a fitted readout absorbs and then mis-applies. It turned a
  1.32x cross-embodiment result into 4.72x. Try more than one reduction and report the range.
- **A script written for one embodiment breaks on the other, and the loud break is the lucky one.**
  Both robots' clips are `.npz` but the B1 stores its command under `action` and the insect under
  `actions`, so `b1_horizon.py` died with a `KeyError` on insect data — visible, cheap. The same
  root cause in `morphology_mix.py` ran to completion and returned a number that was wrong by
  2.55 deg, because it rebuilt the held-out ground truth from disk and so dropped `action_lag`.
  `wm/data/embodiment.py` normalises the key names; a script that indexes the archive directly
  bypasses it. Suspect *every* silent success on an embodiment the script was not written for.
- **Defaults carry the size of the dataset they were written for.** `b1_horizon.py` encoded on the
  CPU by default, which is tolerable for the B1's 14 clips and is 9,240 ViT-g frames on the
  140-clip insect directory — still unfinished after 38 minutes. Same for `--clips`, `--chunk` and
  split counts: re-read the defaults when you point a script at a bigger directory.
- **Accumulating gradient over every batch is not the same as needing to.** `finetune_ftm.py`
  cached embeddings on the CPU to avoid an OOM, then summed gradients over *all* spans before each
  `opt.step()`. That made `--steps` count epochs rather than updates, made cost scale with the clip
  budget, and put the 1/3/5/7-clip sweep at 374,400 forward+backward passes and 13 hours. Only the
  *transfer to the device* has to be batched; sampling one batch per step is 15.6x cheaper and
  gives every cell an equal number of updates, which is what a budget comparison needs anyway.
