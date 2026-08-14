# scripts

Every file here has a docstring stating the question it answers and, where it matters, the
measurement trap it exists to avoid. Read that before running one — several of these exist because
an earlier, more obvious version of the same measurement gave a confidently wrong answer.

Run everything from the repository root with `.venv/bin/python3`, never bare `python3`.
Add `--encode_device cpu` to anything that encodes frames when a training run holds the GPU.

```
scripts/
  vjepa2_encoder.py   the frozen encoder wrapper, imported by almost everything
  *.sh                run sheets: retrain_stage1, com7_train, retrain_stage2_clean_adv
  diagnostics/        what the model learned, and whether it works
  dataset/            build, audit and inspect the recorded data
  figures/            figures for the report that are not measurements
  finished/           answered questions, kept because the findings cite them
  amp/                the parked reinforcement-learning branch
  _archive/           superseded; not expected to run
```

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

**What the decoder uses**

| | |
|---|---|
| `swap_pathway.py` | Give the decoder body A's frame with body B's latent. Which input tells it what body it is looking at? |
| `morphology_mix.py` | What mixture of training bodies does the model's answer look like — is it interpolating or copying one? |
| `morphology_axis.py` | Where a held-out body lands between two training bodies, at each stage of the pipeline. |
| `plot_action_trace.py` | Predicted against ground-truth joint commands, per joint. Aggregate error hides which joints failed. |
| `score_body.py` | One checkpoint against several held-out bodies, with both constant baselines and R². Use instead of retraining per test body. |

**What the latent contains**

| | |
|---|---|
| `z_content.py` | Is the latent still a behaviour representation, or was it hollowed out? |
| `z_dynamics.py` | Does the latent carry the transition, or only the pose at time t? |
| `z_body_share.py` | How much of the latent is "which body is this", on trained and on held-out bodies. |
| `z_embodiment_share.py` | The same across a hexapod and a quadruped, using stance fraction as a shared phase label. |
| `z_identity_ablation.py` | Is the embodiment in the latent load-bearing or only present? Project it out and compare against deleting random directions. |
| `leg_contact_probe.py` | Does "is this leg loaded" read across embodiments? Needs no shared gait phase. |
| `wm_umap.py`, `cross_embodiment_umap.py` | The encoder embedding beside the learned latent. |

**Whether the forward model works**

| | |
|---|---|
| `latent_rollout.py` | Closes the forward model on its own output and rolls it forward against two no-learning baselines. The only script that scores it on the task it is actually for. |
| `aug_noise.py` | How much of the reconstruction target is augmentation noise rather than motion. |

**Whether the setting supports the claim**

| | |
|---|---|
| `cross_embodiment_probe.py` | Does the frozen encoder place a hexapod and a quadruped in a shared space? Run before spending GPU on Stage 2. |
| `pairing_feasibility.py` | Can a cross-embodiment `L_cross` be defined at all? Scores each candidate pairing label on coverage *and* on whether it pins down the command within one robot. A label that fails the second gives the decoder a wrong target, not a noisy one. |
| `swap_embodiment.py` | Does a latent inferred from an unseen embodiment drive the decoder, or is that embodiment simply read as a training body? The cross-embodiment form of `swap_pathway.py`, definable only for the 4-leg, which shares expert episodes with the hexapod. |
| `b1_horizon.py` | Is the B1's command as determined by a single frame as the insect's is? |
| `occlusion_dynamics.py` | Does the second frame matter more when one frame cannot fix the gait phase? |

**Cross-embodiment transfer**

| | |
|---|---|
| `fit_4leg_head.py` | Fit a new output head on a held-out embodiment with the backbone frozen. |
| `sweep_4leg_fewshot.py` | The same, swept over how many clips the new head gets. |

**Does the command actually walk**

| | |
|---|---|
| `wm_gait_report.py` | Gait diagram and side-by-side video, predicted commands against IK ground truth, driven through the same physics. |

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

`make_track_figures.py`, `make_stage2_diagram.py`, `plot_morphology_evidence.py`,
`plot_obs_format.py`, `plot_ik_intuition.py`.

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
