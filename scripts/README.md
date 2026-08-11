# scripts

Every file here has a docstring that states the question it answers and, where it matters, the
measurement trap it exists to avoid. Read that before running one — several of these exist because
an earlier, more obvious version of the same measurement gave a confidently wrong answer.

Run everything from the repository root with `.venv/bin/python3`, never bare `python3`.
Add `--encode_device cpu` to anything that encodes frames when a training run holds the GPU.

---

## Imported, not run

| | |
|---|---|
| `vjepa2_encoder.py` | Frozen V-JEPA2 wrapper. Encodes single frames in isolation via the frame-duplication trick. Imported by almost everything below. |

---

## The live toolkit — world model diagnostics

These are the measurements the current findings rest on. `wm/READING_THE_LOG.md` covers the
training log itself; these go deeper.

**What the decoder uses**

| | |
|---|---|
| `swap_pathway.py` | Give the decoder body A's frame with body B's latent. Which input tells it what body it is looking at? |
| `morphology_mix.py` | What mixture of training bodies does the model's answer look like — is it interpolating or copying one? |
| `morphology_axis.py` | Where a held-out body lands between two training bodies, at each stage of the pipeline. |
| `plot_action_trace.py` | Predicted against ground-truth joint commands, per joint. Aggregate error hides which joints failed. |
| `score_body.py` | One checkpoint against several held-out bodies, with both constant baselines and R^2. Use instead of retraining per test body. |

**What the latent contains**

| | |
|---|---|
| `z_content.py` | Is the latent still a behaviour representation, or was it hollowed out? |
| `z_dynamics.py` | Does the latent carry the transition, or only the pose at time t? |
| `z_body_share.py` | How much of the latent is "which body is this", on trained and on held-out bodies. |
| `z_embodiment_share.py` | The same question across a hexapod and a quadruped, using stance fraction as a shared phase label. |
| `z_identity_ablation.py` | Is the embodiment in the latent load-bearing or only present? Project it out and compare against deleting random directions. |
| `wm_umap.py` | UMAP of the encoder embedding beside the learned latent. |

**Whether the forward model works**

| | |
|---|---|
| `latent_rollout.py` | Closes the forward model on its own output and rolls it forward against two no-learning baselines. This is the only script that scores it on the task it is actually for. |
| `aug_noise.py` | How much of the reconstruction target is augmentation noise rather than motion. |
| `gap_signal.py` | How far apart two frames must be before the real change beats that noise. |

**Whether the setting supports the claim**

| | |
|---|---|
| `cross_embodiment_probe.py` | Does the frozen encoder place a hexapod and a quadruped in a shared space? Run before spending GPU on Stage 2. |
| `b1_horizon.py` | Is the B1's command as determined by a single frame as the insect's is? |
| `occlusion_dynamics.py` | Does the second frame matter more when one frame cannot fix the gait phase? |

**Does the command actually walk**

| | |
|---|---|
| `wm_gait_report.py` | Gait diagram and side-by-side video, predicted commands against IK ground truth, driven through the same physics. |

---

## Dataset tools

| | |
|---|---|
| `audit_ik_dataset.py` | Audit an IK dataset for cross-morphology correspondence. |
| `make_ik_equal_windows.py` | Build equal-length clips from longer sources. |
| `plot_ik_walk_gait.py` | Gait and contact diagrams for the IK forward-walk dataset. |
| `plot_gait_quality.py` | Contact raster and duty factor per body, plus the raw forces against the 0.27 N cut. Use before trusting any stance-derived label. |
| `compare_ratio_gaits.py` | Side-by-side video and contact diagram across the femur/tibia boundary, from recorded frames. |
| `render_lock_check.py` | Render-lock check: is the camera and scene identical across bodies? |
| `inspect_coppelia_scene_objects.py` | Compact object summary for a CoppeliaSim scene. |

---

## Finished — Step 0 and Step -1 checks

Answered, kept because the findings cite them. Not part of any current workflow.

| | |
|---|---|
| `test_vjepa2_encoder.py` | Download V-JEPA2 and confirm the output shape. |
| `test_vjepa2_frame_isolation.py` | Confirm the frame-duplication trick gives a context-independent `e_t`. |
| `step0_macro_f1.py` | Foot-contact macro-F1, within against across morphology, with a shuffled-label control. |
| `temporal_similarity.py` | Temporal similarity of frozen embeddings. |
| `plot_sanity_check.py` | Does frozen `e_t` contain useful locomotion information? |
| `plot_step_minus1.py` | Does a morphology gap exist at all? |
| `plot_morphology_evidence.py` | Evidence that frozen `e_t` encodes leg-length morphology. |
| `plot_ik_intuition.py` | Why fixing the command and fixing the foot target give different joint angles. |
| `plot_obs_format.py` | Figure for "why test vision when proprioception is available". |
| `make_stage2_diagram.py` | Stage 2 training and testing diagram. |

---

## Parked — AMP policy work

The reinforcement-learning branch, not part of the world-model pipeline. Kept because the trained
policies in `amp/logs/` are a candidate source of behavioural diversity, which the IK data lacks.

| | |
|---|---|
| `render_rollout.py` | Render a trained AMP policy rollout to mp4 to eyeball the gait. |
| `gait_report.py` | Is the learned policy's gait actually like the expert's? |
| `sweep_gait_checkpoints.py` | Find the checkpoint whose contact pattern is closest to the expert. |
| `diagnose_amp_rollout.py` | Diagnose an AMP checkpoint numerically, without rendering. |
| `measure_g_range.py` | Measure the raw discriminator output over a deterministic rollout. |

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
- **Do not retrain to change a *test* body — but you must retrain to change a *training* set.**
  A held-out body is never trained on, so any body absent from `train_morphs` already tests an
  existing checkpoint, and retraining for it changes the weights as well as the test: doing that
  once produced a "the frame is actively harmful, 1.34x" result the original checkpoint does not
  show. **This only holds when the training set is already sound.** `tib_cross` qualifies — its
  four bodies are all ratio 0.83 with 42-71 mm dead zones. `m3d_cross` does not: two of its five
  training bodies veer, so removing those is a different model and needs a run. Ask which half of
  the experiment is changing before deciding.
- **Say which constant baseline.** Beating the *training* mean pose only says the model noticed
  this is not an average body. R^2 is defined against the *held-out body's own* mean, and a model
  can beat the first while losing badly to the second — here, by 16.16 against 2.45.
- **A generated body is not a valid body until you watch it walk.** Two in `ik_walk_8body`
  collapse and rotate on the spot; they passed a walk check that used unsigned displacement, which
  reads a healthy 0.46 m for something tumbling. Check forward displacement and lateral drift
  separately, signed, and look at the frames.
- **Do not score an animal's gait against a textbook template.** The expert here is a real stick
  insect walking a variable wave, so tripod separation sits at chance for every body and that is
  correct. Judge the labels instead: is the contact threshold in a gap.
- **How you pool the patch tokens is part of the measurement.** Mean-pooling suits quantities
  spread across the frame and buries ones confined to a few patches, and it also preserves a large
  constant offset between datasets that a fitted readout absorbs and then mis-applies. It turned a
  1.32x cross-embodiment result into 4.72x. Try more than one reduction and report the range.
