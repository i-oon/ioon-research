# Wall selection: dynamics vs evaluation, staged diagnostic plan (2026-10-14, draft for review)

Request (user, 2026-10-14): separate DYNAMICS (projector + FTM predict the consequence of a command) from EVALUATION
(a separate readout / cost judges that consequence against the motion goal and the wall). No safety or clearance loss into
FTM, ITM, encoder or projector in this phase; freeze the existing model and diagnose first. Physics clearance / status
labels may train or evaluate the readout, never be an input at deployment.
**This supersedes DATA_PLAN 13 "run 2" (an imagined-clearance term trained through the forward model).** If wall data is
later used to improve the FTM, it is trained only with its original consequence objectives, as a separate experiment
from the evaluator.

## 1. Inference-time information flow (what is available when the robot decides)

```
INPUTS at decision time t (no simulator, no physics labels):
  camera frame I_t ──► frozen V-JEPA2 (grayscale) ──► e_t            [256 tokens x 1408]
  candidate set {a^c}  (same set for every controller in a comparison)
      a^c = the command's own control signal over the next k frames, in the body's projector space:
            B1: 12-D joint targets of its walking policy (recorded chunk), hexapod: 18-D CPG joint commands
  motion goal g (Froude fwd / lat / yaw, from a recorded goal or read from a goal video)

DYNAMICS (frozen; no gradient from anything below):
  z^c_i   = projector_body(a^c chunk i)                     i = 1..5   (5 frames per step, 0.25 s)
  ê^c_i   = FTM(ê^c_{i-1}, z^c_i),  ê^c_0 = e_t             -> 5 imagined embeddings, 0.25 .. 1.25 s

EVALUATION (separate, trained on detached embeddings; physics labels only as TRAINING targets):
  motion term   m^c = Froude head(ITM(ê_{i-1}, ê_i)) averaged over i  vs  g      (existing, F342 goal alignment)
  safety term   s^c = min_i  S(ê^c_i)                        S = the safety readout (section 2)
  cost^c        = goal_error(m^c, g) + lambda * penalty(s^c)        lambda, penalty shape fixed on VALIDATION only

SELECT  c* = argmin cost^c ; execute the first chunk; replan every decision step.
```
Comparators get exactly the same inputs minus what they are defined not to see (section 4).

## 2. Which safety target, for a 1.25 s horizon

Measured on the test branches (all 28 commands, true futures): within 1.25 s a command reaches NEAR or crosses ONLY from
the near band (0.4-0.8 m): B1 16 % (crossing 4 %), c10 30 % (crossing 7 %); from 0.8-2.0 m: 0 % for both bodies. Within
3 s: 5-53 %.
- P(NEAR or crossing within the horizon): degenerate for 2 / 3 of the states (all zeros) -> cannot rank commands there;
  very imbalanced elsewhere. Not the primary target.
- **Clearance at each imagined step (in body lengths), primary.** Defined on every frame, dense, ranks commands even
  when none comes near within 1.25 s, and its trend carries what happens later: the TRUE clearance at 1.25 s ranks the
  commands that cross within 5 s at within-state AUC 0.977 (far-band pilot, DATA_PLAN 13). Body lengths make it
  comparable across bodies. The safety term is the MINIMUM over the imagined steps (a command can dip and recover).
- Clearance CHANGE from now (c(t+h) - c(t)) as a variant: removes the current distance, which the current frame already
  shows; tells whether prediction adds something beyond "how close am I".
- P(NEAR within the held 3 s) from the 1.25 s prediction: the decision-relevant event, reported as a DETECTION metric of
  the clearance readout (threshold fitted on validation), and as a second readout only if clearance fails.

## 3. Staged diagnostics (frozen world model; base `round1_branches_s0_gray_imread`)

Data: wall test (and val for fitting) branches, all 28 commands per state. Matched time points: imagined step i <-> true
frame first_pair - 1 + 5i (0.25 .. 1.25 s). Branches stop at the crossing: frames after it do not exist -> MASKED (no
label, excluded from fits and metrics; a crossed branch counts as crossed for every later step in the detection metric).

| stage | question | readout | reference rows | pass rule (fixed now) |
|---|---|---|---|---|
| D1 | do REAL future embeddings carry the outcome? | ridge on mean-pooled real e_{t+h} -> clearance (body lengths); logistic -> NEAR within 3 s | copy current (e_t), command-only (current distance from e_t + the command's mean change on fit data) | real-future within-state Spearman > command-only at 1.25 s (else the embedding cannot carry it: stop, nothing to imagine) |
| D2 | does the FROZEN imagination carry it? | the D1 readout applied unchanged to imagined ê_{t+h} | same + real-future (ceiling) | report the real-vs-imagined shift (mean / scale per horizon, MAE); no pass rule (diagnostic) |
| D3 | is it decodable from imagined embeddings? | readout trained on IMAGINED ê (detached; world model frozen) -> true clearance; safety = min over steps | copy, command-only, real-future ceiling, D2 | imagined > command-only AND > copy at 1.25 s, on test, both bodies separately |
| D4 | does the imagination add value over a direct risk model? | DIRECT action-conditioned risk: (e_t, candidate command) -> clearance at h, same data / targets / capacity | D3 | D3 > D4 on within-state ranking, else "seeing helps, imagining does not" |

Metrics (per horizon, per distance band, per approach group head-on / oblique / corner, per body): within-state Spearman
across the 28 commands; unsafe detection (recall / precision / AUC for NEAR-or-cross within 3 s); MAE in body lengths;
calibration (reliability of P(NEAR) if a probability readout is used). Fit on train (+ val for hyper-parameters), report
on test; the c10 rows also on c08 later (test only).

Closed loop (only after D3 passes): paired start states, goal "walk forward", same candidate set for every controller:

| controller | sees | judges the future by |
|---|---|---|
| blind direct | goal + command | motion only, no safety term |
| current-image reactive | e_t | safety readout on e_t (no prediction) + motion term |
| direct action-conditioned risk (D4) | e_t + command | predicted clearance without imagined views |
| imagined-future risk (rollout) | e_t + command | safety readout on imagined ê (D3) |
| true-future oracle | true future | true clearance (upper bound, privileged) |

Metrics: boundary crossings, NEAR time, closest clearance, forward progress, Froude goal error, complete escapes.

## 4. What exists, what needs code or data

| item | status |
|---|---|
| frozen model, projector, FTM rollout, Froude motion term (planner `_score_window`, stride 5, window 21) | EXISTS |
| D1 / D2 / D3 readouts with copy and command-only controls | EXISTS for the old 24-command, 2 s held set (`scripts/diagnostics/wall_pilot/b1_wall_predcheck.py`, F334 / F338) -- needs: the new wall data (28 commands, B1 + c10), masking after the crossing, min over steps, body lengths, per band / group, NEAR-within-3 s detection |
| wall data, all 28 commands per test / val state, labels per frame | B1 done; c10 being balanced (gates pass) |
| D4 direct action-conditioned risk model | NEW code (small: pooled e_t + projector z or the command -> clearance; trained on the same branches) |
| safety term in the planner (cost = goal error + lambda * penalty) | NEW code (planner option; the safety readout is a separate file, loaded at run time) |
| closed-loop wall environment (virtual boundary, paired starts, episode end at crossing) | NEW code: B1 in the MuJoCo loop (`b1_mujoco_env`), hexapod in the CoppeliaSim loop |
| the 4 spin commands in the projector | GAP: the projectors were fitted on the 24 tuned clips; spins are off-library for both bodies -> check projector / FTM accuracy on spin branches (section 5) before using them |

## 5. Command space (audit) and the continuous-candidate experiment

- What the projectors accept: per-frame CONTROL signals in chunks of 5 frames, not task commands. B1: 12-D joint targets
  produced by its walking policy (state-dependent: the same (vx, vy, wz) gives different joint targets in different
  states); hexapod: 18-D CPG joint commands (open-loop given the recipe and the gait phase).
- Consequence: a candidate must be a joint-target chunk. For the 28-command benchmark, every candidate's chunk comes
  from the recorded branch / library (controlled benchmark, comparable with earlier results).
- Continuous candidates (separate experiment, not in the benchmark):
  - hexapod: any feasible CPG recipe (amplitudes, spin, strafe, lead within the tuned ranges) gives its joint chunk
    in closed form (`cpg_clock.commands_at` at the current phase) -> can be scored directly.
  - B1: a new (vx, vy, wz) has no joint chunk without running the policy from the current state (privileged at
    deployment). Options: (a) a B1 COMMAND projector (vx, vy, wz) -> z, fitted on the recorded `command` field (clips +
    wall data); (b) a learned joint-chunk generator. (a) is simpler; it changes the projector input, so it is its own
    run.
  - Before control: projector / FTM accuracy on commands NOT in the recorded library (held-out recipes / commands
    collected as branches): real-vs-imagined read on those, like D2. Every controller in a comparison gets the same
    candidate set (the benchmark's 28, or the same sampled continuous set).

## 6. Retreat and escape with one-step scoring

- Evidence so far (F332, B1, TRUE futures, old 24 commands, decision every 0.1 s): one-action receding-horizon scoring
  avoided every crossing (0 / 48) and kept progress 0.58-0.61, mostly by side-stepping along the wall; it did not need a
  plan. With spins in the library, backing up / spinning are available to it as single held commands.
- Open: whether one-step scoring with the LEARNED cost escapes or dithers (back up -> forward -> back up). Test with the
  true-future oracle first on the new data (no model): episodes from the wall starts, decision every 2 frames, cost as in
  section 1; count crossings, dithering (sign flips of forward command in < 1 s), escapes completed.
- If one-step cannot escape: a small SEQUENCE test, two-segment candidates (retreat or spin for 0.5-1 s, then forward),
  scored over 2.5 s with the same oracle, then with the model; reported separately from any claim about the world model
  (the gain would come from the planner, not from imagination).

## 7. Which result supports which claim

| result | supports | does not support |
|---|---|---|
| D1 passes | wall outcome is in the visual embedding | anything about prediction |
| D3 passes, D4 not better | the frozen imagination carries the outcome | that imagining beats a direct risk model |
| D3 > D4 and closed loop: rollout < crossings / > progress vs current-image and direct-risk | the imagined visual future improves the choice (per body) | cross-body, unless shown on each body |
| same on B1 and c10 (c08 test) with equal data | predictive wall selection across bodies | -- |
| sequence planner needed for escapes | the planner matters | gains attributed to the world model |
| continuous candidates work after the off-library check | the method is not tied to the recorded library | -- |

## 8. Order

1. D1-D3 on the frozen base model, B1 then c10 (extend `b1_wall_predcheck.py`; no training of the world model).
2. Off-library check of the 4 spins (D2-style on spin branches).
3. D4 direct risk model.
4. Oracle one-step escape test (no model); sequence test only if needed.
5. Closed loop (needs the wall environment + planner safety term).
6. Separate experiments: FTM trained on wall data with its ORIGINAL objectives only; continuous candidates.

## 9. LOCKED protocol for D1-D3 (2026-10-14, before any result; user: common 24 commands first, FTM frozen, report
## before changing the world model or building closed loop; continuous action search out of scope)

- Model: `round1_branches_s0_gray_imread` (grayscale, uncentred), frozen; B1 = `ckpt/b1.pt`, c10 = `ckpt/hex.pt`.
  Bodies analysed separately; c08 later (test only).
- Data: the BALANCED wall data (cells equal across bodies, `balance_wall_data.py`), gates passed. Commands: the common
  24 (indices 0-23); spins (24-27) excluded from fits and metrics in this round.
- Splits: readouts FITTED on train only (train states render 14 of 28 commands -> ~12 of the 24); ridge alpha by RidgeCV
  (efficient leave-one-out) on train. REPORTED on test (primary, all 24 per state) and val (secondary). No refit on val /
  test, no threshold tuned on test; the NEAR threshold of the detection metric is the margin itself (0.13 body lengths).
- Embeddings: mean-pooled over the 256 tokens. Decision frame e_t = row first_pair-1 (last shared frame); imagined
  e^_i = FTM^i(e_t, projector(command chunk i)), chunk i = rows first_pair + 5(i-1) .. +4 (= the planner's
  `action_chunk_at(actions, t + 1 + 5(i-1), 5)`); real e_{t+i} = row first_pair-1+5i; i = 1..5 (0.25-1.25 s).
- Targets: clearance in BODY LENGTHS at row first_pair-1+5i. A branch that crossed before that row has no frame there:
  MASKED in every fit and in MAE; in the within-state RANKING it counts as worse than every command still present (its
  value = its last stored clearance, <= 0). Detection label: NEAR or CROSSED within 3 s after the branch point (rows
  first_pair .. first_pair+59); score = MINIMUM predicted clearance over steps 1..i (the planner's safety term).
- Readouts and baselines (all fitted on train only):
  real-future   R_real (ridge on train REAL frames, every 5th row, all 24 commands) applied to the TRUE e_{t+i} (ceiling)
  D2 imagined   R_real applied unchanged to imagined e^_i
  D3 imagined   R_imag,i (ridge on train IMAGINED e^_i, detached -> true clearance at step i), one per horizon
  copy          R_real(e_t), the same for every command (cannot rank: Spearman undefined; detection pooled only)
  command-only  R_real(e_t) + the train-set mean change of true clearance of THAT command at step i (no prediction
                from the image beyond the current distance)
  command-only, true start   the same with the TRUE start clearance (privileged reference)
- Metrics per horizon (test; also per distance band and approach group): within-state Spearman across the 24 commands
  (mean over states); MAE (body lengths, unmasked rows); detection AUC for NEAR-or-cross within 3 s, pooled and
  within-state (pairs in one state); real-vs-imagined shift = mean and std of (R_real(e^_i) - R_real(e_{t+i})).
- Uncertainty: 95 % bootstrap over EPISODES (states of one episode are correlated), 2000 resamples, of every difference
  used in a pass rule.
- PASS rules (test, at 1.25 s, per body):
  D1: Spearman(real-future) - Spearman(command-only) > 0 with the bootstrap lower bound > 0. Fail -> stop (nothing to
      imagine).
  D2: diagnostic only (shift, MAE, Spearman reported).
  D3: Spearman(D3) - Spearman(command-only) lower bound > 0 AND detection AUC within-state (D3) > command-only's AND
      pooled AUC (D3) > copy's. Fail -> the frozen imagination does not carry the outcome beyond what the command and
      the current distance give.
- Script: `scripts/diagnostics/wall_pilot/wall_predcheck.py` (new data layout; `b1_wall_predcheck.py` stays for the
  F334 / F338 sets). Output `results/check/wall_predcheck_v3/<body>_<model>.txt` + cached embeddings.
