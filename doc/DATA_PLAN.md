# Data plan v2 — counterfactual pretraining data (both bodies identical by design)

> **Status 2026-10-02:** stages 1-3 built (`data/counterfactual_walks/`); hexapod main clips + branches being re-collected
> deterministically (section 8). Changes made after the draft below, all in FINDINGS:
> - Froude labels at the **centre of mass** for both bodies (`com_pos`, CoM height) — F301/F302; B1 commands tuned to
>   the hexapod's CoM Froude on all three channels (24/24 within 0.002).
> - B1 v4 main clips re-rendered with the camera along the body (`b1_clips_*`; old in `_superseded/b1_camera_yawed/`) — F304.
> - Every file records `cam_pose` (camera world pose per frame), for the camera-vs-body experiments.
> - Velocity differencing and smoothing stop at command switches; training pairs start at `first_pair` in every stage — F303.
> - val / heldout branches use 3 branch points like train (1,728 files each); pair counts per body: main 2,688 / 1,344 /
>   1,344, branches 38,016 / 19,008 / 19,008.
> - Hexapod branches by replay had ~6 cm start noise (Bullet + scene reload, F304) -> section 8.

> Draft for review before anything is generated (2026-10-01). Replaces: the hexapod beh24 clean clips (F300
> duplicates, F297/rooms), hexapod switching clips (kept as optional extra), the room assignment of all B1 data.
> Conventions: memory `project_timescale_conventions` (0.05 s frames, 1 s edge-correct switch-aware labels).

## 0. Corrections after review (2026-10-01)

1. **A long walk is rendered live in one room, but its windows need different rooms.** Hexapod collection will
   therefore record the physics (base pose, actual joint positions every frame) and render each window afterwards
   with a **hexapod replay renderer** (new; the B1 already works this way via `render_b1_replay.py`). Gate: a replay
   render of a live-rendered clip must reproduce its frames (pixel corr ~1.0) before any data is made.
2. **Window position in the room.** Each window is re-centred so its first frame is at the room centre with the
   same spawn convention for both bodies (as the B1 v3 renders do); otherwise later windows start near a wall.
3. **Condition matching across bodies** (for shared room seeds) is by family and level order: speed fwd 1-4,
   speed bwd 1-4, turn left 1-4, turn right 1-4, side L 1-4, side R 1-4. Checked: signs agree on both bodies
   (turn = +yaw / left, `_neg` = -yaw, side_L = +lateral, side_R = -lateral).
4. **Hexapod branches** are simulated by replay (physics only), then rendered with the same replay renderer in the
   source window's room, so main clips and branches share one rendering path.

## 1. Main clips (one command per clip)

| | hexapod c10f10t10 | B1 |
|---|---|---|
| behaviours | the 24 beh24 conditions (`collect_beh24.py` corrected recipe, F298) | the 24 B1 beh24 conditions |
| how a clip is made | **one long walk per condition, clips cut at different times** (new; replaces the 4 identical repeats, F300) | windows of one long rollout per condition (existing, F293-corrected poses) |
| clips per condition | train 2, val 1, heldout 1 (4 windows of the same long walk, non-overlapping) | same |
| clips per split | 48 / 24 / 24 | 48 / 24 / 24 |
| length | 66 frames (3.3 s) | 66 frames |

Rule: same-condition clips must differ in start state (gait phase, body pose, position). Checked (section 6).

## 2. Rooms (identical for both bodies)

| split | room seeds | |
|---|---|---|
| train | 0–47 | main clip of (condition c, copy k) -> seed = 2·index(c) + k, **same seed for both bodies** |
| val | 100–123 | seed = 100 + index(c) |
| heldout | 200–223 | seed = 200 + index(c) |

- One file = one room for its whole length. Branches inherit their source clip's room.
- Both bodies see exactly the same rooms per split; no room in two splits.
- Room recipe unchanged (`ego_camera`: size by camera height, random wall/floor texture and colour per seed). Lighting fixed (randomised later, round 2).
- Hexapod clips must be re-collected (rendered live); B1 clips and branches re-rendered (no physics rerun).

## 3. Counterfactual branches (same structure for both bodies)

- From every main clip, **3 branch points**, chosen so the gait phase at the branch is spread (early / middle /
  late in the gait cycle; hexapod phase from the plan, B1 from foot contacts), with ≥ 10 frames before and 20 after.
- At each branch point, **all 24 commands** (incl. the clip's own command = no-switch control).
- File layout: 10 frames before the branch (source) + branch frame + 20 frames of the new command = 31 frames;
  fields `segment`, `first_pair` (= 10), `froude_height`, source settings, `cf_*`.
- Counts per body: train 48·3·24 = **3,456**; val 24·3·24 = **1,728**; heldout **1,728**. All **576 ordered
  (source behaviour -> target behaviour) pairs** in every split (train ×6, val ×3, heldout ×3).

| | B1 | hexapod |
|---|---|---|
| how the branch state is reached | MuJoCo exact state restore (prefix identical across the 24 branches) | **replay** of the source commands from frame 0 (Bullet is not deterministic: prefix matches a repeat of the source to ~2–13 mm / 0.003–0.05 rad, side up to 0.2 rad; measured) |
| each branch's own prefix | the source's frames | its **own** replayed frames (labels and frames exact for that file) |
| switch shape | the walking policy smooths the command change | **causal 4-frame cross-fade** starting at the branch: cmd = (1−w)·cmd_old + w·cmd_new on the same continuous gait phase (no joint jump; prefix commands unchanged) |
| noise floor | none (exact) | the no-switch branch (own command) of every group measures the replay noise; stored and reported |

Known difference, written down rather than hidden: the B1's 24 branches start from one identical state, the
hexapod's from states a few mm apart. The effect of the action exceeds this from ~5 frames after the switch
(side -> turn ~10 frames).

## 4. Equal amounts

Training pairs (stride 5, rollout 2): main clips ~2.7k per body; branches 3,456 × 11 = 38,016 per body (train),
1,728 × 11 = 19,008 (val, heldout). Equal by construction.

## 5. What is dropped / kept

- Old hexapod beh24 clean clips (identical repeats, 4 rooms): archived, not used.
- Hexapod switching clips (one physics run, 3–4 behaviours): archived; optional extra in a later round (with B1
  switching clips added so both bodies match).
- Existing B1 branches: regenerated (new branch points, new rooms).

## 6. Checks (automatic, all must pass before training) + human review

1. Splits: no frame, file or source clip in two splits (`check_splits.py`).
2. Rooms: both bodies use exactly the same seed set per split; every room contains all 24 behaviours.
3. Coverage: all 576 source->target pairs present in every split, for both bodies; equal pair counts.
4. Distinctness: same-condition main clips differ in start state (no duplicate motion); new check in `check_splits.py`.
5. Branch integrity: prefix identical across a group (B1) / within the measured noise floor (hexapod); own-command
   branch matches the source (B1 ~1e-6; hexapod within noise); branch-point gait phases spread.
6. Labels: `tests/test_froude_labels.py`; labels after the switch independent of the prefix.
7. Commands: hexapod main clips reproduce the recipe; no falls.
8. **Human review:** sample videos — same branch point, several commands, both bodies side by side; rooms per
   split; one same-condition pair of main clips.

## 7. Cost (CPU / CoppeliaSim / MuJoCo; GPU not needed)

Hexapod main clips ~10 min; hexapod branches (3,456 + 1,728 + 1,728 ≈ 6.9k × ~7 s, 7 instances) ~2 h; B1 main
re-render ~15 min; B1 branches (6.9k, physics + render) ~1.5 h; checks + videos ~30 min.

## 8. Hexapod determinism (2026-10-02)

Re-running a plan after `sim.loadScene` is not bit-identical (differences from physics step 4, ~1e-16, growing to cm).
Loading the scene once and re-running with stop/startSimulation, after warm-up runs, is bit-identical (classes differ
between instances, so a behaviour's walk and all its branches run on one instance). Hexapod main walks and branches are
re-collected this way (`scripts/dataset/collect_c10_walks_and_branches.py`, `sim/collect/scene_reuse.py`) into `c10_walks`, `c10_clips_*`, `c10_branches_*` (`data/counterfactual_walks/`):
every branch's state at the branch frame must equal the walk's state bit-exactly (else re-run), and the own-command
branch equals the walk. B1 targets are re-checked against the new hexapod labels (stop if out of tolerance).


## 9. Four-leg bodies (adaptation test) — plan, 2026-10-04, not yet built

**Question.** Does the pretrained model adapt to a new body with a different gait from **babbling only** (no behaviour
labels, no counterfactual branches of the new body in adaptation), measured with the same heldout tests as c10 / B1?
**Bodies.** c10f10t10 with one leg pair ghost-removed (`collect_ik.ghost_remove_legs`, `drive_and_record(remove_legs=,
active_legs=)`): `hind_loss` first, then `middle_loss`, `front_loss` with the identical pipeline. Feasibility
(`results/check/four_leg_feasibility/summary.txt`): c10 commands do not reproduce c10 behaviours on any variant;
hind_loss babble 0.84 falls per upright minute, 90 % usable, 17/24 c10 targets within 0.05 Froude, misses backward
(segments reach fwd -0.091 vs target -0.236) and the strongest turns.

**Each variant = its own embodiment**: 12-D action (joint targets of the 4 remaining legs, `collect_ik.LEGS` order),
own projector head, own Froude height H4 = median CoM z over the upright frames of its train babble episodes (hind_loss
≈ 0.11 m), stored as `froude_height` in every file so all windows and branches of a body share one scale
(`wm/data/embodiment.py:122` `_com_reference` honours it). Labels = measured CoM Froude (fwd, lat, yaw), loader
conventions (1 s edge-correct, segment-aware, read-out window 21).

Names below for hind_loss (`h4hind`); `h4mid`, `h4front` analogous. All dirs under `data/counterfactual_walks/`.

### 9.1 Adaptation data (train / val): babbling

| | 4-leg babble | B1 reference (eval_suite `hexonly`) |
|---|---|---|
| files | `h4hind_babble_train` **48** windows, `h4hind_babble_val` **24** windows, 66 frames each | `b1_clips_train` 48 / `b1_clips_val` 24, 66 frames |
| adaptation budget | `wm.adapt --clips 44 --test_clips 4` on the 48 train windows = **44 x 66 = 2,904 frames (145 s)** | same flags, `--stratify` (see 9.3) |
| projector fit | 48 train windows (fit_projector's own clip-level 20 % val) | 48 train clips |
| rooms | train window k -> seed k (0-47), val window k -> 100 + k (100-123); one room per file | train 2i+k, val 100+i |
| source | train: 16 episodes, val: 8 episodes, disjoint seeds; **no episode in two splits** | windows of one long walk per behaviour, shared across splits |

- **Episode**: 20 frames settle (holding the first pose, `DRIVE_KW warmup=20`) + 290 frames babble; windows = frames
  [20 + 66j, 86 + 66j), j = 0..3; first 48 (train) / 24 (val) usable windows in (episode, j) order are kept, the rest
  archived. Usable = no frame of the window, nor the 10 after it, is fallen by the feasibility rule (CoM z < 0.5 H6 or
  tilt > 45 deg, held 10 frames; H6 = 0.1422 m). A fall ends the episode (offline truncation = reset: the next episode
  restarts from spawn; nothing after a fall is used). Gate: >= 48 / 24 usable windows, else collect more episodes
  with new seeds (never re-use or hand-pick).
- **Rooms**: babble has no behaviours, so no behaviour<->room link can exist; rooms are assigned by window order only.
  Same seed sets as c10 / B1 (the pretrained model has seen rooms 0-47 with c10 / B1 behaviours; 100-123 / 200-223
  never). Unlike c10 train rooms (one behaviour each, F307), a babble window contains 2-3 command segments, so its room
  is not a label of its motion. Check: |Pearson r| between room seed and window-mean Froude reported (no rule needed).
- **Babble generator** = `four_leg_feasibility.sample_segment` / `babble_plan` (CPG per leg on a shared clock,
  20-40-frame segments, 4-frame cross-fade, per-leg gain U(0.7, 1.3), phase pattern {tripod, trot, random},
  joint offsets N(0, 0.05), OU command noise 0.03 rad / tau 5; recorded action = the executed, perturbed command;
  `clean_actions`, `segment`, segment metadata stored), with the feasibility report's changes:

  | knob | feasibility | plan |
  |---|---|---|
  | family odds fwd / bwd / turn / side / free | 0.2 each | 0.15 / **0.30** / 0.20 / 0.20 / 0.15 |
  | bwd pace (cycles / 66 fr) | 3.0-5.5, lead 0.75 | **3.0-9.5**, lead 0.75, a1/a2 U(0.2, 0.3) |
  | turn spin | +-0..0.7 | **+-0..1.0** |
  | side strafe | L -1.5..-0.4, R 0.4..1.0; spin per side | **both signs symmetric**: sign +-1, abs U(0.4, 1.5), spin N(0, 0.15) |
  | falls | measured only | episode ends at the fall (reset), windows by the rule above |

  Pilot gate before collection (physics only, 16 episodes, as the feasibility run): falls <= 1.5 per upright minute,
  usable fraction >= 0.8, and coverage >= 20/24 c10 targets within 0.05 with some segment at fwd <= -0.15. If backward
  stays out of reach, the knobs are not tuned further: the uncovered goals are reported as such (decision D2).
- **Rendering**: physics first (scene loaded once, `scene_reuse`), then each window rendered by replay in its room with
  the same re-centring / spawn convention as c10 (window's first frame at the room centre). `render_hex_replay.py` has
  no ghost-leg support (only `collect_ik.py:98/671`, `render_leg_loss_walk.py` do): add `remove_legs` (removed legs'
  shapes hidden) and gate: replay render of a live-rendered window reproduces its frames (pixel corr ~1.0, as section 0.1).
- **Fields** (the hexapod reader `wm/data/embodiment.py:139-159` needs `actions`, `forces`, `head`, `body_quat`,
  `morph`, `expert_episode`): those + `com_pos`, `cam_pose`, `froude_height` (= H4), `segment`, `room_seed`, `dt`,
  `variant`, `legs`, `babble_episode`, `window_start`, `clean_actions`, `noise_seed`, `segments` (json). No
  `condition` field (nothing may read one; see 9.3).

### 9.2 Heldout test set (rooms 200-223), built like c10

| part | 4-leg | c10 |
|---|---|---|
| (a) selection library | `h4hind_clips_heldout`: **24** clips x 66 fr, probe command j in room 200 + j | `c10_clips_heldout` 24 |
| (b) branches | `h4hind_branches_heldout`: 24 clips x 3 branch points x **24 probe commands = 1,728**, 31 frames (10 + branch + 20), all 576 pairs | `c10_branches_heldout` 1,728 |
| (c) goals | the 24 `c10_clips_heldout` clips (hexapod goals, as for every body) | same |

- **Probe commands (K = 24)**: a constant babble knob set (one segment held, per-leg gain / phase / offset fixed, no OU
  noise) = the 4-leg analogue of a c10 behaviour. Chosen from a **pilot pool** of 200 sets drawn with the plan's babble
  distribution under a seed disjoint from all train / val episodes, each run 150 frames on the body. **Chosen from the
  new body's own reachable motion, never by reference to c10 or to the goals** (user, 2026-10-04: matching test commands
  to the seen body's behaviours is circular -- it shapes the test library around the goals and inflates the oracle and
  the selection score): k-means (k = 24) on the fall-free sets' standardised measured mean Froude (frames 20..150), one
  set per cluster = the member nearest the centroid. The chosen 24 and their Froude are stored (`h4hind_probes.json`);
  the distance from each c10 goal to the nearest probe is computed only afterwards and only reported (it shows which
  goals the body can reach, it never selects anything).
- **Clips + branches**: exactly `collect_c10_walks_and_branches.py` with the probe sets in place of the 24 beh24 plans
  and `remove_legs`: one long walk per probe, heldout window = the c10 heldout window position, 3 branch points from the
  CPG clock spread over gait phase (>= 10 frames before, 20 after), the 24 probe commands with the causal 4-frame
  cross-fade (`build_branches.py:65` FADE = 4); scene reuse, run classes, a probe's walk and its 72 branches on one
  instance. Gates as F305: every branch's state at the branch frame bit-identical to the walk; own-command branch == walk
  on all 31 frames (state, labels, pixels); no fall in any branch (rule above); labels after the branch independent of
  the prefix. Only the heldout split is built (no 4-leg branches are used for adaptation).
- Branch file fields as c10 (`cf_source`, `cf_t`, `first_pair` = 10, `segment`, `cf_command_index`, `cf_own`, ...),
  `cf_command_name` = `probe_jj` (cluster index; no c10 behaviour attached).

### 9.3 What the tools assume, and the minimal changes (c10 / B1 paths unchanged)

| where | assumption | change |
|---|---|---|
| `wm/data/embodiment.py:380-384` | REGISTRY = hexapod 18-D / b1 / gecko | add `h4hind`, `h4mid`, `h4front` = `Embodiment(name, 12, 4, reader)`; reader = `_hexapod` (same axes / `forward_axis("hexapod")`) with `contact` sliced to the 4 active feet and `body` = variant name |
| `counterfactual_readout.py:51, 75` | `choices=("hexapod","b1")`; action dim `18 if hexapod else 12` | `choices=tuple(REGISTRY)`, dim = `REGISTRY[e].action_dim` (identical result for hexapod / b1) |
| `wm/fit_projector.py:138` | fits only `("hexapod", hex_dir), ("b1", b1_dir)` | add repeatable `--extra NAME=DIR`; heads are independent (`ActionProjector.nets` ModuleDict, per-embodiment stats), so fitting `--hex_dir c10 --b1_dir "" --extra h4hind=...` leaves the hexapod head's task unchanged |
| `wm/adapt.py:53-60` | `--stratify` groups clips by `condition` | do **not** pass `--stratify` for babble: plain seeded permutation (lines 81-83) picks 44 adapt + 4 test of the 48 windows; no code change. `--anchor_froude` reads labels via `REGISTRY[args.embodiment]` (registry entry suffices) |
| `wm/policy/planner.py:51-76` | `load_candidates` groups by `condition_of`; falls back to the file name | none: with `per_condition=999` (selection_eval) every file is a candidate; library clips may carry `condition = probe_jj` |
| `selection_eval.py:88, 113` | goals grouped by `condition` | none: goals are c10 heldout clips (have it). `--embodiment h4hind` already free text |
| `selection_eval.py:100-107` | oracle / random bounds | **already label-free**: oracle = per decision step the candidate with the smallest true-Froude error to the goal (ground truth), random = mean error over all candidates; normalised score = (random - selector) / (random - oracle). Unchanged; report per-goal oracle error too, so unreachable goals (backward) are visible |
| `wm/evaluate.py:66` `offset_for` | `offsets[embodiment]` KeyError if a checkpoint was trained with `center_embeddings` | current runs have it off (`wm/config.py:63`); evaluate only such checkpoints, refuse otherwise |
| `wm/models/ftm.py:75-78` | per-embodiment FTM token raises for an unknown body | off in current runs (`wm/config.py:62`); same rule |
| `scripts/run/eval_suite.sh` | B1 / c08 / c10 only | **untouched**. New `scripts/run/eval_newbody.sh NAME VARIANT PT` mirroring its `hexonly` branch: `wm.adapt --data h4hind_babble_train --embodiment h4hind --clips 44 --test_clips 4 --lambda_hinge 0.5 --hinge_margin 0.1 --lora_rank 8 --steps 3000 --anchor_froude 1.0`; `fit_projector` (c10 + `--extra`); merge; selection (recorded goal, vision-read goal, w = 21) on `h4hind_clips_heldout`; read-out `--pairs 1 11` on `h4hind_branches_heldout`; plus the same with projector only (no LoRA) as the "no adaptation" row |
| `render_hex_replay.py` | six visible legs | `remove_legs` option + replay-vs-live gate (9.1) |
| `check_splits.py`, `tests/test_froude_labels.py` | c10 / B1 / c08 dirs | add the 4-leg dirs (`FROUDE_TEST_HEX=h4hind`) |

### 9.4 Checks before use, cost

1. Splits: no episode / frame in two splits; heldout probe walks separate from babble; rooms train 0-47 / val 100-123 /
   heldout 200-223 exactly once each (`check_splits.py`).
2. Determinism (heldout): F305 gates (branch start bit-identical, own-command == walk on every field, all 576 pairs).
3. Labels: `tests/test_froude_labels.py` on the new dirs; `froude_height` = H4 in every file; segment-aware labels after
   a switch independent of the prefix; CoM collector vs `wm.data.com` with the removed legs' mass included.
4. Falls removed: no fallen frame (rule) in any kept window, clip or branch; max tilt and min CoM z reported.
5. Coverage: babble pilot gate (9.1); per c10 goal, nearest probe distance and oracle error; babble train Froude
   histogram vs c10 targets (as `coverage_<variant>.png`).
6. Rendering: replay == live gate; ego view checked (FOV 90, `check_ego_view`); removed legs invisible in the ego view.
7. **Human review videos**: 6 babble windows (ego + third person), the 24 probe clips as a grid labelled with Froude
   and the matched c10 behaviour, one branch group (same state, 6 commands side by side), one room per split.

Cost per variant (CPU only, <= 6 own CoppeliaSim instances, load < ~12, no GPU until evaluation): babble pilot ~15 min;
babble 24 episodes physics ~30 min + render 72 windows ~15 min; probe pool 200 x 150 frames ~30 min; probe walks +
1,728 branches with scene reuse ~2-3 h (24 probes / 6 instances, ~72 branches each instead of c10's 288) + render ~1 h;
checks + videos ~30 min. **~5 h per variant**, hind_loss first; middle / front only after hind_loss's gates and review pass.
Evaluation (GPU, one job): adapt 3000 steps + projector + selection + read-out ~ as `eval_suite` hexonly B1 part.

### 9.5 Open decisions (recommendation first)

| | decision | recommended |
|---|---|---|
| D1 | probe commands for library + branches | **decided (user): from the body's own babble only** — k-means on its reachable Froude, one per cluster; never matched to c10 or the goals |
| D2 | goals the body cannot reach (hind: backward) | keep all 24 goals (same test as every body); report per family and with the oracle, which shows the ceiling |
| D3 | base checkpoint | the hexapod-only checkpoint (arm H; B1 and 4-leg both unseen), the joint round-1 B checkpoint as second row |
| D4 | baseline rows | projector only (no LoRA), and LoRA + projector (current recipe); both on the same 44 windows |
| D5 | babble windows contain segment switches | yes (that is babbling; labels / pairs are segment-aware) vs constant-command windows |
| D6 | Froude height | one constant H4 per body (babble postures vary, a per-clip median would rescale clips differently) |
| D7 | budget | 44 windows = B1's; a budget sweep (e.g. 11 / 22 / 44 / 88 windows, 4-panel video per point) only after the 44 result |
| D8 | order | hind_loss end-to-end incl. evaluation before collecting middle / front |

**Claims rule (user, 2026-10-04).** c10 is the pretraining (seen) body: its numbers are a sanity reference, never evidence
for transfer. Claims rest on bodies the model was not pretrained on -- c08 (zero-shot), B1 under hexapod-only pretraining
+ adaptation, the 4-leg variants. Nothing in a new body's test set may be chosen using the seen body or the goals. Goals
from c10 heldout clips are the task (imitate another body's demonstration) and stay; a second goal source (B1 heldout) can
be added to show the result does not depend on the hexapod's goals.

## 10. B1 as a new robot: babbling adaptation, expert library as upper bound (user, 2026-10-04)

The claim is driving a robot with **no prior knowledge** of it. B1's existing clips use commands tuned to the hexapod's
Froude (knowledge of the task), so they cannot be the adaptation data for that claim. After hexapod-only pretraining:

| row | adaptation data | candidate library (heldout rooms 200-223) | role |
|---|---|---|---|
| realistic | B1 babbling (train / val rooms), labels = measured CoM Froude | B1 babbling clips, 24 picked by k-means on B1's own reachable Froude (never by c10 or the goals) | the claim |
| upper bound | the same adapted model | `b1_clips_heldout` (tuned behaviour clips, "expert" walking-policy commands) | cost of babbling candidates vs the best |
| reference | none (B1 in joint pretraining, round-1 arm B) | `b1_clips_heldout` | ceiling of the method |

B1 babbling: random walking-policy commands (vx, vy, wz) with switches every 1-2 s over the policy's safe range, exact
MuJoCo, same window length (66 frames), budget and room seeds as the 4-leg babble (9.1). The hexapod expert CSV stays
banned; "expert" here means only B1's own tuned walking-policy clips.

**Correction 2026-10-08 (F325):** these "babbling" commands go through B1's TRAINED walking policy, so the B1 rows are
"no task knowledge", not "no prior knowledge". The claim test is the four-legged hexapod with CPG babbling (section 9).


## 11. Corrections and next data plan (2026-10-08, F323-F326)

**What was wrong**
1. **One room per clip, in every room setup** (original rooms sized to the body and `render_random_room.py` random-size
   rooms alike): each room holds one behaviour, a cue the model can use. Section 3's "several rooms per start state" was never
   built. Corrected 2026-10-09 (F324): this is NOT what separates B1's +0.70 (rooms sized to the body) from +0.44 (random-size
   rooms) -- both have one colour room per clip; only room size and start position differ.
2. **View randomisation** (Egocentric VSM strength, planned 2026-10-01) never applied: pretraining still crop 0.85-1,
   brightness/contrast +-0.2, no hue / saturation / blur / noise, lighting fixed; adaptation has no augmentation at all.
3. **B1 babbling** uses the trained walking policy (F325).
4. **Hexapod physics loop** switches candidates without gait continuity (F323) - execution, not data, but it invalidates this
   week's hexapod physics numbers.

**Plan (one variable at a time)**

| step | change | question |
|---|---|---|
| A | B1 random-command adaptation in the ORIGINAL rooms (render babble at the body's room size, no rr) | with room size fixed, does it reach the tuned-clip level (+0.70)? |
| B | projector-only adaptation (ITM + FTM frozen) | is the adapted ITM what learns the room? (skipped by agreement 2026-10-08) |
| A2 | same 44 tuned clips, adaptation in 3 random-size rooms each (rr + rrv2 + rrv3), control 1 of 3 at random (`b1_rooms_adapt.sh`) | does room-size variety at adaptation close +0.44 -> +0.71? |
| C | multi-version rooms: v2 / v3 of every training clip + branches (rendered 2026-10-08), joint retrain, same steps | does z drop the room (STATUS criteria a-d)? |
| D | adaptation data with many commands per room, rooms shared across clips; colour augmentation during adaptation | does adaptation in random rooms approach +0.71? |
| E | lighting / texture sets / Egocentric VSM augmentation strength | does selection hold in unseen scenes? |
| F | four-legged hexapod, CPG babbling only (no policy), test library from its own babble (k-means), never from c10 / goals | the no-prior-knowledge claim |

Physics loop fix first for any hexapod physics number: CPG recipe on one gait clock + cross-fade at switches, as the branches
(`collect_ik.cpg_commands(xfade=...)`), verified by single-clip replay and c10 direct on turn / speed goals.
