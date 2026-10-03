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

