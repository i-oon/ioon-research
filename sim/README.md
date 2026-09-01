# sim

CoppeliaSim + MuJoCo side of the pipeline: build scenes, collect data, render/replay, and the
one diagnostic that predates the `wm/` measurement toolkit. Run everything from the repository
root with `.venv/bin/python3`, never bare `python3`, matching `scripts/README.md`'s convention.

```
sim/
  scene/         build or modify a .ttt scene: camera, floor, leg geometry, the B1 scene
  collect/       record a dataset: IK retargeting, the Step 0 pilot, MuJoCo/AIRL rollouts
  render/        replay commands through physics, watch/export video, one gait diagnostic input
  diagnostics/   step_minus1_morphology_gap.py, predates scripts/diagnostics/
  env/           the .ttt scene files and heavy CSVs everything above reads and writes
  assets/        URDF/meshes (B1 description) and policy checkpoints scene-builders import
  _archive/      superseded, not expected to run — includes coppeliasim_env.py, never imported
                 by anything and unused since the initial commit
  SOURCES.md, environment.yml   reference docs, not scripts
```

## scene/

| | |
|---|---|
| `add_camera.py` | Add the fixed `vjepa_cam` vision sensor to a Medauroidea scene variant. |
| `build_b1_scene.py` | Build `sim/env/b1_flat.ttt`: fresh scene + floor + imported B1 + fixed camera. |
| `make_leg_morphology.py` | Generate a leg-length/segment-scale variant of the base stick insect. Refuses to generate a body that violates the reach constraint. |
| `match_b1_camera.py` | Copy the insect scene's camera onto the B1 scene so both embodiments render alike. |
| `set_floor_texture.py` | Replace the checkerboard floor with a V-JEPA2-friendly surface. |

## collect/

| | |
|---|---|
| `collect_ik.py` | Step A collector: IK-retargeted dataset with the fixed camera. The source of every `ik_walk_*` dataset. |
| `collect_step0.py` | The Step 0 pilot dataset: walk episodes from all 3 original morphologies. |
| `rollout_b1_mujoco.py` | Native-MuJoCo rollout of the B1 policy to a trajectory `.npz`, no ROS. |
| `rollout_insect_airl.py` | Roll out a trained AIRL insect policy in CoppeliaSim, capture frames. Parked alongside `scripts/amp/`. |

## render/

| | |
|---|---|
| `render_wm_prediction.py` | Drive the robot with world-model predicted joint commands next to the IK ground truth, open loop. Feeds `scripts/diagnostics/wm_gait_report.py`. |
| `render_b1_replay.py` | Render a native-MuJoCo B1 trajectory in CoppeliaSim (kinematic replay) to a dataset `.npz`. |
| `play_ik.py` | Watch the IK-retargeted gait in the CoppeliaSim GUI, no recording. |
| `npz_to_video.py` | Export recorded `.npz` episodes as watchable `.mp4` (+ a grid overview). |
| `render_leg_loss_walk.py`, `render_leg_loss_preview.py` | Walking/preview renders of the 4-leg leg-loss variants (the Stage 2 held-out embodiment). |
| `render_bumpy_morph_walk.py`, `render_terrain_morph_walk.py`, `render_original_uneven_morph_walk.py`, `render_uneven_scene_walk.py` | Terrain-robustness renders from an earlier, now-inactive branch of work. Kept because nothing has archived them yet — treat as lower-confidence than the rest of `render/`. |

## diagnostics/

`step_minus1_morphology_gap.py` — the Step -1 morphology gap check. **Its long-body figure does not reproduce** (4.125 documented, 4.404 recomputed; see the note in `doc/FINDINGS.md`), and it is still quoted in `doc/PROGRESS.md` and `sim/SOURCES.md`. Predates `scripts/diagnostics/`; the newer measurement toolkit lives there instead.

## Do not redefine scene/body paths here

Scene `.ttt` paths are mostly hardcoded absolute paths under `sim/env/`, not relative to a
script's own location — moving a script does not change what scene it points to. Body validity
(which bodies walk, which are excluded) is decided in `wm/bodies.py`, not here; see
`scripts/README.md`'s "Do not redefine these" section.
