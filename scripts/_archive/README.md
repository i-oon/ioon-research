# scripts/_archive

Moved here 2026-10-12 (user: clean up, but never delete scripts). Each file keeps its original relative path
(`scripts/_archive/<dir>/<file>` was `scripts/<dir>/<file>`). Criterion: the script belongs to a dropped or superseded
direction AND no active script in `scripts/`, `wm/` or `sim/` imports or calls it (checked by grep before moving).

| original folder | files | direction |
|---|---|---|
| dataset/ | collect_b1_babble, collect_b1_sbabble, collect_babble_b1, collect_babble_hex | babbling (paused 2026-10-09) |
| dataset/ | render_random_room, render_multiversion | random-size / multi-version rooms (dropped 2026-10-09) |
| dataset/ | four_leg_feasibility | 4-leg body adaptation pilot (paused; DATA_PLAN section 9) |
| dataset/ | build_beh24_b1_ego_flat, build_beh24_hex_ego_flat, make_clean_split_beh24, check_beh24_froude_match | beh24 data era (superseded by counterfactual_walks) |
| figures/ | render_babble_review, render_babble_selection, plot_babble_goal_reading, plot_4leg_fewshot_and_ablation, make_stage2_diagram | figures of those directions |
| finished/ | collect_b1_babble_batch.sh, sweep_b1_coppelia_babble, refit_b1_babble_projector | babbling |
| diagnostics/objective_experiments/ | babble_library_eval, rl_imagination_isolation_v2 / _v3, switch_readout_check | babbling / RL-imagination / switching-data diagnostics |
| diagnostics/cross_embodiment/ | sweep_4leg_fewshot, align_head_only, align_window_mean, shared_in_align_space | 4-leg and z-alignment diagnostics |

**To run one again:** move it back to its original folder first -- scripts locate the repo root by their folder depth
(`os.path.dirname` chains), which is one level off inside `_archive/`. Their FINDINGS entries stay valid.
Older dropped run scripts are in `scripts/run/_archive/` (2026-10-09).

- `dataset/{collect_intent,collect_offrhythm,confirm_counterfactual,ego_derisk,noise_sweep}.sh` (moved 2026-10-15): September launchers for the intent / off-rhythm / egocentric de-risk / noise-sweep collections; their data is deleted (doc/DATA.md).
