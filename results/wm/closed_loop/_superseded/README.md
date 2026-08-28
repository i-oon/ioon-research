# Superseded closed-loop runs

Kept rather than deleted because several are the *evidence* for a defect, and a reader who finds a
number quoted in an older FINDINGS entry should be able to reach the run it came from. **Nothing
here should be used for a new comparison.**

| directory | why it was replaced |
|---|---|
| `b1_physics`, `b1_physics2`, `video_b1_physics*` | the camera followed the robot; every clip the model trained on uses a camera placed once (F101). `b1_physics` also started from a standing pose. Superseded by `b1_physics3` |
| `kine_adapted`, `kine_nce` | the kinematic loop, which cannot fall and reports survival by construction. Superseded by the physics loop once F101 showed it was available |
| `det_A`, `det_B` | determinism probes for the kinematic loop |
| `det_b1_A`, `det_b1_B` | determinism probe that established MuJoCo repeats bit for bit (F105) |
| `hsweep_h*` | planner-horizon sweep run before the camera defect was found |
| `hex_side_c*` | single-run commitment sweep whose apparent trend F105 showed to be variance |
| `hex_commit1`, `hex_commit3` | one run per setting; F105 measured the spread of a repeated hexapod configuration at 37-71%, so a single run cannot carry a comparison. Superseded by `hex_rep5` and `hex_c3_rep5` |
| `c1`, `c5`, `c10`, `w0`, `w5`, `w10` | commitment and warm-start sweeps on the **trained** body, from before the repeatability of CoppeliaSim physics was known. One run each |
| `kinematic`, `clean_R1`, `clean_R2` | the kinematic B1 loop, which poses the body and cannot fall. Superseded once F101 showed a physics loop was available |
| `video_heldout`, `video_heldout_zeroshot` | rendered from run directories whose correspondence to `heldout` and `heldout_fewshot` could not be established from their names. Re-rendered as `video_heldout_fewshot` |
| `video_b1_physics3` | rendered from the `--commit 1` runs; `video_b1_commit3` is the configuration now reported |

**Current runs live one level up**: `b1_commit3` (B1, camera fixed and `--commit 3` -- the best
configuration measured, speed 2/3), `b1_physics3` (the same with `--commit 1`, speed 0/3), `hex_rep5` and `hex_c3_rep5`
(hexapod, five repeats each), `hex_side_fixed` (hexapod sideways after the dataset correction),
`heldout_fewshot` (F95).
