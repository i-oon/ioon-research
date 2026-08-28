# beh12_b1_v2 — the corrected B1 set

Replaces `beh12_b1_flat`, which four separate defects made unusable for the turning half of the
matched behaviour set. **Do not mix the two.**

| | `beh12_b1_flat` | `beh12_b1_v2` | hexapod |
|---|---|---|---|
| frames touching an image edge | 61% (100% on every sideways clip) | **0%** | 0% |
| background spread between clips | 2.79 grey levels | **0.26** | 0.14 |
| weakest turn level | the forward clip relabelled -- identical in both channels | a real turn | -- |
| turn direction | **opposite to the insect** | matches | -- |
| policy recorded in the file | no | `policy` field | -- |

**Turning is matched on what the robots achieve, per policy.** The insect reaches -0.0072 /
-0.0241 / -0.0372 / -0.0878; this set reaches **-0.0083 / -0.0242 / -0.0371 / -0.0750**, three of
them within 1%. The two B1 policies need different commands for the same result -- at the weakest
level `sym` takes `--wz -0.023` and `gait3` `-0.081` -- so conditions are named for the achieved
rate and never for a command.

**The strongest level is 15% short and that is the robot, not the calibration.** Pushing `--wz`
from -0.475 to -0.664 moved yaw only -0.063 to -0.075 while forward Froude fell 0.117 to 0.097:
the quadruped buys rotation with forward speed and cannot hold 0.13 forward at the insect's hardest
turn. **The insect's `turn_s0.56` has no true counterpart here.**

Built by `scripts/dataset/rerender_b1_framing.py` (speed and sideways, re-rendered from the stored
states) and `scripts/dataset/recollect_b1_turns.py` (turning, re-rolled in MuJoCo). Camera:
`--cam_fov 24 --spawn 0 0 --floor_scale 3`. **A loop reading these clips has to use the same three**
or it plans on frames that differ from the ones it was adapted on.
