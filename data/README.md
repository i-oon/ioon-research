# Datasets

Recorded clips. Each `.npz` holds `frames` (T, 256, 256, 3), the commands, foot forces and the
body pose, plus scalar fields naming the body and the source expert episode.

**Nothing here is regenerable without CoppeliaSim**, so treat it as evidence rather than cache.
`results/wm/cache/` is the disposable one.

## Which set to use

| directory | what it is | used by |
|---|---|---|
| `ik_walk_8body` | the base hexapod set, one speed, 9 bodies | Stage 1 and the original Stage 2 |
| `ik_walk_m3d_clean`, `ik_walk_cov_*`, `ik_walk_bracket` | **symlinks into `ik_walk_8body`**, selecting subsets | Stage 1 experiments |
| `ik_walk_speed5` | 5 constant speeds, 67 clips | the first `lambda_body` pair |
| **`ik_walk_speed7`** | **5 constant + 2 ramped, 91 clips** | **current Stage 2 work** |
| `b1_framed` | the quadruped, 14 clips, 2 policies x 7 speeds | every cross-embodiment run |
| `ik_4leg_c08f09t09_clean10` | 4-leg build cut from a **held-out** body | slide 15 |
| `ik_4leg_middleloss_clean9` | 4-leg build cut from a **training** body | superseded, see F59 |

**The symlink directories look empty at 4.0K and are not.** Deleting `ik_walk_8body` breaks five
datasets at once, including everything Stage 1 in the deck reports.

## The speed sets, and why they exist

The expert is a real stick insect walking **one speed** -- 1.9 percent variation across 1,000
episodes. That made every body-level question unanswerable: a readout fitted on the B1 learns
commanded speed, one fitted on the insect learns how far the body rocks within a stride, and asking
one to transfer to the other is not a question with an answer (F57).

`collect_ik.py --speed` resamples the shared foot path along time -- same path, fewer frames, so
the robot covers the same ground faster. **Every leg is resampled by the same time map**, so the
inter-leg phase relationships are untouched and the gait stays the animal's.

`--speed_end` sweeps the speed *within* a clip. That exists because five constant speeds gave the
shared decoding head 12 distinct values to memorise rather than a function to learn (F58). Ramping
made the target continuous and moved `insect->b1` from unstable to positive (F60).

Episode numbers carry the condition as a block of a thousand -- `_ep2020` is block 2, source
episode 20 -- so cross-body pairing only ever happens within one condition. A 92-frame clip and a
60-frame clip must never both claim to be "episode 6".

## Two collection flags that are not optional

`--cam_dx -0.6 --spawn 0 0` are the defaults now, and were not for a long time. Without them the
robot starts outside the right image edge and **56-70 percent of frames are clipped**, unequally
per body, so morphology decodability partly measures framing.

`--morphs NAME=SCENE` must be passed for the `cXXfYYtZZ` bodies. The built-in `SCENES` list is
still the old three-length set, so omitting it silently collects bodies nothing else uses.
