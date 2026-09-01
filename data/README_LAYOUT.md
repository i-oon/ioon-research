# data/ layout

    allocentric/   every set collected before 2026-09-01: the fixed third-person camera
    egocentric/    head-mounted camera, randomised room

**The split is the camera, not the behaviour.** The same twelve conditions exist on both sides, and
a number measured under one view cannot be compared with a number measured under the other -- that
difference is the whole subject of F170.

**Moving these directories breaks symlinked sets.** `beh12_b1_flat_9clips` and
`beh10_c10f10t10_intent2_flat` are symlink farms into their parents, with absolute targets; the move
on 2026-09-01 broke 34 of them and they were repaired by rewriting the targets, not by re-collecting.
**A broken symlink reads as a missing file, several scripts deep.**

`data/allocentric/README.md` carries the naming rule and what each set is.
