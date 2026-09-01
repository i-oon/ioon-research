# beh10_c10f10t10_intent2_flat

**`data/beh12_c10f10t10_intent2_flat` minus the two conditions the physical gate rejected**, as
symlinks. `side_L_stopmid` and `side_R_stopmid` read 0.34x and 1.21x the steady arm's own envelope
(`scripts/dataset/check_within_clip_intent.py`), so the stop never became visible in the body
coordinate.

**That is a property of the sideways recipe, not of the scheduler.** It under-drives this body --
`side_L_lvl0` walks at Froude 0.016-0.019 in the clean arm -- and **a stop is invisible in a robot
that is barely moving**. The other ten conditions clear the gate at 2.1x to 5.6x.

Ten conditions, 20 clips: four speed, four turn, two sideways. **The axes are deliberately unbalanced
and `merge_behaviour_dirs` would reject this**, which is why it is built as links rather than
collected. Per-family rows for `side` rest on two conditions and should be read as such.
