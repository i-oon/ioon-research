# The 24 B1 clips stage 3 fits on, and therefore the only clips stage 1 may adapt on.
#
# **Sourced by both sheets so the two cannot drift apart.** `wm/adapt3.py` splits the 48-clip
# directory into these 24, twelve candidates -- the first clip of each condition in sorted order,
# which is the library the planner picks from -- and twelve validation clips. Stage 1 permuting the
# whole directory at seed 0 lands on two of the candidates and three of the validation clips, so
# the forward model would arrive at stage 3 having already seen part of what stage 3 scores it on.
#
# Two of each condition, the `...1` and `...2` of each block: twelve behaviours covered evenly.
CLIPS="
  b1_ep1.npz b1_ep1001.npz b1_ep1002.npz b1_ep101.npz
  b1_ep102.npz b1_ep1101.npz b1_ep1102.npz b1_ep1201.npz
  b1_ep1202.npz b1_ep1301.npz b1_ep1302.npz b1_ep2.npz
  b1_ep2001.npz b1_ep2002.npz b1_ep201.npz b1_ep202.npz
  b1_ep2101.npz b1_ep2102.npz b1_ep2201.npz b1_ep2202.npz
  b1_ep2301.npz b1_ep2302.npz b1_ep301.npz b1_ep302.npz
"

# The other 24: twelve candidates and twelve validation clips. **Stage 2 must not fit on
# these.** The projector it produces is only stage 3's starting point, and stage 3 retrains it
# for 15k steps at lr 1e-3 on a different objective, so the leak is faint -- but "faint" was
# never measured, and excluding them costs one flag. Names are given in full, with `.npz`:
# `fit_projector --exclude` matches by **prefix**, and `b1_ep100` is a prefix of `b1_ep1000`
# through `b1_ep1003`, so a bare stem silently drops four extra clips.
HOLDOUT="
  b1_ep0.npz b1_ep100.npz b1_ep1000.npz b1_ep1003.npz
  b1_ep103.npz b1_ep1100.npz b1_ep1103.npz b1_ep1200.npz
  b1_ep1203.npz b1_ep1300.npz b1_ep1303.npz b1_ep200.npz
  b1_ep2000.npz b1_ep2003.npz b1_ep203.npz b1_ep2100.npz
  b1_ep2103.npz b1_ep2200.npz b1_ep2203.npz b1_ep2300.npz
  b1_ep2303.npz b1_ep3.npz b1_ep300.npz b1_ep303.npz
"
