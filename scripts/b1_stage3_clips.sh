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
