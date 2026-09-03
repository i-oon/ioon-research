# The insect clips stage 3 fits on, and the 24 it must not see.
#
# **Same rule as the B1's** (`b1_stage3_clips.sh`): two of each of the twelve conditions, the
# `...1` and `...2` of each block, so all twelve behaviours are covered evenly. The `...0` and
# `...3` of each block are held out -- which includes `hexapod_ep100`, the clip F142 takes its
# goal and its `D_real` reference from, so the teacher never trains on the episode it is graded
# against.
#
# `wm/adapt3` reads its training clips from a stage-1 checkpoint's own record when it has one. A
# **pretrain** carries no such record, and passing one without `--train_clips` fails with "no
# training clips" -- which is what the first attempt at `com7_stage3_hexapod.sh` did.
HEX_CLIPS="
  hexapod_ep2001.npz hexapod_ep2002.npz hexapod_ep2101.npz hexapod_ep2102.npz
  hexapod_ep2201.npz hexapod_ep2202.npz hexapod_ep2301.npz hexapod_ep2302.npz
  hexapod_ep1.npz hexapod_ep2.npz hexapod_ep101.npz hexapod_ep102.npz
  hexapod_ep201.npz hexapod_ep202.npz hexapod_ep301.npz hexapod_ep302.npz
  hexapod_ep1001.npz hexapod_ep1002.npz hexapod_ep1101.npz hexapod_ep1102.npz
  hexapod_ep1201.npz hexapod_ep1202.npz hexapod_ep1301.npz hexapod_ep1302.npz
"

HEX_HOLDOUT="
  hexapod_ep2000.npz hexapod_ep2003.npz hexapod_ep2100.npz hexapod_ep2103.npz
  hexapod_ep2200.npz hexapod_ep2203.npz hexapod_ep2300.npz hexapod_ep2303.npz
  hexapod_ep0.npz hexapod_ep3.npz hexapod_ep100.npz hexapod_ep103.npz
  hexapod_ep200.npz hexapod_ep203.npz hexapod_ep300.npz hexapod_ep303.npz
  hexapod_ep1000.npz hexapod_ep1003.npz hexapod_ep1100.npz hexapod_ep1103.npz
  hexapod_ep1200.npz hexapod_ep1203.npz hexapod_ep1300.npz hexapod_ep1303.npz
"
