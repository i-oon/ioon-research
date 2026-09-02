#!/usr/bin/env bash
#
# STEP 2's gate: do the action-emitting components still work in egocentric?
#
#   bash scripts/step2_gate_projector.sh     # wherever BOTH checkpoints are; needs a GPU
#
# **This runs before teacher-student, not alongside it.** GATE C (F172) showed the forward model now
# uses the action, which is what a teacher needs. But teacher-student, the action projector and every
# path that emits a joint command run through two components this run has not measured egocentric:
#
#   MotionDecoder   `(e_t, z) -> a`.  Trained on egocentric it reaches 0.076 on train motion and
#                   never leaves 1.53 on validation -- above 1.0, worse than predicting the mean.
#   ActionProjector `a -> z`.  Never fitted on an egocentric checkpoint at all.
#
# **Building a student on either of those without measuring them first is how F123, F126 and F128
# were built.** Both arms of every comparison are run here in the same pass, on the same settings,
# so no number is quoted from an older run against different data.
#
#   GATE D1  the decoder's ceiling. A dual ridge on the decoder's own input, split by clip. If the
#            ridge recovers the command and the trained head does not, the head overfits and is
#            repairable. If the ridge cannot either, the command is not recoverable from an
#            egocentric frame plus `z`, and nothing that emits a joint command can be built on it.
#   GATE D2  the projector, fitted and scored egocentric against the allocentric arm. **`rollout
#            gap` is the number**, not `z MSE` -- below 1.0 is better than knowing nothing.
#
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python3
EGO=wm/runs/beh12_ego/best.pt
ALLO=wm/runs/beh12_hex-b1_body3/best.pt
EGO_HEX=data/egocentric/beh12_c08f09t09_ego_flat
EGO_TRAIN_HEX=data/egocentric/beh12_c10f10t10_ego_flat
EGO_B1=data/egocentric/beh12_b1_ego_flat
ALLO_HEX=data/allocentric/beh12_c08f09t09_flat
ALLO_TRAIN_HEX=data/allocentric/beh12_c10f10t10_flat
ALLO_B1=data/allocentric/beh12_b1_flat

for f in "$EGO" "$ALLO"; do
  [ -f "$f" ] || { echo "missing $f -- both arms must be present or this measures one thing twice"; exit 2; }
done

echo "############ GATE D1 -- is the command readable from what the decoder is shown? ############"
#
# **Three columns per body, and the verdict is not on the last one.** Egocentrically `e_t` is
# *expected* to fall -- Q1 is single-frame action R2 0.779 to 0.293, and that fall is the thing that
# made the forward model use the action at all. So `[e_t, z]` landing under the allocentric value is
# not by itself a failure. **The question is whether the burden shifted to `z`**: F168 has `z`
# carrying the action, and if `z`-only holds near its allocentric level while `e_t`-only drops, the
# command is still in what the decoder is shown and the head only has to be refitted to read it from
# `z` rather than from the frame.
#
#
# **The allocentric hexapod arm is already measured** -- e_t 0.773, z 0.903, [e_t, z] 0.938, on
# `beh12_hex-b1_body3` against held-out `c08f09t09`. `e_t` at 0.773 reproduces F159's 0.779 from a
# different script, which is what says the instrument is sound. It is passed to the egocentric
# hexapod arm as `--reference` rather than re-run. **The allocentric B1 arm is measured too** --
# e_t 0.166, z 0.790, [e_t, z] 0.789 -- and it does not resemble the insect's at all: the quadruped's
# pose never determined its command, and the frame adds nothing over `z`. Its egocentric arm has to
# be read against those numbers and not against the insect's.
# $PY scripts/diagnostics/motion_decoder_ceiling.py --ckpt "$ALLO" --data "$ALLO_B1" \
#     --embodiment b1 --cache results/wm/cache/fid_b1.pt

# `--bar` is the trained head's own R2 on the same scale, `1 - val_motion`. The egocentric run ends
# at val motion 1.53, so -0.53.
$PY scripts/diagnostics/motion_decoder_ceiling.py --ckpt "$EGO" --data "$EGO_HEX" \
    --embodiment hexapod --bar -0.53 --reference 0.773 0.903 0.938 \
    --cache results/wm/cache/ego_hex.pt
$PY scripts/diagnostics/motion_decoder_ceiling.py --ckpt "$EGO" --data "$EGO_B1" \
    --embodiment b1 --bar -0.53 --reference 0.166 0.790 0.789 \
    --cache results/wm/cache/ego_b1.pt

echo
echo "############ GATE D2 -- the action projector, egocentric against allocentric ############"
$PY -m wm.fit_projector --ckpt "$EGO" --hex_dir "$EGO_TRAIN_HEX" --b1_dir "$EGO_B1" \
    --cache results/wm/cache/proj_ego.pt --out wm/runs/beh12_ego/projector_ego.pt
$PY -m wm.fit_projector --ckpt "$ALLO" --hex_dir "$ALLO_TRAIN_HEX" --b1_dir "$ALLO_B1" \
    --cache results/wm/cache/proj_allo.pt --out /tmp/projector_allo.pt

echo
echo "############ how to read it ############"
echo "PASS  z-only holds near its allocentric level even though e_t-only fell, so the command"
echo "      survives in what the decoder is shown and the head has to be refitted to read it from z"
echo "      rather than from the frame -- and the egocentric rollout gap is within reach of the"
echo "      allocentric arm s. Teacher-student opens, with the decoder refitted first."
echo "FAIL  e_t-only AND z-only both fall. Neither input carries the command, no refit recovers it,"
echo "      and teacher-student needs rethinking rather than repairing."
echo
echo "Do NOT judge on [e_t, z] against the allocentric 0.938 alone: e_t is SUPPOSED to have fallen,"
echo "so that column reads the intended change as a defect."
