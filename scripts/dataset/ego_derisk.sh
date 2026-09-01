#!/usr/bin/env bash
#
# The egocentric de-risk gate: does a head view break the redundancy without losing the coordinate?
#
#   bash scripts/dataset/ego_derisk.sh       # CoppeliaSim GUI, exactly ONE instance
#
# **The cheapest thing that can answer the question, and nothing more.** Egocentric needs a world to
# look at, so four coloured textured walls get built around the spawn and the existing `vjepa_cam`
# is parented to the robot. **This is not an environment.** A polished one is worth building only
# after both questions pass.
#
# **STEP 0 IS A HARD GATE AND IT IS NOT OPTIONAL.** The camera pose is now derived from geometry --
# the insect's from `head - abdomen`, the B1's from the direction its base actually travels -- rather
# than from an axis convention, because the first version guessed and mounted the insect's camera on
# `/abdomen`, the **rear** segment.
# **A camera facing the wrong way still produces 66 frames, still passes every downstream script,
# and would answer Q1 with a resounding false pass**, because a view of the sky reveals no action
# either. F165 was voided by exactly this class of error.
#
# **Two questions, both must pass:**
#
#   Q1  does egocentric BREAK the single-frame redundancy?
#       third-person single-frame action R2 is 0.779 on the insect (F159). If a head view still
#       reads the command from one frame, the view change did not fix anything -- **report what
#       leaked**: legs in shot, a room simple enough to localise from, or the camera seeing the body.
#
#   Q2  does egocentric PRESERVE the shared cross-body coordinate?
#       fit forward/lateral/yaw on the insect's egocentric embeddings, test on the B1's **without
#       refitting**. Camera height and gait-induced bob differ enormously between a stick insect and
#       a quadruped, so "how the world moves" may simply not mean the same thing on the two bodies.
#
#   Q1 pass AND Q2 pass  -> the direction is alive and the full environment is worth building
#   Q1 pass,  Q2 fail    -> the world model is fixed and cross-embodiment is lost. **A trade-off,
#                           not a solution.** Report it; do not proceed on the strength of Q1
#   Q1 fail              -> the view change did not break the redundancy. Report the leak
#
# **Do not build the full environment and do not collect until both pass.**
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
OUT=data/allocentric/ego_derisk
VID=results/ego_derisk
BOX=6.0
MORPH=c10f10t10=medauroidea_c10f10t10.ttt
COMMON="--gait cpg --scale 0.65 --behavior walk --port 23000 --travel 0.8 --repeats 1"
mkdir -p "$OUT" "$VID"

echo "############ STEP 0 -- LOOK AT THE CAMERA BEFORE ANYTHING ELSE ############"
[ -d "$OUT/look_insect" ] || $PY sim/collect/collect_ik.py $COMMON --episodes 0 --morphs "$MORPH" \
    --cycles 7.1 --ego --ego_box $BOX --out "$OUT/look_insect"
$PY sim/render/npz_to_video.py --data "$OUT/look_insect" --out "$VID" || true
echo
echo "  **Watch $VID/ now.** The frame must show the room from the robot's head, moving as it walks."
echo "  If it shows the sky, the floor, or the robot's own body, fix --ego_euler / --ego_offset and"
echo "  rerun this step. Nothing below is meaningful until that frame is right."
echo "  the direction is MEASURED, not assumed: insect from head-minus-abdomen, B1 from the"
echo "  direction the base actually travels. The camera sits on /head (insect) and on the base"
echo "  (B1), 3 cm ahead and 2 cm up. Override with --ego_forward / --ego_offset if the frame"
echo "  still looks wrong -- and say what it looks like, since that names the fault."
echo
read -r -p "  does the frame look like a forward head view? [y/N] " OK
[ "$OK" = "y" ] || { echo "  stopping, as intended"; exit 1; }

echo
echo "############ COLLECT -- twelve conditions per body, egocentric ############"
[ -d "${OUT}/insect_flat" ] || {
  $PY scripts/dataset/collect_beh12.py --morph "$MORPH" --out "$OUT/insect_raw" --repeats 2 \
      --extra --ego --ego_box $BOX
  $PY scripts/dataset/merge_behaviour_dirs.py --src "$OUT/insect_raw" --out "$OUT/insect_flat" \
      --embodiment hexapod
}
echo "  B1: re-render the stored rollouts through the same room and head camera"
[ -d "$OUT/b1_flat" ] || $PY scripts/dataset/rerender_b1_framing.py --out "$OUT/b1_flat" \
    --extra --ego --ego_box $BOX || echo "  (if rerender_b1_framing has no --extra, call "\
"render_b1_replay.py --ego --ego_box $BOX per trajectory)"

echo
echo "############ GUARD -- is heading readable from the room's colour? ############"
echo "Q1 is not to be read until this is at chance. A room that still works as a landmark hands"
echo "back the single-frame pose readability the whole view change exists to remove, and Q1 would"
echo "report that leak as a property of egocentric views. The fix is more seeds."
$PY scripts/diagnostics/check_appearance_leak.py --data "$OUT/insect_flat" --embodiment hexapod
[ -d "$OUT/b1_flat" ] && $PY scripts/diagnostics/check_appearance_leak.py --data "$OUT/b1_flat" \
    --embodiment b1

echo
echo "############ Q1 -- is the action still readable from ONE egocentric frame? ############"
echo "  third-person baseline, F159: insect 0.779 single frame, 0.887 pair"
for SPEC in "hexapod $OUT/insect_flat ego_hex" "b1 $OUT/b1_flat ego_b1"; do
  set -- $SPEC
  [ -d "$2" ] && $PY scripts/diagnostics/inverse_dynamics_r2.py \
      --ckpt wm/runs/beh12_hex-b1_body3/best.pt --data "$2" --embodiment "$1" \
      --cache "results/wm/cache/$3.pt" --target action
done

echo
echo "############ Q2 -- does the shared coordinate still cross bodies? ############"
echo "  read the 'B1 unrefitted' row; the 'insect held-out' row only says the fit worked at all"
[ -d "$OUT/b1_flat" ] && $PY scripts/diagnostics/motion_rep_check.py \
    --ckpt wm/runs/beh12_hex-b1_body3/best.pt \
    --hex_data "$OUT/insect_flat" --b1_data "$OUT/b1_flat"

echo
echo "############ RENDER -- the same behaviour through both heads ############"
for C in speed_c7.1 turn_s0.29 side_R_lvl1; do
  A=$(ls "$OUT/insect_flat"/*.npz 2>/dev/null | head -1)
  B=$(ls "$OUT/b1_flat"/*.npz 2>/dev/null | head -1)
  [ -n "$A" ] && [ -n "$B" ] && $PY scripts/render/merge_counterfactual.py --a "$A" --b "$B" \
      --branch 1 --label_a "insect head" --label_b "B1 head" --out "$VID/heads_$C.mp4" || true
done
echo
echo "=== done  $(date '+%F %T')   send back this log and $VID/"
