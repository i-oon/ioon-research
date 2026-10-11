#!/usr/bin/env bash
# Does room variety in the ADAPTATION data fix B1 rollout in random-size rooms? (2026-10-09)
# Base: round1_hexonly_s0_rr (hexapod only, with branches, random-size rooms). B1 adapted on the SAME 44 tuned clips
# (commands tuned to the hexapod's behaviours; picked exactly as b1_tuned_control.sh: --stratify, seed 0), LoRA r8
# 3000 steps (equal updates in every row), projector fitted on rr_b1_clips_train as in the reference. One variable:
#   one     each clip in its 1 random-size room (rr_b1_clips_train)                 = reference (+0.94 / +0.44 on 10-06)
#   three   each clip in 3 random-size rooms (rr + rrv2 + rrv3), 132 files
#   mixed   each clip in ONE of its 3 rooms, drawn at random (seed 0): same frames as `one`, rooms differ across clips
# Evaluation as b1_tuned_control.sh: tuned held-out library (random-size rooms), goals = rr_c10_clips_heldout, w = 21.
cd "$(dirname "$0")/../.."
BASE=${BASE:-wm/runs/round1_hexonly_s0_rr/best.pt}
ROWS=${*:-one three mixed}
PY=.venv/bin/python3; CW=data/counterfactual_walks; OUT=results/eval/b1_rooms_adapt
SEL=scripts/diagnostics/objective_experiments/selection_eval.py; RO=scripts/diagnostics/objective_experiments/counterfactual_readout.py
quiet() { grep -v -i "warn\|Loading weights\|sit still"; }
for ROW in $ROWS; do
mkdir -p $OUT/$ROW
{
echo "=== row $ROW  base=$BASE  $(date)"
$PY - "$ROW" "$BASE" "$OUT/$ROW/adapted.pt" <<'PYEOF' 2>&1 | quiet | grep -E "^adapt|^->|clips" | tail -4
import glob, os, sys
import numpy as np
import wm.adapt as A
row, base, out = sys.argv[1:4]
cw = "data/counterfactual_walks"
paths = sorted(glob.glob(f"{cw}/rr_b1_clips_train/*.npz"))
train, test = A.select_clips(paths, 44, 4, 0, True)          # identical to the reference's choice
names = [os.path.basename(p) for p in train]
dirs = {"rr": "rr_b1_clips_train", "v2": "rrv2_b1_clips_train", "v3": "rrv3_b1_clips_train"}
if row == "one":
    pick = [f"{cw}/{dirs['rr']}/{n}" for n in names]
elif row == "three":
    pick = [f"{cw}/{dirs[v]}/{n}" for n in names for v in ("rr", "v2", "v3")]
elif row == "mixed":
    v = np.random.default_rng(0).choice(["rr", "v2", "v3"], len(names))
    pick = [f"{cw}/{dirs[vv]}/{n}" for n, vv in zip(names, v)]
    print("mixed rooms per clip: " + " ".join(f"{n[3:-4]}:{vv}" for n, vv in zip(names, v)))
for p in pick:
    assert os.path.exists(p), p
print(f"adapt clips {len(pick)} ({len(set(names))} distinct motions), test clips {len(test)}")
A.select_clips = lambda *a, **k: (pick, test)
sys.argv = ["wm.adapt", "--ckpt", base, "--data", f"{cw}/rr_b1_clips_train", "--embodiment", "b1", "--clips", "44",
            "--test_clips", "4", "--lambda_hinge", "0.5", "--hinge_margin", "0.1", "--lora_rank", "8", "--steps", "3000",
            "--anchor_froude", "1.0", "--out", out]
A.main()
PYEOF
$PY -m wm.fit_projector --ckpt $OUT/$ROW/adapted.pt --hex_dir $CW/rr_c10_clips_train --b1_dir $CW/rr_b1_clips_train \
    --cache_dir results/wm/cache/fitproj_files --out $OUT/$ROW/projector.pt 2>&1 | quiet | tail -1
$PY - "$OUT/$ROW" <<'PYEOF'
import sys, torch
from wm.models.action_projector import action_dims_from
o = sys.argv[1] + "/"
ck = torch.load(o + "adapted.pt", map_location="cpu", weights_only=False)
sv = torch.load(o + "projector.pt", map_location="cpu", weights_only=False)
ck["projector"] = sv.get("projector", sv); ck["action_dims"] = action_dims_from(sv)
torch.save(ck, o + "b1.pt")
PYEOF
echo "--- selection B1, tuned held-out library (random-size rooms)"
$PY $SEL --conditions all --candidates_dir $CW/rr_b1_clips_heldout --goal_dir $CW/rr_c10_clips_heldout \
    --cache results/wm/cache/test_v4_b1_heldout.pt --windows 21 --ckpt $ROW=$OUT/$ROW/b1.pt 2>&1 | grep -E "w=|bounds"
echo "--- read-out B1 (random-size-room held-out branches)"
$PY $RO --embodiment b1 --cf_dir $CW/rr_b1_branches_heldout --ckpt $ROW=$OUT/$ROW/b1.pt --pairs 11 2>&1 | quiet | grep -A3 "^==="
rm -f $OUT/$ROW/adapted.pt
} 2>&1 | tee $OUT/$ROW/summary.txt
sleep 300   # GPU cool-down between rows
done
