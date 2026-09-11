#!/bin/bash
# Q21 follow-up: re-run the babble pipeline applying the margin-aware generation rule found in
# F194 (see memory `babble-generation-rule-target-pretrain-range.md` and
# `sim/collect/collect_b1_cpg_babble.py`'s own docstring note) -- F194 measured 19% of the v1
# babble pool sitting at a ZERO-margin family boundary (gap 0.0005, same as the "safe" gap), which
# is the direct, measured cause of babble scoring worse than expert under a vision-read goal.
#
# What changed from `b1_babble_clean_redo.sh` (v1), and why, each verified before locking in:
#   - lateral: strafe_amp 1.0 -> 1.2 (still --thigh_amp 0.1 --calf_amp 0.4, no fall) -- measured
#     margin (top family magnitude minus runner-up) improved 1.2->1.2x+ before choosing this;
#     0.1/1.2/0.4 gave margin=0.0285 against v1's much weaker, boundary-hugging lateral clips.
#   - yaw: same proven amplitudes (thigh 0.6, pivot 1.5, calf 0.6) but --steps 100 -> 200 -- the
#     already-known lever (F190: heading keeps building past 100 steps) measured to also improve
#     margin (0.0145 at 200 steps vs ~0.001-0.004 at 100-150).
#   - forward: unchanged -- F194's own check found forward already well-matched to hexapod's own
#     range ([0.12, 0.19]) with zero cross-family risk in its own right.
#   - ALL families at 100 steps, including yaw -- NOT 200 as first tried (all-families) or
#     second tried (yaw-only at 200, fwd/lat at 100). Both OOM-killed `wm.fit_body_head`'s
#     combined babble+hexapod gather on the local 31GB machine (confirmed live with `free -h`
#     climbing to the ceiling before each kill). Yaw's 200-step margin improvement (F190/F194
#     follow-up: margin 0.0145 at 200 steps vs ~0.001-0.004 at 100-150) is REAL and worth having,
#     but this machine cannot fit it for the combined rehearsal gather -- deferred to a remote
#     machine (BIAS-2 or com7, both have far more headroom) rather than spending more of this
#     session fighting local RAM. This run's yaw candidates are the same amplitude/pivot values,
#     just shorter, so still an improvement over v1's untargeted margins on fwd/lat at least.
#   - NO allocentric rendering of the raw babble clips (explicitly declined -- 36 individual
#     candidate clips are not what's worth watching). Instead, stage 7 renders the actual CLOSED
#     LOOP result (`close_loop_direct_froude.py`, allocentric+egocentric) once the v2 checkpoint
#     exists -- the system actually selecting candidates toward a goal is the meaningful video.
#
# Versioned as v2 (separate output names throughout) so the F194-validated v1 dataset/checkpoint
# (`data/egocentric/b1_babble_ego_flat`, `wm/runs/b1_babble_clean`) is not clobbered until v2 is
# itself confirmed at least as good.
#
# Usage:
#   bash scripts/run/b1_babble_v2_redo.sh 2>&1 | tee results/wm/dataset/b1_babble/v2_redo_log.txt
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=.venv/bin/python3
S=results/wm/dataset/b1_babble/batch_v2        # pre-render, intermediate -- redundant once rendered
R=data/egocentric/b1_babble_v2_ego_flat        # final, self-sufficient dataset (frames+actions+motion)
OUT=wm/runs/b1_babble_v2
HEXGOAL=data/egocentric/beh12_c10f10t10_ego_flat
EXPERT_CKPT=wm/runs/b1_adapt/body_head_b1.pt
EXPERT_DIR=data/egocentric/beh12_b1_ego_flat

echo "=== stage 0: collect a margin-aware babble batch (forward unchanged, lateral/yaw pushed "
echo "    for decisive family margin per F194's rule) ==="
mkdir -p "$S"
for freq in 1.0 1.5 2.0; do
  for seed in 0 1 2 3; do
    $PY sim/collect/collect_b1_cpg_babble.py --freq "$freq" --thigh_amp 1.0 --calf_amp 0.5 \
      --noise 0.03 --seed "$seed" --steps 100 --out "$S/fwd_f${freq}_s${seed}.npz"
  done
done
for freq in 0.8 1.0 1.2; do
  for seed in 0 1 2 3; do
    $PY sim/collect/collect_b1_cpg_babble.py --freq "$freq" --thigh_amp 0.1 --strafe_amp 1.2 \
      --calf_amp 0.4 --noise 0.03 --seed "$seed" --steps 100 --out "$S/lat_f${freq}_s${seed}.npz"
  done
done
for pv in 1.2 1.5 1.8; do
  for seed in 0 1 2 3; do
    $PY sim/collect/collect_b1_cpg_babble.py --freq 1.0 --thigh_amp 0.6 --pivot_amp "$pv" \
      --calf_amp 0.6 --noise 0.03 --seed "$seed" --steps 100 --out "$S/yaw_p${pv}_s${seed}.npz"
  done
done
echo "collected $(ls "$S"/*.npz | wc -l) clips"

echo "=== stage 0b: verify the margin property directly before spending time on a full fit --"
echo "    F194's own lesson: range coverage alone is not sufficient, check cross-family risk too"
$PY - <<'PYEOF'
import glob, numpy as np, sys
sys.path.insert(0, '.')
from wm.data.embodiment import REGISTRY, load
spec = REGISTRY['b1']
fams = ['fwd', 'lat', 'yaw']
rows = []
for p in sorted(glob.glob('results/wm/dataset/b1_babble/batch_v2/*.npz')):
    clip = load(p, spec)
    fr = np.asarray(clip['body_motion'])[:, :3].mean(0)
    fam = int(np.argmax(np.abs(fr)))
    rows.append((p.split('/')[-1], fr, fam))
from collections import Counter
counts = Counter(fams[f] for _, _, f in rows)
print('family counts:', dict(counts))
preds = np.stack([r[1] for r in rows]); true_fams = np.array([r[2] for r in rows])
dmat = np.abs(preds[:, None, :] - preds[None, :, :]).sum(-1)
np.fill_diagonal(dmat, np.inf)
nn = dmat.argmin(1)
cross = true_fams != true_fams[nn]
print(f'cross-family nearest-neighbour (TRUE-froude only, pre-fit sanity check): {cross.mean():.0%}')
PYEOF

echo "=== stage 0c: render egocentric video for every clip (needed for stage 1's ITM/FTM fit) ==="
mkdir -p "$R"
for f in "$S"/*.npz; do
  $PY sim/render/render_b1_replay.py --scene sim/env/b1_flat.ttt \
    --traj "$f" --out "$R" --ego --align_yaw --ego_seed 0
done

echo "=== stage 1-2: wm.finetune_new_body ==="
$PY -m wm.finetune_new_body \
  --base_ckpt wm/runs/beh12_hexonly_stopgrad/best.pt \
  --embodiment b1 --data "$R" --out_dir "$OUT"

echo "=== stage 4b: refit body_head WITH hexapod rehearsal ==="
$PY -m wm.fit_body_head \
  --ckpt "$OUT/teacher_b1.pt" --data "$R" --embodiment b1 \
  --also "hexapod=$HEXGOAL" --latent both --epochs 400 \
  --out "$OUT/body_head_b1_hex.pt"

echo "=== stage 5: the corrected, single-mechanism 2x2x2 ==="
echo
echo "--- expert-fit + expert-candidates ---"
$PY scripts/diagnostics/objective_experiments/final_2x2x2_test.py \
  --pool expert --ckpt "$EXPERT_CKPT" --candidates_dir "$EXPERT_DIR" --per_condition 1 \
  --goal_dir "$HEXGOAL"

echo "--- babble-fit + babble-candidates (v2, margin-aware) ---"
$PY scripts/diagnostics/objective_experiments/final_2x2x2_test.py \
  --pool babble --ckpt "$OUT/body_head_b1_hex.pt" --candidates_dir "$R" \
  --goal_dir "$HEXGOAL"

echo "=== stage 6: cross-family nearest-neighbour margin, v2 vs v1 (F194's own diagnostic) ==="
$PY - <<'PYEOF'
import glob, os, sys
import numpy as np, torch
sys.path.insert(0, '.')
from wm.config import from_checkpoint
from wm.data.embodiment import REGISTRY, load
from wm.models.action_projector import ActionProjector, action_dims_from
from wm.models.motion_decoder import MotionDecoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cond_of(p):
    with np.load(p, allow_pickle=True) as d:
        if 'condition' in d.files:
            return str(d['condition'])
    return os.path.basename(p)

def family(v):
    return int(np.argmax(np.abs(v)))

def pool_boundary_risk(ckpt_path, candidates_dir, per_condition):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = from_checkpoint(ck['config'])
    channels = [int(c) for c in cfg.body_channels]
    mean = np.asarray(ck['body_stats'][0]).ravel()[:len(channels)]
    std = np.asarray(ck['body_stats'][1]).ravel()[:len(channels)]
    adim = load(glob.glob(candidates_dir + '/*.npz')[0], REGISTRY['b1'])['actions'].shape[1]
    md = MotionDecoder(cfg, {'b1': adim}).to(device).eval(); md.load_state_dict(ck['md'], strict=False)
    proj = ActionProjector(cfg, action_dims_from(ck)).to(device).eval(); proj.load_state_dict(ck['projector'])
    by_cond = {}
    for p in sorted(glob.glob(candidates_dir + '/*.npz')):
        by_cond.setdefault(cond_of(p), []).append(p)
    preds, true_fams = [], []
    for cond, paths in by_cond.items():
        for p in (paths[:per_condition] if per_condition else paths):
            clip = load(p, REGISTRY['b1'])
            a = torch.as_tensor(clip['actions'], dtype=torch.float32, device=device)
            with torch.no_grad():
                z = proj(a, 'b1')
                pred = md.body(None, z).mean(0).cpu().numpy() * std + mean
            preds.append(pred)
            bm = np.asarray(clip['body_motion'])[:, :3].mean(0)
            true_fams.append(family(bm))
    preds = np.stack(preds); true_fams = np.array(true_fams)
    dmat = np.abs(preds[:, None, :] - preds[None, :, :]).sum(-1)
    np.fill_diagonal(dmat, np.inf)
    nn_idx = dmat.argmin(1)
    nn_gap = dmat.min(1)
    cross = true_fams != true_fams[nn_idx]
    return len(preds), cross.mean(), (nn_gap[cross].mean() if cross.any() else float('nan')), nn_gap[~cross].mean()

n, frac, xg, sg = pool_boundary_risk('wm/runs/b1_babble_v2/body_head_b1_hex.pt',
                                     'data/egocentric/b1_babble_v2_ego_flat', None)
print(f'v2 babble: n={n}  frac_cross_family_NN={frac:.0%} (v1 was 19%, expert is 0%)  '
      f'cross-family gap={xg:.4f}  same-family gap={sg:.4f}')
PYEOF

echo "=== stage 7: render the actual CLOSED LOOP result (allocentric+egocentric), locked and free ==="
CL_GOAL=data/egocentric/beh12_c10f10t10_ego_flat/hexapod_ep100.npz  # speed_c7.1, trustworthy fwd goal
CL_DEMO=$(ls "$R"/fwd_f1.5_s0.npz)
mkdir -p results/wm/closed_loop/b1_babble_v2
$PY sim/control/close_loop_direct_froude.py \
  --ckpt "$OUT/body_head_b1_hex.pt" --demo "$CL_DEMO" --goal "$CL_GOAL" \
  --goal_embodiment hexapod --embodiment b1 --candidates_dir "$R" \
  --mechanism direct --goal_source physics --scene sim/env/b1_flat.ttt \
  --out results/wm/closed_loop/b1_babble_v2/locked
$PY sim/control/close_loop_direct_froude.py \
  --ckpt "$OUT/body_head_b1_hex.pt" --demo "$CL_DEMO" --goal "$CL_GOAL" \
  --goal_embodiment hexapod --embodiment b1 --candidates_dir "$R" \
  --mechanism direct --goal_source physics --free_offset --scene sim/env/b1_flat.ttt \
  --out results/wm/closed_loop/b1_babble_v2/free

echo "=== DONE ==="
echo "data:      $S (pre-render), $R (final egocentric)"
echo "checkpoints: $OUT/{adapted_b1,projector_b1,teacher_b1,body_head_b1,body_head_b1_hex}.pt"
echo "closed-loop videos (npz, allo+ego frames inside): results/wm/closed_loop/b1_babble_v2/{locked,free}/"
