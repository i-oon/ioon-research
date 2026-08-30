#!/usr/bin/env bash
#
# Is the state-fidelity / action-sensitivity tension tunable, or structural?
#
#   bash scripts/com7_lambda_sweep.sh          # on com7, under tmux; about four hours
#
# **The two endpoints are already measured and this fills between them** (F139), one stage-3
# adaptation per lambda, everything else held identical -- same stage-1 checkpoint, same projector,
# same 24 clips, same 15,000 steps, same batch, same seed:
#
#   lambda 0    state fidelity 0.592   /mean-z 0.985   good predictor, deaf to the action
#   lambda 1    state fidelity 1.370   /mean-z 0.49    reads the action, worse than holding still
#
# **The question, and it decides weeks against months.** Does any intermediate lambda give state
# fidelity below about 0.8 *and* a meaningful action dependence at the same time? If one does, the
# tension is a knob and that lambda is the pretraining target. If every lambda pays the full price
# -- fidelity degrading in step with sensitivity, no elbow -- the tension is structural, one
# forward model cannot do both jobs, and the architecture has to split them.
#
# **This is a proxy and must be reported as one.** It sweeps the term in *adaptation*, which is
# hours, not in pretraining, which is days. Adaptation and pretraining differ in what else is being
# fitted, so an elbow found here is a reason to spend a pretrain, not a substitute for it.
#
# **Two numbers per lambda, both at one step:**
#   `/mean-z` from adapt3's own log -- the projector path, the one a distilled policy would drive
#   state fidelity from `rollout_fidelity.py` -- rolled state against holding still, plus the
#   perturbed-latent delta
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python3
. scripts/b1_stage3_clips.sh
RUN=wm/runs/beh12_hexonly

for LAM in 0.1 0.25 0.5 0.75; do
  OUT=$RUN/stage3_b1_lam${LAM}.pt
  echo
  echo "=== lambda $LAM  $(date '+%F %T')"
  if [ -f "$OUT" ]; then echo "skip $OUT"; else
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u -m wm.adapt3 \
      --ckpt $RUN/adapted_b1.pt --projector $RUN/projector_b1_adapted.pt \
      --data data/beh12_b1_flat --embodiment b1 --train_clips $CLIPS \
      --steps 15000 --lambda_nce "$LAM" --batch 8 --seed 0 \
      --cache results/wm/cache/b1.pt --out "$OUT"
  fi
  echo "--- state fidelity, lambda $LAM"
  $PY -u scripts/diagnostics/rollout_fidelity.py --ckpt "$OUT" \
    --data data/beh12_b1_flat --embodiment b1 --cache results/wm/cache/b1.pt \
    --mean_z --horizons 1 3 5
done

echo
echo "=== done  $(date '+%F %T')"
echo "send back the whole log; the checkpoints can stay unless an elbow appears"
echo
echo "reading it: the last /mean-z printed by each adapt3 run is the action sensitivity through"
echo "the projector, and the ratio at horizon 1 is the state fidelity. **An elbow is a lambda"
echo "where the ratio is still under ~0.8 while /mean-z has already fallen well below 1.0.**"
echo "No elbow -- fidelity and sensitivity moving together across every lambda -- means one"
echo "forward model cannot hold both and the architecture, not the weighting, is what has to change."
