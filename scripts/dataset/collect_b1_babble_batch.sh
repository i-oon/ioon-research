#!/bin/bash
# B1 generic CPG babble batch -- for Q21's candidate-source test (does babble substitute for the
# recorded expert-clip candidate library, F188/W15-1). Freq range picked from a small sweep to span
# roughly B1's own expert-clip Froude range (mean 0.091, p95 0.181): 0.6-1.2 Hz gave 0.029-0.174.
# turn_bias and noise add diversity the same way gecko's babble sweep did.
set -e
cd /home/aria/ioon-research
OUT=results/wm/dataset/b1_babble/batch
mkdir -p "$OUT"
rm -f "$OUT"/*.npz
i=0
for freq in 0.6 0.8 1.0 1.2 1.4; do
  for turn in -0.15 0.0 0.15; do
    for seed in 0 1; do
      i=$((i+1))
      echo "=== clip $i: freq=$freq turn=$turn seed=$seed ==="
      .venv/bin/python3 sim/collect/collect_b1_cpg_babble.py \
        --freq "$freq" --turn_bias "$turn" --noise 0.03 --seed "$seed" \
        --out "$OUT/b1babble_${i}.npz" || echo "  (skipped/failed)"
    done
  done
done
echo "DONE: $(ls $OUT/*.npz 2>/dev/null | wc -l) clips saved"
