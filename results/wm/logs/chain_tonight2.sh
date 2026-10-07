#!/usr/bin/env bash
cd /home/fibo07/ioon/ioon-research
bash scripts/run/b1_babble_sweep.sh wm/runs/round1_hexonly_s0_rr/best.pt 176
sleep 300
bash scripts/run/b1_babble_variants.sh wm/runs/round1_hexonly_s0_rr/best.pt 44 88
echo CHAIN_DONE
