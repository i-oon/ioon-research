#!/usr/bin/env bash
# waits for the adaptation job (b1_rooms_adapt), then the original-room physics loops on the GPU, 10 min apart
cd /home/fibo07/ioon/ioon-research
while kill -0 $(cat results/wm/logs/b1_rooms_adapt.pid) 2>/dev/null; do sleep 30; done
sleep 600
PORT=25700 bash scripts/run/physics_orig_b1.sh
sleep 600
PORT=25702 BODY=c10 bash scripts/run/physics_orig_hex.sh
sleep 300
PORT=25702 BODY=c08 bash scripts/run/physics_orig_hex.sh
echo QUEUE_DONE
