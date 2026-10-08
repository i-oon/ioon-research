#!/usr/bin/env bash
cd /home/fibo07/ioon/ioon-research
P=$(cat results/check/multiversion/render.pid)
while kill -0 $P 2>/dev/null; do sleep 30; done
.venv/bin/python3 scripts/dataset/render_multiversion.py stop
.venv/bin/python3 scripts/dataset/render_multiversion.py launch --ports 25600
sleep 15
PORT=25600 bash scripts/run/physics_rr_c10.sh
.venv/bin/python3 scripts/dataset/render_multiversion.py stop
