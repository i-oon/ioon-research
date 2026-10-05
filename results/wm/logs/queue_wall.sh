#!/usr/bin/env bash
cd /home/fibo07/ioon/ioon-research
while kill -0 2591175 2>/dev/null; do sleep 60; done
sleep 300
P=$(bash scripts/tools/launch_guarded.sh wallbias results/wm/logs/wall_distance_bias.log .venv/bin/python3 scripts/diagnostics/egocentric_view/wall_distance_bias.py)
while kill -0 $P 2>/dev/null; do sleep 15; done
echo WALL_DONE
