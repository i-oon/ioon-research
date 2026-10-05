#!/usr/bin/env bash
cd /home/fibo07/ioon/ioon-research
F=scripts/diagnostics/forward_model/frame_head_leak_check.py
.venv/bin/python3 $F --embodiment hexapod --body c10 --groups 8 2>&1 | grep -vi "warn\|Loading weights"
.venv/bin/python3 $F --embodiment b1 --body b1 --groups 8 2>&1 | grep -vi "warn\|Loading weights"
echo LEAK_DONE
