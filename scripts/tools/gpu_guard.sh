#!/usr/bin/env bash
# Pause a GPU job while the card is too hot, resume it when it has cooled (resource rule, 2026-10-04: the local GPU
# reached 84 C at its power limit after ~30 h of back-to-back jobs).
#   bash scripts/tools/gpu_guard.sh PID [HOT=83] [COOL=75]
# Sends SIGSTOP to PID and all its descendants above HOT C, SIGCONT below COOL C; terminates them if MemAvailable
# drops below MIN_RAM_GB (default 5); exits when PID ends.
# Only ever signals the given PID's own process tree.
PID=${1:?usage: gpu_guard.sh PID [HOT] [COOL]}; HOT=${2:-83}; COOL=${3:-75}
tree() { local p=$1; echo "$p"; for c in $(ps -o pid= --ppid "$p"); do tree "$c"; done; }
paused=0
while kill -0 "$PID" 2>/dev/null; do
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if [ "$avail" -lt "${MIN_RAM_GB:-5}" ]; then      # RAM guard (2026-10-06): stop the job before the OOM killer takes VS Code
    kill -CONT $(tree "$PID") 2>/dev/null; kill -TERM $(tree "$PID") 2>/dev/null
    echo "$(date +%T) MemAvailable ${avail} GB < ${MIN_RAM_GB:-5} GB: terminated $PID"; break
  fi
  t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
  if [ "$paused" = 0 ] && [ "$t" -ge "$HOT" ]; then
    kill -STOP $(tree "$PID") 2>/dev/null; paused=1; echo "$(date +%T) GPU ${t}C >= ${HOT}: paused $PID"
  elif [ "$paused" = 1 ] && [ "$t" -le "$COOL" ]; then
    kill -CONT $(tree "$PID") 2>/dev/null; paused=0; echo "$(date +%T) GPU ${t}C <= ${COOL}: resumed $PID"
  fi
  sleep 5
done
[ "$paused" = 1 ] && kill -CONT $(tree "$PID") 2>/dev/null
echo "$(date +%T) $PID ended"
