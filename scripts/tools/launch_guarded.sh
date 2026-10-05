#!/usr/bin/env bash
# Launch a long GPU job detached, with gpu_guard.sh attached to the job's REAL pid (written by the job itself),
# so no one has to find the pid with ps/grep (that picked a short-lived wrapper three times, 2026-10-04/05).
#   bash scripts/tools/launch_guarded.sh NAME LOGFILE command args...
# Writes results/wm/logs/NAME.pid (the job's pid) and results/wm/logs/gpu_guard_NAME.log; prints the pid.
set -u
cd "$(dirname "$0")/../.."
NAME=${1:?usage: launch_guarded.sh NAME LOGFILE command...}; LOG=${2:?}; shift 2
PIDF=results/wm/logs/$NAME.pid; rm -f "$PIDF"
setsid nohup bash -c 'echo $$ > "$0"; exec "$@"' "$PIDF" "$@" > "$LOG" 2>&1 < /dev/null &
for _ in $(seq 50); do [ -s "$PIDF" ] && break; sleep 0.1; done
PID=$(cat "$PIDF")
setsid nohup bash scripts/tools/gpu_guard.sh "$PID" > "results/wm/logs/gpu_guard_$NAME.log" 2>&1 < /dev/null &
echo "$PID"
