#!/usr/bin/env bash
#
# The only script com7 runs. Builds the three clean dataset directories, then trains the five
# Stage 1 models into wm/runs/.
#
#   bash scripts/com7_train.sh
#
# com7 trains and does nothing else -- no edits to code or documents, no measurements, no commits.
# If a run fails, stop and send the log; the fix is made on the main machine and pushed. An
# experiment is only controlled if the code that produced it is the code in the repository.
#
# Runs in the foreground so the log is visible. It takes roughly 10 hours end to end, so run it
# under tmux or screen if the connection may drop.
#
# When it finishes, send back two files per run and nothing else:
#   wm/runs/<run>/best.pt        the trained weights
#   wm/runs/<run>/config.yaml    what it actually trained under
#
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python3

# Stamp the commit into the log. Every checkpoint has to be traceable to the code that made it,
# and "the version we had that day" is not traceable.
echo "=== repository $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
if [ -n "$(git status --porcelain -- wm scripts 2>/dev/null)" ]; then
  echo "wm/ or scripts/ has uncommitted changes on this machine." >&2
  echo "com7 trains the committed code only. Reset it, or report what differs." >&2
  git status --short -- wm scripts >&2
  exit 1
fi

echo "=== building dataset directories  $(date '+%F %T')"
$PY scripts/build_stage1_dirs.py

echo
echo "=== training  $(date '+%F %T')"
bash scripts/retrain_stage1.sh

echo
echo "=== all done  $(date '+%F %T')"
echo "send back best.pt and config.yaml for each of the five runs; leave everything else here."
