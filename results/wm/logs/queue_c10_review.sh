#!/usr/bin/env bash
# c10 wall data: 10-episode REVIEW set (4 pilot + 6 new: 3 approach, 3 corner), physics -> render -> review videos.
cd /home/fibo07/ioon/ioon-research
PY=.venv/bin/python3; S=scripts/dataset/collect_c10_wall_data.py
$PY $S physics --split pilot --n 3 --ep0 9004 --look0 9004 --ports 25780 &
$PY $S physics --split pilot --n 3 --ep0 9104 --look0 9104 --ports 25782 --corner &
wait
$PY $S report --splits pilot
$PY $S stats --split pilot
$PY $S render --split pilot --ports 25780 25782
$PY -c "
import sys; sys.argv=['x','review','--split','pilot']; sys.path.insert(0,'scripts/dataset')
import collect_c10_wall_data as C, argparse
a=argparse.Namespace(split='pilot', n=10, splits=['pilot'])
C.W.do_review(a); C.W.do_branchgrid(a); C.W.do_allo(a)"
echo QUEUE_C10_REVIEW_DONE
