"""Generic diagonal duty-cycle CPG plus per-step motor-noise collection entry point.

The phase layout, joint roles, and duty factor are intentionally fixed here.  A final experiment
may predeclare a parameter distribution for frequency/amplitude/lift/noise, but it must preserve
every sampled rollout and must not use measured Froude or survival to select clips.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collect_b1_coppelia_babble import main


if __name__ == "__main__":
    forbidden = {"--cpg-mode", "--generic-phase-layout", "--generic-joint-role-layout",
                 "--generic-gait-shape", "--generic-duty-factor"}
    if forbidden.intersection(sys.argv):
        raise SystemExit("CPG mode, layout, shape, and duty factor are fixed by this entry point")
    sys.argv.extend(["--cpg-mode", "generic", "--generic-phase-layout", "trot",
                     "--generic-joint-role-layout", "quadruped-trot",
                     "--generic-gait-shape", "duty-cycle", "--generic-duty-factor", "0.65"])
    main()
