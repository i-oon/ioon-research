"""Generic quadruped stance/swing CPG plus motor-noise entry point."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collect_b1_coppelia_babble import main


if __name__ == "__main__":
    forbidden = {"--cpg-mode", "--generic-phase-layout", "--generic-joint-role-layout",
                 "--generic-gait-shape"}
    if forbidden.intersection(sys.argv):
        raise SystemExit("CPG mode, layout, and gait shape are fixed by this entry point")
    sys.argv.extend(["--cpg-mode", "generic", "--generic-phase-layout", "trot",
                     "--generic-joint-role-layout", "quadruped-trot",
                     "--generic-gait-shape", "stance-swing"])
    main()
