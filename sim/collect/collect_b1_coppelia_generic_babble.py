"""Claim-honest entry point for generic B1 CPG exploration under CoppeliaSim Bullet."""
import sys

from collect_b1_coppelia_babble import main


if __name__ == "__main__":
    if "--cpg-mode" in sys.argv:
        raise SystemExit("--cpg-mode is fixed to generic by this entry point")
    sys.argv.extend(["--cpg-mode", "generic"])
    main()
