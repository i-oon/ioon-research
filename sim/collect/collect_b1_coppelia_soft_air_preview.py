"""Softened version of the fastest generic B1 babble preview.

This keeps the fast-air coupled-duty gait's load-bearing settings -- 5.0 Hz, amplitude 0.26,
coupling ratio 0.6, duty factor 0.55 -- but reduces the swing clearance arc from 1.5x to 1.1x
amplitude.  The goal is to remove the tip-toe/prancing look without giving up the speed gain
that made the fast-air preview useful.

This is still a fixed preview point, not the final babble collection distribution.
"""
import sys

from collect_b1_coppelia_babble import main


if __name__ == "__main__":
    fixed = {
        "--cpg-mode", "--generic-phase-layout", "--generic-gait-shape", "--generic-duty-factor",
        "--generic-frequency", "--generic-amplitude", "--generic-coupling-ratio",
        "--generic-clearance-ratio", "--generic-trot-pairing", "--generic-thigh-sign-layout", "--generic-hip-ratio",
    }
    if fixed.intersection(sys.argv):
        raise SystemExit("soft-air preview fixes CPG shape, phase, coupling, clearance, duty, "
                         "frequency, and amplitude")
    sys.argv.extend([
        "--cpg-mode", "generic",
        "--generic-phase-layout", "trot",
        "--generic-gait-shape", "coupled-duty",
        "--generic-frequency", "5.0",
        "--generic-amplitude", "0.26",
        "--generic-coupling-ratio", "0.6",
        "--generic-clearance-ratio", "1.1",
        "--generic-duty-factor", "0.55",
        "--generic-trot-pairing", "diagonal",
        "--generic-thigh-sign-layout", "same",
        "--generic-hip-ratio", "0.0",
    ])
    main()
