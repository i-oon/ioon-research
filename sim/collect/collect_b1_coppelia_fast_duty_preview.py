"""Fixed preview of the current safest native-Coppelia generic B1 babble seed.

This is a rerunnable preview handle, not a final dataset collector.  It fixes the best
development setting seen on 2026-09-12: generic diagonal duty-cycle CPG, 2.0 Hz, amplitude
0.24, calf ratio 2.5, plus per-step motor noise.  The final babble dataset still needs a
predeclared parameter distribution and must retain every sampled rollout.
"""
import sys

from collect_b1_coppelia_babble import main


if __name__ == "__main__":
    fixed = {
        "--cpg-mode", "--generic-phase-layout", "--generic-joint-role-layout",
        "--generic-gait-shape", "--generic-duty-factor", "--generic-frequency",
        "--generic-amplitude", "--generic-calf-ratio", "--generic-trot-pairing",
        "--generic-thigh-sign-layout", "--generic-calf-sign-layout",
    }
    if fixed.intersection(sys.argv):
        raise SystemExit("fast duty preview fixes CPG shape, phase, signs, frequency, and amplitude")
    sys.argv.extend([
        "--cpg-mode", "generic",
        "--generic-phase-layout", "trot",
        "--generic-joint-role-layout", "quadruped-trot",
        "--generic-gait-shape", "duty-cycle",
        "--generic-duty-factor", "0.65",
        "--generic-frequency", "2.0",
        "--generic-amplitude", "0.24",
        "--generic-calf-ratio", "2.5",
        "--generic-trot-pairing", "diagonal",
        "--generic-thigh-sign-layout", "same",
        "--generic-calf-sign-layout", "same",
    ])
    main()
