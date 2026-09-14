"""Lateral/yaw-heavy edge preview for B1 generic babble.

This uses the same stance-preloaded coupled-duty CPG family, with a stronger global hip oscillator
and diagonal hip sign layout.  It can produce much larger lateral/yaw components, but validation
showed it is a Bullet fall-boundary sample (mixed survival), so it is diagnostic coverage evidence,
not a safe default.

Preview only, not a final frozen collection distribution.
"""
import sys

from collect_b1_coppelia_babble import main


if __name__ == "__main__":
    fixed = {
        "--cpg-mode", "--generic-phase-layout", "--generic-gait-shape", "--generic-duty-factor",
        "--generic-frequency", "--generic-amplitude", "--generic-coupling-ratio",
        "--generic-clearance-ratio", "--generic-trot-pairing", "--generic-thigh-sign-layout",
        "--generic-hip-ratio", "--generic-hip-clock", "--generic-hip-sign-layout",
        "--generic-hip-phase", "--generic-stance-calf-bias",
    }
    if fixed.intersection(sys.argv):
        raise SystemExit("side-yaw edge preview fixes CPG shape, coupling, duty, frequency, "
                         "amplitude, hip oscillator, and stance preload")
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
        "--generic-stance-calf-bias", "-1.0",
        "--generic-hip-ratio", "0.8",
        "--generic-hip-clock", "global",
        "--generic-hip-sign-layout", "diagonal",
        "--generic-hip-phase", "0.0",
    ])
    main()
