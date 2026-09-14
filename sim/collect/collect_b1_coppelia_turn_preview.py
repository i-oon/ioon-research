"""Turning/mixed B1 generic-babble preview using the same coupled-duty CPG family.

This adds a global hip oscillator with a fixed left/right sign layout to the stance-preloaded
coupled-duty gait.  It is still one generic parameterization (frequency, amplitude, duty,
coupling, stance preload, hip sign/clock/ratio), not a separate scripted turn controller.

Screen-only validation across seeds 10-14: 5/5 upright, forward Froude around 0.10, yaw around
0.045-0.052, lateral around -0.02.

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
        raise SystemExit("turn preview fixes CPG shape, coupling, duty, frequency, amplitude, "
                         "hip oscillator, and stance preload")
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
        "--generic-hip-sign-layout", "left-right",
        "--generic-hip-phase", "0.0",
    ])
    main()
