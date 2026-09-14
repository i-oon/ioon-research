"""Fast forward B1 generic-babble preview with stance preload.

Same coupled-duty CPG family as the previous previews, plus a generic planted-leg calf preload
(`stance_calf_bias=-1.2`) discovered in the 2026-09-12 contact/speed screen.  This made forward
Froude reach the useful lower band (~0.12) while staying upright across seeds 10-14 in screen-only
validation.

Preview only, not a final frozen collection distribution.
"""
import sys

from collect_b1_coppelia_babble import main


if __name__ == "__main__":
    fixed = {
        "--cpg-mode", "--generic-phase-layout", "--generic-gait-shape", "--generic-duty-factor",
        "--generic-frequency", "--generic-amplitude", "--generic-coupling-ratio",
        "--generic-clearance-ratio", "--generic-trot-pairing", "--generic-thigh-sign-layout",
        "--generic-hip-ratio", "--generic-stance-calf-bias",
    }
    if fixed.intersection(sys.argv):
        raise SystemExit("target-forward preview fixes CPG shape, coupling, duty, frequency, "
                         "amplitude, hip, and stance preload")
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
        "--generic-stance-calf-bias", "-1.2",
    ])
    main()
