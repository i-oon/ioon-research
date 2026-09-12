"""Fixed preview of the coupled-duty B1 babble seed (2026-09-12), the current best generic config.

This is a rerunnable preview handle, not a final dataset collector. It fixes the mechanism found
by directly reading Egocentric VSM's own reference code (`doc/ref/Egocentric_VSM/env_agent.py`'s
`move_altas`): their gait algebraically COUPLES the ankle/knee to the hip (`knee = 0.6 - hip`)
rather than driving every joint with an independent sine -- a design-time kinematic prior that
keeps the foot's orientation coherent through the stride, not real-time feedback (it never reads
robot state). `generic_coupled_duty_action_at` ports this to B1: `calf = -coupling_ratio * thigh`
during stance (foot orientation follows the thigh, planted), plus one active clearance arc during
swing so the foot still lifts (pure coupling alone, tested first, barely moved -- it never lifts
the foot at all).

Config: frequency 2.0 Hz, amplitude 0.28, coupling_ratio 0.6, duty_factor 0.65. Confirmed by a
real sweep, not a single run: **20/20 upright** across amplitude 0.20-0.28 (4 seeds each),
lateral Froude consistently under 0.003 -- roughly 10-40x straighter than every uncoupled config
tried in the earlier amplitude/duty-factor push (`q22_handoff_prompt.md`), where lateral drift
ranged 0.006-0.13. Survival starts dropping at amplitude 0.29+ (6/8). Forward Froude at 0.28 is
~0.021-0.026, comparable to (not higher than) the uncoupled best -- the win here is reliability
and straightness, not raw speed; the ~0.03 ceiling from the earlier push is not broken by this.

The final babble dataset still needs a predeclared parameter distribution and must retain every
sampled rollout, including falls -- this file fixes ONE point for preview/rendering, not the
final collection distribution.
"""
import sys

from collect_b1_coppelia_babble import main


if __name__ == "__main__":
    fixed = {
        "--cpg-mode", "--generic-phase-layout", "--generic-gait-shape", "--generic-duty-factor",
        "--generic-frequency", "--generic-amplitude", "--generic-coupling-ratio",
        "--generic-trot-pairing", "--generic-thigh-sign-layout", "--generic-hip-ratio",
    }
    if fixed.intersection(sys.argv):
        raise SystemExit("coupled-duty preview fixes CPG shape, phase, coupling ratio, duty "
                         "factor, frequency, and amplitude")
    sys.argv.extend([
        "--cpg-mode", "generic",
        "--generic-phase-layout", "trot",
        "--generic-gait-shape", "coupled-duty",
        "--generic-frequency", "2.0",
        "--generic-amplitude", "0.28",
        "--generic-coupling-ratio", "0.6",
        "--generic-duty-factor", "0.65",
        "--generic-trot-pairing", "diagonal",
        "--generic-thigh-sign-layout", "same",
        "--generic-hip-ratio", "0.0",
    ])
    main()
