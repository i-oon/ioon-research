"""Fixed preview of the current safest native-Coppelia generic B1 babble seed (2026-09-12, v2).

This is a rerunnable preview handle, not a final dataset collector. It fixes the best
development setting found by a real multi-seed amplitude/duty-factor sweep on 2026-09-12:
generic diagonal duty-cycle CPG, 2.0 Hz, amplitude 0.32, duty factor 0.75 (up from the prior
preview's 0.65 -- more of the cycle spent planted, a bigger support margin, the same mechanism
that fixed gecko's own diagonal-trot instability earlier this session), calf ratio 2.5, plus
per-step motor noise.

Confirmed 3/3 upright across seeds in the sweep (vs. the amplitude-0.24/duty-0.65 preview's
untested single-seed survival), lateral Froude ~0.006 (down 10x from the old preview's
0.02-0.13 range depending on seed), forward Froude 0.024-0.028 -- comparable to, not higher
than, the old preview's 0.0316-0.0328. **The improvement here is stability and cleanliness, not
speed.** The forward-Froude ceiling (~0.03) did not move; see q22_handoff_prompt.md's
"Amplitude/duty-factor push" section for the full sweep this config was chosen from, including
why amplitude beyond ~0.34 reliably falls regardless of duty factor.

The final babble dataset still needs a predeclared parameter distribution and must retain every
sampled rollout, including falls -- this file fixes ONE point for preview/rendering, not the
final collection distribution.
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
        raise SystemExit("wide-stance preview fixes CPG shape, phase, signs, frequency, duty "
                         "factor, and amplitude")
    sys.argv.extend([
        "--cpg-mode", "generic",
        "--generic-phase-layout", "trot",
        "--generic-joint-role-layout", "quadruped-trot",
        "--generic-gait-shape", "duty-cycle",
        "--generic-duty-factor", "0.75",
        "--generic-frequency", "2.0",
        "--generic-amplitude", "0.32",
        "--generic-calf-ratio", "2.5",
        "--generic-trot-pairing", "diagonal",
        "--generic-thigh-sign-layout", "same",
        "--generic-calf-sign-layout", "same",
    ])
    main()
