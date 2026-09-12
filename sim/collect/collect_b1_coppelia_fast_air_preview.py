"""Fixed preview of the fastest reliable native-Coppelia generic B1 babble seed (2026-09-12, v3).

This is a rerunnable preview handle, not a final dataset collector. Fixes the best config found
by a real multi-seed sweep on 2026-09-12: coupled-duty gait (see
`collect_b1_coppelia_coupled_duty_preview.py`'s docstring for the joint-coupling mechanism, ported
from Egocentric VSM's own reference code), frequency pushed to 5.0 Hz, amplitude 0.26, and --
the key addition here -- duty_factor LOWERED to 0.55 (more swing/air time, less stance) rather
than the earlier 0.65/0.75 values. More airtime, at this frequency and amplitude, turned out to
be a genuinely better lever than more amplitude: `amp=0.26, duty=0.65` was only ~67% reliable
(4/6 survival, seed-dependent); `amp=0.26, duty=0.55` is **11/11 (100%)** across every seed
tested, INCLUDING three full egocentric renders, not just screen-only screening.

Forward Froude **0.050-0.063** -- roughly double the original uncoupled preview's ~0.03 ceiling,
and the closest this whole investigation has gotten to the 0.10-0.20 pretraining/expert target
(still short, not matched). Lateral Froude 0.006-0.023, yaw 0.011-0.028 -- higher than the
straightest configs found (duty=0.65's ~0.003) but still far below the 0.03-0.13 range every
uncoupled config produced. See `q22_handoff_prompt.md`'s "Frequency push" and later sections for
the full sweep this was chosen from, including why amplitude beyond ~0.27-0.28 at this frequency
reliably falls regardless of duty factor.

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
        raise SystemExit("fast-air preview fixes CPG shape, phase, coupling ratio, duty factor, "
                         "frequency, and amplitude")
    sys.argv.extend([
        "--cpg-mode", "generic",
        "--generic-phase-layout", "trot",
        "--generic-gait-shape", "coupled-duty",
        "--generic-frequency", "5.0",
        "--generic-amplitude", "0.26",
        "--generic-coupling-ratio", "0.6",
        "--generic-duty-factor", "0.55",
        "--generic-trot-pairing", "diagonal",
        "--generic-thigh-sign-layout", "same",
        "--generic-hip-ratio", "0.0",
    ])
    main()
