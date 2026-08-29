# data — what each set is

**Naming.** `<behaviours>_<body>_<variant>`, the same rule as `wm/runs/`.

| field | values |
|---|---|
| behaviours | `beh12` -- the twelve matched conditions, speed / turn / sideways; `fwd` -- forward walking only |
| body | `c10f10t10`, `c08f09t09` for the two insect bodies by their segment ratios; `b1`; `hex8body`, `m3d`, `bracket` for multi-body Stage 1 sets |
| variant | `flat` -- one clip per file with labels; `50hz` -- the rate it was rendered at, when that is the thing about it |

**Bodies are named, never called "hexapod".** Two insect bodies with the same generic name cost a
week: the pretraining body and the held-out body turned opposite ways and every table said
"hexapod" (F117).

## The live sets

| | clips | what it is |
|---|---|---|
| `beh12_c10f10t10_flat` | 48 | the body the world model is pretrained on |
| `beh12_c08f09t09_flat` | 48 | a held-out body of the same embodiment |
| `beh12_b1_flat` | 48 | the quadruped -- the only executable candidates |
| `beh12_b1_flat_9clips` | 9 | symlinks into the above; the few-shot budget |

**All three turn the same way**, anticlockwise on screen, since 2026-08-29. They did not before, and
nothing that predates that can be compared across them (F115, F117). `beh12_b1_flat/README.md`
lists the four defects corrected on the quadruped side.

## Stage 1 sets, forward walking only

`fwd_hex8body`, `fwd_hex7speed`, `fwd_m3d`, `fwd_bracket`, `fwd_cov_wide`, `fwd_cov_narrow`,
`fwd_decoupled` -- the multi-body insect sets Stage 1 was measured on, and `fwd_b1_50hz`, the
quadruped set the `s2_*` runs used.

**The `fwd` prefix is the point.** Everything known about the shared body target was measured on
these, so it was measured on **one behaviour**. `fwd_b1_50hz` also names its own defect: it was
rendered at 50 Hz against the insect's 20, which made every cross-embodiment number computed with it
span mismatched durations (F74).

Several are symlink farms into `fwd_hex8body` and `fwd_decoupled` -- cheap views, not copies.

## `_archive/`

Speed and ramp sweeps, the 4-leg body sets, the B1 trajectory dumps, and raw pre-flatten
directories. **Nothing in the current pipeline reads them and their results are written up in
`doc/FINDINGS.md`.** Kept so that deleting them is a decision rather than an accident. ~700 MB.
