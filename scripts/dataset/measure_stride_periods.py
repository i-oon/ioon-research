"""Stride period per hexapod behaviour, for stride-matched Froude labels (2026-10-15, F348; Ajan Blink week 20).

The c10 CPG stride depends on the behaviour (measured 1.06-2.65 Hz, median 1.82 Hz), so one label window cannot span whole
strides for every behaviour. Per condition: dominant frequency of the joint commands (`actions`, 18-D, mean removed,
zero-padded FFT x8, > 0.3 Hz) of the c10 TRAINING main clips (66 frames each), median over the clips of that condition.
c08 runs the same walk plans (same condition names, same CPG pace). B1 strides at 2.01 Hz in every behaviour, so its 1 s
window already spans exactly 2 strides; it is not in the table.

    .venv/bin/python3 scripts/dataset/measure_stride_periods.py   # -> wm/data/stride_periods_hexapod.json (tracked in git)
"""
import collections
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "data/counterfactual_walks/c10_clips_train")
OUT = os.path.join(ROOT, "wm/data/stride_periods_hexapod.json")


def main():
    per = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(SRC, "*.npz"))):
        x = np.load(f, allow_pickle=True)
        A = np.asarray(x["actions"], float); A = A - A.mean(0); n = len(A); dt = float(x["dt"]) if "dt" in x.files else 0.05
        fr = np.fft.rfftfreq(n * 8, dt); sp = np.abs(np.fft.rfft(A, n=n * 8, axis=0)).sum(1); ok = fr > 0.3
        per[str(x["condition"])].append(float(fr[ok][np.argmax(sp[ok])]))
    table = {c: round(1.0 / float(np.median(v)), 4) for c, v in sorted(per.items())}
    json.dump({"_source": "c10_clips_train, dominant joint-command frequency, median per condition (seconds per stride)",
               "period_s": table}, open(OUT, "w"), indent=1)
    print(f"{len(table)} conditions, period {min(table.values()):.2f}-{max(table.values()):.2f} s -> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
