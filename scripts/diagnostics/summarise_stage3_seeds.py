"""Average a stage-3 seed sweep over its late window, per arm, from the training log.

    .venv/bin/python3 scripts/diagnostics/summarise_stage3_seeds.py paste_from_com7.md

**Why a log and not the checkpoints.** `wm/adapt3.py` writes only the *final* `top1` and `family`
into its checkpoint, and the final step is not the run: `family` wanders about four points between
adjacent evaluations, so nce seed 0 ends on 57% against a late-window mean of 53.7%. Quoting the
last line is a cherry-pick that nothing in the saved file can correct, and the per-step history
exists only in the log.

**The window is fixed in advance and applied to both arms**, so it cannot be chosen after seeing
which arm it favours. `--from` moves it; the default is step 10000 of 15000.

Chance is not 1/12: the families hold unequal numbers of conditions and `adapt3` accumulates the
rate per pick. It prints 28% for `family` and 8% for `cond`.
"""
import argparse
import re
import statistics as st

ROW = re.compile(r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)%\s+(\d+)%")
HEAD = re.compile(r"=== (\w+) seed (\d+) ===")


def parse(path):
    runs, key = {}, None
    for line in open(path):
        head = HEAD.match(line)
        if head:
            key = (head.group(1), int(head.group(2)))
            runs[key] = []
            continue
        row = ROW.match(line)
        if row and key:
            runs[key].append({"step": int(row.group(1)), "hold": float(row.group(3)),
                              "mean_z": float(row.group(4)), "cond": int(row.group(6)),
                              "family": int(row.group(7))})
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--from", dest="start", type=int, default=10000)
    args = ap.parse_args()

    runs = parse(args.log)
    if not runs:
        raise SystemExit(f"no `=== <arm> seed <n> ===` blocks in {args.log}")
    per_arm = {}
    print(f"{'run':<10}{'family':>9}{'cond':>8}{'/mean-z':>9}{'/hold':>8}{'ckpt':>6}{'final':>7}")
    for key in sorted(runs):
        late = [r for r in runs[key] if r["step"] >= args.start]
        if not late:
            raise SystemExit(f"{key} has nothing at or after step {args.start}")
        row = tuple(st.mean(r[k] for r in late) for k in ("family", "cond", "mean_z", "hold"))
        per_arm.setdefault(key[0], []).append(row)
        print(f"{key[0] + '_s' + str(key[1]):<10}{row[0]:>8.1f}%{row[1]:>7.1f}%{row[2]:>9.3f}"
              f"{row[3]:>8.3f}{len(late):>6}{runs[key][-1]['family']:>6}%")

    print(f"\nmean and spread across seeds, from step {args.start}")
    for arm, rows in sorted(per_arm.items()):
        spread = st.stdev(r[0] for r in rows) if len(rows) > 1 else float("nan")
        print(f"  {arm:<5} family {st.mean(r[0] for r in rows):.1f}% +/- {spread:.1f}  "
              f"cond {st.mean(r[1] for r in rows):.1f}%  "
              f"/mean-z {st.mean(r[2] for r in rows):.3f}  "
              f"/hold {st.mean(r[3] for r in rows):.3f}  ({len(rows)} seeds)")
    print("\nchance is 28% for family and 8% for cond; `/mean-z` near 1.0 means the forward model")
    print("gives the same answer for the real action as for the mean one -- it is not reading it.")


if __name__ == "__main__":
    main()
