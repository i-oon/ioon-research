"""Aggressive target-aware screen for the live-Bullet B1 CPG primitives.

Screening deliberately tolerates many falls. Each attempt leaves a YAML outcome; stable high-motion
configs are ranked for a later, rendered egocentric rerun. This is designed-primitive exploration,
not evidence for undirected babble.
"""
import argparse
import csv
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
COLLECTOR = ROOT / "sim/collect/collect_b1_coppelia_babble.py"


def configs():
    rows = []
    # Push cadence and joint excursion independently; previous tests changed mostly amplitude.
    drive = [(0.23, 0.25), (0.28, 0.32), (0.34, 0.40), (0.45, 0.52), (0.60, 0.65)]
    for freq, (thigh, calf) in itertools.product((0.5, 0.8, 1.2, 1.8), drive):
        rows.append(dict(family="forward", freq=freq, thigh_amp=thigh, calf_amp=calf,
                         strafe_amp=0.0, pivot_amp=0.0))
    # Both lateral signs are necessary; same-sign hip motion is the collector's measured primitive.
    for freq, strafe in itertools.product((0.5, 0.9, 1.4), (-0.5, 0.5, -0.9, 0.9, -1.3, 1.3)):
        rows.append(dict(family="lateral", freq=freq, thigh_amp=0.10, calf_amp=0.40,
                         strafe_amp=strafe, pivot_amp=0.0))
    # There is no useful yaw-dominant pretraining goal, but retain a smaller two-sign screen for
    # the requested future yaw controller and reject it unless yaw becomes decisively dominant.
    for freq, pivot in itertools.product((0.6, 1.0), (-0.8, 0.8, -1.4, 1.4, -2.0, 2.0)):
        rows.append(dict(family="yaw", freq=freq, thigh_amp=0.45, calf_amp=0.50,
                         strafe_amp=0.0, pivot_amp=pivot))
    return rows


def cli_name(index, cfg, repeat):
    return f"{index:03d}_{cfg['family']}_r{repeat}"


def distance_to_goal(family, motion):
    channel = {"forward": 0, "lateral": 1, "yaw": 2}[family]
    value = float(motion[channel])
    if family == "forward":
        lo, hi = 0.12, 0.19
        return max(lo - value, 0.0, value - hi)
    if family == "lateral":
        # The measured source spans both signs; zero itself is intentionally not a useful target.
        magnitude = abs(value)
        return max(0.03 - magnitude, 0.0, magnitude - 0.1233)
    return -abs(value)  # no trustworthy yaw target range: rank decisive yaw, do not claim coverage


def summarize(out_dir, expected_family):
    attempts = []
    for path in sorted(out_dir.glob("attempts/*.yaml")):
        record = yaml.safe_load(path.read_text())
        family = expected_family[path.stem.rsplit("_r", 1)[0]]
        motion = np.asarray(record["mean_body_froude"], dtype=float)
        dominant = ("forward", "lateral", "yaw")[int(np.argmax(np.abs(motion)))]
        attempts.append({
            "attempt": path.stem,
            "family": family,
            "status": record["status"],
            "dominant_family": dominant,
            "family_dominant": dominant == family,
            "target_distance": distance_to_goal(family, motion),
            "forward": motion[0], "lateral": motion[1], "yaw": motion[2],
            "min_height": record["min_height"], "min_upright": record["min_upright"],
        })
    fields = list(attempts[0]) if attempts else []
    with (out_dir / "attempts.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader(); writer.writerows(attempts)

    grouped = {}
    for row in attempts:
        key = row["attempt"].rsplit("_r", 1)[0]
        grouped.setdefault(key, []).append(row)
    ranking = []
    for key, group in grouped.items():
        stable = [r for r in group if r["status"] == "upright"]
        valid = [r for r in stable if r["family_dominant"]]
        ranking.append({
            "config": key,
            "family": group[0]["family"],
            "attempts": len(group),
            "survival_rate": len(stable) / len(group),
            "dominant_rate": len(valid) / len(group),
            "best_target_distance": min((r["target_distance"] for r in valid), default=float("inf")),
            "best_attempt": min(valid, key=lambda r: r["target_distance"])["attempt"] if valid else "",
        })
    ranking.sort(key=lambda r: (-r["dominant_rate"], r["best_target_distance"], -r["survival_rate"]))
    with (out_dir / "ranking.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ranking[0]) if ranking else [])
        writer.writeheader(); writer.writerows(ranking)
    return attempts, ranking


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/wm/dataset/b1_babble/coppelia_aggressive_pilot")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--noise", type=float, default=0.03)
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--limit", type=int, default=0, help="debug: run only the first N configs")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    out_dir = ROOT / args.out
    (out_dir / "attempts").mkdir(parents=True, exist_ok=True)
    grid = configs()[:args.limit or None]
    expected = {cli_name(i, cfg, 0).rsplit("_r", 1)[0]: cfg["family"]
                for i, cfg in enumerate(grid)}
    manifest = {"purpose": "aggressive target-aware designed-CPG screening",
                "goal_source": "data/egocentric/beh12_c10f10t10_ego_flat",
                "goal_ranges": {"forward": [0.12, 0.19], "lateral": [-0.1233, 0.0694],
                                "yaw": None},
                "repeats": args.repeats, "steps": args.steps, "noise": args.noise,
                "configs": grid}
    (out_dir / "config.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    if not args.summarize_only:
        total = len(grid) * args.repeats
        done = 0
        for index, cfg in enumerate(grid):
            for repeat in range(args.repeats):
                name = cli_name(index, cfg, repeat)
                output = out_dir / "attempts" / f"{name}.npz"
                cmd = [sys.executable, str(COLLECTOR), "--screen-only", "--steps", str(args.steps),
                       "--noise", str(args.noise), "--seed", str(repeat), "--port", str(args.port),
                       "--out", str(output)]
                for key in ("freq", "thigh_amp", "calf_amp", "strafe_amp", "pivot_amp"):
                    cmd += ["--" + key.replace("_", "-"), str(cfg[key])]
                done += 1
                print(f"\n=== [{done}/{total}] {name}: {cfg} ===", flush=True)
                subprocess.run(cmd, cwd=ROOT, check=True)
    attempts, ranking = summarize(out_dir, expected)
    print(f"\n{len(attempts)} attempts -> {out_dir / 'attempts.csv'}")
    print(f"ranked configs -> {out_dir / 'ranking.csv'}")
    for row in ranking[:10]:
        print(row)


if __name__ == "__main__":
    main()
