"""Audit an IK dataset for cross-morphology correspondence.

Checks walk clips only:
  * actions differ between morphologies;
  * measured contacts match the source expert and each other;
  * body-relative Cartesian foot paths reconstructed by FK match;
  * forward displacement is similar and lateral drift is small.
"""
import argparse
import itertools
import os
import re

import numpy as np
import pandas as pd
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "sim/env/expert_66k_aug3c_fcontact.csv")
ENV = os.path.join(ROOT, "sim/env")
MORPHS = ("long", "medium", "short")
SCENES = {
    "long": "medauroidea_stick_insect.ttt",
    "medium": "medauroidea_stick_insect_medium.ttt",
    "short": "medauroidea_stick_insect_short.ttt",
}
LEGS = ("FL", "ML", "HL", "FR", "MR", "HR")
JOINTS = ("m1", "m2", "m3")
WALK_RE = re.compile(r"^(long|medium|short)_ep(\d+)(?:_r(\d+))?\.npz$")


def majority3(x):
    y = x.copy()
    if len(x) > 2:
        y[1:-1] = ((x[:-2] + x[1:-1] + x[2:]) >= 2).astype(x.dtype)
    return y


def load_groups(path):
    groups = {}
    for name in sorted(os.listdir(path)):
        match = WALK_RE.match(name)
        if not match:
            continue
        morph, ep, rep = match.groups()
        key = (int(ep), int(rep or 0))
        groups.setdefault(key, {})[morph] = os.path.join(path, name)
    return {key: value for key, value in groups.items() if set(value) == set(MORPHS)}


def fk_paths(sim, groups):
    result = {}
    for morph in MORPHS:
        sim.loadScene(os.path.join(ENV, SCENES[morph]))
        abdomen = sim.getObjectParent(sim.getObject("/m1_FL"))
        joints = [sim.getObject(f"/{joint}_{leg}") for leg in LEGS for joint in JOINTS]
        feet = [sim.getObject(f"/foot_{leg}") for leg in LEGS]
        for key, paths in groups.items():
            actions = np.load(paths[morph])["actions"]
            xyz = np.empty((len(actions), len(LEGS), 3), dtype=np.float64)
            for t, action in enumerate(actions):
                for handle, angle in zip(joints, action):
                    sim.setJointPosition(handle, float(angle))
                xyz[t] = [sim.getObjectPosition(foot, abdomen) for foot in feet]
            result[(key, morph)] = xyz
    return result


def describe(values):
    a = np.asarray(values, dtype=float)
    return f"mean={a.mean():.4f} median={np.median(a):.4f} range={a.min():.4f}..{a.max():.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--contact-threshold", type=float, default=0.27)
    parser.add_argument("--skip-fk", action="store_true")
    args = parser.parse_args()

    groups = load_groups(args.dataset)
    if not groups:
        raise SystemExit("no complete long/medium/short walk triplets found")
    expert = pd.read_csv(CSV)
    metrics = {name: [] for name in
               ("action_rms", "contact_cross", "contact_expert", "dx_cv", "drift_ratio")}

    print(f"dataset={args.dataset} complete walk triplets={len(groups)}")
    for (ep, rep), paths in sorted(groups.items()):
        data = {m: np.load(p) for m, p in paths.items()}
        contacts = {m: majority3((data[m]["forces"] > args.contact_threshold).astype(np.int8))
                    for m in MORPHS}
        n = min(len(data[m]["actions"]) for m in MORPHS)
        exp = expert.iloc[ep * 66:ep * 66 + n][[f"contact_{leg}" for leg in LEGS]].to_numpy(np.int8)
        exp = majority3(exp)
        action_rms = [np.sqrt(np.mean((data[a]["actions"][:n] - data[b]["actions"][:n]) ** 2))
                      for a, b in itertools.combinations(MORPHS, 2)]
        cross = [np.mean(contacts[a][:n] != contacts[b][:n])
                 for a, b in itertools.combinations(MORPHS, 2)]
        exp_err = [np.mean(contacts[m][:n] != exp) for m in MORPHS]
        delta = {m: data[m]["head"][n - 1, :2] - data[m]["head"][0, :2] for m in MORPHS}
        dx = np.asarray([delta[m][0] for m in MORPHS])
        drift = [abs(delta[m][1]) / max(abs(delta[m][0]), 1e-6) for m in MORPHS]
        metrics["action_rms"].extend(action_rms)
        metrics["contact_cross"].extend(cross)
        metrics["contact_expert"].extend(exp_err)
        metrics["dx_cv"].append(float(np.std(dx) / max(abs(np.mean(dx)), 1e-6)))
        metrics["drift_ratio"].extend(drift)

    print("action pairwise RMS (rad):       ", describe(metrics["action_rms"]))
    print("contact cross-morph mismatch:   ", describe(metrics["contact_cross"]))
    print("contact mismatch vs expert:     ", describe(metrics["contact_expert"]))
    print("forward displacement CV/triplet:", describe(metrics["dx_cv"]))
    print("absolute lateral/forward drift: ", describe(metrics["drift_ratio"]))

    if not args.skip_fk:
        client = RemoteAPIClient("localhost", port=args.port)
        sim = client.require("sim")
        fk = fk_paths(sim, groups)
        errors = []
        for key in groups:
            for a, b in itertools.combinations(MORPHS, 2):
                n = min(len(fk[(key, a)]), len(fk[(key, b)]))
                errors.append(np.sqrt(np.mean((fk[(key, a)][:n] - fk[(key, b)][:n]) ** 2)) * 1000.0)
        print("FK task-space pairwise RMS (mm):", describe(errors))


if __name__ == "__main__":
    main()
