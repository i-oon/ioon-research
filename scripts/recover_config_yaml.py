"""Write `config.yaml` for a run that predates automatic config saving.

`wm/train.py` writes the yaml at startup, but runs trained before that only have their config
inside the checkpoint -- which is fine until the checkpoint is deleted to free disk, at which
point the record goes with it. Nine run directories are already empty for exactly that reason.

Recovers what the checkpoint knows. The data breakdown a live run records (pairs per embodiment,
clips per body, the balance ratio) is not in the checkpoint, so it is marked as unavailable rather
than guessed.

  .venv/bin/python3 scripts/recover_config_yaml.py wm/runs/stage2_clean
"""
import os
import sys

import torch
import yaml


def main(run_dir):
    for name in ("best.pt", "epoch060.pt", "last.pt"):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            break
    else:
        raise SystemExit(f"{run_dir}: no checkpoint to recover from")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    out = os.path.join(run_dir, "config.yaml")
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists; refusing to overwrite a live record")
    record = {
        "name": os.path.basename(run_dir.rstrip("/")),
        "config": checkpoint["config"],
        "recovered_from": name,
        "epoch_reached": checkpoint.get("epoch", -1),
        "data": "not recorded -- this run predates config.yaml, and the per-embodiment pair "
                "counts are not stored in the checkpoint",
    }
    with open(out, "w") as fh:
        yaml.safe_dump(record, fh, sort_keys=False, default_flow_style=False)
    print(f"-> {out}  (from {name}, epoch {record['epoch_reached']})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__))
