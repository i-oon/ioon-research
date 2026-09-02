"""Put the egocentric teacher into one file, because `load_teacher` wants one file.

    .venv/bin/python3 -m wm.assemble_teacher --base wm/runs/beh12_ego/best.pt \\
        --projector wm/runs/beh12_ego/projector_ego.pt \\
        --decoder wm/runs/beh12_ego/md_refit.pt \\
        --out wm/runs/beh12_ego/teacher_ego.pt

**P1, and it deliberately does not run stage 3 first.** `sim/control/teacher_student_insect.py`'s
`load_teacher` needs `itm`, `ftm`, `md` and `projector` in one checkpoint. The egocentric run is a
stage-1 pretrain and carries no projector; the missing parts were fitted separately and each is the
one the gates were measured on:

    itm, ftm     `beh12_ego/best.pt`      -- GATE B and GATE C (F172) are this model
    projector    `projector_ego.pt`       -- GATE D2 (F173), rollout gap ratio 0.355 / 0.169
    md           `md_refit.pt`            -- F174, held-out R2 0.847 insect / 0.778 B1

**Adapting with `wm/adapt3` first would put the F145 gate on a model nothing has measured**, and
stage 3's seed ordering is still unresolved. Every number the gate will be read against was taken on
these exact weights, so they are what gets assembled.

**Nothing is retrained here and no tensor is altered** -- this is a merge, and it prints what came
from where so the resulting file cannot be mistaken for a training artefact.
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="the pretrain: supplies itm, ftm and config")
    ap.add_argument("--projector", required=True)
    ap.add_argument("--decoder", default="", help="the refitted md; omit to keep the base's")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    load = lambda p: torch.load(os.path.join(ROOT, p), map_location="cpu", weights_only=False)
    base = load(args.base)
    proj = load(args.projector)
    out = dict(base)

    # `wm/fit_projector` saves either the bare state dict or a dict carrying it, and the two have
    # been mixed up before; take whichever is present rather than assuming.
    out["projector"] = proj.get("projector", proj) if isinstance(proj, dict) else proj
    if isinstance(proj, dict) and "action_dims" in proj:
        out["action_dims"] = proj["action_dims"]
    src = {"itm": args.base, "ftm": args.base, "projector": args.projector, "md": args.base}

    if args.decoder:
        dec = load(args.decoder)
        out["md"] = dec.get("md", dec) if isinstance(dec, dict) else dec
        src["md"] = args.decoder
        if isinstance(dec, dict) and "best_test_r2" in dec:
            out["md_test_r2"] = dec["best_test_r2"]

    for key in ("itm", "ftm", "md", "projector"):
        if key not in out:
            raise SystemExit(f"missing {key!r} -- load_teacher would fail at run time instead")

    path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(out, path)

    print(f"-> {args.out}")
    for key in ("itm", "ftm", "md", "projector"):
        print(f"  {key:>10}  from {src[key]}")
    if "md_test_r2" in out:
        print("\n  the decoder's own held-out R2, carried so the file states its own limits: "
              + "  ".join(f"{k} {v:.3f}" for k, v in out["md_test_r2"].items()))
    print("\n  **This is a merge of measured parts, not a trained model.** Stage 3 was deliberately")
    print("  not run: every gate this teacher will be judged against was measured on these weights.")


if __name__ == "__main__":
    main()
