"""One command, the correct LAC-WM staged adaptation for a body absent from pretrain.

  .venv/bin/python3 -m wm.finetune_new_body \\
      --base_ckpt wm/runs/beh12_hexonly_stopgrad/best.pt \\
      --embodiment b1 --data data/egocentric/beh12_b1_ego_flat \\
      --out_dir wm/runs/b1_adapt

**Read this before running anything -- it is the fine-tune manual, not just a script.**

This project spent real time this session (F200, F200a) running the WRONG mechanism: `wm.train
--init_ckpt` warm-started and then jointly retrained ITM+FTM+the full MotionDecoder (every action
head)+body_head+the adversarial probe together, under the FULL multi-task pretrain loss. That is
not adaptation -- it is closer to "resume pretraining with a new source mixed in", and it disturbs
far more than a new body's grounding needs. This project already has, and had previously
VALIDATED (F131: B1 calibrated to +0.79 correlation; F132: cross-embodiment selection cleared
chance at 35-38%), a much narrower staged procedure, taken from LAC-WM and extended once (F130/F131)
for the reason given at stage 4 below. **This file exists so nobody has to remember four separate
scripts and their argument names again.**

    stage 1  wm.adapt          fine-tune ONLY the ITM and FTM on the new body's own clips.
                                Encoder frozen (as everywhere in this project), decoder and
                                body_head frozen too -- "the frozen model is worse than assuming
                                the frame does not move" is the problem this solves, and nothing
                                downstream of the ITM/FTM needs to move to fix it.

    stage 2  (in this file)    fit the action projector -- `a -> z` -- against the NOW-ADAPTED
                                ITM's z, not the original pretrain's. Stage 1 moved what z means
                                for this body, so refitting against the stale z would be fitting
                                the wrong target. "The inverse model cannot run in the loop, so
                                something must turn an action into z" is the reason a projector
                                exists at all (see `wm/models/action_projector.py`).
                                `wm/fit_projector.py` is hardcoded to exactly two embodiment names
                                (hexapod, b1) and cannot fit a third; this stage is a small,
                                generic, single-embodiment version of the same fitting loop.

    stage 3  wm.adapt3         OPTIONAL: jointly fine-tune projector and FTM together. Only
                                needed if stage 2's plain MSE regression fails, which `wm/adapt3.py`
                                traces to a specific cause (F97): if the SAME action recurs in
                                different states and is followed by different transitions --
                                true of B1's PPO-policy actions, a response to state -- then
                                `a -> z` is one-to-many and no amount of stage-2 data fixes it.
                                "Freezing the forward model and making the projector chase z
                                exactly is not enough -- the forward model has to move to meet
                                it" is stage 3's whole point. Open-loop CPG/babble actions (gecko,
                                and any body whose babble is a scripted gait, not a trained
                                policy) likely do NOT have this problem -- the same 16 numbers
                                mean the same thing regardless of state -- so stage 3 is skippable
                                by default here and only worth turning on if stage 2's rollout gap
                                looks bad.

    stage 4  wm.fit_body_head  refit ONLY the shared body_head (~8k parameters, everything else
                                frozen) against whichever latent the planner will actually consume
                                (the projector's z, matching control time -- not the ITM's, which
                                is only ever available with the future frame in hand). This step
                                did not exist in vanilla LAC-WM; it was added here (F130/F131)
                                because F128 put `body_head(proj(a))` on the actual control path,
                                so a body_head fit only once, on the original pretrain bodies,
                                and never revisited, is exactly the kind of stale-target mismatch
                                stage 2 already exists to avoid -- just one level up the pipeline.

**For a genuinely held-out body (never in ANY pretrain), `--data` must be its BABBLE clips, not a
scripted/expert demonstration set.** A real deployment on an unseen body has motor babbling and
nothing else -- that is the whole premise of claim (3). B1's own adaptation runs in this project's
history (F97, F130-F136) used its PPO-policy clips because that is what existed for B1 at the
time; a body with no policy at all (gecko, or any future third+ body) uses babble for every stage
here, same as the projector fit already does.

**If another paper's cross-embodiment method skips these steps, ask what stands in for them.**
Any method that claims a new body's video is enough, with no adaptation of its own dynamics model
and no adaptation of its inverse map, is claiming something LAC-WM's own ablations (and this
project's F97) found does not hold for a body whose actions are not simply invertible. It may be
that a different architecture does not need a stage-1-equivalent (e.g. one that never encodes a
per-embodiment inverse map at all) -- but "we didn't adapt anything and it still worked" is worth
checking for a hidden stage 2/3/4 equivalent before taking it at face value.
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import from_checkpoint  # noqa: E402
from wm.fit_projector import gather  # noqa: E402
from wm.models.action_projector import ActionProjector  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

PY = sys.executable


def run(cmd):
    print(f"\n$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"stage failed (exit {result.returncode}): {' '.join(cmd)}")


def stage2_fit_projector(adapted_ckpt, embodiment, data_dir, cache, epochs, lr, val_frac, out):
    """Generic, single-embodiment version of `wm/fit_projector.py`'s fitting loop -- that script
    is hardcoded to ("hexapod", "b1") and cannot fit a third name. Same method, same reporting
    convention (z MSE and rollout gap, both against predicting the mean z), any embodiment name.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(adapted_ckpt, map_location="cpu", weights_only=False)
    cfg = from_checkpoint(checkpoint["config"])
    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(checkpoint["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(checkpoint["ftm"])
    for p in list(itm.parameters()) + list(ftm.parameters()):
        p.requires_grad_(False)

    cache_path = os.path.join(ROOT, cache)
    cache_dict = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    before = len(cache_dict)
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    lag = max(1, cfg.action_lag)

    e, z, a, c, paths = gather(embodiment, os.path.join(ROOT, data_dir), encoder, itm, checkpoint,
                               cache_dict, chunk=2, lag=lag, device=device)
    if len(cache_dict) > before:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cache_dict, cache_path)
    del encoder, cache_dict
    torch.cuda.empty_cache()
    print(f"{embodiment}: {len(paths)} clips, {len(a)} transitions gathered")

    proj = ActionProjector(cfg, {embodiment: a.shape[1]}).to(device)
    proj.set_stats(embodiment, a.mean(0).cpu(), a.std(0).cpu())

    ids = torch.unique(c)
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(0))
    val_ids = ids[order[:max(1, int(val_frac * len(ids)))]]
    val_mask = torch.isin(c, val_ids).to(device)
    val_paths = [paths[i] for i in val_ids.tolist()]
    print(f"{embodiment}: {len(ids)} clips, {int(val_mask.sum())} of {len(c)} transitions held out")

    opt = torch.optim.Adam(proj.parameters(), lr=lr)
    m = ~val_mask
    for epoch in range(epochs):
        proj.train(); opt.zero_grad()
        loss = torch.nn.functional.mse_loss(proj(a[m], embodiment), z[m])
        loss.backward(); opt.step()
        if (epoch + 1) % 50 == 0:
            print(f"epoch {epoch + 1:4d}  train {loss.item():.4f}")

    proj.eval()
    with torch.no_grad():
        zp = proj(a[val_mask], embodiment)
        base = z[m].mean(0, keepdim=True).expand_as(z[val_mask])
        z_mse = torch.nn.functional.mse_loss(zp, z[val_mask]).item()
        z_base = torch.nn.functional.mse_loss(base, z[val_mask]).item()
        idx = torch.nonzero(val_mask.cpu(), as_tuple=True)[0]
        gap_num = gap_den = 0.0
        for i in range(0, len(idx), 32):
            sl = idx[i:i + 32]
            e_b = e[sl].to(device).float()
            truth = ftm(e_b, z[val_mask][i:i + 32])
            gap_num += ((ftm(e_b, zp[i:i + 32]) - truth) ** 2).mean().item() * len(sl)
            gap_den += ((ftm(e_b, base[i:i + 32]) - truth) ** 2).mean().item() * len(sl)
            del e_b, truth
        gap, gap_base = gap_num / len(idx), gap_den / len(idx)

    print(f"\n{'embodiment':<12}{'z MSE':>10}{'vs mean-z':>11}{'rollout gap':>14}{'vs mean-z':>11}")
    print(f"{embodiment:<12}{z_mse:>10.4f}{z_mse / max(z_base, 1e-9):>11.3f}"
         f"{gap:>14.4f}{gap / max(gap_base, 1e-9):>11.3f}")
    print("\nRatio below 1.0 beats predicting the mean z; the rollout column is what a")
    print("planner/selector actually consumes.")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"projector": proj.state_dict(), "ckpt": adapted_ckpt, "val_paths": {embodiment: val_paths},
               "action_dims": {embodiment: a.shape[1]}}, out)
    print(f"-> {os.path.relpath(out, ROOT)}")
    return out, gap / max(gap_base, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", required=True, help="the pretrain checkpoint that never saw "
                    "this body -- stage 1's warm start")
    ap.add_argument("--embodiment", required=True, help="registry name in wm/data/embodiment.py")
    ap.add_argument("--data", required=True, help="the new body's own clips -- BABBLE, not a "
                    "scripted/expert set, for a body genuinely absent from pretrain")
    ap.add_argument("--out_dir", required=True)
    # stage 1
    ap.add_argument("--adapt_clips", type=int, default=9)
    ap.add_argument("--adapt_test_clips", type=int, default=10)
    ap.add_argument("--adapt_steps", type=int, default=1000)
    ap.add_argument("--adapt_lr", type=float, default=1e-4)
    # stage 2
    ap.add_argument("--proj_epochs", type=int, default=200)
    ap.add_argument("--proj_lr", type=float, default=1e-3)
    ap.add_argument("--proj_val_frac", type=float, default=0.2)
    # stage 3 (opt-in)
    ap.add_argument("--stage3", action="store_true", help="only turn on if stage 2's rollout "
                    "gap ratio is bad (near or above 1.0) -- see this file's own docstring for "
                    "why open-loop babble bodies likely do not need it")
    ap.add_argument("--stage3_steps", type=int, default=3000)
    ap.add_argument("--stage3_lambda_nce", type=float, default=0.0)
    # stage 4
    ap.add_argument("--body_head_latent", choices=("itm", "projector", "both"), default="both")
    ap.add_argument("--body_head_epochs", type=int, default=400)
    args = ap.parse_args()

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78); print(f"STAGE 1 -- adapt ITM+FTM to {args.embodiment}, everything else frozen")
    print("=" * 78)
    stage1_out = os.path.join(out_dir, f"adapted_{args.embodiment}.pt")
    run([PY, "-m", "wm.adapt", "--ckpt", args.base_ckpt, "--data", args.data,
        "--embodiment", args.embodiment, "--clips", str(args.adapt_clips),
        "--test_clips", str(args.adapt_test_clips), "--steps", str(args.adapt_steps),
        "--lr", str(args.adapt_lr), "--out", stage1_out])

    print("\n" + "=" * 78); print(f"STAGE 2 -- fit the action projector against the ADAPTED itm")
    print("=" * 78)
    stage2_out = os.path.join(out_dir, f"projector_{args.embodiment}.pt")
    _, gap_ratio = stage2_fit_projector(
        stage1_out, args.embodiment, args.data,
        cache=f"results/wm/cache/finetune_{args.embodiment}_embeddings.pt",
        epochs=args.proj_epochs, lr=args.proj_lr, val_frac=args.proj_val_frac, out=stage2_out)

    if args.stage3:
        print("\n" + "=" * 78); print("STAGE 3 -- joint projector+FTM fine-tune (opted in)")
        print("=" * 78)
        stage3_ckpt = os.path.join(out_dir, f"stage3_{args.embodiment}.pt")
        run([PY, "-m", "wm.adapt3", "--ckpt", stage1_out, "--projector", stage2_out,
            "--data", args.data, "--embodiment", args.embodiment,
            "--steps", str(args.stage3_steps), "--lambda_nce", str(args.stage3_lambda_nce),
            "--out", stage3_ckpt])
        # adapt3's own output already bundles itm/ftm/md/projector together -- no merge needed.
        merged_ckpt = stage3_ckpt
    else:
        print(f"\n(stage 3 skipped -- stage 2's rollout gap ratio was {gap_ratio:.3f}; turn on "
             "--stage3 if this looks bad, near or above 1.0)")
        # `wm.fit_body_head` needs one checkpoint carrying BOTH the (adapted) itm/ftm and the
        # projector; stage 1's output has the former, stage 2's has only the latter, and skipping
        # stage 3 means nothing has bundled them yet. `wm.assemble_teacher` does exactly that.
        print("\n" + "=" * 78); print("merging stage 1 + stage 2 (wm.assemble_teacher) -- "
             "wm.fit_body_head needs one checkpoint carrying both")
        print("=" * 78)
        merged_ckpt = os.path.join(out_dir, f"teacher_{args.embodiment}.pt")
        run([PY, "-m", "wm.assemble_teacher", "--base", stage1_out, "--projector", stage2_out,
            "--out", merged_ckpt])

    print("\n" + "=" * 78); print("STAGE 4 -- refit the shared body_head against the latent it "
         "will actually be shown")
    print("=" * 78)
    body_head_out = os.path.join(out_dir, f"body_head_{args.embodiment}.pt")
    run([PY, "-m", "wm.fit_body_head", "--ckpt", merged_ckpt, "--data", args.data,
        "--embodiment", args.embodiment, "--latent", args.body_head_latent,
        "--epochs", str(args.body_head_epochs), "--out", body_head_out])

    print("\n" + "=" * 78); print("DONE"); print("=" * 78)
    print(f"stage 1 (adapted itm/ftm)  -> {os.path.relpath(stage1_out, ROOT)}")
    print(f"stage 2 (projector)       -> {os.path.relpath(stage2_out, ROOT)}")
    print(f"merged/stage 3 checkpoint -> {os.path.relpath(merged_ckpt, ROOT)}")
    print(f"stage 4 (body_head)       -> {os.path.relpath(body_head_out, ROOT)}")


if __name__ == "__main__":
    main()
