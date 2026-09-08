"""TEST 1 -- ground-truth flatness check, gates the sequence-context kill-gate below.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/ground_truth_action_flatness.py

Pure ground truth, no model in the loop at all: uses only the real recorded actions and the
real (sim-measured, dt-derived) body_motion (Froude) arrays straight from the B1 npz clips.

**Why this was checked.** F198's addenda ruled out loss target, gradient share, and two
architectures (pooled-GRU recurrent state, and a two-frame pooled proxy) as fixes for the FTM's
action-lever staying near +0.042-+0.055 -- the single-step model barely distinguishes the real
recorded action from a generic/mean one when predicting outcome direction. Before scoping a
heavier sequence-context architecture (`sequence_context_killgate.py`), this asks the prior
question directly: is there even a real action->outcome relationship in the DATA for a better
model to capture, or is "reach steady Froude" flat within a fixed behaviour regardless of the
exact action -- in which case no architecture fixes it.

Question: within a fixed behavior/condition (same target speed/turn/side level, 4 real clips
each), does the REAL action a clip actually used differ from the condition's mean action, and
does its REAL resulting Froude differ correspondingly from the condition's mean Froude?

  - If real-action deviations predict real-Froude deviations above chance -> signal exists in the
    data; the model's action-lever gap is a MODEL failure to capture something that is there.
    Proceed to the sequence-context kill-gate.
  - If real-action deviations do NOT predict real-Froude deviations above chance -> the flatness
    is already in the ground truth, not introduced by the model. No amount of history/sequence
    context fixes a target that doesn't depend on the action within a fixed behavior. Stop.

Also reports: within one clip (one behavior instance), is Froude roughly CONSTANT over the
interior of the clip, or does it drift/change? Decides Froude vs delta-Froude as the sequence
kill-gate's target, if that test proceeds.
"""
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from wm.data.embodiment import REGISTRY, load  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data/egocentric/beh12_b1_ego_flat")
CHANNELS = (0, 1, 2)  # forward, lateral, yaw -- same 3 channels used throughout this arc's state work
WARMUP_FRAC = 0.15    # drop this fraction of frames at each end as transient/edge-of-smoothing artifact

paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
by_condition = {}
for p in paths:
    clip = load(p, REGISTRY["b1"])
    cond = str(np.load(p, allow_pickle=True)["condition"])
    n = len(clip["actions"])
    lo, hi = int(n * WARMUP_FRAC), n - int(n * WARMUP_FRAC)
    lo, hi = max(lo, 1), min(hi, n - 1)
    a = clip["actions"][lo:hi]                       # (T,12)
    f = clip["body_motion"][lo:hi][:, CHANNELS]       # (T,3)
    by_condition.setdefault(cond, []).append({"a_mean": a.mean(0), "f_mean": f.mean(0),
                                              "a_seq": a, "f_seq": f})

print(f"{len(paths)} clips, {len(by_condition)} conditions, "
      f"{[len(v) for v in by_condition.values()]} clips/condition\n")

# ---------------------------------------------------------------------------------------------
# Part A: real action deviation -> real Froude deviation, within-condition (behavior fixed)
# ---------------------------------------------------------------------------------------------
da_list, df_list, cond_id_list = [], [], []
for ci, (cond, clips) in enumerate(by_condition.items()):
    a_cond_mean = np.mean([c["a_mean"] for c in clips], axis=0)
    f_cond_mean = np.mean([c["f_mean"] for c in clips], axis=0)
    for c in clips:
        da_list.append(c["a_mean"] - a_cond_mean)
        df_list.append(c["f_mean"] - f_cond_mean)
        cond_id_list.append(ci)

DA = np.stack(da_list)   # (N,12) real-action deviation from its condition's mean action
DF = np.stack(df_list)   # (N,3)  real-Froude deviation from its condition's mean Froude
cond_id = np.array(cond_id_list)
N = len(DA)

# magnitude correlation: does a clip whose action deviates more also have Froude deviate more?
da_mag = np.linalg.norm(DA, axis=1)
df_mag = np.linalg.norm(DF, axis=1)
mag_corr = np.corrcoef(da_mag, df_mag)[0, 1]

# permutation test on the magnitude correlation -- honest given N=48: shuffle which clip's
# action-deviation goes with which clip's Froude-deviation, see how often chance alone beats
# the real correlation. Handles the "12-D fit on 48 points" overfit risk without needing an
# unstable regression at all.
rng = np.random.default_rng(0)
n_perm = 20000
perm_corrs = np.empty(n_perm)
for i in range(n_perm):
    perm_corrs[i] = np.corrcoef(da_mag, rng.permutation(df_mag))[0, 1]
perm_p = float(np.mean(np.abs(perm_corrs) >= abs(mag_corr)))

# ridge-regularized leave-one-condition-out R^2 (honest alternative to an unregularized 12-D OLS
# fit, which severely overfits with only ~44 training points per fold: in-sample R^2 was 0.797
# but unregularized LOO R^2 was -0.621, i.e. worse than predicting zero -- pure overfitting)
RIDGE_LAMBDA = 5.0
loo_res, loo_tot = 0.0, 0.0
for ci in range(len(by_condition)):
    train = cond_id != ci
    test = cond_id == ci
    if train.sum() < 13:
        continue
    Xtr = DA[train]
    Wc = np.linalg.solve(Xtr.T @ Xtr + RIDGE_LAMBDA * np.eye(Xtr.shape[1]), Xtr.T @ DF[train])
    pred = DA[test] @ Wc
    loo_res += ((DF[test] - pred) ** 2).sum()
    loo_tot += (DF[test] ** 2).sum()
r2_loo_ridge = 1 - loo_res / loo_tot

print("=" * 70)
print("PART A: real-action deviation vs real-Froude deviation, within a fixed behavior")
print("=" * 70)
print(f"N = {N} clips across {len(by_condition)} conditions (4 clips/condition)")
print(f"corr(|real action deviation|, |real Froude deviation|):  {mag_corr:+.3f}")
print(f"permutation p-value (n={n_perm}, is this corr above chance?):  {perm_p:.4f}")
print(f"R^2 (ridge-regularized leave-one-condition-out, lambda={RIDGE_LAMBDA}): {r2_loo_ridge:.3f}")

# ---------------------------------------------------------------------------------------------
# Part B: within one clip, is Froude roughly constant over the interior, or does it change?
# ---------------------------------------------------------------------------------------------
within_clip_cv = []   # coefficient of variation of Froude channel-0 (forward) over one clip's interior
within_clip_trend = []  # start-vs-end difference (normalized), a directional drift measure
for clips in by_condition.values():
    for c in clips:
        f0 = c["f_seq"][:, 0]  # forward-speed Froude channel
        mu = np.mean(np.abs(f0)) + 1e-6
        within_clip_cv.append(np.std(f0) / mu)
        k = max(1, len(f0) // 5)
        start, end = f0[:k].mean(), f0[-k:].mean()
        within_clip_trend.append(abs(end - start) / mu)

across_condition_cv = np.std([np.mean([c["f_mean"][0] for c in clips]) for clips in by_condition.values()]) / \
    (np.mean([abs(np.mean([c["f_mean"][0] for c in clips])) for clips in by_condition.values()]) + 1e-6)

print("\n" + "=" * 70)
print("PART B: is Froude constant or changing within one clip? (forward-speed channel)")
print("=" * 70)
print(f"within-clip coefficient of variation (std/mean over the clip's interior): "
      f"{np.mean(within_clip_cv):.3f} +/- {np.std(within_clip_cv):.3f}")
print(f"within-clip start-vs-end normalized drift:                                "
      f"{np.mean(within_clip_trend):.3f} +/- {np.std(within_clip_trend):.3f}")
print(f"(reference) across-condition CV of the condition-level mean Froude:       {across_condition_cv:.3f}")

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
if perm_p < 0.05 and r2_loo_ridge > 0.05:
    print(f"-> SIGNAL EXISTS in ground truth (perm p={perm_p:.4f}, ridge-LOO R^2={r2_loo_ridge:.3f}): "
          "real action differs from mean action, and real Froude differs correspondingly, within "
          "a fixed behavior, above what chance pairing alone would give. The model's action-lever "
          "gap is a MODEL failure to capture something present in the data. The sequence-context "
          "kill-gate is worth running.")
else:
    print(f"-> FLAT in ground truth (perm p={perm_p:.4f}, ridge-LOO R^2={r2_loo_ridge:.3f}): real "
          "action deviations from the behavior's mean do not predict real Froude deviations above "
          "chance. The flatness is in the TASK, not the model -- 'reach steady Froude' has no "
          "within-behavior gradient to learn regardless of architecture. Report and stop; do not "
          "run the sequence-context kill-gate.")
if np.mean(within_clip_trend) > 0.5 * np.mean(within_clip_cv) and np.mean(within_clip_trend) > 0.15:
    print("-> Froude drifts noticeably within a clip (start != end) -- if the sequence kill-gate "
          "proceeds, prefer delta-Froude as the target, not absolute Froude.")
else:
    print("-> Froude is roughly steady within a clip's interior -- absolute Froude is a reasonable "
          "target, delta-Froude not required on this basis alone.")
