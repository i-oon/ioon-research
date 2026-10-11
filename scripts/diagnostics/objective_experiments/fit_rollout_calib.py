"""Per-channel de-shrink of the rollout read (F340), fitted on VALIDATION branches only (2026-10-10).

Input: a `counterfactual_readout.py --dump` on `*_branches_val` (P11, "FTM cf" = the rollout read, "truth"). Within each state
(24 commands) the slope of the read on the truth and the means are pooled over states: slope = sum(tc * rc) / sum(tc * tc),
m_read / m_true = the overall means. The planner then scores read' = (read - m_read) / slope + m_true (Froude units).

    .venv/bin/python3 scripts/diagnostics/objective_experiments/fit_rollout_calib.py DUMP.npz BODY OUT.json
"""
import json, sys
import numpy as np
dump, body, out = sys.argv[1:4]
d = np.load(dump)
T, R = d["B|P11|truth"], d["B|P11|FTM cf"]
tc = T - T.mean(1, keepdims=True); rc = R - R.mean(1, keepdims=True)
slope = (tc * rc).sum((0, 1)) / (tc * tc).sum((0, 1))
cal = dict(slope=slope.tolist(), m_read=R.reshape(-1, 3).mean(0).tolist(), m_true=T.reshape(-1, 3).mean(0).tolist(),
           source=dump, n_states=int(len(T)))
J = json.load(open(out)) if __import__("os").path.exists(out) else {}
J[body] = cal
json.dump(J, open(out, "w"), indent=1)
print(body, {k: np.round(v, 3).tolist() if isinstance(v, list) else v for k, v in cal.items()})
