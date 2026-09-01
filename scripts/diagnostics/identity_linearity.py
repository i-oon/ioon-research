"""Is embodiment identity actually removed from `z`, or only hidden from a straight line?

The probe standardises each embodiment's features before fitting, and a **linear** classifier then
reads the robot at chance -- which is what licenses the claim that its transfer numbers are not
identity leaking through. The `z_umap.png` figure, drawn with UMAP, nonetheless shows the two robots
in different regions, and a picture and a number that appear to disagree is not something to put on
a slide unexplained.

They do not disagree. They answer different questions, because one method is linear and the other is
not. Running classifiers of both kinds on the same features settles it:

    classifier                    raw z   standardised
    linear (logistic)             1.000       0.460
    nonlinear (random forest)     1.000       0.999
    nonlinear (MLP)               1.000       1.000

**Standardising removes identity from what a straight line can use, and from nothing else.** The
information is entirely present; a nonlinear reader recovers it exactly. This is the same lesson as
the `center_embeddings` run, where subtracting each robot's mean embedding left the online probe
free to climb back to 1.000 within 25 epochs -- first and second moments are not where the identity
lives.

Two consequences worth keeping straight:

  the probe's transfer numbers are clean   a linear ridge cannot exploit what a linear classifier
                                           cannot find, so the shared-axis result stands
  `z` is not a unified space               it encodes the robot exactly, just not linearly, so no
                                           claim of the form "the latent forgets the body" is
                                           available to us

  .venv/bin/python3 scripts/diagnostics/identity_linearity.py
"""
import sys, glob, numpy as np, torch
sys.path.insert(0,"/home/aria/ioon-research"); sys.path.insert(0,"/home/aria/ioon-research/scripts")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from wm.config import from_checkpoint
from wm.evaluate import offset_for
from wm.models.itm import InverseTransitionModel
from diagnostics.body_motion_probe import gather, standardise
R="/home/aria/ioon-research"
from vjepa2_encoder import VJEPA2FrameEncoder
enc=VJEPA2FrameEncoder(device="cuda", dtype=torch.float32)
ck=torch.load(f"{R}/wm/runs/s2_fwd_hex7-b1_body0.5/last.pt", map_location="cpu", weights_only=False)
itm=InverseTransitionModel(from_checkpoint(ck["config"])); itm.load_state_dict(ck["itm"]); itm.eval()
d=gather(enc, 2, "data/allocentric/fwd_hex7speed", f"{R}/results/wm/cache/probe_ik_walk_speed7.pt", itm, ck)
del enc
raw=np.concatenate([d[n][0] for n in ("insect","b1")])
std=np.concatenate([standardise(d[n][0]) for n in ("insect","b1")])
who=np.concatenate([np.full(len(d[n][1]),i) for i,n in enumerate(("insect","b1"))])
clip=np.concatenate([d[n][2]+1000*i for i,n in enumerate(("insect","b1"))])
rng=np.random.default_rng(0); ids=np.unique(clip); rng.shuffle(ids)
tr=np.isin(clip, ids[:int(.7*len(ids))])
models={"linear (logistic)": LogisticRegression(max_iter=3000),
        "nonlinear (random forest)": RandomForestClassifier(n_estimators=200, random_state=0),
        "nonlinear (MLP)": MLPClassifier(hidden_layer_sizes=(64,), max_iter=600, random_state=0)}
print(f"{'classifier':<28}{'raw z':>10}{'standardised':>15}")
for name, m in models.items():
    row=[]
    for x in (raw, std):
        m.fit(x[tr], who[tr])
        s=m.predict_proba(x[~tr])[:,1] if hasattr(m,"predict_proba") else m.decision_function(x[~tr])
        row.append(roc_auc_score(who[~tr], s))
    print(f"{name:<28}{row[0]:>10.3f}{row[1]:>15.3f}")
