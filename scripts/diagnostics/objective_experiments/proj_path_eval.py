import glob, os, sys
import numpy as np, torch
from scipy.stats import spearmanr
ROOT = "/home/fibo07/ioon/ioon-research"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from wm.config import from_checkpoint
from wm.data.embodiment import REGISTRY, load
from wm.models.itm import InverseTransitionModel
from wm.models.motion_decoder import MotionDecoder
from wm.models.action_projector import ActionProjector, action_dims_from
sys.path.insert(0, os.path.join(ROOT, "scripts/diagnostics/objective_experiments"))
from eval_body_head_true_heldout import embed_and_target, score

dev = torch.device("cuda")
CK = "wm/runs/beh12_hinge_cleansplit/b1_adapt_clean/body_head_b1_hex_clean.pt"
PJ = "wm/runs/beh12_hinge_cleansplit/b1_adapt_clean/projector_clean.pt"
ck = torch.load(os.path.join(ROOT, CK), map_location="cpu", weights_only=False)
cfg = from_checkpoint(ck["config"]); ch = [int(c) for c in cfg.body_channels]
mean = torch.tensor(np.asarray(ck["body_stats"][0]).ravel(), dtype=torch.float32)
std = torch.tensor(np.asarray(ck["body_stats"][1]).ravel(), dtype=torch.float32)
itm = InverseTransitionModel(cfg).to(dev).eval(); itm.load_state_dict(ck["itm"])
md = MotionDecoder(cfg, None).to(dev).eval(); md.load_state_dict(ck["md"], strict=False)
sv = torch.load(os.path.join(ROOT, PJ), map_location="cpu", weights_only=False)
proj = ActionProjector(cfg, action_dims_from(sv)).to(dev).eval()
proj.load_state_dict(sv.get("projector", sv))
lag = max(1, cfg.action_lag)
spec = REGISTRY["b1"]
held = sorted(glob.glob(os.path.join(ROOT, "data/egocentric/beh12_b1_ego_flat_cleanheldout/*.npz")))
cache = os.path.join(ROOT, "results/wm/cache/eval_true_heldout_b1.pt")
z_itm, y = embed_and_target(None, itm, held, spec, ch, 2, cache)
zp = []
with torch.no_grad():
    for p in held:
        c = load(p, spec)
        n = min(len(np.load(p, allow_pickle=True)["frames"]) - 1, len(np.asarray(c["body_motion"])) - 1)
        a = torch.tensor(np.asarray(c["actions"])[lag:lag + n], dtype=torch.float32, device=dev)
        zp.append(proj(a, "b1").cpu())
z_proj = torch.cat(zp)
print("transitions:", len(z_itm), len(z_proj))
m = min(len(z_itm), len(z_proj))
for name, z in (("ITM(e_t,e_t+1)", z_itm[:m]), ("proj(action)  ", z_proj[:m])):
    err, base, ratio, raw, rhos = score(md, z, y[:m], mean, std, dev)
    print(f"{name}  ratio {ratio:.3f}  rho fwd {rhos[0]:+.3f} lat {rhos[1]:+.3f} yaw {rhos[2]:+.3f}  median {np.median(rhos):+.3f}")
