"""Roll out a trained AIRL insect policy in CoppeliaSim + capture frames (fixed cam).

Uses the AIRL CoppeliaSimEnv (drives joints directly) + a trained actor.pth. We remove
the scene's main_script so its CSV gait can't fight the policy, add the fixed vjepa_cam,
then step actor->env and grab a frame each step. Produces frames + a video for eyeballing
gait quality vs the CSV-gait hexapod_v1.

  python3 sim/collect/rollout_insect_airl.py \
     --scene airl-insect-walking/env/medauroidea_stick_insect.ttt \
     --actor airl-insect-walking/logs/Medauroidea/airl/20250727-2010/model/actor.pth \
     --steps 200 --out /tmp/insect_airl
"""
import argparse, os, sys, time
import numpy as np
import torch

AIRL = "/home/aria/ioon-research/airl-insect-walking"
sys.path.insert(0, AIRL)

# camera framing = same as the insect add_camera.py (elevation 40 side telephoto)
DISTANCE, ELEVATION, AZIMUTH, VIEW_ANGLE, TARGET_Z, RUNWAY_AIM = 8.0, 40.0, 90.0, 15.0, 0.10, 0.75
RES = 256


def cam_offset():
    el, az = np.deg2rad(ELEVATION), np.deg2rad(AZIMUTH)
    h = DISTANCE * np.cos(el)
    return np.array([h*np.cos(az), h*np.sin(az), DISTANCE*np.sin(el)])


def look_at(cam_pos, target):
    z = target-cam_pos; z/=np.linalg.norm(z)
    x = np.cross([0,0,1.0], z); x/=np.linalg.norm(x)
    y = np.cross(z, x)
    return [v for r in range(3) for v in (x[r], y[r], z[r], cam_pos[r])]


def add_camera(sim):
    try: sim.removeObjects([sim.getObject("/vjepa_cam")])
    except Exception: pass
    head = np.array(sim.getObjectPosition(sim.getObject("/head"), sim.handle_world))
    target = np.array([head[0]+RUNWAY_AIM, head[1], TARGET_Z])
    cam_pos = target + cam_offset()
    h = sim.createVisionSensor(1|2|4, [RES,RES,0,0],
                               [0.01,20.0,np.deg2rad(VIEW_ANGLE),0.05,0,0,0,0,0,0,0])
    sim.setObjectAlias(h, "vjepa_cam")
    sim.setObjectMatrix(h, look_at(cam_pos, target))
    sim.setObjectInt32Param(h, sim.objintparam_visibility_layer, 0xFFFF)
    return h


def capture(sim, cam):
    sim.handleVisionSensor(cam)
    buf, res = sim.getVisionSensorImg(cam)
    return np.flipud(np.frombuffer(buf, np.uint8).reshape(res[1], res[0], 3)).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--actor", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--mid", default="", help="npy of correct action mid (overrides drifted env bounds)")
    ap.add_argument("--scale", default="", help="npy of correct action scale")
    ap.add_argument("--out", default="/tmp/insect_airl")
    args = ap.parse_args()

    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    c = RemoteAPIClient("localhost", port=23000); sim = c.require("sim")
    sim.stopSimulation()
    while sim.getSimulationState() != 0: sim.stopSimulation(); time.sleep(0.1)
    sim.loadScene(os.path.abspath(args.scene))
    # remove main_script (child script) so it can't drive the CSV gait against the policy
    for h in sim.getObjectsInTree(sim.handle_scene, sim.handle_all):
        try:
            if sim.getObjectAlias(h) == "script": sim.removeObjects([h]); print("removed main_script")
        except Exception: pass
    cam = add_camera(sim)
    print("camera added")

    from common.normalized_env_66k import CoppeliaSimEnv
    from networks.actor import ActorNetworkPolicy
    env = CoppeliaSimEnv(port=23000)
    obs_dim = env.observation_space.shape[0]; act_dim = int(len(env.action_space_low))
    print(f"env obs_dim={obs_dim} action_dim={act_dim}")
    sd = torch.load(args.actor, map_location="cpu", weights_only=True)
    exp_obs = sd["net.0.weight"].shape[1]; exp_act = sd["net.4.weight"].shape[0]
    print(f"actor expects obs={exp_obs} action={exp_act}")
    assert obs_dim == exp_obs, f"OBS MISMATCH env {obs_dim} vs actor {exp_obs}"

    actor = ActorNetworkPolicy(state_shape=(obs_dim,), action_shape=(act_dim,),
                               hidden_units=(64,64), hidden_activation=torch.nn.Tanh())
    actor.load_state_dict(sd); actor.eval()

    # override the env's drifted action denormalization with the log's true mid/scale
    if args.mid and args.scale:
        env._action_mid = np.load(args.mid).astype(float)
        env._action_scale = np.load(args.scale).astype(float)
        print("overrode action mid/scale from log; mid[0]=%.3f" % env._action_mid[0])

    cam = env.sim.getObject("/vjepa_cam")
    st = env.reset()
    if isinstance(st, tuple): st = st[0]
    frames, heads = [], []
    for t in range(args.steps):
        with torch.no_grad():
            a = actor(torch.tensor(np.asarray(st), dtype=torch.float32).unsqueeze(0)).numpy()[0]
        frames.append(capture(env.sim, cam))
        heads.append(env.sim.getObjectPosition(env.sim.getObject("/head"), env.sim.handle_world))
        out = env.step(a)
        st = out[0]; term = out[2]; trunc = out[3]
        if term or trunc:
            print(f"episode ended at step {t} (term={term} trunc={trunc})"); break
    env.stop()

    frames = np.asarray(frames, np.uint8); heads = np.asarray(heads, np.float32)
    dist = float(np.linalg.norm(heads[-1,:2]-heads[0,:2])) if len(heads) else 0.0
    print(f"frames={frames.shape} moved={dist:.3f}m final_z={heads[-1,2]:.3f}")

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out,"clip.npz"), frames=frames, head=heads)
    import imageio.v2 as imageio
    imageio.mimwrite(os.path.join(args.out,"walk.mp4"), list(frames), fps=30, macro_block_size=None)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    idx = np.linspace(0, len(frames)-1, 5).astype(int)
    fig,ax=plt.subplots(1,5,figsize=(20,4))
    for i,t in enumerate(idx): ax[i].imshow(frames[t]); ax[i].set_title(f"step {t}"); ax[i].axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(args.out,"frames.png"),dpi=90)
    print("saved video + frames.png ->", args.out)


if __name__ == "__main__":
    main()
