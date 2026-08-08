"""Copy the insect scene's camera onto the B1 scene so both embodiments are rendered alike.

Stage 2 asks whether embodiment is decodable from the latent. If the two embodiments are
rendered by different cameras, any such measurement is partly reading the camera. The scenes
already share a floor (5 x 5 m at z = -0.1) and a camera orientation, but not the field of
view: the insect scene uses a 0.2618 rad perspective angle and b1_flat.ttt uses 0.4189 rad, 60
percent wider. The wider view reaches past the floor edge, so B1 frames contain a horizon band
the insect frames do not, and the floor texture appears at a different scale. Measured on
median background images, insect bodies differ from each other by 0.16 to 0.34 grey levels out
of 255 while insect and B1 differ by 5.03 with 12.8 percent of pixels off by more than 10.

This copies position, orientation, field of view, clipping planes and resolution from the
insect camera to the B1 camera. Both collectors then re-anchor the camera to their own robot's
start pose, so the remaining geometry is identical by construction.

Run from the repository root with a CoppeliaSim listening:
  .venv/bin/python3 sim/match_b1_camera.py --port 23077
"""
import argparse
import os
import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = os.path.join(ROOT, "sim", "env")
SENSOR = "/vjepa_cam"
# each collector pins the camera to its own robot's start pose, so what has to match between
# scenes is the camera's offset from the tracked object, not its absolute position
TRACKED = {"medauroidea_stick_insect.ttt": "/head", "b1_flat.ttt": "/base_visual"}

# sim.visionfloatparam_* and sim.visionintparam_*, addressed numerically so the script does not
# depend on which constants a given CoppeliaSim build exposes through the remote API
FLOAT_PARAMS = {"nearClipping": 1000, "farClipping": 1001, "perspectiveAngle": 1004}
INT_PARAMS = {"resolutionX": 1002, "resolutionY": 1003}


def read_camera(sim, scene, track):
    sim.loadScene(os.path.join(ENV, scene))
    time.sleep(0.4)
    cam = sim.getObject(SENSOR)
    anchor = sim.getObjectPosition(sim.getObject(track), sim.handle_world)
    position = sim.getObjectPosition(cam, sim.handle_world)
    return {
        "offset": [position[0] - anchor[0], position[1] - anchor[1], position[2]],
        "anchor": anchor,
        "position": position,
        "orientation": sim.getObjectOrientation(cam, sim.handle_world),
        "floats": {k: sim.getObjectFloatParam(cam, v) for k, v in FLOAT_PARAMS.items()},
        "ints": {k: sim.getObjectInt32Param(cam, v) for k, v in INT_PARAMS.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--source", default="medauroidea_stick_insect.ttt")
    ap.add_argument("--target", default="b1_flat.ttt")
    ap.add_argument("--source_track", default="", help="object the source camera is pinned to")
    ap.add_argument("--target_track", default="", help="object the target camera is pinned to")
    ap.add_argument("--out", default="", help="defaults to overwriting the target scene")
    args = ap.parse_args()

    sim = RemoteAPIClient(port=args.port).require("sim")
    source_track = args.source_track or TRACKED[args.source]
    target_track = args.target_track or TRACKED[args.target]
    reference = read_camera(sim, args.source, source_track)
    before = read_camera(sim, args.target, target_track)

    cam = sim.getObject(SENSOR)
    # same offset from this scene's own tracked object, so both collectors end up with the
    # camera at the same place relative to a robot spawned at the same world point
    sim.setObjectPosition(cam, sim.handle_world,
                          [before["anchor"][0] + reference["offset"][0],
                           before["anchor"][1] + reference["offset"][1],
                           reference["offset"][2]])
    sim.setObjectOrientation(cam, sim.handle_world, reference["orientation"])
    for name, key in FLOAT_PARAMS.items():
        sim.setObjectFloatParam(cam, key, reference["floats"][name])
    for name, key in INT_PARAMS.items():
        sim.setObjectInt32Param(cam, key, reference["ints"][name])

    after = read_camera_current(sim, cam, before["anchor"])
    print(f"{'property':<18}{'insect':>28}{'B1 before':>28}{'B1 after':>28}")
    print(f"{'offset from track':<18}{fmt(reference['offset']):>28}{fmt(before['offset']):>28}{fmt(after['offset']):>28}")
    print(f"{'orientation':<18}{fmt(reference['orientation']):>28}{fmt(before['orientation']):>28}{fmt(after['orientation']):>28}")
    for name in FLOAT_PARAMS:
        print(f"{name:<18}{reference['floats'][name]:>28.4f}{before['floats'][name]:>28.4f}{after['floats'][name]:>28.4f}")
    for name in INT_PARAMS:
        print(f"{name:<18}{reference['ints'][name]:>28d}{before['ints'][name]:>28d}{after['ints'][name]:>28d}")

    out = args.out or os.path.join(ENV, args.target)
    sim.saveScene(out if os.path.isabs(out) else os.path.join(ENV, out))
    print(f"\nsaved: {out}")
    print("re-collect B1 after this; existing clips were rendered with the old camera")


def read_camera_current(sim, cam, anchor):
    position = sim.getObjectPosition(cam, sim.handle_world)
    return {
        "offset": [position[0] - anchor[0], position[1] - anchor[1], position[2]],
        "position": position,
        "orientation": sim.getObjectOrientation(cam, sim.handle_world),
        "floats": {k: sim.getObjectFloatParam(cam, v) for k, v in FLOAT_PARAMS.items()},
        "ints": {k: sim.getObjectInt32Param(cam, v) for k, v in INT_PARAMS.items()},
    }


def fmt(values):
    return "[" + ", ".join(f"{v:.3f}" for v in values) + "]"


if __name__ == "__main__":
    main()
