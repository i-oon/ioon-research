"""List what is inside a CoppeliaSim scene: joints, their order, and any attached scripts.

Written for `olaf_6legs_model.ttt`, a six-leg model from elsewhere in the lab. Two questions have
to be answered before any of this project's code can be pointed at it:

    is there a controller in here?   the scene file itself is opaque -- no readable strings, no
                                    inflatable streams -- so the only way to see an embedded script
                                    is to have CoppeliaSim open the scene and hand it over.
    does the joint layout match?     `wm/predict_actions.py` assumes 18 joints in leg-major order,
                                    `LEGS x SEG`. A different name or order silently mislabels every
                                    column of every action array downstream.

Run CoppeliaSim first, then this against it. It only reads, and it never starts the simulation.

  cd /home/aria/CoppeliaSim && ./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23000
  .venv/bin/python3 scripts/tools/inspect_scene.py --scene olaf_6legs_model.ttt
"""
import argparse
import os

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--scene", default="olaf_6legs_model.ttt")
    ap.add_argument("--load", action="store_true",
                    help="load the scene first; omit if you already opened it in the GUI")
    args = ap.parse_args()

    sim = RemoteAPIClient(port=args.port).require("sim")
    if args.load:
        path = args.scene if os.path.isabs(args.scene) else os.path.join(ROOT, args.scene)
        sim.loadScene(path)
        print(f"loaded {path}\n")

    objects = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)

    # What kind of thing is each object? Printing the census first because the previous version of
    # this script asked every object "do you have a script" and got the *scene's main script* back
    # each time, then reported it 76 times as if 76 controllers existed. Counting types is how you
    # see that immediately.
    census, joints = {}, []
    type_names = {getattr(sim, a): a.replace("object_", "").replace("_type", "")
                  for a in dir(sim) if a.startswith("object_") and a.endswith("_type")}
    type_names.update({getattr(sim, a): a.replace("sceneobject_", "")
                       for a in dir(sim) if a.startswith("sceneobject_")})
    for h in objects:
        kind = sim.getObjectType(h)
        census[type_names.get(kind, f"type {kind}")] = census.get(
            type_names.get(kind, f"type {kind}"), 0) + 1
        if kind == sim.object_joint_type:
            joints.append((h, sim.getObjectAlias(h, 1)))

    print(f"{len(objects)} objects, {len(joints)} joints")
    print("object census: " + ", ".join(f"{k} x{v}" for k, v in sorted(census.items())) + "\n")

    # In 4.6+ scripts are scene objects in their own right, so they appear in the tree above.
    # In older versions they hang off an object and have to be asked for by (type, object).
    scripts = []
    script_type = getattr(sim, "sceneobject_script", getattr(sim, "object_script_type", None))
    if script_type is not None:
        for h in objects:
            if sim.getObjectType(h) == script_type:
                scripts.append((sim.getObjectAlias(h, 1), "script object", h))
    if not scripts:
        for h in objects:
            for attr in ("scripttype_simulation", "scripttype_childscript",
                         "scripttype_customization", "scripttype_customizationscript"):
                kind = getattr(sim, attr, None)
                fn = getattr(sim, "getScript", None)
                if kind is None or fn is None:
                    continue
                try:
                    s = fn(kind, h)
                except Exception:
                    continue
                if s and s != -1:
                    scripts.append((sim.getObjectAlias(h, 1), attr, s))
    print(f"{len(scripts)} attached script(s) -- the scene main script is excluded\n")

    print("JOINTS, in tree order -- compare against LEGS x SEG in wm/predict_actions.py")
    for i, (h, name) in enumerate(joints):
        pos = sim.getJointPosition(h)
        print(f"  {i:>2}  {name:<44} {pos:+.4f} rad")

    if not scripts:
        print("\nNo attached scripts. The controller, if there is one, lives outside this file.")
        return

    print("\nSCRIPTS")
    for name, kind, handle in scripts:
        text = ""
        # every spelling this API has used for "give me the source of that script"
        attempts = [
            ("getScriptText", (handle,)),
            ("getScriptStringParam", (handle, 6001)),
            ("getScriptStringParam", (handle, sim.scriptstringparam_text
                                     if hasattr(sim, "scriptstringparam_text") else 6001)),
            ("getScriptProperty", (handle,)),
        ]
        for getter, argv in attempts:
            fn = getattr(sim, getter, None)
            if fn is None:
                continue
            try:
                out = fn(*argv)
            except Exception:
                continue
            if isinstance(out, str) and len(out) > 20:
                text = out
                break
        print(f"\n=== {name}  [{kind}]  handle {handle}  {len(text)} chars " + "=" * 24)
        if text:
            print(text)
        else:
            names = sorted(a for a in dir(sim) if "cript" in a)
            print("  could not read it through this API version. Script-related calls available:")
            print("   ", ", ".join(names))


if __name__ == "__main__":
    main()
