"""Build a legs-only, camera-equipped gecko scene from the raw `RL_slalom_sim2real_gamma_cfoot`
asset, for the held-out-body reconnaissance scoped after inspecting `sim/assets/gecko/`.

The raw asset has 27 revolute joints: 16 leg joints (4 per leg x 4 legs: `lf`/`rf`/`lh`/`rh`),
3 spine joints (`joint_b1/b2/b3`), and 8 small-range foot-compliance joints (`Leg{1-4}_ball{1,2}`).
Neither existing body (18-DOF hexapod, 12-DOF B1) has body-frame articulation, so the gecko is
made comparable by treating only the 16 leg joints as actively driven; the spine is left free
(soft spring to neutral, not commanded) and the foot joints are locked. No embedded controller
survives -- `/geckobotiv`'s script only `includeRel()`s a path that does not exist in this repo.

Usage (CoppeliaSim must already be running -- on this machine that means the DISPLAY=:0 launch in
the coppeliasim-headless memory, not `-h`, which exits immediately here):

  .venv/bin/python3 sim/scene/build_gecko_scene.py
  .venv/bin/python3 sim/scene/build_gecko_scene.py --preview
"""
import argparse
import os
import sys

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ego_camera import (  # noqa: E402  reuse the measured-not-assumed mounting geometry
    attach_ego, build_texture_box, clear_box, randomise_ground, room_for, scale_floor,
)

SOURCE = os.path.join(ROOT, "sim", "assets", "gecko", "RL_slalom_sim2real_gamma_cfoot_cover.ttt")
OUT = os.path.join(ROOT, "sim", "env", "gecko_legs.ttt")
SENSOR_NAME = "vjepa_cam"
RESOLUTION = 256

# The 8 foot-compliance ("ball") joints, locked with a stiff dynamic PID (see `build`).
FOOT_LOCKED = [
    "/Leg1_ball1", "/Leg1_ball2", "/Leg2_ball1", "/Leg2_ball2",
    "/Leg3_ball1", "/Leg3_ball2", "/Leg4_ball1", "/Leg4_ball2",
]
# The 3 spine joints -- soft spring to neutral, not commanded (see `build`).
SPINE_FREE = ["/joint_b1", "/joint_b2", "/joint_b3"]

# The 16 primary leg joints left actuated, in a fixed order this project can rely on downstream:
# leg-major, joint1..4 per leg. LF/RF/LH/RH names come straight from the asset's own convention.
LEG_ORDER = ["lf", "rf", "lh", "rh"]
ACTIVE_JOINTS = [f"/joint{j}_{leg}" for leg in LEG_ORDER for j in (1, 2, 3, 4)]


def gecko_forward(sim):
    """`body_part_1 - body_part_4`, in world coordinates. Measured, not assumed -- `body_part_1`
    sits ahead of `body_part_4` on world-x, consistent with the asset's `lf`/`rf` (front) vs
    `lh`/`rh` (hind) naming. Same convention as `ego_camera.py`'s `insect_forward`."""
    h1 = np.asarray(sim.getObjectPosition(sim.getObject("/body_part_1"), sim.handle_world), float)
    h4_candidates = [h for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type)
                      if sim.getObjectAlias(h, 0) == "body_part_4"]
    h4 = np.asarray(sim.getObjectPosition(h4_candidates[0], sim.handle_world), float)
    d = h1 - h4
    d[2] = 0.0
    return d


def add_camera(sim, preview=False):
    try:
        old = sim.getObject("/" + SENSOR_NAME)
        sim.removeObjects([old])
        print("  removed existing camera")
    except Exception:
        pass

    options = 1 | 2 | 4
    int_params = [RESOLUTION, RESOLUTION, 0, 0]
    float_params = [0.01, 20.0, np.deg2rad(60.0), 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cam = sim.createVisionSensor(options, int_params, float_params)
    sim.setObjectAlias(cam, SENSOR_NAME)
    sim.setObjectInt32Param(cam, sim.objintparam_visibility_layer, 0xFFFF)

    # ego_camera.py's room helpers look for an object aliased "Floor" (capital F); the gecko
    # asset's own floor is lowercase, so rename it to match rather than editing shared code.
    floor = sim.getObject("/floor")
    sim.setObjectAlias(floor, "Floor")

    head = sim.getObject("/body_part_1")
    forward = gecko_forward(sim)
    info = attach_ego(sim, cam, head, forward, offset_frac=(0.0, 0.551, 0.662))
    print(f"  camera mounted on /body_part_1, forward={np.round(forward, 3)}")
    print(f"  {info}")

    # Same textured room the other two bodies get (ego_camera.py), scaled by this body's own mount
    # height so it subtends the same angles from the gecko's eye as it does from the insect's/B1's.
    mount_z = info["position"][2]
    room = room_for(mount_z)
    clear_box(sim)
    build_texture_box(sim, size=room["size"], height=room["height"])
    scale_floor(sim, room["size"])
    randomise_ground(sim, seed=0, uv=room["ground_uv"])
    print(f"  room: mount_z={mount_z:.3f}  size={room['size']:.2f}m  height={room['height']:.2f}m")
    return cam


# Real per-body masses (kg), read from the upstream USD asset this project's copy was exported
# from -- github.com/worasuch/IsaacLab-LocoNets, `slalom_fixedbody_16dof_origin_addmass.usd`, via
# UsdPhysics.MassAPI. That USD's fused torso is `/geckobotiv` here, not `/body_part_1` (see
# `apply_masses`). Its `motor2/motor3/motor4/link4/foot_{leg}` prims match ours by name. Inertia
# was left unset in the USD too, so `sim.setShapeMass` (derive from geometry) is used, not
# `setShapeMassAndInertia`.
BASE_MASS = 0.70
LEG_LINK_MASS = {"motor2": 0.10, "motor3": 0.10, "motor4": 0.10, "link4": 0.04, "foot": 0.08}


def apply_masses(sim):
    """Every shape in this asset ships `static=1, respondable=0` -- inert, no gravity, no ground
    contact. This makes the real dynamic chain (base + per-leg links) dynamic and respondable with
    the mass recovered above; cosmetic cover-mesh duplicates, walls and floor are left as-is.

    The real torso every leg and the spine joint attach to is `/geckobotiv` (checked via
    `getObjectParent` on every joint), not `/body_part_1` or `/body_part_2/3/4` -- those are
    plain, joint-less decorative siblings of the real chain (`/geckobotiv`,
    `/link_body2/3/4`) and are left exactly as shipped (static). `link_body2/3/4` already ship
    dynamic with a real 0.14 kg mass each, untouched here."""
    root = sim.getObject("/geckobotiv")
    sim.setObjectInt32Param(root, sim.shapeintparam_static, 0)
    sim.setObjectInt32Param(root, sim.shapeintparam_respondable, 1)
    sim.setShapeMass(root, BASE_MASS)
    print(f"  /geckobotiv  mass={BASE_MASS} kg  dynamic, respondable (the real torso)")
    print("  /body_part_1..4 left static (decorative, not chain members)")

    for leg in LEG_ORDER:
        leg_total = 0.0
        for part, mass in LEG_LINK_MASS.items():
            try:
                h = sim.getObject(f"/{part}_{leg}")
            except Exception:
                # `rh`'s cover mesh is missing a `link4_rh` shape by that name (an authoring gap
                # in the source asset). Its 0.04 kg is small next to the leg's 0.42 kg total.
                print(f"    (no /{part}_{leg} shape in this asset -- skipped, {mass} kg not applied)")
                continue
            sim.setObjectInt32Param(h, sim.shapeintparam_static, 0)
            sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1)
            sim.setShapeMass(h, mass)
            leg_total += mass
        print(f"  {leg}: dynamic+respondable, {leg_total:.2f} kg")

    print(f"  nominal total (if nothing were skipped): {BASE_MASS + 4 * sum(LEG_LINK_MASS.values()):.2f} kg")


# Every consecutive pair of shapes in a leg's kinematic chain geometrically overlaps at rest (the
# joint housings are modelled generously), which is harmless while static but a self-collision
# explosion once `apply_masses` makes them dynamic. Each chain position gets its own collision bit
# so no two robot shapes ever physically respond to each other; cross-leg pairs at the same chain
# position (e.g. motor2_lf and motor2_rf) never geometrically overlap, so sharing a bit is safe.
# `/geckobotiv` and `/link_body2/3/4` get bits for the same reason once they became dynamic; floor
# and walls keep their default full mask so ground contact is untouched.
CHAIN_BIT = {"motor2": 1 << 0, "motor3": 1 << 1, "motor4": 1 << 2, "link4": 1 << 3, "foot": 1 << 4}
SPINE_BIT = {"geckobotiv": 1 << 5, "link_body2": 1 << 6, "link_body3": 1 << 7, "link_body4": 1 << 8}


def disable_self_collision(sim):
    for name, bit in SPINE_BIT.items():
        h = sim.getObject(f"/{name}")
        sim.setObjectInt32Param(h, sim.shapeintparam_respondable_mask, bit)
    for leg in LEG_ORDER:
        for part, bit in CHAIN_BIT.items():
            try:
                h = sim.getObject(f"/{part}_{leg}")
            except Exception:
                continue  # /link4_rh -- same missing shape apply_masses already skips
            sim.setObjectInt32Param(h, sim.shapeintparam_respondable_mask, bit)
    print("  self-collision disabled: each chain position on its own collision bit; "
          "floor/walls untouched (still 0xFFFF, full ground contact)")


def disable_dangling_controller_script(sim):
    """The `/geckobotiv` simulation script only `includeRel()`s a path that does not exist in this
    repo, which auto-pauses the simulation a few steps into any run. Disabling it removes the
    error at the source; nothing here replaces its behaviour, since no working controller shipped
    with this asset."""
    obj = sim.getObject("/geckobotiv")
    script = sim.getScript(sim.scripttype_simulation, obj)
    sim.setScriptAttribute(script, sim.scriptattribute_enabled, False)
    print("  disabled /geckobotiv's dangling controller-include script (was auto-pausing sim)")


def set_finer_timestep(sim):
    """The scene ships with a 0.1625 s timestep (~6 Hz), unusually coarse next to the hexapod's
    0.05 s or B1's 0.02 s. At that timestep, under CPG motion, the robot tunnelled clean through
    the floor within one second of simulated time. Matching B1's 0.02 s keeps floor contact
    continuous."""
    sim.setFloatParam(sim.floatparam_simulation_time_step, 0.02)
    print("  simulation timestep: 0.1625s -> 0.02s (matches B1; scene's own value tunnelled "
          "through the floor under CPG motion)")


def set_bullet_engine(sim):
    """The scene ships on Vortex, unregistered per the CoppeliaSim startup log: under Vortex,
    `setJointTargetPosition` accepts the call but the joint never actually moves. Bullet is also
    what the hexapod scene uses, so this keeps both bodies on the same engine."""
    sim.setInt32Param(sim.intparam_dynamic_engine, sim.physics_bullet)
    print("  physics engine: Bullet (scene shipped on unregistered Vortex, which silently ignored "
          "joint position targets)")


# Same allocentric camera convention as `sim/scene/add_camera.py` (used for the hexapod): AZIMUTH=90
# (pure side view) and the same DISTANCE/VIEW_ANGLE combination, scaled by a body-size factor `k`.
#
# **`k` from hip/mount height was wrong -- tried first, visibly too close, corrected here.** The
# gecko's sprawled 4-legged stance means its actual horizontal footprint (measured directly:
# ~0.33 x 0.31 m bounding box across all leg links) is close to the hexapod's *absolute* size, not
# proportionally smaller the way its hip height alone suggests (mount height ratio gives k=0.30,
# which under-frames it). The empirically-correct distance for this footprint was found by direct
# visual check to be 5.0 m (against the hexapod's 8.0 m) -- k=5.0/8.0=0.625 is that ratio, applied
# consistently to DISTANCE, TARGET_Z and RUNWAY_AIM together, not derived from mount height at all.
#
# Aim direction also needs a gecko-specific correction the hexapod doesn't: the gecko's confirmed
# walking direction is world -x (`gecko_forward()`'s "+x = forward" was our own labelling choice
# from the asset's own `lf`/`rf` vs `lh`/`rh` naming, not a guarantee the working gait walks that
# way -- it measurably does not), so the runway is aimed at `head[0] - RUNWAY_AIM`, not `+`.
# AZIMUTH=-90, not the hexapod's +90: worked out directly (camera's local "right" = cross(up, view
# direction)) that +90 puts world -x (the gecko's confirmed walking direction) toward camera-left,
# i.e. the robot would walk right-to-left; -90 puts it toward camera-right, i.e. left-to-right,
# matching the hexapod's own convention of the robot crossing the frame left to right.
ALLO_DISTANCE, ALLO_ELEVATION, ALLO_AZIMUTH, ALLO_VIEW_ANGLE = 8.0, 40.0, -90.0, 15.0
ALLO_TARGET_Z, ALLO_RUNWAY_AIM = 0.10, 0.75
ALLO_SCALE_K = 5.0 / 8.0  # footprint-based, not mount-height-based -- see comment above


def add_allocentric_camera(sim):
    k = ALLO_SCALE_K

    root = sim.getObject("/geckobotiv")
    head = np.array(sim.getObjectPosition(root, sim.handle_world))
    target = np.array([head[0] - ALLO_RUNWAY_AIM * k, head[1], ALLO_TARGET_Z * k])

    el, az = np.deg2rad(ALLO_ELEVATION), np.deg2rad(ALLO_AZIMUTH)
    horiz = ALLO_DISTANCE * k * np.cos(el)
    offset = np.array([horiz * np.cos(az), horiz * np.sin(az), ALLO_DISTANCE * k * np.sin(el)])
    cam_pos = target + offset

    z = target - cam_pos
    z = z / np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = []
    for r in range(3):
        m += [x[r], y[r], z[r], cam_pos[r]]

    try:
        old = sim.getObject("/vjepa_cam_allo")
        sim.removeObjects([old])
    except Exception:
        pass
    options = 1 | 2 | 4
    int_params = [512, 512, 0, 0]
    float_params = [0.01, 20.0, np.deg2rad(ALLO_VIEW_ANGLE), 0.05, 0, 0, 0, 0, 0, 0, 0]
    cam = sim.createVisionSensor(options, int_params, float_params)
    sim.setObjectAlias(cam, "vjepa_cam_allo")
    sim.setObjectInt32Param(cam, sim.objintparam_visibility_layer, 0xFFFF)
    sim.setObjectMatrix(cam, sim.handle_world, m)
    print(f"  /vjepa_cam_allo  k={k:.3f}  pos={np.round(cam_pos,3)}  target={np.round(target,3)}")
    return cam


def set_default_viewport(sim):
    """Point `/DefaultCamera` (the GUI's own interactive viewport, saved as part of the scene) at
    the robot, at a fixed third-person distance, so opening or re-running this scene always shows
    the same framing -- it shipped 3.9/-4.1/4.0 m away, sane for the original life-size slalom
    course, useless for a ~0.25 m robot. Unlike a vision sensor (+z forward), CoppeliaSim's
    interactive `camera` object type looks down its own -z (standard OpenGL convention)."""
    cam = sim.getObject("/DefaultCamera")
    root = sim.getObject("/geckobotiv")
    target = np.array(sim.getObjectPosition(root, sim.handle_world)) + np.array([0, 0, 0.03])
    cam_pos = target + np.array([-0.35, -0.45, 0.30])

    z = target - cam_pos
    z = z / np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    # camera looks down -z, so the matrix's z-column is -z (points away from the target)
    m = []
    for r in range(3):
        m += [x[r], y[r], -z[r], cam_pos[r]]
    sim.setObjectMatrix(cam, sim.handle_world, m)
    print(f"  /DefaultCamera fixed on the robot: pos={np.round(cam_pos,3)} target={np.round(target,3)}")


def build(sim, preview=False):
    sim.loadScene(SOURCE)
    disable_dangling_controller_script(sim)
    set_bullet_engine(sim)
    set_finer_timestep(sim)
    apply_masses(sim)
    disable_self_collision(sim)

    # `jointmode_kinematic` breaks Bullet's rigid-body chain: everything downstream of a kinematic
    # joint separates from the rest of the body under dynamics. Foot joints stay in `jointmode_force`
    # (same mode as the active joints) and are held rigid with a stiff PID instead. Gains are
    # Bullet-engine-specific (`bullet_joint_pospid1/3`), not the generic `jointfloatparam_pid_p`,
    # which silently does nothing under Bullet.
    print(f"locking {len(FOOT_LOCKED)} foot-compliance joints via a stiff dynamic PID, "
          f"not kinematic mode:")
    for name in FOOT_LOCKED:
        h = sim.getObject(name)
        pos = sim.getJointPosition(h)
        sim.setJointMode(h, sim.jointmode_force, 0)
        sim.setEngineFloatParam(sim.bullet_joint_pospid1, h, 5.0)   # P; active joints ship at 0.1
        sim.setEngineFloatParam(sim.bullet_joint_pospid3, h, 0.5)   # D; active joints ship at 0.0
        sim.setJointMaxForce(h, 50.0)                                 # well above what a 2.4 kg robot needs
        sim.setJointTargetPosition(h, pos)
        print(f"  {name:<14} locked at {pos:+.4f} rad (bullet pospid1=5.0, pospid3=0.5, maxForce=50 N*m)")

    # Spine: soft spring to neutral (0 rad), not a rigid lock and not fully free. A real lizard
    # spine undulates actively with the leg cycle, which this project does not model; a rigid lock
    # forces the body into one plank, which no sprawled quadruped is. Fully free (jointdynctrl_free,
    # zero restoring force) was tried and rejected: with no notion of a neutral pose, the joint only
    # integrates whatever net torque it receives and drifts to one side and stays there under the
    # gait's roughly-constant torque, rather than flexing and returning.
    #
    # Gain was swept (0.3 / 0.5 / 1.0 / 0.1) and measured, not guessed: net displacement over a 3 s
    # CPG test was 0.030 m / 0.021 m / 0.021 m / TBD, against the fully-free spine's 0.037 m and the
    # rigid-locked spine's 0.022 m. Centring improves with gain but displacement drops sharply past
    # ~0.3-0.5 -- there is no smooth middle, it behaves like a cliff between "compliant enough to
    # help" and "stiff enough to act locked." Currently set to prioritise displacement.
    print(f"springing {len(SPINE_FREE)} spine joints toward neutral (soft PID, not locked, not free):")
    SPINE_PID_P = 0.1
    SPINE_PID_D = SPINE_PID_P / 6
    SPINE_MAX_FORCE = SPINE_PID_P * 10
    for name in SPINE_FREE:
        h = sim.getObject(name)
        sim.setJointMode(h, sim.jointmode_force, 0)
        sim.setObjectInt32Param(h, sim.jointintparam_dynctrlmode, sim.jointdynctrl_position)
        sim.setEngineFloatParam(sim.bullet_joint_pospid1, h, SPINE_PID_P)
        sim.setEngineFloatParam(sim.bullet_joint_pospid3, h, SPINE_PID_D)
        sim.setJointMaxForce(h, SPINE_MAX_FORCE)
        sim.setJointTargetPosition(h, 0.0)
        print(f"  {name:<14} soft spring to 0 rad (pospid1={SPINE_PID_P}, pospid3={SPINE_PID_D:.3f}, "
              f"maxForce={SPINE_MAX_FORCE} N*m)")

    # These joints shipped rate-limited (maxvel=1.047 rad/s, maxforce=4.1 N*m) -- too weak to track
    # a 1 Hz, 0.35 rad sinusoidal CPG command (needs ~1.4 rad/s at the steepest point). Measured
    # directly: a 0.9 rad step command covered barely half the distance in 0.4 s. The achieved leg
    # motion was a small, lagging fraction of what was commanded, regardless of phase or amplitude
    # tuning, until this was fixed.
    print(f"\n{len(ACTIVE_JOINTS)} active leg joints, leg-major order:")
    active_handles = []
    for name in ACTIVE_JOINTS:
        h = sim.getObject(name)
        active_handles.append(h)
        sim.setObjectFloatParam(h, sim.jointfloatparam_maxvel, 6.0)   # was 1.047 rad/s
        sim.setJointMaxForce(h, 20.0)                                  # was 4.1 N*m
        cyclic, (lo, rng) = sim.getJointInterval(h)
        print(f"  {name:<14} interval=[{lo:+.3f}, {lo + rng:+.3f}]  maxvel=6.0 rad/s maxForce=20 N*m")

    print()
    add_camera(sim, preview=preview)
    add_allocentric_camera(sim)
    set_default_viewport(sim)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sim.saveScene(OUT)
    print(f"\nsaved: {OUT}")
    return active_handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    sim = RemoteAPIClient("localhost", port=23000).require("sim")
    build(sim, preview=args.preview)


if __name__ == "__main__":
    main()
