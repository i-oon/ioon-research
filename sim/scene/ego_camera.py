"""A head camera and a cheap textured room, for the egocentric de-risk gate.

**Deliberately the minimum that can answer the question.** Egocentric needs a world to see, so
something must be built -- but a polished environment is only worth building *after* the gate
passes, so this makes four coloured, mildly textured walls around the spawn and mounts the existing
`vjepa_cam` on the robot. Nothing here is meant to survive into a final environment.

**Two conventions that are not guessable and have cost this project time before.** A CoppeliaSim
vision sensor looks along its own **+z**. The insect's body frame has **z pointing aft** off
`/abdomen` (F71, F117), so its camera must be turned to face **-z_body**; the B1's base has **x
forward**, so its camera faces **+x_body**. Both defaults are below and both are **guesses until
somebody looks at a frame** -- `--ego_euler` exists so they can be corrected without editing code,
and the run sheet renders before it measures.

**Texture, not checkerboard and not blank.** `set_floor_texture.py` measured both extremes as
actively harmful: repeating high-contrast edges alias under sub-pixel motion (r = -0.16) and
featureless patches get repurposed as ViT scratch space (r = -0.20). The walls here follow the same
recipe -- matte, mildly textured, non-repeating at the scale the camera sees -- because **optical
flow is the entire signal the egocentric view is supposed to provide**, and a wall with no texture
provides none.
"""
import os
import sys
import tempfile

import numpy as np
from PIL import Image

RECIPE = "v3_oct4_8_16"          # bump when the texture changes, so old files cannot be picked up

# **The insect's camera mount, in metres, and the unit every other length is expressed in.**
# Measured live rather than tabulated: an earlier version scaled the room by median *walking* height
# (0.175 and 0.559, ratio 3.19) while the camera sat on the mount at spawn (0.272 and 0.599, ratio
# 2.20). Two height definitions for one scale left the B1's frame 46% floor against the insect's
# 32%, which is a difference of view sitting inside a cross-body test.
REF_MOUNT_Z = 0.272

# **Mean pitch of each body's own camera axis while walking, in degrees, measured over all 48 clips
# of `data/allocentric/beh12_*`.** The insect carries its head up: +7.53 (sd 0.64 across clips, and
# 7.15 / 7.29 / 8.15 across the three behaviour families), the B1 +1.02 (sd 0.66). The camera is
# parented to the mount, so it inherits that -- which put the insect's wall/floor junction 13.6
# degrees below centre where the geometry predicts 6.0, and left its frame 32% floor against the
# B1's 44%.
#
# **Only the constant is removed. The swing stays.** Yaw swings 15.6 degrees on the insect against
# 6.8 on the B1 and the head bobs 30 mm; those are the body moving and are what an egocentric view
# is supposed to contain. Levelling the *mean* is the same kind of correction as `--align_yaw`:
# it stops the two cameras aiming at different parts of the room, and stabilises nothing else.
WALK_PITCH = {"hexapod": 7.53, "b1": 1.02}


def room_for(mount_z, size=8.0, height=3.0, tile=6.0, ground_uv=0.5):
    """Every length scaled by body height, so the room **subtends the same angles from each eye**.

    **Without this the two robots are not doing the same experiment.** A B1 is 3.19x the insect's
    height and covers 1.73x the ground per clip; dropped into the insect's 8 m room it stands in a
    space that is proportionally a third the size, sees the walls at a different angle and the floor
    at a different rate, and Q2 would be asking whether a coordinate survives **a change of room**
    as much as a change of body.

    **The scale is the camera mount's own height above the floor, read from the scene**, so the
    room, the ground tile and the camera offset all derive from one number and cannot drift apart.

    **The two travel ratios cannot both be matched, and that is a fact about the robots.** Travel per
    body height is 3.39 on the insect and 1.85 on the B1, so no room size equalises both the wall
    sweep and the ground flow. **Height is the one chosen**, because the shared coordinate is already
    Froude-scaled -- normalised by `sqrt(h/g)` -- so scaling the world by height is the convention
    the measurement itself uses, and because task-space quantities in this project scale with body
    size while joint-space ones do not.

    The camera offset scales with it for the same reason: 15 cm above an insect's head is a
    different vantage from 15 cm above a B1's back.

    **`ground_uv` scales too, and it is the one that decides whether sideways is visible at all.**
    The floor's authored 6 m tile was chosen for a third-person shot; an insect 42 cm above it sees
    about a metre of floor, so a sixth of one tile fills the frame and the ground reads as flat. The
    floor is the nearest surface and therefore where parallax lives -- 0.5 m tiles put real structure
    under the camera at insect scale, and 1.6 m does the same for the B1.
    """
    k = float(mount_z) / REF_MOUNT_Z
    return dict(size=size * k, height=height * k, tile=0.0, ground_uv=ground_uv * k,
                offset_frac=(0.0, 0.551, 0.662),   # of the mount's height; identical on both bodies
                scale=k)


def _basis(forward, up=(0.0, 0.0, 1.0)):
    """A CoppeliaSim object matrix whose **+z is `forward`** -- the axis a vision sensor looks down."""
    z = np.asarray(forward, float); z /= max(np.linalg.norm(z), 1e-9)
    u = np.asarray(up, float)
    x = np.cross(u, z); n = np.linalg.norm(x)
    if n < 1e-6:                                   # forward is vertical; pick any perpendicular
        x = np.cross(np.array([1.0, 0.0, 0.0]), z); n = np.linalg.norm(x)
    x /= n
    y = np.cross(z, x)
    return x, y, z


def attach_ego(sim, cam, parent, forward_world, offset=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0),
               offset_frac=None, pitch_comp=0.0):
    """Mount the sensor on `parent`, looking along `forward_world`, then let it ride.

    **The pose is derived from geometry, never from a per-robot Euler convention.** The first
    version of this guessed: it assumed the insect's forward was `-z` of `/abdomen` and mounted the
    camera on the abdomen, which is the **rear** segment -- so the camera was neither at the head nor
    necessarily facing forward, and a wrong-facing camera still produces frames, still passes every
    downstream script, and would have answered Q1 with a false pass.

    `forward_world` is measured rather than assumed: for the insect it is `head - abdomen`, for the
    B1 the direction the base actually travels in a forward clip. **Both come out of the scene, so
    neither can be wrong about a convention.**

    `offset` is `(right, up, forward)` **in the camera's own basis**, in metres, so "10 cm ahead of
    the head and 3 cm above it" is written as `(0, 0.03, 0.10)` for either robot.
    """
    if pitch_comp:
        f = np.asarray(forward_world, float)
        f = f / max(np.linalg.norm(f), 1e-9)
        r = np.cross(np.asarray(up, float), f)
        r = r / max(np.linalg.norm(r), 1e-9)
        # **Sign checked against a render, not derived on paper.** With `up` = +z and the
        # basis below, a positive angle here rotates the forward axis downward; the first
        # attempt used the negative and moved the insect's floor edge from 13.6 to 21.3
        # degrees, which is the compensation applied backwards.
        a = np.radians(pitch_comp)
        forward_world = f * np.cos(a) + np.cross(r, f) * np.sin(a)
    x, y, z = _basis(forward_world, up)
    p = np.asarray(sim.getObjectPosition(parent, sim.handle_world), float)
    if offset_frac is not None:
        # **A fraction of the mount's own height above the floor, not a fixed distance.** With a
        # fixed 15 cm the insect's camera sat at 1.55x its head height and the B1's at 1.80x its
        # base, so the horizon fell in a different place and the B1's frame was 45% floor against
        # the insect's 34% -- a difference of view, not of body, sitting directly inside Q2.
        offset = tuple(f * float(p[2]) for f in offset_frac)
    p = p + x * offset[0] + y * offset[1] + z * offset[2]
    sim.setObjectMatrix(cam, sim.handle_world,
                        [float(x[0]), float(y[0]), float(z[0]), float(p[0]),
                         float(x[1]), float(y[1]), float(z[1]), float(p[1]),
                         float(x[2]), float(y[2]), float(z[2]), float(p[2])])
    sim.setObjectParent(cam, parent, True)         # keepInPlace, so the world pose above survives
    return dict(forward=[round(float(v), 3) for v in z],
                position=[round(float(v), 3) for v in p])


def insect_forward(sim):
    """`head - abdomen`, in world coordinates. **Geometry, not a stored convention.**"""
    h = np.asarray(sim.getObjectPosition(sim.getObject("/head"), sim.handle_world), float)
    a = np.asarray(sim.getObjectPosition(sim.getObject("/abdomen"), sim.handle_world), float)
    d = h - a
    d[2] = 0.0                                     # look level, not along the body's slope
    return d


def randomise_ground(sim, seed=0, uv=0.5, octaves=((4, .4), (8, .25), (16, .2), (32, .15))):
    """A fresh ground texture per episode -- **the surface the head camera actually spends its
    frames looking at.**

    Finer octaves than the walls carry on purpose: the floor is metres closer, so it is seen at far
    higher magnification, and what is fine detail on a wall 4 m away is coarse detail on ground 30
    cm below the camera. **Whether this octave mix is right is not settled** -- it was chosen against
    a 2D pan, which has no depth, no parallax and no shadows, and the real question is what a walking
    robot's rendered view carries. That is measured in the 3D scene, not tuned here.
    """
    path = os.path.join(tempfile.gettempdir(), f"ego_ground_{RECIPE}_{seed}.png")
    grey = _octave_noise(1024, seed * 31 + 7, octaves)
    rng = np.random.default_rng(seed * 31 + 7)
    tint = 0.55 + 0.25 * rng.random(3)
    Image.fromarray(_tinted(grey, tuple(tint), lift=0.35)).save(path)
    carrier, tid, _res = sim.createTexture(path, 0)
    n = 0
    for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type):
        try:
            if sim.getObjectAlias(h, 1).startswith("/Floor"):
                sim.setShapeTexture(h, tid, sim.texturemap_cube, 4 | 8, [uv, uv])
                n += 1
        except Exception:
            pass
    sim.removeObjects([carrier])
    print(f"    ground randomised, seed {seed}, applied to {n} shape(s)")
    return n


def scale_floor(sim, room_size, margin=1.3):
    """Grow the scene floor until it reaches past the walls.

    **A room larger than its floor is not a room.** Both scenes ship a 5 m floor; the B1's replay
    already scales it by 3 for its own framing, the insect's collector never did, so an 8 m box put
    the walls a metre and a half beyond the floor's edge and the view ended in void before it
    reached them.

    `sim.scaleObjects` grows a box **without moving its centre**, so the walking surface rises by
    half the added thickness and has to be put back or the robot stands underground -- the same
    correction `render_b1_replay` makes, kept in one place now.
    """
    floors = [h for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type)
              if sim.getObjectAlias(h, 1).startswith("/Floor")]
    if not floors:
        return 0.0
    top = sim.getObject("/Floor")
    def surface():
        q = sim.getObjectPosition(top, sim.handle_world)
        bb = sim.getShapeBB(top)
        return q[2] + (bb[0] if isinstance(bb[0], list) else bb)[2] / 2
    bb = sim.getShapeBB(top)
    width = float((bb[0] if isinstance(bb[0], list) else bb)[0])
    k = max(1.0, room_size * margin / max(width, 1e-6))
    if k <= 1.001:
        return 1.0
    before = surface()
    sim.scaleObjects(floors, float(k), False)
    drop = surface() - before
    for h in floors:
        q = sim.getObjectPosition(h, sim.handle_world)
        sim.setObjectPosition(h, sim.handle_world, [q[0], q[1], q[2] - drop])
    print(f"    floor x{k:.2f}: {width:.1f} m -> {width * k:.1f} m, surface held at {surface():+.3f}")
    return k


def clear_box(sim):
    """Remove any walls a previous run left behind, so variants do not accumulate."""
    gone = 0
    for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type):
        try:
            if sim.getObjectAlias(h, 0).startswith("egoWall"):
                sim.removeObjects([h])
                gone += 1
        except Exception:
            pass
    return gone


def _wall_texture(seed, n=512):
    """Value noise at octaves 4, 8 and 16 -- **measured, not inherited.**

    The first version of this file used `set_floor_texture.py`'s recipe on the assumption that a
    good floor makes a good wall. **They are not the same job.** The floor recipe exists to stop a
    third-person background drowning the robot; an egocentric wall *is* the signal, and the property
    it needs is that embedding distance grows with camera motion.

    Measured directly (`scripts/diagnostics/egocentric_view/texture_for_vjepa.py`, a texture panned past a 256 px
    viewport through the same encoder), correlation between true pan distance and embedding
    distance:

        checkerboard                -0.149     (reproduces the project's own -0.16)
        white noise                  0.199
        floor recipe                 0.478
        **octaves 4, 8, 16           0.723**   <- this
        octaves 8, 16, 32            0.613

    **The floor recipe's fine octaves at 64 and 256 are what cost it**, which is the same
    high-frequency aliasing the checkerboard finding was about, in a milder form.

    **None of them is good**, and that is worth carrying forward rather than burying: the best
    surface tested reaches r = 0.72 and moves the embedding only about twice as far as a
    one-pixel wobble does. Egocentric rests on this relation being strong, and measured, it is not.
    """
    return _octave_noise(n, seed, ((4, 0.5), (8, 0.3), (16, 0.2)))


def _octave_noise(n, seed, octaves):
    rng = np.random.default_rng(seed)
    acc = np.zeros((n, n))
    for octave, weight in octaves:
        small = rng.random((octave, octave))
        idx = np.linspace(0, octave - 1, n)
        i0 = np.floor(idx).astype(int)
        i1 = np.minimum(i0 + 1, octave - 1)
        f = idx - i0
        top = small[np.ix_(i0, i0)] * (1 - f) + small[np.ix_(i0, i1)] * f
        bot = small[np.ix_(i1, i0)] * (1 - f) + small[np.ix_(i1, i1)] * f
        acc += weight * (top * (1 - f[:, None]) + bot * f[:, None])
    return (acc - acc.min()) / max(acc.max() - acc.min(), 1e-9)


def _tinted(grey, colour, lift=0.45):
    """Grey noise carried into one wall's colour, kept **bright**.

    **The first version multiplied a 0.3-0.75 colour by a 0.55-1.0 noise and the walls came out
    near-black**, while the floor stayed bright because its shape colour is white and its texture is
    pale. Lifting toward white keeps the hue distinct enough to read bearing from without turning
    the room into a cave: the tint is a *label*, and the texture is the signal.
    """
    base = np.asarray(colour, float)[None, None, :]
    base = base + lift * (1.0 - base)                     # toward white, hue preserved
    img = base * (0.70 + 0.30 * grey[..., None])
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def build_texture_box(sim, size=8.0, height=3.0, thickness=0.05, seed=0, tint=True, tile=6.0,
                      ceiling=True, centre=(0.0, 0.0)):
    """Four walls around the origin, each with its own texture and colour.

    **Walls are structure, not landmarks.** They exist so a turn produces *some* image change at
    all; they must not be identifiable, or the view reintroduces the readability it was meant to
    remove. Appearance is drawn fresh from `seed` each episode.

    **The ground is the primary flow source, not the walls.** A head camera on a walking robot sees
    mostly floor, and prior egocentric locomotion work reads flow from ground texture rather than
    from wall landmarks (Hu et al. 2207.03386). `randomise_ground` is what should carry the signal
    here; the box is a backdrop that keeps the frame from ending in empty space.

    **Height and distance both matter and the first version had both wrong.** 1.2 m walls at 3 m
    left the top third of the frame looking past them into empty space, and a 3 m box filled the
    whole frame with one flat wall.

    **The texture tile is the wall's own size: one tile per surface, no repeat and no seam.**
    Anything smaller draws a visible boundary where the pattern wraps, and a repeating wall also
    costs the property the whole view change exists for:

        no repeat     r = 0.723
        repeat 2x2    r = 0.598
        repeat 4x4    r = 0.474
        repeat 8x8    r = 0.626

    **A repeating surface is ambiguous about where you are**, which is the checkerboard failure in
    a gentler form: two different positions look the same, so embedding distance stops tracking
    distance travelled.
    """
    half = size / 2.0
    cx, cy = float(centre[0]), float(centre[1])   # the room follows the robot's spawn, not the origin
    # **Randomised per episode, and no longer one fixed colour per side.** The first design gave
    # each wall its own permanent colour so bearing could be read from the frame -- which is
    # **exactly the single-frame pose readability the egocentric view exists to break**. "See red,
    # facing north" is a landmark, and a landmark is a pose label. Domain randomisation
    # (1703.06907) exists for this: draw the appearance fresh each episode so no colour means a
    # direction across the dataset, and a probe split by clip cannot use it on held-out clips.
    rng = np.random.default_rng(seed * 977 + 11)
    colours = [tuple(0.45 + 0.35 * rng.random(3)) for _ in range(5)]
    specs = [((0, half, height / 2), (size, thickness, height)),
             ((0, -half, height / 2), (size, thickness, height)),
             ((half, 0, height / 2), (thickness, size, height)),
             ((-half, 0, height / 2), (thickness, size, height))]
    if ceiling:
        # **A lid, because the frame looks over the wall.** At 8 m with a 90 degree field the top of
        # the image sits 8 m above the camera, so a 3 m wall leaves a black band across the upper
        # third -- a large region of literally no information, which is the flat-surface failure in
        # its most extreme form.
        specs.append(((0, 0, height), (size, size, thickness)))
    placed = []
    for i, ((x, y, z), (sx, sy, sz)) in enumerate(specs):
        h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [sx, sy, sz], 0)
        sim.setObjectPosition(h, sim.handle_world, [cx + x, cy + y, z])
        sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
        sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 0)
        # **White, because the texture already carries the colour.** Tinting the shape *and* the
        # texture multiplies the two, which is the other half of why the walls were dark.
        sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, [1.0, 1.0, 1.0])
        try:
            # **Through a file, the way `set_floor_texture.py` already does it.** The in-memory
            # route was guessed: `createTexture` returns three values where two were unpacked, so
            # every wall silently came out flat colour -- the blank-surface case measured at
            # r = -0.20, and exactly what the first preview showed. Options 4|8 repeat along U and
            # V; without them one tile stretches across a whole wall.
            # **Written every time, and the name carries the recipe.** Caching on the filename
            # alone silently reused the previous run's textures: the speckled ones from the version
            # this file replaced were still on disk, so a "fixed" recipe rendered with the old
            # images and looked unchanged. A stale cache is indistinguishable from a fix that did
            # not work.
            path = os.path.join(tempfile.gettempdir(), f"ego_wall_{RECIPE}_{seed + i}.png")
            Image.fromarray(_tinted(_wall_texture(seed + i),
                                    colours[i % len(colours)])).save(path)
            carrier, tid, _res = sim.createTexture(path, 0)
            # **Cube mapping, one tile spanning the wall.** The vertical lines that split each
            # wall into sections were a texture seam: a 6 m tile on an 8 m wall repeats 1.33 times,
            # so the wrap was visible. Setting the tile to the wall's own size removes the repeat,
            # and it is also the non-repeating case measured as best -- r 0.723 against 0.474 tiled.
            #
            # **Plane mapping was tried as the fix and is worse**: it projects along one axis and
            # smears the texture into vertical bands on a standing wall. Compared side by side,
            # cube is clean, plane stripes and sphere leaves half the wall dark. A flat-colour wall
            # with no texture shows no lines at all, which is what located the fault.
            span = max(sx, sy, sz)
            sim.setShapeTexture(h, tid, sim.texturemap_cube, 0, [span, span])
            sim.removeObjects([carrier])           # createTexture leaves a carrier plane behind
        except Exception as exc:                   # colour still gives bearing, but no flow
            print(f"    wall {i}: texture FAILED ({exc}) -- this wall carries no optical flow")
        sim.setObjectAlias(h, f"egoWall{i}")
        placed.append(h)
    print(f"    built {len(placed)} walls, {size:.1f} m box, {height:.1f} m high, tile {tile} m")
    return placed
