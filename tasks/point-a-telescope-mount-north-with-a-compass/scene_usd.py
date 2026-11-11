"""The toolbox, the compass and the telescope mount: layout, meshes and USD files.

Positions are metres in the station frame (x forward, y to the robot's left); heights are above the table top. A bearing
is degrees clockwise from straight ahead (+x) seen from above; a heading is anticlockwise from +x (heading = -bearing).
build() writes the files into build/ (not committed: they are generated each time the task file loads).

- toolbox.usda: a fixed (kinematic) body in station coordinates: origin on the table under the arm bases' midpoint.
- compass.usda: a floating articulation, origin at the centre of the housing's base. The housing is the root link; the
  needle is a collider-free link on a revolute joint about the housing's axis, pointing along the housing's +x at joint
  zero. The task drives the joint toward the computed field every control step (task.py).
- mount.usda: a fixed-base articulation, origin on the table at the mount's axis. A hidden pin is the fixed base; the
  mount (column, three legs, equatorial head) turns on it about the vertical. The N leg lies along the file's +x.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import field as F
import usdkit as U

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"

# -- layout -----------------------------------------------------------------------------------------------------------
BOX_LO, BOX_HI = F.BOX_LO, F.BOX_HI                 # x 0.37-0.53, y 0.14-0.46, 0.11 tall
LID_SEAM = 0.085                                    # the lid is the top 2.5 cm
COMPASS_XY = (0.415, 0.195)                         # on the lid, near its inboard end
COMPASS_R, COMPASS_H = 0.0375, 0.025
COMPASS_MASS = 0.09
FACE_Z = 0.0128                                     # the dial face, above the housing's base
PIVOT_Z = 0.016                                     # the needle's pivot, above the housing's base
NEEDLE_L, NEEDLE_W, NEEDLE_T = 0.065, 0.010, 0.0015
COMPASS_START = (*COMPASS_XY, BOX_HI[2])            # the housing's base rests on the lid

MOUNT_AXIS = (0.33, -0.16)
MOUNT_START_HEADING = math.radians(175.0)           # the N leg at bearing 185: pointing at the robot
COLUMN_R, COLUMN_TOP = 0.02, 0.16
COLLAR = (0.028, 0.035, 0.065)                      # radius, z0, z1
LEG_W, LEG_T = 0.024, 0.015
LEG_R0, LEG_Z0 = 0.02, 0.05                         # the leg bar's centre line at the collar
FOOT_R, FOOT_H, FOOT_AT = 0.012, 0.008, 0.12        # rubber feet, 1 mm above the table, centred 12 cm out
FOOT_LIFT = 0.001
LEG_Z1 = FOOT_LIFT + FOOT_H + LEG_T / 2             # the bar's centre line over the foot
LEG_SLOPE = math.atan2(LEG_Z0 - LEG_Z1, FOOT_AT - LEG_R0)
LEG_LEN = math.hypot(FOOT_AT - LEG_R0, LEG_Z0 - LEG_Z1)
LEG_HEADINGS = (0.0, 120.0, 240.0)                  # in the mount's frame; the N leg is the first
MOUNT_MASS = 1.2


def leg_z(r: float) -> float:
    """Height of a leg bar's centre line at distance r from the axis."""
    return LEG_Z0 + (LEG_Z1 - LEG_Z0) * (r - LEG_R0) / (FOOT_AT - LEG_R0)


def _lin(c):
    c = np.asarray(c, float)
    return tuple(float(v) for v in np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4))


LOOKS = {
    "paint_red": U.Look(color=(1.0, 1.0, 1.0), roughness=0.42, texture="paint_red.png", tile=0.16),
    "seam": U.Look(color=_lin((0.10, 0.03, 0.03)), roughness=0.7),
    "steel": U.Look(color=_lin((0.62, 0.63, 0.65)), metallic=1.0, roughness=0.33),
    "grip_black": U.Look(color=_lin((0.05, 0.05, 0.05)), roughness=0.75),
    "abs_black": U.Look(color=_lin((0.04, 0.04, 0.045)), roughness=0.38),
    "face_black": U.Look(color=_lin((0.025, 0.025, 0.03)), roughness=0.9),
    "tick_white": U.Look(color=_lin((0.92, 0.92, 0.90)), roughness=0.6),
    "needle_red": U.Look(color=_lin((0.97, 0.08, 0.04)), roughness=0.45),
    "needle_white": U.Look(color=_lin((0.95, 0.95, 0.93)), roughness=0.5),
    "brushed": U.Look(color=(1.0, 1.0, 1.0), metallic=0.75, roughness=0.32, texture="brushed.png", tile=0.10),
    "anodised_black": U.Look(color=_lin((0.035, 0.035, 0.04)), metallic=0.3, roughness=0.5),
    "paint_yellow": U.Look(color=_lin((0.97, 0.78, 0.05)), roughness=0.45),
    "letter_black": U.Look(color=_lin((0.02, 0.02, 0.02)), roughness=0.6),
    "rubber": U.Look(color=_lin((0.03, 0.03, 0.03)), roughness=0.95),
    "cast": U.Look(color=(1.0, 1.0, 1.0), roughness=0.45, texture="cast.png", tile=0.10),
}


# -- helpers ------------------------------------------------------------------------------------------------------------
def _rot(axis: str, deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def _quat_wxyz(r: np.ndarray) -> tuple[float, float, float, float]:
    w = math.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    if w > 1e-6:
        return (w, (r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w))
    i = int(np.argmax(np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    q = [0.0, 0.0, 0.0]
    q[i] = math.sqrt(max(0.0, 1 + r[i, i] - r[j, j] - r[k, k])) / 2
    q[j] = (r[j, i] + r[i, j]) / (4 * q[i])
    q[k] = (r[k, i] + r[i, k]) / (4 * q[i])
    return ((r[k, j] - r[j, k]) / (4 * q[i]), *q)


def _apply(mesh: U.Mesh, r: np.ndarray, t) -> U.Mesh:
    return U.Mesh(mesh.v @ r.T + np.asarray(t, float), mesh.f)


def rcollider(name: str, size, center, r: np.ndarray, root: str, material: str = "PhysicsMaterial") -> str:
    """A box collider turned by the rotation matrix r (not rendered)."""
    sx, sy, sz = size
    cx, cy, cz = center
    w, x, y, z = _quat_wxyz(r)
    return f"""
        def Cube "{name}" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            rel material:binding:physics = </{root}/{material}>
            uniform token purpose = "guide"
            double size = 1
            float physxCollision:contactOffset = 0.003
            float physxCollision:restOffset = 0
            double3 xformOp:translate = ({cx:.6f}, {cy:.6f}, {cz:.6f})
            quatf xformOp:orient = ({w:.6f}, {x:.6f}, {y:.6f}, {z:.6f})
            float3 xformOp:scale = ({sx:.6f}, {sy:.6f}, {sz:.6f})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
        }}"""


def hull_collider(name: str, pts: np.ndarray, faces: list, root: str, material: str = "PhysicsMaterial") -> str:
    return f"""
        def Mesh "{name}" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            uniform token physics:approximation = "convexHull"
            rel material:binding:physics = </{root}/{material}>
            uniform token purpose = "guide"
            float physxCollision:contactOffset = 0.002
            float physxCollision:restOffset = 0
            int[] faceVertexCounts = [{", ".join(str(len(f)) for f in faces)}]
            int[] faceVertexIndices = [{", ".join(str(i) for f in faces for i in f)}]
            point3f[] points = [{U._fmt(pts, 6)}]
        }}"""


def _visual(parts, root: str, indent: str = "            ") -> str:
    return "".join(U._mesh_usda(n, m, look, root, LOOKS[look], 35.0, indent=indent) for n, m, look in parts)


def _materials(parts, root: str) -> str:
    return "".join(U._material_usda(m, root, LOOKS[m]) for m in sorted({p[2] for p in parts}))


def _header(root: str) -> str:
    return f"""#usda 1.0
(
    defaultPrim = "{root}"
    metersPerUnit = 1
    upAxis = "Z"
)
"""


# -- toolbox --------------------------------------------------------------------------------------------------------------
def toolbox_parts():
    lo, hi = BOX_LO, BOX_HI
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    sx, sy = hi[0] - lo[0], hi[1] - lo[1]
    parts = [
        ("body", U.box((sx - 0.002, sy - 0.002, LID_SEAM - 0.001), (cx, cy, (LID_SEAM - 0.001) / 2), bevel=0.002, round_xy=0.004), "paint_red"),
        ("seam", U.box((sx - 0.006, sy - 0.006, 0.003), (cx, cy, LID_SEAM)), "seam"),
        ("lid", U.box((sx, sy, hi[2] - LID_SEAM - 0.001), (cx, cy, (hi[2] + LID_SEAM + 0.001) / 2), bevel=0.003, round_xy=0.005), "paint_red"),
        ("hinge", U.cylinder(0.004, sy - 0.02, (hi[0] + 0.001, cy, LID_SEAM), axis="y", segments=12), "steel"),
    ]
    for k, y in enumerate((cy - 0.08, cy + 0.08)):
        parts.append((f"latch_plate_{k}", U.box((0.003, 0.032, 0.026), (lo[0] - 0.0015, y, LID_SEAM - 0.012), bevel=0.0008), "steel"))
        parts.append((f"latch_lever_{k}", U.box((0.004, 0.022, 0.034), (lo[0] - 0.004, y, LID_SEAM - 0.004), bevel=0.001), "steel"))
        parts.append((f"latch_keeper_{k}", U.box((0.003, 0.018, 0.012), (lo[0] - 0.0015, y, LID_SEAM + 0.011), bevel=0.0008), "steel"))
    # A U-shaped steel handle folded flat along the middle of the lid, its pivots in two brackets, a black grip on it.
    top = hi[2]
    y0, y1 = cy - 0.05, cy + 0.05
    for k, y in enumerate((y0, y1)):
        parts.append((f"bracket_{k}", U.box((0.014, 0.012, 0.010), (cx, y, top + 0.005), bevel=0.001), "steel"))
    bar = [(cx, y0, top + 0.006), (cx + 0.026, y0, top + 0.006), (cx + 0.028, y0 + 0.004, top + 0.006),
           (cx + 0.028, y1 - 0.004, top + 0.006), (cx + 0.026, y1, top + 0.006), (cx, y1, top + 0.006)]
    parts.append(("handle", U.tube(bar, 0.004, sides=10), "steel"))
    parts.append(("grip", U.cylinder(0.0075, 0.07, (cx + 0.028, cy, top + 0.0075), axis="y", segments=16, bevel=0.002), "grip_black"))
    return parts


def _toolbox() -> str:
    root = "Toolbox"
    lo, hi = BOX_LO, BOX_HI
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    cols = [U.collider("c_box", hi - lo, (lo + hi) / 2, root),
            U.collider("c_handle", (0.045, 0.11, 0.016), (cx + 0.018, cy, hi[2] + 0.008), root)]
    U.write_usda(str(BUILD / "toolbox.usda"), root, toolbox_parts(), LOOKS, cols, mass=6.0, friction=(0.5, 0.4))
    return "build/toolbox.usda"


# -- compass --------------------------------------------------------------------------------------------------------------
def compass_parts():
    r, h = COMPASS_R, COMPASS_H
    housing = U.lathe([(0.0, 0.0), (r - 0.001, 0.0), (r, 0.001), (r, h - 0.0015), (r - 0.0015, h), (r - 0.0045, h),
                       (r - 0.0052, h - 0.001), (r - 0.0055, FACE_Z), (0.0, FACE_Z)], 40)
    parts = [("housing", housing, "abs_black"),
             ("face", U.cylinder(r - 0.0055, 0.0006, (0, 0, FACE_Z + 0.0003), segments=40), "face_black")]
    ticks = []
    for k in range(36):
        long = k % 9 == 0
        r0, r1 = (0.0225 if long else 0.0255), 0.0305
        t = U.box((r1 - r0, 0.0013 if long else 0.001, 0.0004), ((r0 + r1) / 2, 0.0, FACE_Z + 0.0008))
        ticks.append(t.turned(10.0 * k, "z"))
    parts.append(("ticks", U.join(*ticks), "tick_white"))
    return parts


def needle_parts():
    """In the needle link's frame: the pivot at the origin, the north (red) half along +x."""
    w, half, t = NEEDLE_W / 2, NEEDLE_L / 2, NEEDLE_T
    north = U.prism([(0.0, -w), (half, 0.0), (0.0, w)], -t / 2, t / 2)
    south = U.prism([(0.0, w), (-half, 0.0), (0.0, -w)], -t / 2, t / 2)
    cap = U.cylinder(0.0028, 0.0025, (0.0, 0.0, t / 2 + 0.00125), segments=16)
    return [("north", north, "needle_red"), ("south", south, "needle_white"), ("cap", cap, "steel")]


def compass_hull(n: int = 24):
    th = 2 * np.pi * np.arange(n) / n
    pts = [(COMPASS_R * math.cos(a), COMPASS_R * math.sin(a), z) for z in (0.0, COMPASS_H) for a in th]
    faces = [tuple(reversed(range(n))), tuple(n + i for i in range(n))]
    faces += [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    return np.array(pts), faces


def _compass() -> str:
    root = "Compass"
    housing, needle = compass_parts(), needle_parts()
    pts, faces = compass_hull()
    text = _header(root) + f"""
def Xform "{root}"
{{{U.physics_material("PhysicsMaterial", 0.8, 0.7)}

    def Scope "Looks"
    {{{_materials(housing + needle, root)}
    }}

    def Xform "housing" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI", "PhysicsArticulationRootAPI", "PhysxArticulationAPI"]
    )
    {{
        float physics:mass = {COMPASS_MASS}
        float3 physics:centerOfMass = (0, 0, {COMPASS_H / 2:.4f})
        bool physxArticulation:enabledSelfCollisions = 0
        int physxArticulation:solverPositionIterationCount = 16
        int physxArticulation:solverVelocityIterationCount = 1
        float physxRigidBody:maxDepenetrationVelocity = 1
{hull_collider("collider", pts, faces, root)}

        def Xform "visual"
        {{{_visual(housing, root)}
        }}
    }}

    def Xform "needle" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
    )
    {{
        double3 xformOp:translate = (0, 0, {PIVOT_Z})
        uniform token[] xformOpOrder = ["xformOp:translate"]
        float physics:mass = 0.002
        float3 physics:diagonalInertia = (0.0000002, 0.000001, 0.000001)
        float3 physics:centerOfMass = (0, 0, 0)

        def Xform "visual"
        {{{_visual(needle, root)}
        }}
    }}

    def Scope "joints"
    {{
        def PhysicsRevoluteJoint "needle_joint" (
            prepend apiSchemas = ["PhysicsDriveAPI:angular", "PhysxJointAPI"]
        )
        {{
            uniform token physics:axis = "Z"
            rel physics:body0 = </{root}/housing>
            rel physics:body1 = </{root}/needle>
            point3f physics:localPos0 = (0, 0, {PIVOT_Z})
            quatf physics:localRot0 = (1, 0, 0, 0)
            point3f physics:localPos1 = (0, 0, 0)
            quatf physics:localRot1 = (1, 0, 0, 0)
            uniform token drive:angular:physics:type = "force"
            float drive:angular:physics:stiffness = 0
            float drive:angular:physics:damping = 0
            float drive:angular:physics:maxForce = 1
            float physxJoint:armature = 0.00001
        }}
    }}
}}
"""
    (BUILD / "compass.usda").write_text(text)
    return "build/compass.usda"


# -- mount ----------------------------------------------------------------------------------------------------------------
def _leg_rot(heading_deg: float) -> np.ndarray:
    return _rot("z", heading_deg) @ _rot("y", math.degrees(LEG_SLOPE))


def _leg_centre(heading_deg: float):
    rm = (LEG_R0 + FOOT_AT) / 2
    a = math.radians(heading_deg)
    return (rm * math.cos(a), rm * math.sin(a), (LEG_Z0 + LEG_Z1) / 2)


def _letter_n() -> U.Mesh:
    """A block letter N in a leg bar's own frame (x along the bar toward the foot, z up from its top face), reading upright
    with its top toward the foot."""
    hgt, wid, st, th = 0.022, 0.015, 0.0032, 0.0006
    xc = 0.022                                       # from the bar's middle toward the foot
    z = LEG_T / 2 + th / 2
    left = U.box((hgt, st, th), (xc, wid / 2 - st / 2, z))
    right = U.box((hgt, st, th), (xc, -wid / 2 + st / 2, z))
    diag_len = math.hypot(hgt, wid - st)
    diag = U.box((diag_len, st * 1.1, th), (0, 0, 0)).turned(math.degrees(math.atan2(wid - st, hgt)), "z").moved((xc, 0, z + 0.00005))
    return U.join(left, right, diag)


def mount_parts():
    parts = [("column", U.cylinder(COLUMN_R, COLUMN_TOP - 0.003, (0, 0, (COLUMN_TOP + 0.003) / 2), segments=32, bevel=0.001), "brushed"),
             ("collar", U.cylinder(COLLAR[0], COLLAR[2] - COLLAR[1], (0, 0, (COLLAR[1] + COLLAR[2]) / 2), segments=32, bevel=0.002), "anodised_black")]
    for k, hd in enumerate(LEG_HEADINGS):
        r = _leg_rot(hd)
        bar = _apply(U.box((LEG_LEN, LEG_W, LEG_T), (0, 0, 0), bevel=0.0015), r, _leg_centre(hd))
        parts.append((f"leg_{k}", bar, "paint_yellow" if k == 0 else "anodised_black"))
        a = math.radians(hd)
        foot = U.cylinder(FOOT_R, FOOT_H, (FOOT_AT * math.cos(a), FOOT_AT * math.sin(a), FOOT_LIFT + FOOT_H / 2), segments=20, bevel=0.0015)
        parts.append((f"foot_{k}", foot, "rubber"))
    parts.append(("letter_n", _apply(_letter_n(), _leg_rot(0.0), _leg_centre(0.0)), "letter_black"))
    # The equatorial head: azimuth base, latitude block, polar-axis housing tilted 45 degrees up over the N leg,
    # declination housing across it, with a dovetail saddle on one end (no telescope or counterweight shaft yet).
    z0 = COLUMN_TOP
    parts.append(("az_base", U.cylinder(0.034, 0.018, (0, 0, z0 + 0.009), segments=36, bevel=0.002), "cast"))
    parts.append(("lat_block", U.box((0.056, 0.050, 0.030), (-0.004, 0, z0 + 0.033), bevel=0.003), "cast"))
    tilt = _rot("y", 45.0)                            # a z-axis cylinder turned to rise toward +x
    pa_c = np.array([0.004, 0.0, z0 + 0.052])
    parts.append(("polar_housing", _apply(U.cylinder(0.022, 0.085, (0, 0, 0), segments=32, bevel=0.002), tilt, pa_c), "cast"))
    axis = tilt @ np.array([0.0, 0.0, 1.0])
    top = pa_c + axis * 0.0425
    parts.append(("polar_cap", _apply(U.cylinder(0.012, 0.012, (0, 0, 0), segments=24), tilt, top + axis * 0.004), "anodised_black"))
    dec_c = top - axis * 0.012
    parts.append(("dec_housing", U.cylinder(0.019, 0.080, tuple(dec_c), axis="y", segments=32, bevel=0.002), "cast"))
    parts.append(("saddle", U.box((0.070, 0.016, 0.030), (dec_c[0], dec_c[1] + SADDLE_Y, dec_c[2]), bevel=0.002), "anodised_black"))
    for k, s in enumerate((-1, 1)):
        parts.append((f"lat_knob_{k}", U.cylinder(0.006, 0.016, (-0.036, s * 0.014, z0 + 0.03), axis="x", segments=16, bevel=0.001), "anodised_black"))
        parts.append((f"az_knob_{k}", U.cylinder(0.006, 0.014, (0.0, s * 0.04, z0 + 0.012), axis="y", segments=16, bevel=0.001), "anodised_black"))
    return parts


SADDLE_Y = -0.048       # the saddle on the declination housing's -y end (toward the robot's left at the start)
_HEAD_TILT = _rot("y", 45.0)
_PA_C = np.array([0.004, 0.0, COLUMN_TOP + 0.052])
_DEC_C = _PA_C + _HEAD_TILT @ np.array([0.0, 0.0, 1.0]) * (0.0425 - 0.012)


def collider_boxes():
    """The mount's colliders as (name, centre, rotation, size) in its own frame; the scorer measures release to these."""
    out = [(f"leg_{k}", _leg_centre(hd), _leg_rot(hd), (LEG_LEN, LEG_W, LEG_T)) for k, hd in enumerate(LEG_HEADINGS)]
    out.append(("column", (0.0, 0.0, (COLUMN_TOP + 0.003) / 2), np.eye(3), (2 * COLUMN_R, 2 * COLUMN_R, COLUMN_TOP - 0.003)))
    out.append(("base", (-0.002, 0.0, COLUMN_TOP + 0.025), np.eye(3), (0.07, 0.07, 0.05)))
    out.append(("polar", tuple(_PA_C), _HEAD_TILT, (0.044, 0.044, 0.095)))
    out.append(("dec", tuple(_DEC_C), np.eye(3), (0.04, 0.08, 0.04)))
    out.append(("saddle", (_DEC_C[0], _DEC_C[1] + SADDLE_Y, _DEC_C[2]), np.eye(3), (0.07, 0.016, 0.03)))
    return out


def _mount() -> str:
    root = "Mount"
    parts = mount_parts()
    cols = [rcollider(f"c_{n}", size, c, r, root) for n, c, r, size in collider_boxes()]
    text = _header(root) + f"""
def Xform "{root}"
{{{U.physics_material("PhysicsMaterial", 0.9, 0.8)}

    def Scope "Looks"
    {{{_materials(parts, root)}
    }}

    def Xform "pin" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI", "PhysicsArticulationRootAPI", "PhysxArticulationAPI"]
    )
    {{
        float physics:mass = 0.2
        float3 physics:diagonalInertia = (0.0001, 0.0001, 0.0001)
        bool physxArticulation:enabledSelfCollisions = 0
        int physxArticulation:solverPositionIterationCount = 16
        int physxArticulation:solverVelocityIterationCount = 1
    }}

    def Xform "body" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
    )
    {{
        float physics:mass = {MOUNT_MASS}
        float3 physics:centerOfMass = (0, 0, 0.12)
        float3 physics:diagonalInertia = (0.012, 0.012, 0.006)
        float physxRigidBody:maxDepenetrationVelocity = 1
{"".join(cols)}

        def Xform "visual"
        {{{_visual(parts, root)}
        }}
    }}

    def Scope "joints"
    {{
        def PhysicsFixedJoint "root_joint"
        {{
            rel physics:body0 = </{root}>
            rel physics:body1 = </{root}/pin>
            point3f physics:localPos0 = (0, 0, 0)
            quatf physics:localRot0 = (1, 0, 0, 0)
            point3f physics:localPos1 = (0, 0, 0)
            quatf physics:localRot1 = (1, 0, 0, 0)
        }}

        def PhysicsRevoluteJoint "turn" (
            prepend apiSchemas = ["PhysicsDriveAPI:angular", "PhysxJointAPI"]
        )
        {{
            uniform token physics:axis = "Z"
            rel physics:body0 = </{root}/pin>
            rel physics:body1 = </{root}/body>
            point3f physics:localPos0 = (0, 0, 0)
            quatf physics:localRot0 = (1, 0, 0, 0)
            point3f physics:localPos1 = (0, 0, 0)
            quatf physics:localRot1 = (1, 0, 0, 0)
            uniform token drive:angular:physics:type = "force"
            float drive:angular:physics:stiffness = 0
            float drive:angular:physics:damping = 0
            float drive:angular:physics:maxForce = 1000
            float physxJoint:armature = 0.001
        }}
    }}
}}
"""
    (BUILD / "mount.usda").write_text(text)
    return "build/mount.usda"


def build() -> dict[str, str]:
    BUILD.mkdir(exist_ok=True)
    return {"toolbox": _toolbox(), "compass": _compass(), "mount": _mount()}


if __name__ == "__main__":
    print(build())
