"""The teleoperation rig: layout, kinematics and USD files.

Positions are metres in the station frame (x forward, y to the robot's left); heights are above the table top. Every
file has its origin on the table under the arm bases' midpoint, so its parts sit at the station xy below. build() writes
them into build/ (not committed: they are generated each time the task file loads).

Both arms are two-link arms with vertical joint axes, like small SCARA arms, mounted the same way round. A shoulder angle
is link 1's direction (base axis to elbow axis) from +x, counter-clockwise seen from above; an elbow angle is the turn
from link 1's direction to link 2's (elbow axis to tip axis), also counter-clockwise. Each arm is authored in one pose
(FOLLOWER_AUTHORED, LEADER_AUTHORED), where its joints read 0, so an angle is the authored angle plus the joint's reading.

- Follower (black): fixed base, shoulder and elbow on stiff position drives that the task file aims every control step.
- Leader (yellow): a 0.75-scale replica. Its shoulder and elbow are damped and unsprung, so it stays where it is pushed;
  a knob over its tip turns freely, so a gripper holding the knob at a fixed yaw can drag the tip anywhere.
- ENABLE box: a fixed enclosure with a sprung push button on a sliding joint. The button has two caps, unlit and lit;
  the task file shows the lit one once teleoperation is on.
- Glass: a loose tumbler. Mats, the dot and the cables are fixed.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import usdkit as U

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"

# -- layout -----------------------------------------------------------------------------------------------------------
BOARD_X, BOARD_Y, BOARD_T = (0.49, 0.90), (-0.28, 0.28), 0.006       # light grey HDPE board under the follower
MAT_CENTRE, MAT_SIZE = (0.28, 0.27), (0.32, 0.36, 0.003)              # black rubber mat under the leader

F0 = (0.55, 0.0)                     # follower's base axis
F_L1, F_L2 = 0.14, 0.16              # between axes
F_W, F_T = 0.028, 0.016              # link width and thickness
F_BASE_R, F_BASE_TOP = 0.035, 0.050  # base column
F_Z1 = F_BASE_TOP                    # link 1 from here up F_T, link 2 on top of it
F_TIP_R, F_POINTER_R, F_POINTER_LOW = 0.015, 0.006, BOARD_T + 0.005

SCALE = 0.75                         # the leader is a 0.75-scale replica
L0 = (0.24, 0.26)                    # leader's base axis
L_L1, L_L2 = F_L1 * SCALE, F_L2 * SCALE
L_W, L_T = 0.021, 0.012
L_BASE_R, L_BASE_TOP = 0.025, 0.020
L_Z1 = L_BASE_TOP
KNOB_R, KNOB_H = 0.0125, 0.050

SHOULDER_RANGE = (-135.0, 110.0)     # deg, both arms
ELBOW_RANGE = (0.0, 150.0)           # deg: the elbow bends counter-clockwise only

FOLLOWER_START = (-56.0, 30.0)       # deg (shoulder, elbow): parked reaching out to the right
LEADER_START = (64.0, 140.0)         # deg: folded, about 120 deg away
DOT_POSE = (24.0, 30.0)              # deg: the follower pose that puts its tip on the dot
# The poses the arms are authored in, where their joints read 0. The follower is authored at its start; the leader in the
# middle of its ranges, so that no joint ever turns more than 180 deg from where it reads 0. A reset event sets the start.
FOLLOWER_AUTHORED = FOLLOWER_START
LEADER_AUTHORED = ((SHOULDER_RANGE[0] + SHOULDER_RANGE[1]) / 2, (ELBOW_RANGE[0] + ELBOW_RANGE[1]) / 2)

GLASS_XY = (0.793, 0.034)            # the tumbler's base centre
GLASS_R_BASE, GLASS_R_TOP, GLASS_H, GLASS_MASS = 0.032, 0.037, 0.100, 0.25
DOT_R = 0.025

BOX_XY, BOX_SIZE = (0.22, -0.08), (0.08, 0.08, 0.045)                 # ENABLE box
BUTTON_R, BUTTON_PROUD, BUTTON_TRAVEL = 0.015, 0.008, 0.006
BUTTON_XY = (BOX_XY[0] + 0.01, BOX_XY[1])                             # the button sits 1 cm beyond the box centre
BUTTON_TOP = BOX_SIZE[2] + BUTTON_PROUD
BEZEL_R, BEZEL_PROUD = 0.019, 0.002


# -- kinematics --------------------------------------------------------------------------------------------------------
def fk(base, l1, l2, q1_deg, q2_deg):
    """Elbow and tip axis positions (x, y) for a shoulder and elbow angle in degrees."""
    a, b = math.radians(q1_deg), math.radians(q1_deg + q2_deg)
    e = np.array([base[0] + l1 * math.cos(a), base[1] + l1 * math.sin(a)])
    return e, e + l2 * np.array([math.cos(b), math.sin(b)])


def ik(base, l1, l2, p):
    """(shoulder, elbow) in degrees for a tip axis at p, elbow bent counter-clockwise (the only way it bends)."""
    d = np.asarray(p, float) - np.asarray(base, float)
    r = float(np.linalg.norm(d))
    c = (r * r - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    q2 = math.acos(max(-1.0, min(1.0, c)))
    q1 = math.atan2(d[1], d[0]) - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
    return math.degrees(q1), math.degrees(q2)


def follower_tip(q1, q2):
    return fk(F0, F_L1, F_L2, q1, q2)[1]


def leader_tip(q1, q2):
    return fk(L0, L_L1, L_L2, q1, q2)[1]


def leader_ik(p):
    return ik(L0, L_L1, L_L2, p)


DOT_XY = tuple(float(v) for v in follower_tip(*DOT_POSE))


# -- looks --------------------------------------------------------------------------------------------------------------
def _lin(c):
    c = np.asarray(c, float)
    return tuple(float(v) for v in np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4))


LOOKS = {
    "anodised": U.Look(color=_lin((0.075, 0.075, 0.085)), metallic=0.6, roughness=0.42),
    "silver": U.Look(color=_lin((0.78, 0.79, 0.80)), metallic=1.0, roughness=0.28),
    "steel": U.Look(color=_lin((0.62, 0.63, 0.65)), metallic=1.0, roughness=0.35),
    "pla_yellow": U.Look(color=_lin((0.98, 0.76, 0.06)), roughness=0.5),
    "servo_black": U.Look(color=_lin((0.05, 0.05, 0.055)), roughness=0.55),
    "knob_rubber": U.Look(color=_lin((0.06, 0.06, 0.06)), roughness=0.85),
    "rubber": U.Look(color=(1.0, 1.0, 1.0), roughness=0.9, texture="rubber.png", tile=0.10),
    "hdpe": U.Look(color=_lin((0.80, 0.81, 0.80)), roughness=0.6),
    "dot_green": U.Look(color=_lin((0.12, 0.72, 0.22)), roughness=0.55),
    "glass": U.Look(color=_lin((0.30, 0.55, 0.92)), roughness=0.05, opacity=0.45),
    "abs_black": U.Look(color=_lin((0.09, 0.09, 0.095)), roughness=0.55),
    "label": U.Look(color=(1.0, 1.0, 1.0), roughness=0.55, texture="enable_label.png", tile=1.0),
    "button_off": U.Look(color=_lin((0.06, 0.17, 0.48)), roughness=0.35),
    "button_on": U.Look(color=_lin((0.35, 0.62, 1.0)), roughness=0.3, emissive=(0.25, 0.6, 2.0)),
    "cable": U.Look(color=_lin((0.30, 0.31, 0.32)), roughness=0.6),
}


# -- meshes -------------------------------------------------------------------------------------------------------------
def stadium(length: float, width: float, seg: int = 10) -> np.ndarray:
    """A link's outline in its own frame: from the axis at (0, 0) to the axis at (length, 0), round ends, counter-clockwise."""
    r = width / 2
    pts = [(length + r * math.cos(a), r * math.sin(a)) for a in np.linspace(-math.pi / 2, math.pi / 2, seg + 1)]
    pts += [(r * math.cos(a), r * math.sin(a)) for a in np.linspace(math.pi / 2, 3 * math.pi / 2, seg + 1)]
    return np.array(pts)


def disc(radius: float, z0: float, z1: float, xy=(0.0, 0.0), seg: int = 32, bevel: float = 0.0) -> U.Mesh:
    return U.cylinder(radius, z1 - z0, (xy[0], xy[1], (z0 + z1) / 2), segments=seg, bevel=bevel)


def _hull(name: str, mesh: U.Mesh, root: str, material: str, indent: str = "        ") -> str:
    """A convex-hull collider from a mesh's points (not rendered: purpose "guide")."""
    f = mesh.f
    return f"""
{indent}def Mesh "{name}" (
{indent}    prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
{indent})
{indent}{{
{indent}    uniform token physics:approximation = "convexHull"
{indent}    rel material:binding:physics = </{root}/{material}>
{indent}    uniform token purpose = "guide"
{indent}    float physxCollision:contactOffset = 0.002
{indent}    float physxCollision:restOffset = 0
{indent}    int[] faceVertexCounts = [{", ".join("3" for _ in range(len(f)))}]
{indent}    int[] faceVertexIndices = [{", ".join(str(int(i)) for i in f.reshape(-1))}]
{indent}    point3f[] points = [{U._fmt(mesh.v, 6)}]
{indent}}}"""


def _visuals(parts, root: str, indent: str = "            ") -> str:
    return "".join(U._mesh_usda(p[0], p[1], p[2], root, LOOKS[p[2]], 40.0, indent=indent, hidden=len(p) > 3 and p[3])
                   for p in parts)


def _quat(yaw_deg: float) -> str:
    h = math.radians(yaw_deg) / 2
    return f"({math.cos(h):.8f}, 0, 0, {math.sin(h):.8f})"


def _body(name: str, xy, mass: float, colliders: str, parts, root: str, extra_api: str = "", extra: str = "",
          label: str = "", yaw: float = 0.0) -> str:
    api = '"PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"' + extra_api
    return f"""
    def Xform "{name}" (
        prepend apiSchemas = [{api}]
    )
    {{
        double3 xformOp:translate = ({xy[0]:.6f}, {xy[1]:.6f}, 0)
        quatf xformOp:orient = {_quat(yaw)}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
        float physics:mass = {mass:.4f}
        float physxRigidBody:maxDepenetrationVelocity = 1{extra}
{colliders}

        def Xform "visual"
        {{{_visuals(parts, root)}{label}
        }}
    }}"""


def _quad_usda(name: str, corners, z: float, mat: str, root: str, indent: str = "            ") -> str:
    """A flat quad facing up, its texture mapped once: corners in order (u, v) = (0, 0), (1, 0), (1, 1), (0, 1)."""
    pts = ", ".join(f"({x:.6f}, {y:.6f}, {z:.6f})" for x, y in corners)
    return f"""
{indent}def Mesh "{name}" (
{indent}    prepend apiSchemas = ["MaterialBindingAPI"]
{indent})
{indent}{{
{indent}    rel material:binding = </{root}/Looks/{mat}>
{indent}    int[] faceVertexCounts = [4]
{indent}    int[] faceVertexIndices = [0, 1, 2, 3]
{indent}    point3f[] points = [{pts}]
{indent}    normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (
{indent}        interpolation = "faceVarying"
{indent}    )
{indent}    texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
{indent}        interpolation = "faceVarying"
{indent}    )
{indent}    uniform token subdivisionScheme = "none"
{indent}}}"""


def _root_api() -> str:
    return ', "PhysicsArticulationRootAPI", "PhysxArticulationAPI"'


def _root_extra(iters: int = 32) -> str:
    return f"""
        bool physxArticulation:enabledSelfCollisions = 0
        int physxArticulation:solverPositionIterationCount = {iters}
        int physxArticulation:solverVelocityIterationCount = 4"""


def _fixed(root: str, body: str) -> str:
    return f"""
        def PhysicsFixedJoint "root_joint"
        {{
            rel physics:body0 = </{root}>
            rel physics:body1 = </{root}/{body}>
            point3f physics:localPos0 = (0, 0, 0)
            quatf physics:localRot0 = (1, 0, 0, 0)
            point3f physics:localPos1 = (0, 0, 0)
            quatf physics:localRot1 = (1, 0, 0, 0)
        }}"""


def _revolute(name: str, root: str, b0: str, b1: str, pos0, limits=None, rot0: float = 0.0) -> str:
    lim = "" if limits is None else f"""
            float physics:lowerLimit = {limits[0]:.4f}
            float physics:upperLimit = {limits[1]:.4f}"""
    return f"""
        def PhysicsRevoluteJoint "{name}" (
            prepend apiSchemas = ["PhysicsDriveAPI:angular", "PhysxJointAPI"]
        )
        {{
            uniform token physics:axis = "Z"
            rel physics:body0 = </{root}/{b0}>
            rel physics:body1 = </{root}/{b1}>
            point3f physics:localPos0 = ({pos0[0]:.6f}, {pos0[1]:.6f}, {pos0[2]:.6f})
            quatf physics:localRot0 = {_quat(rot0)}
            point3f physics:localPos1 = (0, 0, {pos0[2]:.6f})
            quatf physics:localRot1 = (1, 0, 0, 0){lim}
            uniform token drive:angular:physics:type = "force"
            float drive:angular:physics:stiffness = 0
            float drive:angular:physics:damping = 0
            float drive:angular:physics:maxForce = 1000
            float physxJoint:armature = 0.0001
        }}"""


def _usda(root: str, body: str) -> str:
    return f"""#usda 1.0
(
    defaultPrim = "{root}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}"
{{{body}
}}
"""


def _looks(root: str, names) -> str:
    return f"""
    def Scope "Looks"
    {{{"".join(U._material_usda(m, root, LOOKS[m]) for m in sorted(set(names)))}
    }}"""


def _arm(root: str, base_xy, l1, l2, w, t, base_r, z1, look_link, look_cap, masses, follower: bool, authored) -> str:
    """One arm, authored at shoulder and elbow angles `authored` (deg), where its joints read 0. Bodies: base (fixed, the
    articulation root), link1, link2 (and knob for the leader). Each body's origin is on the table under its joint axis,
    and each link's own x runs along it. Joint limits are the arm's ranges less the authored angles."""
    a1, a2 = authored
    elbow_xy, tip_xy = fk(base_xy, l1, l2, a1, a2)
    z2 = z1 + t
    used = {look_link, look_cap}
    # base: a column on a flange, with a cable gland at its back
    base_parts = [("column", disc(base_r, 0.0, z1 - 0.001, seg=40, bevel=0.002), look_cap if not follower else "anodised"),
                  ("flange", disc(base_r * 1.45, 0.0, 0.004 if follower else 0.003, seg=40, bevel=0.001), "anodised" if follower else "servo_black"),
                  ("gland", U.cylinder(0.006, 0.014, (-base_r - 0.005, 0.0, 0.012 if follower else 0.008), axis="x", segments=16), "anodised" if follower else "servo_black")]
    base_parts = [(n, m.moved((base_xy[0], base_xy[1], 0.0)), look) for n, m, look in base_parts]
    used |= {p[2] for p in base_parts}
    base_col = _hull("c_column", disc(base_r, 0.0, z1 - 0.001, xy=base_xy, seg=24), root, "PhysicsMaterial")
    # link 1: bar from the shoulder axis to the elbow axis, a joint cap over the shoulder
    out1 = stadium(l1, w)
    link1_parts = [("bar", U.prism(out1, z1, z2, bevel=0.0012), look_link),
                   ("shoulder_cap", disc(w * 0.62, z2, z2 + 0.004, seg=32, bevel=0.001), look_cap)]
    link1_col = _hull("c_bar", U.prism(out1, z1, z2), root, "PhysicsMaterial")
    # link 2: bar from the elbow axis to the tip axis, a joint cap over the elbow
    z3 = z2 + t
    out2 = stadium(l2, w)
    link2_parts = [("bar", U.prism(out2, z2, z3, bevel=0.0012), look_link),
                   ("elbow_cap", disc(w * 0.62, z3, z3 + 0.004, seg=32, bevel=0.001), look_cap)]
    link2_col = _hull("c_bar", U.prism(out2, z2, z3), root, "PhysicsMaterial")
    bodies = ""
    if follower:
        # the tip: a silver end cap over a steel pointer that ends just above the board
        link2_parts += [("tip_cap", disc(F_TIP_R, z3, z3 + 0.006, xy=(l2, 0.0), seg=32, bevel=0.0015), "silver"),
                        ("pointer", disc(F_POINTER_R, F_POINTER_LOW + 0.003, z2, xy=(l2, 0.0), seg=20), "steel"),
                        ("pointer_end", U.lathe([(0.0, F_POINTER_LOW), (F_POINTER_R * 0.5, F_POINTER_LOW + 0.0005),
                                                 (F_POINTER_R, F_POINTER_LOW + 0.003)], 20).moved((l2, 0.0, 0.0)), "steel")]
        link2_col += _hull("c_pointer", disc(F_POINTER_R, F_POINTER_LOW, z2, xy=(l2, 0.0), seg=12), root, "PhysicsMaterial")
        used |= {"silver", "steel"}
    else:
        # GELLO-style: a servo body at the elbow under link 2's cap
        link1_parts.append(("elbow_servo", U.box((0.030, 0.024, t + 0.008), (l1, 0.0, z1 - 0.008 + (t + 0.008) / 2), bevel=0.001, round_xy=0.002), "servo_black"))
        used.add("servo_black")
    # The root body sits at the file's origin: a reset puts the root at the object's pose (the origin), not at its axis.
    bodies += _body("base", (0.0, 0.0), masses[0], base_col, base_parts, root, _root_api(), _root_extra())
    bodies += _body("link1", base_xy, masses[1], link1_col, link1_parts, root, yaw=a1)
    bodies += _body("link2", elbow_xy, masses[2], link2_col, link2_parts, root, yaw=a1 + a2)
    joints = _fixed(root, "base")
    joints += _revolute("shoulder", root, "base", "link1", (base_xy[0], base_xy[1], z1), (SHOULDER_RANGE[0] - a1, SHOULDER_RANGE[1] - a1), rot0=a1)
    joints += _revolute("elbow", root, "link1", "link2", (l1, 0.0, z2), (ELBOW_RANGE[0] - a2, ELBOW_RANGE[1] - a2), rot0=a2)
    if not follower:
        kz0 = z3
        knob = U.lathe([(0.0, kz0), (KNOB_R * 0.8, kz0), (KNOB_R, kz0 + 0.002), (KNOB_R, kz0 + KNOB_H - 0.003),
                        (KNOB_R - 0.002, kz0 + KNOB_H), (0.0, kz0 + KNOB_H)], 32)
        ribs = [(f"rib_{i}", U.box((0.0016, 0.0016, KNOB_H - 0.012), (KNOB_R * math.cos(a), KNOB_R * math.sin(a), kz0 + KNOB_H / 2), bevel=0.0),
                 "knob_rubber") for i, a in enumerate(np.linspace(0, 2 * math.pi, 16, endpoint=False))]
        knob_parts = [("knob", knob, "knob_rubber")] + ribs
        used.add("knob_rubber")
        knob_col = _hull("c_knob", disc(KNOB_R + 0.0005, kz0 + 0.001, kz0 + KNOB_H, seg=24), root, "KnobMaterial")
        bodies += _body("knob", tip_xy, masses[3], knob_col, knob_parts, root, yaw=a1 + a2)
        joints += _revolute("knob_spin", root, "link2", "knob", (l2, 0.0, kz0))
    return f"""{U.physics_material("PhysicsMaterial", 0.5, 0.4)}{U.physics_material("KnobMaterial", 1.0, 0.9)}{_looks(root, used)}
{bodies}

    def Scope "joints"
    {{{joints}
    }}"""


def _follower() -> str:
    root = "Follower"
    text = _usda(root, _arm(root, F0, F_L1, F_L2, F_W, F_T, F_BASE_R, F_Z1, "anodised", "silver",
                            (2.0, 0.25, 0.20), follower=True, authored=FOLLOWER_AUTHORED))
    (BUILD / "follower.usda").write_text(text)
    return "build/follower.usda"


def _leader() -> str:
    root = "Leader"
    text = _usda(root, _arm(root, L0, L_L1, L_L2, L_W, L_T, L_BASE_R, L_Z1, "pla_yellow", "servo_black",
                            (0.5, 0.035, 0.030, 0.015), follower=False, authored=LEADER_AUTHORED))
    (BUILD / "leader.usda").write_text(text)
    return "build/leader.usda"


def _enable_box() -> str:
    """The enclosure (fixed base) and the push button on a sliding joint along z; joint 0 is the button up."""
    root = "EnableBox"
    sx, sy, sz = BOX_SIZE
    bx, by = BOX_XY            # the box body sits at the file's origin (a reset puts the root there)
    ux, uy = BUTTON_XY
    box_parts = [("enclosure", U.box((sx, sy, sz), (bx, by, sz / 2), bevel=0.003, round_xy=0.006), "abs_black"),
                 ("bezel", U.lathe([(BUTTON_R + 0.0005, sz), (BEZEL_R, sz), (BEZEL_R, sz + BEZEL_PROUD - 0.0006),
                                    (BEZEL_R - 0.0008, sz + BEZEL_PROUD), (BUTTON_R + 0.0005, sz + BEZEL_PROUD)], 40).moved((ux, uy, 0.0)), "silver")]
    # The label on the near half of the lid, reading left to right from the robot's side.
    lx0, lx1 = bx - sx / 2 + 0.006, bx - sx / 2 + 0.020
    ly0, ly1 = by - 0.026, by + 0.026
    label = _quad_usda("label", [(lx0, ly1), (lx0, ly0), (lx1, ly0), (lx1, ly1)], sz + 0.0003, "label", root)
    box_cols = U.collider("c_enclosure", (sx, sy, sz), (bx, by, sz / 2), root)
    for i, a in enumerate(np.linspace(0, 2 * math.pi, 8, endpoint=False)):   # the bezel ring, as 8 small blocks
        c = np.array(BUTTON_XY) + (BUTTON_R + BEZEL_R) / 2 * np.array([math.cos(a), math.sin(a)])
        box_cols += U.collider(f"c_bezel_{i}", (0.004, 0.004, BEZEL_PROUD), (c[0], c[1], sz + BEZEL_PROUD / 2), root)
    cap_lo, cap_hi = BUTTON_TOP - 0.014, BUTTON_TOP
    cap = U.lathe([(0.0, cap_lo), (BUTTON_R - 0.0004, cap_lo), (BUTTON_R - 0.0004, cap_hi - 0.0025),
                   (BUTTON_R - 0.0016, cap_hi - 0.0004), (BUTTON_R - 0.0035, cap_hi), (0.0, cap_hi)], 40)
    button_parts = [("cap_off", cap, "button_off"), ("cap_on", cap, "button_on", True)]
    button_col = _hull("c_cap", disc(BUTTON_R - 0.0004, cap_lo, cap_hi, seg=24), root, "PhysicsMaterial")
    used = {p[2] for p in box_parts} | {"button_off", "button_on", "label"}
    bodies = _body("box", (0.0, 0.0), 0.3, box_cols, box_parts, root, _root_api(), _root_extra(16), label=label)
    bodies += _body("button", BUTTON_XY, 0.010, button_col, button_parts, root)
    joint = f"""
        def PhysicsPrismaticJoint "press" (
            prepend apiSchemas = ["PhysicsDriveAPI:linear", "PhysxJointAPI"]
        )
        {{
            uniform token physics:axis = "Z"
            rel physics:body0 = </{root}/box>
            rel physics:body1 = </{root}/button>
            point3f physics:localPos0 = ({ux:.6f}, {uy:.6f}, {cap_lo:.6f})
            quatf physics:localRot0 = (1, 0, 0, 0)
            point3f physics:localPos1 = (0, 0, {cap_lo:.6f})
            quatf physics:localRot1 = (1, 0, 0, 0)
            float physics:lowerLimit = {-BUTTON_TRAVEL:.6f}
            float physics:upperLimit = 0
            uniform token drive:linear:physics:type = "force"
            float drive:linear:physics:stiffness = 0
            float drive:linear:physics:damping = 0
            float drive:linear:physics:maxForce = 1000
            float physxJoint:armature = 0.00001
        }}"""
    body = f"""{U.physics_material("PhysicsMaterial", 0.6, 0.5)}{_looks(root, used)}
{bodies}

    def Scope "joints"
    {{{_fixed(root, "box")}{joint}
    }}"""
    (BUILD / "enable_box.usda").write_text(_usda(root, body))
    return "build/enable_box.usda"


def _glass() -> str:
    """A loose tumbler, origin at its base centre: tapered, 2.5 mm wall, 8 mm base. Collider: the outer hull."""
    root = "Glass"
    rb, rt, h = GLASS_R_BASE, GLASS_R_TOP, GLASS_H
    wall, base = 0.0025, 0.008
    profile = [(0.0, 0.0), (rb - 0.002, 0.0), (rb, 0.002), (rt, h - 0.001), (rt - 0.0008, h), (rt - wall + 0.0004, h),
               (rt - wall, h - 0.001), (rb - wall + (rt - rb) * base / h, base + 0.002), (rb - wall - 0.003, base), (0.0, base)]
    parts = [("tumbler", U.lathe(profile, 48), "glass")]
    col = _hull("c_glass", U.lathe([(0.0, 0.0), (rb, 0.0), (rt, h), (0.0, h)], 20), root, "PhysicsMaterial")
    U.write_usda(str(BUILD / "glass.usda"), root, parts, LOOKS, [col], mass=GLASS_MASS, friction=(0.4, 0.35),
                 crease=50.0)
    return "build/glass.usda"


def _cable_path(points, z: float, n: int = 40) -> np.ndarray:
    """A smooth path through 2-D points (Catmull-Rom), at height z."""
    p = np.asarray(points, float)
    p = np.vstack([p[0], p, p[-1]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for t in np.linspace(0, 1, n // (len(p) - 3) + 1)[:-1]:
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(p[-2])
    out = np.array(out)
    return np.c_[out, np.full(len(out), z)]


def _mats() -> str:
    """The follower's board with the green dot, the leader's mat, and the two cables (visual only). Fixed."""
    root = "Mats"
    (cx, cy), (sx, sy, sz) = MAT_CENTRE, MAT_SIZE
    bx, by = (BOARD_X[0] + BOARD_X[1]) / 2, (BOARD_Y[0] + BOARD_Y[1]) / 2
    bsx, bsy = BOARD_X[1] - BOARD_X[0], BOARD_Y[1] - BOARD_Y[0]
    r = 0.003
    lead = _cable_path([(L0[0] - L_BASE_R - 0.012, L0[1]), (0.17, 0.22), (0.15, 0.08), (0.17, -0.03),
                        (BOX_XY[0] - 0.02, BOX_XY[1] + BOX_SIZE[1] / 2 + 0.004)], r)
    lead[:, 2] = np.where(np.hypot(lead[:, 0] - cx, lead[:, 1] - cy) < 0.2, sz + r, r)
    lead[:, 2] = np.where((np.abs(lead[:, 0] - cx) < sx / 2) & (np.abs(lead[:, 1] - cy) < sy / 2), sz + r, r)
    foll = _cable_path([(F0[0] - F_BASE_R * 1.45 - 0.004, F0[1] - 0.01), (0.47, -0.03), (0.36, -0.10), (0.29, -0.11),
                        (BOX_XY[0] + BOX_SIZE[0] / 2 + 0.004, BOX_XY[1] - 0.01)], r)
    foll[:, 2] = np.where(foll[:, 0] > BOARD_X[0], BOARD_T + r, r)
    parts = [("board", U.box((bsx, bsy, BOARD_T), (bx, by, BOARD_T / 2), bevel=0.0015, round_xy=0.012), "hdpe"),
             ("dot", disc(DOT_R, BOARD_T, BOARD_T + 0.0004, xy=DOT_XY, seg=48), "dot_green"),
             ("mat", U.box((sx, sy, sz), (cx, cy, sz / 2), bevel=0.0008, round_xy=0.01), "rubber"),
             ("cable_leader", U.tube(lead, r, 10), "cable"),
             ("cable_follower", U.tube(foll, r, 10), "cable")]
    cols = [U.collider("c_board", (bsx, bsy, BOARD_T), (bx, by, BOARD_T / 2), root),
            U.collider("c_mat", (sx, sy, sz), (cx, cy, sz / 2), root)]
    U.write_usda(str(BUILD / "mats.usda"), root, parts, LOOKS, cols, mass=1.0, friction=(0.6, 0.5))  # kinematic: no mass in the task file
    return "build/mats.usda"


def build() -> dict[str, str]:
    BUILD.mkdir(exist_ok=True)
    return {"follower": _follower(), "leader": _leader(), "enable_box": _enable_box(), "glass": _glass(), "mats": _mats()}


if __name__ == "__main__":
    print(build())
    print("dot", DOT_XY, "follower tip at start", follower_tip(*FOLLOWER_START), "leader knob at start", leader_tip(*LEADER_START))
