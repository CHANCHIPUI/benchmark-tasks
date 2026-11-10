"""The soroban and its mat: layout and USD files.

Positions are metres in the station frame (x forward, y to the robot's left); heights are above the table top. Both
files have their origin on the table under the arm bases' midpoint, so their parts sit at the station xy below.
build() writes them into build/ (not committed: they are generated each time the task file loads).

The soroban is one articulation. Its fixed base is the frame (bars and beam, box colliders), and each of its 25 beads
is a link on a prismatic joint along x. Joint zero is the cleared position. A 1-bead (below the beam, nearer the robot)
moves +x toward the beam, a 5-bead (above the beam, further away) moves -x; each can travel TRAVEL to its stop at the
beam. Beads on one rod collide with each other (self-collision is on), so pushing one pushes those ahead of it. The rods
are thin box colliders on the frame, which a bead never touches (a jointed pair does not collide).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import usdkit as U

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"

# -- layout -----------------------------------------------------------------------------------------------------------
MAT_CENTRE, MAT_SIZE = (0.29, 0.0), (0.36, 0.40, 0.003)
BASE = MAT_SIZE[2]                     # the frame stands on the mat
FRAME_X = (0.15, 0.43)                 # outer, along x (280 mm)
FRAME_HALF_Y = 0.16                    # outer half width (320 mm)
FRAME_H = 0.050                        # frame and beam thickness, lying flat
FRAME_TOP = BASE + FRAME_H
SIDE_W, NEAR_W, FAR_W = 0.025, 0.040, 0.036
BEAM_X = (0.318, 0.338)                # the reckoning bar
ROD_Z = BASE + FRAME_H / 2             # rods at mid-thickness
ROD_R = 0.002
ROD_Y = (0.088, 0.044, 0.0, -0.044, -0.088)   # rods 1..5, robot's left to right
DOT_RODS = (1, 4)                      # the unit dots on the beam (rod 4 is the ones rod)
DOT_R = 0.007

BEAD_R, BEAD_T, HUB_R = 0.020, 0.024, 0.007   # 40 mm across, 24 mm thick along the rod
BEAD_MASS = 0.008
TRAVEL = 0.032                         # free travel of each group, cleared stop to the beam
EARTH = ("e1", "e2", "e3", "e4")       # 1-beads, e1 nearest the beam
EARTH_X = {b: FRAME_X[0] + NEAR_W + BEAD_T * (4 - k) - BEAD_T / 2 for k, b in enumerate(EARTH)}  # cleared centres
HEAVEN_X = FRAME_X[1] - FAR_W - BEAD_T / 2                                                      # cleared centre
BEAD_TOP = ROD_Z + BEAD_R

assert abs(EARTH_X["e1"] + BEAD_T / 2 + TRAVEL - BEAM_X[0]) < 1e-9
assert abs(HEAVEN_X - BEAD_T / 2 - TRAVEL - BEAM_X[1]) < 1e-9


def bead_names() -> list[tuple[int, str]]:
    return [(r, b) for r in range(1, 6) for b in ("h", *EARTH)]


def joint_name(rod: int, bead: str) -> str:
    return f"slide_{rod}_{bead}"


def cleared_xyz(rod: int, bead: str) -> tuple[float, float, float]:
    x = HEAVEN_X if bead == "h" else EARTH_X[bead]
    return (x, ROD_Y[rod - 1], ROD_Z)


# -- looks --------------------------------------------------------------------------------------------------------------
def _lin(c):
    c = np.asarray(c, float)
    return tuple(float(v) for v in np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4))


LOOKS = {
    "wood": U.Look(color=(1.0, 1.0, 1.0), roughness=0.45, texture="wood.png", tile=0.10),
    "wood_v": U.Look(color=(1.0, 1.0, 1.0), roughness=0.45, texture="wood_v.png", tile=0.10),
    "beam_white": U.Look(color=_lin((0.93, 0.93, 0.90)), roughness=0.35),
    "dot_black": U.Look(color=_lin((0.04, 0.04, 0.04)), roughness=0.5),
    "bead_yellow": U.Look(color=_lin((0.98, 0.80, 0.08)), roughness=0.32),
    "rod_black": U.Look(color=_lin((0.06, 0.06, 0.06)), metallic=0.6, roughness=0.35),
    "rubber": U.Look(color=(1.0, 1.0, 1.0), roughness=0.9, texture="rubber.png", tile=0.10),
}


# -- meshes -------------------------------------------------------------------------------------------------------------
def _box(lo, hi, bevel=0.0) -> U.Mesh:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return U.box(hi - lo, (lo + hi) / 2, bevel=bevel)


def bead_mesh(segments: int = 32) -> U.Mesh:
    """A double cone with flat hubs and a slightly rounded rim, its axis along x, centred on its origin."""
    h, r = BEAD_T / 2, BEAD_R
    profile = [(0.0, -h), (HUB_R, -h), (HUB_R + 0.0008, -h + 0.0006), (r - 0.0006, -0.0014), (r, -0.0004),
               (r, 0.0004), (r - 0.0006, 0.0014), (HUB_R + 0.0008, h - 0.0006), (HUB_R, h), (0.0, h)]
    return U.lathe(profile, segments).turned(90.0, "y")


def bead_hull(segments: int = 12) -> np.ndarray:
    """The collider's points: a 12-sided double cone, 0.1 mm thinner on each face than the bead so stacked beads do
    not touch until pushed, with a flat on top so a fingertip pressing down meets an edge, not a corner."""
    h = BEAD_T / 2 - 0.0001
    rim = BEAD_R / np.cos(np.pi / segments) * 0.985          # mean radius about the bead's 20 mm
    th = np.pi / 2 + np.pi / segments + 2 * np.pi * np.arange(segments) / segments
    pts = []
    for x, rad in ((-h, HUB_R), (-0.0005, rim), (0.0005, rim), (h, HUB_R)):
        pts += [(x, rad * np.cos(t), rad * np.sin(t)) for t in th]
    return np.array(pts)


def _hull_usda(name: str, pts: np.ndarray, root: str, indent: str) -> str:
    n = len(pts) // 4
    faces = []
    for k in range(3):
        for i in range(n):
            j = (i + 1) % n
            faces.append((k * n + i, k * n + j, (k + 1) * n + j, (k + 1) * n + i))
    faces.append(tuple(reversed(range(n))))
    faces.append(tuple(3 * n + i for i in range(n)))
    return f"""
{indent}def Mesh "{name}" (
{indent}    prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
{indent})
{indent}{{
{indent}    uniform token physics:approximation = "convexHull"
{indent}    rel material:binding:physics = </{root}/BeadPhysics>
{indent}    uniform token purpose = "guide"
{indent}    float physxCollision:contactOffset = 0.001
{indent}    float physxCollision:restOffset = 0
{indent}    int[] faceVertexCounts = [{", ".join(str(len(f)) for f in faces)}]
{indent}    int[] faceVertexIndices = [{", ".join(str(i) for f in faces for i in f)}]
{indent}    point3f[] points = [{U._fmt(pts, 6)}]
{indent}}}"""


def _frame_parts():
    """Visual parts and box colliders of the frame, in the soroban file's frame (origin on the table at x 0, y 0)."""
    x0, x1 = FRAME_X
    hy = FRAME_HALF_Y
    z0, z1 = BASE, FRAME_TOP
    pieces = {
        "near_bar": ((x0, -hy, z0), (x0 + NEAR_W, hy, z1), "wood_v"),
        "far_bar": ((x1 - FAR_W, -hy, z0), (x1, hy, z1), "wood_v"),
        "left_bar": ((x0 + NEAR_W, hy - SIDE_W, z0), (x1 - FAR_W, hy, z1), "wood"),
        "right_bar": ((x0 + NEAR_W, -hy, z0), (x1 - FAR_W, -hy + SIDE_W, z1), "wood"),
        "beam": ((BEAM_X[0], -hy + SIDE_W, z0), (BEAM_X[1], hy - SIDE_W, z1), "beam_white"),
    }
    parts, cols = [], []
    for name, (lo, hi, look) in pieces.items():
        parts.append((name, _box(lo, hi, bevel=0.0015), look))
        lo, hi = np.asarray(lo), np.asarray(hi)
        cols.append(U.collider(f"c_{name}", hi - lo, (lo + hi) / 2, "Soroban"))
    for r in DOT_RODS:
        c = ((BEAM_X[0] + BEAM_X[1]) / 2, ROD_Y[r - 1], FRAME_TOP + 0.0002)
        parts.append((f"dot_{r}", U.cylinder(DOT_R, 0.0004, c, segments=24), "dot_black"))
    for r, y in enumerate(ROD_Y, start=1):
        length = x1 - x0 - 0.01
        parts.append((f"rod_{r}", U.cylinder(ROD_R, length, ((x0 + x1) / 2, y, ROD_Z), axis="x", segments=10), "rod_black"))
        # The rod stops a fingertip in the gaps; the beads on it are jointed to the frame, so they never touch it.
        cols.append(U.collider(f"c_rod_{r}", (length, 2 * ROD_R, 2 * ROD_R), ((x0 + x1) / 2, y, ROD_Z), "Soroban"))
    return parts, cols


def _soroban() -> str:
    root = "Soroban"
    frame_parts, frame_cols = _frame_parts()
    used = sorted({p[2] for p in frame_parts} | {"bead_yellow"})
    materials = "".join(U._material_usda(m, root, LOOKS[m]) for m in used)
    frame_meshes = "".join(U._mesh_usda(n, m, look, root, LOOKS[look], 35.0, indent="            ") for n, m, look in frame_parts)
    bead_vis = bead_mesh()
    hull = bead_hull()
    beads, joints = [], []
    for rod, b in bead_names():
        name = f"bead_{rod}_{b}"
        x, y, z = cleared_xyz(rod, b)
        vis = U._mesh_usda("shell", bead_vis, "bead_yellow", root, LOOKS["bead_yellow"], 50.0, indent="            ")
        beads.append(f"""
    def Xform "{name}" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
    )
    {{
        double3 xformOp:translate = ({x:.6f}, {y:.6f}, {z:.6f})
        uniform token[] xformOpOrder = ["xformOp:translate"]
        float physics:mass = {BEAD_MASS}
        float physxRigidBody:maxDepenetrationVelocity = 1
{_hull_usda("collider", hull, root, "        ")}

        def Xform "visual"
        {{{vis}
        }}
    }}""")
        lo, hi = (-TRAVEL, 0.0) if b == "h" else (0.0, TRAVEL)
        joints.append(f"""
        def PhysicsPrismaticJoint "{joint_name(rod, b)}" (
            prepend apiSchemas = ["PhysicsDriveAPI:linear", "PhysxJointAPI"]
        )
        {{
            uniform token physics:axis = "X"
            rel physics:body0 = </{root}/frame>
            rel physics:body1 = </{root}/{name}>
            point3f physics:localPos0 = ({x:.6f}, {y:.6f}, {z:.6f})
            quatf physics:localRot0 = (1, 0, 0, 0)
            point3f physics:localPos1 = (0, 0, 0)
            quatf physics:localRot1 = (1, 0, 0, 0)
            float physics:lowerLimit = {lo:.6f}
            float physics:upperLimit = {hi:.6f}
            uniform token drive:linear:physics:type = "force"
            float drive:linear:physics:stiffness = 0
            float drive:linear:physics:damping = 1
            float drive:linear:physics:maxForce = 1000
            float physxJoint:armature = 0.00001
        }}""")
    text = f"""#usda 1.0
(
    defaultPrim = "{root}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}"
{{{U.physics_material("PhysicsMaterial", 0.5, 0.4)}{U.physics_material("BeadPhysics", 0.6, 0.5)}

    def Scope "Looks"
    {{{materials}
    }}

    def Xform "frame" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsArticulationRootAPI", "PhysxArticulationAPI"]
    )
    {{
        float physics:mass = 0.6
        bool physxArticulation:enabledSelfCollisions = 1
        int physxArticulation:solverPositionIterationCount = 16
        int physxArticulation:solverVelocityIterationCount = 1
{"".join(frame_cols)}

        def Xform "visual"
        {{{frame_meshes}
        }}
    }}
{"".join(beads)}

    def Scope "joints"
    {{
        def PhysicsFixedJoint "root_joint"
        {{
            rel physics:body0 = </{root}>
            rel physics:body1 = </{root}/frame>
            point3f physics:localPos0 = (0, 0, 0)
            quatf physics:localRot0 = (1, 0, 0, 0)
            point3f physics:localPos1 = (0, 0, 0)
            quatf physics:localRot1 = (1, 0, 0, 0)
        }}{"".join(joints)}
    }}
}}
"""
    BUILD.mkdir(exist_ok=True)
    (BUILD / "soroban.usda").write_text(text)
    return "build/soroban.usda"


def _mat() -> str:
    root = "DeskMat"
    (cx, cy), (sx, sy, sz) = MAT_CENTRE, MAT_SIZE
    parts = [("mat", U.box((sx, sy, sz), (cx, cy, sz / 2), bevel=0.0008, round_xy=0.01), "rubber")]
    cols = [U.collider("c_mat", (sx, sy, sz), (cx, cy, sz / 2), root)]
    U.write_usda(str(BUILD / "mat.usda"), root, parts, LOOKS, cols, mass=1.0, friction=(0.9, 0.8))  # kinematic: the task file gives it no mass
    return "build/mat.usda"


def build() -> dict[str, str]:
    BUILD.mkdir(exist_ok=True)
    return {"soroban": _soroban(), "mat": _mat()}


if __name__ == "__main__":
    print(build())
