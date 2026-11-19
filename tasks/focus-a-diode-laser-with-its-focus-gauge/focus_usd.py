"""The task's objects as plain-text USD: the open-frame diode laser engraver (frame, gantry, carriage), its honeycomb
working panel, the 3 mm plywood sheet, and the laser module as a fixed-base articulation: a vertical slide (module on
its carriage), the focal length setting bar on a hinge on its +y side, and the thumb-screw handle on a hinge on its -y
side.

Visuals are meshes with the real product's proportions; colliders are separate plain boxes and cylinders (purpose
"guide"). Units: metres. Each file's origin is stated with its builder.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import usdkit as uk
from usdkit import Look

HERE = Path(__file__).resolve().parent
OUT = HERE / "build"

# -- the laser module (20 W diode module of an open-frame engraver) -----------------------------------------------------
# Module frame (articulation file origin): on the module's vertical axis, at the light shield's lower edge, at the start.
BODY = 0.060                 # body 6 x 6 cm in plan
BODY_H = 0.128               # shield edge to the fan housing's top
SHIELD_H = 0.026             # orange light shield, from the edge up
SHIELD_W = 0.046
MODULE_MASS = 0.80
SLIDE_LOW, SLIDE_HIGH = -0.018, 0.022     # slide travel from the start (shield edge 4.5 cm up): 0.5 cm above the
                                          # honeycomb at the bottom, 4 cm of travel

# Focal length setting bar: a steel strip on a hinge on the +y face; axis along y.
BAR_PIVOT = np.array([-0.015, BODY / 2 + 0.0115, 0.040])   # 1.5 cm behind the front face, 4.0 cm above the shield edge
BAR_L = 0.048                # axis to tip
BAR_W, BAR_T = 0.008, 0.003  # width (in its swing plane), thickness (along y)
BAR_COL_T = 0.007            # collider thickness along y (thicker than the 3 mm strip so a fingertip cannot pass)
BAR_TOP = 0.004              # the strip's rounded end above the axis
BAR_MASS = 0.009
STOW_DEG = 170.0             # stowed: 10 degrees short of upright, leaning toward the robot (-x)
HANG_DEG = 0.0               # hanging stop: straight down
BAR_STOP_HI = 2.0            # degrees past the stowed angle (the magnet's seat)
BAR_COM_OFF = -0.002         # the bar's centre of mass lies 2 mm ahead of its line (toward -x hanging), so it hangs back on its stop

# Thumb-screw handle: a lever on a hinge on the -y face; axis along y, 2 cm out from the face.
HANDLE_AXIS = np.array([0.015, -BODY / 2 - 0.020, 0.060])
HANDLE_L = 0.030             # axis to tip; at its tight stop it points at the robot (-x), level
HANDLE_W, HANDLE_T = 0.010, 0.006   # along y, along z
HANDLE_MASS = 0.010
HANDLE_LOW = -55.0           # degrees; tip-down loosens


def bar_dir(theta_deg: float) -> np.ndarray:
    """Unit vector from the bar's axis to its tip at angle theta (0 hanging straight down, + swings the tip toward -x)."""
    t = math.radians(theta_deg)
    return np.array([-math.sin(t), 0.0, -math.cos(t)])


def bar_reach() -> float:
    """G: how far the bar's lowest point (its rounded end) lies below the shield edge, hanging on its stop."""
    c = BAR_PIVOT + (BAR_L - BAR_W / 2) * bar_dir(HANG_DEG)
    return -(c[2] - BAR_W / 2)


LOOKS = {
    "red": Look(color=(0.62, 0.05, 0.09), metallic=0.75, roughness=0.38, texture="brushed.png", tile=0.08),
    "red_dark": Look(color=(0.36, 0.03, 0.05), metallic=0.7, roughness=0.45),
    "gold": Look(color=(0.80, 0.55, 0.16), metallic=0.85, roughness=0.32, texture="brushed.png", tile=0.10),
    "shield": Look(color=(1.0, 0.42, 0.02), roughness=0.15, opacity=0.72),
    "lens": Look(color=(0.05, 0.05, 0.06), metallic=0.3, roughness=0.2),
    "fan": Look(texture="fan.png", tile_uv=(BODY, BODY), origin=(-BODY / 2, -BODY / 2, 0.0), roughness=0.6),
    "steel": Look(color=(0.62, 0.63, 0.65), metallic=0.9, roughness=0.3),
    "black": Look(color=(0.025, 0.025, 0.028), roughness=0.55),
    "magnet": Look(color=(0.12, 0.12, 0.13), metallic=0.6, roughness=0.35),
    "white": Look(color=(0.85, 0.85, 0.83), roughness=0.4),
    "cable": Look(color=(0.02, 0.02, 0.02), roughness=0.7),
    "belt": Look(color=(0.03, 0.03, 0.03), roughness=0.8),
    "honeycomb": Look(texture="honeycomb.png", tile=0.10, metallic=0.6, roughness=0.4),
    "alu": Look(color=(0.75, 0.76, 0.78), metallic=0.85, roughness=0.35, texture="brushed.png", tile=0.15),
    "wood": Look(texture="basswood.png", tile=0.10, roughness=0.75),
    "ply_edge": Look(texture="ply_edge.png", tile_uv=(0.01, 0.003), roughness=0.8),
    "rubber": Look(color=(0.05, 0.05, 0.05), roughness=0.9),
}


# -- helpers --------------------------------------------------------------------------------------------------------
def _fmt(a, d=6):
    return ", ".join("(" + ", ".join(f"{x:.{d}f}" for x in row) + ")" for row in np.atleast_2d(a))


def box_col(name, size, center, root, rot=None, offset=0.001, material="PhysicsMaterial", indent="    ") -> str:
    sx, sy, sz = size
    cx, cy, cz = center
    w, qx, qy, qz = rot if rot is not None else (1.0, 0.0, 0.0, 0.0)
    text = f"""
    def Cube "{name}" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        rel material:binding:physics = </{root}/{material}>
        uniform token purpose = "guide"
        double size = 1
        float physxCollision:contactOffset = {offset}
        float physxCollision:restOffset = 0
        double3 xformOp:translate = ({cx:.6f}, {cy:.6f}, {cz:.6f})
        quatf xformOp:orient = ({w:.6f}, {qx:.6f}, {qy:.6f}, {qz:.6f})
        float3 xformOp:scale = ({sx:.6f}, {sy:.6f}, {sz:.6f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }}"""
    return text.replace("\n    ", "\n" + indent)


def _quat_y(deg):
    h = math.radians(deg) / 2
    return (math.cos(h), 0.0, math.sin(h), 0.0)


def _meshes(parts, root, indent="        ") -> str:
    return "".join(uk._mesh_usda(p[0], p[1], p[2], root, LOOKS[p[2]], 35.0, indent=indent) for p in parts)


def _looks(parts, root) -> str:
    return "".join(uk._material_usda(m, root, LOOKS[m]) for m in sorted({p[2] for p in parts}))


def _write(name: str, text: str) -> str:
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.usda"
    rel = Path("..") / "textures"
    text = text.replace(f"@{(HERE / 'textures').as_posix()}/", f"@{rel.as_posix()}/")
    path.write_text(text)
    return str(path.relative_to(HERE))


def _body(name, root, parts, colliders, mass=None, friction=(0.6, 0.5), solver=16, extra_mat="", depen=1.0) -> str:
    kin = mass is None
    m = 1.0 if kin else mass
    head = """ (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
)"""
    body = f"""
    float physics:mass = {m:.6f}
    int physxRigidBody:solverPositionIterationCount = {solver}
    int physxRigidBody:solverVelocityIterationCount = 4
    float physxRigidBody:maxDepenetrationVelocity = {depen}
    float physxRigidBody:sleepThreshold = 0
    float physxRigidBody:stabilizationThreshold = 0{chr(10) + "    bool physics:kinematicEnabled = 1" if kin else ""}
"""
    text = f"""#usda 1.0
(
    defaultPrim = "{root}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}"{head}
{{{body}{uk.physics_material("PhysicsMaterial", *friction)}{extra_mat}
{colliders}

    def Scope "Looks"
    {{{_looks(parts, root)}
    }}

    def Xform "visual"
    {{{_meshes(parts, root)}
    }}
}}
"""
    return _write(name, text)


def box(size, center, bevel=0.0):
    return uk.box(size, center=center, bevel=bevel)


# -- the machine: frame, gantry, carriage (fixed). Origin: table top, below the module's axis. -------------------------
# Station layout (m, from the arms' midpoint): frame x 0.20..0.79, y -0.30..+0.30; module axis at (0.29, 0.06).
FRAME_X = (0.19, 0.78)
FRAME_Y = (-0.30, 0.30)
RAIL_W, RAIL_H = 0.040, 0.050
MODULE_XY = np.array([0.29, 0.06])
BEAM_X = (MODULE_XY[0] + 0.066, MODULE_XY[0] + 0.106)      # gantry beam (station x)
BEAM_Z = (0.110, 0.150)
CARRIAGE_X = (MODULE_XY[0] + 0.031, MODULE_XY[0] + 0.036)
HONEY = (0.245, 0.745, -0.25, 0.25, 0.022)   # x0, x1, y0, y1, height


def machine() -> str:
    """Origin: the table top at the module's axis (station (0.29, 0.06))."""
    root = "Engraver"
    ox, oy = MODULE_XY
    x0, x1 = FRAME_X[0] - ox, FRAME_X[1] - ox
    y0, y1 = FRAME_Y[0] - oy, FRAME_Y[1] - oy
    w, h = RAIL_W, RAIL_H
    parts, cols = [], ""
    rails = {
        "rail_front": ((w, y1 - y0, h), (x0 + w / 2, (y0 + y1) / 2, h / 2)),
        "rail_back": ((w, y1 - y0, h), (x1 - w / 2, (y0 + y1) / 2, h / 2)),
        "rail_left": ((x1 - x0 - 2 * w, w, h), ((x0 + x1) / 2, y1 - w / 2, h / 2)),
        "rail_right": ((x1 - x0 - 2 * w, w, h), ((x0 + x1) / 2, y0 + w / 2, h / 2)),
    }
    for n, (s, c) in rails.items():
        parts.append((n, box(s, c, bevel=0.002), "red"))
        cols += box_col(f"col_{n}", s, c, root)
    # belts and slots along the side rails' inner faces, feet under the corners
    for side, yy in (("l", y1 - w - 0.0008), ("r", y0 + w + 0.0008)):
        parts.append((f"belt_{side}", box((x1 - x0 - 2 * w - 0.01, 0.0015, 0.008), ((x0 + x1) / 2, yy, 0.032)), "belt"))
        parts.append((f"slot_{side}", box((x1 - x0 - 2 * w - 0.01, 0.0012, 0.004), ((x0 + x1) / 2, yy, 0.014)), "black"))
    for cx in (x0 + 0.02, x1 - 0.02):
        for cy in (y0 + 0.02, y1 - 0.02):
            parts.append((f"foot_{len(parts)}", uk.cylinder(0.009, 0.004, center=(cx, cy, 0.002), segments=16), "red_dark"))
    # gantry beam with its end plates riding on the side rails
    bx0, bx1 = BEAM_X[0] - ox, BEAM_X[1] - ox
    bz0, bz1 = BEAM_Z
    beam = ((bx1 - bx0, y1 - y0 - 2 * w + 0.004, bz1 - bz0), ((bx0 + bx1) / 2, (y0 + y1) / 2, (bz0 + bz1) / 2))
    parts.append(("beam", box(*beam, bevel=0.002), "gold"))
    cols += box_col("col_beam", *beam, root)
    parts.append(("beam_slot", box((0.0015, y1 - y0 - 2 * w, 0.004), (bx0 - 0.0006, (y0 + y1) / 2, bz0 + 0.012)), "black"))
    for side, yy in (("l", y1 - w - 0.004), ("r", y0 + w + 0.004)):
        plate = ((0.075, 0.008, 0.125), ((bx0 + bx1) / 2, yy, h + 0.0625))
        parts.append((f"end_{side}", box(*plate, bevel=0.0015), "red"))
        cols += box_col(f"col_end_{side}", *plate, root)
    # the carriage plate behind the module, and its wheels on the beam
    cx0, cx1 = CARRIAGE_X[0] - ox, CARRIAGE_X[1] - ox
    car = ((cx1 - cx0, 0.070, 0.110), ((cx0 + cx1) / 2, 0.0, 0.115))
    parts.append(("carriage", box(*car, bevel=0.001), "red_dark"))
    cols += box_col("col_carriage", *car, root)
    bridge = ((bx0 - cx1 + 0.001, 0.070, BEAM_Z[1] - BEAM_Z[0]), ((cx1 + bx0) / 2, 0.0, (BEAM_Z[0] + BEAM_Z[1]) / 2))
    parts.append(("bridge", box(*bridge, bevel=0.001), "red_dark"))
    cols += box_col("col_bridge", *bridge, root)
    return _body("machine", root, parts, cols, mass=None, friction=(0.5, 0.45))


def honeycomb() -> str:
    """Origin: the table top at the panel's centre."""
    root = "Honeycomb"
    hx0, hx1, hy0, hy1, hh = HONEY
    sx, sy = hx1 - hx0, hy1 - hy0
    parts = [
        ("core", box((sx - 0.006, sy - 0.006, 0.0005), (0, 0, hh - 0.00025)), "honeycomb"),
        ("base", box((sx, sy, hh - 0.0005), (0, 0, (hh - 0.0005) / 2)), "alu"),
    ]
    for n, s, c in (("edge_f", (0.003, sy, 0.0007), (-sx / 2 + 0.0015, 0, hh - 0.0003)),
                    ("edge_b", (0.003, sy, 0.0007), (sx / 2 - 0.0015, 0, hh - 0.0003)),
                    ("edge_l", (sx, 0.003, 0.0007), (0, sy / 2 - 0.0015, hh - 0.0003)),
                    ("edge_r", (sx, 0.003, 0.0007), (0, -sy / 2 + 0.0015, hh - 0.0003))):
        parts.append((n, box(s, c), "alu"))
    cols = box_col("col_panel", (sx, sy, hh), (0, 0, hh / 2), root, offset=0.002)
    return _body("honeycomb", root, parts, cols, mass=None, friction=(0.5, 0.45))


SHEET = (0.200, 0.300, 0.003)   # x by y by thickness
SHEET_MASS = 0.080


def sheet() -> str:
    """Origin: the sheet's centre on its bottom face."""
    root = "Sheet"
    sx, sy, t = SHEET
    parts = [
        ("face", box((sx, sy, 0.0004), (0, 0, t - 0.0002)), "wood"),
        ("ply", box((sx - 0.0002, sy - 0.0002, t - 0.0004), (0, 0, (t - 0.0004) / 2)), "ply_edge"),
    ]
    cols = box_col("col_sheet", (sx, sy, t), (0, 0, t / 2), root, offset=0.001)
    return _body("sheet", root, parts, cols, mass=SHEET_MASS, friction=(0.5, 0.45), solver=32, depen=0.5)


# -- the laser module articulation --------------------------------------------------------------------------------------
def module_parts():
    """(name, mesh, look) in the module frame, at the start pose (shield edge at z 0)."""
    h = BODY / 2
    parts = []
    # light shield (orange acrylic box, open at the bottom) and the lens barrel seen through it
    parts.append(("shield", uk.box((SHIELD_W, SHIELD_W, SHIELD_H), center=(0, 0, SHIELD_H / 2), bevel=0.0015), "shield"))
    parts.append(("lens", uk.cylinder(0.008, 0.020, center=(0, 0, 0.012), segments=20), "lens"))
    # air-assist nozzle connector (white) at the shield's front-right corner
    parts.append(("nozzle", uk.cylinder(0.005, 0.016, center=(-0.016, -0.026, 0.016), segments=16), "white"))
    # body: lower heat-sink block and upper fan housing, red anodised
    parts.append(("body", uk.box((BODY, BODY, BODY_H - SHIELD_H - 0.004), center=(0, 0, SHIELD_H + (BODY_H - SHIELD_H - 0.004) / 2),
                                 bevel=0.0025), "red"))
    parts.append(("seam", uk.box((BODY - 0.002, BODY - 0.002, 0.004), center=(0, 0, BODY_H - 0.034)), "red_dark"))
    parts.append(("top", uk.box((BODY, BODY, 0.004), center=(0, 0, BODY_H - 0.002), bevel=0.0012), "red"))
    parts.append(("fan", uk.box((BODY - 0.004, BODY - 0.004, 0.0004), center=(0, 0, BODY_H + 0.0002)), "fan"))
    # bar bracket (hinge block) and the magnet that holds the stowed bar
    px, py, pz = BAR_PIVOT
    parts.append(("bracket", uk.box((0.012, py - h - BAR_COL_T / 2 - 0.0005, 0.012),
                                    center=(px, (h + py - BAR_COL_T / 2 - 0.0005) / 2, pz), bevel=0.001), "black"))
    top = BAR_PIVOT + (BAR_L - 0.006) * bar_dir(STOW_DEG)
    parts.append(("magnet", uk.cylinder(0.004, py - h - BAR_COL_T / 2 - 0.0005, center=(top[0], (h + py - BAR_COL_T / 2 - 0.0005) / 2, top[2]),
                                        axis="y", segments=16), "magnet"))
    # the power and signal cable from the fan housing's top, back over the gantry
    pts = [(0.006, -0.012, BODY_H), (0.006, -0.012, BODY_H + 0.030), (0.025, -0.012, BODY_H + 0.050),
           (0.055, -0.012, BODY_H + 0.045)]
    parts.append(("cable", uk.tube(pts, 0.003, 8), "cable"))
    # thumb-screw boss on the -y face
    ax, ay, az = HANDLE_AXIS
    parts.append(("boss", uk.cylinder(0.0055, -ay - h, center=(ax, (ay - h) / 2, az), axis="y", segments=20), "steel"))
    return parts


def module_colliders(root, ind):
    cols = box_col("col_body", (BODY, BODY, BODY_H - SHIELD_H), (0, 0, SHIELD_H + (BODY_H - SHIELD_H) / 2), root, indent=ind)
    cols += box_col("col_shield", (SHIELD_W, SHIELD_W, SHIELD_H), (0, 0, SHIELD_H / 2), root, indent=ind)
    return cols


def bar_parts():
    """The bar in its stowed pose (module frame). Its free end is rounded (radius half its width)."""
    rt = BAR_W / 2
    t = BAR_TOP + BAR_L - rt
    strip = uk.box((BAR_W, BAR_T, t), center=(0, 0, -t / 2 + BAR_TOP), bevel=0.0008)
    end = uk.cylinder(rt, BAR_T, center=(0, 0, -(BAR_L - rt)), axis="y", segments=20)
    knuckle = uk.cylinder(BAR_W / 2, BAR_T + 0.001, center=(0, 0, 0), axis="y", segments=16)
    pin = uk.cylinder(0.0018, BAR_T + 0.004, center=(0, 0, 0), axis="y", segments=12)
    out = []
    for n, m, look in (("strip", strip, "steel"), ("end", end, "steel"), ("knuckle", knuckle, "steel"), ("pin", pin, "black")):
        out.append((n, m.turned(STOW_DEG, "y").moved(BAR_PIVOT), look))
    return out


def bar_colliders(root, ind):
    """A box along the strip and a cylinder (axis y) at its rounded end: the end rolls on the sheet."""
    rt = BAR_W / 2
    t = BAR_TOP + BAR_L - rt
    centre = BAR_PIVOT + (t / 2 - BAR_TOP) * bar_dir(STOW_DEG)
    cols = box_col("col_bar", (BAR_W, BAR_COL_T, t), centre, root, rot=_quat_y(STOW_DEG), offset=0.002,
                   material="BarMaterial", indent=ind)
    c = BAR_PIVOT + (BAR_L - rt) * bar_dir(STOW_DEG)
    cols += f"""
{ind}def Cylinder "col_bar_end" (
{ind}    prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
{ind})
{ind}{{
{ind}    rel material:binding:physics = </{root}/BarMaterial>
{ind}    uniform token purpose = "guide"
{ind}    uniform token axis = "Y"
{ind}    double radius = {rt:.6f}
{ind}    double height = {BAR_COL_T:.6f}
{ind}    float physxCollision:contactOffset = 0.002
{ind}    float physxCollision:restOffset = 0
{ind}    double3 xformOp:translate = ({c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f})
{ind}    uniform token[] xformOpOrder = ["xformOp:translate"]
{ind}}}"""
    return cols


def handle_parts():
    """The lever at its tight stop, pointing -x (module frame)."""
    ax, ay, az = HANDLE_AXIS
    lever = uk.box((HANDLE_L + 0.004, HANDLE_W, HANDLE_T), center=(-(HANDLE_L + 0.004) / 2 + 0.004, 0, 0), bevel=0.0015)
    hub = uk.cylinder(0.0075, HANDLE_W + 0.002, center=(0, 0, 0), axis="y", segments=20)
    grip = uk.box((0.008, HANDLE_W + 0.001, HANDLE_T + 0.002), center=(-HANDLE_L + 0.004, 0, 0), bevel=0.0015)
    return [(n, m.moved(HANDLE_AXIS), look) for n, m, look in (("lever", lever, "black"), ("hub", hub, "steel"),
                                                                  ("grip", grip, "black"))]


def handle_colliders(root, ind):
    return box_col("col_handle", (HANDLE_L + 0.006, HANDLE_W + 0.003, HANDLE_T + 0.002),
                   tuple(HANDLE_AXIS + np.array([-(HANDLE_L + 0.006) / 2 + 0.006, 0, 0])), root, offset=0.002,
                   material="HandleMaterial", indent=ind)


def _inertia_box(m, s):
    a, b, c = s
    return (m * (b * b + c * c) / 12, m * (a * a + c * c) / 12, m * (a * a + b * b) / 12)


BAR_MAT = """
    def Material "BarMaterial" (
        prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]
    )
    {
        float physics:staticFriction = 0.15
        float physics:dynamicFriction = 0.12
        float physics:restitution = 0
        uniform token physxMaterial:frictionCombineMode = "min"
    }
"""


def laser() -> str:
    root = "Laser"
    ind = "        "

    def link(name, mass, com, inertia, cols="", parts=(), art=False):
        api = '"PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"'
        extra = ""
        if art:
            api += ', "PhysicsArticulationRootAPI", "PhysxArticulationAPI"'
            extra = """
        bool physxArticulation:enabledSelfCollisions = 0
        int physxArticulation:solverPositionIterationCount = 32
        int physxArticulation:solverVelocityIterationCount = 4
        float physxArticulation:sleepThreshold = 0
        float physxArticulation:stabilizationThreshold = 0"""
        vis = f"""

        def Xform "visual"
        {{{_meshes(parts, root, indent="            ")}
        }}""" if parts else ""
        return f"""
    def Xform "{name}" (
        prepend apiSchemas = [{api}]
    )
    {{
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        float physics:mass = {mass:.6f}
        point3f physics:centerOfMass = ({com[0]:.6f}, {com[1]:.6f}, {com[2]:.6f})
        float3 physics:diagonalInertia = ({inertia[0]:.3e}, {inertia[1]:.3e}, {inertia[2]:.3e})
        float physxRigidBody:maxDepenetrationVelocity = 0.5
        float physxRigidBody:sleepThreshold = 0
        float physxRigidBody:stabilizationThreshold = 0{extra}
{cols}{vis}
    }}"""

    mparts, bparts, hparts = module_parts(), bar_parts(), handle_parts()
    t = math.radians(STOW_DEG)
    behind = np.array([math.cos(t), 0.0, -math.sin(t)])          # across the strip, toward +x when hanging
    bar_com = BAR_PIVOT + (BAR_L / 2 - BAR_TOP / 2) * bar_dir(STOW_DEG) + BAR_COM_OFF * behind
    links = link("carriage", 0.05, (0, 0, 0.06), (1e-5, 1e-5, 1e-5), art=True)
    links += link("module", MODULE_MASS, (0, 0, 0.065), _inertia_box(MODULE_MASS, (BODY, BODY, BODY_H)),
                  module_colliders(root, ind), mparts)
    bi = BAR_MASS * (BAR_L + BAR_TOP) ** 2 / 12
    links += link("bar", BAR_MASS, bar_com, (bi, bi, BAR_MASS * BAR_W ** 2 / 12 + 1e-8), bar_colliders(root, ind), bparts)
    hc = HANDLE_AXIS + np.array([-HANDLE_L / 2, 0, 0])
    hi = HANDLE_MASS * HANDLE_L ** 2 / 12
    links += link("handle", HANDLE_MASS, hc, (HANDLE_MASS * HANDLE_W ** 2 / 12 + 1e-8, hi, hi), handle_colliders(root, ind),
                  hparts)

    def joint(name, kind, b0, b1, pos, lo, hi_, axis):
        drive = "linear" if kind == "Prismatic" else "angular"
        return f"""
        def Physics{kind}Joint "{name}" (
            prepend apiSchemas = ["PhysicsDriveAPI:{drive}", "PhysxJointAPI"]
        )
        {{
            uniform token physics:axis = "{axis}"
            rel physics:body0 = </{root}/{b0}>
            rel physics:body1 = </{root}/{b1}>
            point3f physics:localPos0 = ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f})
            quatf physics:localRot0 = (1, 0, 0, 0)
            point3f physics:localPos1 = ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f})
            quatf physics:localRot1 = (1, 0, 0, 0)
            float physics:lowerLimit = {lo:.6f}
            float physics:upperLimit = {hi_:.6f}
            float physxJoint:armature = 0.00001
            float physxJoint:maxJointVelocity = 1000000
            uniform token drive:{drive}:physics:type = "force"
            float drive:{drive}:physics:targetPosition = 0
            float drive:{drive}:physics:stiffness = 0
            float drive:{drive}:physics:damping = 0
        }}
"""

    joints = joint("slide", "Prismatic", "carriage", "module", (0, 0, 0), SLIDE_LOW, SLIDE_HIGH, "Z")
    joints += joint("bar", "Revolute", "module", "bar", BAR_PIVOT, HANG_DEG - STOW_DEG, BAR_STOP_HI, "Y")
    joints += joint("handle", "Revolute", "module", "handle", HANDLE_AXIS, HANDLE_LOW, 0.0, "Y")
    all_parts = mparts + bparts + hparts
    text = f"""#usda 1.0
(
    defaultPrim = "{root}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}"
{{{uk.physics_material("PhysicsMaterial", 0.8, 0.7)}{BAR_MAT}{uk.physics_material("HandleMaterial", 0.8, 0.7)}

    def Scope "Looks"
    {{{_looks(all_parts, root)}
    }}
{links}

    def Scope "joints"
    {{{joints}
    }}
}}
"""
    return _write("laser", text)


def build() -> dict:
    return {"machine": machine(), "honeycomb": honeycomb(), "sheet": sheet(), "laser": laser()}


if __name__ == "__main__":
    print(build())
    print("bar reach G (mm):", round(1000 * bar_reach(), 3))
