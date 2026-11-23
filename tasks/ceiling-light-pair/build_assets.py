"""Writes assets/housing.usda (the light's bare steel housing, a rigid body), assets/driver.usda (a push-to-start
cordless screwdriver, a rigid body) and assets/screws.usda (both screws: one articulation fixed to the ceiling, each
screw on its own chain x, y, z (prismatic) and spin (revolute) from its pilot hole). Needs pxr only (not the
simulator): python build_assets.py"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import geometry as g
from geometry import subtract

OUT = Path(__file__).resolve().parent / "assets"
OFF = (0.0005, 0.0)

WHITE = (0.93, 0.93, 0.91)
STEEL_IN = (0.82, 0.83, 0.82)
LED_BOARD = (0.90, 0.90, 0.86)
LED = (0.98, 0.97, 0.85)
TRIM = (0.06, 0.06, 0.06)
CHANNEL = (0.75, 0.76, 0.75)
BOSCH = (0.02, 0.30, 0.52)
RUBBER = (0.10, 0.11, 0.12)
BIT = (0.55, 0.56, 0.58)
ZINC = (0.80, 0.80, 0.76)


# -- USD helpers -------------------------------------------------------------------------------------------------------
def stage_for(name: str):
    path = OUT / f"{name}.usda"
    path.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    return stage, root


def material(stage, path: str, color, roughness=0.5, metallic=0.0):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def bind(prim, mat) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim())
    UsdShade.MaterialBindingAPI(prim.GetPrim()).Bind(mat)


def offsets(prim, contact=0.0005, rest=0.0):
    prim = prim.GetPrim()
    prim.AddAppliedSchema("PhysxCollisionAPI")
    prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(contact)
    prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(rest)


def box(stage, path, lo, hi, mat=None, collide=True, off=OFF):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    c.AddTranslateOp().Set(Gf.Vec3d(*map(float, (lo + hi) / 2)))
    c.AddScaleOp().Set(Gf.Vec3f(*map(float, hi - lo)))
    if mat is not None:
        bind(c, mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        offsets(c, *off)
    return c


def cylinder(stage, path, radius, z0, z1, mat=None, collide=False, off=OFF, xy=(0.0, 0.0)):
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(float(radius))
    c.CreateHeightAttr(float(z1 - z0))
    c.CreateAxisAttr("Z")
    c.AddTranslateOp().Set(Gf.Vec3d(float(xy[0]), float(xy[1]), float((z0 + z1) / 2)))
    if mat is not None:
        bind(c, mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        offsets(c, *off)
    return c


def capsule(stage, path, radius, z0, z1, mat=None, collide=False, off=OFF):
    c = UsdGeom.Capsule.Define(stage, path)
    c.CreateRadiusAttr(float(radius))
    c.CreateHeightAttr(float(z1 - z0 - 2 * radius))
    c.CreateAxisAttr("Z")
    c.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float((z0 + z1) / 2)))
    if mat is not None:
        bind(c, mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        offsets(c, *off)
    return c


def rigid(stage, path, mass, com, inertia, pos=(0.0, 0.0, 0.0)):
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
    m = UsdPhysics.MassAPI.Apply(xf.GetPrim())
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, com)))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, inertia)))
    return xf


def rigid_root(root, mass, com, inertia):
    p = root.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(p)
    m = UsdPhysics.MassAPI.Apply(p)
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, com)))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, inertia)))


def keyhole_cuts(cx: float, cy: float, outward: float) -> list[tuple]:
    """A keyhole around the screw spot (cx, cy) in the housing's frame: the narrow slot from SLOT_IN inboard of the
    spot to the round end SLOT_TO_ROUND outboard (`outward` = +1 toward +y), the round end as its bounding square."""
    s0, s1 = sorted((cy - outward * g.SLOT_IN, cy + outward * g.SLOT_TO_ROUND))
    r0, r1 = sorted((cy + outward * (g.SLOT_TO_ROUND - g.ROUND_R), cy + outward * (g.SLOT_TO_ROUND + g.ROUND_R)))
    w = g.SLOT_W / 2
    return [(cx - w, cx + w, s0, s1), (cx - g.ROUND_R, cx + g.ROUND_R, r0, r1)]


# -- the housing ---------------------------------------------------------------------------------------------------------
def build_housing():
    """Origin: the middle of the back's top face (flush with the ceiling when mounted). x across (the near long side
    at -W/2 faces the robots), y along, z up; the open side faces down."""
    stage, root = stage_for("housing")
    r = "/housing"
    white = material(stage, f"{r}/Looks/white", WHITE, roughness=0.35, metallic=0.2)
    inner = material(stage, f"{r}/Looks/inner", STEEL_IN, roughness=0.4, metallic=0.3)
    board = material(stage, f"{r}/Looks/board", LED_BOARD, roughness=0.6)
    led = material(stage, f"{r}/Looks/led", LED, roughness=0.2)
    trim = material(stage, f"{r}/Looks/trim", TRIM, roughness=0.8)
    chan = material(stage, f"{r}/Looks/channel", CHANNEL, roughness=0.4, metallic=0.4)
    L, W, D, T, WT = g.HOUSING_L, g.HOUSING_W, g.HOUSING_D, g.SHEET_T, g.WALL_T
    m = g.HOUSING_MASS
    inertia = (m * (L * L + D * D) / 12, m * (W * W + D * D) / 12, m * (L * L + W * W) / 12)
    rigid_root(root, m, (0.0, 0.0, -0.018), inertia)
    b = r
    # the back plate, with its four keyholes cut out of the collider (and so seen from below)
    rects = [(-W / 2, W / 2, -L / 2, L / 2)]
    for ky in (+1.0, -1.0):
        for kx in g.KEY_X:
            for cut in keyhole_cuts(kx - g.HOUSING_X, ky * g.KEY_DY, ky):
                rects = subtract(rects, cut)
    for i, (x0, x1, y0, y1) in enumerate(rects):
        box(stage, f"{b}/back_{i}", (x0, y0, -T), (x1, y1, 0.0), white)
    # long walls (near at -x, far at +x) and end walls
    box(stage, f"{b}/wall_near", (-W / 2, -L / 2, -D), (-W / 2 + WT, L / 2, -T), white)
    box(stage, f"{b}/wall_far", (W / 2 - WT, -L / 2, -D), (W / 2, L / 2, -T), white)
    box(stage, f"{b}/end_pos", (-W / 2 + WT, L / 2 - WT, -D), (W / 2 - WT, L / 2, -T), white)
    box(stage, f"{b}/end_neg", (-W / 2 + WT, -L / 2, -D), (W / 2 - WT, -L / 2 + WT, -T), white)
    # inside (visual): the inner face of the back, four LED boards and the driver channel in the middle
    rects_in = [(-W / 2 + WT, W / 2 - WT, -L / 2 + WT, L / 2 - WT)]
    for ky in (+1.0, -1.0):
        for kx in g.KEY_X:
            for cut in keyhole_cuts(kx - g.HOUSING_X, ky * g.KEY_DY, ky):
                rects_in = subtract(rects_in, (cut[0] - 0.003, cut[1] + 0.003, cut[2] - 0.003, cut[3] + 0.003))
    for i, (x0, x1, y0, y1) in enumerate(rects_in):
        box(stage, f"{b}/inside_{i}", (x0, y0, -T - 0.0004), (x1, y1, -T - 0.0001), inner, collide=False)
    for i, x in enumerate((-0.095, -0.055, 0.055, 0.095)):
        box(stage, f"{b}/board_{i}", (x - 0.012, -0.46, -T - 0.004), (x + 0.012, 0.46, -T - 0.0005), board, collide=False)
        for k in range(24):
            y = -0.44 + k * (0.88 / 23)
            box(stage, f"{b}/led_{i}_{k}", (x - 0.0025, y - 0.0025, -T - 0.0055), (x + 0.0025, y + 0.0025, -T - 0.004),
                led, collide=False)
    box(stage, f"{b}/channel", (-0.025, -0.40, -T - 0.030), (0.025, 0.40, -T - 0.0005), chan, collide=False)
    # rubber edge trim (push-on U-channel, 8 mm across, 24 mm tall) on the end walls' lower edges at the four grip spots
    rubber = UsdShade.Material.Define(stage, f"{r}/Looks/rubber_phys")
    pm = UsdPhysics.MaterialAPI.Apply(rubber.GetPrim())
    pm.CreateStaticFrictionAttr(1.1)
    pm.CreateDynamicFrictionAttr(1.0)
    pm.CreateRestitutionAttr(0.0)
    for sy, end in ((1.0, "pos"), (-1.0, "neg")):
        yw = sy * (L / 2 - WT / 2)
        for name, x0, x1 in (("far", 0.06, 0.14), ("near", -0.08, 0.00)):
            c = box(stage, f"{b}/trim_{end}_{name}", (x0, yw - g.TRIM_T / 2, -D - 0.002),
                    (x1, yw + g.TRIM_T / 2, -D + g.TRIM_H - 0.002), trim)
            UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(rubber, UsdShade.Tokens.weakerThanDescendants, "physics")
    stage.GetRootLayer().Save()


# -- the screwdriver -----------------------------------------------------------------------------------------------------
def build_driver():
    """Origin: the middle of its bottom end; axis +z toward the bit."""
    stage, root = stage_for("driver")
    r = "/driver"
    blue = material(stage, f"{r}/Looks/blue", BOSCH, roughness=0.45)
    rubber = material(stage, f"{r}/Looks/rubber", RUBBER, roughness=0.9)
    steel = material(stage, f"{r}/Looks/steel", BIT, roughness=0.25, metallic=0.9)
    m, R, Lb = g.DRV_MASS, g.DRV_R, g.DRV_BODY_L
    inertia = (m * (3 * R * R + Lb * Lb) / 12, m * (3 * R * R + Lb * Lb) / 12, m * R * R / 2)
    rigid_root(root, m, (0.0, 0.0, 0.075), inertia)
    b = r
    cylinder(stage, f"{b}/grip", R, 0.0, Lb, rubber, collide=True)
    cylinder(stage, f"{b}/shell_top", R + 0.0008, 0.060, Lb, blue)
    cylinder(stage, f"{b}/end_cap", R + 0.0008, 0.0, 0.006, blue)
    box(stage, f"{b}/switch", (-0.006, R - 0.0005, 0.095), (0.006, R + 0.003, 0.125), rubber, collide=False)
    cylinder(stage, f"{b}/collar", g.DRV_COLLAR_R, Lb, Lb + g.DRV_COLLAR_L, steel, collide=True)
    h = g.DRV_BIT_HALF
    box(stage, f"{b}/bit", (-h, -h, Lb + g.DRV_COLLAR_L), (h, h, g.DRV_TIP), steel, collide=False)   # contact: task code
    stage.GetRootLayer().Save()


# -- the two screws ------------------------------------------------------------------------------------------------------
def build_screws():
    """One articulation, fixed to the world at its origin (place it at (0, 0, 0)). For each screw: base -> x -> y -> z
    (prismatic) -> spin (revolute), all frames at its pilot hole at the ceiling when the joints are at 0. The screw's
    frame origin is its tip; it points up (+z), its head below."""
    stage, root = stage_for("screws")
    r = "/screws"
    zinc = material(stage, f"{r}/Looks/zinc", ZINC, roughness=0.3, metallic=0.9)
    dark = material(stage, f"{r}/Looks/dark", (0.15, 0.15, 0.15), roughness=0.5)
    base = rigid(stage, r + "/base", 0.05, (0.0, 0.0, 0.0), (1e-5, 1e-5, 1e-5)).GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    root.GetPrim().AddAppliedSchema("PhysxArticulationAPI")
    root.GetPrim().CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    root.GetPrim().CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    root.GetPrim().CreateAttribute("physxArticulation:sleepThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
    root.GetPrim().CreateAttribute("physxArticulation:stabilizationThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
    fixed = UsdPhysics.FixedJoint.Define(stage, r + "/fix")
    fixed.CreateBody1Rel().SetTargets([base.GetPath()])
    for s, (hx, hy) in g.SCREW_XY.items():
        hole = (hx, hy, g.CEIL)
        parent = base.GetPath()
        for j, (kind, axis) in enumerate((("prismatic", "X"), ("prismatic", "Y"), ("prismatic", "Z"),
                                          ("revolute", "Z"))):
            name = ("px", "py", "pz", "spin")[j]
            last = name == "spin"
            path = f"{r}/{s}_screw" if last else f"{r}/{s}_link_{name}"
            mass = g.SCREW_MASS if last else 0.01
            inertia = (2e-7, 2e-7, 5e-8) if last else (1e-6, 1e-6, 1e-6)
            link = rigid(stage, path, mass, (0.0, 0.0, -0.009 if last else 0.0), inertia, pos=hole)
            jt = (UsdPhysics.RevoluteJoint if kind == "revolute" else UsdPhysics.PrismaticJoint).Define(
                stage, f"{r}/{s}_{name}")
            jt.CreateAxisAttr(axis)
            jt.CreateBody0Rel().SetTargets([parent])
            jt.CreateBody1Rel().SetTargets([link.GetPath()])
            pos0 = hole if parent == base.GetPath() else (0.0, 0.0, 0.0)
            jt.CreateLocalPos0Attr(Gf.Vec3f(*map(float, pos0)))
            jt.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
            jt.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
            jt.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
            if kind == "prismatic":
                lo, hi = {"px": (-0.6, 0.6), "py": (-0.6, 0.6), "pz": (-0.8, 0.0)}[name]
                jt.CreateLowerLimitAttr(lo)
                jt.CreateUpperLimitAttr(hi)
            drive = UsdPhysics.DriveAPI.Apply(jt.GetPrim(), "angular" if kind == "revolute" else "linear")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(0.0)
            parent = link.GetPath()
        sc = f"{r}/{s}_screw"
        capsule(stage, f"{sc}/shank", g.SHANK_R, -g.SCREW_L, 0.0, zinc, collide=True)
        cylinder(stage, f"{sc}/head", g.HEAD_R, -g.SCREW_L - g.HEAD_H, -g.SCREW_L, zinc, collide=True)
        h = g.DRV_BIT_HALF + 0.0003
        box(stage, f"{sc}/recess", (-h, -h, -g.SCREW_L - g.HEAD_H - 0.0002), (h, h, -g.SCREW_L - g.HEAD_H + 0.0002),
            dark, collide=False)
        box(stage, f"{sc}/mark", (h, -0.0005, -g.SCREW_L - g.HEAD_H - 0.0002),
            (g.HEAD_R - 0.0005, 0.0005, -g.SCREW_L - g.HEAD_H + 0.0002), dark, collide=False)
    stage.GetRootLayer().Save()


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    build_housing()
    build_driver()
    build_screws()
    print("wrote", sorted(p.name for p in OUT.glob("*.usda")))
