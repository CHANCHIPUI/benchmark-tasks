"""Write the scene's USD assets from geometry.py.

    python build_assets.py          # needs pxr (Isaac Sim's Python), not the simulator

- assets/tank.usda: the Mini-Sub Cell GT base, a fixed (kinematic) body: floor, walls, the gel stage between the two
  buffer chambers, and at each end the QuickSnap electrode assembly (black cathode at +y, red anode at -y) with its
  banana plug. Origin on the table under its middle.
- assets/guards.usda: the two electrode wire guards with their platinum wires, a separate fixed body (so contacts
  with it can be read). Same origin as the tank.
- assets/tray.usda: the UV-transparent tray, an articulation with a free base: the tray (floor, side walls with two
  comb slots each) and the comb's hold, an invisible catch over each end of the comb's bar on a stiff vertical slide
  (task.py releases it at the release force). Origin at the middle of the tray's bottom face, x along the run axis.
- assets/gel.usda: the agarose slab with its 8 well pockets, a free rigid body. Origin at the middle of its bottom face.
- assets/comb.usda: the 8-well, 1.5 mm comb, a free rigid body. Origin at the middle of its bar's bottom edge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import geometry as g

OUT = Path(__file__).resolve().parent / "assets"
CLEAR = (0.80, 0.86, 0.90)          # clear moulded base, seen against the table
STAGE = (0.86, 0.89, 0.91)
ACRYLIC = (0.72, 0.84, 0.92)        # UV-transparent acrylic tray, a faint blue cast
GEL = (0.86, 0.88, 0.84)            # 1% agarose: translucent, faintly opalescent
WELL = (0.30, 0.32, 0.34)           # the wells' floors read dark through the gel
COMB = (0.12, 0.12, 0.14)           # black moulded comb
BLACK = (0.06, 0.06, 0.07)
RED = (0.75, 0.08, 0.08)
GOLD = (0.83, 0.69, 0.30)
PLATINUM = (0.80, 0.80, 0.82)


def stage_for(name: str):
    path = OUT / f"{name}.usda"
    path.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    return stage, root


def material(stage, path: str, color, roughness=0.5, metallic=0.0, opacity=1.0):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    if opacity < 1.0:
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def physics_material(stage, path: str, static: float, dynamic: float):
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(0.0)
    mat.GetPrim().AddAppliedSchema("PhysxMaterialAPI")
    mat.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("min")
    return mat


def bind(prim, mat) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim())
    UsdShade.MaterialBindingAPI(prim.GetPrim()).Bind(mat)


def bind_physics(prim, mat) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim())
    UsdShade.MaterialBindingAPI(prim.GetPrim()).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def contact_offsets(prim, contact=0.001, rest=0.0):
    prim.AddAppliedSchema("PhysxCollisionAPI")
    prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(contact)
    prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(rest)


def box(stage, path, size, center, mat=None, collide=True, visible=True, offsets=True):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    c.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
    c.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
    if mat is not None:
        bind(c, mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        if offsets:
            contact_offsets(c.GetPrim())
    if not visible:
        c.CreatePurposeAttr(UsdGeom.Tokens.guide)
    return c


def span(lo, hi):
    """(size, centre) of an interval."""
    return hi - lo, (lo + hi) / 2


def abox(stage, path, x, y, z, mat=None, **kw):
    """A box from intervals x, y, z = (lo, hi)."""
    (sx, cx), (sy, cy), (sz, cz) = span(*x), span(*y), span(*z)
    return box(stage, path, (sx, sy, sz), (cx, cy, cz), mat, **kw)


def cylinder(stage, path, radius, height, center, axis, mat, collide=False):
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(radius)
    c.CreateHeightAttr(height)
    c.CreateAxisAttr(axis)
    c.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
    bind(c, mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    return c


def kinematic(prim):
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)


def rigid(prim, mass):
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)


# -- the Mini-Sub Cell GT base ------------------------------------------------------------------------------------------
def build_tank():
    stage, root = stage_for("tank")
    r = "/tank"
    kinematic(root.GetPrim())
    clear = material(stage, f"{r}/Looks/clear", CLEAR, roughness=0.15, opacity=0.45)
    frosted = material(stage, f"{r}/Looks/stage", STAGE, roughness=0.3, opacity=0.8)
    black = material(stage, f"{r}/Looks/black", BLACK, roughness=0.4)
    red = material(stage, f"{r}/Looks/red", RED, roughness=0.4)
    gold = material(stage, f"{r}/Looks/gold", GOLD, roughness=0.25, metallic=0.9)
    hl, hw, H, t = g.BASE_L / 2, g.BASE_W / 2, g.BASE_H, g.BASE_WALL
    iw = g.TANK_INNER_W / 2
    abox(stage, f"{r}/floor", (-hw, hw), (-hl, hl), (0.0, g.BASE_FLOOR), clear)
    for s in (-1, 1):                                   # the long side walls (near and far)
        abox(stage, f"{r}/side_{'nf'[s > 0]}", (hw - t, hw) if s > 0 else (-hw, -hw + t), (-hl, hl), (0.0, H), clear)
    for s in (-1, 1):                                   # the end walls
        y0, y1 = (hl - t, hl) if s > 0 else (-hl, -hl + t)
        abox(stage, f"{r}/end_{'pm'[s < 0]}", (-hw, hw), (y0, y1), (0.0, H), clear)
    # the gel stage: a raised block between the buffer chambers, full inner width
    sl = g.STAGE_L / 2
    abox(stage, f"{r}/stage", (-iw, iw), (-sl, sl), (g.BASE_FLOOR, g.STAGE_TOP), frosted)
    # the QuickSnap electrode assemblies: a coloured moulded block over each end wall's inside, with the banana plug
    for s, mat, n in ((1, black, "cathode"), (-1, red, "anode")):
        yo = s * (hl - t)
        abox(stage, f"{r}/{n}_assembly", (-0.020, 0.020), sorted((yo, yo - s * 0.008)), (H - 0.010, H + 0.004), mat)
        abox(stage, f"{r}/{n}_post", (0.010, 0.018), sorted((yo, yo - s * 0.006)), (g.BASE_FLOOR, H - 0.010), mat)
        cylinder(stage, f"{r}/{n}_plug", g.PLUG_R, g.PLUG_L * 0.6, (0.014, s * g.PLUG_Y, H + 0.004 + g.PLUG_L * 0.3),
                 "Z", gold)
        cylinder(stage, f"{r}/{n}_collar", 0.004, 0.006, (0.014, s * g.PLUG_Y, H + 0.007), "Z", mat)
        # a colour band along the end wall's outside (the base's colour-keyed ends)
        abox(stage, f"{r}/{n}_band", (-hw, hw), sorted((s * hl, s * (hl + 0.0008))), (H - 0.012, H - 0.002), mat,
             collide=False)
    stage.Save()


def build_guards():
    stage, root = stage_for("guards")
    r = "/guards"
    kinematic(root.GetPrim())
    clear = material(stage, f"{r}/Looks/clear", CLEAR, roughness=0.2, opacity=0.6)
    plat = material(stage, f"{r}/Looks/platinum", PLATINUM, roughness=0.2, metallic=1.0)
    iw = g.TANK_INNER_W / 2
    for s, n in ((1, "cathode"), (-1, "anode")):
        abox(stage, f"{r}/{n}_guard", (-iw, iw), sorted((s * g.GUARD_Y[0], s * g.GUARD_Y[1])),
             (g.BASE_FLOOR, g.GUARD_H), clear)
        cylinder(stage, f"{r}/{n}_wire", 0.0006, 2 * iw - 0.004, (0.0, s * g.WIRE_Y, g.WIRE_Z), "X", plat)
        cylinder(stage, f"{r}/{n}_riser", 0.0006, g.BASE_H - g.WIRE_Z - 0.010, (0.014, s * g.WIRE_Y,
                 (g.WIRE_Z + g.BASE_H - 0.010) / 2), "Z", plat)
        # the riser's collider: a 3 mm square post round the wire, so a touch on it is read like one on the guard
        abox(stage, f"{r}/{n}_riser_c", (0.0125, 0.0155), (s * g.WIRE_Y - 0.0015, s * g.WIRE_Y + 0.0015),
             (g.GUARD_H, g.BASE_H - 0.010), visible=False)
    stage.Save()


# -- the tray (an articulation: the tray and the comb's hold) -------------------------------------------------------------
def build_tray():
    stage, root = stage_for("tray")
    r = "/tray"
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    root.GetPrim().AddAppliedSchema("PhysxArticulationAPI")
    root.GetPrim().CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    acrylic = material(stage, f"{r}/Looks/acrylic", ACRYLIC, roughness=0.1, opacity=0.55)
    slick = physics_material(stage, f"{r}/Looks/slick", 0.0, 0.0)
    base = f"{r}/base"
    xf = UsdGeom.Xform.Define(stage, base)
    rigid(xf.GetPrim(), g.TRAY_MASS)
    hl, hw, ft, wt, wh = g.TRAY_L / 2, g.TRAY_W / 2, g.FLOOR_T, g.WALL_T, g.WALL_H
    abox(stage, f"{base}/floor", (-hl, hl), (-hw + wt, hw - wt), (0.0, ft), acrylic)
    sw = g.SLOT_W / 2
    for s, n in ((1, "l"), (-1, "r")):
        y = sorted((s * (hw - wt), s * hw))
        xs = [-hl, g.SLOT_X[0] - sw, g.SLOT_X[0] + sw, g.SLOT_X[1] - sw, g.SLOT_X[1] + sw, hl]
        for k in range(5):
            top = wh - g.SLOT_D if k in (1, 3) else wh          # the slots: notches in the wall's top edge
            abox(stage, f"{base}/wall_{n}{k}", (xs[k], xs[k + 1]), y, (0.0, top), acrylic)
    # the hold: an invisible catch over each end of the comb's bar, on a stiff vertical slide (task.py)
    catch = f"{r}/catch"
    cx = UsdGeom.Xform.Define(stage, catch)
    cx.AddTranslateOp().Set(Gf.Vec3d(g.WELL_X, 0.0, g.CATCH_Z))
    rigid(cx.GetPrim(), 0.0005)
    for s, n in ((1, "l"), (-1, "r")):
        c = box(stage, f"{catch}/end_{n}", g.CATCH_SIZE, (0.0, s * g.CATCH_Y, 0.0), visible=False)
        bind_physics(c, slick)
    j = UsdPhysics.PrismaticJoint.Define(stage, f"{r}/joints/hold")
    j.CreateAxisAttr("Z")
    j.CreateBody0Rel().SetTargets([Sdf.Path(base)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(catch)])
    j.CreateLocalPos0Attr(Gf.Vec3f(g.WELL_X, 0.0, g.CATCH_Z))
    j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    j.CreateLowerLimitAttr(g.CATCH_PARK - 0.002)
    j.CreateUpperLimitAttr(0.05)
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(g.CATCH_K)
    drive.CreateDampingAttr(10.0)
    drive.CreateTargetPositionAttr(0.0)
    stage.Save()


# -- the gel -------------------------------------------------------------------------------------------------------------
def build_gel():
    stage, root = stage_for("gel")
    r = "/gel"
    rigid(root.GetPrim(), g.GEL_MASS)
    gel = material(stage, f"{r}/Looks/agarose", GEL, roughness=0.35, opacity=0.8)
    well = material(stage, f"{r}/Looks/well", WELL, roughness=0.6)
    hl, hw, T = g.GEL_L / 2, g.GEL_W / 2, g.GEL_T
    px0, px1 = g.WELL_X - g.POCKET_X / 2, g.WELL_X + g.POCKET_X / 2
    abox(stage, f"{r}/slab_back", (-hl, px0), (-hw, hw), (0.0, T), gel)
    abox(stage, f"{r}/slab_front", (px1, hl), (-hw, hw), (0.0, T), gel)
    ys = sorted(g.WELL_Y)
    edges = [-hw] + [v for y in ys for v in (y - g.POCKET_Y / 2, y + g.POCKET_Y / 2)] + [hw]
    for k in range(0, len(edges), 2):                   # the gel between the pockets (and at the two sides)
        abox(stage, f"{r}/rib_{k // 2}", (px0, px1), (edges[k], edges[k + 1]), (0.0, T), gel)
    # shown only: each well's floor (the 1 mm of gel under the tooth); the collider pockets go through
    floor_z = T - g.CAST_DEPTH
    for i, y in enumerate(g.WELL_Y):
        abox(stage, f"{r}/well_{i + 1}", (g.WELL_X - g.TOOTH_T / 2, g.WELL_X + g.TOOTH_T / 2),
             (y - g.TOOTH_W / 2, y + g.TOOTH_W / 2), (0.0, floor_z), well, collide=False)
    stage.Save()


# -- the comb ------------------------------------------------------------------------------------------------------------
def build_comb():
    stage, root = stage_for("comb")
    r = "/comb"
    rigid(root.GetPrim(), g.COMB_MASS)
    comb = material(stage, f"{r}/Looks/comb", COMB, roughness=0.45)
    box(stage, f"{r}/bar", (g.BAR_T, g.BAR_L, g.BAR_H), (0.0, 0.0, g.BAR_H / 2), comb)
    for i, y in enumerate(g.WELL_Y):
        box(stage, f"{r}/tooth_{i + 1}", (g.TOOTH_T, g.TOOTH_W, g.TOOTH_L), (0.0, y, -g.TOOTH_L / 2), comb)
    stage.Save()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build_tank()
    build_guards()
    build_tray()
    build_gel()
    build_comb()
    for p in sorted(OUT.glob("*.usda")):
        print(p.name, p.stat().st_size, "bytes")
