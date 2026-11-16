"""Write the scene's USD assets from geometry.py.

    python build_assets.py          # needs pxr (Isaac Sim's Python), not the simulator

- assets/fryer.usda: the Avantco F100 countertop fryer, a fixed body. Colliders: the body under the tank, the deck
  round the tank's opening (its inner faces are the tank's walls), the control head, the bottom grate in the tank, and
  the basket hanger (the head's rolled top front edge) and the thermostat knob. Shown only: the oil's surface, the
  control panel, the carrying handles and feet.
  Its origin is on the table at the middle of its front face; x runs back (u), y across (v).
- assets/basket.usda: the fry basket, a free rigid body. Colliders: thin shells for its floor and four walls, the
  handle (a flat coated bar running forward and up from the front wall's top) and the hook (a tab running back from the
  back wall's top and a lip down behind it). Shown: a wire mesh, the coated handle and the hook. Its origin is the
  middle of its floor's underside; x runs back toward the hook, y across, z up.
"""

from __future__ import annotations

import math
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import geometry as g

OUT = Path(__file__).resolve().parent / "assets"
STEEL = (0.62, 0.64, 0.66)
STEEL_DARK = (0.35, 0.36, 0.38)
PANEL = (0.62, 0.10, 0.10)          # the F100's red control panel
KNOB = (0.08, 0.08, 0.09)
OIL = (0.40, 0.24, 0.05)            # used frying oil, golden brown
CHROME = (0.80, 0.81, 0.83)
COATING = (0.06, 0.06, 0.07)        # the handle's black coating


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


def _orient(rot_y: float = 0.0):
    h = rot_y / 2
    return Gf.Quatf(math.cos(h), Gf.Vec3f(0.0, math.sin(h), 0.0))


def box(stage, path, size, center, mat=None, collide=True, visible=True, rot_y=0.0):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    c.AddTranslateOp().Set(Gf.Vec3d(*center))
    if rot_y:
        c.AddOrientOp().Set(_orient(rot_y))
    c.AddScaleOp().Set(Gf.Vec3f(*size))
    if mat is not None:
        bind(c, mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    if not visible:
        c.CreatePurposeAttr(UsdGeom.Tokens.guide)       # a collider that isn't rendered
    return c


def cylinder(stage, path, radius, height, center, axis, mat, collide=False, rot_y=0.0):
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(radius)
    c.CreateHeightAttr(height)
    c.CreateAxisAttr(axis)
    c.AddTranslateOp().Set(Gf.Vec3d(*center))
    if rot_y:
        c.AddOrientOp().Set(_orient(rot_y))
    bind(c, mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    return c


def capsule(stage, path, radius, height, center, axis, mat):
    c = UsdGeom.Capsule.Define(stage, path)
    c.CreateRadiusAttr(radius)
    c.CreateHeightAttr(height)
    c.CreateAxisAttr(axis)
    c.AddTranslateOp().Set(Gf.Vec3d(*center))
    bind(c, mat)
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    return c


# -- the fryer -----------------------------------------------------------------------------------------------------------
def build_fryer():
    stage, root = stage_for("fryer")
    r = "/fryer"
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    steel = material(stage, f"{r}/Looks/steel", STEEL, roughness=0.35, metallic=0.6)
    dark = material(stage, f"{r}/Looks/steel_dark", STEEL_DARK, roughness=0.5, metallic=0.4)
    panel = material(stage, f"{r}/Looks/panel", PANEL, roughness=0.5)
    knob = material(stage, f"{r}/Looks/knob", KNOB, roughness=0.4)
    oil = material(stage, f"{r}/Looks/oil", OIL, roughness=0.15)
    W, D, H = g.WIDTH, g.DEPTH, g.HEIGHT
    u0, u1 = g.VAT_U
    vh = g.VAT_HALF
    feet = 0.02
    # the cabinet under the vat, on four feet
    box(stage, f"{r}/cabinet", (D, W, g.VAT_FLOOR - feet), (D / 2, 0.0, feet + (g.VAT_FLOOR - feet) / 2), steel)
    for i, (a, b) in enumerate(((0.03, 0.03), (0.03, -0.03), (D - 0.03, 0.03), (D - 0.03, -0.03))):
        cylinder(stage, f"{r}/foot_{i}", 0.012, feet, (a, (W / 2 - 0.03) * (1 if b > 0 else -1), feet / 2), "Z", knob)
    # the deck round the vat, from the vat's floor up: its inner faces are the vat's walls
    hz = g.DECK - g.VAT_FLOOR
    zc = g.VAT_FLOOR + hz / 2
    box(stage, f"{r}/deck_front", (u0, W, hz), (u0 / 2, 0.0, zc), steel)
    box(stage, f"{r}/deck_back", (D - u1, W, hz), ((u1 + D) / 2, 0.0, zc), steel)
    side = W / 2 - vh
    for s in (-1, 1):
        box(stage, f"{r}/deck_{'lr'[s > 0]}", (u1 - u0, side, hz), ((u0 + u1) / 2, s * (vh + side / 2), zc), steel)
    # the vat's floor (dark) and the basket rack above it
    box(stage, f"{r}/vat_floor", (u1 - u0, 2 * vh, 0.002), ((u0 + u1) / 2, 0.0, g.VAT_FLOOR + 0.001), dark, collide=False)
    box(stage, f"{r}/rack", (u1 - u0 - 0.01, 2 * vh - 0.01, 0.004), ((u0 + u1) / 2, 0.0, g.RACK - 0.002), dark)
    # the control head at the back, its red panel facing forward, and the knob
    box(stage, f"{r}/head", (D - g.HEAD_U, W, g.HEAD_TOP - g.DECK), ((g.HEAD_U + D) / 2, 0.0, (g.DECK + g.HEAD_TOP) / 2),
        steel)
    box(stage, f"{r}/panel", (0.002, W * 0.97, 0.07), (g.HEAD_U - 0.001, 0.0, g.DECK + 0.05), panel, collide=False)
    cylinder(stage, f"{r}/knob", g.KNOB_R, 0.016, (g.HEAD_U - 0.008, -g.KNOB_V, g.DECK + 0.06), "X", knob, collide=True)
    # the basket hanger: the head's rolled top front edge, a round lip along it
    capsule(stage, f"{r}/hanger_lip", g.ROD_R, 2 * g.ROD_HALF, (g.ROD_U, 0.0, g.ROD_Z), "Y", steel)
    # carrying handles on the sides (shown only)
    for s_ in (-1, 1):
        box(stage, f"{r}/carry_{'lr'[s_ > 0]}", (0.06, 0.015, 0.025), (0.12, s_ * (W / 2 + 0.0075), g.DECK - 0.05), knob,
            collide=False)
    # the oil's surface (shown only) and the drain valve on the front
    box(stage, f"{r}/oil", (u1 - u0, 2 * vh, 0.002), ((u0 + u1) / 2, 0.0, g.OIL - 0.001), oil, collide=False)
    stage.GetRootLayer().Save()


# -- the basket ------------------------------------------------------------------------------------------------------------
def build_basket():
    stage, root = stage_for("basket")
    r = "/basket"
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass.CreateMassAttr(g.BASKET_MASS)
    L, W, H = g.BASKET
    t = g.WALL
    chrome = material(stage, f"{r}/Looks/chrome", CHROME, roughness=0.25, metallic=0.8)
    coat = material(stage, f"{r}/Looks/coating", COATING, roughness=0.6)
    # collision shells, not rendered
    box(stage, f"{r}/c_floor", (L, W, t), (0.0, 0.0, t / 2), visible=False)
    for s, name in ((-1, "front"), (1, "back")):
        box(stage, f"{r}/c_{name}", (t, W, H), (s * (L / 2 - t / 2), 0.0, H / 2), visible=False)
    for s in (-1, 1):
        box(stage, f"{r}/c_side_{'lr'[s > 0]}", (L, t, H), (0.0, s * (W / 2 - t / 2), H / 2), visible=False)
    # the wire mesh: uprights every 15 mm round the walls, rings every 20 mm up, a floor grid
    wr = 0.0012
    n = 0
    for s in (-1, 1):
        for k in range(int(W / 0.015) + 1):
            y = -W / 2 + k * W / int(W / 0.015)
            cylinder(stage, f"{r}/w{n}", wr, H, (s * L / 2, y, H / 2), "Z", chrome)
            n += 1
        for k in range(1, int(L / 0.015)):
            x = -L / 2 + k * L / int(L / 0.015)
            cylinder(stage, f"{r}/w{n}", wr, H, (x, s * W / 2, H / 2), "Z", chrome)
            n += 1
    for k in range(int(H / 0.02) + 1):
        z = min(H, k * 0.02 + 0.002)
        for s in (-1, 1):
            cylinder(stage, f"{r}/w{n}", wr, W, (s * L / 2, 0.0, z), "Y", chrome)
            cylinder(stage, f"{r}/w{n + 1}", wr, L, (0.0, s * W / 2, z), "X", chrome)
            n += 2
    for k in range(1, int(W / 0.02)):
        cylinder(stage, f"{r}/w{n}", wr, L, (0.0, -W / 2 + k * W / int(W / 0.02), 0.002), "X", chrome)
        n += 1
    for k in range(1, int(L / 0.02)):
        cylinder(stage, f"{r}/w{n}", wr, W, (-L / 2 + k * L / int(L / 0.02), 0.0, 0.002), "Y", chrome)
        n += 1
    # the handle: a bare wire riser up from the front wall's top, then a flat coated bar forward and up from its top
    box(stage, f"{r}/handle_stem", (g.STEM_T, g.STEM_T, g.HANDLE_STEM), (-L / 2, 0.0, H + g.HANDLE_STEM / 2), chrome)
    p = g.HANDLE_PITCH
    d = (-math.cos(p), 0.0, math.sin(p))
    base = (-L / 2, 0.0, H + g.HANDLE_STEM)
    centre = tuple(base[i] + d[i] * g.HANDLE_LEN / 2 for i in range(3))
    box(stage, f"{r}/handle", (g.HANDLE_LEN, g.HANDLE[0], g.HANDLE[1]), centre, coat, rot_y=math.pi + p)
    # the hook: a tab back from the back wall's top, then a lip down behind the rod
    tab_c = (L / 2 + g.HOOK_REACH / 2, 0.0, H - g.HOOK_T / 2)
    box(stage, f"{r}/hook_tab", (g.HOOK_REACH, g.HOOK_W, g.HOOK_T), tab_c, chrome)
    lip_c = (L / 2 + g.HOOK_REACH - g.HOOK_T / 2, 0.0, H - g.HOOK_LIP / 2)
    box(stage, f"{r}/hook_lip", (g.HOOK_T, g.HOOK_W, g.HOOK_LIP), lip_c, chrome)
    stage.GetRootLayer().Save()
    return n


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build_fryer()
    wires = build_basket()
    print("basket wires", wires)
    for p in sorted(OUT.rglob("*.usda")):
        print(p.relative_to(OUT), p.stat().st_size, "bytes")
