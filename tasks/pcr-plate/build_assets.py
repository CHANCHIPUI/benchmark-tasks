"""Write the scene's USD assets from geometry.py.

    python build_assets.py          # needs pxr (Isaac Sim's Python), not the simulator

- assets/cycler.usda: the GeneAmp 9700, an articulation with a fixed base: the base unit (body, sloped keypad panel,
  the module's surround and the cover's side rails), the heated cover (a carriage on a prismatic slide along u, with
  an end stop at fully forward), the lever (a revolute joint on the cover's front top edge) and the platen inside the
  cover (a prismatic joint along h, lowered by the lever's cam: task.py sets its drive target from the lever's angle).
  Origin on the table at the middle of its front face; x runs back (u), y across (v).
- assets/block.usda: the 96-well sample block, a fixed (kinematic) body with a triangle-mesh collider: its top with 96
  conical wells, and its sides down to the surround. Origin at the middle of its top.
- assets/support.usda: the Splash-Free 96-Well Support Base, the same kind of body: a pedestal with 96 wells on a wider
  foot. Origin at the middle of its top.
- assets/plate.usda: the capped plate, a free rigid body: 96 convex well hulls, the deck, the semi-skirt and the cap
  mat (the full plate cover). Origin at the middle of the deck's top; x toward row A, y toward column 1.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

import geometry as g

OUT = Path(__file__).resolve().parent / "assets"
BEIGE = (0.84, 0.83, 0.79)          # the 9700's off-white housing
GREY = (0.55, 0.56, 0.58)
DARK = (0.16, 0.17, 0.19)
PANEL = (0.24, 0.30, 0.38)          # the keypad / display panel
ALU = (0.70, 0.71, 0.73)            # the block's anodised aluminium
PLATE = (0.90, 0.91, 0.90)          # natural polypropylene, frosted
CAPS = (0.82, 0.86, 0.90)           # optical caps, clear
SUPPORT = (0.10, 0.10, 0.11)        # the black support base
MARK = (0.05, 0.05, 0.05)


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


def bind(prim, mat) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim())
    UsdShade.MaterialBindingAPI(prim.GetPrim()).Bind(mat)


def box(stage, path, size, center, mat=None, collide=True, visible=True):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    c.AddTranslateOp().Set(Gf.Vec3d(*center))
    c.AddScaleOp().Set(Gf.Vec3f(*size))
    if mat is not None:
        bind(c, mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    if not visible:
        c.CreatePurposeAttr(UsdGeom.Tokens.guide)
    return c


def cylinder(stage, path, radius, height, center, axis, mat, collide=False):
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(radius)
    c.CreateHeightAttr(height)
    c.CreateAxisAttr(axis)
    c.AddTranslateOp().Set(Gf.Vec3d(*center))
    bind(c, mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    return c


def mesh(stage, path, V, F, mat=None, collide=False, approx="none", visible=True):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in V]))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(F)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in np.asarray(F).ravel()]))
    m.CreateSubdivisionSchemeAttr("none")
    if mat is not None:
        bind(m, mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr(approx)
    if not visible:
        m.CreatePurposeAttr("guide")
    return m


def contact_offsets(prim, contact=0.001, rest=0.0):
    prim.AddAppliedSchema("PhysxCollisionAPI")
    prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(contact)
    prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(rest)


def frustum(r0, r1, z0, z1, n=16, x=0.0, y=0.0):
    """A closed cone frustum along z (radius r0 at z0, r1 at z1) as triangles."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    bot = np.c_[x + r0 * np.cos(a), y + r0 * np.sin(a), np.full(n, z0)]
    top = np.c_[x + r1 * np.cos(a), y + r1 * np.sin(a), np.full(n, z1)]
    V = np.vstack([bot, top, [[x, y, z0], [x, y, z1]]])
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i], [2 * n, j, i], [2 * n + 1, n + i, n + j]]
    return V, np.array(F)


def well_block_mesh(size_xy, depth_body: float, n=24):
    """A block's top face (z = 0) with the 96 wells (conical, flat bottomed, g.hole_radius), its four sides down to
    z = -depth_body and a bottom. Triangles, facing out. Wells on the plate's 9 mm grid, A1 at +x +y."""
    X, Y = size_xy
    V, F = [], []

    def add(pts):
        V.extend(pts)
        return len(V) - len(pts)

    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # the depths at which the wall is sampled
    ds = np.linspace(0.0, g.HOLE_DEPTH, 7)
    cells = []
    for r in range(g.ROWS):
        for c in range(g.COLS):
            cx, cy = g.well_xy(r, c)
            cells.append((cx, cy))
    # the top face: each cell a 9 mm square round its hole, triangulated between the hole's rim and the square, sampled
    # along the same rays (n a multiple of 8, so the rays at 45 degrees hit the square's corners); margins as quads
    half = g.PITCH / 2
    for cx, cy in cells:
        ring = [add([(cx + g.hole_radius(d) * math.cos(t), cy + g.hole_radius(d) * math.sin(t), -d) for t in a])
                for d in ds]
        # square boundary points along the same rays
        sq = []
        for t in a:
            ct, st = math.cos(t), math.sin(t)
            s = half / max(abs(ct), abs(st))
            sq.append((cx + s * ct, cy + s * st, 0.0))
        sq0 = add(sq)
        for i in range(n):
            j = (i + 1) % n
            F.append([sq0 + i, sq0 + j, ring[0] + j])
            F.append([sq0 + i, ring[0] + j, ring[0] + i])
            for k in range(len(ds) - 1):     # the wall, facing into the hole
                F.append([ring[k] + i, ring[k + 1] + j, ring[k] + j])
                F.append([ring[k] + i, ring[k + 1] + i, ring[k + 1] + j])
        bottom = add([(cx, cy, -g.HOLE_DEPTH)])
        for i in range(n):
            F.append([ring[-1] + i, ring[-1] + (i + 1) % n, bottom])
    # margins round the well grid, out to the block's edge (top face, z = 0)
    gx0, gx1 = g.well_xy(g.ROWS - 1, 0)[0] - half, g.well_xy(0, 0)[0] + half
    gy0, gy1 = g.well_xy(0, g.COLS - 1)[1] - half, g.well_xy(0, 0)[1] + half
    xs = [-X / 2, gx0, gx1, X / 2]
    ys = [-Y / 2, gy0, gy1, Y / 2]
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue
            b = add([(xs[i], ys[j], 0), (xs[i + 1], ys[j], 0), (xs[i + 1], ys[j + 1], 0), (xs[i], ys[j + 1], 0)])
            F += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    # sides and bottom
    z1 = -depth_body
    c = [(-X / 2, -Y / 2), (X / 2, -Y / 2), (X / 2, Y / 2), (-X / 2, Y / 2)]
    for k in range(4):
        (x0, y0), (x1, y1) = c[k], c[(k + 1) % 4]
        b = add([(x0, y0, 0), (x1, y1, 0), (x1, y1, z1), (x0, y0, z1)])
        F += [[b, b + 2, b + 1], [b, b + 3, b + 2]]
    b = add([(x, y, z1) for x, y in c])
    F += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    return np.array(V), np.array(F)


def kinematic(prim):
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)


# -- the block and the support base ------------------------------------------------------------------------------------
def build_block():
    stage, root = stage_for("block")
    kinematic(root.GetPrim())
    alu = material(stage, "/block/Looks/alu", ALU, roughness=0.45, metallic=0.5)
    V, F = well_block_mesh(g.BLOCK_SIZE, g.BLOCK_BODY)
    m = mesh(stage, "/block/top", V, F, alu, collide=True, approx="none")
    contact_offsets(m.GetPrim())
    stage.Save()
    return len(F)


def build_support():
    stage, root = stage_for("support")
    kinematic(root.GetPrim())
    black = material(stage, "/support/Looks/black", SUPPORT, roughness=0.6)
    V, F = well_block_mesh(g.SUPPORT_TOP_SIZE, g.SUPPORT_TOP - g.SUPPORT_FOOT[2])
    m = mesh(stage, "/support/top", V, F, black, collide=True, approx="none")
    contact_offsets(m.GetPrim())
    fx, fy, fz = g.SUPPORT_FOOT
    box(stage, "/support/foot", (fx, fy, fz), (0, 0, -g.SUPPORT_TOP + fz / 2), black)
    stage.Save()


# -- the plate ------------------------------------------------------------------------------------------------------------
def build_plate():
    stage, root = stage_for("plate")
    p = root.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(p)
    UsdPhysics.MassAPI.Apply(p).CreateMassAttr(g.PLATE_MASS)
    pp = material(stage, "/plate/Looks/pp", PLATE, roughness=0.7)
    caps = material(stage, "/plate/Looks/caps", CAPS, roughness=0.3)
    mark = material(stage, "/plate/Looks/mark", MARK, roughness=0.6)
    X, Y = g.PLATE_X, g.PLATE_Y
    # colliders
    d = box(stage, "/plate/c_deck", (X, Y, g.DECK_T), (0, 0, -g.DECK_T / 2), visible=False)
    contact_offsets(d.GetPrim())
    t, h = g.SKIRT_COLLIDE, g.SKIRT_H
    for s in (-1, 1):
        c = box(stage, f"/plate/c_skirt_x{'mp'[s > 0]}", (t, Y, h), (s * (X / 2 - t / 2), 0, -h / 2), visible=False)
        contact_offsets(c.GetPrim())
        c = box(stage, f"/plate/c_skirt_y{'mp'[s > 0]}", (X - 2 * t, t, h), (0, s * (Y / 2 - t / 2), -h / 2), visible=False)
        contact_offsets(c.GetPrim())
    mx = g.A1_X + g.CAP_MARGIN
    my = g.A1_Y + g.CAP_MARGIN
    c = box(stage, "/plate/c_caps", (2 * mx, 2 * my, g.CAP_H), (0, 0, g.CAP_H / 2), visible=False)
    contact_offsets(c.GetPrim())
    for i, (x, y) in enumerate(g.WELLS):
        V, F = frustum(g.WELL_R_TIP, g.WELL_R_TOP, -g.WELL_TIP, -g.DECK_T, n=12, x=x, y=y)
        m = mesh(stage, f"/plate/c_well_{i:02d}", V, F, None, collide=True, approx="convexHull", visible=False)
        contact_offsets(m.GetPrim())
    # shown: the deck (with the cut corner at A12, back right), the skirt, the wells, the cap mat and caps
    cut = g.CUT
    outline = [(X / 2, Y / 2), (-X / 2, Y / 2), (-X / 2, -Y / 2), (X / 2 - cut, -Y / 2), (X / 2, -Y / 2 + cut)]
    V, F = prism(outline, -g.SKIRT_H, 0.0)
    mesh(stage, "/plate/deck", V, F, pp)
    for i, (x, y) in enumerate(g.WELLS):
        V, F = frustum(g.WELL_R_TIP, g.WELL_R_TOP, -g.WELL_TIP, -g.DECK_T, n=16, x=x, y=y)
        mesh(stage, f"/plate/well_{i:02d}", V, F, pp)
    box(stage, "/plate/cap_sheet", (2 * mx, 2 * my, 0.0015), (0, 0, 0.00075), caps, collide=False)
    for i, (x, y) in enumerate(g.WELLS):
        cylinder(stage, f"/plate/cap_{i:02d}", 0.0029, g.CAP_H - 0.0015, (x, y, 0.0015 + (g.CAP_H - 0.0015) / 2),
                 "Z", caps)
    # printed labels: a dark tab beside A1 on the deck's margin, row letters' strip at the left, column numbers' at the back
    x1, y1 = g.A1_X, g.A1_Y
    box(stage, "/plate/a1_mark", (0.004, 0.004, 0.0002), (x1 + 0.0075, y1 + 0.0075, 0.0001), mark, collide=False)
    for r in range(g.ROWS):
        x, _ = g.well_xy(r, 0)
        box(stage, f"/plate/row_{r}", (0.0025, 0.0015, 0.0002), (x, g.PLATE_Y / 2 - 0.004, 0.0001), mark, collide=False)
    for c in range(g.COLS):
        _, y = g.well_xy(0, c)
        box(stage, f"/plate/col_{c}", (0.0015, 0.0025, 0.0002), (g.PLATE_X / 2 - 0.004, y, 0.0001), mark,
            collide=False)
    stage.Save()


def prism(outline, z0, z1):
    """A closed prism over a convex polygon (counter-clockwise, xy)."""
    n = len(outline)
    V = [(x, y, z0) for x, y in outline] + [(x, y, z1) for x, y in outline]
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i]]
    for i in range(1, n - 1):
        F += [[0, i + 1, i], [n, n + i, n + i + 1]]
    return np.array(V), np.array(F)


# -- the cycler -----------------------------------------------------------------------------------------------------------
def body(stage, path, mass, pos=(0.0, 0.0, 0.0)):
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
    UsdPhysics.MassAPI.Apply(xf.GetPrim()).CreateMassAttr(mass)
    return xf


def joint(stage, kind, path, body0, body1, pos0, axis, limits):
    j = (UsdPhysics.RevoluteJoint if kind == "revolute" else UsdPhysics.PrismaticJoint).Define(stage, path)
    j.CreateAxisAttr(axis)
    j.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    j.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
    j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    j.CreateLowerLimitAttr(limits[0])
    j.CreateUpperLimitAttr(limits[1])
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular" if kind == "revolute" else "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(0.0)
    drive.CreateTargetPositionAttr(0.0)
    return j


def build_cycler():
    stage, root = stage_for("cycler")
    r = "/cycler"
    beige = material(stage, f"{r}/Looks/beige", BEIGE, roughness=0.55)
    grey = material(stage, f"{r}/Looks/grey", GREY, roughness=0.45, metallic=0.3)
    dark = material(stage, f"{r}/Looks/dark", DARK, roughness=0.5)
    panel = material(stage, f"{r}/Looks/panel", PANEL, roughness=0.35)
    # the fixed base
    base = f"{r}/base"
    body(stage, base, 10.0)
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    root.GetPrim().AddAppliedSchema("PhysxArticulationAPI")
    root.GetPrim().CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    UsdPhysics.FixedJoint.Define(stage, f"{r}/joints/root_joint").CreateBody1Rel().SetTargets([Sdf.Path(base)])
    W, D = g.WIDTH, g.DEPTH
    pu, (ph0, ph1) = g.PANEL_U, g.PANEL_H
    # the body, with a pocket under the block (the block's own body stands in it)
    low = g.DECK_H - g.POCKET
    box(stage, f"{base}/body", (D - pu, W, low), ((pu + D) / 2, 0, low / 2), beige)
    bu0, bu1 = g.BLOCK_U - g.BLOCK_SIZE[0] / 2, g.BLOCK_U + g.BLOCK_SIZE[0] / 2
    bv = g.BLOCK_SIZE[1] / 2
    zc = low + g.POCKET / 2
    box(stage, f"{base}/ring_f", (bu0 - pu, W, g.POCKET), ((pu + bu0) / 2, 0, zc), beige)
    box(stage, f"{base}/ring_b", (D - bu1, W, g.POCKET), ((bu1 + D) / 2, 0, zc), beige)
    for s in (-1, 1):
        box(stage, f"{base}/ring_{'lr'[s > 0]}", (bu1 - bu0, W / 2 - bv, g.POCKET),
            ((bu0 + bu1) / 2, s * (bv + (W / 2 - bv) / 2), zc), beige)
    box(stage, f"{base}/front", (pu, W, ph0), (pu / 2, 0, ph0 / 2), beige)
    # the sloped keypad panel (a wedge, convex hull) from ph0 at the front edge up to ph1 at the deck
    wedge = [(0, ph0), (pu, ph0), (pu, ph1)]
    V = [(u, s * W / 2, h) for s in (-1, 1) for u, h in wedge]
    F = [[0, 1, 2], [3, 5, 4], [0, 3, 4], [0, 4, 1], [1, 4, 5], [1, 5, 2], [0, 2, 5], [0, 5, 3]]
    mesh(stage, f"{base}/panel", V, F, panel, collide=True, approx="convexHull")
    box(stage, f"{base}/display", (0.004, 0.10, 0.035), (pu * 0.62, 0.04, (ph0 + ph1) / 2 + 0.012), dark,
        collide=False)
    # the module's surround (dark top) and the cover's side rails
    box(stage, f"{base}/surround", (D - pu, W, 0.002), ((pu + D) / 2, 0, g.DECK_H - 0.001), dark, collide=False)
    for s in (-1, 1):
        box(stage, f"{base}/rail_{'lr'[s > 0]}", (D - pu, g.RAIL_V[1] - g.RAIL_V[0], g.RAIL_H - g.DECK_H),
            ((pu + D) / 2, s * sum(g.RAIL_V) / 2, (g.DECK_H + g.RAIL_H) / 2), beige)
    # the heated cover: a carriage, its origin at its front face's middle, at its underside
    cover = f"{r}/cover"
    body(stage, cover, 0.8, (g.COVER_BACK_U, 0, g.COVER_UNDER))
    L, Wc, Hc, t = g.COVER_L, g.COVER_W, g.COVER_TOP - g.COVER_UNDER, g.COVER_WALL
    box(stage, f"{cover}/top", (L, Wc, t), (L / 2, 0, Hc - t / 2), grey)
    box(stage, f"{cover}/front", (t, Wc, Hc - t), (t / 2, 0, (Hc - t) / 2), grey)
    box(stage, f"{cover}/back", (t, Wc, Hc - t), (L - t / 2, 0, (Hc - t) / 2), grey)
    for s in (-1, 1):
        box(stage, f"{cover}/side_{'lr'[s > 0]}", (L - 2 * t, t, Hc - t), (L / 2, s * (Wc / 2 - t / 2), (Hc - t) / 2),
            grey)
    box(stage, f"{cover}/badge", (0.002, 0.06, 0.012), (-0.001, 0.0, Hc * 0.45), dark, collide=False)
    joint(stage, "prismatic", f"{r}/joints/slide", base, cover, (g.COVER_BACK_U, 0, g.COVER_UNDER), "X",
          (-g.COVER_TRAVEL, 0.0))
    # the platen inside the cover, raised: its underside level with the cover's
    platen = f"{r}/platen"
    px, py, pz = g.PLATEN_SIZE
    body(stage, platen, 0.3, (g.COVER_BACK_U + L / 2, 0, g.COVER_UNDER))
    box(stage, f"{platen}/plate", (px, py, pz), (0, 0, pz / 2), dark)
    joint(stage, "prismatic", f"{r}/joints/platen", cover, platen, (L / 2, 0, 0), "Z",
          (-g.PLATEN_TRAVEL - 0.003, 0.0))
    # the lever: two side arms up from the pivot on the cover's front top edge, the handle bar across their tops
    lever = f"{r}/lever"
    body(stage, lever, 0.12, (g.COVER_BACK_U, 0, g.COVER_TOP))
    for s in (-1, 1):
        box(stage, f"{lever}/arm_{'lr'[s > 0]}", (0.010, 0.008, g.LEVER_R + 0.008),
            (0, s * g.LEVER_ARM_V, (g.LEVER_R + 0.008) / 2), grey)
    cylinder(stage, f"{lever}/bar", g.LEVER_BAR_R, g.LEVER_BAR_L, (0, 0, g.LEVER_R), "Y", dark, collide=True)
    joint(stage, "revolute", f"{r}/joints/lever", cover, lever, (0, 0, g.COVER_TOP - g.COVER_UNDER), "Y",
          (math.degrees(g.LEVER_DOWN), 0.0))
    stage.Save()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print("block triangles", build_block())
    build_support()
    build_plate()
    build_cycler()
    for p in sorted(OUT.glob("*.usda")):
        print(p.name, p.stat().st_size, "bytes")
