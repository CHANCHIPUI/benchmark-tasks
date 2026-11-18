"""Write the scene's USD assets.

    python build_head.py build/dl_makehuman/base.obj    # first: the head mesh and the scoring samples
    python build_assets.py                              # needs pxr (Isaac Sim's Python), not the simulator

- assets/head.usda: the manikin head, a kinematic body with a triangle-mesh collider (the scanned surface as is) and
  contact reporting. Its origin is the middle of the mouth; axes as the station's.
- assets/stick.usda: the lip balm stick, an articulation: the tube (with its neck) and the bullet on a sprung slide
  along the tube's axis (1 N/mm, 3 mm of travel). Its origin is the middle of the tube's base end; +z runs to the tip.
- assets/cap.usda: the cap, a free body: a thin-walled cup (a ring of wall pieces round a 7.4 mm bore and a closed
  end). Its origin is the middle of its open rim; +z runs to its closed end.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

import geometry as g

HERE = Path(__file__).resolve().parent
CAP_SOLID = True                   # the cap as one solid collider, filtered against the stick; task.py holds it on
CAP_ARCS = False                   # cap colliders as convex arc pieces (tried; it made the handover unreliable)
OUT = HERE / "assets"
SKIN = (0.74, 0.55, 0.45)          # the practice head's one vinyl skin tone
EYE = (0.93, 0.92, 0.90)
TUBE = (0.86, 0.12, 0.18)          # the tube and cap, glossy red-purple
BALM = (0.55, 0.08, 0.14)          # Red Dahlia, a deep tinted red


def stage_for(name: str):
    path = OUT / f"{name}.usda"
    path.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    return stage, root


def material(stage, path: str, color, roughness=0.5):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def bind(prim, mat) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim())
    UsdShade.MaterialBindingAPI(prim.GetPrim()).Bind(mat)


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


def cylinder_mesh(r, z0, z1, n=24, r_top=None):
    """A closed cylinder (or cone frustum) along z as triangles."""
    r_top = r if r_top is None else r_top
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    bot = np.c_[r * np.cos(a), r * np.sin(a), np.full(n, z0)]
    top = np.c_[r_top * np.cos(a), r_top * np.sin(a), np.full(n, z1)]
    V = np.vstack([bot, top, [[0, 0, z0], [0, 0, z1]]])
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i], [2 * n, j, i], [2 * n + 1, n + i, n + j]]
    return V, np.array(F)


def dome_mesh(r, z0, z_tip, n=24, rings=6):
    """The bullet: a cylinder from z0 rounding into a hemisphere whose tip is at z_tip."""
    zc = z_tip - r
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rows = [np.c_[r * np.cos(a), r * np.sin(a), np.full(n, z0)]]
    for k in range(rings):
        t = k / rings * np.pi / 2
        rows.append(np.c_[r * math.cos(t) * np.cos(a), r * math.cos(t) * np.sin(a), np.full(n, zc + r * math.sin(t))])
    V = np.vstack(rows + [[[0, 0, z_tip], [0, 0, z0]]])
    F = []
    for k in range(len(rows) - 1):
        for i in range(n):
            j = (i + 1) % n
            F += [[k * n + i, k * n + j, (k + 1) * n + j], [k * n + i, (k + 1) * n + j, (k + 1) * n + i]]
    tip, bot = len(V) - 2, len(V) - 1
    last = (len(rows) - 1) * n
    for i in range(n):
        j = (i + 1) % n
        F += [[last + i, last + j, tip], [bot, j, i]]
    return V, np.array(F)


def rigid(prim, mass=None, kinematic=False):
    UsdPhysics.RigidBodyAPI.Apply(prim)
    if kinematic:
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
    if mass is not None:
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)


def contact_report(prim):
    prim.AddAppliedSchema("PhysxContactReportAPI")
    prim.CreateAttribute("physxContactReport:threshold", Sdf.ValueTypeNames.Float).Set(0.0)


def build_head():
    h = np.load(HERE / "build" / "head.npz", allow_pickle=True)
    stage, root = stage_for("head")
    rigid(root.GetPrim(), kinematic=True)
    contact_report(root.GetPrim())
    skin = material(stage, "/head/Looks/skin", SKIN, 0.55)
    eye = material(stage, "/head/Looks/eye", EYE, 0.3)
    mesh(stage, "/head/surface", h["V"], h["F"], skin, collide=True, approx="none")
    for i, (V, F) in enumerate(h["eyes"]):
        mesh(stage, f"/head/eye_{i}", V, F, eye)
    stage.Save()


def build_stick():
    stage, root = stage_for("stick")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    root.GetPrim().AddAppliedSchema("PhysxArticulationAPI")
    root.GetPrim().CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    red = material(stage, "/stick/Looks/tube", TUBE, 0.3)
    balm = material(stage, "/stick/Looks/balm", BALM, 0.45)
    tube = UsdGeom.Xform.Define(stage, "/stick/tube")
    rigid(tube.GetPrim(), g.TUBE_MASS)
    contact_report(tube.GetPrim())
    V, F = cylinder_mesh(g.TUBE_R, 0.0, g.TUBE_L)
    mesh(stage, "/stick/tube/body", V, F, red, collide=True, approx="convexHull")
    V, F = cylinder_mesh(g.NECK_R, g.TUBE_L - 0.001, g.TUBE_L + g.NECK_L)
    mesh(stage, "/stick/tube/neck", V, F, red, collide=True, approx="convexHull")
    bullet = UsdGeom.Xform.Define(stage, "/stick/bullet")
    rigid(bullet.GetPrim(), g.BULLET_MASS)
    contact_report(bullet.GetPrim())
    tip = g.tip_from_base()
    V, F = dome_mesh(g.BULLET_R, g.TUBE_L + g.NECK_L - 0.006, tip)
    mesh(stage, "/stick/bullet/wax", V, F, balm, collide=True, approx="convexHull")
    j = UsdPhysics.PrismaticJoint.Define(stage, "/stick/slide")
    j.CreateBody0Rel().SetTargets([tube.GetPath()])
    j.CreateBody1Rel().SetTargets([bullet.GetPath()])
    j.CreateAxisAttr("Z")
    j.CreateLowerLimitAttr(-g.SLIDE_TRAVEL)
    j.CreateUpperLimitAttr(0.0)
    j.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(g.SLIDE_K)
    drive.CreateDampingAttr(2.0)
    drive.CreateTargetPositionAttr(0.0)
    stage.Save()


def arc_segment(r_in, r_out, a0, a1, z0, z1, k=4):
    """A closed convex piece of a thick ring: the annular sector between angles a0 and a1 (its convex hull)."""
    a = np.linspace(a0, a1, k)
    ring = [np.c_[r * np.cos(a), r * np.sin(a)] for r in (r_in, r_out)]
    xy = np.vstack(ring)
    V = np.vstack([np.c_[xy, np.full(len(xy), z0)], np.c_[xy, np.full(len(xy), z1)]])
    import itertools
    from scipy.spatial import ConvexHull
    hull = ConvexHull(V)
    return V, hull.simplices


def build_cap_arcs():
    """Alternative cap colliders: 16 convex arc pieces (inner radius CAP_BORE, CAP_WALL_COLLIDE thick)."""
    stage, root = stage_for("cap")
    rigid(root.GetPrim(), g.CAP_MASS)
    contact_report(root.GetPrim())
    red = material(stage, "/cap/Looks/cap", TUBE, 0.3)
    V, F = cylinder_mesh(g.CAP_R, 0.0, g.CAP_L)
    mesh(stage, "/cap/shell", V, F, red)
    n = 16
    for i in range(n):
        a0, a1 = 2 * math.pi * i / n - 0.01, 2 * math.pi * (i + 1) / n + 0.01
        V, F = arc_segment(g.CAP_BORE, g.CAP_BORE + g.CAP_WALL_COLLIDE, a0, a1, 0.0, g.CAP_L - g.CAP_TOP)
        mesh(stage, f"/cap/arc_{i}", V, F, None, collide=True, approx="convexHull", visible=False)
    c = UsdGeom.Cylinder.Define(stage, "/cap/end")
    c.CreateRadiusAttr(g.CAP_BORE + g.CAP_WALL_COLLIDE)
    c.CreateHeightAttr(g.CAP_TOP)
    c.CreateAxisAttr("Z")
    c.AddTranslateOp().Set(Gf.Vec3d(0, 0, g.CAP_L - g.CAP_TOP / 2))
    c.CreatePurposeAttr("guide")
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    stage.Save()


def build_cap_solid():
    """The cap as one solid collider (no thin wall to squeeze through); the task filters its collisions with the tube
    and the bullet, and holds it on the tube in code (see task.py)."""
    stage, root = stage_for("cap")
    rigid(root.GetPrim(), g.CAP_MASS)
    contact_report(root.GetPrim())
    red = material(stage, "/cap/Looks/cap", TUBE, 0.3)
    V, F = cylinder_mesh(g.CAP_R, 0.0, g.CAP_L)
    mesh(stage, "/cap/shell", V, F, red)
    V, F = cylinder_mesh(g.CAP_SOLID_R, 0.0, g.CAP_L)
    mesh(stage, "/cap/solid", V, F, None, collide=True, approx="convexHull", visible=False)
    stage.Save()


def build_cap():
    stage, root = stage_for("cap")
    rigid(root.GetPrim(), g.CAP_MASS)
    contact_report(root.GetPrim())
    red = material(stage, "/cap/Looks/cap", TUBE, 0.3)
    V, F = cylinder_mesh(g.CAP_R, 0.0, g.CAP_L)
    mesh(stage, "/cap/shell", V, F, red)                      # shown: the closed cylinder
    # colliders: wall pieces round the bore (1.6 mm thick, so the fingers can't squeeze through them), and the end
    n, t = 12, g.CAP_WALL_COLLIDE
    w = 2 * (g.CAP_BORE + t) * math.tan(math.pi / n) + 0.0004
    for i in range(n):
        a = 2 * math.pi * i / n
        c = UsdGeom.Cube.Define(stage, f"/cap/wall_{i}")
        c.CreateSizeAttr(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d((g.CAP_BORE + t / 2) * math.cos(a), (g.CAP_BORE + t / 2) * math.sin(a),
                                        g.CAP_L / 2))
        c.AddRotateZOp().Set(math.degrees(a))
        c.AddScaleOp().Set(Gf.Vec3f(t, w, g.CAP_L))
        c.CreatePurposeAttr("guide")
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    c = UsdGeom.Cylinder.Define(stage, "/cap/end")
    c.CreateRadiusAttr(g.CAP_BORE + g.CAP_WALL_COLLIDE)
    c.CreateHeightAttr(g.CAP_TOP)
    c.CreateAxisAttr("Z")
    c.AddTranslateOp().Set(Gf.Vec3d(0, 0, g.CAP_L - g.CAP_TOP / 2))
    c.CreatePurposeAttr("guide")
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    stage.Save()


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    build_head()
    build_stick()
    build_cap_solid() if CAP_SOLID else (build_cap_arcs() if CAP_ARCS else build_cap())
    print("wrote", sorted(p.name for p in OUT.glob("*.usda")))
