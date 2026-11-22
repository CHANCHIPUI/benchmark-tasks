"""Write the scene's USD assets and textures from geometry.py.

    python build_assets.py          # needs pxr, PIL, scipy and the DejaVu fonts (Isaac Sim's Python), not the simulator

- assets/alarm.usda: the smoke alarm, one floating articulation, built face up (origin on its axis at its back's
  middle, z toward the front cover):
  - body (root): the flat back, the side band and the front cover; a slot in the body at the rim holds the battery
    compartment, and the front cover over the slot's hinge end (the shroud) stays over the hinge end of the battery
    when the compartment is open; over the slot's free end the cover has a finger recess for the latch;
  - tray (revolute about z at the rim, the compartment's far end): the swing-out battery compartment, its pocket open
    toward the front: its floor, low walls, the outer skirt that follows the rim, and the contact block at its free
    end with two open-topped slots (the + slot too narrow for the battery's - snap);
  - tab (prismatic along z on the tray): the latch tab on the free end, in the recess.
- assets/battery_old.usda, assets/battery_new.usda: a 9 V battery, one rigid body each: the case and its two snaps
  (the old one with a red tape band round its bottom end); origin at the case's centre, +x toward the snaps.
- assets/card.usda: the torn-open retail card, kinematic; origin at its centre on the table.
- assets/bracket.usda: the twist-on mounting bracket, kinematic; origin at its centre on its face.
- assets/textures/*.png: drawn here (the alarm's front and back, the battery label, the card print).
"""

from __future__ import annotations

import math
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import geometry as g

OUT = Path(__file__).resolve().parent / "assets"
TEX = OUT / "textures"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
WHITE = (0.90, 0.90, 0.88)
OFFWHITE = (0.84, 0.84, 0.81)
GREY = (0.45, 0.46, 0.48)
DARK = (0.12, 0.12, 0.13)
STEEL = (0.75, 0.75, 0.77)
RED = (0.80, 0.08, 0.06)
OFF = (0.0005, 0.0)                 # contact and rest offsets of the small parts' colliders


# -- textures ------------------------------------------------------------------------------------------------------------
def back_label(path: Path) -> None:
    """The alarm's back, seen from above once it is turned face down (image up = -x of the alarm frame, image left =
    +y), across 2 R: a moulded ring and the printed back label (drawn here, no brand artwork)."""
    n = 1024
    im = Image.new("RGB", (n, n), (226, 226, 220))
    d = ImageDraw.Draw(im)
    c = n / 2
    ppm = n / (2 * g.ALARM_R)
    for r, w in ((0.050, 5), (0.046, 2)):
        rr = r * ppm
        d.ellipse([c - rr, c - rr, c + rr, c + rr], outline=(190, 190, 184), width=w)
    big = ImageFont.truetype(FONT_B, 54)
    small = ImageFont.truetype(FONT, 30)
    d.rectangle([c - 300, c - 170, c + 300, c + 130], fill=(240, 240, 236))
    d.text((c, c - 110), "SMOKE ALARM", font=big, fill=(60, 60, 64), anchor="mm")
    d.text((c, c - 40), "Battery powered  9V", font=small, fill=(80, 80, 84), anchor="mm")
    d.text((c, c + 5), "Test weekly", font=small, fill=(80, 80, 84), anchor="mm")
    d.text((c, c + 60), "Install battery before mounting", font=small, fill=(80, 80, 84), anchor="mm")
    im.save(path)


def front_print(path: Path) -> None:
    """The front cover's flat middle, seen from above with the alarm face up (image up = +x, image left = +y), across
    2 FLAT_R: the words SMOKE ALARM by a pin-hole LED, a vent grille and the Test/Silence button toward the robot.
    Drawn here."""
    n = 1024
    im = Image.new("RGB", (n, n), tuple(int(255 * v) for v in WHITE))
    d = ImageDraw.Draw(im)
    c = n / 2
    ppm = n / (2 * g.FLAT_R)
    small = ImageFont.truetype(FONT_B, 34)
    d.text((c, c - 0.033 * ppm), "SMOKE ALARM", font=small, fill=(90, 90, 94), anchor="mm")
    d.ellipse([c - 9, c - 0.025 * ppm - 9, c + 9, c - 0.025 * ppm + 9], fill=(30, 30, 30))
    for k in range(5):                                   # vent slots
        y = c + 0.004 * ppm + (k - 2) * 26
        d.rounded_rectangle([c + 0.010 * ppm, y - 7, c + 0.030 * ppm, y + 7], radius=7, fill=(70, 70, 74))
    bx, by, br = c - 0.012 * ppm, c + 0.034 * ppm, 0.009 * ppm      # the Test/Silence button
    d.ellipse([bx - br, by - br, bx + br, by + br], fill=(222, 222, 216), outline=(150, 150, 146), width=5)
    t = ImageFont.truetype(FONT, 22)
    d.text((bx, by - 12), "TEST", font=t, fill=(110, 110, 114), anchor="mm")
    d.text((bx, by + 14), "SILENCE", font=t, fill=(110, 110, 114), anchor="mm")
    im.save(path)


def battery_label(path: Path) -> None:
    """A 9 V label for the battery's broad faces (image left = the snaps' end, image up = +y of the battery): a dark
    case, a copper band at the snap end, "9V" and the + and - marks beside the snaps they name. Drawn here."""
    w, h = 918, 530
    im = Image.new("RGB", (w, h), (28, 30, 34))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 250, h], fill=(176, 104, 46))
    big = ImageFont.truetype(FONT_B, 190)
    mid = ImageFont.truetype(FONT_B, 70)
    small = ImageFont.truetype(FONT_B, 46)
    d.text((590, 250), "9V", font=big, fill=(236, 236, 236), anchor="mm")
    d.text((590, 420), "ALKALINE", font=small, fill=(200, 200, 205), anchor="mm")
    # + toward +y (image up), - toward -y (image down), at the snap end
    d.text((125, 130), "+", font=mid, fill=(20, 20, 20), anchor="mm")
    d.text((125, 400), "−", font=mid, fill=(20, 20, 20), anchor="mm")
    im.save(path)


def card_print(path: Path) -> None:
    """The torn-open card: printed cardboard with the blister's outline, seen from above (image up = +x)."""
    w, h = 600, 400          # image x along the card's y (150 mm), image y along its x (100 mm)
    im = Image.new("RGB", (w, h), (188, 160, 118))
    d = ImageDraw.Draw(im)
    d.rectangle([12, 12, w - 12, h - 12], outline=(120, 92, 60), width=6)
    f = ImageFont.truetype(FONT_B, 40)
    d.text((w / 2, 46), "9V  1 PACK", font=f, fill=(70, 50, 30), anchor="mm")
    d.rectangle([w / 2 - 70, 110, w / 2 + 70, 330], outline=(150, 120, 84), width=4)   # where the blister was
    im.save(path)


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


def material(stage, path: str, color, roughness=0.5, texture: str | None = None, metallic=0.0):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    if texture:
        st = UsdShade.Shader.Define(stage, f"{path}/st")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tx = UsdShade.Shader.Define(stage, f"{path}/texture")
        tx.CreateIdAttr("UsdUVTexture")
        tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
        tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        # a fallback colour of its own, so that no renderer cache can mistake this material for another task's
        fb = 0.5 + (zlib.crc32(texture.encode()) % 997) / 4000.0
        tx.CreateInput("fallback", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(fb, fb, fb, 1.0))
        tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
        tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    else:
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
    else:
        c.CreatePurposeAttr(UsdGeom.Tokens.default_)
    return c


def cylinder(stage, path, radius, center, length, axis, mat=None, collide=False, off=OFF):
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(float(radius))
    c.CreateHeightAttr(float(length))
    c.CreateAxisAttr(axis)
    c.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
    if mat is not None:
        bind(c, mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        offsets(c, *off)
    return c


def mesh(stage, path, V, F, mat=None, collide=False, off=OFF, uv=None, smooth=False):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in V])
    m.CreateFaceVertexCountsAttr([len(f) for f in F])
    m.CreateFaceVertexIndicesAttr([int(i) for f in F for i in f])
    m.CreateSubdivisionSchemeAttr("catmullClark" if smooth else "none")
    if uv is not None:
        UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                             UsdGeom.Tokens.vertex).Set([Gf.Vec2f(*map(float, t)) for t in uv])
    m.CreateDoubleSidedAttr(True)
    if mat is not None:
        bind(m, mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("convexHull")
        offsets(m, *off)
    return m


def prism(stage, path, poly, z0, z1, mat=None, collide=True, off=OFF):
    """A vertical prism over a convex polygon (counterclockwise from above)."""
    n = len(poly)
    V = [(x, y, z0) for x, y in poly] + [(x, y, z1) for x, y in poly]
    F = [list(range(n))[::-1], [n + k for k in range(n)]] + [[k, (k + 1) % n, n + (k + 1) % n, n + k] for k in range(n)]
    return mesh(stage, path, V, F, mat, collide, off)


def body(stage, path, mass, pos=(0.0, 0.0, 0.0), com=None, inertia=None):
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
    m = UsdPhysics.MassAPI.Apply(xf.GetPrim())
    m.CreateMassAttr(float(mass))
    if com is not None:
        m.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, com)))
    if inertia is not None:
        m.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, inertia)))
    return xf


def joint(stage, kind, path, body0, body1, pos0, axis, limits=None):
    j = (UsdPhysics.RevoluteJoint if kind == "revolute" else UsdPhysics.PrismaticJoint).Define(stage, path)
    j.CreateAxisAttr(axis)
    j.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, pos0)))
    j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    if limits is not None:
        j.CreateLowerLimitAttr(float(limits[0]))
        j.CreateUpperLimitAttr(float(limits[1]))
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular" if kind == "revolute" else "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(0.0)
    drive.CreateTargetPositionAttr(0.0)
    return j


def revolve(profile, segs=96):
    """A surface of revolution about z from (r, z) points; returns V, F."""
    V, F = [], []
    m = len(profile)
    for k in range(segs):
        a = 2 * math.pi * k / segs
        for r, z in profile:
            V.append((r * math.cos(a), r * math.sin(a), z))
    for k in range(segs):
        k1 = (k + 1) % segs
        for i in range(m - 1):
            F.append([k * m + i, k1 * m + i, k1 * m + i + 1, k * m + i + 1])
    return V, F


def disc_poly(r, a0, a1, n=48):
    return [(r * math.cos(a), r * math.sin(a)) for a in np.linspace(a0, a1, n)]


# -- the alarm ---------------------------------------------------------------------------------------------------------
def _dome_piece(poly, z0, z_floor=None, n_in=7):
    """Points of a convex body over the convex footprint ``poly`` (x, y): from z0 up to the front cover (cover_z), or
    from ``z_floor`` (a flat underside) when given. Returns the convex hull's vertices and triangles (a collider)."""
    from scipy.spatial import ConvexHull

    P = np.asarray(poly, float)
    lo = P.min(axis=0)
    hi = P.max(axis=0)
    pts = [(x, y) for x, y in P]
    for x in np.linspace(lo[0], hi[0], n_in):
        for y in np.linspace(lo[1], hi[1], n_in):
            if _inside(P, (x, y)):
                pts.append((x, y))
    V = [(x, y, z0 if z_floor is None else z_floor) for x, y in P]
    V += [(x, y, g.cover_z(math.hypot(x, y))) for x, y in pts]
    V = np.asarray(V)
    h = ConvexHull(V)
    used = sorted(set(h.simplices.ravel()))
    idx = {k: i for i, k in enumerate(used)}
    F = []
    c = V[used].mean(axis=0)
    for s in h.simplices:
        a, b, d = V[s[0]], V[s[1]], V[s[2]]
        n = np.cross(b - a, d - a)
        tri = [idx[s[0]], idx[s[1]], idx[s[2]]]
        F.append(tri if n @ (a - c) > 0 else tri[::-1])
    return V[used], F


def _inside(P, q) -> bool:
    """q strictly inside the convex polygon P (either winding)."""
    s = None
    for k in range(len(P)):
        a, b = P[k], P[(k + 1) % len(P)]
        cr = (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0])
        if abs(cr) < 1e-12:
            return False
        if s is None:
            s = cr > 0
        elif (cr > 0) != s:
            return False
    return True


def _in_recess(x, y) -> bool:
    return g.NOTCH_X[0] < x < g.REC_X and y < g.NOTCH_Y


def _in_slot(x, y) -> bool:
    return g.NOTCH_X[0] < x < g.NOTCH_X[1] and y < g.NOTCH_Y


def build_alarm():
    stage, root = stage_for("alarm")
    r = "/alarm"
    white = material(stage, f"{r}/Looks/white", WHITE, roughness=0.45)
    offwhite = material(stage, f"{r}/Looks/offwhite", OFFWHITE, roughness=0.5)
    grey = material(stage, f"{r}/Looks/grey", GREY, roughness=0.6)
    dark = material(stage, f"{r}/Looks/dark", DARK, roughness=0.7)
    steel = material(stage, f"{r}/Looks/steel", STEEL, roughness=0.3, metallic=0.8)
    label = material(stage, f"{r}/Looks/label", (1, 1, 1), roughness=0.5, texture="./textures/sa_alarm_back.png")
    face = material(stage, f"{r}/Looks/face", (1, 1, 1), roughness=0.45, texture="./textures/sa_alarm_front.png")
    markm = material(stage, f"{r}/Looks/mark", (0.97, 0.97, 0.97), roughness=0.4)

    # body: root of a floating articulation
    b = f"{r}/body"
    R, H = g.ALARM_R, g.ALARM_H
    Izz = 0.5 * g.BODY_MASS * R * R
    Ixx = g.BODY_MASS * (3 * R * R + H * H) / 12
    prim = body(stage, b, g.BODY_MASS, com=(0.0, 0.0, g.BODY_COM_Z), inertia=(Ixx, Ixx, Izz)).GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    prim.AddAppliedSchema("PhysxArticulationAPI")
    prim.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    prim.CreateAttribute("physxArticulation:sleepThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
    prim.CreateAttribute("physxArticulation:stabilizationThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
    off = (0.001, 0.0)
    # colliders (invisible): the back plate, three pieces up to the front cover around the compartment's slot, and
    # the shroud (the cover over the slot's hinge end, above the battery)
    disc = disc_poly(R, 0.0, 2 * math.pi, 49)[:-1]
    prism(stage, f"{b}/c_back", disc, 0.0, g.BACK_T, None, off=off).CreatePurposeAttr(UsdGeom.Tokens.guide)
    ny, (nx0, nx1) = g.NOTCH_Y, g.NOTCH_X
    a0 = math.asin(ny / R)
    polyA = [(math.sqrt(R * R - ny * ny), ny)] + disc_poly(R, a0, math.pi - a0)[1:-1] + [(-math.sqrt(R * R - ny * ny), ny)]
    aB0 = -math.acos(nx1 / R)
    polyB = [(nx1, -math.sqrt(R * R - nx1 * nx1))] + disc_poly(R, aB0, a0, 12)[1:-1] + [(math.sqrt(R * R - ny * ny), ny), (nx1, ny)]
    aC1 = 2 * math.pi - math.acos(nx0 / R)
    polyC = [(-math.sqrt(R * R - ny * ny), ny)] + disc_poly(R, math.pi - a0, aC1, 12)[1:-1] + [(nx0, -math.sqrt(R * R - nx0 * nx0)), (nx0, ny)]
    rx0 = g.REC_X
    aS0, aS1 = -math.acos(nx1 / R), -math.acos(rx0 / R)
    polyS = [(rx0, ny)] + disc_poly(R, aS1, aS0, 14) + [(nx1, ny)]
    for name, poly, zf in (("c_a", polyA, None), ("c_b", polyB, None), ("c_c", polyC, None), ("c_shroud", polyS, g.SHROUD_Z)):
        V, F = _dome_piece(poly, g.BACK_T, zf)
        m = mesh(stage, f"{b}/{name}", V, F, None, collide=True, off=off)
        m.CreatePurposeAttr(UsdGeom.Tokens.guide)
        # the hands' side grips hold the band with the fingertips' flat side: a contact patch, so it resists
        # twisting about the contact normal (PhysX torsional friction)
        m.GetPrim().CreateAttribute("physxCollision:torsionalPatchRadius", Sdf.ValueTypeNames.Float).Set(g.TORSION_R)
        m.GetPrim().CreateAttribute("physxCollision:minTorsionalPatchRadius", Sdf.ValueTypeNames.Float).Set(0.8 * g.TORSION_R)
    # what is seen: the front cover (open over the latch's recess), its printed middle, the side band (open over the
    # compartment's slot below the shroud), the back with its label, and the slot's walls, floor and the shroud's
    # underside
    nr, na = 36, 144
    rs = np.linspace(0.0, R, nr)
    As = np.linspace(0.0, 2 * math.pi, na + 1)
    V, F = [], []
    for a in As:
        for rr in rs:
            V.append((rr * math.cos(a), rr * math.sin(a), g.cover_z(rr)))
    for i in range(na):
        for j in range(nr - 1):
            k0, k1 = i * nr + j, (i + 1) * nr + j
            xm, ym = np.mean([V[k0][:2], V[k1 + 1][:2]], axis=0)
            if _in_recess(xm, ym):
                continue
            F.append([k0, k0 + 1, k1 + 1, k1])
    mesh(stage, f"{b}/front", V, F, white)
    n = 64
    Vf = [(g.FLAT_R * 0.98 * math.cos(a), g.FLAT_R * 0.98 * math.sin(a), H + 0.0002) for a in np.linspace(0, 2 * math.pi, n + 1)[:-1]]
    uv = [(0.5 - y / (2 * g.FLAT_R), 0.5 + x / (2 * g.FLAT_R)) for x, y, _ in Vf]
    mesh(stage, f"{b}/front_print", Vf, [list(range(n))], face, uv=uv)
    zr = g.cover_z(R)
    V, F = [], []
    for k, a in enumerate(As):
        x, y = R * math.cos(a), R * math.sin(a)
        z0 = g.SHROUD_Z if _in_slot(x, y + 0.001) else 0.0
        V += [(x, y, z0), (x, y, zr)]
    for k in range(na):
        F.append([2 * k, 2 * k + 2, 2 * k + 3, 2 * k + 1])
    mesh(stage, f"{b}/band", V, F, white)
    # band below the slot: the back plate's edge
    V, F = [], []
    for k, a in enumerate(As):
        V += [(R * math.cos(a), R * math.sin(a), 0.0), (R * math.cos(a), R * math.sin(a), g.BACK_T)]
    for k in range(na):
        F.append([2 * k, 2 * k + 2, 2 * k + 3, 2 * k + 1])
    mesh(stage, f"{b}/band_low", V, F, white)
    Vb = [(x, y, -0.0002) for x, y in disc]
    uv = [(0.5 - y / (2 * R), 0.5 - x / (2 * R)) for x, y in disc]     # reads right once the alarm is turned over
    mesh(stage, f"{b}/back_label", Vb, [list(range(len(disc)))[::-1]], label, uv=uv)
    # the slot: its floor (the back plate's top), inner wall, end walls, the shroud's underside and its edge
    xl = math.sqrt(R * R - nx0 * nx0)
    xr = math.sqrt(R * R - nx1 * nx1)
    mesh(stage, f"{b}/slot_floor", [(nx0, -xl, g.BACK_T), (nx1, -xr, g.BACK_T), (nx1, ny, g.BACK_T), (nx0, ny, g.BACK_T)],
         [[0, 1, 2, 3]], grey)
    zi = g.cover_z(abs(ny) if abs(ny) > 0 else 0.0)
    mesh(stage, f"{b}/slot_wall_in", [(nx0, ny, g.BACK_T), (nx1, ny, g.BACK_T), (nx1, ny, g.SHROUD_Z), (rx0, ny, g.SHROUD_Z),
                                      (rx0, ny, g.cover_z(math.hypot(rx0, ny))), (nx0, ny, g.cover_z(math.hypot(nx0, ny)))],
         [[0, 1, 2, 3, 4, 5]], offwhite)
    _ = zi
    mesh(stage, f"{b}/slot_end_far", [(nx1, ny, g.BACK_T), (nx1, -xr, g.BACK_T), (nx1, -xr, g.SHROUD_Z), (nx1, ny, g.SHROUD_Z)],
         [[0, 1, 2, 3]], offwhite)
    ze0 = g.cover_z(math.hypot(nx0, ny))
    mesh(stage, f"{b}/slot_end_near", [(nx0, -xl, g.BACK_T), (nx0, ny, g.BACK_T), (nx0, ny, ze0), (nx0, -xl, g.cover_z(R))],
         [[0, 1, 2, 3]], offwhite)
    xs = math.sqrt(R * R - rx0 * rx0)
    mesh(stage, f"{b}/shroud_under", [(rx0, -xs, g.SHROUD_Z), (nx1, -xr, g.SHROUD_Z), (nx1, ny, g.SHROUD_Z), (rx0, ny, g.SHROUD_Z)],
         [[0, 1, 2, 3]], grey)
    ys = np.linspace(-xs, ny, 12)
    Ve = [(rx0, y, g.SHROUD_Z) for y in ys] + [(rx0, y, g.cover_z(math.hypot(rx0, y))) for y in ys[::-1]]
    mesh(stage, f"{b}/shroud_edge", Ve, [list(range(len(Ve)))], offwhite)

    # the battery compartment (tray), origin on the hinge axis; its pocket opens toward the front (+z)
    t = f"{r}/tray"
    hx, hy = g.HINGE
    body(stage, t, g.TRAY_MASS, pos=(hx, hy, 0.0), com=(-g.TRAY_L / 2, g.POCKET_YC, g.FLOOR_Z0 + 0.005),
         inertia=(2e-6, 6e-6, 7e-6))
    x0, x1 = -g.TRAY_L, 0.0
    y0, y1 = g.POCKET_Y0, g.POCKET_Y1
    z0, zf, zw = g.FLOOR_Z0, g.FLOOR_TOP, g.WALL_TOP
    box(stage, f"{t}/floor", (x0, y0 - 0.0005, z0), (x1, y1 + g.WALL_T, zf), offwhite)
    # the side walls, their top 2.5 mm leaning out 1.5 mm (a lead-in)
    li, lh = g.LEAD_IN
    box(stage, f"{t}/wall_in", (x0, y1, zf), (x1, y1 + g.WALL_T, zw - lh), offwhite)
    _ramp(stage, f"{t}/lead_in", x0, x1, y1, +1, zw - lh, zw, li, g.WALL_T, offwhite)
    _ramp(stage, f"{t}/lead_out", x0, x1, y0, -1, zw - lh, zw, li, 0.004, white)
    box(stage, f"{t}/wall_near", (g.NEAR_WALL[0], y0, zf), (g.NEAR_WALL[1], y1, zw), offwhite)
    # the outer skirt (the door's outer wall): the rim's arc from the hinge to the free end; full height (up to the
    # shroud) over its hinge half, cut down to the walls' height over its free half, where a hand pinches the battery
    arc = g.skirt_outline(16)                                    # from the free end to the hinge
    xcut = -0.024
    arc_lo = [p for p in arc if p[0] <= xcut + 1e-9] + [(xcut, g.rim_y(g.HINGE[0] + xcut) - g.HINGE[1])]
    arc_hi = [(xcut, g.rim_y(g.HINGE[0] + xcut) - g.HINGE[1])] + [p for p in arc if p[0] > xcut]
    prism(stage, f"{t}/skirt_lo", _ccw(arc_lo + [(xcut, y0), (x0, y0)]), z0, zw - g.LEAD_IN[1], white)
    prism(stage, f"{t}/skirt_hi", _ccw(arc_hi + [(x1, y0), (xcut, y0)]), z0, g.SHROUD_Z - 0.0008, white)
    # the contact block at the free end: three pieces leaving the two slots open on top, and its back
    bx0, bx1 = g.BLOCK_X
    sx = g.BLOCK_FACE - g.SLOT_DEPTH
    ps0, ps1 = g.POS_SLOT_Y - g.POS_SLOT / 2, g.POS_SLOT_Y + g.POS_SLOT / 2
    ns0, ns1 = g.NEG_SLOT_Y - g.NEG_SLOT / 2, g.NEG_SLOT_Y + g.NEG_SLOT / 2
    box(stage, f"{t}/block_back", (x0, y0, zf), (sx, y1, g.BLOCK_TOP), grey)
    box(stage, f"{t}/block_out", (sx, y0, zf), (bx1, ps0, g.BLOCK_TOP), grey)
    box(stage, f"{t}/block_mid", (sx, ps1, zf), (bx1, ns0, g.BLOCK_TOP), grey)
    box(stage, f"{t}/block_in", (sx, ns1, zf), (bx1, y1, g.BLOCK_TOP), grey)
    # the contacts at the slots' backs (visual) and the + / - marks on the block's top (visual)
    zc = zf + g.BATT_T / 2
    box(stage, f"{t}/contact_pos", (sx, ps0 + 0.001, zc - 0.003), (sx + 0.0003, ps1 - 0.001, zc + 0.003), steel, collide=False)
    box(stage, f"{t}/contact_neg", (sx, ns0 + 0.001, zc - 0.004), (sx + 0.0003, ns1 - 0.001, zc + 0.004), steel, collide=False)
    zt = g.BLOCK_TOP + 0.0002
    mx = (x0 + 0.004 + sx) / 2
    cy = (ps0 + ps1) / 2
    box(stage, f"{t}/mark_pos_b", (mx - 0.0004, cy - 0.0022, zt), (mx + 0.0004, cy + 0.0022, zt + 0.0003), markm, collide=False)
    box(stage, f"{t}/mark_pos_c", (mx - 0.0022, cy - 0.0004, zt), (mx + 0.0022, cy + 0.0004, zt + 0.0003), markm, collide=False)
    cy = (ns0 + ns1) / 2
    box(stage, f"{t}/mark_neg", (mx - 0.0022, cy - 0.0004, zt), (mx + 0.0022, cy + 0.0004, zt + 0.0003), markm, collide=False)
    _ = dark

    # the latch tab, on the free end, in the front cover's finger recess
    k = f"{r}/tab"
    tcx, tcy = (g.TAB_X[0] + g.TAB_X[1]) / 2, (g.TAB_Y[0] + g.TAB_Y[1]) / 2
    body(stage, k, g.TAB_MASS, pos=(hx + tcx, hy + tcy, 0.0), com=(0.0, 0.0, (g.TAB_Z[0] + g.TAB_Z[1]) / 2),
         inertia=(5e-8, 5e-8, 5e-8))
    hw, hd = (g.TAB_X[1] - g.TAB_X[0]) / 2, (g.TAB_Y[1] - g.TAB_Y[0]) / 2
    box(stage, f"{k}/tab", (-hw, -hd, g.TAB_Z[0]), (hw, hd, g.TAB_Z[1]), grey)
    for j in range(3):          # ridges on its top (visual)
        y = -hd + (j + 1) * 2 * hd / 4
        box(stage, f"{k}/ridge_{j}", (-hw + 0.0005, y - 0.0005, g.TAB_Z[1]), (hw - 0.0005, y + 0.0005, g.TAB_Z[1] + 0.0004),
            dark, collide=False)

    joint(stage, "revolute", f"{r}/joints/hinge", b, t, (hx, hy, 0.0), "Z", (0.0, math.degrees(g.SWING)))
    joint(stage, "prismatic", f"{r}/joints/tab", t, k, (tcx, tcy, 0.0), "Z", (-g.TAB_TRAVEL, 0.0))
    stage.Save()


def _ramp(stage, path, x0, x1, y_face, out, z0, z1, lean, thick, mat):
    """A wall's top as a ramp along x: its pocket face at y_face at z0, leaning ``lean`` outward (``out`` = +1: +y)
    by z1; ``thick`` thick at the bottom."""
    yb0, yb1 = y_face, y_face + out * thick
    yt0 = y_face + out * lean
    V = []
    for x in (x0, x1):
        V += [(x, yb0, z0), (x, yb1, z0), (x, yb1, z1), (x, yt0, z1)]
    F = [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]]
    return mesh(stage, path, V, F, mat, collide=True)


def _ccw(poly):
    """The polygon's points in counterclockwise order (seen from above)."""
    p = np.asarray(poly, float)
    area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    return [tuple(v) for v in (p if area > 0 else p[::-1])]


# -- the batteries ----------------------------------------------------------------------------------------------------
def build_battery(name: str, taped: bool):
    stage, root = stage_for(name)
    p = root.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(p)
    m = UsdPhysics.MassAPI.Apply(p)
    m.CreateMassAttr(g.BATT_MASS)
    L, W, T = g.BODY_L, g.BATT_W, g.BATT_T
    m.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(g.BATT_MASS * (W * W + T * T) / 12, g.BATT_MASS * (L * L + T * T) / 12,
                                         g.BATT_MASS * (L * L + W * W) / 12))
    r = f"/{name}"
    case = material(stage, f"{r}/Looks/case", (0.11, 0.12, 0.14), roughness=0.4)
    copper = material(stage, f"{r}/Looks/copper", (0.69, 0.41, 0.18), roughness=0.35, metallic=0.4)
    lab = material(stage, f"{r}/Looks/label", (1, 1, 1), roughness=0.4, texture="./textures/sa_battery_label.png")
    steel = material(stage, f"{r}/Looks/steel", STEEL, roughness=0.25, metallic=0.9)
    red = material(stage, f"{r}/Looks/tape", RED, roughness=0.35)
    off = (0.0005, 0.0)
    # the case (collider, sides in the case colour) and the label printed on both broad faces
    box(stage, f"{r}/case", (-L / 2, -W / 2, -T / 2), (L / 2, W / 2, T / 2), case, off=off)
    for s, z in ((1, T / 2 + 0.0002), (-1, -T / 2 - 0.0002)):
        V = [(-L / 2, -W / 2, z), (L / 2, -W / 2, z), (L / 2, W / 2, z), (-L / 2, W / 2, z)]
        # image left = the snap end (+x), image up = +y; on the underside the image is mirrored so it reads right
        uv = [(1, 0), (0, 0), (0, 1), (1, 1)] if s > 0 else [(1, 1), (0, 1), (0, 0), (1, 0)]
        mesh(stage, f"{r}/label_{'top' if s > 0 else 'bottom'}", V, [[0, 1, 2, 3] if s > 0 else [3, 2, 1, 0]], lab, uv=uv)
    # copper band on the narrow sides at the snap end
    box(stage, f"{r}/band", (L / 2 - 0.0125, -W / 2 - 0.0002, -T / 2 - 0.0001), (L / 2 - 0.0005, W / 2 + 0.0002, T / 2 + 0.0001),
        copper, collide=False)
    # the snap end's metal top plate (visual) and the two snaps (colliders)
    box(stage, f"{r}/top_plate", (L / 2 - 0.0005, -W / 2 + 0.0005, -T / 2 + 0.0005), (L / 2 + 0.0002, W / 2 - 0.0005, T / 2 - 0.0005),
        steel, collide=False)
    cylinder(stage, f"{r}/snap_pos", g.POS_R, (L / 2 + g.TERM_L / 2, g.POS_Y, 0.0), g.TERM_L, "X", steel, collide=True, off=off)
    cylinder(stage, f"{r}/snap_neg", g.NEG_R, (L / 2 + g.TERM_L / 2, g.NEG_Y, 0.0), g.TERM_L, "X", steel, collide=True, off=off)
    # the - snap's hexagonal look (visual): a ring of 6 flats
    for k in range(6):
        a = k * math.pi / 3
        c = (L / 2 + g.TERM_L / 2, g.NEG_Y + (g.NEG_R + 0.0001) * math.cos(a), (g.NEG_R + 0.0001) * math.sin(a))
        cyl = cylinder(stage, f"{r}/hex_{k}", 0.0006, c, g.TERM_L, "X", steel)
        _ = cyl
    if taped:
        box(stage, f"{r}/tape", (-L / 2 - 0.0002, -W / 2 - 0.0003, -T / 2 - 0.0003),
            (-L / 2 + g.TAPE_W, W / 2 + 0.0003, T / 2 + 0.0003), red, collide=False)
    stage.Save()


# -- the card ---------------------------------------------------------------------------------------------------------
def build_card():
    stage, root = stage_for("card")
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.RigidBodyAPI(root.GetPrim()).CreateKinematicEnabledAttr(True)
    r = "/card"
    cx, cy, ct = g.CARD
    plain = material(stage, f"{r}/Looks/plain", (0.70, 0.60, 0.45), roughness=0.9)
    pr = material(stage, f"{r}/Looks/print", (1, 1, 1), roughness=0.85, texture="./textures/sa_card.png")
    box(stage, f"{r}/board", (-cx / 2, -cy / 2, 0.0), (cx / 2, cy / 2, ct), plain, off=(0.0005, 0.0))
    z = ct + 0.0002
    V = [(-cx / 2, -cy / 2, z), (cx / 2, -cy / 2, z), (cx / 2, cy / 2, z), (-cx / 2, cy / 2, z)]
    uv = [(1, 0), (1, 1), (0, 1), (0, 0)]       # image x along -y, image y along -x (image up = +x)
    mesh(stage, f"{r}/print", V, [[0, 1, 2, 3]], pr, uv=uv)
    stage.Save()


# -- the mounting bracket ---------------------------------------------------------------------------------------------
def build_bracket():
    """The SMI100's twist-on mounting bracket: a white plastic ring, kinematic; origin at its centre on its face (the
    underside the alarm's back meets), z up into the shelf board above it."""
    stage, root = stage_for("bracket")
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.RigidBodyAPI(root.GetPrim()).CreateKinematicEnabledAttr(True)
    r = "/bracket"
    white = material(stage, f"{r}/Looks/white", (0.93, 0.93, 0.91), roughness=0.45)
    grey = material(stage, f"{r}/Looks/grey", (0.62, 0.62, 0.60), roughness=0.55)
    dark = material(stage, f"{r}/Looks/dark", (0.20, 0.20, 0.21), roughness=0.6)
    cylinder(stage, f"{r}/ring", g.RING_R, (0.0, 0.0, g.RING_T / 2), g.RING_T, "Z", white, collide=True,
             off=(0.0005, 0.0))
    cylinder(stage, f"{r}/face", g.RING_R - 0.010, (0.0, 0.0, -0.0002), 0.0003, "Z", grey)
    for k in range(6):                       # the six lugs' slots, 60 degrees apart (visual)
        a = k * g.ENTRY_STEP
        c = (0.047 * math.cos(a), 0.047 * math.sin(a), -0.0004)
        sl = UsdGeom.Cube.Define(stage, f"{r}/slot_{k}")
        sl.CreateSizeAttr(1.0)
        sl.AddTranslateOp().Set(Gf.Vec3d(*c))
        sl.AddOrientOp().Set(Gf.Quatf(math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)))
        sl.AddScaleOp().Set(Gf.Vec3f(0.006, 0.014, 0.0004))
        bind(sl, dark)
    for k in range(3):                       # screw heads (visual)
        a = math.radians(30 + 120 * k)
        cylinder(stage, f"{r}/screw_{k}", 0.004, (0.030 * math.cos(a), 0.030 * math.sin(a), -0.0005), 0.0004, "Z", grey)
    stage.Save()


if __name__ == "__main__":
    TEX.mkdir(parents=True, exist_ok=True)
    back_label(TEX / "sa_alarm_back.png")
    front_print(TEX / "sa_alarm_front.png")
    battery_label(TEX / "sa_battery_label.png")
    card_print(TEX / "sa_card.png")
    build_alarm()
    build_battery("battery_old", taped=True)
    build_battery("battery_new", taped=False)
    build_card()
    build_bracket()
    for p in sorted(OUT.rglob("*.*")):
        print(p.relative_to(OUT), p.stat().st_size, "bytes")
