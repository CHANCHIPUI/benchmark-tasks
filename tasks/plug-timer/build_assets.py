"""Write the scene's USD assets and textures from geometry.py.

    python build_assets.py          # needs pxr, PIL and the DejaVu fonts (Isaac Sim's Python), not the simulator

- assets/timer.usda: the power strip with the GE 15119 timer in its third socket, one floating articulation:
  - base (root body): the strip, the timer's white housing, the fixed white centre with its pointer, the timer's
    outlet with the lamp's plug in it (visual);
  - dial (revolute about z): the black wheel with its ribbed side and the printed ring of hours;
  - seg_00 .. seg_47 (prismatic along the dial's axis, children of the dial): the 48 half-hour segments, 2 mm of
    travel, wedge colliders;
  - switch (prismatic along y): the OUTLET ON / TIMER ON slide.
  Origin: on the dial's axis at the table top (the spawn pose puts it there).
- assets/pen.usda: the ballpoint, one rigid body; origin at the lowest point of its 1 mm ball, z along the pen.
- assets/mug.usda: the ceramic mug, one rigid body (16 wall boxes, a floor, a handle); origin at its base centre.
- assets/lamp.usda: the table lamp, kinematic, and the cord to the timer's outlet (visual); origin at its base centre.
- assets/textures/*.png: the dial's printed ring, the centre cap, the ribbed side.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import geometry as g

OUT = Path(__file__).resolve().parent / "assets"
TEX = OUT / "textures"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"
WHITE = (0.92, 0.92, 0.90)          # the timer's and strip's white plastic
STRIP = (0.95, 0.95, 0.94)
BLACK = (0.035, 0.035, 0.04)        # the dial
SEGC = (0.10, 0.10, 0.11)           # the segments (black plastic, a little lighter than the dial's body)
DARK = (0.02, 0.02, 0.02)
RED = (0.75, 0.10, 0.08)
INK = (12, 12, 14)
PAPER = (238, 238, 232)


# -- textures ------------------------------------------------------------------------------------------------------------
def _xy(c, ppm, r, theta):
    """Image pixel of a point at radius r (m) and dial angle theta (CCW from the dial's +x); image up is +x, image left
    is +y (the dial seen from above with its pointer side up)."""
    return c - r * math.sin(theta) * ppm, c - r * math.cos(theta) * ppm


def dial_ring(path: Path) -> None:
    """The printed ring: hours 1 to 12 twice (a sun at noon, a moon at midnight), white with black figures by day and
    black with white figures from 6:30 PM to 6:30 AM; each figure at its hour, its top toward the rim."""
    n = 1024
    R = g.NUM_R[1]
    ppm = n / (2 * R)
    c = n / 2
    im = Image.new("RGB", (n, n), (8, 8, 9))
    d = ImageDraw.Draw(im)
    # night band: from 6:30 PM (18.5) through midnight to 6:30 AM (6.5); angles CCW grow with time
    a_night0 = g.seg_angle_dial(37) - g.SEG_ANG / 2      # 18:30
    a_night1 = g.seg_angle_dial(13) - g.SEG_ANG / 2 + 2 * math.pi   # 06:30 next day
    r_in, r_out = g.NUM_R[0], R
    steps = 720
    for k in range(steps):
        a0 = 2 * math.pi * k / steps
        a1 = 2 * math.pi * (k + 1) / steps
        am = (a0 + a1) / 2
        rel = (am - a_night0) % (2 * math.pi)
        night = rel < (a_night1 - a_night0)
        col = (12, 12, 14) if night else PAPER
        poly = [_xy(c, ppm, r_in, a0), _xy(c, ppm, r_out, a0), _xy(c, ppm, r_out, a1), _xy(c, ppm, r_in, a1)]
        d.polygon(poly, fill=col)
    font = ImageFont.truetype(FONT, int(0.0029 * ppm))
    r_txt = (r_in + r_out) / 2
    for h in range(24):
        a = math.radians(g.DEG_PER_H * (h - g.START_TIME))
        rel = (a - a_night0) % (2 * math.pi)
        night = rel < (a_night1 - a_night0)
        fg = PAPER if night else INK
        label = str((h - 1) % 12 + 1)
        glyph = Image.new("RGBA", (160, 120), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glyph)
        if h == 12:          # the sun
            gd.ellipse([62, 42, 98, 78], fill=fg)
            for k in range(12):
                t = 2 * math.pi * k / 12
                gd.line([(80 + 22 * math.cos(t), 60 + 22 * math.sin(t)), (80 + 32 * math.cos(t), 60 + 32 * math.sin(t))],
                        fill=fg, width=5)
        elif h == 0:         # the moon
            gd.ellipse([56, 30, 112, 86], fill=fg)
            gd.ellipse([74, 22, 128, 78], fill=(0, 0, 0, 0))
        else:
            gd.text((80, 60), label, font=font, fill=fg, anchor="mm")
        # top of the glyph toward the rim: rotate by the angle (PIL rotates anticlockwise in degrees)
        glyph = glyph.rotate(math.degrees(a), resample=Image.BICUBIC, expand=True)
        x, y = _xy(c, ppm, r_txt, a)
        im.paste(glyph, (int(x - glyph.width / 2), int(y - glyph.height / 2)), glyph)
    # thin black circles at the ring's edges
    for r in (r_in, r_out - 0.0002):
        rr = r * ppm
        d.ellipse([c - rr, c - rr, c + rr, c + rr], outline=(8, 8, 9), width=4)
    im.save(path)


def cap_top(path: Path) -> None:
    """The fixed white centre: two curved clockwise arrows and the pointer (a black triangle at the +x edge)."""
    n = 512
    ppm = n / (2 * g.CAP_R)
    c = n / 2
    im = Image.new("RGB", (n, n), (236, 236, 230))
    d = ImageDraw.Draw(im)
    r = 0.0062 * ppm
    for start in (110, 290):           # arcs (PIL angles clockwise from +x of the image)
        d.arc([c - r, c - r, c + r, c + r], start, start + 120, fill=INK, width=18)
        e = math.radians(start + 120)
        tip = (c + r * math.cos(e), c + r * math.sin(e))
        tang = (-math.sin(e), math.cos(e))
        nrm = (math.cos(e), math.sin(e))
        d.polygon([(tip[0] + 34 * tang[0], tip[1] + 34 * tang[1]),
                   (tip[0] + 20 * nrm[0], tip[1] + 20 * nrm[1]),
                   (tip[0] - 20 * nrm[0], tip[1] - 20 * nrm[1])], fill=INK)
    # the pointer: a triangle at the top of the image (+x), its tip on the edge
    d.polygon([(c, 6), (c - 34, 78), (c + 34, 78)], fill=INK)
    im.save(path)


def ribs(path: Path) -> None:
    """The dial's ribbed side, unrolled: 96 ribs."""
    w, h = 2048, 64
    im = Image.new("RGB", (w, h), (8, 8, 9))
    d = ImageDraw.Draw(im)
    for k in range(96):
        x = w * k / 96
        d.rectangle([x, 0, x + 7, h], fill=(48, 48, 52))
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
        tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def bind(prim, mat) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim())
    UsdShade.MaterialBindingAPI(prim.GetPrim()).Bind(mat)


def offsets(prim, contact=0.0002, rest=0.0):
    prim = prim.GetPrim()
    prim.AddAppliedSchema("PhysxCollisionAPI")
    prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(contact)
    prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(rest)


def box(stage, path, size, center, mat=None, collide=True, rot=None, off=None):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    c.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
    if rot is not None:
        c.AddOrientOp().Set(Gf.Quatf(*map(float, rot)))
    c.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
    if mat is not None:
        bind(c, mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        if off is not None:
            offsets(c, *off)
    return c


def cylinder(stage, path, radius, z0, z1, mat, collide=False, center_xy=(0.0, 0.0), axis="Z", off=None, rot=None):
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(float(radius))
    c.CreateHeightAttr(float(z1 - z0))
    c.CreateAxisAttr(axis)
    if axis == "Z":
        c.AddTranslateOp().Set(Gf.Vec3d(float(center_xy[0]), float(center_xy[1]), float((z0 + z1) / 2)))
    else:
        c.AddTranslateOp().Set(Gf.Vec3d(*map(float, center_xy)))
    if rot is not None:
        c.AddOrientOp().Set(Gf.Quatf(*map(float, rot)))
    if mat is not None:
        bind(c, mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        if off is not None:
            offsets(c, *off)
    return c


def mesh(stage, path, V, F, mat=None, collide=False, off=None):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in V])
    m.CreateFaceVertexCountsAttr([len(f) for f in F])
    m.CreateFaceVertexIndicesAttr([int(i) for f in F for i in f])
    m.CreateSubdivisionSchemeAttr("none")
    if mat is not None:
        bind(m, mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("convexHull")
        if off is not None:
            offsets(m, *off)
    return m


def band(stage, path, radius, z0, z1, mat, segs=96):
    """A textured band round the z axis; image x runs anticlockwise seen from above."""
    pts, uvs, idx, counts = [], [], [], []
    for j in range(segs + 1):
        a = 2 * math.pi * j / segs
        for z in (z0, z1):
            pts.append(Gf.Vec3f(radius * math.cos(a), radius * math.sin(a), z))
    for j in range(segs):
        p, q, r, s = 2 * j, 2 * j + 2, 2 * j + 3, 2 * j + 1
        counts.append(4)
        idx += [p, q, r, s]
        u0, u1 = j / segs, (j + 1) / segs
        uvs += [Gf.Vec2f(u0, 0), Gf.Vec2f(u1, 0), Gf.Vec2f(u1, 1), Gf.Vec2f(u0, 1)]
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr(counts)
    m.CreateFaceVertexIndicesAttr(idx)
    UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set(uvs)
    m.CreateDoubleSidedAttr(True)
    bind(m, mat)
    return m


def disc(stage, path, radius, z, mat, segs=128, r_in=0.0):
    """A textured disc (or ring from r_in) facing up; image up is +x, image left is +y."""
    ang = np.linspace(0, 2 * math.pi, segs, endpoint=False)

    def uv(r, a):
        return Gf.Vec2f(0.5 - 0.5 * (r / radius) * math.sin(a), 0.5 + 0.5 * (r / radius) * math.cos(a))

    pts, uvs, idx, counts = [], [], [], []
    for k in range(segs):
        a0, a1 = ang[k], ang[(k + 1) % segs]
        quad = [(r_in, a0), (radius, a0), (radius, a1), (r_in, a1)]
        base = len(pts)
        for r, a in quad:
            pts.append(Gf.Vec3f(r * math.cos(a), r * math.sin(a), z))
            uvs.append(uv(r, a))
        idx += [base, base + 1, base + 2, base + 3]
        counts.append(4)
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr(counts)
    m.CreateFaceVertexIndicesAttr(idx)
    m.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * len(idx))
    UsdGeom.Mesh(m).SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set(uvs)
    m.CreateDoubleSidedAttr(True)
    bind(m, mat)
    return m


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


def kinematic(prim):
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)


def wedge(r0, r1, a0, a1, z0, z1):
    """A wedge's 8 corners and 6 quad faces (outward)."""
    V = []
    for z in (z0, z1):
        for r, a in ((r0, a0), (r1, a0), (r1, a1), (r0, a1)):
            V.append((r * math.cos(a), r * math.sin(a), z))
    F = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    return V, F


# -- the strip and timer -----------------------------------------------------------------------------------------------
def build_timer():
    stage, root = stage_for("timer")
    r = "/timer"
    white = material(stage, f"{r}/Looks/white", WHITE, roughness=0.45)
    strip_m = material(stage, f"{r}/Looks/strip", STRIP, roughness=0.5)
    black = material(stage, f"{r}/Looks/black", BLACK, roughness=0.55)
    segm = material(stage, f"{r}/Looks/seg", SEGC, roughness=0.5)
    dark = material(stage, f"{r}/Looks/dark", DARK, roughness=0.8)
    red = material(stage, f"{r}/Looks/red", RED, roughness=0.4)
    grey = material(stage, f"{r}/Looks/grey", (0.30, 0.30, 0.32), roughness=0.6)
    ring_m = material(stage, f"{r}/Looks/ring", (1, 1, 1), roughness=0.5, texture="./textures/dial_ring.png")
    cap_m = material(stage, f"{r}/Looks/cap", (1, 1, 1), roughness=0.45, texture="./textures/cap_top.png")
    rib_m = material(stage, f"{r}/Looks/ribs", (1, 1, 1), roughness=0.55, texture="./textures/ribs.png")
    off = (0.0002, 0.0)

    # base: the strip and the timer's housing (root body of a floating articulation)
    base = f"{r}/base"
    prim = body(stage, base, g.STRIP_MASS + g.TIMER_MASS - g.DIAL_MASS).GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    prim.AddAppliedSchema("PhysxArticulationAPI")
    prim.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    prim.CreateAttribute("physxArticulation:sleepThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
    prim.CreateAttribute("physxArticulation:stabilizationThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
    sy = g.STRIP_Y - g.TIMER_Y
    box(stage, f"{base}/strip", (g.STRIP_W, g.STRIP_L, g.STRIP_H), (0.0, sy, g.STRIP_H / 2), strip_m)
    # the strip's six sockets (visual; the timer covers the two either side of its own) and its rocker switch
    for k in range(6):
        y = sy + g.SOCKET_1 + k * g.SOCKET_PITCH
        box(stage, f"{base}/socket_{k + 1}", (0.032, 0.026, 0.0004), (0.0, y, g.STRIP_H + 0.0002), white, collide=False)
        for s in (-1, 1):
            box(stage, f"{base}/socket_{k + 1}_slot{'ab'[s > 0]}", (0.0075, 0.0018, 0.0003),
                (s * 0.0063, y + 0.002, g.STRIP_H + 0.0005), dark, collide=False)
        cylinder(stage, f"{base}/socket_{k + 1}_earth", 0.0024, g.STRIP_H + 0.0002, g.STRIP_H + 0.0006, dark,
                 center_xy=(0.0, y - 0.008))
    ysw = sy + g.STRIP_SWITCH_Y
    box(stage, f"{base}/strip_switch", (0.016, 0.024, 0.006), (0.0, ysw, g.STRIP_H + 0.003), red)
    # its cord, straight back off the table (visual)
    cylinder(stage, f"{base}/strip_cord", 0.0035, 0.0, 0.62, dark, axis="X",
             center_xy=(g.STRIP_W / 2 + 0.31, sy - g.STRIP_L / 2 + 0.015, 0.006))
    # the housing
    hx, hy, hz = g.HOUSING
    box(stage, f"{base}/housing", (hx, hy, hz), (0.0, 0.0, g.HOUSING_Z0 + hz / 2), white, off=off)
    # the timer's outlet on its +y side, the lamp's plug in it (visual)
    box(stage, f"{base}/outlet", (0.022, 0.0006, 0.018), (0.0, hy / 2 + 0.0003, g.OUTLET_Z), grey, collide=False)
    box(stage, f"{base}/plug", (0.020, 0.016, 0.014), (0.0, hy / 2 + 0.008, g.OUTLET_Z), white, collide=False)
    # the slide switch's slot and the fixed white centre with its pointer
    box(stage, f"{base}/switch_slot", (0.0004, 0.014, 0.0045), (hx / 2 + 0.0002, 0.002, g.SWITCH_Z), dark, collide=False)
    cylinder(stage, f"{base}/cap", g.CAP_R, g.FACE_H, g.CAP_H, white, collide=True, off=off)
    disc(stage, f"{base}/cap_top", g.CAP_R - 0.0001, g.CAP_H + 0.00005, cap_m)

    # the dial: origin on its axis at its top face
    dial = f"{r}/dial"
    body(stage, dial, g.DIAL_MASS, (0.0, 0.0, g.DIAL_TOP_H))
    zf = g.FACE_H - g.DIAL_TOP_H
    cylinder(stage, f"{dial}/wheel", g.RIB_R, zf, 0.0, black, collide=True, off=off)
    band(stage, f"{dial}/ribs", g.RIB_R + 0.00005, zf + 0.0003, -0.0003, rib_m)
    disc(stage, f"{dial}/ring", g.NUM_R[1], 0.00005, ring_m, r_in=g.NUM_R[0] - 0.0004)
    joint(stage, "revolute", f"{r}/joints/dial", base, dial, (0.0, 0.0, g.DIAL_TOP_H), "Z")

    # the 48 segments: prismatic along the dial's axis, each a wedge 2 mm tall standing on the dial's top face (up)
    for i in range(g.SEG_N):
        seg = f"{r}/seg_{i:02d}"
        body(stage, seg, g.SEG_MASS, (0.0, 0.0, g.DIAL_TOP_H))
        a = g.seg_angle_dial(i)
        V, F = wedge(g.SEG_R_IN, g.SEG_R_OUT, a - g.SEG_ANG_W / 2, a + g.SEG_ANG_W / 2, 0.0, g.SEG_TRAVEL)
        mesh(stage, f"{seg}/wedge", V, F, segm, collide=True, off=off)
        joint(stage, "prismatic", f"{r}/joints/seg_{i:02d}", dial, seg, (0.0, 0.0, 0.0), "Z", (-g.SEG_TRAVEL, 0.0))

    # the slide switch: its knob proud of the +x face; joint 0 = TIMER ON, +4 mm (+y) = OUTLET ON
    sw = f"{r}/switch"
    body(stage, sw, 0.002, (hx / 2, 0.0, g.SWITCH_Z))
    kx, ky, kz = g.SWITCH_KNOB
    box(stage, f"{sw}/knob", (kx, ky, kz), (kx / 2, 0.0, 0.0), dark, off=off)
    joint(stage, "prismatic", f"{r}/joints/switch", base, sw, (hx / 2, 0.0, g.SWITCH_Z), "Y",
          g.SWITCH_STOPS)
    stage.Save()


# -- the pen ---------------------------------------------------------------------------------------------------------------
def build_pen():
    stage, root = stage_for("pen")
    p = root.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(p)
    UsdPhysics.MassAPI.Apply(p).CreateMassAttr(g.PEN_MASS)
    r = "/pen"
    barrel = material(stage, f"{r}/Looks/barrel", (0.92, 0.92, 0.90), roughness=0.4)
    blue = material(stage, f"{r}/Looks/blue", (0.10, 0.22, 0.62), roughness=0.4)
    metal = material(stage, f"{r}/Looks/metal", (0.70, 0.70, 0.72), roughness=0.3, metallic=0.8)
    off = (0.0002, 0.0)
    br = g.PEN_BALL_R
    UsdPhysics.CollisionAPI.Apply(s := UsdGeom.Sphere.Define(stage, f"{r}/ball").GetPrim())
    UsdGeom.Sphere(s).CreateRadiusAttr(br)
    UsdGeom.Sphere(s).AddTranslateOp().Set(Gf.Vec3d(0, 0, br))
    bind(UsdGeom.Sphere(s), metal)
    offsets(UsdGeom.Sphere(s), *off)
    # the tapered point: a frustum from just above the ball to the barrel (convex hull)
    n = 16
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    z0, r0 = br * 1.2, br * 0.9
    V = [(r0 * math.cos(t), r0 * math.sin(t), z0) for t in a] + \
        [(g.PEN_R * math.cos(t), g.PEN_R * math.sin(t), g.PEN_CONE) for t in a]
    F = [[k for k in range(n)][::-1], [n + k for k in range(n)]] + [[k, (k + 1) % n, n + (k + 1) % n, n + k] for k in range(n)]
    mesh(stage, f"{r}/point", V, F, metal, collide=True, off=off)
    cylinder(stage, f"{r}/barrel", g.PEN_R, g.PEN_CONE, g.PEN_L - 0.010, barrel, collide=True, off=off)
    cylinder(stage, f"{r}/end", g.PEN_R, g.PEN_L - 0.010, g.PEN_L, blue, collide=True, off=off)
    cylinder(stage, f"{r}/ink", 0.0012, g.PEN_CONE, g.PEN_L - 0.012, blue)      # the refill, seen through the barrel
    stage.Save()


# -- the mug -----------------------------------------------------------------------------------------------------------
def build_mug():
    stage, root = stage_for("mug")
    p = root.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(p)
    UsdPhysics.MassAPI.Apply(p).CreateMassAttr(g.MUG_MASS)
    r = "/mug"
    glaze = material(stage, f"{r}/Looks/glaze", (0.16, 0.36, 0.52), roughness=0.25)
    inner = material(stage, f"{r}/Looks/inner", (0.88, 0.88, 0.85), roughness=0.25)
    n = 24
    R, t, H = g.MUG_R, g.MUG_WALL, g.MUG_H
    rc = R - t / 2
    wseg = 2 * rc * math.tan(math.pi / n) + 0.0006
    for k in range(n):
        a = 2 * math.pi * k / n
        q = (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))
        box(stage, f"{r}/wall_{k:02d}", (t, wseg, H - g.MUG_FLOOR), (rc * math.cos(a), rc * math.sin(a),
                                                                       g.MUG_FLOOR + (H - g.MUG_FLOOR) / 2),
            None, rot=q).CreatePurposeAttr(UsdGeom.Tokens.guide)
    cylinder(stage, f"{r}/floor", R - 0.001, 0.0, g.MUG_FLOOR, glaze, collide=True)
    # visual shell
    band(stage, f"{r}/outside", R, 0.0, H, glaze)
    band(stage, f"{r}/inside", R - t, g.MUG_FLOOR, H, inner)
    disc(stage, f"{r}/rim", R, H, glaze, r_in=R - t)
    disc(stage, f"{r}/bottom", R - t, g.MUG_FLOOR + 0.0001, inner)
    # the handle on the -y side (away from the arms' path), a collider
    box(stage, f"{r}/handle_a", (0.012, 0.022, 0.008), (0.0, -R - 0.010, H - 0.020), glaze)
    box(stage, f"{r}/handle_b", (0.012, 0.022, 0.008), (0.0, -R - 0.010, 0.022), glaze)
    box(stage, f"{r}/handle_c", (0.012, 0.008, 0.061), (0.0, -R - 0.019, H / 2 + 0.006), glaze)
    stage.Save()


# -- the lamp ----------------------------------------------------------------------------------------------------------
def build_lamp():
    stage, root = stage_for("table_lamp")
    kinematic(root.GetPrim())
    r = "/table_lamp"
    white = material(stage, f"{r}/Looks/white", (0.95, 0.95, 0.93), roughness=0.6)
    shade = material(stage, f"{r}/Looks/shade", (0.97, 0.96, 0.92), roughness=0.8)
    dark = material(stage, f"{r}/Looks/dark", (0.94, 0.94, 0.94), roughness=0.6)
    # LAMPAN: a conical base 13 cm across narrowing up to the bulb holder, under a conical shade 19 cm across at its rim
    n = 32
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)

    def frustum(path, r0, r1, z0, z1, mat, collide):
        V = [(r0 * math.cos(t), r0 * math.sin(t), z0) for t in a] + [(r1 * math.cos(t), r1 * math.sin(t), z1) for t in a]
        F = [[k for k in range(n)][::-1], [n + k for k in range(n)]] + \
            [[k, (k + 1) % n, n + (k + 1) % n, n + k] for k in range(n)]
        mesh(stage, path, V, F, mat, collide=collide)

    frustum(f"{r}/base", g.LAMP_BASE_R, 0.015, 0.0, 0.160, white, True)
    frustum(f"{r}/shade", g.LAMP_SHADE_R, 0.032, 0.120, g.LAMP_H, shade, True)
    # the cord from the base to the timer's plug, along the table, behind the strip (visual)
    plug = np.array([g.TIMER_X - g.LAMP_X, g.TIMER_Y + g.HOUSING[1] / 2 + 0.016 - g.LAMP_Y, g.OUTLET_Z])
    pts = [np.array([-0.04, -0.05, 0.004]), np.array([-0.20, -0.20, 0.004]),
           np.array([plug[0] + 0.050, plug[1] + 0.030, 0.004]), np.array([plug[0], plug[1] + 0.012, 0.004]),
           np.array([plug[0], plug[1] + 0.006, plug[2] - 0.004]), plug + np.array([0.0, 0.0, 0.0])]
    for k in range(len(pts) - 1):
        p0, p1 = pts[k], pts[k + 1]
        d = p1 - p0
        L = float(np.linalg.norm(d))
        z = np.array([0.0, 0.0, 1.0])
        v = d / L
        ax = np.cross(z, v)
        s = np.linalg.norm(ax)
        ang = math.atan2(s, float(z @ v))
        q = (1.0, 0.0, 0.0, 0.0) if s < 1e-9 else (math.cos(ang / 2), *(ax / s * math.sin(ang / 2)))
        c = UsdGeom.Cylinder.Define(stage, f"{r}/cord_{k}")
        c.CreateRadiusAttr(0.0028)
        c.CreateHeightAttr(L)
        c.AddTranslateOp().Set(Gf.Vec3d(*map(float, (p0 + p1) / 2)))
        c.AddOrientOp().Set(Gf.Quatf(*map(float, q)))
        bind(c, dark)
    stage.Save()


if __name__ == "__main__":
    TEX.mkdir(parents=True, exist_ok=True)
    dial_ring(TEX / "dial_ring.png")
    cap_top(TEX / "cap_top.png")
    ribs(TEX / "ribs.png")
    build_timer()
    build_pen()
    build_mug()
    build_lamp()
    for p in sorted(OUT.rglob("*.*")):
        print(p.relative_to(OUT), p.stat().st_size, "bytes")
