"""The robot car, its test stand and the vial rack: sizes and USD files.

Lengths are metres. The car's own frame has its origin on the ground under the car's centre, x toward its nose, y to its
left, z up. The stand's and rack's origins are at the middle of their undersides. build() writes the files into build/
(not committed: they are generated each time the task file loads).

The car is one articulation with a free base: the chassis. Its four wheels turn on revolute joints about y, driven by
velocity drives with a torque limit (the gearmotors), and its power button's cap slides on a prismatic joint along z
with a spring (the switch). The lit and unlit looks of the button are separate meshes; the task shows one of them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import usdkit as U

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"

# -- the car ------------------------------------------------------------------------------------------------------------
CAR_MASS = 0.40
WHEEL_MASS, BUTTON_MASS = 0.025, 0.005
CHASSIS_MASS = CAR_MASS - 4 * WHEEL_MASS - BUTTON_MASS
LENGTH, DECK_W, DECK_T = 0.20, 0.086, 0.003
BELLY = 0.022                                   # underside of the lower deck
LOWER = (BELLY, BELLY + DECK_T)
UPPER = (0.050, 0.053)
DECK_TOP = UPPER[1]
WHEEL_R, WHEEL_W = 0.0325, 0.024
AXLE_X, AXLE_Z = 0.055, 0.0325
WHEEL_Y = 0.046 + WHEEL_W / 2                    # inner faces 9.2 cm apart
WHEELS = {"fl": (AXLE_X, WHEEL_Y), "fr": (AXLE_X, -WHEEL_Y), "rl": (-AXLE_X, WHEEL_Y), "rr": (-AXLE_X, -WHEEL_Y)}
HANDLE_X, HANDLE_W = 0.035, 0.012               # bar from -3.5 to +3.5 cm along the car, 1.2 cm square
HANDLE_Z = (0.073, 0.085)
POST_T = 0.007                                   # posts at the bar's ends, 0.7 cm thick along the car
BUTTON_X = -0.060
CAP_R, CAP_H = 0.0095, 0.008                    # cap 19 mm across, 8 mm proud of the deck
BEZEL_R = 0.0125                                 # 25 mm
BUTTON_TRAVEL = 0.005
COM_Z = 0.038
CHASSIS_COM_Z = (CAR_MASS * COM_Z - 4 * WHEEL_MASS * AXLE_Z - BUTTON_MASS * (DECK_TOP + CAP_H / 2)) / CHASSIS_MASS
TOP_Z = HANDLE_Z[1]

# -- the stand and the rack -----------------------------------------------------------------------------------------------
STAND_SIZE = (0.06, 0.26, 0.060)                 # across, along (its long side, longer than the car), tall with the pad
PAD_T = 0.003
STAND_MASS = 0.60                                # beech, about 0.68 g/cm3
RACK_SIZE = (0.07, 0.15, 0.030)
VIAL_R, VIAL_TOP, CAP_TOP = 0.014, 0.054, 0.066
VIALS = [(x, y) for x in (-0.0175, 0.0175) for y in (-0.06, -0.03, 0.0, 0.03, 0.06)]
RACK_MASS = 0.38
RACK_ENVELOPE = (RACK_SIZE[0], RACK_SIZE[1], CAP_TOP)  # the rack and its vials, for the scorer


def _lin(c):
    c = np.asarray(c, float)
    return tuple(float(v) for v in np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4))


LOOKS = {
    "acrylic": U.Look(color=_lin((0.035, 0.035, 0.04)), roughness=0.18),
    "tt_yellow": U.Look(color=_lin((0.96, 0.78, 0.10)), roughness=0.45),
    "hub_yellow": U.Look(color=_lin((0.98, 0.80, 0.12)), roughness=0.40),
    "hub_dark": U.Look(color=_lin((0.10, 0.09, 0.07)), roughness=0.6),
    "tyre": U.Look(color=_lin((0.05, 0.05, 0.055)), roughness=0.85),
    "can": U.Look(color=_lin((0.75, 0.76, 0.78)), metallic=0.9, roughness=0.3),
    "brass": U.Look(color=_lin((0.78, 0.62, 0.30)), metallic=0.9, roughness=0.3),
    "battery": U.Look(color=_lin((0.10, 0.28, 0.62)), roughness=0.35),
    "holder": U.Look(color=_lin((0.06, 0.06, 0.06)), roughness=0.5),
    "aluminium": U.Look(color=(1.0, 1.0, 1.0), metallic=0.8, roughness=0.35, texture="aluminium.png", tile=0.05),
    "pcb": U.Look(color=(1.0, 1.0, 1.0), roughness=0.4, texture="pcb.png", tile=0.10),
    "sensor_pcb": U.Look(color=_lin((0.10, 0.30, 0.70)), roughness=0.4),
    "chip": U.Look(color=_lin((0.04, 0.04, 0.04)), roughness=0.4),
    "white_plastic": U.Look(color=_lin((0.92, 0.92, 0.90)), roughness=0.5),
    "wire_red": U.Look(color=_lin((0.75, 0.06, 0.05)), roughness=0.5),
    "wire_black": U.Look(color=_lin((0.04, 0.04, 0.04)), roughness=0.5),
    "chrome": U.Look(color=_lin((0.82, 0.83, 0.85)), metallic=1.0, roughness=0.15),
    "cap_off": U.Look(color=_lin((0.10, 0.11, 0.10)), roughness=0.3),
    "led_off": U.Look(color=_lin((0.30, 0.30, 0.30)), roughness=0.25),
    "led_on": U.Look(color=_lin((0.20, 0.95, 0.30)), roughness=0.3, emissive=(0.6, 6.0, 1.0)),
    "beech": U.Look(color=(1.0, 1.0, 1.0), roughness=0.55, texture="beech.jpg", tile=0.05),
    "rubber_ribbed": U.Look(color=(1.0, 1.0, 1.0), roughness=0.9, texture="rubber_ribbed.png", tile=0.10),
    "rack_white": U.Look(color=_lin((0.93, 0.93, 0.91)), roughness=0.55),
    "hole_dark": U.Look(color=_lin((0.35, 0.35, 0.35)), roughness=0.7),
    "glass": U.Look(color=_lin((0.88, 0.93, 0.95)), roughness=0.05, opacity=0.35),
    "sample": U.Look(color=_lin((0.85, 0.55, 0.15)), roughness=0.1, opacity=0.8),
    "vial_cap": U.Look(color=_lin((0.05, 0.05, 0.05)), roughness=0.45),
}


def _box(lo, hi, bevel=0.0, round_xy=0.0) -> U.Mesh:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return U.box(hi - lo, (lo + hi) / 2, bevel=bevel, round_xy=round_xy)


def _mesh_collider(name: str, mesh: U.Mesh, root: str, material: str, indent: str = "    ") -> str:
    """A convex-hull collider through a mesh's points (not rendered)."""
    return f"""
{indent}def Mesh "{name}" (
{indent}    prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "PhysicsMeshCollisionAPI", "MaterialBindingAPI"]
{indent})
{indent}{{
{indent}    rel material:binding:physics = </{root}/{material}>
{indent}    uniform token purpose = "guide"
{indent}    uniform token physics:approximation = "convexHull"
{indent}    float physxCollision:contactOffset = 0.002
{indent}    float physxCollision:restOffset = 0
{indent}    int[] faceVertexCounts = [{", ".join("3" for _ in mesh.f)}]
{indent}    int[] faceVertexIndices = [{", ".join(str(int(i)) for i in mesh.f.reshape(-1))}]
{indent}    point3f[] points = [{U._fmt(mesh.v, 6)}]
{indent}}}"""


def _box_collider(name, lo, hi, root, material, indent="        ") -> str:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    text = U.collider(name, hi - lo, (lo + hi) / 2, root, material)
    return text.replace("\n    ", "\n" + indent).replace("contactOffset = 0.004", "contactOffset = 0.002")


def _sector(r0, r1, a0, a1, z0, z1, seg=8) -> U.Mesh:
    """A ring sector (a spoke gap) between radii r0 and r1, angles a0 to a1 (rad), from z0 to z1."""
    a = np.linspace(a0, a1, seg + 1)
    outline = np.r_[np.c_[r1 * np.cos(a), r1 * np.sin(a)], np.c_[r0 * np.cos(a[::-1]), r0 * np.sin(a[::-1])]]
    return U.prism(outline, z0, z1)


def wheel_parts(outer_sign: float):
    """Tyre, hub, spoke gaps and boss, about the wheel's centre with its axle along y; the outer face toward
    outer_sign * y."""
    h = WHEEL_W / 2
    r, rim = WHEEL_R, 0.0235
    tyre = U.lathe([(rim, -h), (r - 0.003, -h), (r - 0.0005, -h + 0.0025), (r, -h + 0.005), (r, -0.0035),
                    (r - 0.0012, -0.0025), (r - 0.0012, -0.0010), (r, 0.0), (r, h - 0.005),
                    (r - 0.0005, h - 0.0025), (r - 0.003, h), (rim, h), (rim, -h)][:-1], 36)
    hub = U.cylinder(rim + 0.0003, WHEEL_W - 0.004, (0, 0, -0.001), segments=36)
    face = h - 0.003                                   # hub face, 1 mm inside the tyre's outer face
    gaps = [_sector(0.0085, 0.0195, k * 2 * np.pi / 3 + 0.25, k * 2 * np.pi / 3 + 1.55, face, face + 0.0004)
            for k in range(3)]
    boss = U.cylinder(0.0065, 0.004, (0, 0, face + 0.002), segments=20)
    parts = [("tyre", tyre, "tyre"), ("hub", hub, "hub_yellow"), ("gaps", U.join(*gaps), "hub_dark"),
             ("boss", boss, "hub_yellow")]
    turn = -90.0 if outer_sign > 0 else 90.0           # lathe +z onto +y (left wheels) or -y (right wheels)
    return [(n, m.turned(turn, "x"), look) for n, m, look in parts]


def wheel_hull() -> U.Mesh:
    return U.cylinder(WHEEL_R, WHEEL_W, axis="y", segments=24)


def chassis_parts():
    L2, W2 = LENGTH / 2, DECK_W / 2
    parts = [
        ("lower_deck", _box((-L2, -W2, LOWER[0]), (L2, W2, LOWER[1]), bevel=0.0005, round_xy=0.012), "acrylic"),
        ("upper_deck", _box((-L2, -W2, UPPER[0]), (L2, W2, UPPER[1]), bevel=0.0005, round_xy=0.012), "acrylic"),
        ("battery_holder", _box((-0.037, -0.0195, LOWER[1]), (0.037, 0.0195, LOWER[1] + 0.003)), "holder"),
    ]
    for k, y in enumerate((-0.0095, 0.0095)):
        parts.append((f"cell_{k}", U.cylinder(0.0090, 0.065, (0.0, y, LOWER[1] + 0.012), axis="x", segments=20), "battery"))
    for name, (x, y) in WHEELS.items():
        s = np.sign(y)
        cx = np.sign(x) * 0.052
        parts.append((f"gearbox_{name}", _box((cx - 0.0185, s * 0.019, LOWER[1] + 0.0005), (cx + 0.0185, s * 0.041, LOWER[1] + 0.0195),
                                            bevel=0.001), "tt_yellow"))
        parts.append((f"motor_{name}", U.cylinder(0.0095, 0.020, (np.sign(x) * 0.0235, s * 0.030, LOWER[1] + 0.0100),
                                                    axis="x", segments=20), "can"))
        parts.append((f"shaft_{name}", U.cylinder(0.0035, 0.008, (x, s * 0.044, AXLE_Z), axis="y", segments=12), "white_plastic"))
    for k, (x, y) in enumerate((x, y) for x in (-0.088, 0.0, 0.088) for y in (-0.036, 0.036)):
        if True:
            parts.append((f"standoff_{k}", U.cylinder(0.0025, UPPER[0] - LOWER[1], (x, y, (UPPER[0] + LOWER[1]) / 2),
                                                                    segments=6), "brass"))
    # Carry handle: a bar on two posts.
    parts.append(("handle_bar", _box((-HANDLE_X, -HANDLE_W / 2, HANDLE_Z[0]), (HANDLE_X, HANDLE_W / 2, HANDLE_Z[1]), bevel=0.0012),
                  "aluminium"))
    for s in (-1, 1):
        parts.append((f"handle_post_{'p' if s > 0 else 'n'}", _box((min(s * HANDLE_X, s * (HANDLE_X - POST_T)), -HANDLE_W / 2, DECK_TOP),
                                                 (max(s * HANDLE_X, s * (HANDLE_X - POST_T)), HANDLE_W / 2, HANDLE_Z[0] + 0.001)),
                      "aluminium"))
    # Controller board with a few parts, on short standoffs at the front.
    bz = DECK_TOP + 0.005
    parts += [
        ("board", _box((0.042, -0.030, bz), (0.082, 0.030, bz + 0.0016)), "pcb"),
        ("chip", _box((0.055, -0.008, bz + 0.0016), (0.069, 0.006, bz + 0.0036)), "chip"),
        ("usb", _box((0.072, 0.012, bz + 0.0016), (0.083, 0.024, bz + 0.0066)), "can"),
        ("terminal", _box((0.046, -0.027, bz + 0.0016), (0.054, -0.012, bz + 0.0096)), "sensor_pcb"),
    ]
    for k, (x, y) in enumerate((x, y) for x in (0.045, 0.079) for y in (-0.027, 0.027)):
        if True:
            parts.append((f"board_post_{k}", U.cylinder(0.002, 0.005, (x, y, DECK_TOP + 0.0025), segments=6), "brass"))
    # Ultrasonic distance sensor at the nose.
    parts += [
        ("sensor_bracket", _box((0.080, -0.010, DECK_TOP), (0.086, 0.010, DECK_TOP + 0.004)), "holder"),
        ("sensor_board", _box((0.084, -0.023, DECK_TOP + 0.002), (0.0856, 0.023, DECK_TOP + 0.023)), "sensor_pcb"),
    ]
    for k, y in enumerate((-0.013, 0.013)):
        parts.append((f"sensor_eye_{k}", U.cylinder(0.008, 0.012, (0.0916, y, DECK_TOP + 0.0125), axis="x", segments=24), "can"))
    # The motor wires come up through the deck beside the board and into its terminal block.
    for s_ in (-1, 1):
        for dx, look in ((0.0, "wire_red"), (0.0028, "wire_black")):
            pts = [(0.036 + dx, s_ * 0.034, UPPER[0] - 0.006), (0.036 + dx, s_ * 0.034, DECK_TOP + 0.003),
                   (0.042 + dx, s_ * 0.030, bz + 0.004), (0.048 + dx, -0.020, bz + 0.006)]
            parts.append((f"wire_{'l' if s_ > 0 else 'r'}_{look[5:]}", U.tube(np.array(pts), 0.0009, 6), look))
    # Power button: chrome bezel and its LED ring (unlit and lit).
    bezel = U.ring_prism(U.rounded_rect(2 * BEZEL_R, 2 * BEZEL_R, BEZEL_R - 1e-4, 8), U.rounded_rect(2 * 0.0112, 2 * 0.0112, 0.0111, 8),
                         DECK_TOP, DECK_TOP + 0.0018).moved((BUTTON_X, 0, 0))
    ring = U.ring_prism(U.rounded_rect(2 * 0.0112, 2 * 0.0112, 0.0111, 8), U.rounded_rect(2 * 0.0099, 2 * 0.0099, 0.0098, 8),
                        DECK_TOP, DECK_TOP + 0.0016).moved((BUTTON_X, 0, 0))
    parts += [("bezel", bezel, "chrome"), ("led_ring_off", ring, "led_off"), ("led_ring_on", ring, "led_on", True)]
    return parts


def chassis_colliders(root: str) -> list[str]:
    L2, W2 = LENGTH / 2, DECK_W / 2
    m = "CarPhysics"
    c = [
        _box_collider("c_lower_deck", (-L2, -W2, LOWER[0]), (L2, W2, LOWER[1]), root, m),
        _box_collider("c_middle", (-0.075, -0.041, LOWER[1]), (0.075, 0.041, UPPER[0]), root, m),
        _box_collider("c_upper_deck", (-L2, -W2, UPPER[0]), (L2, W2, UPPER[1]), root, m),
        _box_collider("c_handle_bar", (-HANDLE_X, -HANDLE_W / 2, HANDLE_Z[0]), (HANDLE_X, HANDLE_W / 2, HANDLE_Z[1]), root, m),
        _box_collider("c_board", (0.042, -0.030, DECK_TOP), (0.083, 0.030, DECK_TOP + 0.0116), root, m),
        _box_collider("c_sensor", (0.080, -0.023, DECK_TOP), (0.0976, 0.023, DECK_TOP + 0.023), root, m),
    ]
    for s in (-1, 1):
        lo_x, hi_x = sorted((s * HANDLE_X, s * (HANDLE_X - POST_T)))
        c.append(_box_collider(f"c_post_{'p' if s > 0 else 'n'}", (lo_x, -HANDLE_W / 2, DECK_TOP), (hi_x, HANDLE_W / 2, HANDLE_Z[0]), root, m))
    return c


def button_parts():
    """The cap in the button link's frame (origin on the deck top at the button's axis)."""
    cap = U.cylinder(CAP_R, CAP_H, (0, 0, CAP_H / 2), segments=28, bevel=0.0008)
    return [("cap_off", cap, "cap_off"), ("cap_on", cap, "led_on", True)]


def _material_block(root: str, names) -> str:
    return "".join(U._material_usda(n, root, LOOKS[n]) for n in sorted(set(names)))


def _meshes(parts, root: str, indent: str) -> str:
    return "".join(U._mesh_usda(p[0], p[1], p[2], root, LOOKS[p[2]], 35.0, indent=indent, hidden=len(p) > 3 and p[3])
                   for p in parts)


def _body(name: str, pos, mass: float, com, inertia, colliders: str, visuals: str, extra_api: str = "", extra: str = "") -> str:
    return f"""
    def Xform "{name}" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"{extra_api}]
    )
    {{
        double3 xformOp:translate = ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f})
        uniform token[] xformOpOrder = ["xformOp:translate"]
        float physics:mass = {mass:.5f}
        point3f physics:centerOfMass = ({com[0]:.6f}, {com[1]:.6f}, {com[2]:.6f})
        float3 physics:diagonalInertia = ({inertia[0]:.3e}, {inertia[1]:.3e}, {inertia[2]:.3e})
        float physxRigidBody:maxDepenetrationVelocity = 1
        float physxRigidBody:sleepThreshold = 0
        float physxRigidBody:stabilizationThreshold = 0{extra}
{colliders}

        def Xform "visual"
        {{{visuals}
        }}
    }}"""


def _car() -> str:
    root = "RobotCar"
    chassis = chassis_parts()
    wheels = {n: wheel_parts(np.sign(y)) for n, (x, y) in WHEELS.items()}
    button = button_parts()
    used = [p[2] for p in chassis] + [p[2] for w in wheels.values() for p in w] + [p[2] for p in button]
    ind = "            "
    sx, sy, sz = LENGTH, DECK_W, UPPER[1] - LOWER[0]
    inertia = (CHASSIS_MASS / 12 * (sy * sy + sz * sz), CHASSIS_MASS / 12 * (sx * sx + sz * sz), CHASSIS_MASS / 12 * (sx * sx + sy * sy))
    bodies = [_body("chassis", (0, 0, 0), CHASSIS_MASS, (0, 0, CHASSIS_COM_Z), inertia, "".join(chassis_colliders(root)),
                    _meshes(chassis, root, ind), ', "PhysicsArticulationRootAPI", "PhysxArticulationAPI"',
                    """
        bool physxArticulation:enabledSelfCollisions = 0
        int physxArticulation:solverPositionIterationCount = 32
        int physxArticulation:solverVelocityIterationCount = 1
        float physxArticulation:sleepThreshold = 0
        float physxArticulation:stabilizationThreshold = 0""")]
    wi_axis = 0.5 * WHEEL_MASS * WHEEL_R ** 2
    wi_other = WHEEL_MASS / 12 * (3 * WHEEL_R ** 2 + WHEEL_W ** 2)
    joints = []
    for n, (x, y) in WHEELS.items():
        col = _mesh_collider("collider", wheel_hull(), root, "TyrePhysics", indent="        ")
        bodies.append(_body(f"wheel_{n}", (x, y, AXLE_Z), WHEEL_MASS, (0, 0, 0), (wi_other, wi_axis, wi_other), col,
                            _meshes(wheels[n], root, ind)))
        joints.append(f"""
        def PhysicsRevoluteJoint "axle_{n}" (
            prepend apiSchemas = ["PhysicsDriveAPI:angular", "PhysxJointAPI"]
        )
        {{
            uniform token physics:axis = "Y"
            rel physics:body0 = </{root}/chassis>
            rel physics:body1 = </{root}/wheel_{n}>
            point3f physics:localPos0 = ({x:.6f}, {y:.6f}, {AXLE_Z:.6f})
            quatf physics:localRot0 = (1, 0, 0, 0)
            point3f physics:localPos1 = (0, 0, 0)
            quatf physics:localRot1 = (1, 0, 0, 0)
            uniform token drive:angular:physics:type = "force"
            float drive:angular:physics:stiffness = 0
            float drive:angular:physics:damping = 0.05
            float drive:angular:physics:maxForce = 0.1
            float physxJoint:armature = 0.00001
        }}""")
    cap_col = _box_collider("collider", (-0.0085, -0.0085, 0.0), (0.0085, 0.0085, CAP_H), root, "CarPhysics")
    bodies.append(_body("button", (BUTTON_X, 0, DECK_TOP), BUTTON_MASS, (0, 0, CAP_H / 2), (2e-7, 2e-7, 2e-7), cap_col,
                        _meshes(button, root, ind)))
    joints.append(f"""
        def PhysicsPrismaticJoint "button" (
            prepend apiSchemas = ["PhysicsDriveAPI:linear", "PhysxJointAPI"]
        )
        {{
            uniform token physics:axis = "Z"
            rel physics:body0 = </{root}/chassis>
            rel physics:body1 = </{root}/button>
            point3f physics:localPos0 = ({BUTTON_X:.6f}, 0, {DECK_TOP:.6f})
            quatf physics:localRot0 = (1, 0, 0, 0)
            point3f physics:localPos1 = (0, 0, 0)
            quatf physics:localRot1 = (1, 0, 0, 0)
            float physics:lowerLimit = {-BUTTON_TRAVEL:.6f}
            float physics:upperLimit = 0
            uniform token drive:linear:physics:type = "force"
            float drive:linear:physics:stiffness = 500
            float drive:linear:physics:damping = 2
            float drive:linear:physics:maxForce = 100
            float physxJoint:armature = 0.00001
        }}""")
    text = f"""#usda 1.0
(
    defaultPrim = "{root}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}"
{{{U.physics_material("CarPhysics", 0.9, 0.8)}{U.physics_material("TyrePhysics", 1.0, 0.9)}

    def Scope "Looks"
    {{{_material_block(root, used)}
    }}
{"".join(bodies)}

    def Scope "joints"
    {{{"".join(joints)}
    }}
}}
"""
    return _write("robot_car.usda", text)


def _write(name: str, text: str) -> str:
    BUILD.mkdir(exist_ok=True)
    text = text.replace(f"@{(HERE / 'textures').as_posix()}/", "@../textures/")
    (BUILD / name).write_text(text)
    return f"build/{name}"


def _stand() -> str:
    root = "TestStand"
    sx, sy, sz = STAND_SIZE
    block = _box((-sx / 2, -sy / 2, 0), (sx / 2, sy / 2, sz - PAD_T), bevel=0.0015, round_xy=0.002)
    pad = _box((-sx / 2 + 0.0005, -sy / 2 + 0.0005, sz - PAD_T - 0.0002), (sx / 2 - 0.0005, sy / 2 - 0.0005, sz), bevel=0.0006)
    parts = [("block", block, "beech"), ("pad", pad, "rubber_ribbed")]
    cols = [U.collider("c_block", (sx, sy, sz), (0, 0, sz / 2), root)]
    BUILD.mkdir(exist_ok=True)
    U.write_usda(str(BUILD / "stand.usda"), root, parts, LOOKS, cols, mass=STAND_MASS, friction=(0.9, 0.8))
    return "build/stand.usda"


def _rack() -> str:
    root = "VialRack"
    rx, ry, rz = RACK_SIZE
    parts = [("rack", _box((-rx / 2, -ry / 2, 0), (rx / 2, ry / 2, rz), bevel=0.0015, round_xy=0.004), "rack_white")]
    cols = [U.collider("c_rack", (rx, ry, rz), (0, 0, rz / 2), root)]
    for k, (x, y) in enumerate(VIALS):
        parts += [
            (f"hole_{k}", U.cylinder(VIAL_R + 0.0012, 0.0004, (x, y, rz + 0.0002), segments=24), "hole_dark"),
            (f"vial_{k}", U.cylinder(VIAL_R, VIAL_TOP - rz + 0.004, (x, y, (VIAL_TOP + rz - 0.004) / 2), segments=28, bevel=0.0008), "glass"),
            (f"sample_{k}", U.cylinder(VIAL_R - 0.0012, 0.016, (x, y, rz + 0.008), segments=24), "sample"),
            (f"cap_{k}", U.cylinder(VIAL_R - 0.0004, CAP_TOP - VIAL_TOP + 0.0015, (x, y, (CAP_TOP + VIAL_TOP - 0.0015) / 2),
                                    segments=28, bevel=0.0008), "vial_cap"),
        ]
        hull = U.cylinder(VIAL_R, CAP_TOP - rz, (x, y, (CAP_TOP + rz) / 2), segments=12)
        cols.append(_mesh_collider(f"c_vial_{k}", hull, root, "PhysicsMaterial"))
    U.write_usda(str(BUILD / "rack.usda"), root, parts, LOOKS, cols, mass=RACK_MASS, friction=(0.5, 0.4))
    return "build/rack.usda"


def build() -> dict[str, str]:
    return {"car": _car(), "stand": _stand(), "rack": _rack()}


if __name__ == "__main__":
    print(build())
    print("chassis mass", round(CHASSIS_MASS, 4), "chassis CoM z", round(CHASSIS_COM_Z, 4))
