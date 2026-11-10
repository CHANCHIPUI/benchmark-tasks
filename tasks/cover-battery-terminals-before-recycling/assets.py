"""Geometry of the battery, its terminal cover and the collection box, and the USD files built from it (into assets/,
at load time). Pure NumPy. Every size is in metres; each body's origin is the centre of its outer envelope, except the
box, whose origin is the centre of its base.

Battery (sized from the common 7.0 V, 2,280 mAh camera pack: 40 x 56 x 20.5 mm, 80 g): a black body with the contact
end up. The top 9 mm is a neck stepped in on every side, with a 1.5 mm lead-in chamfer round its top edge and four
gold contacts on its top face. The cover is an open-bottom sleeve whose outside is flush with the body once it is on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import usdkit as uk

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets"

# -- battery ------------------------------------------------------------------------------------------------------
BAT_W, BAT_T, BAT_H = 0.040, 0.0205, 0.056  # x (width), y (thickness), z (height)
BAT_KG = 0.080
NECK_H = 0.009
COVER_WALL = 0.001
CLEAR = 0.0005  # per side, between the cover's inside and the neck
STEP = COVER_WALL + CLEAR  # how far the neck steps in from the body on every side
NECK_W, NECK_T = BAT_W - 2 * STEP, BAT_T - 2 * STEP
CHAMFER = 0.0015  # lead-in round the neck's top edge
SHOULDER_Z = BAT_H / 2 - NECK_H  # body frame: where the neck starts

# -- cover --------------------------------------------------------------------------------------------------------
COVER_W, COVER_T, COVER_H = BAT_W, BAT_T, 0.0105
COVER_TOP = 0.0015
COVER_KG = 0.005
SKIRT = COVER_H - COVER_TOP  # equals NECK_H, so a seated cover meets the shoulder
# A fully pushed-on cover's centre in the battery's frame: its top plate rests on the neck's top face.
SEAT = np.array([0.0, 0.0, BAT_H / 2 + COVER_TOP - COVER_H / 2])

# -- collection box -----------------------------------------------------------------------------------------------
BOX_L, BOX_W, BOX_H = 0.20, 0.30, 0.12  # x, y, height (a 300 x 200 x 120 mm Euro stacking box)
BOX_WALL = 0.003
INNER_HX, INNER_HY = BOX_L / 2 - BOX_WALL, BOX_W / 2 - BOX_WALL

FRICTION = (0.5, 0.4)  # static, dynamic: moulded plastic


def _battery() -> str:
    looks = {
        "case": uk.Look(color=(0.018, 0.018, 0.02), roughness=0.55),
        "gold": uk.Look(color=(0.95, 0.68, 0.30), metallic=1.0, roughness=0.28),
        "label": uk.Look(texture="battery_label.png", color=(1, 1, 1), roughness=0.45,
                         origin=(-0.017, 0.0, -0.022), tile_uv=(0.034, 0.030)),
        "label_m": uk.Look(texture="battery_label_m.png", color=(1, 1, 1), roughness=0.45,
                           origin=(-0.017, 0.0, -0.022), tile_uv=(0.034, 0.030)),
        "end": uk.Look(texture="battery_end_label.png", color=(1, 1, 1), roughness=0.45,
                       origin=(0.0, -0.009, -0.022), tile_uv=(0.018, 0.030)),
        "end_m": uk.Look(texture="battery_end_label_m.png", color=(1, 1, 1), roughness=0.45,
                         origin=(0.0, -0.009, -0.022), tile_uv=(0.018, 0.030)),
    }
    body_h = BAT_H - NECK_H
    body = uk.box((BAT_W, BAT_T, body_h), (0, 0, -BAT_H / 2 + body_h / 2), bevel=0.0008, round_xy=0.0015)
    neck = uk.prism(uk.rounded_rect(NECK_W, NECK_T, 0.0008), SHOULDER_Z - 0.0005, BAT_H / 2, bevel=CHAMFER)
    contacts = uk.join(*[uk.box((0.004, 0.009, 0.0003), (x, 0.0, BAT_H / 2 + 0.00015)) for x in (-0.0105, -0.0035, 0.0035, 0.0105)])
    label_y = BAT_T / 2 + 0.00012
    label_neg = uk.box((0.034, 0.0002, 0.030), (0.0, -label_y, -0.007))  # read from the -y side: u runs with +x
    label_pos = uk.box((0.034, 0.0002, 0.030), (0.0, label_y, -0.007))  # read from the +y side: u runs against +x
    end_x = BAT_W / 2 + 0.00012
    end_neg = uk.box((0.0002, 0.018, 0.030), (-end_x, 0.0, -0.007))  # read from the -x side: u runs against +y
    end_pos = uk.box((0.0002, 0.018, 0.030), (end_x, 0.0, -0.007))  # read from the +x side: u runs with +y
    parts = [("case_body", body, "case"), ("case_neck", neck, "case"), ("contacts", contacts, "gold"),
             ("label_a", label_neg, "label"), ("label_b", label_pos, "label_m"),
             ("label_c", end_neg, "end_m"), ("label_d", end_pos, "end")]
    # Colliders: the body box, and the neck as a convex hull with the chamfered top edge.
    hx, hy = NECK_W / 2, NECK_T / 2
    hull = [(sx * hx, sy * hy, z) for z in (SHOULDER_Z - 0.0005, BAT_H / 2 - CHAMFER) for sx in (-1, 1) for sy in (-1, 1)]
    hull += [(sx * (hx - CHAMFER), sy * (hy - CHAMFER), BAT_H / 2) for sx in (-1, 1) for sy in (-1, 1)]
    cols = [uk.collider("col_body", (BAT_W, BAT_T, body_h), (0, 0, -BAT_H / 2 + body_h / 2), "Battery"),
            uk.hull_collider("col_neck", hull, "Battery")]
    return uk.write_usda(str(OUT / "battery.usda"), "Battery", parts, looks, cols, mass=BAT_KG, friction=FRICTION)


def _cover() -> str:
    looks = {"cover": uk.Look(color=(0.035, 0.036, 0.04), roughness=0.3)}
    z0, z_top0 = -COVER_H / 2, COVER_H / 2 - COVER_TOP
    inner_w, inner_t = NECK_W + 2 * CLEAR, NECK_T + 2 * CLEAR
    skirt = uk.ring_prism(uk.rounded_rect(COVER_W, COVER_T, 0.0015), uk.rounded_rect(inner_w, inner_t, 0.0005), z0, z_top0 + 0.0002)
    top = uk.box((COVER_W, COVER_T, COVER_TOP), (0, 0, COVER_H / 2 - COVER_TOP / 2), bevel=0.0005, round_xy=0.0015)
    zc = z0 + SKIRT / 2
    cols = [
        uk.collider("col_top", (COVER_W, COVER_T, COVER_TOP), (0, 0, COVER_H / 2 - COVER_TOP / 2), "Cover"),
        uk.collider("col_side_a", (COVER_W, COVER_WALL, SKIRT), (0, COVER_T / 2 - COVER_WALL / 2, zc), "Cover"),
        uk.collider("col_side_b", (COVER_W, COVER_WALL, SKIRT), (0, -COVER_T / 2 + COVER_WALL / 2, zc), "Cover"),
        uk.collider("col_end_a", (COVER_WALL, inner_t, SKIRT), (COVER_W / 2 - COVER_WALL / 2, 0, zc), "Cover"),
        uk.collider("col_end_b", (COVER_WALL, inner_t, SKIRT), (-COVER_W / 2 + COVER_WALL / 2, 0, zc), "Cover"),
    ]
    parts = [("skirt", skirt, "cover"), ("top", top, "cover")]
    return uk.write_usda(str(OUT / "cover.usda"), "Cover", parts, looks, cols, mass=COVER_KG, friction=FRICTION)


def _box() -> str:
    looks = {
        "pp": uk.Look(color=(0.30, 0.31, 0.32), roughness=0.62),
        "label_m": uk.Look(texture="box_label_m.png", color=(1, 1, 1), roughness=0.5,
                           origin=(0.0, -0.06, 0.035), tile_uv=(0.12, 0.06)),
    }
    r_out, r_in = 0.012, 0.009
    walls = uk.ring_prism(uk.rounded_rect(BOX_L, BOX_W, r_out), uk.rounded_rect(BOX_L - 2 * BOX_WALL, BOX_W - 2 * BOX_WALL, r_in),
                          BOX_WALL, BOX_H)
    floor = uk.box((BOX_L, BOX_W, BOX_WALL), (0, 0, BOX_WALL / 2), round_xy=r_out)
    lip = uk.ring_prism(uk.rounded_rect(BOX_L + 0.004, BOX_W + 0.004, r_out + 0.002), uk.rounded_rect(BOX_L - 0.001, BOX_W - 0.001, r_out),
                        BOX_H - 0.009, BOX_H)
    rib = uk.ring_prism(uk.rounded_rect(BOX_L + 0.002, BOX_W + 0.002, r_out + 0.001), uk.rounded_rect(BOX_L - 0.001, BOX_W - 0.001, r_out),
                        0.004, 0.012)
    label = uk.box((0.0002, 0.12, 0.06), (-BOX_L / 2 - 0.0012, 0.0, 0.065))  # front face, read from the robot's side
    parts = [("walls", walls, "pp"), ("floor", floor, "pp"), ("lip", lip, "pp"), ("foot_rib", rib, "pp"),
             ("label", label, "label_m")]
    zc = BOX_WALL + (BOX_H - BOX_WALL) / 2
    hw = BOX_H - BOX_WALL
    cols = [
        uk.collider("col_floor", (BOX_L, BOX_W, BOX_WALL), (0, 0, BOX_WALL / 2), "Box"),
        uk.collider("col_front", (BOX_WALL, BOX_W, hw), (-BOX_L / 2 + BOX_WALL / 2, 0, zc), "Box"),
        uk.collider("col_back", (BOX_WALL, BOX_W, hw), (BOX_L / 2 - BOX_WALL / 2, 0, zc), "Box"),
        uk.collider("col_left", (BOX_L - 2 * BOX_WALL, BOX_WALL, hw), (0, BOX_W / 2 - BOX_WALL / 2, zc), "Box"),
        uk.collider("col_right", (BOX_L - 2 * BOX_WALL, BOX_WALL, hw), (0, -BOX_W / 2 + BOX_WALL / 2, zc), "Box"),
    ]
    return uk.write_usda(str(OUT / "box.usda"), "Box", parts, looks, cols, mass=1.2, friction=(0.5, 0.4))  # kinematic in the scene (no mass given there)


def build() -> None:
    _battery()
    _cover()
    _box()


if __name__ == "__main__":
    build()
    print("wrote", sorted(p.name for p in OUT.glob("*.usda")))
