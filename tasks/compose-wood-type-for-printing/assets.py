"""Geometry of the wood type and the composing bench, and the USD files built from it (into assets/, at load time).
Pure NumPy. Sizes in metres.

Wood type: 18-line poster sorts (body 18 picas = 76.2 mm head to foot), type-high (0.918 in = 23.3 mm), end-grain hard
maple (about 0.72 g/cm3). Each sort's origin is its centre; its +x is the head of its letter and +z its printing face.

Bench (one fixed body whose origin is the table top under the arms' midpoint): a steel imposing surface; on it a cast-iron
chase whose inside is divided by three fixed hardwood furniture strips into four bays; and a rubber bench mat in front.
"""

from __future__ import annotations

from pathlib import Path

import usdkit as uk

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets"

# -- wood type ---------------------------------------------------------------------------------------------------
BODY = 0.0762  # head to foot (x)
TYPE_HIGH = 0.0233  # z
FACE_T = 0.0004  # the printing face's skin on top of the body (visual)
MAPLE_KG_M3 = 720.0
SET_WIDTH = {"K": 0.047, "E": 0.036, "L": 0.036, "P": 0.046, "J": 0.028}  # y; Oswald Bold at 69 mm cap height, 2.3 to 2.5 mm shoulders
MASS = {c: round(MAPLE_KG_M3 * BODY * w * TYPE_HIGH, 3) for c, w in SET_WIDTH.items()}
SORT_FRICTION = (0.6, 0.5)  # maple on steel, rubber and maple

# -- bench --------------------------------------------------------------------------------------------------------
CX = 0.36  # the chase's centre, x
SURFACE = (0.20, 0.34, 0.006)  # imposing surface, centred (CX, 0)
BED = SURFACE[2]  # the bed's height above the table top
BAY_W, BAY_L = 0.060, 0.130  # each bay's inside, y and x
FURN_W = 0.0085  # 2-pica furniture
FURN_H = 0.016  # below type-high, so it never prints
BAR_W, BAR_H = 0.015, 0.016  # chase bars
INNER_W = 4 * BAY_W + 3 * FURN_W  # 0.2655
BAY_Y = [INNER_W / 2 - BAY_W / 2 - k * (BAY_W + FURN_W) for k in range(4)]  # bays 1..4, left to right from the arms
FURN_Y = [INNER_W / 2 - BAY_W - FURN_W / 2 - k * (BAY_W + FURN_W) for k in range(3)]
MAT = (0.12, 0.60, 0.003)  # bench mat, x, y, thickness
MAT_XY = (0.195, -0.03)


def _sort(letter: str) -> str:
    w = SET_WIDTH[letter]
    looks = {
        "side": uk.Look(texture="maple_side.jpg", color=(1, 1, 1), roughness=0.62, tile=0.05),
        "face": uk.Look(texture=f"face_{letter}.png", color=(1, 1, 1), roughness=0.5,
                        origin=(-BODY / 2, -w / 2, 0.0), tile_uv=(BODY, w)),
    }
    body = uk.box((BODY, w, TYPE_HIGH - FACE_T), (0, 0, -FACE_T / 2))
    skin = uk.box((BODY, w, FACE_T), (0, 0, TYPE_HIGH / 2 - FACE_T / 2))
    parts = [("body", body, "side"), ("face", skin, "face")]
    cols = [uk.collider("col_sort", (BODY, w, TYPE_HIGH), (0, 0, 0), f"Sort_{letter}")]
    return uk.write_usda(str(OUT / f"sort_{letter}.usda"), f"Sort_{letter}", parts, looks, cols, mass=MASS[letter],
                         friction=SORT_FRICTION)


def _bench() -> str:
    looks = {
        "steel": uk.Look(color=(0.30, 0.31, 0.33), metallic=0.85, roughness=0.38),
        "iron": uk.Look(color=(0.045, 0.046, 0.05), metallic=0.55, roughness=0.6),
        "furniture": uk.Look(texture="furniture.jpg", color=(1, 1, 1), roughness=0.65, tile=0.05),
        "rubber": uk.Look(color=(0.022, 0.023, 0.025), roughness=0.92),
    }
    sx, sy, sz = SURFACE
    z_bar = BED + BAR_H / 2
    out_l, out_w = BAY_L + 2 * BAR_W, INNER_W + 2 * BAR_W
    chase = uk.ring_prism(uk.rounded_rect(out_l, out_w, 0.004), uk.rounded_rect(BAY_L, INNER_W, 0.0015), BED, BED + BAR_H)
    chase = uk.Mesh(chase.v + [CX, 0, 0], chase.f)
    parts = [
        ("surface", uk.box(SURFACE, (CX, 0, sz / 2), bevel=0.001), "steel"),
        ("chase", chase, "iron"),
        ("mat", uk.box((MAT[0], MAT[1], MAT[2]), (MAT_XY[0], MAT_XY[1], MAT[2] / 2), round_xy=0.006), "rubber"),
    ]
    parts += [(f"furniture_{k + 1}", uk.box((BAY_L - 0.0004, FURN_W, FURN_H), (CX, y, BED + FURN_H / 2)), "furniture")
              for k, y in enumerate(FURN_Y)]
    cols = [
        uk.collider("col_surface", SURFACE, (CX, 0, sz / 2), "Bench"),
        uk.collider("col_bar_far", (BAR_W, out_w, BAR_H), (CX + BAY_L / 2 + BAR_W / 2, 0, z_bar), "Bench"),
        uk.collider("col_bar_near", (BAR_W, out_w, BAR_H), (CX - BAY_L / 2 - BAR_W / 2, 0, z_bar), "Bench"),
        uk.collider("col_bar_left", (BAY_L, BAR_W, BAR_H), (CX, INNER_W / 2 + BAR_W / 2, z_bar), "Bench"),
        uk.collider("col_bar_right", (BAY_L, BAR_W, BAR_H), (CX, -INNER_W / 2 - BAR_W / 2, z_bar), "Bench"),
        uk.collider("col_mat", MAT, (MAT_XY[0], MAT_XY[1], MAT[2] / 2), "Bench"),
    ]
    cols += [uk.collider(f"col_furniture_{k + 1}", (BAY_L, FURN_W, FURN_H), (CX, y, BED + FURN_H / 2), "Bench")
             for k, y in enumerate(FURN_Y)]
    # Written as a rigid body (the loader needs one); the task spawns it without a mass, which makes it kinematic: fixed.
    return uk.write_usda(str(OUT / "bench.usda"), "Bench", parts, looks, cols, mass=8.0, friction=(0.5, 0.4))


def build() -> None:
    for letter in SET_WIDTH:
        _sort(letter)
    _bench()


if __name__ == "__main__":
    build()
    print("wrote", sorted(p.name for p in OUT.glob("*.usda")), "masses", MASS, "bays", [round(y, 5) for y in BAY_Y])
