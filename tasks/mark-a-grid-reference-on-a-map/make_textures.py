"""Draws the task's textures as image files in textures/ (our own artwork; the grid numbers are set in Oswald Bold, SIL Open
Font License 1.1, fonts/OFL.txt). Run once with any Python that has Pillow and NumPy: python make_textures.py. The task
loads the image files; it does not need Pillow itself.

- map.png: the map board's printed face, an invented 1:10,000 sheet with a 1 km grid (10 cm squares). It is drawn north
  up, as a reader at the near (south) edge sees it, then turned a quarter turn so that usdkit's box projection (u along
  the world's x, v along its y) shows it north toward +x (the far side) and west toward +y (the robot's left).
- hardboard.jpg: the board's brown edges.

The layout numbers come from layout.py, which the scorer uses too.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import layout as L

HERE = Path(__file__).resolve().parent
OUT = HERE / "textures"
FONT = HERE / "fonts" / "Oswald.ttf"

W_PX = 1024  # across the board (its y, 58 cm)
S = W_PX / (L.BOARD_Y[1] - L.BOARD_Y[0]) / 100  # px per cm
H_PX = int(round((L.BOARD_X[1] - L.BOARD_X[0]) * 100 * S))  # along the board (its x)

MARGIN = (244, 243, 236)
PAPER = (248, 244, 229)
GRID = (22, 64, 160)
INK = (22, 22, 24)
CONTOUR = (178, 104, 52)
WOOD = (192, 223, 168)
TREE = (118, 170, 100)
WATER = (172, 214, 238)
WATER_EDGE = (60, 140, 205)
ROAD = (250, 214, 120)
CASING = (120, 110, 100)
BUILDING = (90, 88, 92)


def px(x_cm: float, y_cm: float) -> tuple[float, float]:
    """Board point (cm, world frame) -> (column, row) in the north-up image."""
    return ((L.BOARD_Y[1] * 100 - y_cm) * S, (L.BOARD_X[1] * 100 - x_cm) * S)


def grid_xy():
    """Pixel arrays of the face's world coordinates (cm), for the field-based drawing."""
    rows, cols = np.mgrid[0:H_PX, 0:W_PX].astype(float) + 0.5
    return L.BOARD_X[1] * 100 - rows / S, L.BOARD_Y[1] * 100 - cols / S


def _blend(img: np.ndarray, mask: np.ndarray, color) -> None:
    m = np.clip(mask, 0, 1)[..., None]
    img[:] = img * (1 - m) + np.array(color, float) * m


def terrain(img: np.ndarray) -> None:
    x, y = grid_xy()
    fx0, fx1 = L.FACE_X[0] * 100, L.FACE_X[1] * 100
    fy0, fy1 = L.FACE_Y[0] * 100, L.FACE_Y[1] * 100
    face = (x >= fx0) & (x <= fx1) & (y >= fy0) & (y <= fy1)
    img[face] = PAPER
    # Height field: a hill in the north-east, a lower rise in the west, a valley with a lake in the south-west.
    def bump(cx, cy, sx, sy, h):
        return h * np.exp(-(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))
    hgt = 180 + bump(39, -11, 9, 11, 160) + bump(41, 13, 8, 7, 70) + bump(22, -2, 10, 8, 45) - bump(25, 12, 9, 8, 50)
    hgt += 6 * np.sin(x / 3.1) * np.cos(y / 4.3)
    # Woods (soft-edged blobs) with tree dots.
    wood = np.zeros_like(x)
    for cx, cy, rx, ry in ((40.5, 13.0, 5.5, 4.0), (20.5, -12.5, 4.5, 3.2), (31.0, 3.0, 3.0, 2.2), (45.5, -2.0, 2.6, 3.4)):
        wood = np.maximum(wood, ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1 + 0.12 * np.sin(5 * np.arctan2(y - cy, x - cx)))
    _blend(img, wood * face, WOOD)
    rng = np.random.default_rng(11)
    dots = (rng.random(x.shape) < 0.018) & (wood > 0) & face
    img[dots] = TREE
    # Contours every 20 m, every fifth one thicker, antialiased by their distance in pixels.
    f = hgt / 20.0
    gy, gx = np.gradient(f)
    dist = np.abs(f - np.round(f)) / np.maximum(np.hypot(gx, gy), 1e-6)
    index = np.round(f).astype(int) % 5 == 0
    width = np.where(index, 1.9, 1.0)
    _blend(img, np.clip(width + 0.5 - dist, 0, 1) * face * 0.9, CONTOUR)
    # Lake and its outflow stream.
    lake = ((x - 25.0) / 3.6) ** 2 + ((y - 12.5) / 2.4) ** 2
    _blend(img, (lake <= 1.0) * face, WATER)
    edge = np.abs(np.sqrt(lake) - 1.0) * 2.4 * S
    _blend(img, np.clip(1.2 - edge, 0, 1) * face, WATER_EDGE)
    pil = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    stream = [px(x0, y0) for x0, y0 in ((21.4, 12.0), (19.5, 10.0), (17.5, 9.2), (15.0, 6.5), (13.0, 5.8))]
    d.line(stream, fill=WATER_EDGE, width=max(2, int(0.25 * S)), joint="curve")
    # A minor road (cased yellow) and a track (dashed).
    road = [px(13.0, -4.0), px(19.0, -1.5), px(26.5, -3.5), px(33.5, 0.5), px(41.0, -1.0), px(49.0, 4.5)]
    d.line(road, fill=CASING, width=int(0.55 * S), joint="curve")
    d.line(road, fill=ROAD, width=int(0.32 * S), joint="curve")
    track = [(26.5, -3.5), (29.0, -8.0), (33.0, -12.5), (36.0, -17.0)]
    for (xa, ya), (xb, yb) in zip(track[:-1], track[1:]):
        for t in np.arange(0, 1, 0.16):
            u0, u1 = t, min(1.0, t + 0.09)
            d.line([px(xa + (xb - xa) * u0, ya + (yb - ya) * u0), px(xa + (xb - xa) * u1, ya + (yb - ya) * u1)],
                   fill=CASING, width=max(2, int(0.15 * S)))
    # A farm by the road and a hut on the track.
    for cx, cy, sx, sy in ((19.8, -3.2, 0.9, 1.3), (21.2, -3.0, 0.6, 0.6), (33.3, -12.0, 0.6, 0.6)):
        a, b = px(cx + sx / 2, cy + sy / 2), px(cx - sx / 2, cy - sy / 2)
        d.rectangle([a, b], fill=BUILDING)
    img[:] = np.asarray(pil, float)


def grid_and_numbers() -> Image.Image:
    img = np.zeros((H_PX, W_PX, 3), float) + MARGIN
    terrain(img)
    pil = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    lw = L.LINE_W * 100 * S
    fx0, fx1 = L.FACE_X[0] * 100, L.FACE_X[1] * 100
    fy0, fy1 = L.FACE_Y[0] * 100, L.FACE_Y[1] * 100
    # Neat line round the face.
    d.rectangle([px(fx1, fy1), px(fx0, fy0)], outline=INK, width=max(2, int(0.15 * S)))
    tick = L.TICK * 100
    for e in L.EASTINGS:
        y = L.easting_y(e) * 100
        (c0, r0), (_, r1) = px(fx1 + tick, y), px(fx0 - tick, y)
        d.rectangle([c0 - lw / 2, r0, c0 + lw / 2, r1], fill=GRID)
    for n in L.NORTHINGS:
        x = L.northing_x(n) * 100
        (c0, r0), (c1, _) = px(x, fy1 + tick), px(x, fy0 - tick)
        d.rectangle([c0, r0 - lw / 2, c1, r0 + lw / 2], fill=GRID)
    font = _font(L.DIGIT_H * 100 * S)
    for e in L.EASTINGS:
        y = L.easting_y(e) * 100
        for x in (L.EAST_LABEL_X[0] * 100, L.EAST_LABEL_X[1] * 100):
            _label(d, font, str(e), px(x, y))
    for n in L.NORTHINGS:
        x = L.northing_x(n) * 100
        for y in (L.NORTH_LABEL_Y * 100, -L.NORTH_LABEL_Y * 100):
            _label(d, font, str(n), px(x, y))
    _north_arrow(d)
    return pil


def _font(cap_px: float) -> ImageFont.FreeTypeFont:
    """Oswald Bold sized so that its digits are cap_px tall."""
    probe = ImageFont.truetype(str(FONT), 1000)
    probe.set_variation_by_name("Bold")
    box = probe.getbbox("0123458", anchor="ls")
    size = int(round(1000 * cap_px / (box[3] - box[1])))
    font = ImageFont.truetype(str(FONT), size)
    font.set_variation_by_name("Bold")
    return font


def _label(d: ImageDraw.ImageDraw, font, text: str, at) -> None:
    """Centre the digits' ink (not the font's line box) on `at`."""
    box = d.textbbox((0, 0), text, font=font, anchor="ls")
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    d.text((at[0] - cx, at[1] - cy), text, font=font, fill=INK, anchor="ls")


def _north_arrow(d: ImageDraw.ImageDraw) -> None:
    x, y = L.ARROW_XY[0] * 100, L.ARROW_XY[1] * 100
    tip = px(x + 1.8, y)
    left, right, notch = px(x - 1.8, y + 1.0), px(x - 1.8, y - 1.0), px(x - 1.0, y)
    d.polygon([tip, left, notch], fill=INK)
    d.polygon([tip, notch, right], outline=INK, fill=MARGIN, width=2)
    font = _font(1.5 * S)
    _label(d, font, "N", px(x + 3.0, y))


def hardboard() -> Image.Image:
    rng = np.random.default_rng(5)
    base = np.array([112, 82, 56], float)
    n = rng.normal(0, 1, (256, 256))
    img = Image.fromarray(((n * 6 + 128).clip(0, 255)).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))
    a = (np.asarray(img, float) - 128) / 128
    return Image.fromarray((base * (1 + 0.25 * a[..., None])).clip(0, 255).astype(np.uint8))


def build() -> None:
    OUT.mkdir(exist_ok=True)
    north_up = grid_and_numbers()
    north_up.save(OUT / "map_north_up.png")  # for people to look at; the task uses map.png
    # Quarter turn: texture column <- world x (south to north), texture row from the bottom <- world y (east to west).
    north_up.transpose(Image.Transpose.ROTATE_270).save(OUT / "map.png")
    hardboard().save(OUT / "hardboard.jpg", quality=90)


if __name__ == "__main__":
    build()
    print("map", W_PX, "x", H_PX, "px,", round(S, 2), "px/cm")
