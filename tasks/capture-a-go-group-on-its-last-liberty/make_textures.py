"""Draws the task's textures as image files in textures/ (our own artwork). Run once with any Python that has Pillow and
NumPy: python make_textures.py. The task loads the image files; it does not need Pillow itself.

- board_print.png: the board's top: beech veneer with a golden finish, the 9x9 grid in matte black (5 mm lines, 50 mm
  apart) and five star points. Geometry comes from assets.py. Laid out for usdkit's box projection: columns run with u
  (the board's +x, away from the arms), rows from top to bottom with decreasing v (-y).
- board_side.jpg: the same veneer for the board's edges.
- chestnut.jpg: turned chestnut for the bowls and lids.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import assets as a

HERE = Path(__file__).resolve().parent
OUT = HERE / "textures"
SS = 4  # supersampling
PX = 1024  # the board's 47 cm fills 1,024 px each way
GOLD = np.array([226, 182, 104], float)  # beech veneer with a golden board finish, sRGB
INK = np.array([22, 20, 18], float)


def _noise(shape, scale: float, rng) -> np.ndarray:
    """Smooth value noise in [-1, 1] with features about `scale` pixels across."""
    h, w = shape
    gh, gw = max(2, int(h / scale) + 2), max(2, int(w / scale) + 2)
    g = Image.fromarray((rng.random((gh, gw)) * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return np.asarray(g, float) / 127.5 - 1.0


def grain(tint, h: int, w: int, rng, mm: float, along_rows: bool = True, contrast: float = 1.0) -> np.ndarray:
    """Long straight grain, with the fine flecks of beech, as an (h, w, 3) float image. Grain lines run along rows."""
    px_mm = w / mm
    ys = np.arange(h, dtype=float)
    streak = np.zeros(h)
    for period, amp in ((1.1, 0.03), (3.1, 0.045), (8.0, 0.05), (21.0, 0.03)):
        streak += contrast * amp * np.sin(2 * np.pi * ys / (period * px_mm) + rng.uniform(0, 6.3))
    wav = _noise((h, w), 120, rng) * 8.0
    rows = (ys[:, None] + wav) % h
    tone_s = np.interp(rows, ys, streak)
    tone = np.asarray(tint, float) * (1.0 - tone_s[..., None]) * (1.0 + 0.025 * _noise((h, w), 40, rng)[..., None])
    flecks = (rng.random((h, w)) < 0.0015).astype(float)
    flecks = np.asarray(Image.fromarray((flecks * 255).astype(np.uint8)).resize((w // 4, h)).resize((w, h))
                        .filter(ImageFilter.GaussianBlur(0.7)), float)
    tone *= 1.0 - 0.08 * (flecks / max(1.0, flecks.max()))[..., None]
    return tone if along_rows else tone.transpose(1, 0, 2)


def board_print() -> Image.Image:
    """The board's top. Pixel column c is x = x0 + c / PX * 0.47, pixel row r is y = y1 - r / PX * 0.47."""
    bx, by = a.BOARD[0], a.BOARD[1]
    x0, y1 = a.BOARD_XY[0] - bx / 2, a.BOARD_XY[1] + by / 2
    s = PX / bx * SS

    def px(x: float, y: float) -> tuple[float, float]:
        return (x - x0) * s, (y1 - y) * s

    ink = Image.new("L", (PX * SS, PX * SS), 0)
    d = ImageDraw.Draw(ink)
    xs, ys = list(a.ROW_X.values()), list(a.COL_Y.values())
    h = a.LINE_T / 2
    for x in xs:  # lines across the board (along y), one per row
        (u0, v0), (u1, v1) = px(x - h, max(ys) + h), px(x + h, min(ys) - h)
        d.rectangle([u0, v0, u1, v1], fill=255)
    for y in ys:  # lines along x, one per column
        (u0, v0), (u1, v1) = px(min(xs) - h, y + h), px(max(xs) + h, y - h)
        d.rectangle([u0, v0, u1, v1], fill=255)
    for name in a.STARS:
        x, y = a.point(name)
        (u0, v0), (u1, v1) = px(x - a.STAR_D / 2, y + a.STAR_D / 2), px(x + a.STAR_D / 2, y - a.STAR_D / 2)
        d.ellipse([u0, v0, u1, v1], fill=255)
    mask = np.asarray(ink.resize((PX, PX), Image.LANCZOS), float) / 255.0
    rng = np.random.default_rng(11)
    wood = grain(GOLD, PX, PX, rng, mm=470.0, along_rows=False)  # the veneer's grain runs along x, away from the arms
    printed = INK * (1.0 + 0.05 * _noise((PX, PX), 30, rng)[..., None])
    img = wood * (1 - mask[..., None]) + printed * mask[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    board_print().save(OUT / "board_print.png", optimize=True)
    rng = np.random.default_rng(4)
    Image.fromarray(np.clip(grain(GOLD * 0.93, 512, 512, rng, mm=120.0), 0, 255).astype(np.uint8)).save(
        OUT / "board_side.jpg", quality=90)
    rng = np.random.default_rng(9)
    chestnut = grain((132, 78, 44), 512, 512, rng, mm=80.0, contrast=2.2)
    Image.fromarray(np.clip(chestnut, 0, 255).astype(np.uint8)).save(OUT / "chestnut.jpg", quality=90)
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
