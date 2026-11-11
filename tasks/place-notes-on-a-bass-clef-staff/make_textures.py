"""Draws the task's textures as image files in textures/ (our own artwork; the bass clef is the F-clef glyph of Bravura,
SIL Open Font License 1.1, fonts/OFL.txt). Run once with any Python that has Pillow and NumPy: python make_textures.py.
The task loads the image files; it does not need Pillow itself.

- staff_print.png: the staff board's printed top: five staff lines, the bass clef at the left end, four bar lines and a
  final double bar, in matte black on white melamine. Geometry comes from assets.py. The image is turned for usdkit's box
  projection: columns run with u (the board's +x, away from the arms), rows top to bottom with decreasing v (-y).
- beech.jpg: long-grain beech for the trays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import assets as a

HERE = Path(__file__).resolve().parent
OUT = HERE / "textures"
FONT = HERE / "fonts" / "Bravura.otf"
F_CLEF = ""  # SMuFL fClef: origin on the F line, em = 4 staff spaces

SS = 4  # supersampling
PX_PER_M = 1024 / a.BOARD[1]  # the board's 62 cm along y fills 1,024 px
WHITE = np.array([243, 243, 238], float)  # white melamine, sRGB
INK = np.array([18, 18, 20], float)


def _noise(shape, scale: float, rng) -> np.ndarray:
    """Smooth value noise in [-1, 1] with features about `scale` pixels across."""
    h, w = shape
    gh, gw = max(2, int(h / scale) + 2), max(2, int(w / scale) + 2)
    g = Image.fromarray((rng.random((gh, gw)) * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return np.asarray(g, float) / 127.5 - 1.0


def staff_print() -> Image.Image:
    """The board seen from above, as the arms see it: the robot's left (+y) on the left, away (+x) at the top."""
    bx, by = a.BOARD[0], a.BOARD[1]
    x0, y1 = a.BOARD_XY[0] - bx / 2, a.BOARD_XY[1] + by / 2  # the image's top-left corner is (far x, left y)
    w, h = int(round(by * PX_PER_M)), int(round(bx * PX_PER_M))
    s = PX_PER_M * SS

    def px(x: float, y: float) -> tuple[float, float]:
        return (y1 - y) * s, (x0 + bx - x) * s

    ink = Image.new("L", (w * SS, h * SS), 0)
    d = ImageDraw.Draw(ink)
    half = a.LINE_T / 2
    for x in a.LINE_X:  # staff lines
        (u0, v0), (u1, v1) = px(x + half, a.STAFF_Y[0]), px(x - half, a.STAFF_Y[1])
        d.rectangle([u0, v0, u1, v1], fill=255)
    top, bottom = a.LINE_X[-1] + half, a.LINE_X[0] - half
    for y in a.BAR_Y:  # bar lines, from line 1 to line 5
        (u0, v0), (u1, v1) = px(top, y + a.BAR_T / 2), px(bottom, y - a.BAR_T / 2)
        d.rectangle([u0, v0, u1, v1], fill=255)
    for y, t in ((a.FINAL_THIN_Y, a.FINAL_THIN_T), (a.FINAL_THICK_Y, a.FINAL_THICK_T)):  # final double bar
        (u0, v0), (u1, v1) = px(top, y + t / 2), px(bottom, y - t / 2)
        d.rectangle([u0, v0, u1, v1], fill=255)

    # The bass clef: Bravura's em is four staff spaces, and the glyph's origin sits on the F line (line 4).
    size = int(round(4 * a.SPACE * s))
    font = ImageFont.truetype(str(FONT), size)
    ox, oy = px(a.LINE_X[3], a.CLEF_LEFT_Y)
    d.text((ox, oy), F_CLEF, font=font, fill=255, anchor="ls")

    mask = np.asarray(ink.resize((w, h), Image.LANCZOS), float) / 255.0
    rng = np.random.default_rng(5)
    board = WHITE * (1.0 + 0.012 * _noise((h, w), 60, rng)[..., None])  # faint unevenness of the melamine
    printed = INK * (1.0 + 0.05 * _noise((h, w), 30, rng)[..., None])
    img = board * (1 - mask[..., None]) + printed * mask[..., None]
    pic = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))  # as seen from above, the arms at the bottom
    return pic.transpose(Image.Transpose.ROTATE_270)  # columns along +x, rows along -y (usdkit's box projection)


def side_grain(tint, size: int = 512, seed: int = 7, mm: float = 50.0, along_u: bool = False) -> Image.Image:
    """Long grain with beech's fine flecks. By default it runs along v (usdkit maps z to v on side faces); with along_u
    it runs along u (a part's length along x)."""
    rng = np.random.default_rng(seed)
    px_mm = size / mm
    xs = np.arange(size, dtype=float)
    streak = np.zeros(size)
    for period, amp in ((0.9, 0.035), (2.7, 0.045), (7.0, 0.04)):
        streak += amp * np.sin(2 * np.pi * xs / (period * px_mm) + rng.uniform(0, 6.3))
    wav = _noise((size, size), 90, rng) * 6.0
    cols = (xs[None, :] + wav) % size
    tone_s = np.interp(cols, xs, streak)
    tone = np.asarray(tint, float) * (1.0 - tone_s[..., None]) * (1.0 + 0.03 * _noise((size, size), 25, rng)[..., None])
    flecks = Image.fromarray(((rng.random((size, size)) < 0.002) * 255).astype(np.uint8))
    flecks = np.asarray(flecks.resize((size, size * 4)).resize((size, size)).filter(ImageFilter.GaussianBlur(0.6)), float)
    tone *= 1.0 - 0.10 * (flecks / max(1.0, flecks.max()))[..., None]  # short ray flecks, typical of beech
    img = Image.fromarray(np.clip(tone, 0, 255).astype(np.uint8))
    return img.transpose(Image.Transpose.ROTATE_90) if along_u else img


def main() -> None:
    OUT.mkdir(exist_ok=True)
    staff_print().save(OUT / "staff_print.png", optimize=True)
    side_grain((214, 178, 136), seed=4, along_u=True).save(OUT / "beech.jpg", quality=90)
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
