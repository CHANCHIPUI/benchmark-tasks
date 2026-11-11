"""Draws the task's textures as image files in textures/ (our own artwork). Run once with any Python that has Pillow and
NumPy: python make_textures.py. The task loads the image files; it does not need Pillow itself.

Images are turned for usdkit's box projection on a top face: columns run with +x, rows from top to bottom with -y.

- fretboard.png: dark rosewood with white pearloid dots (8 mm) in the 3rd, 5th, 7th, 9th, 15th and 17th fret spaces and
  two in the 12th. Geometry from assets.py.
- top.png: the soundboard: spruce, a white binding round the edge, the soundhole with its rosette, and a black pickguard
  on the treble (near) side.
- mahogany.png, rosewood.png, beech.png: wood grain for the neck and sides, the bridge, and the neck rest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import assets as a

HERE = Path(__file__).resolve().parent
OUT = HERE / "textures"
SS = 4


def _noise(shape, scale: float, rng) -> np.ndarray:
    """Smooth value noise in [-1, 1] with features about `scale` pixels across."""
    h, w = shape
    gh, gw = max(2, int(h / scale) + 2), max(2, int(w / scale) + 2)
    g = Image.fromarray((rng.random((gh, gw)) * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return np.asarray(g, float) / 127.5 - 1.0


def grain(size, tint, px_per_mm: float, seed: int, lines_mm: float, contrast: float, wobble: float = 8.0) -> np.ndarray:
    """Wood grain running down the image's rows (along v): an RGB float array."""
    h, w = size
    rng = np.random.default_rng(seed)
    xs = np.arange(w, dtype=float)[None, :] + wobble * _noise((h, w), 120, rng)
    phase = 2 * np.pi * xs / (lines_mm * px_per_mm)
    rings = 0.5 + 0.5 * np.sin(phase + 1.7 * _noise((h, w), 200, rng))
    rings = rings ** 3
    tone = 1.0 - contrast * rings + 0.05 * _noise((h, w), 40, rng) + 0.04 * _noise((h, w), 8, rng)
    return np.asarray(tint, float)[None, None, :] * tone[..., None]


def _save(img: np.ndarray, name: str) -> None:
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(OUT / name, optimize=True)


def fretboard() -> None:
    width_m, length_m = 0.064, a.FB_END - a.NUT_Y + 0.002
    h = 1024
    w = int(round(h * width_m / length_m))
    ppm = h / (length_m * 1000)  # px per mm
    wood = grain((h, w), (52, 30, 22), ppm, seed=3, lines_mm=1.6, contrast=0.45, wobble=3.0)
    # rows: top = far end (FB_END), bottom = the nut; columns: +x
    big = Image.new("L", (w * SS, h * SS), 0)
    d = ImageDraw.Draw(big)

    def px(x: float, y: float) -> tuple[float, float]:
        return (x - (a.X - 0.032)) / width_m * w * SS, (a.FB_END + 0.001 - y) / length_m * h * SS

    r = a.DOT_D / 2
    spots = [(n, 0.0) for n in a.DOT_SPACES] + [(12, -0.0115), (12, 0.0115)]
    for n, off in spots:
        y = a.NUT_Y + 0.5 * (a.fret_d(n - 1) + a.fret_d(n))
        (u0, v0), (u1, v1) = px(a.X + off - r, y + r), px(a.X + off + r, y - r)
        d.ellipse([u0, v0, u1, v1], fill=255)
    mask = np.asarray(big.resize((w, h), Image.LANCZOS), float)[..., None] / 255.0
    rng = np.random.default_rng(11)
    pearl = np.array([236, 232, 222], float)[None, None, :] * (1 + 0.04 * _noise((h, w), 3, rng))[..., None]
    _save(wood * (1 - mask) + pearl * mask, "fretboard.png")


def top() -> None:
    w_m, l_m = 0.40, a.BODY_L
    h = 1024
    w = int(round(h * w_m / l_m))
    ppm = h / (l_m * 1000)
    spruce = grain((h, w), (224, 192, 140), ppm, seed=5, lines_mm=1.4, contrast=0.16, wobble=2.0)
    # the image is drawn with rows along y and columns along x; grain must run along the body (y): rows -> along y
    big = (w * SS, h * SS)

    def px(x: float, y: float) -> tuple[float, float]:
        return (x - (a.X - 0.20)) / w_m * w * SS, (a.BODY_Y0 + l_m - y) / l_m * h * SS

    layers = []  # (mask image, colour)
    # binding: a white line just inside the outline, and a dark purfling line inside it
    outline = a.body_outline(0.0)
    for inset, width_px, col in ((0.0012, 2.4, (238, 232, 218)), (0.0034, 1.2, (40, 26, 20))):
        o = a.body_outline(inset)
        m = Image.new("L", big, 0)
        ImageDraw.Draw(m).line([px(x, y) for x, y in o] + [px(*o[0])], fill=255, width=int(width_px * SS))
        layers.append((m, col))
    cx, cy, rad = a.SOUNDHOLE
    # pickguard: a teardrop on the near (treble) side of the soundhole
    pg = Image.new("L", big, 0)
    t = np.linspace(0, 2 * np.pi, 120)
    gx = cx - 0.046 - 0.040 * (1 + np.cos(t)) * 0.55
    gy = cy + 0.012 + 0.060 * np.sin(t) * (0.75 + 0.25 * np.cos(t))
    ImageDraw.Draw(pg).polygon([px(x, y) for x, y in zip(gx, gy)], fill=255)
    hole_cut = Image.new("L", big, 0)
    ImageDraw.Draw(hole_cut).ellipse([*px(cx - rad - 0.006, cy + rad + 0.006), *px(cx + rad + 0.006, cy - rad - 0.006)], fill=255)
    pg = Image.fromarray(np.minimum(np.asarray(pg), 255 - np.asarray(hole_cut)).astype(np.uint8))
    layers.append((pg, (16, 12, 10)))
    # rosette: rings round the soundhole
    for r_out, r_in, col in ((rad + 0.0115, rad + 0.0095, (30, 22, 18)), (rad + 0.0090, rad + 0.0060, (225, 220, 205)),
                             (rad + 0.0058, rad + 0.0045, (30, 22, 18)), (rad + 0.0040, rad + 0.0030, (225, 220, 205))):
        m = Image.new("L", big, 0)
        dd = ImageDraw.Draw(m)
        dd.ellipse([*px(cx - r_out, cy + r_out), *px(cx + r_out, cy - r_out)], fill=255)
        dd.ellipse([*px(cx - r_in, cy + r_in), *px(cx + r_in, cy - r_in)], fill=0)
        layers.append((m, col))
    hole = Image.new("L", big, 0)
    ImageDraw.Draw(hole).ellipse([*px(cx - rad, cy + rad), *px(cx + rad, cy - rad)], fill=255)
    layers.append((hole, (8, 6, 5)))
    img = spruce
    for m, col in layers:
        mk = np.asarray(m.resize((w, h), Image.LANCZOS), float)[..., None] / 255.0
        img = img * (1 - mk) + np.asarray(col, float)[None, None, :] * mk
    _save(img, "top.png")


def woods() -> None:
    s = 512
    _save(np.transpose(grain((s, s), (120, 58, 36), s / 120, seed=7, lines_mm=2.5, contrast=0.3), (1, 0, 2)), "mahogany.png")
    _save(grain((s, s), (60, 34, 24), s / 80, seed=9, lines_mm=1.5, contrast=0.4), "rosewood.png")
    _save(grain((s, s), (214, 178, 136), s / 60, seed=4, lines_mm=2.0, contrast=0.12), "beech.png")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    fretboard()
    top()
    woods()
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
