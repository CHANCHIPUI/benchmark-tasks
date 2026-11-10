"""Draws the task's textures as image files in textures/ (our own artwork; the letters are set in Oswald Bold, SIL Open
Font License 1.1, fonts/OFL.txt). Run once with any Python that has Pillow and NumPy: python make_textures.py. The task
loads the image files; it does not need Pillow itself.

- face_<letter>.png: each wood-type sort's printing face, 76.2 mm (head to foot) by the sort's set width. The capital is
  cut mirror-reversed, as all type is, and its raised face carries dried black ink; the routed-down shoulders around it
  are bare end-grain maple. The image is turned so that the box projection in usdkit (u along the sort's x, its head;
  v along its y) shows the letter upright, head toward +x, when seen from above.
- maple_side.jpg: side-grain hard maple for the sorts' sides (the grain runs head-high, since the face is end grain).
- furniture.jpg: the older, darker hardwood of the furniture strips.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
OUT = HERE / "textures"
FONT = HERE / "fonts" / "Oswald.ttf"

BODY_MM = 76.2  # 18-line body, head to foot
CAP_MM = 69.0  # cap height on the face; the shoulders take the rest
PX_PER_MM = 1024 / BODY_MM
SET_WIDTH_MM = {"K": 47.0, "E": 36.0, "L": 36.0, "P": 46.0, "J": 28.0}

MAPLE = np.array([226, 204, 164], float)  # pale end-grain hard maple, sRGB
INK = np.array([24, 22, 24], float)


def _noise(shape, scale: float, rng) -> np.ndarray:
    """Smooth value noise in [-1, 1] with features about `scale` pixels across."""
    h, w = shape
    gh, gw = max(2, int(h / scale) + 2), max(2, int(w / scale) + 2)
    g = Image.fromarray((rng.random((gh, gw)) * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return np.asarray(g, float) / 127.5 - 1.0


def end_grain(h: int, w: int, rng) -> np.ndarray:
    """End-grain maple: faint growth-ring arcs about 1.6 mm apart, fine pores and slow colour drift."""
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    cx, cy = rng.uniform(-3, -2) * w, rng.uniform(-1, 2) * h  # the pith is far off the block
    r = np.hypot(xx - cx, yy - cy) / PX_PER_MM
    rings = 0.5 + 0.5 * np.sin(2 * np.pi * r / 1.6 + _noise((h, w), 120, rng) * 1.5)
    tone = MAPLE * (1.0 - 0.07 * rings[..., None]) * (1.0 + 0.04 * _noise((h, w), 300, rng)[..., None])
    pores = rng.random((h, w)) < 0.004
    tone[pores] *= 0.85
    return tone


def face(letter: str, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    hgt = 1024
    wid = int(round(SET_WIDTH_MM[letter] * PX_PER_MM))
    base = end_grain(hgt, wid, rng)

    # The letter, mirror-reversed, cap height CAP_MM, centred across, its cap line CAP_MM below the head shoulder.
    font = ImageFont.truetype(str(FONT), 1000)
    font.set_variation_by_name("Bold")
    probe = Image.new("L", (1400, 1600), 0)
    ImageDraw.Draw(probe).text((200, 100), "H", font=font, fill=255)
    ys = np.nonzero(np.asarray(probe) > 127)[0]
    cap_units = ys.max() - ys.min() + 1  # cap height at size 1000
    size = int(round(1000 * CAP_MM * PX_PER_MM / cap_units))
    font = ImageFont.truetype(str(FONT), size)
    font.set_variation_by_name("Bold")
    glyph = Image.new("L", (wid * 3, hgt * 2), 0)
    ImageDraw.Draw(glyph).text((wid, hgt // 2), letter, font=font, fill=255)
    g = np.asarray(glyph) > 127
    gy, gx = np.nonzero(g)
    # Cap line: the top of the ink for every capital used here.
    top = gy.min()
    left, right = gx.min(), gx.max()
    shoulder = int(round((BODY_MM - CAP_MM) / 2 * PX_PER_MM))
    crop = glyph.crop((left, top - shoulder, left + (right - left + 1), top - shoulder + hgt))
    letter_img = Image.new("L", (wid, hgt), 0)
    letter_img.paste(crop, ((wid - crop.size[0]) // 2, 0))
    letter_img = ImageOps.mirror(letter_img)  # type is cut reversed
    mask = np.asarray(letter_img, float) / 255.0

    # The routed wall round the raised letter: a narrow band, a little darker, that fades out.
    halo = np.asarray(letter_img.filter(ImageFilter.MaxFilter(17)).filter(ImageFilter.GaussianBlur(7)), float) / 255.0
    halo = np.clip(halo - mask, 0, 1)
    img = base * (1.0 - 0.22 * halo[..., None])

    # Dried ink on the printing face: near black, slightly uneven, a little worn at the edges.
    wear = 0.06 * _noise((hgt, wid), 40, rng) + 0.04 * rng.random((hgt, wid))
    ink = INK[None, None, :] * (1.0 + wear[..., None]) + 10 * np.clip(wear, 0, None)[..., None]
    img = img * (1 - mask[..., None]) + ink * mask[..., None]

    # A thin darker line where the face meets each edge of the block (the cut edge catches dirt).
    edge = np.zeros((hgt, wid))
    e = 4
    edge[:e, :] = edge[-e:, :] = 1
    edge[:, :e] = edge[:, -e:] = 1
    img = img * (1 - 0.18 * edge[..., None])

    a = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))  # as seen from above: head up, the robot's left on the left
    # Texture: columns run with u (the sort's +x, its head), rows top to bottom with decreasing v (the sort's -y).
    return a.transpose(Image.Transpose.ROTATE_270)


def side_grain(tint, size: int = 512, seed: int = 7, mm: float = 50.0, along_u: bool = False) -> Image.Image:
    """Long grain. By default it runs along v (usdkit maps z to v on side faces, so the grain stands upright): the
    streaks vary across columns. With along_u it runs along u (a strip's length along x)."""
    rng = np.random.default_rng(seed)
    px_mm = size / mm
    xs = np.arange(size, dtype=float)
    streak = np.zeros(size)
    for period, amp in ((0.9, 0.05), (2.7, 0.06), (7.0, 0.05)):
        streak += amp * np.sin(2 * np.pi * xs / (period * px_mm) + rng.uniform(0, 6.3))
    wav = _noise((size, size), 90, rng) * 6.0
    cols = (xs[None, :] + wav) % size
    s = np.interp(cols, xs, streak)
    tone = np.asarray(tint, float) * (1.0 - s[..., None]) * (1.0 + 0.03 * _noise((size, size), 25, rng)[..., None])
    img = Image.fromarray(np.clip(tone, 0, 255).astype(np.uint8))
    return img.transpose(Image.Transpose.ROTATE_90) if along_u else img


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for i, letter in enumerate(SET_WIDTH_MM):
        face(letter, 11 + i).save(OUT / f"face_{letter}.png", optimize=True)
    side_grain((222, 196, 150)).save(OUT / "maple_side.jpg", quality=90)
    side_grain((112, 80, 56), seed=3, along_u=True).save(OUT / "furniture.jpg", quality=90)
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
