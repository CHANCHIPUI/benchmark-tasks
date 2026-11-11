"""Writes the task's textures into textures/ (all drawn here; the only third-party file is the Oswald font, OFL 1.1):

- rubber.png: a black rubber mat's fine matte stipple, 512 px for a 10 cm tile.
- enable_label.png: the ENABLE box's printed label, white Oswald capitals on black, 512 x 140 px for 52 x 14 mm.

Run once with the harness's Python: python make_textures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE / "textures"


def _smooth_noise(rng, shape, k: int) -> np.ndarray:
    """Wrapped box-blurred noise along both axes, so the tile repeats without a seam."""
    n = rng.normal(size=shape)
    for axis in (0, 1):
        pad = np.concatenate([n, np.take(n, range(k), axis=axis)], axis=axis)
        c = np.cumsum(pad, axis=axis)
        n = (np.take(c, range(k, k + shape[axis]), axis=axis) - np.take(c, range(0, shape[axis]), axis=axis)) / np.sqrt(k)
    return n


def rubber(size: int = 512, seed: int = 11) -> Image.Image:
    rng = np.random.default_rng(seed)
    v = 0.075 + 0.012 * _smooth_noise(rng, (size, size), 3) + 0.006 * rng.normal(size=(size, size))
    rgb = np.clip(np.stack([v, v, v * 1.03], axis=-1), 0, 1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def label(w: int = 512, h: int = 140) -> Image.Image:
    img = Image.new("RGB", (w, h), (23, 23, 24))
    font = ImageFont.truetype(str(HERE / "fonts" / "Oswald.ttf"), 104)
    try:
        font.set_variation_by_name("Bold")
    except Exception:
        pass
    d = ImageDraw.Draw(img)
    box = d.textbbox((0, 0), "ENABLE", font=font)
    d.text(((w - (box[2] - box[0])) / 2 - box[0], (h - (box[3] - box[1])) / 2 - box[1]), "ENABLE", font=font, fill=(238, 238, 236))
    return img


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    rubber().save(OUT / "rubber.png")
    label().save(OUT / "enable_label.png")
    print("wrote", sorted(p.name for p in OUT.iterdir()))
