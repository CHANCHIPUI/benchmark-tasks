"""Writes the task's two textures into textures/ (both drawn here, no third-party images):

- brushed_aluminium.png: the modules' brushed aluminium casing, 512 px for a 10 cm tile.
- hv_warning.png: a yellow high-voltage warning triangle (ISO 7010 W012 style) on a white square label. The image is
  mirrored left-right and rolled by half a tile, so that a label face centred on a tile line shows it the right way round.

Run once with the harness's Python: python make_textures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent / "textures"


def brushed_aluminium(size: int = 512, seed: int = 7) -> Image.Image:
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(size, size * 3))
    # Long streaks along u: a running mean over 60 px, wrapped so the tile repeats without a seam.
    k = 60
    padded = np.concatenate([noise, noise[:, :k]], axis=1)
    c = np.cumsum(padded, axis=1)
    streaks = (c[:, k:] - c[:, :-k])[:, :size] / np.sqrt(k)
    fine = rng.normal(size=(size, size)) * 0.25
    v = 0.78 + 0.035 * streaks + 0.01 * fine
    rgb = np.clip(np.stack([v * 0.98, v * 0.99, v * 1.0], axis=-1), 0, 1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def hv_warning(size: int = 256) -> Image.Image:
    img = Image.new("RGB", (size, size), (245, 245, 240))
    d = ImageDraw.Draw(img)
    m = size * 0.08
    top, left, right, bottom = (size / 2, m), (m, size - m * 1.6), (size - m, size - m * 1.6), size - m * 1.6
    d.polygon([top, right, left], fill=(20, 20, 20))
    inset = size * 0.075
    d.polygon([(size / 2, m + inset * 1.9), (size - m - inset * 1.7, bottom - inset), (m + inset * 1.7, bottom - inset)],
              fill=(250, 200, 0))
    s = size / 256
    bolt = [(140, 92), (110, 150), (130, 150), (114, 196), (152, 134), (131, 134), (150, 92)]
    d.polygon([(x * s, y * s) for x, y in bolt], fill=(20, 20, 20))
    a = np.asarray(img)[:, ::-1]
    a = np.roll(a, (size // 2, size // 2), axis=(0, 1))
    return Image.fromarray(np.ascontiguousarray(a))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    brushed_aluminium().save(OUT / "brushed_aluminium.png")
    hv_warning().save(OUT / "hv_warning.png")
    print("wrote", sorted(p.name for p in OUT.iterdir()))
