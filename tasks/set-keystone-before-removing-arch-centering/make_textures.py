"""Draws the task's textures as image files in textures/ (our own artwork, no third-party images). Run once with any
Python that has Pillow and NumPy: python make_textures.py. The task loads the image files; it does not need Pillow.

- sandstone_a/b/c.jpg: blue-grey sandstone, three slightly different stones, with fine grain, darker specks and faint
  bedding lines (a 12 cm tile).
- pine.jpg: planed pine for the centering, the grain along u (15 cm tile).
- beech.jpg: beech for the pull peg (5 cm tile).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
OUT = HERE / "textures"


def _noise(shape, scale: float, rng) -> np.ndarray:
    """Smooth value noise in [-1, 1] with features about `scale` pixels across."""
    h, w = shape
    gh, gw = max(2, int(h / scale) + 2), max(2, int(w / scale) + 2)
    g = Image.fromarray((rng.random((gh, gw)) * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return np.asarray(g, float) / 127.5 - 1.0


def sandstone(seed: int, tint, size: int = 512) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = np.array(tint, float)
    v = (1.0 + 0.10 * _noise((size, size), 120, rng) + 0.06 * _noise((size, size), 30, rng)
         + 0.05 * _noise((size, size), 6, rng) + 0.05 * rng.normal(size=(size, size)))
    rows = np.arange(size)[:, None]
    bed = 0.035 * np.sin(2 * np.pi * (rows + 12 * _noise((size, size), 160, rng)) / rng.uniform(40, 70))
    v = v + bed
    rgb = base[None, None, :] * v[..., None]
    specks = (rng.random((size, size)) < 0.012).astype(float)
    specks = np.asarray(Image.fromarray((specks * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7)), float) / 255
    rgb *= 1.0 - 0.35 * specks[..., None]
    light = (rng.random((size, size)) < 0.006).astype(float)
    light = np.asarray(Image.fromarray((light * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6)), float) / 255
    rgb += 60 * light[..., None]
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def _grain(size: int, seed: int, tint, mm: float, periods, contrast: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    px_mm = size / mm
    xs = np.arange(size, dtype=float)
    streak = np.zeros(size)
    for period, amp in periods:
        streak += amp * np.sin(2 * np.pi * xs / (period * px_mm) + rng.uniform(0, 6.3))
    wav = _noise((size, size), 110, rng) * 9.0
    cols = (xs[None, :] + wav) % size
    tone_s = np.interp(cols, xs, streak)
    tone = np.array(tint, float) * (1.0 - contrast * tone_s[..., None])
    tone *= 1.0 + 0.03 * _noise((size, size), 25, rng)[..., None]
    return tone


def pine(size: int = 512) -> Image.Image:
    """Planed pine: pale with darker latewood lines a few mm apart, running along u."""
    tone = _grain(size, 5, (226, 196, 150), 150.0, ((4.5, 0.6), (11.0, 0.4), (2.0, 0.2)), 0.12)
    return Image.fromarray(np.clip(tone, 0, 255).astype(np.uint8)).transpose(Image.Transpose.ROTATE_90)


def beech(size: int = 512) -> Image.Image:
    tone = _grain(size, 4, (214, 178, 136), 50.0, ((0.9, 0.35), (2.7, 0.45), (7.0, 0.4)), 0.10)
    return Image.fromarray(np.clip(tone, 0, 255).astype(np.uint8)).transpose(Image.Transpose.ROTATE_90)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    sandstone(31, (128, 134, 140)).save(OUT / "sandstone_a.jpg", quality=90)
    sandstone(32, (110, 119, 130)).save(OUT / "sandstone_b.jpg", quality=90)
    sandstone(33, (142, 142, 139)).save(OUT / "sandstone_c.jpg", quality=90)
    pine().save(OUT / "pine.jpg", quality=90)
    beech().save(OUT / "beech.jpg", quality=90)
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
