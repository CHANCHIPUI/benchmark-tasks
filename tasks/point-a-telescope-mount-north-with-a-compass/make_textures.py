"""Writes the task's textures into textures/ (all drawn here, no third-party images):

- paint_red.png: a steel toolbox's red enamel, with fine orange-peel, scuffs and small chips worn to bare grey steel,
  512 px for a 16 cm tile.
- brushed.png: brushed aluminium, grain along u, 512 px for a 10 cm tile.
- cast.png: a light grey powder-coated casting's fine stipple, 512 px for a 10 cm tile.

Run once with the harness's Python: python make_textures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent / "textures"


def _smooth_noise(rng, shape, k: int) -> np.ndarray:
    """Wrapped box-blurred noise along both axes, so the tile repeats without a seam."""
    n = rng.normal(size=shape)
    for axis in (0, 1):
        pad = np.concatenate([n, np.take(n, range(k), axis=axis)], axis=axis)
        c = np.cumsum(pad, axis=axis)
        n = (np.take(c, range(k, k + shape[axis]), axis=axis) - np.take(c, range(0, shape[axis]), axis=axis)) / np.sqrt(k)
    return n


def paint_red(size: int = 512, seed: int = 5) -> Image.Image:
    rng = np.random.default_rng(seed)
    peel = 0.035 * _smooth_noise(rng, (size, size), 6)
    scuff = 0.05 * np.clip(_smooth_noise(rng, (size, size), 40), 0, None)
    base = np.array([0.66, 0.08, 0.06])
    rgb = base[None, None, :] * (1.0 + peel[..., None]) + scuff[..., None] * np.array([0.25, 0.2, 0.2])
    # Chips worn through to bare steel: small irregular blobs.
    yy, xx = np.mgrid[0:size, 0:size]
    steel = np.array([0.40, 0.41, 0.43])
    for _ in range(14):
        cx, cy = rng.integers(0, size, 2)
        r = rng.uniform(3.0, 9.0)
        dx = (xx - cx + size // 2) % size - size // 2
        dy = (yy - cy + size // 2) % size - size // 2
        wob = 1 + 0.35 * np.sin(np.arctan2(dy, dx) * rng.integers(3, 6) + rng.uniform(0, 6))
        mask = (dx**2 + dy**2) < (r * wob) ** 2
        rgb[mask] = steel * (0.85 + 0.15 * rng.random()) * (1 + 0.08 * peel[mask][:, None])
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))


def brushed(size: int = 512, seed: int = 8) -> Image.Image:
    rng = np.random.default_rng(seed)
    lines = rng.normal(size=(size, 1)) * 0.03 + _smooth_noise(rng, (size, size), 3) * 0.012
    v = 0.74 + lines
    rgb = np.clip(np.stack([v, v * 1.01, v * 1.03], axis=-1), 0, 1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def cast(size: int = 512, seed: int = 9) -> Image.Image:
    rng = np.random.default_rng(seed)
    v = 0.86 + 0.02 * _smooth_noise(rng, (size, size), 2) + 0.01 * _smooth_noise(rng, (size, size), 12)
    rgb = np.clip(np.stack([v, v, v * 1.02], axis=-1), 0, 1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    paint_red().save(OUT / "paint_red.png")
    brushed().save(OUT / "brushed.png")
    cast().save(OUT / "cast.png")
    print("wrote", sorted(p.name for p in OUT.iterdir()))
