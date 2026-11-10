"""Writes the task's textures into textures/ (all drawn here, no third-party images):

- wood.png, wood_v.png: lacquered mid-brown hardwood for the soroban's frame, 512 px for a 10 cm tile, the grain running
  along u (bars along x) or along v (bars along y).
- rubber.png: a black rubber desk mat's fine matte stipple, 512 px for a 10 cm tile.

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


def wood(size: int = 512, seed: int = 3) -> Image.Image:
    rng = np.random.default_rng(seed)
    v = np.arange(size)[:, None] / size
    u = np.arange(size)[None, :] / size
    warp = 0.035 * _smooth_noise(rng, (size, size), 48)
    # Growth rings seen on a flat-sawn face: bands across v, gently wavy along u (the grain direction).
    rings = np.sin(2 * np.pi * (16 * v + 0.05 * np.sin(2 * np.pi * u) + warp * 3))
    pores = _smooth_noise(rng, (size, size), 2) * 0.5
    streak = _smooth_noise(rng, (size, size), 6)
    shade = 0.5 + 0.18 * rings + 0.05 * streak + 0.03 * pores
    base = np.array([0.50, 0.30, 0.15])
    dark = np.array([0.33, 0.18, 0.08])
    t = np.clip(shade, 0, 1)[..., None]
    rgb = dark + (base - dark) * t * 1.6
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))


def rubber(size: int = 512, seed: int = 11) -> Image.Image:
    rng = np.random.default_rng(seed)
    v = 0.075 + 0.012 * _smooth_noise(rng, (size, size), 3) + 0.006 * rng.normal(size=(size, size))
    rgb = np.clip(np.stack([v, v, v * 1.03], axis=-1), 0, 1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    w = wood()
    w.save(OUT / "wood.png")
    w.transpose(Image.Transpose.TRANSPOSE).save(OUT / "wood_v.png")
    rubber().save(OUT / "rubber.png")
    print("wrote", sorted(p.name for p in OUT.iterdir()))
