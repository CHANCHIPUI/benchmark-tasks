"""Writes the task's textures into textures/ (all drawn here, no third-party images):

- pulp.png: grey moulded paper pulp (recycled-paper egg carton), mottled with short fibres, 512 px for a 10 cm tile.
- pine.png: planed pine with its grain along u, 512 px for a 10 cm tile.
- board.png: dark grey painted hardboard, a fine matte stipple, 512 px for a 10 cm tile.

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


def pulp(size: int = 512, seed: int = 5) -> Image.Image:
    rng = np.random.default_rng(seed)
    blotch = _smooth_noise(rng, (size, size), 40)
    mottle = _smooth_noise(rng, (size, size), 8)
    grain = rng.normal(size=(size, size))
    v = 0.66 + 0.035 * blotch + 0.03 * mottle + 0.02 * grain
    img = np.stack([v * 1.0, v * 0.985, v * 0.94], axis=-1)
    # Short fibres: a few hundred dark and light flecks a few pixels long, as recycled paper shows.
    for _ in range(900):
        x, y = rng.integers(0, size, 2)
        ang = rng.uniform(0, np.pi)
        length = rng.integers(3, 10)
        tone = rng.choice([0.45, 0.55, 0.82])
        for t in range(length):
            xi = int(x + t * np.cos(ang)) % size
            yi = int(y + t * np.sin(ang)) % size
            img[yi, xi] = img[yi, xi] * 0.5 + tone * 0.5
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def pine(size: int = 512, seed: int = 7) -> Image.Image:
    rng = np.random.default_rng(seed)
    v = np.arange(size)[:, None] / size
    u = np.arange(size)[None, :] / size
    warp = 0.03 * _smooth_noise(rng, (size, size), 64)
    rings = np.sin(2 * np.pi * (9 * v + 0.04 * np.sin(2 * np.pi * u) + warp * 4))
    late = (np.clip(rings, 0.4, 1.0) - 0.4) / 0.6  # narrow darker latewood lines
    streak = _smooth_noise(rng, (size, size), 16)
    light = np.array([0.86, 0.70, 0.48])
    dark = np.array([0.66, 0.46, 0.25])
    t = np.clip(0.12 + 0.75 * late ** 2 + 0.05 * streak, 0, 1)[..., None]
    rgb = light + (dark - light) * t
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))


def board(size: int = 512, seed: int = 11) -> Image.Image:
    rng = np.random.default_rng(seed)
    v = 0.16 + 0.012 * _smooth_noise(rng, (size, size), 3) + 0.006 * rng.normal(size=(size, size))
    rgb = np.clip(np.stack([v, v, v * 1.02], axis=-1), 0, 1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    pulp().save(OUT / "pulp.png")
    pine().save(OUT / "pine.png")
    board().save(OUT / "board.png")
    print("wrote", sorted(p.name for p in OUT.iterdir()))
