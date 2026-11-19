"""Draws the task's textures into textures/ (run once; needs NumPy and Pillow). All are our own, drawn here.

- knurl.png: the wrench body's diamond knurl (1 mm pitch at the tile size used in tap_usd.py).
- brushed.png: turned and brushed aluminium for the adapter plate.
- hammertone.png: the vise's painted cast iron.
- pla.png: the stand's grey PLA with its 0.2 mm print layers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent / "textures"
rng = np.random.default_rng(7)


def save(name: str, a: np.ndarray) -> None:
    OUT.mkdir(exist_ok=True)
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(OUT / name, optimize=True)


def knurl(n=256, cells=8):
    y, x = np.mgrid[0:n, 0:n] / n * cells
    u, v = (x + y) % 1.0, (x - y) % 1.0
    ridge = np.minimum(np.abs(u - 0.5), np.abs(v - 0.5)) * 2      # 0 in the grooves, 1 on the diamonds' tops
    shade = 0.45 + 0.55 * np.clip(ridge * 1.6, 0, 1) + 0.12 * np.sign(np.sin(2 * np.pi * u))
    g = 150 * shade + rng.normal(0, 6, (n, n))
    return np.dstack([g, g, g * 1.02])


def brushed(n=512):
    streak = rng.normal(0, 1, n)
    streak = np.convolve(streak, np.ones(3) / 3, mode="same")
    a = 205 + 14 * streak[None, :] + rng.normal(0, 3, (n, n))
    return np.dstack([a, a, a * 1.01])


def hammertone(n=256):
    a = rng.normal(0, 1, (n, n))
    k = np.ones((5, 5)) / 25
    from numpy.lib.stride_tricks import sliding_window_view
    p = np.pad(a, 2, mode="wrap")
    a = (sliding_window_view(p, (5, 5)) * k).sum((-1, -2))
    a = 200 + 40 * a / a.std()
    return np.dstack([a * 0.92, a * 0.97, a])


def pla(n=256, layers=32):
    y = np.arange(n)[:, None] / n * layers
    line = 0.88 + 0.12 * np.abs(np.sin(np.pi * y))
    a = 215 * line + rng.normal(0, 3, (n, n))
    return np.dstack([a, a, a])


if __name__ == "__main__":
    save("knurl.png", knurl())
    save("brushed.png", brushed())
    save("hammertone.png", hammertone())
    save("pla.png", pla())
    print(sorted(p.name for p in OUT.iterdir()))
