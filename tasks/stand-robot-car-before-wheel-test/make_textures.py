"""Draws the task's textures as image files in textures/ (our own artwork, no third-party images). Run once with any
Python that has Pillow and NumPy: python make_textures.py. The task loads the image files; it does not need Pillow.

- beech.jpg: long-grain beech for the test stand, the grain along u.
- rubber_ribbed.png: the stand's grey ribbed rubber pad, ribs 4 mm apart along u (512 px for a 10 cm tile).
- aluminium.png: brushed aluminium for the carry handle, streaks along u.
- pcb.png: a green circuit board's solder mask with copper traces and white silkscreen marks (no text).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
OUT = HERE / "textures"


def _noise(shape, scale: float, rng) -> np.ndarray:
    """Smooth value noise in [-1, 1] with features about `scale` pixels across."""
    h, w = shape
    gh, gw = max(2, int(h / scale) + 2), max(2, int(w / scale) + 2)
    g = Image.fromarray((rng.random((gh, gw)) * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return np.asarray(g, float) / 127.5 - 1.0


def beech(size: int = 512, seed: int = 4, mm: float = 50.0) -> Image.Image:
    """Long grain with beech's fine ray flecks, running along u."""
    rng = np.random.default_rng(seed)
    px_mm = size / mm
    xs = np.arange(size, dtype=float)
    streak = np.zeros(size)
    for period, amp in ((0.9, 0.035), (2.7, 0.045), (7.0, 0.04)):
        streak += amp * np.sin(2 * np.pi * xs / (period * px_mm) + rng.uniform(0, 6.3))
    wav = _noise((size, size), 90, rng) * 6.0
    cols = (xs[None, :] + wav) % size
    tone_s = np.interp(cols, xs, streak)
    tint = np.array((214, 178, 136), float)
    tone = tint * (1.0 - tone_s[..., None]) * (1.0 + 0.03 * _noise((size, size), 25, rng)[..., None])
    flecks = Image.fromarray(((rng.random((size, size)) < 0.002) * 255).astype(np.uint8))
    flecks = np.asarray(flecks.resize((size, size * 4)).resize((size, size)).filter(ImageFilter.GaussianBlur(0.6)), float)
    tone *= 1.0 - 0.10 * (flecks / max(1.0, flecks.max()))[..., None]
    return Image.fromarray(np.clip(tone, 0, 255).astype(np.uint8)).transpose(Image.Transpose.ROTATE_90)


def rubber_ribbed(size: int = 512, seed: int = 11, tile_mm: float = 100.0, pitch_mm: float = 4.0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u = np.arange(size)[None, :] * tile_mm / size
    rib = 0.5 + 0.5 * np.cos(2 * np.pi * u / pitch_mm)  # ribs across u
    v = 0.30 + 0.06 * rib ** 3 + 0.008 * _noise((size, size), 3, rng) + 0.004 * rng.normal(size=(size, size))
    rgb = np.clip(np.stack([v, v, v * 1.03], axis=-1), 0, 1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def aluminium(size: int = 512, seed: int = 7) -> Image.Image:
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(size, size * 3))
    k = 60
    padded = np.concatenate([noise, noise[:, :k]], axis=1)
    c = np.cumsum(padded, axis=1)
    streaks = (c[:, k:] - c[:, :-k])[:, :size] / np.sqrt(k)
    v = 0.78 + 0.035 * streaks + 0.01 * rng.normal(size=(size, size)) * 0.25
    rgb = np.clip(np.stack([v * 0.98, v * 0.99, v * 1.0], axis=-1), 0, 1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def pcb(size: int = 512, seed: int = 21) -> Image.Image:
    """A 10 cm tile of green solder mask: copper traces, round pads and white component outlines."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (size, size), (22, 92, 48))
    d = ImageDraw.Draw(img)
    trace = (40, 128, 62)
    for _ in range(70):
        x, y = rng.integers(0, size, 2)
        for _ in range(rng.integers(2, 5)):
            if rng.random() < 0.5:
                x2, y2 = x + rng.integers(-120, 120), y
            else:
                x2, y2 = x, y + rng.integers(-120, 120)
            d.line([(x, y), (x2, y2)], fill=trace, width=int(rng.integers(2, 5)))
            x, y = x2, y2
    for _ in range(140):
        x, y = rng.integers(0, size, 2)
        r = int(rng.integers(3, 6))
        d.ellipse([x - r, y - r, x + r, y + r], fill=(196, 168, 92))
    for _ in range(18):
        x, y = rng.integers(10, size - 60, 2)
        w, h = rng.integers(20, 60), rng.integers(10, 40)
        d.rectangle([x, y, x + w, y + h], outline=(225, 230, 225), width=2)
    arr = np.asarray(img, float) * (1.0 + 0.03 * _noise((size, size), 40, rng)[..., None])
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    beech().save(OUT / "beech.jpg", quality=90)
    rubber_ribbed().save(OUT / "rubber_ribbed.png", optimize=True)
    aluminium().save(OUT / "aluminium.png", optimize=True)
    pcb().save(OUT / "pcb.png", optimize=True)
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
