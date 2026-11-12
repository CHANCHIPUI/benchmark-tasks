"""Writes the task's two textures into textures/ (both drawn here, no third-party images):

- brushed_aluminium.png: the modules' brushed aluminium casing, 512 px for a 10 cm tile.
- hv_warning.png: a yellow high-voltage warning triangle (ISO 7010 W012 style) on a white square label. The image is
  mirrored left-right and rolled by half a tile, so that a label face centred on a tile line shows it the right way round.
- lcd_<reading>.png: the multimeter's display showing a DC voltage, in seven-segment digits drawn here with polygons
  (no font), with a "V" and the DC symbol.

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


SEGMENTS = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd", "6": "afgedc", "7": "abc",
            "8": "abcdefg", "9": "abcdfg", " ": ""}
LCD_BG, LCD_INK = (178, 190, 168), (18, 22, 18)
READINGS = (0.0, 25.6, -25.6, 51.2, -51.2, 76.8, -76.8, 102.4, -102.4)


def _segment(d: ImageDraw.ImageDraw, x0: float, y0: float, w: float, h: float, t: float, seg: str) -> None:
    """One bar of a seven-segment digit whose cell starts at (x0, y0), w wide and h tall, bars t thick."""
    m = h / 2
    horiz = {"a": y0, "g": y0 + m, "d": y0 + h}
    vert = {"f": (x0, y0, y0 + m), "b": (x0 + w, y0, y0 + m), "e": (x0, y0 + m, y0 + h), "c": (x0 + w, y0 + m, y0 + h)}
    g = t * 0.15
    if seg in horiz:
        y = horiz[seg]
        a, b = x0 + t / 2 + g, x0 + w - t / 2 - g
        d.polygon([(a, y), (a + t / 2, y - t / 2), (b - t / 2, y - t / 2), (b, y), (b - t / 2, y + t / 2), (a + t / 2, y + t / 2)],
                  fill=LCD_INK)
    else:
        x, ya, yb = vert[seg]
        a, b = ya + t / 2 + g, yb - t / 2 - g
        d.polygon([(x, a), (x + t / 2, a + t / 2), (x + t / 2, b - t / 2), (x, b), (x - t / 2, b - t / 2), (x - t / 2, a + t / 2)],
                  fill=LCD_INK)


def lcd(value: float, size=(512, 256)) -> Image.Image:
    """The display reading `value` volts DC: sign, four digits with one decimal, then V and the DC symbol."""
    img = Image.new("RGB", size, LCD_BG)
    d = ImageDraw.Draw(img)
    text = f"{abs(value):5.1f}".replace(".", "")          # e.g. "  00", " 256", "1024"
    text = text.rjust(4)[-4:]
    if text[2] == " ":
        text = text[:2] + "0" + text[3:]
    w, h, t, top = 52, 150, 15, 52
    xs = [72 + k * 92 for k in range(4)]
    for x, ch in zip(xs, text):
        for seg in SEGMENTS[ch]:
            _segment(d, x, top, w, h, t, seg)
    dot = xs[3] - 20
    d.rectangle([dot - 8, top + h - 8, dot + 8, top + h + 8], fill=LCD_INK)        # decimal point
    if value < 0:
        d.polygon([(10, top + h / 2 - 8), (48, top + h / 2 - 8), (48, top + h / 2 + 8), (10, top + h / 2 + 8)], fill=LCD_INK)
    vx, vy = 430, top + 60
    d.polygon([(vx, vy), (vx + 14, vy), (vx + 32, vy + 66), (vx + 50, vy), (vx + 64, vy), (vx + 40, vy + 90), (vx + 24, vy + 90)],
              fill=LCD_INK)                                                          # V
    d.rectangle([vx, vy - 34, vx + 64, vy - 26], fill=LCD_INK)                        # DC symbol: a line ...
    for k in range(3):                                                               # ... over a dashed line
        d.rectangle([vx + k * 24, vy - 18, vx + k * 24 + 14, vy - 11], fill=LCD_INK)
    return img


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    brushed_aluminium().save(OUT / "brushed_aluminium.png")
    hv_warning().save(OUT / "hv_warning.png")
    for v in READINGS:
        lcd(v).save(OUT / f"lcd_{v:+.1f}.png")
    print("wrote", sorted(p.name for p in OUT.iterdir()))
