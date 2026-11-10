"""Draws the task's two printed labels as PNG files in textures/ (our own artwork, no third-party images). Run once with
any Python that has Pillow: python make_textures.py. The task loads the PNG files; it does not need Pillow itself.

- battery_label.png, battery_end_label.png: the grey label on each battery's large faces (34 x 30 mm) and narrow
  end faces (18 x 30 mm), plain text, no brand.
- box_label.png: the collection box's front label (120 x 60 mm): a battery pictogram and recycling arrows.
Each has a mirrored twin (_m) for faces whose texture coordinates run right to left.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

OUT = Path(__file__).resolve().parent / "textures"
FONT_DIRS = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu"]


def font(size: int, bold: bool = True):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for d in FONT_DIRS:
        p = Path(d) / name
        if p.is_file():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size=size)


def centred(draw, y, text, f, fill):
    w = draw.textlength(text, font=f)
    draw.text(((draw.im.size[0] - w) / 2, y), text, font=f, fill=fill)


def arrows(draw, cx, cy, r, width, fill):
    """Three chasing arrows round a circle (a generic recycling mark)."""
    for k in range(3):
        a0 = math.radians(90 + 120 * k + 12)
        a1 = math.radians(90 + 120 * k + 95)
        box = (cx - r, cy - r, cx + r, cy + r)
        draw.arc(box, math.degrees(-a1), math.degrees(-a0), fill=fill, width=width)
        tip = (cx + r * math.cos(a1 + 0.12), cy - r * math.sin(a1 + 0.12))
        back = a1 - 0.25
        p1 = (cx + (r - width * 1.4) * math.cos(back), cy - (r - width * 1.4) * math.sin(back))
        p2 = (cx + (r + width * 1.4) * math.cos(back), cy - (r + width * 1.4) * math.sin(back))
        draw.polygon([tip, p1, p2], fill=fill)


def battery_label() -> Image.Image:
    w, h = 1020, 900  # 34 x 30 mm at 30 px/mm
    im = Image.new("RGB", (w, h), (150, 152, 156))
    d = ImageDraw.Draw(im)
    ink = (40, 42, 46)
    d.rectangle((0, 0, w - 1, 150), fill=(110, 112, 118))
    centred(d, 30, "Li-ion BATTERY PACK", font(76), (235, 235, 235))
    centred(d, 210, "7.0 V  2280 mAh  16 Wh", font(66), ink)
    centred(d, 310, "Rechargeable lithium-ion", font(50, bold=False), ink)
    d.line((60, 400, w - 60, 400), fill=ink, width=4)
    # crossed-out wheeled bin
    bx, by = 140, 470
    d.rectangle((bx, by + 40, bx + 150, by + 260), outline=ink, width=12)
    d.rectangle((bx - 20, by + 10, bx + 170, by + 40), fill=ink)
    d.line((bx - 40, by - 10, bx + 190, by + 300), fill=ink, width=14)
    d.line((bx + 190, by - 10, bx - 40, by + 300), fill=ink, width=14)
    arrows(d, 520, 600, 120, 22, ink)
    centred_x = 840
    t = "Li-ion"
    tw = d.textlength(t, font=font(60))
    d.text((centred_x - tw / 2, 570), t, font=font(60), fill=ink)
    d.text((80, 800), "Do not short-circuit. Recycle.", font=font(46, bold=False), fill=ink)
    return im


def battery_end_label() -> Image.Image:
    """The narrow end faces (18 x 30 mm): the same grey label wrapping round."""
    w, h = 540, 900
    im = Image.new("RGB", (w, h), (150, 152, 156))
    d = ImageDraw.Draw(im)
    ink = (40, 42, 46)
    d.rectangle((0, 0, w - 1, 150), fill=(110, 112, 118))
    centred(d, 34, "Li-ion", font(80), (235, 235, 235))
    centred(d, 220, "7.0 V", font(96), ink)
    centred(d, 350, "2280 mAh", font(70), ink)
    arrows(d, w // 2, 640, 120, 24, ink)
    return im


def box_label() -> Image.Image:
    w, h = 1024, 512  # 120 x 60 mm
    im = Image.new("RGB", (w, h), (246, 246, 240))
    d = ImageDraw.Draw(im)
    green = (30, 120, 60)
    d.rectangle((0, 0, w - 1, h - 1), outline=green, width=26)
    # battery pictogram
    d.rounded_rectangle((90, 150, 330, 290), radius=14, outline=(25, 25, 25), width=16)
    d.rectangle((330, 190, 362, 250), fill=(25, 25, 25))
    d.text((130, 172), "+  -", font=font(70), fill=(25, 25, 25))
    arrows(d, 600, 225, 105, 22, green)
    d.text((740, 150), "Li", font=font(88), fill=green)
    centred(d, 370, "USED BATTERIES", font(78), (25, 25, 25))
    return im


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, im in (("battery_label", battery_label()), ("battery_end_label", battery_end_label()), ("box_label", box_label())):
        im.save(OUT / f"{name}.png", optimize=True)
        ImageOps.mirror(im).save(OUT / f"{name}_m.png", optimize=True)
        print(name, im.size)


if __name__ == "__main__":
    main()
