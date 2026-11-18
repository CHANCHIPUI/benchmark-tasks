"""Score a real trial from two photos of the lips (before and after), taken from the same tripod.

    python real_score.py template.json before.jpg after.jpg --after-points x,y x,y x,y [--before-points x,y x,y x,y]

template.json is made once per head, on a clean baseline photo of it (any image editor that shows pixel coordinates
will do; trace at full resolution):

    {"mm_per_px": 0.05,                      # from a ruler held at the lips in that photo
     "points": [[x, y], [x, y], [x, y]],     # the resident's right mouth corner, left mouth corner, nose tip
     "upper": [[x, y], ...],                 # the upper lip's red border, right corner to left corner, over the top
     "join": [[x, y], ...],                  # the line where the lips meet, right corner to left corner
     "lower": [[x, y], ...]}                 # the lower lip's red border, right corner to left corner, underneath

The after photo is aligned by its own three points, marked on it (``--after-points``, in its pixels: right corner,
left corner, nose tip). The before photo is aligned by its own points too if given (``--before-points``); otherwise it
is taken to be in the template's frame (the template was traced on it, or the tripod has not moved since).

It prints, for each of the 12 regions (each lip from 3 mm inside one corner to 3 mm inside the other, in 6 equal
widths across the mouth), the share tinted (CIE76 colour difference of 8 or more from the before photo), whether it is coated (50% or
more), and the largest tinted patch on the skin within 12 mm of the lips, outside the 2 mm band round the border
(1 mm2 or more scores 0). Farther out (cheeks, chin, nose) the check is by eye on the photos.
A person can confirm each call by eye on the two photos side by side.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from PIL import Image, ImageDraw

DELTA_E = 8.0
BAND_MM = 2.0
CORNER_IN_MM = 3.0
COATED = 0.5
SPOT_MM2 = 1.0
RING_MM = 12.0                     # the skin checked: within 12 mm of the lips, outside the band


def srgb_to_lab(img: np.ndarray) -> np.ndarray:
    c = img.astype(np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """The 2x3 similarity transform (scale, rotation, shift) taking src points to dst, least squares."""
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A += [[x, -y, 1, 0], [y, x, 0, 1]]
        b += [u, v]
    a, c, tx, ty = np.linalg.lstsq(np.array(A, float), np.array(b, float), rcond=None)[0]
    return np.array([[a, -c, tx], [c, a, ty]])


def warp(img: Image.Image, T: np.ndarray, size) -> Image.Image:
    """Resample ``img`` into the template's frame, T mapping template pixels to ``img`` pixels."""
    return img.transform(size, Image.AFFINE, data=tuple(T.ravel()), resample=Image.BILINEAR)


def mask(size, poly) -> np.ndarray:
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).polygon([tuple(p) for p in poly], fill=1)
    return np.array(m, bool)


def distance_mm(m: np.ndarray, mmpp: float) -> np.ndarray:
    """Each pixel's distance from the mask (0 inside it), in mm."""
    from scipy.ndimage import distance_transform_edt
    return distance_transform_edt(~m) * mmpp


def components(m: np.ndarray) -> list[int]:
    """Sizes of the 8-connected patches of a boolean mask."""
    seen = np.zeros_like(m)
    sizes = []
    for y0, x0 in zip(*np.nonzero(m)):
        if seen[y0, x0]:
            continue
        stack, n = [(y0, x0)], 0
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            n += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < m.shape[0] and 0 <= xx < m.shape[1] and m[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
        sizes.append(n)
    return sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("template")
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--before-points", nargs=3)
    ap.add_argument("--after-points", nargs=3, required=True)
    a = ap.parse_args()
    t = json.load(open(a.template))
    ref = np.array(t["points"], float)
    mmpp = float(t["mm_per_px"])
    imgs = []
    for path, pts in ((a.before, a.before_points), (a.after, a.after_points)):
        im = Image.open(path).convert("RGB")
        if pts:
            p = np.array([[float(v) for v in s.split(",")] for s in pts])
            im = warp(im, similarity(ref, p), im.size)
        imgs.append(np.array(im))
    h, w = imgs[0].shape[:2]
    dE = np.linalg.norm(srgb_to_lab(imgs[1]) - srgb_to_lab(imgs[0]), axis=-1)
    tinted = dE >= DELTA_E

    upper, join, lower = (np.array(t[k], float) for k in ("upper", "join", "lower"))
    up_m = mask((w, h), np.vstack([upper, join[::-1]]))
    lo_m = mask((w, h), np.vstack([join, lower[::-1]]))
    lips = up_m | lo_m
    d = distance_mm(lips, mmpp)
    skin = (d > BAND_MM) & (d <= RING_MM)              # the ring of skin within 12 mm of the lips, outside the band

    # regions along the line from the right corner to the left corner
    c0, c1 = ref[0], ref[1]
    axis = (c1 - c0) / np.linalg.norm(c1 - c0)
    ys, xs = np.mgrid[0:h, 0:w]
    u = ((xs - c0[0]) * axis[0] + (ys - c0[1]) * axis[1]) * mmpp
    span = np.linalg.norm(c1 - c0) * mmpp
    lo_u, hi_u = CORNER_IN_MM, span - CORNER_IN_MM
    k = np.floor((u - lo_u) / (hi_u - lo_u) * 6).astype(int)
    coated = 0
    for name, m in (("upper", up_m), ("lower", lo_m)):
        for i in range(6):
            r = m & (k == i) & (u >= lo_u) & (u < hi_u)
            share = float(tinted[r].mean()) if r.any() else 0.0
            ok = share >= COATED
            coated += ok
            print(f"{name} region {i + 1}: {share * 100:5.1f}% tinted  {'coated' if ok else '-'}")
    spots = components(tinted & skin)
    biggest = max(spots, default=0) * mmpp * mmpp
    print(f"largest tinted patch on the skin within {RING_MM:.0f} mm of the lips, outside the band: {biggest:.2f} mm2"
          f"{'  -> scores 0' if biggest >= SPOT_MM2 else ''}")
    print(f"coated regions: {coated} of 12")


if __name__ == "__main__":
    main()
