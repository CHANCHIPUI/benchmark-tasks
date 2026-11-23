"""dev/overlay.py LOG FRAMES OUT: re-render a demo video in yam-pair-video's layout and encoding, with a status line
for the light and for each station burnt into the overview panel. The status comes from the run's monitor log
(dev/runs/NAME.log.gz or dev/NAME.log): the task's monitor prints the screws' state every 10 steps and each station's
script prints its phase every 2 s; each value holds until the next. Step 0 is left out, because the overview camera is
placed at the reset, after it.

    python dev/overlay.py dev/runs/d43.log.gz dev/logs/d43 media/demo.mp4
"""

from __future__ import annotations

import gzip
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from yam_pair import media

HZ = 20.0
PHASES = {
    "raise": "lifting the light",
    "line up": "lining its slot up under the pilot hole",
    "press": "pressing the light flush",
    "hold while A drives": "holding its end flush while A drives",
    "to slot": "bringing the screw under the slot",
    "into the slot": "raising the screw through the slot",
    "push and drive": "pushing the screw in and driving it",
    "seat": "driving the screw tight",
    "park the driver": "parking its screwdriver, still in hand",
    "hand over the load": "passing its share of the load to its outer hand",
    "inner hand clear": "inner hand letting go of the light",
    "fetch driver": "taking its screwdriver",
    "hold until both are in": "outer hand staying under its end until both screws are in",
}
START = {"a": "taking hold; the inner hand takes its screwdriver", "b": "taking hold of the light"}
STATION = re.compile(r"(?P<s>[ab]): bit (?P<bit>\w+) mag (?P<mag>\w+) push (?P<push>[-\d.]+) pz (?P<pz>[-+\d.]+) .*?"
                     r"mot (?P<mot>\w+) slip \w+ F [-\d]+ clamp (?P<clamp>[-\d]+) gap (?P<gap>[-+\d.]+) .*?"
                     r"tight (?P<tight>\w+)")


def read_log(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    clp, scr, events = {}, {"a": [], "b": []}, []
    with opener(path, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("[clp] step "):
                step = int(line.split()[2])
                hz = float(re.search(r" hz ([-+\d.]+)", line).group(1))
                st = {m["s"]: m.groupdict() for m in STATION.finditer(line)}
                clp[step] = (hz, st)
            elif line.startswith("[clp] event step "):
                m = re.match(r"\[clp\] event step (\d+): (.*)", line)
                events.append((int(m.group(1)), m.group(2)))
            elif line.startswith("[scr] ") and " t " in line and " slot err" in line:
                m = re.match(r"\[scr\] ([ab]) t ([\d.]+) (.*?) slot err", line)
                if m:
                    scr[m.group(1)].append((float(m.group(2)), m.group(3)))
    return clp, scr, events


def light_text(hz: float, step: int, events) -> str:
    if any(s <= step and t.startswith("dropped") for s, t in events):
        return "Light: dropped"
    hang = [s for s, t in events if t.startswith("hangs on its own") and "tier 0.0" not in t]
    if hang and hang[0] - 60 <= step:
        return "Light: hangs on its two screws, no hand on it"
    if hz < -0.095:
        return "Light: on its stands"
    if hz > -0.002:
        return "Light: flush with the ceiling"
    return f"Light: {-hz * 1000:.0f} mm below the ceiling"


def phase_text(s: str, t: float, scr) -> str:
    rows = [(tt, p) for tt, p in scr[s] if tt <= t + 1e-6]
    if not rows:
        return START[s]
    tt, p = rows[-1]
    if t - tt > 2.2 and tt >= scr[s][-1][0]:
        return "has let go of its end"
    return PHASES.get(p, p)


def screw_text(st: dict, prev: dict | None, bite_step: int | None, step: int) -> str:
    """The screw's state at this sample; "driving" if the motor runs now or the head rose since the previous sample
    (the motor cuts in and out between the 0.5 s samples)."""
    bit, mag = st["bit"] == "True", st["mag"] == "True"
    pz, push, gap, clamp = float(st["pz"]), float(st["push"]), float(st["gap"]), float(st["clamp"])
    if not bit:
        if not mag:
            return "screw off the bit"
        if pz <= -60:
            return "screw on the bit"
        if pz < -1.5:
            return f"screw on the bit, tip {-pz:.0f} mm below the pilot hole"
        return f"screw tip on the pilot hole, pushed with {push:.1f} N"
    if bite_step is not None and step - bite_step < 20:
        return "screw has bitten: started in the pilot hole"
    if st["tight"] == "True":
        return f"tight: head on the housing, clamping it with {clamp:.0f} N"
    rose = prev is not None and prev["bit"] == "True" and float(prev["gap"]) - gap > 0.05
    if st["mot"] == "True" or (rose and gap > 0.3):
        return f"driving, screw head {max(gap, 0.0):.1f} mm below the housing"
    if gap > 0.3:
        return f"motor stopped, screw head {gap:.1f} mm below the housing"
    return f"head on the housing, clamping it with {clamp:.0f} N"


def main(log: str, frames: str, out: str) -> None:
    import imageio.v2 as iio

    clp, scr, events = read_log(Path(log))
    bites = {s: next((st for st, t in events if t.startswith(f"{s} screw bit")), None) for s in "ab"}
    trial, cams = media.find_frames(Path(frames))
    steps = [t for t in sorted(set().union(*(set(v) for v in cams.values()))) if t > 0]
    logged = sorted(clp)
    font, small = media._font(14), media._font(16)
    with iio.get_writer(str(out), fps=HZ, codec="libx264", macro_block_size=1, pixelformat="yuv420p",
                        output_params=["-movflags", "+faststart", "-crf", "22"]) as writer:
        for t in steps:
            frames_t = {c: np.load(paths[t]) for c, paths in cams.items() if t in paths}
            grid = media.compose(frames_t, t, HZ, font, True, None)
            prev = [s for s in logged if s <= t]
            hz, st = clp[prev[-1]] if prev else (-0.1, {})
            before = clp[prev[-2]][1] if len(prev) > 1 else {}
            lines = [light_text(hz, t, events)]
            for s in "ab":
                lines.append(f"{s.upper()}: {phase_text(s, t / HZ, scr)}")
                if s in st:
                    lines.append(f"    {screw_text(st[s], before.get(s), bites[s], t)}")
            im = Image.fromarray(grid)
            d = ImageDraw.Draw(im, "RGBA")
            w = max(d.textlength(x, font=small) for x in lines)
            y0 = 30
            d.rectangle([4, y0, 14 + w, y0 + 6 + 21 * len(lines)], fill=(0, 0, 0, 140))
            for i, x in enumerate(lines):
                d.text((9, y0 + 3 + 21 * i), x, font=small, fill=(255, 255, 255))
            writer.append_data(np.asarray(im))
    print(f"{out}: {len(steps)} frames of trial {trial}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
