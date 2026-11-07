"""On the station table, in front of the arms: a 19-inch two-post open-frame rack (8U) clamped to the table, its front
facing the arms. A 2U server fills U1-U2, a 1U switch U3 and a 1U patch panel U8; U4 to U7 are empty. A 1U tool-less
blanking panel stands upright in a slotted stand between the arms and the rack.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
         --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from scenekit import UsdBody, on_reset, on_step  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "Fit the blanking panel at U6."

TOP = g.TOP

# -- objects --------------------------------------------------------------------------------------------------------
RACK = UsdBody("rack", usd="assets/rack.usda", pos=(0.0, 0.0, TOP), size=(0.08, 2 * g.POST_OUT, g.TOP_Z + g.BAR_H))
STAND = UsdBody("stand", usd="assets/stand.usda", pos=(0.0, 0.0, TOP), size=(0.05, 0.4, g.STAND_H))
PANEL = UsdBody("panel", usd="assets/panel.usda", pos=(g.STAND_X, 0.0, TOP + g.PANEL_START_Z),
                size=(g.PANEL_T, g.PANEL_W, g.PANEL_H), mass=g.PANEL_MASS, friction=(0.5, 0.4))
OBJECTS = [RACK, STAND, PANEL]

# -- monitor ---------------------------------------------------------------------------------------------------------
_STATE: dict = {}
HOLES = np.array([g.hole_z(h) for h in range(1, 3 * g.NU + 1)])
CHAIN = ("base", "link1", "link2", "link3", "link4", "link5", "gripper")   # each arm's links, shoulder to wrist
ARM_R = 0.03              # m: each arm link is checked as a capsule of this radius between its joints
GRIPPER_R = 0.025         # m: the gripper body, from the wrist to the tool point
FINGER_R = 0.01           # m: each finger, from its base to its tip
HOLD_STEPS = 20           # control steps (1 s) fitted with the arms clear before it counts as fitted and let go


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(ends={}, clearance=1.0, clear=True, clear_steps=0)


_reset_state()


def _ends(state):
    """Per end of the panel ("left" is the robot's left): the hole its clip is level with (within HOLE_TOL, or None),
    whether the clip is in that hole (within 1.75 mm of its centre across and up), and whether the end is pushed
    home (the clip in its hole and the panel's back against the post: no more than 2 mm in front, 1 mm behind)."""
    R, pos = state.rotation(), np.asarray(state.pos, float)
    out = {}
    for s, side in ((1, "left"), (-1, "right")):
        back = pos + R @ np.array([0.0, s * g.PY, 0.0])              # the back face where this end's clip leaves it
        clip = pos + R @ np.array([g.PEG_L / 2, s * g.PY, 0.0])       # the clip's middle
        z = clip[2] - TOP
        k = int(np.argmin(np.abs(HOLES - z)))
        level = abs(HOLES[k] - z) <= g.HOLE_TOL
        across = clip[1] - s * g.PY
        gap = g.RX - back[0]                                         # the panel's back in front of the post
        in_hole = bool(level and abs(HOLES[k] - z) <= g.CLIP_TOL and abs(across) <= g.CLIP_TOL and gap <= g.PEG_L)
        out[side] = {"hole": k + 1 if level else None, "in_hole": in_hole, "across": float(across), "gap": float(gap),
                     "home": bool(in_hole and -g.BEHIND_TOL <= gap <= g.SEAT_TOL)}
    return out


def _segments(environment):
    """Both arms as capsules (end, end, radius): the links from the base to the wrist, the gripper body to its tool
    point, and each finger from its base out to the fingertips (the finger links' origins are at their bases)."""
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, PAD_TIP_PAST_TCP

    from scenekit import arm_joints, robot_links

    for side, links in robot_links(environment).items():
        chain = [np.asarray(links[n], float) for n in CHAIN if n in links]
        yield from ((a, b, ARM_R) for a, b in zip(chain[:-1], chain[1:]))
        tcp, rot = ik.tcp_pose(arm_joints(environment, side, ARM_JOINTS), station.BASES[side])
        tip = np.asarray(tcp, float) + np.asarray(rot, float)[:, 2] * PAD_TIP_PAST_TCP
        yield chain[-1], np.asarray(tcp, float), GRIPPER_R
        bases = [np.asarray(links[t], float) for t in ("tip_left", "tip_right") if t in links]
        if bases:
            reach = tip - np.mean(bases, axis=0)                     # finger base to fingertip, the same for both
            yield from ((b, b + reach, FINGER_R) for b in bases)


def _clearance(environment, state) -> float:
    """How far the nearest part of either arm is from the panel (its plate and clips, as one box)."""
    R, pos = state.rotation(), np.asarray(state.pos, float)
    lo = np.array([-g.PANEL_T, -g.PANEL_W / 2, -g.PANEL_H / 2])
    hi = np.array([g.PEG_L, g.PANEL_W / 2, g.PANEL_H / 2])
    best = np.inf
    for a, b, r in _segments(environment):
        pts = np.asarray(a, float) + np.linspace(0.0, 1.0, 16)[:, None] * (np.asarray(b, float) - np.asarray(a, float))
        local = (pts - pos) @ R                                      # rows: R.T @ (p - pos)
        d = np.linalg.norm(np.maximum(0.0, np.maximum(lo - local, local - hi)), axis=1) - r
        best = min(best, float(d.min()))
    return best


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    state = object_states(environment, ["panel"])["panel"]
    S = _STATE
    S["ends"] = _ends(state)
    S["clearance"] = _clearance(environment, state)
    S["clear"] = S["clearance"] >= g.CLEAR
    S["clear_steps"] = S["clear_steps"] + 1 if _fitted_at(g.TARGET_HOLE) and S["clear"] else 0


def _fitted_at(hole: int) -> bool:
    e = _STATE["ends"]
    return bool(e) and all(e[s]["hole"] == hole and e[s]["home"] for s in ("left", "right"))


def success(objects) -> bool:
    """The panel at U6: both clips in U6's middle hole and both ends pushed home, with no part of either arm within
    1 cm of it (yam_arena ends the trial once this has held for SUCCESS_HOLD_S)."""
    return _fitted_at(g.TARGET_HOLE) and _STATE["clear"]


def score(objects) -> float:
    """At the end of the trial (success scores 1.0): 1.0 fitted at U6 with the arms clear for the last 1 s; 0.5 fitted
    at U6 but not let go for the last 1 s, or with one end pushed home and the other end's clip lined up in front of
    its hole (level, within 1.75 mm across, no more than 15 mm out); 0 anything else, including one hole off (the
    panel straddling two units) and the two ends in different holes."""
    e = _STATE["ends"]
    if not e:
        return 0.0
    if _fitted_at(g.TARGET_HOLE):
        return 1.0 if _STATE["clear_steps"] >= HOLD_STEPS else 0.5
    home = [s for s in ("left", "right") if e[s]["home"] and e[s]["hole"] == g.TARGET_HOLE]
    if len(home) == 1:
        o = e["right" if home[0] == "left" else "left"]
        if o["hole"] == g.TARGET_HOLE and abs(o["across"]) <= g.CLIP_TOL and -g.BEHIND_TOL <= o["gap"] <= g.OUT_TOL:
            return 0.5
    return 0.0
