"""Scripted full-score demo for the lock-before-loading task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 3600] [--no-record] [--variant naive]

1. The right arm closes its gripper and presses the raised right end of the platform down until it is level,
   closed loop on the measured hinge angle (it never presses past level: the hand is underneath).
2. The left arm takes the pin from its cradle by the flat handle, lines it up with the holes and pushes it through
   both side walls and the platform's lug, closed loop on the measured pin pose, then lets go.
3. Once the pin locks the hinge, the right arm lifts off. Each arm puts its block on the mark at its end of the
   platform, and both arms return to the ready pose.

``--variant naive`` skips steps 1 and 2 (it loads the platform as it is), to show the scorer's violation.
Writes one MP4 per camera, setup.png (the first top-camera frame) and result.json to the output folder.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

CONTROL_HZ = 20
DT = 1.0 / CONTROL_HZ
DEBUG = bool(os.environ.get("DEMO_DEBUG"))
PAD_BELOW_TCP = 0.013   # yam_arena.constants.PAD_TIP_PAST_TCP: fingertips below the tool point


class Arm:
    """One arm commanded in Cartesian space (IK on the URDF), each move starting where the last command left it."""

    def __init__(self, side: str):
        from yam_arena import ik, station
        from yam_arena.constants import READY

        self.side, self.base = side, np.asarray(station.BASES[side], float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.home = (self.pos.copy(), self.rot.copy())
        self.worst_mm = 0.0

    def aim(self, pos, rot) -> None:
        from yam_arena import ik

        q, err, _ = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), q_init=self.q, base_pos=self.base, iters=80)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), np.asarray(rot, float)
        self.worst_mm = max(self.worst_mm, err * 1000)

    def move(self, pos, rot=None, seconds: float = 1.0, grip: float | None = None):
        """Generator: a smooth straight-line move to ``pos``/``rot``; ``grip`` 1 open .. 0 closed."""
        from yam_arena import ik

        rot = self.rot if rot is None else rot
        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(seconds * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self.aim(p0 + s * (np.asarray(pos, float) - p0), ik.slerp_matrix(r0, rot, s))
            yield

    def set_grip(self, grip: float, seconds: float = 0.8):
        yield from self.move(self.pos, self.rot, seconds, grip)

    def command(self) -> np.ndarray:
        return np.r_[self.q, self.grip]


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def together(*gens):
    """Run generators side by side, one step each per control step."""
    gens = [iter(g) for g in gens]
    while gens:
        for gen in list(gens):
            try:
                next(gen)
            except StopIteration:
                gens.remove(gen)
        if gens:
            yield


class Ctx:
    def __init__(self, emb, task):
        self.emb, self.task = emb, task
        self.left, self.right = Arm("left"), Arm("right")
        self.flags = {"locked": False, "level": False}

    def states(self):
        from yam_arena.scene import object_states

        return object_states(self.emb._env, self.emb.task.names)

    def hinge(self) -> float:
        """The hinge angle the task's monitor measured after the last control step."""
        if "hinge" in self.task._STATE:
            return float(self.task._STATE["hinge"])
        return float(self.states()["platform"].joints["hinge"])

    def pin(self) -> tuple[np.ndarray, np.ndarray]:
        """The pin's handle centre and shaft direction (station frame) after the last control step."""
        S = self.task._STATE
        if "pin_pos" in S:
            return S["pin_pos"], S["pin_axis"]
        from task import _rot

        p = self.states()["pin"]
        return p.pos, _rot(p.quat_xyzw) @ np.array([-1.0, 0.0, 0.0])

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.command(), self.right.command()))


def press_level(ctx, g):
    """Right arm: press the raised end down until the platform is level and hold it there until the pin locks."""
    from yam_arena.demo import grip_rotation

    R = ctx.right
    xy = g.place((g.XC, 0.12, 0.0))[:2]          # 12 cm from the hinge, on the raised (hand) side
    level_tcp = g.PL_TOP + g.DZ + PAD_BELOW_TCP  # tool point with the fingertips on the level plate
    down = grip_rotation(0.0)
    yield from R.move((*xy, 0.893), down, 3.0, grip=0.0)
    z = 0.893
    while not ctx.flags["locked"]:
        hinge = ctx.hinge()
        if hinge > math.radians(0.15):
            z -= 0.008 * DT                        # 8 mm/s down while the raised end is above level
        elif hinge < math.radians(-0.3):
            z += 0.004 * DT
        z = max(z, level_tcp - 0.004)              # never more than 4 mm past level: the hand is underneath
        ctx.flags["level"] = abs(hinge) < math.radians(0.6)
        R.aim((*xy, z), down)
        yield
    yield from R.move(R.pos + [0, 0, 0.04], down, 1.5)
    yield from R.move(R.pos + [0, -0.06, 0.0], down, 1.5)


def _turn(axis, angle: float) -> np.ndarray:
    """Rotation by ``angle`` about the unit vector ``axis``."""
    k = np.asarray(axis, float)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def insert_pin(ctx, g):
    """Left arm: take the pin by the sides of its handle, line it up with the holes, push it home while the
    platform is held level, and let go."""
    from yam_arena.demo import grip_rotation

    L = ctx.left
    side = grip_rotation(0.0)                     # fingers close along y, on the handle's sides
    handle = ctx.pin()[0].copy()
    yield from L.move(handle + [0, 0, 0.06], side, 2.5, grip=1.0)
    yield from L.move(handle, side, 1.5)
    yield from L.set_grip(0.0, 1.0)
    yield from wait(0.4)
    yield from L.move(L.pos + [0, 0, 0.03], side, 1.0)

    line = g.place(g.TAB_IN)                       # handle centre when inserted (station)
    shaft = g.TAB_HALF[0] + g.PIN_LEN              # handle centre to tip
    near_face = g.STAND_XY[0] - (g.CHEEK_X + g.CHEEK_HX)
    start_x = near_face - 0.012 - shaft            # tip 12 mm short of the near wall
    grip_off = L.pos - ctx.pin()[0]                # tool point relative to the handle, as held
    yield from L.move(np.array([start_x, line[1], line[2] + 0.02]) + grip_off, side, 2.5)
    yield from L.move(np.array([start_x, line[1], line[2]]) + grip_off, side, 1.5)
    while not ctx.flags["level"]:                  # the lug's hole lines up only with the platform level
        yield

    # Closed loop on the measured pin: keep the handle on the holes' axis and the shaft parallel to it (wrist yaw
    # and pitch), advance at 1.5 cm/s while the pin keeps up and is lined up, and back off 3 mm when it stalls.
    target_x, corr, yaw, pitch, stalled, locked_for = start_x, np.zeros(2), 0.0, 0.0, 0, 0
    for i in range(int(40 * CONTROL_HZ)):
        p, a = ctx.pin()
        a_yaw, a_pitch = math.atan2(a[1], a[0]), math.asin(max(-1.0, min(1.0, a[2])))
        yaw -= 2.0 * DT * a_yaw
        pitch += 2.0 * DT * a_pitch
        err = np.array([line[1] - p[1], line[2] - p[2]])
        corr += 1.5 * DT * err
        tip_in = p[0] + shaft > near_face + 0.004
        lined_up = np.abs(err).max() < 0.0015 and max(abs(a_yaw), abs(a_pitch)) < math.radians(0.6)
        if p[0] > target_x - 0.004:
            stalled = 0
            if tip_in or lined_up:
                target_x = min(target_x + 0.015 * DT, line[0])
        else:
            stalled += 1
            if stalled > CONTROL_HZ:
                target_x, stalled = target_x - 0.003, 0
        rot = _turn([0, 0, 1], yaw) @ _turn([0, 1, 0], pitch) @ side
        L.aim(np.array([target_x, line[1] + corr[0], line[2] + corr[1]]) + grip_off, rot)
        if DEBUG and i % 10 == 0:
            print(f"insert handle {np.round(p, 4)} err_mm {np.round(err * 1000, 1)} yaw {math.degrees(a_yaw):.2f} "
                  f"pitch {math.degrees(a_pitch):.2f} target_x {target_x:.4f} tip_in {tip_in} "
                  f"locked {ctx.task._STATE['locked']} hinge {math.degrees(ctx.hinge()):.2f}", flush=True)
        locked_for = locked_for + 1 if ctx.task._STATE["locked"] else 0
        if locked_for and (p[0] > line[0] - 0.002 or locked_for > 1.5 * CONTROL_HZ):
            break
        yield
    yield from wait(0.5)
    yield from L.set_grip(1.0, 0.8)
    yield from L.move(L.pos + [0, 0, 0.04], L.rot, 1.0)
    yield from L.move(L.pos + [-0.06, 0, 0], L.rot, 1.2)
    ctx.flags["locked"] = bool(ctx.task._STATE["locked"])


def place_block(arm, ctx, g, name, mark_y_build):
    from yam_arena.demo import grip_rotation

    down = grip_rotation(0.0)
    block = ctx.states()[name].pos.copy()
    mark = g.place((g.XC, mark_y_build, g.PL_TOP + g.BLK + 0.004))
    yield from arm.move(block + [0, 0, 0.10], down, 2.5, grip=1.0)
    yield from arm.move(block, down, 1.5)
    yield from arm.set_grip(0.0, 1.0)
    yield from wait(0.3)
    yield from arm.move(block + [0, 0, 0.12], down, 1.5)
    yield from arm.move((mark[0], mark[1], 0.888), down, 2.5)
    yield from arm.move(mark, down, 1.5)
    yield from arm.set_grip(1.0, 1.0)
    yield from wait(0.3)
    yield from arm.move((mark[0], mark[1], 0.888), down, 1.2)


def go_home(arm):
    yield from arm.move(arm.home[0], arm.home[1], 2.5, grip=1.0)


def script(ctx, g, variant: str):
    if variant != "naive":
        yield from together(press_level(ctx, g), insert_pin(ctx, g))
    yield from together(place_block(ctx.left, ctx, g, "orange_block", -g.MARK_Y),
                        place_block(ctx.right, ctx, g, "blue_block", g.MARK_Y))
    yield from together(go_home(ctx.left), go_home(ctx.right))


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 3600, record: bool = True, variant: str = "",
        cam_width: int = 320, cam_height: int = 176, hold_s: float = 4.0) -> dict:
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    task = emb.task.module
    sys.modules.setdefault("task", task)
    import geometry as g

    obs = emb.reset(Scene(id="demo", instruction=task.INSTRUCTION))
    ctx = Ctx(emb, task)
    gen = script(ctx, g, variant)
    frames: dict[str, list] = {}

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    snap(obs)
    steps, ended, success, info = 0, None, False, {}
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(ctx.action())
        steps += 1
        info = res.info
        snap(res.observation)
        if steps % 100 == 0:
            S = task._STATE
            print(f"step {steps} score {info.get('score')} hinge {math.degrees(S.get('hinge', 0)):.2f} "
                  f"locked {S['locked']} on_mark {S['on_mark']} clear {S['clear']} opening {S.get('opening')} "
                  f"why {S['why']}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    S = task._STATE
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "violations": list(S["why"]),
              "locked": bool(S["locked"]), "on_mark": dict(S["on_mark"]),
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)}}
    if record:
        import imageio

        for cam, fs in frames.items():
            _write_mp4(fs, out / f"demo_{cam}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=3600)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--variant", default="", choices=["", "naive"])
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=352)
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record, variant=a.variant, cam_width=a.cam_width,
        cam_height=a.cam_height)
