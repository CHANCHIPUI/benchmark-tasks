"""Offline checks of the scorer on hand-built states (no simulator; the harness's Python): python check_score.py

Each case is a pose; `tier()` from task.py scores the end state with the latches given. The expected scores are those the
task's scoring text gives, including for shortcuts such as N on its side."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
import task as T  # noqa: E402


def rz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rx(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def case(pos, R=None):
    R = np.eye(3) if R is None else R
    pos = np.asarray(pos, float)
    return dict(pos=pos, R=R, v=np.zeros(3), w=np.zeros(3), c=T.corners(pos, R))


def start():
    return {n: case(L.start_pos(n)) for n in L.CASES}


def shelf_z(h=L.CASE[2] / 2):
    return L.H(L.WORK_TOP) + h


def with_room(cs, d1=0.05, d2=0.05):
    for n, d in (("C1", d1), ("C2", -d2)):
        p = cs[n]["pos"].copy()
        p[1] += d
        cs[n] = case(p)
    return cs


def n_in_slot(cs, front=L.FRONT_X, y=0.0, R=None):
    cs["N"] = case((front + L.CASE[0] / 2, y, shelf_z()), R)
    return cs


NO = dict(dropped=False, over_edge=False, out_back=False)


def check():
    out = []

    def run(label, cs, latches=NO, clear=True, still=True, want=None):
        got = T.tier(cs, dict(latches), clear, still)[0]
        ok = want is None or abs(got - want) < 1e-9
        out.append((label, got, want, ok))

    run("start state, nothing done", start(), want=0.0)
    run("room made only", with_room(start()), want=0.25)
    run("full: room made, N in the slot, flush, upright, clear, still", n_in_slot(with_room(start())), want=1.0)
    run("N in, hands not clear", n_in_slot(with_room(start())), clear=False, want=0.5)
    run("N left 5 cm proud of the front", n_in_slot(with_room(start()), front=L.FRONT_X - 0.05), want=0.5)
    on_side = with_room(start())
    on_side["N"] = case((L.FRONT_X + L.CASE[0] / 2, 0.0, L.H(L.WORK_TOP) + L.CASE[1] / 2), rx(90))
    run("room made, N stood on its side in the slot", on_side, want=0.0)
    gap_side = start()
    gap_side["N"] = case((L.FRONT_X + L.CASE[0] / 2, 0.0, L.H(L.WORK_TOP) + L.CASE[1] / 2), rx(90))
    run("no room made, N stood on its side in the 16 cm gap", gap_side, want=0.0)
    run("N upside down in the slot", n_in_slot(with_room(start()), R=rx(180)), want=0.0)
    lean = with_room(start())
    Rl = rx(30)
    lean["N"] = case((L.FRONT_X + L.CASE[0] / 2, 0.0, 0.0), Rl)
    low = lean["N"]["c"][:, 2].min()
    lean["N"] = case((L.FRONT_X + L.CASE[0] / 2, 0.0, L.H(L.WORK_TOP) - low), Rl)
    run("N leaning 30 degrees in the slot", lean, want=0.5)
    held = with_room(start())
    held["N"] = case((L.FRONT_X + L.CASE[0] / 2 - 0.10, 0.0, shelf_z() + 0.03))
    run("room made, N held 3 cm above the work shelf", held, want=0.25)
    top = with_room(start())
    top["N"] = case((L.ROW_X, 0.0, L.H(L.TOP_SHELF) + L.CASE[2] / 2))
    run("room made, N on the top shelf", top, want=0.0)
    on_stack = with_room(start())
    run("room made, N still on the stack", on_stack, want=0.25)
    run("N in, a case went out of the back and was pulled back", n_in_slot(with_room(start())),
        latches=dict(NO, out_back=True), want=0.25)
    moved = n_in_slot(with_room(start()))
    p = moved["S1"]["pos"].copy()
    p[0] -= 0.04
    moved["S1"] = case(p)
    run("N in, S1 pushed 4 cm along the deck", moved, want=0.25)
    run("N in, a neighbour pushed past the deck edge", n_in_slot(with_room(start())), latches=dict(NO, over_edge=True), want=0.0)
    behind = n_in_slot(with_room(start()), front=L.FRONT_X + 0.05)
    run("N set on the shelf 5 cm behind the front-edge line", behind, want=0.0)
    on_top = with_room(start())
    on_top["N"] = case((L.ROW_X, L.ROW["C1"] + 0.05, shelf_z() + L.CASE[2]))
    run("N set on top of C1", on_top, want=0.0)
    run("a stack case knocked off (dropped)", start(), latches=dict(NO, dropped=True), want=0.0)
    run("N in, but not still", n_in_slot(with_room(start())), still=False, want=0.5)
    over = n_in_slot(with_room(start()))
    p3 = over["C3"]["pos"].copy()
    p3[1] = L.SHELF_Y[0] + L.CASE[1] / 2 - 0.03          # C3 3 cm past the shelf's right end, its centre still over it
    over["C3"] = case(p3)
    run("N in, C3 3 cm past the shelf's end (not wholly on it)", over, want=0.5)
    turned = n_in_slot(with_room(start()), R=rz(20))
    run("N in, turned 20 degrees across the row", turned, want=0.0)
    return out


if __name__ == "__main__":
    bad = 0
    for label, got, want, ok in check():
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {got:4.2f} (want {want}) {label}")
    sys.exit(1 if bad else 0)
