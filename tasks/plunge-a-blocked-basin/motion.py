"""Cartesian plunger and overflow-cover motions.
Only arm joint actions are produced; object poses and dynamics are never changed.
"""
import math
import os
import numpy as np
import yam_arena.demo as D
from yam_arena.demo import Waypoint

class Grip(float):
    def __new__(cls, yaw, tilt=0.0, roll=0.0):
        obj = float.__new__(cls, yaw)
        obj.tilt = tilt
        obj.roll = roll
        return obj

_down = getattr(D.grip_rotation, 'straight_down', D.grip_rotation)
def rotation(yaw=0.0):
    a = getattr(yaw, 'tilt', 0.0)
    c, s = math.cos(a), math.sin(a)
    r=getattr(yaw,'roll',0.);cr,sr=math.cos(r),math.sin(r)
    return np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]]) @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ _down(float(yaw))
rotation.straight_down = _down
D.grip_rotation = rotation

class Plan:
    def __init__(self, top):
        self.top, self.w = top, []
    def add(self,t,p,f,label,yaw=0.0,tilt=0.0,roll=0.0):
        self.w.append(Waypoint(t,(p[0],p[1],self.top+p[2]),f,label,Grip(yaw,tilt,roll)))
        return self
    @property
    def time(self):return sum(w.duration for w in self.w)
    def wait(self,t):
        w=self.w[-1];self.w.append(Waypoint(t,w.pos,w.finger,'hold',w.yaw));return self
    def until(self,t):
        if t>self.time:self.wait(t-self.time)
