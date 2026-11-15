"""Conservative Euclidean clearance from the waste disk to the whole cup.

The mouth is closed for this test: a hollow cup sitting over a drain obstructs
it even though its rubber rim is 60 mm from the centre. No centre-only shortcut.
"""
import numpy as np

def cup_to_waste(rim, rotation, compression, lip_positions, waste):
    angles=np.arange(128)*2*np.pi/128
    disk=np.vstack([np.array([[0.,0.,0.]]),*[np.c_[r*np.cos(angles),r*np.sin(angles),np.zeros(128)] for r in [.0065,.013,.0195,.026]]])+waste
    local=(disk-rim)@rotation
    points=np.c_[np.linalg.norm(local[:,:2],axis=1),local[:,2]]
    lower=min(0.,float(min(lip_positions)))
    scale=np.clip(1+compression/.055,.2,1.2)
    profile=np.array([(0,lower),(.062,lower),(.062,.010),(.056,.013),(.050,.031),(.044,.049),(.040,.058),(.018,.062),(0,.062)])
    profile[3:,1]=.010+(profile[3:,1]-.010)*scale
    inside=np.ones(len(points),dtype=bool);distance=np.full(len(points),np.inf)
    for a,b in zip(profile,np.roll(profile,-1,axis=0)):
        d=b-a;v=points-a
        inside &= d[0]*v[:,1]-d[1]*v[:,0]>=-1e-12
        t=np.clip(v@d/(d@d),0,1)
        distance=np.minimum(distance,np.linalg.norm(v-t[:,None]*d,axis=1))
    distance[inside]=0
    # 1.7 mm covers the radial/angular sample spacing on the drain disk.
    return max(0.,float(min(distance))-.0017)
