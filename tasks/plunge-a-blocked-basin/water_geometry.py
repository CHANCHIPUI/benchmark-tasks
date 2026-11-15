"""Water contours and volume from the same parametric bowl as its contact mesh."""
import numpy as np
import usdkit as U
N=64
ANGLES=np.arange(N)*2*np.pi/N
C,S=np.cos(ANGLES),np.sin(ANGLES)

def outline(z, floor=.065, rim=.160):
    edge=.011*((.110*C/.100)**2+(.137*S/.135)**2)**2
    head=np.maximum(0,np.asarray(z)-floor)
    fraction=np.minimum(1,(head[...,None]/edge)**.25)
    inner=np.stack([.110*C,.137*S],axis=-1)
    outer=np.stack([.126*C,.040+.150*S],axis=-1)
    k=np.clip((head[...,None]-edge)/(rim-floor-edge),0,1)
    return inner*fraction[...,None]+k[...,None]*(outer-inner)

LEVELS=np.linspace(.065,.160,1025)
POLYGONS=outline(LEVELS)
AREAS=.5*np.sum(POLYGONS[:,:,0]*np.roll(POLYGONS[:,:,1],-1,axis=1)-POLYGONS[:,:,1]*np.roll(POLYGONS[:,:,0],-1,axis=1),axis=1)
VOLUMES=np.r_[0,np.cumsum((AREAS[1:]+AREAS[:-1])*.5*np.diff(LEVELS))]

def height(volume):return float(np.interp(max(0,volume),VOLUMES,LEVELS))
def volume(height):return float(np.interp(height,LEVELS,VOLUMES))
def mesh(z):
    boundary=outline(z)
    verts=np.vstack([[0,0,0],np.c_[boundary,np.zeros(N)]])
    faces=np.array([(0,1+i,1+(i+1)%N) for i in range(N)])
    return U.Mesh(verts,faces)
