"""Rounded 8 x 6 mm rubber sealing bead, split into compliant sectors."""
import numpy as np
import usdkit as U
N=16;R=.056

def mesh(index):
    th=np.linspace((index-.5)*2*np.pi/N,(index+.5)*2*np.pi/N,5)
    phi=np.arange(16)*2*np.pi/16
    v=np.array([[(R+.004*np.sin(p))*np.cos(t),(R+.004*np.sin(p))*np.sin(t),.003-.003*np.cos(p)] for t in th for p in phi])
    f=[]
    for i in range(4):
        for j in range(16):
            a=i*16+j;b=(i+1)*16+j;c=(i+1)*16+(j+1)%16;d=i*16+(j+1)%16
            f.extend([(a,b,c),(a,c,d)])
    for i,reverse in [(0,False),(4,True)]:
        centre=len(v);v=np.vstack([v,[R*np.cos(th[i]),R*np.sin(th[i]),.003]])
        for j in range(16):
            tri=(centre,i*16+j,i*16+(j+1)%16)
            f.append(tri[::-1] if reverse else tri)
    return U.Mesh(v,np.array(f))

# Actual lower-profile vertices, at the middle and both ends of each sector.
# The minimum radial clearance locates the sealing line; the largest angular
# clearance retains any break in that line instead of requiring a flat footprint.
CONTACTS=np.array([[[[(R+.004*np.sin(p))*np.cos(t),(R+.004*np.sin(p))*np.sin(t),.003-.003*np.cos(p)] for p in np.linspace(-np.pi/2,np.pi/2,9)] for t in [(i-.5)*2*np.pi/N,i*2*np.pi/N,(i+.5)*2*np.pi/N]] for i in range(N)])
