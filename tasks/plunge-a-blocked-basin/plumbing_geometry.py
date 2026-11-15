"""Hollow bench drain with an actual side-port opening and inspection chamber."""
import numpy as np
import usdkit as U
OUTLET_Z=.038

def cylinder_wall(radius,z0,z1,inside=False):
    critical=np.arcsin(.016/radius)
    angles=np.unique(np.r_[np.linspace(-np.pi,np.pi,97),-critical,critical])
    vertices=[];faces=[]
    for a,b in zip(angles[:-1],angles[1:]):
        def spans(t):
            y=radius*np.sin(t)
            half=np.sqrt(max(0.,.016**2-y*y)) if np.cos(t)>0 and abs(y)<=.016 else 0.
            return OUTLET_Z-half,OUTLET_Z+half
        la,ha=spans(a);lb,hb=spans(b)
        for lo0,hi0,lo1,hi1 in [(z0,min(z1,la),z0,min(z1,lb)),(max(z0,ha),z1,max(z0,hb),z1)]:
            pts=[[radius*np.cos(a),radius*np.sin(a),lo0],[radius*np.cos(b),radius*np.sin(b),lo1],[radius*np.cos(b),radius*np.sin(b),hi1],[radius*np.cos(a),radius*np.sin(a),hi0]]
            n=len(vertices);vertices.extend(pts)
            ff=[(n,n+1,n+2),(n,n+2,n+3)]
            faces.extend([f[::-1] for f in ff] if inside else ff)
    return U.Mesh(np.array(vertices),np.array(faces))

def chamber():
    outer=cylinder_wall(.025,.015,.054)
    inner=cylinder_wall(.022,.019,.054,True)
    roof=U.lathe([(.016,.054),(.025,.054),(.025,.058),(.016,.058)],64)
    bottom=U.lathe([(0,.015),(.025,.015),(.025,.019),(.022,.019),(0,.019)],64)
    verts=[];faces=[]
    angles=np.linspace(0,2*np.pi,65)
    for a,b in zip(angles[:-1],angles[1:]):
        quad=[]
        for radius,t in [(.025,a),(.022,a),(.022,b),(.025,b)]:
            y=.016*np.cos(t);z=OUTLET_Z+.016*np.sin(t)
            quad.append([np.sqrt(radius**2-y*y),y,z])
        n=len(verts);verts.extend(quad);faces.extend([(n,n+2,n+1),(n,n+3,n+2)])
    return U.join(outer,inner,roof,bottom,U.Mesh(np.array(verts),np.array(faces)))

def pipe(length):
    return U.lathe([(.019,0),(.019,length),(.016,length),(.016,0)],48)
