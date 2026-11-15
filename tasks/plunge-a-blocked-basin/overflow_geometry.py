"""A physical circular overflow opening in the sloping ceramic back panel."""
import numpy as np
import usdkit as U

def panels(drain, floor_height, rim):
    theta=np.pi*7/16
    b=np.array([drain[0]+.110*np.cos(theta),drain[1]+.137*np.sin(theta),0.])
    b[2]=floor_height(*b[:2])
    t=np.array([drain[0]+.126*np.cos(theta),.150*np.sin(theta),rim])
    tangent=np.array([0,t[1]-b[1],t[2]-b[2]]);tangent/=np.linalg.norm(tangent)
    normal=np.cross(tangent,[1,0,0])
    centre=np.array([drain[0], b[1]+(.139-b[2])/(t[2]-b[2])*(t[1]-b[1]), .139])
    corners=np.array([[b[0]-centre[0],(b-centre)@tangent],[t[0]-centre[0],(t-centre)@tangent],[-t[0]+centre[0],(t-centre)@tangent],[-b[0]+centre[0],(b-centre)@tangent]])
    angles=sorted(set(np.r_[np.linspace(0,2*np.pi,33)[:-1],np.arctan2(corners[:,1],corners[:,0])%(2*np.pi)].tolist()))
    boundary=[]
    for angle in angles:
        ray=np.array([np.cos(angle),np.sin(angle)])
        for a,c in zip(corners,np.roll(corners,-1,axis=0)):
            scale,along=np.linalg.solve(np.c_[ray,a-c],a)
            if scale>0 and -.000001<=along<=1.000001:
                boundary.append((.015*ray,scale*ray));break
    meshes=[]
    faces=np.array([(0,1,2),(0,2,3),(4,6,5),(4,7,6),(0,4,5),(0,5,1),(1,5,6),(1,6,2),(2,6,7),(2,7,3),(3,7,4),(3,4,0)])
    for p,q in zip(boundary,boundary[1:]+boundary[:1]):
        xy=np.array([p[0],p[1],q[1],q[0]])
        front=centre+xy[:,0,None]*[1,0,0]+xy[:,1,None]*tangent
        meshes.append(U.Mesh(np.vstack([front,front+.008*normal]),faces))
    return meshes,centre,normal
