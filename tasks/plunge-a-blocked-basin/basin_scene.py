"""Full-size compact washbasin, short plunger and its compliant lip."""
from pathlib import Path
import math
import numpy as np
import usdkit as U
import overflow_geometry as O
import water_geometry as W
import plumbing_geometry as P
import lip_geometry as LIP
HERE=Path(__file__).resolve().parent
CX,CY=.18,0.
FLOOR=.065
CLOG_Z=.056
RIM=.160
DRAIN=np.array([.18,-.040,FLOOR])
OVERFLOW=np.array([.18,.137,.139])
PLUNGER=np.array([.18,-.095,FLOOR+.007])
PAD=np.array([.16,.285,.030])
N_LIP=16
LIP_R=.056
RECEIVER_R=.166
RECEIVER_BOTTOM=.004
LOOKS={
 'ceramic':U.Look((.93,.95,.94),roughness=.13),
 'rubber':U.Look((.08,.23,.31),roughness=.6),
 'red':U.Look((.72,.12,.07),roughness=.4),
 'chrome':U.Look((.65,.68,.70),metallic=.93,roughness=.2),
 'water':U.Look((.28,.68,.79),roughness=.12,opacity=.48),
 'glass':U.Look((.76,.89,.91),roughness=.18,opacity=.20),
 'pipe':U.Look((.84,.84,.82),roughness=.5),
 'dark':U.Look((.08,.07,.06),roughness=.9),
 'mat':U.Look((.17,.23,.25),roughness=.9),
}

def floor_height(x,y):
    # Smoothly rising real bowl: 1.5 mm at r=60 mm near the waste,
    # increasing rapidly toward the walls. No added sealing platform.
    r2=((x-DRAIN[0])/.100)**2+((y-DRAIN[1])/.135)**2
    return FLOOR+.011*r2*r2

def with_flange(z,x,y):
    r=np.hypot(x-DRAIN[0],y-DRAIN[1])
    if .019<=r<=.022:z=max(z,FLOOR+.002*(r-.019)/.003)
    elif .022<r<=.026:z=max(z,FLOOR+.002*(.026-r)/.004)
    return float(z)

def contact_floor(x,y):
    """Height on the actual piecewise planar collision surface."""
    xx=(x-DRAIN[0])/(.110/.137); yy=y-DRAIN[1]
    rr=np.hypot(xx,yy); sector=int(np.arctan2(yy,xx)%(2*np.pi)/(2*np.pi/32))
    j=int(np.clip(np.searchsorted(np.linspace(.021,.137,8),rr)-1,0,6))
    for ids in [(j*32+sector,j*32+(sector+1)%32,(j+1)*32+(sector+1)%32),(j*32+sector,(j+1)*32+(sector+1)%32,(j+1)*32+sector)]:
        tri=_floor_vertices[list(ids)]; A=np.vstack([tri[:,:2].T,np.ones(3)]); weights=np.linalg.solve(A,[x,y,1])
        if min(weights)>-1e-5:return with_flange(weights@tri[:,2],x,y)
    return with_flange(floor_height(x,y),x,y)

def contact_floor_many(points):
    points=np.asarray(points);x,y=points[:,0],points[:,1]
    xx=(x-DRAIN[0])/(.110/.137);yy=y-DRAIN[1]
    rr=np.hypot(xx,yy);sector=(np.arctan2(yy,xx)%(2*np.pi)/(2*np.pi/32)).astype(int)
    j=np.clip(np.searchsorted(np.linspace(.021,.137,8),rr)-1,0,6)
    result=floor_height(x,y)
    for ids in [np.stack([j*32+sector,j*32+(sector+1)%32,(j+1)*32+(sector+1)%32],axis=1),np.stack([j*32+sector,(j+1)*32+(sector+1)%32,(j+1)*32+sector],axis=1)]:
        tri=_floor_vertices[ids];a=tri[:,0,:2];u=tri[:,1,:2]-a;v=tri[:,2,:2]-a;d=points[:,:2]-a
        determinant=u[:,0]*v[:,1]-u[:,1]*v[:,0]
        bu=(d[:,0]*v[:,1]-d[:,1]*v[:,0])/determinant
        bv=(u[:,0]*d[:,1]-u[:,1]*d[:,0])/determinant
        weights=np.stack([1-bu-bv,bu,bv],axis=1)
        inside=np.min(weights,axis=1)>=-1e-5
        result=np.where(inside,np.sum(weights*tri[:,:,2],axis=1),result)
    r=np.hypot(x-DRAIN[0],y-DRAIN[1])
    result=np.where((r>=.019)&(r<=.022),np.maximum(result,FLOOR+.002*(r-.019)/.003),result)
    return np.where((r>.022)&(r<=.026),np.maximum(result,FLOOR+.002*(.026-r)/.004),result)

def water_height(volume):return W.height(volume)
def bowl_volume(z):return W.volume(z)

def header(root):return f'#usda 1.0\n(defaultPrim="{root}"\n metersPerUnit=1\n upAxis="Z")\n'
def mats(root):return '\ndef Scope "Looks" {'+''.join(U._material_usda(n,root,l) for n,l in LOOKS.items())+'\n}'
def mesh(name,m,look,root):return U._mesh_usda(name,m,look,root,LOOKS[look],35)
def collider(name,size,center,root):return U.collider(name,size,center,root).replace('0.004','0.0005')
def meshcol(name,m,root):
    return f'''\ndef Mesh "{name}" (prepend apiSchemas=["PhysicsCollisionAPI","PhysicsMeshCollisionAPI","PhysxCollisionAPI"]) {{
 uniform token purpose="guide"
 uniform token physics:approximation="convexHull"
 float physxCollision:contactOffset=0.0005
 float physxCollision:restOffset=0
 int[] faceVertexCounts=[{','.join('3' for _ in m.f)}]
 int[] faceVertexIndices=[{','.join(str(int(i)) for i in m.f.reshape(-1))}]
 point3f[] points=[{U._fmt(m.v)}]
 }}'''
def body(name,parts,cols,root,mass=.05,art=False):
    api=',"PhysicsArticulationRootAPI","PhysxArticulationAPI"' if art else ''
    extra='bool physxArticulation:enabledSelfCollisions=0\nint physxArticulation:solverPositionIterationCount=64\nint physxArticulation:solverVelocityIterationCount=16' if art else ''
    return f'''\ndef Xform "{name}" (prepend apiSchemas=["PhysicsRigidBodyAPI","PhysicsMassAPI","PhysxRigidBodyAPI"{api}]) {{
 float physics:mass={mass}
 float3 physics:diagonalInertia=({mass*.001},{mass*.001},{mass*.001})
 float physxRigidBody:maxDepenetrationVelocity=1
 float physxRigidBody:sleepThreshold=0
 {extra}
 {''.join(cols)}
 def Xform "visual" {{{''.join(mesh(n,m,l,root) for n,m,l in parts)}}}
 }}'''
def joint(name,b0,b1,axis,low,high,k,d,root):
    return f'''\ndef PhysicsPrismaticJoint "{name}" (prepend apiSchemas=["PhysicsDriveAPI:linear","PhysxJointAPI"]) {{
 uniform token physics:axis="{axis}"
 rel physics:body0=</{root}/{b0}>
 rel physics:body1=</{root}/{b1}>
 point3f physics:localPos0=(0,0,0)
 point3f physics:localPos1=(0,0,0)
 quatf physics:localRot0=(1,0,0,0)
 quatf physics:localRot1=(1,0,0,0)
 float physics:lowerLimit={low}
 float physics:upperLimit={high}
 uniform token drive:linear:physics:type="force"
 float drive:linear:physics:stiffness={k}
 float drive:linear:physics:damping={d}
 float drive:linear:physics:maxForce=120
 float physxJoint:armature=0.00001
 }}'''

def generate_basin():
    root='Basin'
    # Parametric elliptical bowl. Fine contact patches follow the same surface.
    rings=[]
    ns,nr=32,8
    for rr in np.linspace(.021,.137,nr):
        th=np.arange(ns)*2*np.pi/ns
        x=DRAIN[0]+rr*np.cos(th)*(.110/.137)
        y=DRAIN[1]+rr*np.sin(th)
        z=floor_height(x,y)
        rings.append(np.c_[x,y,z])
    v=np.vstack(rings);f=[];cols=[]
    global _floor_vertices
    _floor_vertices=v
    for j in range(nr-1):
        for i in range(ns):
            a=j*ns+i;b=j*ns+(i+1)%ns;c=(j+1)*ns+(i+1)%ns;d=(j+1)*ns+i
            f.extend([(a,b,c),(a,c,d)])
            # Downward extrusion of each small patch yields a convex tile.
            verts=np.vstack([v[[a,b,c,d]],v[[a,b,c,d]]-[0,0,.006]])
            faces=np.array([(0,1,2),(0,2,3),(4,6,5),(4,7,6),(0,4,5),(0,5,1),(1,5,6),(1,6,2),(2,6,7),(2,7,3),(3,7,4),(3,4,0)])
            cols.append(meshcol(f'floor_{j}_{i}',U.Mesh(verts,faces),root))
    parts=[('bowl',U.Mesh(v,np.array(f)[:,::-1]),'ceramic')]
    # Curved outer wall rises from the shallow bowl to its rim.
    for i in range(ns):
        if i in (7,8):continue
        t0,t1=2*np.pi*i/ns,2*np.pi*(i+1)/ns
        # Leave a small real opening for the overflow at the back wall.
        verts=np.array([[DRAIN[0]+.110*np.cos(t),DRAIN[1]+.137*np.sin(t),floor_height(DRAIN[0]+.110*np.cos(t),DRAIN[1]+.137*np.sin(t))] for t in [t0,t1]]+
                       [[CX+.126*np.cos(t),CY+.150*np.sin(t),RIM] for t in [t1,t0]])
        verts=np.vstack([verts,verts+np.array([0,0,-.008])])
        faces=np.array([(0,1,2),(0,2,3),(4,6,5),(4,7,6),(0,4,5),(0,5,1),(1,5,6),(1,6,2),(2,6,7),(2,7,3),(3,7,4),(3,4,0)])
        m=U.Mesh(verts,faces[:,::-1]);parts.append((f'wall{i}',m,'ceramic'));cols.append(meshcol(f'wall{i}',m,root))
    back,centre,normal=O.panels(DRAIN,floor_height,RIM)
    OVERFLOW[:]=centre
    for i,m in enumerate(back):
        parts.append((f'overflow_wall{i}',m,'ceramic'));cols.append(meshcol(f'overflow_wall{i}',m,root))
    parts += [('rim',U.lathe([(.136,RIM-.008),(.140,RIM),(.134,RIM+.004),(.129,RIM)],64).scaled((1,1.25,1)).moved((CX,CY,0)),'ceramic'),
              ('drain',U.lathe([(.022,FLOOR-.002),(.026,FLOOR),(.022,FLOOR+.002),(.019,FLOOR)],48).moved((DRAIN[0],DRAIN[1],0)),'chrome'),
              ('overflow_channel',U.cylinder(.015,.004,axis='y').turned(-32.2,'x').moved(OVERFLOW+normal*.010),'dark'),
              ('tap',U.tube([(CX-.06,.15,RIM),(CX-.06,.15,RIM+.07),(CX-.06,.09,RIM+.07)],.009,12),'chrome')]
    # A normal fixed cross grate protects the waste throat from fingers/handles.
    for i,size in enumerate([(.032,.003,.003),(.003,.032,.003)]):
        c=(DRAIN[0],DRAIN[1],FLOOR-.002)
        parts.append((f'waste_bar{i}',U.box(size,c),'chrome'))
        cols.append(collider(f'waste_bar{i}',size,c,root))
    # The wall-hung basin has a 140 mm back apron. Two 20 mm bench feet
    # support its normal mounting flange while the underside remains open.
    parts.append(('back_apron',U.box((.22,.012,.140),(CX,.157,.090),bevel=.002),'ceramic'))
    cols.append(collider('back_apron',(.22,.012,.140),(CX,.157,.090),root))
    for idx,x in enumerate([CX-.085,CX+.085]):
        p=(x,.145,.010)
        parts.append((f'support{idx}',U.box((.04,.06,.020),p,bevel=.002),'pipe'))
        cols.append(collider(f'support{idx}',(.04,.06,.020),p,root))
    cols=[meshcol('ceramic_contact',U.join(*(m for _,m,_ in parts)),root).replace('physics:approximation="convexHull"','physics:approximation="none"')]
    U.write_usda(str(HERE/'basin.usda'),root,parts,LOOKS,cols,mass=1.)


def generate_plunger():
    root='Plunger'
    blocks=[];joints=[]
    # Root carries the ring's continuous rubber skin. Pads individually yield
    # to bowl curvature, so a central pose alone never creates a seal.
    parts=[('rim_skin',U.lathe([(.052,.005),(.058,.005),(.058,.007),(.052,.010)],64),'rubber'),('dome',U.lathe([(.052,.009),(.056,.013),(.046,.018),(.053,.022),(.043,.027),(.050,.031),(.040,.036),(.047,.040),(.037,.045),(.044,.049),(.034,.054),(.040,.058),(.040,.062),(.018,.062)],48),'red')]
    blocks.append(body('rim',parts,[],root,.025,True))
    parts=[('handle',U.lathe([(0,.052),(.018,.052),(.016,.060),(.010,.070),(.012,.145),(.018,.150),(.018,.160),(0,.160)],48),'red')]
    cols=[meshcol('upper_endplate',U.cylinder(.040,.004,(0,0,.060)),root),collider('shaft',(.024,.024,.100),(0,0,.108),root),meshcol('cap',U.cylinder(.018,.010,(0,0,.155)),root)]
    blocks.append(body('handle',parts,cols,root,.030))
    joints.append(joint('compression','rim','handle','Z',-.035,.006,350,4,root))
    for i in range(N_LIP):
        m=LIP.mesh(i)
        blocks.append(body(f'lip{i}',[(f'pad{i}',m,'rubber')],[meshcol(f'pad{i}',m,root)],root,.0015))
        joints.append(joint(f'lip{i}','rim',f'lip{i}','Z',-.010,.010,250,1.2,root))
    # The moving dome mesh is non-colliding; its shape follows the measured joint.
    dome=U.lathe([(.052,.009),(.056,.013),(.046,.018),(.053,.022),(.043,.027),(.050,.031),(.040,.036),(.047,.040),(.037,.045),(.044,.049),(.034,.054),(.040,.058),(.040,.062),(.018,.062)],48)
    text=header(root)+f'def Xform "{root}" {{'+U.physics_material('PhysicsMaterial',.9,.8)+mats(root)+''.join(blocks)+'def Scope "joints" {'+''.join(joints)+'}'+'}'
    (HERE/'plunger.usda').write_text(text.replace('}def ','}\ndef ').replace('{def ','{\ndef ').replace('}','}\n'))


def generate_plug():
    root='OverflowPlug'
    parts=[('disc',U.cylinder(.025,.007,axis='y'),'rubber'),('ring',U.tube([(0,-.010,.007),(.010,-.014,.016),(0,-.014,.026),(-.010,-.014,.016),(0,-.010,.007)],.0025,8),'chrome')]
    cols=[meshcol('disc',U.cylinder(.025,.007,axis='y'),root),collider('grip',(.022,.007,.016),(0,-.014,.018),root)]
    U.write_usda(str(HERE/'plug.usda'),root,parts,LOOKS,cols,mass=.018,friction=(1.,.9))


def generate_drain(top):
    root='DrainRig';parts=[];cols=[]
    # Centreline elbow and side outlet remain above the station tabletop.
    parts.append(('outlet',P.pipe(.43-(DRAIN[0]+.016)).turned(90,'y').moved((DRAIN[0]+.016,DRAIN[1],P.OUTLET_Z)),'glass'))
    parts.append(('waste_neck',P.pipe(FLOOR-.052).moved((DRAIN[0],DRAIN[1],.052)),'glass'))
    parts.append(('trap_chamber',P.chamber().moved((DRAIN[0],DRAIN[1],0)),'glass'))
    parts.append(('collector',U.lathe([(.170,0),(.170,.018),(.166,.018),(.166,.004),(0,.004),(0,0)],48).moved((.50,-.04,0)),'glass'))
    for i in range(1,6):parts.append((f'mark{i}',U.box((.045,.0015,.0015),(.50,-.208,RECEIVER_BOTTOM+i*.0002/(np.pi*RECEIVER_R**2))),'dark'))
    # Pipe is enclosed to prevent an arm directly pushing the obstruction.
    cols=[meshcol('drain_contact',U.join(*(m for _,m,_ in parts)),root).replace('0.0005','0.0001').replace('physics:approximation="convexHull"','physics:approximation="none"')]
    U.write_usda(str(HERE/'drain.usda'),root,parts,LOOKS,cols,mass=1.)
    root='Blockage'
    th=np.arange(35)*2*np.pi/35
    hairs=[]
    for i in range(14):
        points=[(.002*np.sin(k*.9+i),.015*np.cos(k*.9+i),.004*np.sin(k*.9+i)) for k in range(12)]
        points=np.array(points);points[:,0]+=np.linspace(-.014,.014,len(points))
        hairs.append(U.tube(points,.0012,5))
    plug=U.join(U.cylinder(.0158,.006),*hairs)
    base=body('base',[],[],root,.1,True)
    block=body('plug',[('hair',plug,'dark')],[meshcol('mat_contact',U.cylinder(.0158,.006),root).replace('0.0005','0.00005')],root,.006)
    fixed=f'\ndef PhysicsFixedJoint "fixed" {{\n rel physics:body1=</{root}/base>\n point3f physics:localPos0=({DRAIN[0]},{DRAIN[1]},{top+CLOG_Z})\n}}\n'
    text=header(root)+f'def Xform "{root}" {{'+mats(root)+base+block+fixed+joint('slide','base','plug','Z',-.033,.0015,0,30,root)+'}'
    (HERE/'blockage.usda').write_text(text.replace('}def ','}\ndef ').replace('{def ','{\ndef ').replace('}','}\n'))


def generate_water():
    for name,r,h in [('basinwater',.105,.001),('receiverwater',RECEIVER_R,.001)]:
        U.write_usda(str(HERE/(name+'.usda')),name,[('surface',W.mesh(W.height(.001)) if name=='basinwater' else U.cylinder(r,h),'water')],LOOKS,[],mass=.001)
        if name=='receiverwater':
            path=HERE/(name+'.usda')
            path.write_text(path.read_text().replace('rel material:binding =','token visibility = \"invisible\"\n            rel material:binding ='))

def generate(top=.75):
    generate_basin();generate_plunger();generate_plug();generate_drain(top);generate_water()
