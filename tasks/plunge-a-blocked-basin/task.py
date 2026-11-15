"""A full-size compact washbasin holds standing water over a blocked waste.
A short rubber sink plunger stands off-centre in the bowl. A rubber basin plug
with a pull ring lies nearby; the basin has an open overflow. The water and
clear outlet make drainage visible. The tap is off and no chemicals are used.
"""
from __future__ import annotations
import os,sys,pathlib,json,math
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import steptime
import scenekit as K
import basin_scene as S
import hydraulics as H
from clearance import cup_to_waste
from visuals import Visuals
import coupling
from yam_arena.task_file import TABLE_TOP
from yam_arena.demo import Waypoint
SCENE='station';TOP=TABLE_TOP[SCENE]
S.generate(TOP)
BASIN=K.UsdBody('basin',pos=(0,0,TOP),size=(.28,.35,.16),usd='basin.usda',friction=(.45,.30))
PLUNGER=K.UsdBody('plunger',pos=tuple(S.PLUNGER+[0,0,TOP]),size=(.12,.12,.16),usd='plunger.usda',mass=.079,articulation=True,
 joint_pos=tuple([('compression',0.)]+[(f'lip{i}',0.) for i in range(S.N_LIP)]),drives=(('compression',350.,4.,120.),('lip.*',250.,1.2,20.)),friction=(.9,.8),solver_iters=64,torsional_patch=.006)
PLUG=K.UsdBody('overflow_plug',pos=tuple(S.PAD+[0,0,TOP]),size=(.05,.025,.05),usd='plug.usda',mass=.018,friction=(1.,.9),solver_iters=32,torsional_patch=.006)
DRAIN=K.UsdBody('outlet',pos=(0,0,TOP),size=(.4,.18,.12),usd='drain.usda')
CLOG=K.UsdBody('clog',pos=(S.DRAIN[0],S.DRAIN[1],TOP+S.CLOG_Z),size=(.032,.032,.008),usd='blockage.usda',articulation=True,joint_pos=(('slide',0.),),drives=(('slide',0.,30.,20.),),solver_iters=32)
WATER=K.UsdBody('basin_water',pos=(S.DRAIN[0],S.DRAIN[1],TOP+S.water_height(.001)),size=(.21,.21,.001),usd='basinwater.usda')
RECEIVER=K.UsdBody('receiver_water',pos=(.50,-.04,TOP+S.RECEIVER_BOTTOM),size=(.332,.332,.001),usd='receiverwater.usda')
OBJECTS=[BASIN,PLUNGER,PLUG,DRAIN,CLOG,WATER,RECEIVER]
_state={}

def _array(a):return getattr(a,'torch',a).detach().cpu().numpy()
def rot(q):
 x,y,z,w=q
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def cup_region(height,rim,compression):
 # Integrate the authored cup section; below its mouth use the exact quartic
 # mean of the bowl profile over the 112 mm sealing disk. Clipping at the mean
 # rather than each floor triangle is the documented small-volume approximation.
 a,b=rim[0]-S.DRAIN[0],rim[1]-S.DRAIN[1];r=.056
 ex4=a**4+1.5*a*a*r*r+r**4/8
 ey4=b**4+1.5*b*b*r*r+r**4/8
 ex2y2=a*a*b*b+(a*a+b*b)*r*r/4+r**4/24
 floor_mean=S.FLOOR+.011*(ex4/.100**4+2*ex2y2/(.100**2*.135**2)+ey4/.135**4)
 rim_height=rim[2]-TOP
 gap=H.MOUTH_A*max(0.,min(height,rim_height)-floor_mean)
 return gap+H.cup_profile_volume(max(0.,height-rim_height),compression)

DROP_POINTS=np.array([[.056*np.sqrt((i+.5)/8)*np.cos(j*2*np.pi/32),.056*np.sqrt((i+.5)/8)*np.sin(j*2*np.pi/32),0.] for i in range(8) for j in range(32)])
def liquid_return_fractions(rim,rotation):
 points=DROP_POINTS@rotation.T+rim
 basin=((points[:,0]-S.CX)/.129)**2+((points[:,1]-S.CY)/(.129*1.25))**2<=1.
 receiver=((points[:,0]-.50)**2+(points[:,1]+.04)**2<=S.RECEIVER_R**2)&~basin
 return float(np.mean(basin)),float(np.mean(receiver))

def outside_height(water,rim,compression):
 if not water.cup_active:return S.water_height(water.volume)
 if water.volume<=1e-9:return S.FLOOR
 lo,hi=S.FLOOR,S.RIM
 for _ in range(24):
  mid=(lo+hi)/2
  if S.bowl_volume(mid)-cup_region(mid,rim,compression)>water.volume:hi=mid
  else:lo=mid
 return (lo+hi)/2

@K.before_reset
def before_reset(e):
 _state.clear()
 for name in ['plunger','overflow_plug']:
  try:e.unwrapped.scene[name].permanent_wrench_composer.reset()
  except (KeyError,AttributeError):pass
 for n,js in [('plunger',dict(PLUNGER.joint_pos)),('clog',{'slide':0.})]:
  try:
   K.write_joint_positions(e,n,js)
   if n=='clog':
    import torch
    a=e.unwrapped.scene[n]
    a.actuators.target_command.set_effort_index(value=torch.tensor([[H.PLUG_MASS*H.G]],device=a.device),joint_ids=[0])
    a.write_joint_damping_to_sim_index(damping=torch.tensor([[H.PLUG_DAMPING]],device=a.device),joint_ids=[0])
  except (KeyError,AttributeError):pass

@K.on_reset
def reset(e):
 _state.update(water=H.Water(),step=0,physics_step=0,last_velocity=None,last_physics_stamp=None,last_structure=None,pending=None,last_force_diff=0.,pressure_error_peak=0.,pressure_work=0.,air_energy_change=0.,network_work=0.,air_residual_total=0.,air_residual_peak=0.,air_input_total=0.,failed=False,clear_hold=0.,sealed=False,cover=0.,pressure_force=0.)
 _state['ids']={n:i for i,n in enumerate(e.unwrapped.scene['plunger'].body_names)}
 _state['clog_open']=False
 _state['trace']=[]
 _state['substeps']=[]
 _state['snapshot']=None
 _state['visuals']=Visuals()
 print('PLUNGER_BODIES',_state['ids'])
 print('CLOG_POSES',_array(e.unwrapped.scene['clog'].data.body_pos_w).tolist())
 scene=e.unwrapped.scene
 if not getattr(scene,'_basin_pressure_hook',False):
  original_write=scene.write_data_to_sim
  def write_with_hydraulics(*args,**kwargs):
   physics_update(e,float(e.unwrapped.physics_dt))
   return original_write(*args,**kwargs)
  scene.write_data_to_sim=write_with_hydraulics
  scene._basin_pressure_hook=True


def _move_surface(asset,z):
 pose=asset.data.root_link_pose_w.clone();pose[...,2]=TOP+z
 asset.write_root_pose_to_sim_index(root_pose=pose)

def physics_update(e,dt):
 if not _state:return
 import torch
 base=e.unwrapped;p=base.scene['plunger']; c=base.scene['clog']; plug=base.scene['overflow_plug']
 st=_state
 stamp=float(p.data._sim_timestamp)
 if st['last_physics_stamp'] is not None:
  elapsed=stamp-st['last_physics_stamp']
  if abs(elapsed)<1e-10:return
  if abs(elapsed-dt)>1e-7:raise RuntimeError(f'Hydraulic cache interval {elapsed} differs from physics dt {dt}')
 st['last_physics_stamp']=stamp
 st['physics_step']+=1
 origin=_array(base.scene.env_origins[0])
 bp=_array(p.data.body_pos_w)[0]-origin
 root=_array(p.data.root_link_pose_w)[0]; R=rot(root[3:7]); normal=R[:,2]
 q=_array(p.data.joint_pos)[0]; qd=_array(p.data.joint_vel)[0]
 ji={n:i for i,n in enumerate(p.joint_names)}
 rim=bp[st['ids']['rim']]; handle=bp[st['ids']['handle']]
 lip_positions=bp[[st['ids'][f'lip{i}'] for i in range(S.N_LIP)]]
 points=S.LIP.CONTACTS@R.T+lip_positions[:,None,None,:]
 surface=S.contact_floor_many(points.reshape(-1,3)).reshape(S.N_LIP,3,9)
 profile_gaps=points[:,:,:,2]-TOP-surface
 gaps=np.max(np.min(profile_gaps,axis=2),axis=1).tolist()
 minimum_gap=float(np.min(profile_gaps))
 centred=np.linalg.norm(rim[:2]-S.DRAIN[:2])+.019<=.052
 sealed=centred and normal[2]>.96 and max(gaps)<.0020 and minimum_gap>-.0020
 plugpose=_array(plug.data.root_link_pose_w)[0];pr=rot(plugpose[3:7]); pp=plugpose[:3]-origin
 delta=pp-(S.OVERFLOW+[0,0,TOP])
 # Either face of the rubber plug can form the seal; it need not have one grip.
 wall_normal=np.array([0,.85,-.527]);wall_normal/=np.linalg.norm(wall_normal)
 face_normal=pr[:,1]*np.sign(pr[:,1]@wall_normal)
 normal_gap=float(delta@wall_normal)
 lateral=np.linalg.norm(delta-normal_gap*wall_normal)
 cover=0.
 if face_normal@wall_normal>.5 and np.linalg.norm(delta)<.06:
  tangent=np.cross(wall_normal,[1.,0.,0.]);tangent/=np.linalg.norm(tangent)
  angles=np.arange(32)*2*np.pi/32
  hole=(S.OVERFLOW+[0,0,TOP])+np.cos(angles)[:,None]*np.array([.015,0,0])+np.sin(angles)[:,None]*.015*tangent
  face=pp+face_normal*.0035
  gap=(hole-face)@face_normal
  projected=hole-face-gap[:,None]*face_normal
  covered=np.linalg.norm(projected,axis=1)<=.025
  if min(gap)>-.003:
   open_area=H.OVERFLOW_A*(1-np.mean(covered))
   open_area+=sum(np.maximum(0,gap-.0005))*2*np.pi*.015/32
   cover=max(0.,1-open_area/H.OVERFLOW_A)
 water=st['water']
 clogq=float(_array(c.data.joint_pos)[0,0]);reported_clog_v=float(_array(c.data.joint_vel)[0,0])
 clogv=(clogq-st['pending']['clogq'])/dt if st['pending'] is not None else 0.
 air_residual=None;pressure_error=None;balance={}
 if st['pending'] is not None:
  previous=st['pending'];old_rim,old_handle,old_normal=st['last_structure']
  average_normal=(old_normal+normal)*.5
  actual_rim_v=float((rim-old_rim)@average_normal)/dt
  actual_handle_v=float((handle-old_handle)@average_normal)/dt
  actual_input=-dt*((H.MOUTH_A-H.PISTON_A)*actual_rim_v+H.PISTON_A*actual_handle_v)
  actual_clog_flow=H.PIPE_A*(min(H.PLUG_LENGTH,max(0.,-clogq))-min(H.PLUG_LENGTH,max(0.,-previous['clogq'])))/dt
  old_pressure=water.pressure
  release_basin,release_receiver=liquid_return_fractions(rim,R)
  water.advance(dt,previous['head'],actual_handle_v,previous['active'],previous['cover'],clogq,clogv,previous['leak'],rim_velocity=actual_rim_v,actual_plug_flow=actual_clog_flow,capture_volume=previous['capture_volume'],cup_capacity=previous['cup_capacity'],release_basin=release_basin,release_receiver=release_receiver,mouth_down=float(normal[2]))
  balance=water.balance
  if balance['active']:
   air_residual=balance['air_change']+balance['loss_volume']-actual_input
   st['air_residual_total']+=air_residual
   st['air_residual_peak']=max(st['air_residual_peak'],abs(air_residual))
   st['air_input_total']+=abs(actual_input)
   pressure_error=previous['force_pressure']-water.pressure
   st['pressure_error_peak']=max(st['pressure_error_peak'],abs(pressure_error))
   st['pressure_work']+=previous['force_pressure']*actual_input
   st['air_energy_change']+=.5*H.COMPLIANCE*(water.pressure**2-old_pressure**2)
   st['network_work']+=water.pressure*balance['loss_volume']
  if water.overflow_flow>0 and H.overflow_escapes(water.pressure,S.OVERFLOW,wall_normal,S.RIM):water.spill_overflow(1.)
 st['last_structure']=(rim.copy(),handle.copy(),normal.copy())
 velocities=_array(p.data.body_lin_vel_w)[0]
 reported_velocity=np.array([velocities[st['ids']['rim']]@normal,velocities[st['ids']['handle']]@normal])
 measured=np.array([actual_rim_v,actual_handle_v]) if st['pending'] is not None else np.zeros(2)
 acceleration=np.zeros(2) if st['last_velocity'] is None else (measured-st['last_velocity'])/dt
 st['last_velocity']=measured.copy()
 contacts=sum(g<.002 and g>-.002 for g in gaps)
 predicted,mobility=coupling.predict(measured,acceleration,dt,contacts)
 rim_v,v=predicted
 water=st['water'];surface_height=outside_height(water,rim,float(q[ji['compression']]))
 st['surface_height']=surface_height
 head=max(0.,surface_height-S.FLOOR)
 pressure_active=centred and normal[2]>.96 and max(gaps)<.010 and minimum_gap>-.002
 leak=sum(max(0.,g-.0003) for g in gaps)*2*np.pi*.056/S.N_LIP if pressure_active else .01
 pressure=water.predict(dt,head,v,pressure_active,cover,clogq,clogv,leak,rim_velocity=rim_v,coupling_mobility=mobility,force_reference=st['last_force_diff'],actual_plug_flow=actual_clog_flow if st['pending'] is not None else 0.)
 st['pending']=dict(head=head,active=pressure_active,cover=cover,clogq=clogq,leak=leak,force_pressure=pressure,capture_volume=cup_region(surface_height,rim,float(q[ji['compression']])),cup_capacity=cup_region(1.,rim,float(q[ji['compression']])))
 diff=pressure-H.RHO*H.G*head if pressure_active else 0.
 st['last_force_diff']=diff
 forces=torch.zeros((1,p.num_bodies,3),device=p.device)
 forces[0,st['ids']['handle']]=torch.tensor(normal*diff*H.PISTON_A,device=p.device)
 forces[0,st['ids']['rim']]=torch.tensor(normal*diff*(H.MOUTH_A-H.PISTON_A),device=p.device)
 p.permanent_wrench_composer.reset()
 p.permanent_wrench_composer.add_forces_and_torques(forces,torch.zeros_like(forces),is_global=True)
 mouth_pressure=pressure-H.RHO*H.G*water.overflow_stored/H.OVERFLOW_A
 cover_force=-wall_normal*mouth_pressure*H.OVERFLOW_A*cover
 plug_com=_array(plug.data.root_com_pose_w)[0,:3]-origin
 cover_torque=np.cross(S.OVERFLOW+[0,0,TOP]-plug_com,cover_force)
 plug.permanent_wrench_composer.reset()
 plug.permanent_wrench_composer.add_forces_and_torques(torch.tensor(cover_force[None,None,:],dtype=torch.float32,device=plug.device),torch.tensor(cover_torque[None,None,:],dtype=torch.float32,device=plug.device),is_global=True)
 fr=H.resistance(clogq,clogv,pressure)
 opened=clogq<=-H.PLUG_LENGTH
 if opened!=st['clog_open']:
  c.write_joint_damping_to_sim_index(damping=torch.tensor([[.08 if opened else H.PLUG_DAMPING]],device=c.device),joint_ids=[0])
  st['clog_open']=opened
 effort=H.pressure_force(clogq,pressure)-fr+H.PLUG_BUOYANCY
 c.actuators.target_command.set_effort_index(value=torch.tensor([[effort]],device=c.device),joint_ids=[0])
 # Only water that has filled the overflow branch can emerge as a jet.
 # Only outside water is counted as a spill; volume removed by the hydraulic
 # outlet is already credited separately. A tool cannot scoop the surface mesh.
 if water.spill>.00005:st['failed']=True
 st.update(sealed=sealed,cover=cover,clogq=clogq,pressure_force=diff*H.MOUTH_A)
 st['snapshot']=(rim.copy(),R.copy(),q.copy(),ji,gaps,pp.copy(),pr.copy(),handle.copy(),pressure,cover_force.copy())
 if 1600<=st['physics_step']<=3800:
  st['substeps'].append({'tick':st['physics_step'],'cache_time':float(p.data._sim_timestamp),'dt':dt,'compression':float(q[ji['compression']]),'compression_v':float(qd[ji['compression']]),'rim_v':float(measured[0]),'handle_v':float(measured[1]),'reported_rim_v':float(reported_velocity[0]),'reported_handle_v':float(reported_velocity[1]),'clog_q':clogq,'clog_v':clogv,'reported_clog_v':reported_clog_v,'pressure':pressure,'force':diff*H.MOUTH_A,'min_gap':minimum_gap,'max_gap':float(max(gaps)),'water_total':water.total,'air_residual_mL':None if air_residual is None else air_residual*1e6,'air_residual_total_mL':st['air_residual_total']*1e6,'air_input_total_mL':st['air_input_total']*1e6,'pressure_mobility_mL':balance.get('mobility_volume',0.)*1e6,'actual_pressure':water.pressure,'pressure_error':pressure_error,'node_work_J':st['pressure_work'],'air_energy_change_J':st['air_energy_change'],'network_work_J':st['network_work']})

@K.on_step
def update(e):
 if not _state or _state['snapshot'] is None:return
 st=_state;st['step']+=1;base=e.unwrapped;water=st['water']
 rim,R,q,ji,gaps,pp,pr,handle,pressure,cover_force=st['snapshot']
 # Visuals and the score remain at the normal 20 Hz task cadence.
 _move_surface(base.scene['basin_water'],st['surface_height'])
 _move_surface(base.scene['receiver_water'],S.RECEIVER_BOTTOM+(water.collected+water.diverted_receiver)/(np.pi*S.RECEIVER_R**2))
 if st['step']%2==0:st['visuals'].update(st['surface_height'],float(q[ji['compression']]),water.volume,water.collected+water.diverted_receiver)
 clearance=cup_to_waste(rim,R,float(q[ji['compression']]),[q[ji[f'lip{i}']] for i in range(S.N_LIP)],S.DRAIN+[0,0,TOP])
 cup_clear=clearance>=.030
 complete=(not st['failed'] and water.collected>=.00090 and H.bore_diameter(st['clogq'])>=.024 and cup_clear)
 st['clear_hold']=st['clear_hold']+.05 if complete else 0.
 st.update(cup_clear=cup_clear,clearance=clearance)
 if st['step']%10==0:
  row={'step':st['step'],'physics_step':st['physics_step'],'rim':rim.tolist(),'rim_normal':R[:,2].tolist(),'lip_q':[float(q[ji[f'lip{i}']]) for i in range(S.N_LIP)],'q':float(q[ji['compression']]),'gaps':list(map(float,gaps)),'sealed':bool(st['sealed']),'cover':st['cover'],'clearance':clearance,'plug':pp.tolist(),'plug_normal':pr[:,1].tolist(),'pressure':pressure,'handle':handle.tolist(),'clog':st['clogq'],'flow':water.flow,'collected':water.collected,'basin_ml':water.volume*1e6,'cup_ml':water.cup_water*1e6,'cup_overfill_peak_ml':water.cup_overfill_peak*1e6,'diverted_receiver_ml':water.diverted_receiver*1e6,'surface_height':st['surface_height'],'overflow_ml':water.overflow_stored*1e6,'neck_ml':water.neck_stored*1e6,'pipe_ml':(water.in_transit+water.sump_stored)*1e6,'spill_ml':water.spill*1e6,'total_ml':water.total*1e6,'force':st['pressure_force'],'cover_force_N':float(np.linalg.norm(cover_force)),'air_residual_total_mL':st['air_residual_total']*1e6,'air_residual_peak_mL':st['air_residual_peak']*1e6,'air_input_total_mL':st['air_input_total']*1e6,'actual_pressure':water.pressure,'pressure_error_peak':st['pressure_error_peak'],'node_work_J':st['pressure_work'],'air_energy_change_J':st['air_energy_change'],'network_work_J':st['network_work']}
  st['trace'].append(row)
  if os.environ.get('TASK_SCORE_DIR'):
   path=pathlib.Path(os.environ['TASK_SCORE_DIR']);path.mkdir(exist_ok=True,parents=True)
   (path/'hydraulic_trace.json').write_text(json.dumps(st['trace']))
   if st['step']%40==0:(path/'hydraulic_substeps.json').write_text(json.dumps(st['substeps']))
  if st['step']%40==0:print('HYDRAULICS',json.dumps(row),flush=True)


def score(objects):
 if not _state or _state.get('failed'):return 0.
 if H.bore_diameter(_state.get('clogq',0.))<.024:return 0.
 v=_state['water'].collected
 if v>=.00090 and _state['clear_hold']>=1.:return 1.
 return .5 if v>=.0005 else .25 if v>=.0002 else 0.

def success(objects):
 steptime.tick();return score(objects)==1.


def demo():
 from motion import Plan
 O,C=-.0469,0.
 L,R=Plan(TOP),Plan(TOP)
 mode=os.environ.get('TASK_DEMO','success')
 if mode=='settle':
  R.add(8,(.20,-.30,.18),O,'observe untouched basin');return {'right':R.w}
 # Approach the short handle from above in the near workspace.
 x,y=S.PLUNGER[:2];z=S.PLUNGER[2]+.115
 R.add(1.5,(.08,-.23,.16),O,'route around basin')
 R.add(1.5,(x-.07,y,z),O,'approach handle from side')
 R.add(1.5,(x,y,z),O,'beside handle')
 R.add(.8,(x,y,z),C,'grip plunger')
 R.add(1.2,(x,y,z+.015),C,'lighten cup')
 R.add(1.5,(S.DRAIN[0],S.DRAIN[1],z+.015),C,'centre cup over waste')
 R.add(1.2,(S.DRAIN[0],S.DRAIN[1],z-.005),C,'seat compliant rim')
 if mode=='seal':R.wait(2);return {'right':R.w}
 # The physical overflow cover is installed by the left arm.
 L.add(2,(S.PAD[0],S.PAD[1]-.014,.13),O,'above overflow plug',yaw=-math.pi/2)
 L.add(1.3,(S.PAD[0],S.PAD[1]-.014,S.PAD[2]+.013),O,'at plug ring',yaw=-math.pi/2)
 L.add(.8,(S.PAD[0],S.PAD[1]-.014,S.PAD[2]+.013),C,'grip plug ring',yaw=-math.pi/2)
 L.add(1.2,(S.PAD[0],S.PAD[1]-.014,.215),C,'lift above basin rim',yaw=-math.pi/2,tilt=-.4)
 L.add(1.6,(S.OVERFLOW[0],S.OVERFLOW[1]-.040,.215),C,'carry plug over rim',yaw=-math.pi/2,tilt=-.4)
 L.add(1.2,(S.OVERFLOW[0],S.OVERFLOW[1]-.035,S.OVERFLOW[2]+.021),C,'lower inside bowl',yaw=-math.pi/2,roll=-.467)
 L.add(1.2,(S.OVERFLOW[0],S.OVERFLOW[1]+.000,S.OVERFLOW[2]+.021),C,'cover overflow',yaw=-math.pi/2,roll=-.467)
 L.wait(1)
 if mode=='cover':L.wait(3);return {'left':L.w}
 R.until(L.time)
 for i in range(2 if mode=='pressure' else 5):
  R.add(.8,(S.DRAIN[0],S.DRAIN[1],z-.060),C,'compress cup',tilt=.030,roll=-.040)
  R.add(.8,(S.DRAIN[0],S.DRAIN[1],z-.008),C,'release cup')
 if mode=='pressure':
  R.wait(2);L.until(R.time);return {'left':L.w,'right':R.w}
 if mode!='failure':
  L.until(R.time)
  L.add(1.,(S.OVERFLOW[0],S.OVERFLOW[1]-.035,S.OVERFLOW[2]+.021),C,'withdraw overflow cover',yaw=-math.pi/2,roll=-.467)
  L.add(1.2,(S.OVERFLOW[0],S.OVERFLOW[1]-.040,.215),C,'lift cover over rim',yaw=-math.pi/2,tilt=-.4)
  L.add(1.4,(S.PAD[0],S.PAD[1]-.014,.215),C,'return cover beside basin',yaw=-math.pi/2,tilt=-.4)
  L.add(1.2,(S.PAD[0],S.PAD[1]-.014,.13),C,'lower above cover rest',yaw=-math.pi/2)
  L.add(1.,(S.PAD[0],S.PAD[1]-.014,S.PAD[2]+.013),C,'set cover down',yaw=-math.pi/2)
  L.add(.6,(S.PAD[0],S.PAD[1]-.014,S.PAD[2]+.013),O,'release cover',yaw=-math.pi/2)
  L.add(.7,(S.PAD[0],S.PAD[1]-.014,.13),O,'clear cover',yaw=-math.pi/2)
  R.until(L.time)
 R.add(1.8,(S.DRAIN[0]+.015,S.DRAIN[1],z+.057),C,'lift cup clear of waste',roll=.40)
 R.wait(7);L.until(R.time)
 return {'right':R.w} if mode=='failure' else {'left':L.w,'right':R.w}
