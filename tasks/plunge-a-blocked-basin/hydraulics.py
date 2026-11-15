"""Lumped water/air hydraulics, real pressure forces and explicit water storage.

Only water crossing the outlet after its transport time enters ``collected``.
Air admitted through the overflow is never counted as water or as a spill.
"""
from dataclasses import dataclass,field
import math
RHO=1000.;AIR_RHO=1.2;G=9.81
PIPE_R=.016;PIPE_A=math.pi*PIPE_R**2
MOUTH_A=math.pi*.056**2;PISTON_A=0.00656001505983
INITIAL_VOLUME=.001
AIR_VOLUME=.00023;ATM=101325.;COMPLIANCE=AIR_VOLUME/ATM
OVERFLOW_H=.074;OVERFLOW_A=math.pi*.015**2;CD=.60
BREAKAWAY=1.20;KINETIC=.50;PLUG_DAMPING=30.;PLUG_LENGTH=.010
OUTLET_DROP=.027
SUMP_CAPACITY=math.pi*.022**2*.003
PLUG_MASS=.006
PLUG_BUOYANCY=RHO*G*.000005
PLUG_WEIGHT=PLUG_MASS*G-PLUG_BUOYANCY
TRANSPORT_TIME=.35

# Internal profile below the moving end plate. The first 7 mm is fixed;
# the corrugated portion shortens by the measured compression joint.
CUP_PROFILE=[(.052,0.),(.052,.007),(.052,.009),(.056,.013),(.046,.018),(.053,.022),(.043,.027),(.050,.031),(.040,.036),(.047,.040),(.037,.045),(.044,.049),(.034,.054),(.040,.058)]
def cup_profile_volume(depth,compression=0.):
    scale=max(.2,1+compression/.051)
    profile=[(r,z if z<=.007 else .007+(z-.007)*scale) for r,z in CUP_PROFILE]
    total=0.
    for (r0,z0),(r1,z1) in zip(profile,profile[1:]):
        h=min(depth,z1)-z0
        if h<=0:break
        r=r0+(r1-r0)*h/(z1-z0)
        total+=math.pi*h*(r0*r0+r0*r+r*r)/3
    return total

def orifice(p,area,density=RHO):
    return CD*area*math.copysign(math.sqrt(2*abs(p)/density),p) if p else 0.

def bore_diameter(q):return 2*PIPE_R if q<=-PLUG_LENGTH else 0.

def flow_area(q):
    area=math.pi*(bore_diameter(q)/2)**2+2.e-7
    grate=PIPE_A-(2*.032*.003-.003**2)
    return 1/math.sqrt(area**-2+grate**-2)

def pressure_force(q,p):
    # The widened chamber equalises pressure on both sides of a released mat.
    return -(p+RHO*G*(.009-q))*PIPE_A if q>-PLUG_LENGTH else 0.

def outlet_flow(p,q,liquid=True):
    density=RHO if liquid else AIR_RHO
    differential=p+(density*G*OUTLET_DROP if q<=-PLUG_LENGTH else 0.)
    return orifice(differential,flow_area(q),AIR_RHO if differential<0 else density)

def resistance(q,v,p):
    if q<=-PLUG_LENGTH:return 0.
    f=pressure_force(q,p)-PLUG_WEIGHT
    if abs(v)<.001 and abs(f)<=BREAKAWAY:return f
    return KINETIC*math.copysign(1.,v if abs(v)>.001 else f)

def overflow_escapes(pressure,origin,wall_normal,rim_height=.160):
    """Ballistic centreline of a water jet, used only after the branch fills."""
    speed=math.sqrt(2*max(0.,pressure/RHO-G*OVERFLOW_H))
    velocity=[-n*speed for n in wall_normal]
    discriminant=velocity[2]**2+2*G*(origin[2]-rim_height)
    if discriminant<=0:return False
    t=(velocity[2]+math.sqrt(discriminant))/G
    x=origin[0]+velocity[0]*t;y=origin[1]+velocity[1]*t
    return ((x-.18)/.140)**2+(y/.175)**2>1.

@dataclass
class Water:
    volume:float=INITIAL_VOLUME
    cup_water:float=0.
    cup_active:bool=False
    cup_overfill_peak:float=0.
    diverted_receiver:float=0.
    collected:float=0.
    spill:float=0.
    overflow_stored:float=0.
    neck_stored:float=0.
    sump_stored:float=0.
    pressure:float=0.
    head_pressure:float=0.
    flow:float=0.
    receiver_flow:float=0.
    overflow_flow:float=0.
    lip_flow:float=0.
    peak_pressure:float=0.
    time:float=0.
    pending:list=field(default_factory=list)
    overflow_amount:float=0.
    balance:dict=field(default_factory=dict)

    @property
    def in_transit(self):return sum(v for _,v in self.pending)
    @property
    def total(self):return self.volume+self.cup_water+self.diverted_receiver+self.collected+self.spill+self.overflow_stored+self.neck_stored+self.sump_stored+self.in_transit

    def _solution(self,dt,head,piston_velocity,sealed,overflow_fraction,plug_q,plug_v,leak_area=0.,rim_velocity=0.,coupling_mobility=0.,actual_plug_flow=None,force_reference=None):
        p0=RHO*G*head
        reference=self.pressure-self.head_pressure if force_reference is None else force_reference
        source=-(PISTON_A*piston_velocity+(MOUTH_A-PISTON_A)*rim_velocity) if sealed else 0.
        area=OVERFLOW_A*(1-min(1.,max(0.,overflow_fraction)))
        capacity=OVERFLOW_A*OVERFLOW_H
        def overflow(p):
            if area<=1e-12:return 0.
            difference=p-RHO*G*self.overflow_stored/OVERFLOW_A
            # Implicit water-column height prevents an artificial instantaneous
            # jet from a branch that still contains air.
            a=RHO/(2*(CD*area)**2);b=RHO*G*dt/OVERFLOW_A
            q=math.copysign(2*abs(difference)/(b+math.sqrt(b*b+4*a*abs(difference))),difference)
            if self.overflow_stored+q*dt>capacity:
                return max(0.,orifice(p-RHO*G*OVERFLOW_H,area))
            if self.overflow_stored+q*dt<0:
                air_fraction=1-self.overflow_stored/max(1e-30,-q*dt)
                return -self.overflow_stored/dt+air_fraction*orifice(min(0.,p),area,AIR_RHO)
            return q
        def plugflow(p):
            if actual_plug_flow is not None:return actual_plug_flow
            return -PIPE_A*plug_v if plug_q>-PLUG_LENGTH else 0.
        liquid=not self.cup_active or self.cup_water+self.neck_stored>1e-9
        def loss(p):return outlet_flow(p,plug_q,liquid)+overflow(p)+orifice(p-p0,leak_area)+plugflow(p)
        if not sealed or self.volume+self.cup_water<=0:
            p=p0
            lip_flow=0.
        else:
            lo,hi=-90000.,120000.
            for _ in range(42):
                mid=(lo+hi)/2
                increment=(mid-p0)-reference
                residual=COMPLIANCE*(mid-self.pressure)/dt+dt*coupling_mobility*increment+loss(mid)-source
                if residual>0:hi=mid
                else:lo=mid
            p=(lo+hi)/2
            lip_flow=orifice(p-p0,leak_area)
        return dict(pressure=p,head=p0,active=bool(sealed and self.volume+self.cup_water>0),
            air_change=COMPLIANCE*(p-self.pressure),loss_volume=dt*loss(p) if sealed and self.volume+self.cup_water>0 else 0.,
            source_volume=dt*source,mobility_volume=dt*dt*coupling_mobility*((p-p0)-reference),
            branch_flow=overflow(p),lip_flow=lip_flow,outlet_flow=outlet_flow(p,plug_q,liquid))

    def predict(self,*args,**kwargs):
        return self._solution(*args,**kwargs)['pressure']

    def _take_liquid(self,amount):
        key='cup_water' if self.cup_active else 'volume'
        taken=min(getattr(self,key),max(0.,amount));setattr(self,key,getattr(self,key)-taken)
        return taken

    def _return_liquid(self,amount):
        key='cup_water' if self.cup_active else 'volume'
        setattr(self,key,getattr(self,key)+amount)

    def advance(self,dt,head,piston_velocity,sealed,overflow_fraction,plug_q,plug_v,leak_area=0.,rim_velocity=0.,coupling_mobility=0.,actual_plug_flow=None,capture_volume=None,cup_capacity=None,release_basin=1.,release_receiver=0.,mouth_down=1.):
        self.time+=dt
        if sealed and not self.cup_active:
            captured=min(self.volume,max(0.,(cup_profile_volume(head) if capture_volume is None else capture_volume)-self.cup_water))
            self.volume-=captured;self.cup_water+=captured
        self.cup_active=bool(sealed)
        if not self.cup_active and self.cup_water>0:
            release=min(self.cup_water,CD*MOUTH_A*math.sqrt(2*G*self.cup_water/MOUTH_A)*dt*max(0.,mouth_down))
            basin_fraction=max(0.,min(1.,release_basin))
            receiver_fraction=max(0.,min(1-basin_fraction,release_receiver))
            self.cup_water-=release
            self.volume+=release*basin_fraction
            self.diverted_receiver+=release*receiver_fraction
            self.spill+=release*(1-basin_fraction-receiver_fraction)
        self.overflow_flow=self.overflow_amount=self.lip_flow=0.
        solution=self._solution(dt,head,piston_velocity,sealed,overflow_fraction,plug_q,plug_v,leak_area,rim_velocity,coupling_mobility,actual_plug_flow)
        p=solution['pressure'];p0=solution['head'];self.balance=solution
        self.lip_flow=solution['lip_flow']
        self.pressure=p;self.head_pressure=p0;self.peak_pressure=max(self.peak_pressure,abs(p-p0))
        capacity=OVERFLOW_A*OVERFLOW_H
        # A sealed cup has a finite liquid inventory. Air passing through it
        # cannot draw from the surrounding basin unless water crosses the lip.
        if self.cup_active:
            if self.lip_flow>=0:self.volume+=self._take_liquid(self.lip_flow*dt)
            else:
                incoming=min(self.volume,-self.lip_flow*dt)
                self.volume-=incoming;self.cup_water+=incoming
        branch=solution['branch_flow']*dt
        if branch>=0:
            taken=self._take_liquid(branch)
            stored=min(taken,capacity-self.overflow_stored);self.overflow_stored+=stored
            self.overflow_amount=taken-stored
            self.volume+=self.overflow_amount # A normal jet returns to the bowl.
            self.overflow_flow=self.overflow_amount/dt
        else:
            returned=min(self.overflow_stored,-branch)
            self.overflow_stored-=returned;self._return_liquid(returned)
        # Water occupying the space vacated by a moving obstruction is stored
        # in the neck. It has not yet crossed the outlet.
        if plug_q>-PLUG_LENGTH:
            desired=PIPE_A*max(0.,-plug_q)
            change=desired-self.neck_stored
            if change>=0:self.neck_stored+=self._take_liquid(change)
            else:self.neck_stored+=change;self._return_liquid(-change)
        self.flow=max(0.,solution['outlet_flow'])
        requested=self.flow*dt
        neck=min(self.neck_stored,requested) if plug_q<=-PLUG_LENGTH else 0.
        self.neck_stored-=neck
        transfer=neck+self._take_liquid(requested-neck)
        if self.cup_active and cup_capacity is not None:
            self.cup_overfill_peak=max(self.cup_overfill_peak,self.cup_water-cup_capacity)
        # No capacity clamp transfers water through a sealed wall. A positive
        # diagnostic requires fixing the physical coupling before using a run.
        sump=min(transfer,SUMP_CAPACITY-self.sump_stored)
        self.sump_stored+=sump;transfer-=sump
        if transfer:self.pending.append((self.time+TRANSPORT_TIME,transfer))
        arrived=sum(v for when,v in self.pending if when<=self.time+1e-10)
        self.pending=[(when,v) for when,v in self.pending if when>self.time+1e-10]
        self.collected+=arrived;self.receiver_flow=arrived/dt
        return p

    def spill_overflow(self,fraction):
        escaped=min(self.volume,self.overflow_amount*max(0.,min(1.,fraction)))
        self.volume-=escaped;self.spill+=escaped
        return escaped
