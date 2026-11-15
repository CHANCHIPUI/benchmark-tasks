"""Offline calibration/regression checks; never used by the task controller."""
import json
import numpy as np
import hydraulics as H
import water_geometry as W
from clearance import cup_to_waste

def simulate(mode):
    water=H.Water();q=v=0.;peak=0.;first_exit=None;largest_residual=0.
    normal=np.array([0.,.85,-.527]);normal/=np.linalg.norm(normal)
    for i in range(700):
        if mode=='idle':speed=0.;active=False;cover=0.
        elif mode=='fast':
            speed=(-.16 if i%12<4 else .08) if i<60 else 0.;active=i<60;cover=0.
        else:
            speed=(-.025 if i%32<16 else .025) if i<160 else 0.;active=i<160;cover=float(mode=='covered')
        p=water.advance(.05,W.height(water.volume)-.065,speed,active,cover,q,v,.000050)
        if H.overflow_escapes(p,[.18,.1338,.139],normal):water.spill_overflow(1.)
        damping=H.PLUG_DAMPING if q>-H.PLUG_LENGTH else .08
        force=H.pressure_force(q,p)-H.PLUG_WEIGHT-H.resistance(q,v,p)
        # Implicit velocity step retains inertia without requiring simulation.
        v=(v+.05*force/H.PLUG_MASS)/(1+.05*damping/H.PLUG_MASS)
        q=float(np.clip(q+.05*v,-.033,.0015))
        if q<=-H.PLUG_LENGTH and first_exit is None:first_exit=i*.05
        peak=max(peak,abs(p-H.RHO*H.G*(W.height(water.volume)-.065)))
        largest_residual=max(largest_residual,abs(water.total-H.INITIAL_VOLUME))
        assert min(water.volume,water.collected,water.spill,water.overflow_stored,water.neck_stored,water.sump_stored)>=-1e-12
        assert largest_residual<1e-12
        if water.overflow_flow>0:assert water.overflow_stored>=H.OVERFLOW_A*H.OVERFLOW_H-1e-12
    return dict(mode=mode,receiver_ml=water.collected*1e6,spill_ml=water.spill*1e6,clog_mm=q*1000,first_clear_s=first_exit,peak_pressure_Pa=peak,max_water_error_ml=largest_residual*1e6)

if __name__=='__main__':
    rows={m:simulate(m) for m in ['idle','uncovered','covered','fast']}
    assert rows['idle']['first_clear_s'] is None
    assert rows['uncovered']['first_clear_s'] is None
    assert rows['covered']['receiver_ml']>=900
    assert rows['covered']['spill_ml']==0
    # There is no pressure-driven clearing event based on stroke count.
    assert rows['uncovered']['receiver_ml']<20
    assert cup_to_waste(np.zeros(3),np.eye(3),0,[0]*16,np.zeros(3))==0
    assert cup_to_waste(np.array([.03,0,0]),np.eye(3),0,[0]*16,np.zeros(3))==0
    assert cup_to_waste(np.array([0,0,.045]),np.eye(3),0,[0]*16,np.zeros(3))>.030
    print(json.dumps(rows,indent=2))
