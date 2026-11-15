"""Closed-air verification with free, constrained and switching mechanical loads."""
import json
import numpy as np
import coupling as C
import hydraulics as H
A=C.AREAS

def run(dt,mode):
    contacts=16;_,K,D=C.matrices(contacts);M=C.matrices(0)[0]
    ext=np.diag([1e5,1e5]) if mode=='gripped' else np.zeros((2,2))
    q=np.array([-.0002,-.0006]) if mode!='gripped' else np.zeros(2)
    v=np.zeros(2);p=-A@q/H.COMPLIANCE;force_p=p
    acceleration=np.linalg.solve(M,-(K+ext)@q+A*p)
    def energy():return .5*v@M@v+.5*q@(K+ext)@q+.5*H.COMPLIANCE*p*p
    initial=previous=peak_energy=energy();max_increase=0.;pv=0.;error=[];pressure=[];applied_work=0.;gas_change=0.
    for n in range(round(.8/dt)):
        t=n*dt
        if mode=='switch' and n==round(.0125/dt):
            contacts=0;_,K,D=C.matrices(contacts);previous=energy()
        vp,mob=C.predict(v,acceleration,dt,contacts)
        next_force=(H.COMPLIANCE*p-dt*A@vp+dt*dt*mob*force_p)/(H.COMPLIANCE+dt*dt*mob)
        target=np.array([0.,-.001*np.sin(np.pi*(t+dt)/.8)**2]) if mode=='gripped' else np.zeros(2)
        nv=np.linalg.solve(M+D*dt+(K+ext)*dt*dt,M@v-dt*(K+ext)@q+dt*ext@target+dt*A*next_force)
        nq=q+dt*nv;npres=-A@nq/H.COMPLIANCE
        actual_input=-A@(nq-q)
        applied_work+=next_force*actual_input;gas_change+=.5*H.COMPLIANCE*(npres*npres-p*p)
        acceleration=(nv-v)/dt;q=nq;v=nv;p=npres;force_p=next_force
        current=energy();peak_energy=max(peak_energy,current)
        if mode!='gripped':max_increase=max(max_increase,current-previous)
        previous=current;pv=max(pv,abs(H.COMPLIANCE*p+A@q));error.append(force_p-p);pressure.append(p)
    rms=float(np.sqrt(np.mean(np.square(error))));peak=max(abs(x) for x in pressure)
    assert pv<1e-12
    assert mode=='gripped' or peak_energy<=initial+1e-10,(dt,mode,peak_energy,initial)
    return dict(hz=1/dt,mode=mode,pv_residual_m3=pv,max_step_energy_gain_J=max_increase,initial_energy_J=initial,peak_energy_J=peak_energy,final_energy_J=previous,pressure_peak_Pa=peak,force_pressure_rms_error_Pa=rms,relative_rms=rms/max(1.,peak),applied_node_work_J=applied_work,air_energy_change_J=gas_change)

if __name__=='__main__':
    rows=[run(1/hz,mode) for mode in ['free','gripped','switch'] for hz in [240,480,960]]
    print(json.dumps(rows,indent=2))
