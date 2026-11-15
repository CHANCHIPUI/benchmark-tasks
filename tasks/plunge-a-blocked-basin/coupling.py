"""Incremental backward-Euler fluid/structure coupling for the real tool mass.

No physical compliance, mass or force limit is changed. The pressure solve
includes the velocity response to its own pressure increment. Acceleration
from consecutive refreshed physics states supplies other applied/contact loads.
"""
import numpy as np
import hydraulics as H
AREAS=np.array([H.MOUTH_A-H.PISTON_A,H.PISTON_A])
PAIR=np.array([[1.,-1.],[-1.,1.]])

def matrices(contacts):
    M=np.diag([.025+(16-contacts)*.0015,.030])
    K=350.*PAIR+np.diag([250.*contacts,0.])
    D=4.*PAIR+np.diag([1.2*contacts,0.])
    return M,K,D

def predict(velocity,acceleration,dt,contacts):
    M,K,D=matrices(contacts)
    B=M+D*dt+K*dt*dt
    predicted=np.linalg.solve(B,(M+D*dt)@velocity+dt*M@acceleration)
    mobility=float(AREAS@np.linalg.solve(B,AREAS))
    return predicted,mobility

if __name__=='__main__':
    import json
    rows=[]
    for dt in [1/20,1/240,1/480,1/960]:
        M,K,D=matrices(16);q=np.array([-.001,-.003]);v=np.zeros(2)
        p=-AREAS@q/H.COMPLIANCE
        acceleration=np.linalg.solve(M,-K@q+AREAS*p)
        def energy():return .5*v@M@v+.5*q@K@q+.5*H.COMPLIANCE*p*p
        initial=previous=energy();increase=0.;volume_error=0.
        for _ in range(round(.5/dt)):
            predicted,mobility=predict(v,acceleration,dt,16)
            p-=dt*(AREAS@predicted)/(H.COMPLIANCE+dt*dt*mobility)
            new=np.linalg.solve(M+D*dt+K*dt*dt,M@v-dt*K@q+dt*AREAS*p)
            acceleration=(new-v)/dt;v=new;q+=dt*v
            current=energy();increase=max(increase,current-previous);previous=current
            volume_error=max(volume_error,abs(H.COMPLIANCE*p+AREAS@q))
        assert increase<1e-12
        assert volume_error<1e-12
        rows.append(dict(hz=1/dt,initial_energy_J=initial,final_energy_J=previous,largest_energy_increase_J=increase,max_air_volume_residual_m3=volume_error))
    print(json.dumps(rows,indent=2))
