"""A vented, perfectly sealed cup cannot drain the surrounding basin as air."""
import json
import hydraulics as H

w=H.Water()
for i in range(2400):
    w.advance(1/240,.025,0.,True,0.,-.033,0.,0.,actual_plug_flow=0.,capture_volume=.0002,cup_capacity=.0004)
    assert abs(w.total-.001)<1e-12
assert abs(w.volume-.0008)<1e-12
assert w.collected<=.0002+1e-12
sealed=dict(outside_ml=w.volume*1e6,cup_ml=w.cup_water*1e6,receiver_ml=w.collected*1e6)
for _ in range(2400):
    w.advance(1/240,.025,0.,False,0.,-.033,0.,actual_plug_flow=0.)
    assert abs(w.total-.001)<1e-12
assert w.collected>.0009
print(json.dumps(dict(sealed=sealed,after_cup_removed_receiver_ml=w.collected*1e6),indent=2))

# A cup carried outside cannot silently return liquid to the basin or earn
# waste-outlet credit. An upward-facing cup retains its liquid until tipped.
for destination in ['table','receiver','retained']:
    w=H.Water(volume=.0008,cup_water=.0002,cup_active=True)
    for _ in range(240):
        w.advance(1/240,.02,0.,False,0.,0.,0.,release_basin=0.,release_receiver=float(destination=='receiver'),mouth_down=-1. if destination=='retained' else 1.)
        assert abs(w.total-.001)<1e-12
    assert w.collected<1e-6
    if destination=='table':assert w.spill>.00019
    elif destination=='receiver':assert w.diverted_receiver>.00019 and w.spill==0
    else:assert w.cup_water==.0002 and w.spill==0

# Even an intentionally impossible overfilled initial condition cannot invoke
# a hidden transfer through a perfectly sealed wall; it is diagnosed instead.
w=H.Water(volume=.0008,cup_water=.0002,cup_active=True)
w.advance(1/240,.025,0.,True,1.,0.,0.,0.,actual_plug_flow=0.,cup_capacity=.0001)
assert w.volume==.0008 and w.cup_overfill_peak>.00009
