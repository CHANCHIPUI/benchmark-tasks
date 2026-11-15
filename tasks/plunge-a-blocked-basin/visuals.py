"""Measured hydraulic state drives only water and rubber-skin appearance."""
import numpy as np
import water_geometry as W

class Visuals:
    def __init__(self):
        import omni.usd
        from pxr import Usd,UsdGeom
        self.meshes={}
        stage=omni.usd.get_context().get_stage()
        for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/envs/env_0')):
            path=str(prim.GetPath())
            if prim.GetTypeName()!='Mesh':continue
            if '/basin_water/' in path and prim.GetName()=='surface':self.meshes['water']=UsdGeom.Mesh(prim)
            if '/receiver_water/' in path and prim.GetName()=='surface':self.meshes['receiver']=UsdGeom.Mesh(prim)
            if '/plunger/' in path and prim.GetName()=='dome':self.meshes['dome']=UsdGeom.Mesh(prim)
        if 'receiver' in self.meshes:self.meshes['receiver'].CreateVisibilityAttr().Set('invisible')
        self.original={key:np.asarray(mesh.GetPointsAttr().Get(),dtype=np.float32) for key,mesh in self.meshes.items()}

    def update(self,height,compression,basin_volume,receiver_volume):
        from pxr import Vt
        for key,volume in [('water',basin_volume),('receiver',receiver_volume)]:
            if key in self.meshes:self.meshes[key].CreateVisibilityAttr().Set('inherited' if volume>.0000005 else 'invisible')
        if 'water' in self.meshes:
            self.meshes['water'].GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(W.mesh(height).v.astype(np.float32)))
        if 'dome' in self.meshes:
            pts=self.original['dome'].copy()
            scale=np.clip(1+compression/.051,.2,1.2)
            pts[:,2]=np.where(pts[:,2]>=.05799,pts[:,2]+compression,.007+(pts[:,2]-.007)*scale)
            self.meshes['dome'].GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts))
