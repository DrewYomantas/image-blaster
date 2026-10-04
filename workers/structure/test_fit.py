import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import fit

class AnalyticalTests(unittest.TestCase):
    def setUp(self):
        self.sizes={f'view{i:02d}':(640,480) for i in range(1,4)}
        self.anchors={v:{'opening.bottomLeft':[230.,300.], 'opening.bottomRight':[410.,300.]} for v in self.sizes}
    def test_projection_convention(self):
        k,pose=fit.camera([0,0,0,0,0,4,1],(640,480))
        uv,z=fit.project([[-.6,0,0],[.6,0,0],[0,1,0]],k,pose)
        self.assertTrue(np.all(z>0)); self.assertLess(uv[0,0],uv[1,0]); self.assertLess(uv[2,1],uv[0,1])
        np.testing.assert_allclose(np.linalg.det(pose[:,:3]),1)
        np.testing.assert_allclose(-pose[:,:3].T@pose[:,3],[0,0,4])
        np.testing.assert_allclose(uv,[[223.5,239.5],[415.5,239.5],[319.5,79.5]])
        np.testing.assert_allclose(z,[4,4,4])
    def test_independent_numeric_projection(self):
        uv,z=fit.project([[1,2,5]],np.array([[100,0,10],[0,100,20],[0,0,1]]),np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
        np.testing.assert_allclose(uv,[[30,60]]); np.testing.assert_allclose(z,[5])
    def test_insufficient_overlap_is_underconstrained(self):
        obs=json.loads(json.dumps(self.anchors)); obs['view01']['hearth.topFrontLeft']=[190,350]; obs['view02']['mantel.topFrontRight']=[450,180]
        with patch.dict(fit.SETTINGS,starts=1,perturbations=0,max_nfev=10):
            structure,_,diagnostics=fit.solve(obs,self.sizes,{})
        self.assertEqual(structure['status'],'underconstrained'); self.assertFalse(diagnostics['numericalGatePassed'])
        self.assertEqual(structure['surfaces'],[]); self.assertTrue(any(v is None for v in structure['parameters'].values()))
    def test_offsets_and_rectangles(self):
        p={k:v[0] for k,v in fit.SPECS.items()}; p.update(hearthBack=.11,hearthDepth=.43,mantelBack=-.07,mantelDepth=.29)
        self.assertAlmostEqual(fit.point('hearth.topFrontLeft',p)[2],.54)
        self.assertAlmostEqual(fit.point('mantel.bottomFrontRight',p)[2],.22)
        self.assertAlmostEqual(fit.point('hearth.topFrontLeft',p)[1]-fit.point('hearth.bottomFrontLeft',p)[1],p['hearthThickness'])
    def test_width_scale(self):
        np.testing.assert_allclose(fit.point('opening.bottomRight',{'_width':1.8})-fit.point('opening.bottomLeft',{'_width':1.8}),[1.8,0,0])
        a=fit.Problem(self.anchors,self.sizes,{},1.2); b=fit.Problem(self.anchors,self.sizes,{},1.8)
        np.testing.assert_allclose(a.residual(a.initial),b.residual(b.initial),atol=1e-10)
    def test_rank_reports_nullspace(self):
        problem=fit.Problem(self.anchors,self.sizes,{})
        rank=fit.rank_report(fit.finite_jacobian(problem,problem.initial))
        self.assertGreater(rank['nullity'],0); self.assertEqual(rank['columns'],21)
    def test_hidden_parameter_null_direction(self):
        obs={v:dict(p,**{'mantel.bottomFrontLeft':[200,200],'mantel.bottomFrontRight':[440,200]}) for v,p in self.anchors.items()}
        problem=fit.Problem(obs,self.sizes,{})
        jac=fit.finite_jacobian(problem,problem.initial)
        np.testing.assert_allclose(jac[:,problem.names.index('mantelBack')],jac[:,problem.names.index('mantelDepth')],atol=1e-6)
    def test_exact_synthetic_projection_residual(self):
        problem=fit.Problem(self.anchors,self.sizes,{})
        p,cameras=problem.unpack(problem.initial)
        for ident in self.anchors:
            uv,_=fit.project([fit.point(k,p) for k in self.anchors[ident]],*cameras[ident])
            problem.obs[ident]=dict(zip(self.anchors[ident],uv.tolist()))
        np.testing.assert_allclose(problem.residual(problem.initial),0,atol=1e-10)
    def test_missing_endpoints(self):
        obs=json.loads(json.dumps(self.anchors)); del obs['view01']['opening.bottomLeft']
        with self.assertRaisesRegex(ValueError,'Missing'): fit.validate_annotations(dict(schemaVersion=1,views=[dict(id=v,points=p) for v,p in obs.items()]),self.sizes)
    def test_conflicting_endpoints(self):
        obs=json.loads(json.dumps(self.anchors)); obs['view01']['opening.bottomLeft']=[500,300]
        with self.assertRaisesRegex(ValueError,'ordering'): fit.validate_annotations(dict(schemaVersion=1,views=[dict(id=v,points=p) for v,p in obs.items()]),self.sizes)
    def test_metric_conflict(self):
        data=dict(schemaVersion=1,anchor=dict(kind='opening-mouth-width',meters=9),views=[dict(id=v,points=p) for v,p in self.anchors.items()])
        with self.assertRaisesRegex(ValueError,'Metric'): fit.validate_annotations(data,self.sizes)
    def test_invalid_points(self):
        data=dict(schemaVersion=1,views=[dict(id=v,points=p) for v,p in self.anchors.items()]); data['views'][0]['points']['opening.topLeft']=[float('nan'),0]
        with self.assertRaisesRegex(ValueError,'Invalid'): fit.validate_annotations(data,self.sizes)
    def test_automatic_boundary(self):
        data=dict(schemaVersion=1,views=[dict(id=v,points=dict(p,**{'opening.topLeft':[200,200]})) for v,p in self.anchors.items()])
        with self.assertRaisesRegex(ValueError,'Automatic'): fit.validate_annotations(data,self.sizes,automatic=True)
    def test_axial_normals(self):
        n=np.tile([0.,0.,-1.],(32,32,1)); mask=np.ones((32,32),bool)
        np.testing.assert_allclose(fit.normal_clusters(n,mask),[[0,0,1]])
        np.testing.assert_allclose(fit.normal_clusters(-n,mask),[[0,0,1]])
    def test_no_aperture_occluder(self):
        obj=fit.obj_text(dict(surfaces=[dict(id='opening.mouth',vertices=[[0,0,0]]*4)]))
        self.assertNotIn('\nf ',obj)
    def test_guard_actual_forbidden_open(self):
        source=Path(fit.__file__).resolve().parent
        code="import sys; from pathlib import Path; import fit; guard=fit.ReadGuard([],Path(sys.argv[1])/'output',Path(fit.__file__).parent); sys.addaudithook(guard);\ntry:\n open(Path(sys.argv[1])/'forbidden.json','rb')\nexcept PermissionError:\n print('DENIED')\nelse:\n raise RuntimeError('Leak')"
        with tempfile.TemporaryDirectory() as temp:
            forbidden=Path(temp)/'forbidden.json'; forbidden.write_text('must-not-read')
            result=subprocess.run([sys.executable,'-c',code,temp],cwd=source,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr); self.assertEqual(result.stdout.strip(),'DENIED')
    def test_guard_direct_read_write_and_network(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); guard=fit.ReadGuard([root/'allowed.json'],root/'output',Path(fit.__file__).parent)
            guard('open',(str(root/'allowed.json'),'r',0))
            with self.assertRaises(PermissionError): guard('open',(str(root/'allowed.json'),'w',0))
            with self.assertRaises(PermissionError): guard('os.listdir',(str(root),))
            with self.assertRaises(PermissionError): guard('socket.connect',(None,None))

if __name__=='__main__': unittest.main()
