import importlib.util
from pathlib import Path
import unittest
import json
import tempfile
from unittest.mock import patch
from PIL import Image
import numpy as np

spec = importlib.util.spec_from_file_location('structural_evaluation', Path(__file__).with_name('evaluate.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EvaluationTests(unittest.TestCase):
    def test_empty_geometry_and_thin_truth_face_serialize(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'inputs').mkdir()
            Image.new('RGB',(4,4)).save(root/'inputs/view-01.png')
            metadata = dict(version='analytical', input_hashes=[dict(file='view-01.png')]*4, dimensions=[])
            truth = dict(metadata=metadata, cloud=np.tile([0.,0.,1.],(4,4,4,1)), labels=np.ones((4,4,4),int), normals=np.tile([0.,0.,1.],(4,4,4,1)), depths=np.ones((4,4,4)), intrinsics=np.tile(np.eye(3),(4,1,1)), extrinsics=np.tile(np.column_stack([np.eye(3),np.zeros(3)]),(4,1,1)))
            structure = dict(coordinates=dict(units='meters',frame='scene-y-up',handedness='right',openingBottomMidpoint=[0,0,0]),points={},surfaces=[],cameras=[],status='underconstrained')
            (root/'structure.json').write_text(json.dumps(structure))
            membership = (np.ones((4,4,4),int),np.zeros((4,4,4),bool),{1:dict(id='thin',normal=[0,0,1],plane_m=1,sign=1)})
            with patch.object(module,'truth_geometry',return_value=truth), patch.object(module,'datum_translation',return_value=np.zeros(3)), patch.object(module,'named_truth_points',return_value={}), patch.object(module,'surface_membership',return_value=membership):
                report = module.evaluate(root,root/'structure.json',root/'out')
            json.dumps(report,allow_nan=False)
            self.assertEqual(report['surfaces'][0]['within_5cm_fraction'],0)
            self.assertIsNone(report['views'][0]['oracle_depth_rmse_m'])
            self.assertTrue(report['surfaces'][0]['missing'])
            self.assertTrue((root/'out/report.json').exists())

    def test_partial_faces_do_not_invent_dimensions(self):
        names = ['hearth.topFrontLeft', 'hearth.topFrontRight', 'hearth.topBackLeft']
        self.assertTrue(module.axis_supported(names, 'hearth', 0))
        self.assertFalse(module.axis_supported(names, 'hearth', 1))
        self.assertTrue(module.axis_supported(names, 'hearth', 2))
        self.assertFalse(module.axis_supported(['room.frontFloorLeft', 'room.backFloorLeft'], 'room', 2))
        self.assertFalse(module.axis_supported(['opening.topLeft', 'opening.bottomLeft'], 'opening-recess', 2))

    def test_negative_axis_face_plane_coordinates(self):
        for vertices, normal, coordinate, sign in [([[-2,0,0],[-2,1,0],[-2,1,1]],[-1,0,0],-2,-1), ([[0,2,0],[1,2,0],[1,2,1]],[0,-1,0],2,-1)]:
            result = module.plane_error(vertices,normal,coordinate*sign)
            self.assertAlmostEqual(result['signed_centroid_offset_m'],0)
            self.assertAlmostEqual(result['vertex_plane_rmse_m'],0)
            self.assertAlmostEqual(result['orientation_error_degrees'],0)

    def test_semantic_wall_absence_despite_geometric_contact(self):
        report = dict(version='v1',surfaces=[dict(primitive='front-left',axis=2,sign=1,missing=False,within_5cm_fraction=.0015)])
        structure = dict(surfaces=[dict(id='mantel.front',vertexIds=['a','b','c','d'])])
        result = module.semantic_report(report,structure,'immutable-sha')
        wall = result['surfaces'][0]
        self.assertEqual(wall['expected_semantic_surface_id'],'room.back')
        self.assertTrue(wall['semantic_missing'])
        self.assertTrue(wall['geometric_contact_within_5cm'])
        self.assertEqual(wall['within_5cm_fraction'],.0015)
        self.assertFalse(wall['missing'])
        self.assertNotIn('semantic_missing',report['surfaces'][0])
        self.assertEqual(result['report_history']['original_sha256'],'immutable-sha')

    def test_semantic_face_mapping_uses_face_axis(self):
        self.assertEqual(module.expected_semantic_surface(dict(primitive='mantel',axis=1,sign=-1)),'mantel.bottom')
        self.assertEqual(module.expected_semantic_surface(dict(primitive='hearth',axis=1,sign=1)),'hearth.top')
        self.assertEqual(module.expected_semantic_surface(dict(primitive='floor',axis=1,sign=1)),'room.floor')

    def test_plane_offset_independent_of_winding(self):
        rectangle = [[-1, 2, 3], [1, 2, 3], [1, 4, 3], [-1, 4, 3]]
        for points in (rectangle, rectangle[::-1]):
            result = module.plane_error(points, [0, 0, 1], 1)
            self.assertAlmostEqual(result['orientation_error_degrees'], 0)
            self.assertAlmostEqual(result['signed_centroid_offset_m'], 2)
            self.assertAlmostEqual(result['vertex_plane_rmse_m'], 2)

    def test_normal_sign_changes_offset_only(self):
        rectangle = [[0, 0, 3], [1, 0, 3], [1, 1, 3]]
        result = module.plane_error(rectangle, [0, 0, -1], -1)
        self.assertAlmostEqual(result['signed_centroid_offset_m'], -2)
        self.assertAlmostEqual(result['orientation_error_degrees'], 0)

    def test_camera_translation_and_positive_depth(self):
        k = [[100, 0, 20], [0, 100, 30], [0, 0, 1]]
        e = [[1, 0, 0, -1], [0, 1, 0, -2], [0, 0, 1, -3]]
        xy, z = module.project([[2, 4, 5], [1, 2, 2]], k, e)
        np.testing.assert_allclose(xy[0], [70, 130])
        np.testing.assert_allclose(z, [2, -1])
        self.assertTrue(np.isnan(xy[1]).all())

    def test_sampling_stays_in_finite_triangle(self):
        samples = module.polygon_samples([[0, 0, 0], [2, 0, 0], [0, 2, 0]], .25)
        self.assertTrue((samples[:, :2] >= 0).all())
        self.assertTrue((samples[:, 0] + samples[:, 1] <= 2+1e-12).all())
        self.assertEqual(samples[:, 2].max(), 0)
        self.assertTrue(any(np.array_equal(p, [2, 0, 0]) for p in samples))

    def test_occluded_and_behind_camera_excluded(self):
        k = np.eye(3)
        e = np.column_stack([np.eye(3), np.zeros(3)])
        index, _, _ = module.visible_samples(np.array([[0, 0, 1], [0, 0, 3], [0, 0, -1]]), k, e, np.ones((2, 2))*2)
        self.assertEqual(index.tolist(), [0])

    def test_alignment_translation_never_refits_width(self):
        metadata = {'primitives': [{'name': 'opening-back', 'low': [-2, 4, -1], 'high': [2, 6, -.5]}, {'name': 'front-left', 'high': [-2, 8, 0]}]}
        t = module.datum_translation(metadata)
        points = np.array([[-.6, 0, 0], [.6, 0, 0]]) + t
        np.testing.assert_allclose(t, [0, 4, 0])
        self.assertAlmostEqual(np.linalg.norm(points[1]-points[0]), 1.2)

    def test_degenerate_plane_reports_unavailable(self):
        self.assertEqual(module.plane_error([[0,0,0], [1,0,0], [2,0,0]], [0,1,0], 0)['status'], 'degenerate')

    def test_vertex_ids_use_aligned_points(self):
        vertices = module.surface_vertices({'vertexIds':['a','b','c'], 'vertices': [[0,0,0]]*3}, {'a':[1,2,3], 'b':[2,2,3], 'c':[2,3,3]})
        np.testing.assert_allclose(vertices[0], [1,2,3])

if __name__ == '__main__':
    unittest.main()
