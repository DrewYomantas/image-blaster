import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("moge2_worker", Path(__file__).with_name("worker.py"))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class RequestBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "input.png"
        self.image.write_bytes(b"fixture")
        self.request = {"inputs": [{"path": str(self.image)}], "outputDir": str(self.root / "out")}
        self.conditioning = {"mode": "known-fov", "provenance": "experiment-oracle", "horizontalFovDegrees": [85]}

    def test_unconditioned_defaults_require_no_model(self):
        paths, output, resolution, threads, conditioning = worker.validate_request(self.request)
        self.assertEqual(paths, [self.image])
        self.assertEqual(output, self.root / "out")
        self.assertEqual((resolution, threads, conditioning), (9, 6, {"mode": "none"}))

    def test_only_explicit_oracle_fov_is_accepted(self):
        self.assertEqual(worker.validate_request({**self.request, "conditioning": self.conditioning})[-1], self.conditioning)
        for key in ("groundTruth", "extrinsics", "depth", "knownScaleMeters", "labels"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                worker.validate_request({**self.request, key: {}})
            with self.subTest(key=key), self.assertRaises(ValueError):
                worker.validate_request({**self.request, "conditioning": {**self.conditioning, key: {}}})

    def test_fov_count_type_bounds_and_provenance(self):
        for values in ([], [85, 85], [True], [None], [1], [179], [float("nan")], [float("inf")], ["85"]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                worker.validate_request({**self.request, "conditioning": {**self.conditioning, "horizontalFovDegrees": values}})
        for provenance in ("measured", "manufacturer-specified", "inferred"):
            with self.subTest(provenance=provenance), self.assertRaises(ValueError):
                worker.validate_request({**self.request, "conditioning": {**self.conditioning, "provenance": provenance}})

    def test_none_cannot_hide_truth(self):
        with self.assertRaises(ValueError):
            worker.validate_request({**self.request, "conditioning": {"mode": "none", "horizontalFovDegrees": [85]}})

    def test_parameters_are_typed_bounded_and_only_requested_controls(self):
        for name, values in (("resolutionLevel", (True, -1, 10, 9.0, "9")), ("threads", (True, 0, 65, 6.0, "6"))):
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    worker.validate_request({**self.request, "parameters": {name: value}})
        with self.assertRaises(ValueError):
            worker.validate_request({**self.request, "parameters": {"forceProjection": False}})

    def test_outputs_and_duplicate_sources_are_protected(self):
        with self.assertRaises(ValueError):
            worker.validate_request({**self.request, "inputs": self.request["inputs"] * 2})
        output = self.root / "out"
        output.mkdir()
        (output / "evidence.npz").touch()
        with self.assertRaises(ValueError):
            worker.validate_request(self.request)


class IdentityBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source = root / "source"
        self.helper = root / "helper"
        self.model = root / "model"
        self.model.mkdir()
        (self.model / "model.pt").write_bytes(b"weights")
        self.trees = {}
        for source, package in ((self.source, "moge"), (self.helper, "utils3d_moge")):
            (source / package).mkdir(parents=True)
            data = b"value = 1\n"
            (source / package / "fixture.py").write_bytes(data)
            blob = worker.hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            self.trees[str(source)] = f"100644 blob {blob}\t{package}/fixture.py\n"
        versions = dict(row.split("==") for row in (worker.ROOT / "requirements.txt").read_text().splitlines())
        def git(args, **kwargs):
            if "rev-parse" in args:
                return worker.CODE_REVISION if args[2] == str(self.source) else worker.HELPER_REVISION
            return self.trees[args[2]]
        for mock in (patch.object(worker, "locations", return_value=(self.source, self.helper, self.model)),
                     patch.object(worker, "MODEL_HASH", worker.sha256(self.model / "model.pt")),
                     patch.object(worker, "MODEL_BYTES", 7),
                     patch.object(worker.subprocess, "check_output", side_effect=git),
                     patch.object(worker.importlib.metadata, "version", side_effect=versions.__getitem__),
                     patch.object(worker.importlib.metadata, "distributions", return_value=[])):
            mock.start()
            self.addCleanup(mock.stop)

    def test_identity_is_model_free_and_covers_two_source_revisions(self):
        identity = worker.identity()
        self.assertEqual(identity["upstreamRevision"], worker.CODE_REVISION)
        self.assertEqual(identity["helperRevision"], worker.HELPER_REVISION)
        self.assertEqual(identity["license"]["weights"], "MIT")

    def test_model_source_helper_and_dependency_drift_fail_closed(self):
        (self.model / "model.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            worker.identity()
        (self.model / "model.pt").write_bytes(b"weights")
        (self.helper / "utils3d_moge/fixture.py").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "modified"):
            worker.identity()
        (self.helper / "utils3d_moge/fixture.py").write_bytes(b"value = 1\n")
        (self.source / "moge/unreviewed.py").touch()
        with self.assertRaisesRegex(ValueError, "unreviewed"):
            worker.identity()
        (self.source / "moge/unreviewed.py").unlink()
        with patch.object(worker.importlib.metadata, "version", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "dependency"):
                worker.identity()


class StrictCheckpointTests(unittest.TestCase):
    def test_official_loader_followed_by_strict_state_loading(self):
        class Model:
            def state_dict(self):
                return {"parameter": object()}
            def load_state_dict(self, state, strict):
                self.strict = strict
        model = Model()
        class Class:
            @staticmethod
            def from_pretrained(path):
                return model
        class Torch:
            @staticmethod
            def load(path, map_location, weights_only):
                if map_location != "cpu" or not weights_only:
                    raise AssertionError("Only bounded CPU weights loading")
                return {"model": {"parameter": object()}, "model_config": {}}
        self.assertIs(worker.load_model(Class, "model.pt", Torch), model)
        self.assertIs(model.strict, True)
        with patch.object(Torch, "load", return_value={"model": {}, "model_config": {}}):
            with self.assertRaisesRegex(ValueError, "missing or unexpected"):
                worker.load_model(Class, "model.pt", Torch)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is optional for model-free request tests")
class ArrayContractTests(unittest.TestCase):
    def arrays(self):
        import numpy as np
        points = np.zeros((1, 2, 3, 3), dtype=np.float32)
        points[..., 2] = 1
        normals = np.zeros_like(points)
        normals[..., 2] = -1
        return {"camera_points": points, "depths": points[..., 2].copy(), "normals": normals,
                "masks": np.ones((1, 2, 3), dtype=bool), "intrinsics": np.eye(3)[None],
                "intrinsics_normalized": np.eye(3)[None], "processed_images": np.zeros_like(points, dtype=np.uint8)}

    def test_invalid_masked_geometry_is_preserved(self):
        import numpy as np
        arrays = self.arrays()
        arrays["masks"][0, 0, 0] = False
        arrays["depths"][0, 0, 0] = np.inf
        arrays["camera_points"][0, 0, 0] = np.inf
        arrays["normals"][0, 0, 0] = 0
        worker.validate_predictions(arrays, 1, 2, 3)
        self.assertTrue(np.isinf(arrays["depths"][0, 0, 0]))

    def test_valid_points_depths_normals_masks_and_camera_shapes_are_enforced(self):
        import numpy as np
        mutations = (("depths", lambda a: a.fill(-1)), ("camera_points", lambda a: a.fill(np.nan)),
                     ("normals", lambda a: a.fill(0)), ("intrinsics", lambda a: a.fill(np.nan)))
        for key, mutate in mutations:
            arrays = self.arrays()
            mutate(arrays[key])
            with self.subTest(key=key), self.assertRaises(ValueError):
                worker.validate_predictions(arrays, 1, 2, 3)
        arrays = self.arrays()
        arrays["masks"] = arrays["masks"].astype(np.uint8)
        with self.assertRaises(ValueError):
            worker.validate_predictions(arrays, 1, 2, 3)


@unittest.skipUnless(importlib.util.find_spec("torch") and worker.locations()[0].is_dir(),
                     "optional installed CPU packages and pinned source")
class OfficialMechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source, helper, _ = worker.locations()
        sys.path[:0] = [str(source), str(helper)]

    def test_official_normalized_to_pixel_intrinsics_maps_halfpixel_centres(self):
        import torch
        import utils3d_moge as utils3d
        normalized = torch.tensor([[350/640, 0, .5], [0, 350/480, .5], [0, 0, 1]])
        actual = utils3d.pt.denormalize_intrinsics(normalized, (480, 640), pixel_convention="integer-center")
        expected = torch.tensor([[350., 0, 319.5], [0, 350., 239.5], [0, 0, 1.]])
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))

    def test_official_normal_orientation_is_camera_facing(self):
        import torch
        import utils3d_moge as utils3d
        depth = torch.ones((1, 5, 7))
        k = torch.tensor([[[1., 0, .5], [0, 1., .5], [0, 0, 1.]]])
        normals = utils3d.pt.depth_map_to_normal_map(depth, k)
        self.assertTrue(torch.allclose(normals[0, 2, 3], torch.tensor([0., 0., -1.])))

    def test_fov_is_postforward_and_scale_head_is_metric_multiplier(self):
        import torch
        from moge.model.v2 import MoGeModel
        model = MoGeModel.__new__(MoGeModel)
        torch.nn.Module.__init__(model)
        model.register_parameter("fixture_parameter", torch.nn.Parameter(torch.zeros(1)))
        model.num_tokens_range = [1200, 2500]
        calls = []
        def forward(image, num_tokens):
            calls.append(num_tokens)
            point = torch.zeros((1, 5, 7, 3))
            point[..., 2] = 1
            normal = torch.zeros_like(point)
            normal[..., 2] = -1
            return {"points": point, "normal": normal, "mask": torch.ones((1, 5, 7)), "metric_scale": torch.tensor([3.])}
        model.forward = forward
        with patch("moge.model.v2.recover_focal_shift", return_value=(torch.ones(1), torch.tensor([2.]))) as recover:
            native = model.infer(torch.zeros(3, 5, 7), resolution_level=9, use_fp16=False)
            self.assertNotIn("focal", recover.call_args.kwargs)
            known = model.infer(torch.zeros(3, 5, 7), resolution_level=9, fov_x=80, use_fp16=False)
            self.assertIn("focal", recover.call_args.kwargs)
        self.assertEqual(calls, [2500, 2500])
        self.assertEqual(tuple(known["points"].shape), (5, 7, 3))
        self.assertTrue(torch.allclose(known["depth"], torch.full((5, 7), 9.)))
        self.assertTrue(torch.allclose(native["depth"], known["depth"]))
        self.assertTrue(torch.all(known["mask"]))


if __name__ == "__main__":
    unittest.main()
