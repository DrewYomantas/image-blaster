import ast
import copy
import importlib.util
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("da3_worker", Path(__file__).with_name("worker.py"))
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

    def test_validation_works_without_model_packages(self):
        paths, output, resolution, threads, profile = worker.validate_request(self.request)
        self.assertEqual(paths, [self.image])
        self.assertEqual(output, self.root / "out")
        self.assertEqual(resolution, 384)
        self.assertEqual(threads, 6)
        self.assertIs(profile, False)

    def test_ground_truth_and_scale_are_rejected(self):
        for key in ("scene", "groundTruth", "cameras", "knownScaleMeters"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                worker.validate_request({**self.request, key: {}})

    def test_camera_parameter_injection_is_rejected(self):
        with self.assertRaises(ValueError):
            worker.validate_request({**self.request, "parameters": {"intrinsics": []}})

    def test_resolution_is_bounded_and_typed(self):
        for resolution in (True, 384.0, 504, "384", None):
            with self.subTest(resolution=resolution), self.assertRaises(ValueError):
                worker.validate_request({**self.request, "parameters": {"processResolution": resolution}})

    def test_nonempty_output_cannot_be_overwritten(self):
        output = self.root / "out"
        output.mkdir()
        (output / "existing.npz").touch()
        with self.assertRaises(ValueError):
            worker.validate_request(self.request)

    def test_duplicate_images_are_rejected(self):
        with self.assertRaises(ValueError):
            worker.validate_request({**self.request, "inputs": self.request["inputs"] * 2})

    def test_thread_limit_is_bounded_and_typed(self):
        for threads in (True, 0, 65, "6", 6.0):
            with self.subTest(threads=threads), self.assertRaises(ValueError):
                worker.validate_request({**self.request, "parameters": {"threads": threads}})

    def test_warm_profile_accepts_only_booleans(self):
        for profile in (0, 1, "true", None, []):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                worker.validate_request({**self.request, "parameters": {"profileWarmInference": profile}})
        self.assertIs(worker.validate_request({**self.request, "parameters": {"profileWarmInference": True}})[-1], True)

    def test_unknown_parameter_is_rejected(self):
        with self.assertRaises(ValueError):
            worker.validate_request({**self.request, "parameters": {"unknown": False}})


class IdentityBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.package = self.source / "src/depth_anything_3"
        self.package.mkdir(parents=True)
        self.module = self.package / "fixture.py"
        self.module.write_bytes(b"value = 1\n")
        self.model = self.root / "model"
        self.model.mkdir()
        self.weights = self.model / "model.safetensors"
        self.weights.write_bytes(b"weights")
        self.config = self.model / "config.json"
        self.config.write_bytes(b"{}")
        data = self.module.read_bytes()
        blob = worker.hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        self.tree = f"100644 blob {blob}\tsrc/depth_anything_3/fixture.py\n"
        versions = dict(row.split("==") for row in (worker.ROOT / "requirements.txt").read_text().splitlines())
        for mock in (
            patch.object(worker, "locations", return_value=(self.source, self.model)),
            patch.object(worker, "MODEL_HASH", worker.sha256(self.weights)),
            patch.object(worker, "CONFIG_HASH", worker.sha256(self.config)),
            patch.object(worker.subprocess, "check_output", side_effect=lambda args, **kwargs: worker.CODE_REVISION if "rev-parse" in args else self.tree),
            patch.object(worker.importlib.metadata, "version", side_effect=versions.__getitem__),
            patch.object(worker.importlib.metadata, "distributions", return_value=[]),
        ):
            mock.start()
            self.addCleanup(mock.stop)

    def test_identity_requires_actual_matching_checkpoint_bytes(self):
        identity = worker.identity()
        self.assertEqual(identity["device"], "cpu")
        self.weights.write_bytes(b"different weights")
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            worker.identity()

    def test_identity_rejects_source_edits_and_unreviewed_modules(self):
        self.module.write_bytes(b"value = 2\n")
        with self.assertRaisesRegex(ValueError, "modified"):
            worker.identity()
        self.module.write_bytes(b"value = 1\n")
        (self.package / "unexpected.py").touch()
        with self.assertRaisesRegex(ValueError, "Unreviewed"):
            worker.identity()

    def test_identity_rejects_dependency_drift(self):
        with patch.object(worker.importlib.metadata, "version", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "dependency"):
                worker.identity()


class CheckpointSharingTests(unittest.TestCase):
    def setUp(self):
        class Tensor:
            shape = (32,)

            def __init__(self, address):
                self.address = address

            def data_ptr(self):
                return self.address

            def stride(self):
                return (1,)

        class Model:
            def __init__(self, state):
                self.state = state
                self.loaded = None

            def state_dict(self):
                return self.state

            def load_state_dict(self, state, strict):
                if not strict or set(state) != set(self.state):
                    raise AssertionError("All checkpoint keys must be loaded strictly")
                self.loaded = state

        shared_weight, shared_bias = Tensor(101), Tensor(102)
        model_state = {}
        for alias, source in worker.CHECKPOINT_ALIASES.items():
            tensor = shared_weight if alias.endswith("weight") else shared_bias
            model_state[alias] = tensor
            model_state[source] = tensor
        self.model = Model(model_state)
        self.state = {f"model.{key}": value for key, value in model_state.items() if key not in worker.CHECKPOINT_ALIASES}
        self.metadata = {f"model.{alias}": f"model.{source}" for alias, source in worker.CHECKPOINT_ALIASES.items()}

    def test_exact_shared_aliases_are_restored_before_strict_loading(self):
        worker.load_checkpoint(self.model, self.state, self.metadata)
        for alias, source in worker.CHECKPOINT_ALIASES.items():
            self.assertIs(self.model.loaded[alias], self.model.loaded[source])

    def test_nonalias_missing_and_unexpected_keys_are_rejected(self):
        self.model.state["backbone.required"] = object()
        with self.assertRaisesRegex(ValueError, "missing or unexpected"):
            worker.load_checkpoint(self.model, self.state, self.metadata)
        del self.model.state["backbone.required"]
        with self.assertRaisesRegex(ValueError, "missing or unexpected"):
            worker.load_checkpoint(self.model, {**self.state, "model.extra": object()}, self.metadata)

    def test_unshared_alias_and_changed_metadata_are_rejected(self):
        alias = next(iter(worker.CHECKPOINT_ALIASES))
        self.model.state[alias] = type(self.model.state[alias])(999)
        with self.assertRaisesRegex(ValueError, "actual model storage"):
            worker.load_checkpoint(self.model, self.state, self.metadata)
        with self.assertRaisesRegex(ValueError, "sharing metadata"):
            worker.load_checkpoint(self.model, self.state, {})


class ConditioningBoundaryTests(unittest.TestCase):
    def setUp(self):
        RequestBoundaryTests.setUp(self)
        self.camera = {"mode": "intrinsics-only", "provenance": "experiment-oracle",
                       "cameraConvention": "opencv-world-to-camera-scene-y-up-meters",
                       "intrinsics": [[[350, 0, 319.5], [0, 350, 239.5], [0, 0, 1]]]}

    def test_explicit_intrinsics_and_none_are_accepted(self):
        worker.validate_request({**self.request, "conditioning": self.camera})
        worker.validate_request({**self.request, "conditioning": {"mode": "none"}})

    def test_conditioning_provenance_convention_and_modes_fail_closed(self):
        for key, value in (("mode", "auto"), ("provenance", "measured"),
                           ("cameraConvention", "camera-to-world"), ("groundTruth", {})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                worker.validate_request({**self.request, "conditioning": {**self.camera, key: value}})
        with self.assertRaises(ValueError):
            worker.validate_request({**self.request, "conditioning": {"mode": "none", "intrinsics": []}})

    def test_camera_count_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "count"):
            worker.validate_request({**self.request, "conditioning": {**self.camera, "intrinsics": []}})

    def test_malformed_intrinsics_are_rejected(self):
        for value in (True, float("nan"), float("inf"), 0, -350):
            camera = copy.deepcopy(self.camera)
            camera["intrinsics"][0][0][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                worker.validate_conditioning(camera, 1)
        for matrix in ([[1, 0], [0, 1]], [[350, 0, 0], [1, 350, 0], [0, 0, 1]],
                       [[350, 0, 0], [0, 350, 0], [0, 1, 1]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                worker.validate_conditioning({**self.camera, "intrinsics": [matrix]}, 1)

    def test_skew_and_degenerate_pose_are_rejected(self):
        skew = copy.deepcopy(self.camera)
        skew["intrinsics"][0][0][1] = .1
        with self.assertRaises(ValueError):
            worker.validate_conditioning(skew, 1)
        camera = {**self.camera, "mode": "pose", "intrinsics": self.camera["intrinsics"] * 3,
                  "extrinsics": [[[1, 0, 0, offset], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] for offset in (0, 1, 2)]}
        with self.assertRaisesRegex(ValueError, "noncollinear"):
            worker.validate_conditioning(camera, 3)

    def test_pose_requires_proper_finite_four_by_four_matrices(self):
        camera = {**self.camera, "mode": "pose", "intrinsics": self.camera["intrinsics"] * 3,
                  "extrinsics": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]] * 3}
        camera["extrinsics"] = [copy.deepcopy(matrix) for matrix in camera["extrinsics"]]
        camera["extrinsics"][1][0][3] = 1
        camera["extrinsics"][2][1][3] = 1
        worker.validate_conditioning(camera, 3)
        for row, column, value in ((0, 0, -1), (0, 0, 2), (3, 3, 0), (0, 3, float("inf")), (1, 1, True)):
            bad = copy.deepcopy(camera)
            bad["extrinsics"][0][row][column] = value
            with self.subTest(row=row, column=column, value=value), self.assertRaises(ValueError):
                worker.validate_conditioning(bad, 3)
        with self.assertRaises(ValueError):
            worker.validate_conditioning({**camera, "extrinsics": []}, 3)
        with self.assertRaises(ValueError):
            worker.validate_conditioning({**camera, "extrinsics": [row[:3] for row in camera["extrinsics"]]}, 3)
        with self.assertRaises(ValueError):
            worker.validate_conditioning({**camera, "intrinsics": camera["intrinsics"][:1], "extrinsics": camera["extrinsics"][:1]}, 1)


@unittest.skipUnless(importlib.util.find_spec("torch") and (worker.locations()[0] / "src").is_dir(),
                     "optional installed CPU preprocessing packages and pinned upstream source")
class OfficialConditioningMechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(worker.locations()[0] / "src"))

    def test_official_fixture_intrinsics_resize_without_crop(self):
        import numpy as np
        from PIL import Image
        from depth_anything_3.utils.io.input_processor import InputProcessor

        original = np.array([[[350, 0, 319.5], [0, 350, 239.5], [0, 0, 1]]], dtype=np.float32)
        extrinsic = np.eye(4, dtype=np.float32)[None]
        for resolution, width, height in ((256, 252, 196), (384, 378, 294)):
            with self.subTest(resolution=resolution):
                images, out_e, out_k = InputProcessor()(
                    [Image.new("RGB", (640, 480))], extrinsics=extrinsic.copy(), intrinsics=original.copy(),
                    process_res=resolution, process_res_method="upper_bound_resize", sequential=True, num_workers=1,
                )
                self.assertEqual(tuple(images.shape), (1, 3, height, width))
                expected = original.copy()
                expected[:, 0, :] *= width / 640
                expected[:, 1, :] *= height / 480
                np.testing.assert_allclose(out_k.numpy(), expected, rtol=1e-6)
                np.testing.assert_array_equal(out_e.numpy(), extrinsic)
                np.testing.assert_array_equal(original, [[[350, 0, 319.5], [0, 350, 239.5], [0, 0, 1]]])
                self.assertAlmostEqual(float(out_k[0, 0, 2]) - (width / 2 - .5), (1-width/640)/2, delta=3e-5)
                self.assertAlmostEqual(float(out_k[0, 1, 2]) - (height / 2 - .5), (1-height/480)/2, delta=3e-5)

    def test_normalization_matches_exact_pinned_api_method(self):
        import torch
        from depth_anything_3.utils.geometry import affine_inverse

        source = worker.locations()[0] / "src/depth_anything_3/api.py"
        api = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "DepthAnything3")
        method = next(node for node in api.body if isinstance(node, ast.FunctionDef) and node.name == "_normalize_extrinsics")
        namespace = {"torch": torch, "affine_inverse": affine_inverse}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        extrinsics = torch.eye(4).repeat(1, 4, 1, 1)
        extrinsics[0, :, 0, 3] = torch.tensor([1., 2., 3., 4.])
        extrinsics[0, :, 1, 3] = torch.tensor([0., .1, -.1, .2])
        actual, median = worker.normalize_extrinsics(extrinsics.clone())
        expected = namespace["_normalize_extrinsics"](None, extrinsics.clone())
        self.assertTrue(torch.equal(actual, expected))
        self.assertGreater(median, 0)
        self.assertTrue(torch.allclose(actual[0, 0], torch.eye(4)))

    def test_intrinsics_only_does_not_activate_official_camera_encoder(self):
        import torch
        from addict import Dict
        from depth_anything_3.model.da3 import DepthAnything3Net

        model = DepthAnything3Net.__new__(DepthAnything3Net)
        torch.nn.Module.__init__(model)
        calls = []
        model.cam_enc = lambda *args: calls.append(args) or torch.ones(1, 3, 384)
        model.backbone = lambda images, **kwargs: (calls.append(kwargs) or [], [])
        model._process_depth_head = lambda *args: Dict(depth=torch.ones(1, 3, 14, 14))
        model._process_camera_estimation = lambda feats, h, w, output: output
        images = torch.ones(1, 3, 3, 14, 14)
        intrinsics = torch.eye(3).repeat(1, 3, 1, 1)
        DepthAnything3Net.forward(model, images, intrinsics=intrinsics)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0]["cam_token"])
        calls.clear()
        DepthAnything3Net.forward(model, images, torch.eye(4).repeat(1, 3, 1, 1), intrinsics)
        self.assertEqual(len(calls), 2)
        self.assertIsNotNone(calls[1]["cam_token"])

    def test_official_scale_alignment_retains_model_depth_and_supplies_output_cameras(self):
        import numpy as np
        import torch
        from types import SimpleNamespace

        supplied = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
        supplied[:, :3, 3] = [[0, 0, 0], [-1, 0, 0], [0, -1, 0], [-1, -1, -.1]]
        predicted = supplied[:, :3].copy()
        predicted[:, :3, 3] *= 2
        depth = np.full((4, 2, 2), 8, dtype=np.float32)
        prediction = SimpleNamespace(depth=depth.copy(), extrinsics=predicted, intrinsics=np.zeros((4, 3, 3)))
        intrinsic = torch.eye(3).repeat(4, 1, 1)
        scale = worker.align_conditioned_prediction(prediction, torch.from_numpy(supplied), intrinsic)
        self.assertAlmostEqual(scale, 2, places=5)
        np.testing.assert_allclose(prediction.depth, 4)
        np.testing.assert_array_equal(depth, 8)
        np.testing.assert_array_equal(prediction.extrinsics, supplied[:, :3])
        np.testing.assert_array_equal(prediction.intrinsics, intrinsic.numpy())


if __name__ == "__main__":
    unittest.main()
