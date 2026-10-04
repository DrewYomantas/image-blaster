import importlib.util
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


if __name__ == "__main__":
    unittest.main()
