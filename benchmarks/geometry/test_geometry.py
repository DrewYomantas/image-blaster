import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from evaluate import boundary_metrics, evaluate, load_prediction
from fixture import generate, intersect, rays


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.fixture = cls.root / "fixture"
        cls.metadata = generate(cls.fixture)
        with np.load(cls.fixture / "ground-truth/truth.npz") as archive:
            cls.truth = {key: np.array(archive[key]) for key in archive}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def prediction(self, name, depths=None, extrinsics=None):
        path = self.root / f"{name}.npz"
        np.savez_compressed(path, depths=self.truth["depths"] if depths is None else depths,
                            confidences=np.ones(self.truth["depths"].shape), intrinsics=self.truth["intrinsics"],
                            extrinsics=self.truth["extrinsics"] if extrinsics is None else extrinsics)
        return path

    def test_deterministic_inputs_and_truth_separation(self):
        second = self.root / "regenerated"
        repeated = generate(second)
        self.assertEqual(self.metadata["input_hashes"], repeated["input_hashes"])
        for item in repeated["input_hashes"]:
            self.assertEqual(item["sha256"], hashlib.sha256((second / "inputs" / item["file"]).read_bytes()).hexdigest())
        self.assertEqual(sorted(p.name for p in (second / "inputs").iterdir()), ["manifest.json", "view-01.png", "view-02.png", "view-03.png", "view-04.png"])
        manifest = json.loads((second / "inputs/manifest.json").read_text())
        self.assertNotIn("extrinsics", manifest)
        self.assertNotIn("intrinsics", manifest)
        self.assertNotIn("labels", manifest)
        self.assertTrue(np.isfinite(self.truth["depths"]).all())
        self.assertTrue((self.truth["depths"] > 0).all())
        for label in [7, 9, 10, 11, 12, 13]:
            self.assertGreater(int((self.truth["labels"] == label).sum()), 100)

    def test_oracle_and_fixed_scale_gauge_recovery(self):
        angle, scale = .4, 2.5
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        translation = np.array([.8, -.2, .5])
        extrinsics = self.truth["extrinsics"].copy()
        extrinsics[:, :, :3] = extrinsics[:, :, :3] @ rotation.T
        extrinsics[:, :, 3] = self.truth["extrinsics"][:, :, 3] * scale - np.einsum("nij,j->ni", extrinsics[:, :, :3], translation)
        path = self.prediction("scaled-gauge", self.truth["depths"] * scale, extrinsics)
        report = evaluate(self.fixture, path, self.root / "scaled-report")
        self.assertAlmostEqual(report["scale_anchor"]["fixed_scale"], 1 / scale, places=4)
        self.assertLess(report["depth"]["ALIGNED"]["abs_rel"], 1e-4)
        self.assertGreater(report["depth"]["INFERENCE"]["abs_rel"], 1)
        self.assertLess(max(report["cameras"]["aligned_translation_error_cm"]), .01)
        self.assertLess(max(report["cameras"]["aligned_rotation_error_degrees"]), .01)
        self.assertGreater(report["boundaries"]["ALIGNED"]["f1"], .99)
        room = next(item for item in report["dimensions"] if item["id"] == "room")
        self.assertFalse(room["axes"][2]["complete_axis_observed"])

    def test_geometry_distortion_remains_after_anchor_alignment(self):
        depths = self.truth["depths"].copy()
        depths[self.truth["labels"] == 13] *= 1.4
        extrinsics = self.truth["extrinsics"].copy()
        extrinsics[0, :, 3] += [.2, 0, 0]
        report = evaluate(self.fixture, self.prediction("distorted", depths, extrinsics), self.root / "distorted-report")
        self.assertGreater(report["depth"]["ALIGNED"]["abs_rel"], .005)
        self.assertGreater(max(report["cameras"]["aligned_translation_error_cm"]), 5)

    def test_boundary_failure_and_invalid_rotation(self):
        target = np.ones((1, 20, 20))
        target[:, :, 10:] = 2
        result = boundary_metrics(np.ones_like(target), target, np.ones(target.shape, dtype=bool))
        self.assertEqual(result["f1"], 0)
        extrinsics = self.truth["extrinsics"].copy()
        extrinsics[0, 0, 0] *= 2
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            load_prediction(self.prediction("bad-rotation", extrinsics=extrinsics))

    def test_resize_only_oracle_and_local_opening_boundaries(self):
        width, height = 378, 280
        intrinsics = self.truth["intrinsics"].copy()
        sx, sy = width / 640, height / 480
        intrinsics[:, 0] *= sx
        intrinsics[:, 1] *= sy
        intrinsics[:, 0, 2] += (sx - 1) / 2
        intrinsics[:, 1, 2] += (sy - 1) / 2
        depths = []
        for k, e in zip(intrinsics, self.truth["extrinsics"]):
            origin, direction = rays(k, e, width, height)
            depth, _, _, _ = intersect(origin, direction, self.metadata["primitives"])
            depths.append(depth)
        path = self.root / "resize-oracle.npz"
        np.savez_compressed(path, depths=depths, confidences=np.ones((4, height, width)), intrinsics=intrinsics, extrinsics=self.truth["extrinsics"])
        report = evaluate(self.fixture, path, self.root / "resize-report")
        self.assertLess(report["depth"]["ALIGNED"]["abs_rel"], 1e-4)
        self.assertGreater(report["boundaries"]["fireplace_opening_ALIGNED"]["f1"], .99)
        self.assertEqual(report["focal"]["absolute_error_pixels"], [[0., 0.]] * 4)


if __name__ == "__main__":
    unittest.main()
