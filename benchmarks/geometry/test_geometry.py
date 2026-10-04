import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from evaluate import boundary_metrics, cross_view_metrics, evaluate, held_out_dimensions, load_prediction, points
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


    def worker_receipt(self, prediction, **conditioning):
        receipt = self.root / f"{prediction.stem}-worker.json"
        metadata = dict(mode="pose", provenance="experiment-oracle", cameraPredictionIndependent=False,
                        depthScaleSource="DA3 supplied-camera path", processedIntrinsics=self.truth["intrinsics"].tolist())
        metadata.update(conditioning)
        receipt.write_text(json.dumps(dict(artifactSha256=hashlib.sha256(prediction.read_bytes()).hexdigest(),
                                          coordinates=dict(units="meters"), conditioning=metadata)))
        return receipt

    def test_native_oracle_and_no_fit_of_perturbed_cameras(self):
        path = self.prediction("native-oracle")
        report = evaluate(self.fixture, path, self.root / "native-oracle-report", self.worker_receipt(path))
        self.assertLess(report["depth"]["NATIVE"]["rmse_m"], 1e-6)
        self.assertFalse(report["cameras"]["independent_model_accuracy"])
        self.assertIn("original_oriented_extent_meters", report["dimensions"][0]["axes"][0])
        self.assertNotIn("original_oriented_extent_relative_units", report["dimensions"][0]["axes"][0])
        self.assertLess(report["cross_view"]["NATIVE"]["reprojection_rmse_pixels"], 1e-4)
        self.assertLess(report["cross_view"]["NATIVE"]["duplicate_surface_scatter_rmse_cm"], .1)
        self.assertTrue((self.root / "native-oracle-report/NATIVE-depth-01.png").is_file())
        extrinsics = self.truth["extrinsics"].copy()
        extrinsics[0, :, 3] += [.2, 0, 0]
        path = self.prediction("native-perturbed", extrinsics=extrinsics)
        report = evaluate(self.fixture, path, self.root / "native-perturbed-report", self.worker_receipt(path))
        self.assertGreater(report["cross_view"]["NATIVE"]["reprojection_rmse_pixels"], 1)
        self.assertGreater(report["cross_view"]["NATIVE"]["duplicate_surface_scatter_rmse_cm"], 5)
        self.assertEqual(report["native_alignment"], "none: supplied camera frame and official DA3 metric scale retained; no rigid fit of perturbed cameras")

    def test_conditioning_receipt_cannot_claim_inferred_oracle_cameras(self):
        path = self.prediction("invalid-oracle")
        with self.assertRaisesRegex(ValueError, "oracle provenance"):
            evaluate(self.fixture, path, self.root / "invalid-oracle-report", self.worker_receipt(path, provenance="measured"))
        receipt = self.worker_receipt(path)
        payload = json.loads(receipt.read_text())
        payload["artifactSha256"] = "0" * 64
        receipt.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            evaluate(self.fixture, path, self.root / "mismatched-oracle-report", receipt)

    def test_conditioned_npz_requires_explicit_matching_receipt(self):
        path = self.root / "marked-conditioned.npz"
        np.savez_compressed(path, depths=self.truth["depths"], confidences=np.ones_like(self.truth["depths"]),
                            intrinsics=self.truth["intrinsics"], extrinsics=self.truth["extrinsics"], conditioning_mode=np.array("pose"))
        with self.assertRaisesRegex(ValueError, "explicit worker receipt"):
            evaluate(self.fixture, path, self.root / "missing-conditioned-receipt")
        with self.assertRaisesRegex(ValueError, "mode differs"):
            evaluate(self.fixture, path, self.root / "wrong-conditioned-receipt", self.worker_receipt(path, mode="none"))

    def test_visible_spans_exclude_calibration_and_missing_geometry(self):
        cloud = np.array([points(d, k, e) for d, k, e in zip(self.truth["depths"], self.truth["intrinsics"], self.truth["extrinsics"])])
        valid = np.ones(self.truth["depths"].shape, dtype=bool)
        dimensions = held_out_dimensions(cloud, cloud, self.truth["labels"], valid, self.metadata)
        room_depth = next(item for item in dimensions if item["id"] == "room")["axes"][2]
        self.assertFalse(room_depth["complete_axis_observed"])
        self.assertEqual(room_depth["held_out_absolute_error_cm"], 0)
        opening_width = next(item for item in dimensions if item["id"] == "opening-back")["axes"][0]
        self.assertTrue(opening_width["calibration_dimension_excluded"])
        self.assertFalse(opening_width["independent_accuracy_claim"])
        self.assertIsNone(opening_width["held_out_absolute_error_cm"])
        opening_plane = next(item for item in dimensions if item["id"] == "opening-back")["axes"][2]
        self.assertTrue(opening_plane["planarity_diagnostic"])
        self.assertFalse(opening_plane["independent_accuracy_claim"])
        self.assertIsNone(opening_plane["held_out_relative_error_percent"])
        self.assertLess(opening_plane["held_out_absolute_error_cm"], 1e-4)
        valid[self.truth["labels"] == 13] = False
        dimensions = held_out_dimensions(cloud, cloud, self.truth["labels"], valid, self.metadata)
        prop = next(item for item in dimensions if item["id"] == "right-seat-box")
        self.assertEqual(prop["predicted_valid_fraction_of_visible_samples"], 0)
        self.assertTrue(all(axis["held_out_absolute_error_cm"] is None for axis in prop["axes"]))

    def test_per_entity_metrics_preserve_global_correspondences_and_missing_coverage(self):
        cloud = np.array([points(d, k, e) for d, k, e in zip(self.truth["depths"], self.truth["intrinsics"], self.truth["extrinsics"])])
        valid = np.ones(self.truth["depths"].shape, dtype=bool)
        original = cross_view_metrics(cloud, cloud, self.truth["depths"], self.truth["labels"], valid, self.truth["intrinsics"], self.truth["extrinsics"])
        grouped = cross_view_metrics(cloud, cloud, self.truth["depths"], self.truth["labels"], valid, self.truth["intrinsics"], self.truth["extrinsics"], self.metadata["dimensions"])
        for key in ["truth_visible_correspondences", "valid_prediction_correspondences", "reprojection_rmse_pixels", "duplicate_surface_scatter_rmse_cm", "pairs"]:
            self.assertEqual(original[key], grouped[key])
        for name in ["tall-left-box", "foreground-stool", "right-seat-box"]:
            self.assertGreater(grouped["entities"][name]["truth_visible_correspondences"], 100)
            self.assertEqual(grouped["entities"][name]["prediction_coverage"], 1)
            self.assertLess(grouped["entities"][name]["duplicate_surface_scatter_rmse_cm"], .2)
        valid[self.truth["labels"] == 13] = False
        missing = cross_view_metrics(cloud, cloud, self.truth["depths"], self.truth["labels"], valid, self.truth["intrinsics"], self.truth["extrinsics"], self.metadata["dimensions"])
        prop = missing["entities"]["right-seat-box"]
        self.assertEqual(prop["truth_visible_correspondences"], grouped["entities"]["right-seat-box"]["truth_visible_correspondences"])
        self.assertEqual(prop["prediction_coverage"], 0)
        self.assertIsNone(prop["duplicate_surface_scatter_rmse_cm"])
        self.assertIsNone(prop["reprojection_rmse_pixels"])

    def test_cross_view_reports_missing_coverage_and_depth_distortion(self):
        cloud = np.array([points(d, k, e) for d, k, e in zip(self.truth["depths"], self.truth["intrinsics"], self.truth["extrinsics"])])
        valid = np.ones(self.truth["depths"].shape, dtype=bool)
        valid[0] = False
        missing = cross_view_metrics(cloud, cloud, self.truth["depths"], self.truth["labels"], valid, self.truth["intrinsics"], self.truth["extrinsics"])
        self.assertGreater(missing["prediction_coverage"], 0)
        self.assertLess(missing["prediction_coverage"], 1)
        self.assertEqual(missing["pairs"][0]["valid_prediction_correspondences"], 0)
        depths = self.truth["depths"].copy()
        depths[0] *= 1.2
        distorted_cloud = np.array([points(d, k, e) for d, k, e in zip(depths, self.truth["intrinsics"], self.truth["extrinsics"])])
        distorted = cross_view_metrics(distorted_cloud, cloud, self.truth["depths"], self.truth["labels"], np.ones_like(valid), self.truth["intrinsics"], self.truth["extrinsics"])
        self.assertEqual(distorted["prediction_coverage"], 1)
        self.assertGreater(distorted["duplicate_surface_scatter_rmse_cm"], 10)
        self.assertGreater(distorted["reprojection_rmse_pixels"], 1)


if __name__ == "__main__":
    unittest.main()
