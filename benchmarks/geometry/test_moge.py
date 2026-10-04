import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from evaluate import held_out_dimensions
from evaluate_moge import evaluate_moge, load_moge, surface_membership, surface_metrics, truth_geometry, world_points
from fixture import generate


class MoGeGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.fixture = cls.root / "fixture"
        generate(cls.fixture)
        cls.truth = truth_geometry(cls.fixture, 640, 480)
        cls.membership = surface_membership(cls.truth["cloud"], cls.truth["labels"], cls.truth["normals"], cls.truth["metadata"])
        cls.camera_points = np.array([cloud @ e[:, :3].T + e[:, 3] for cloud, e in zip(cls.truth["cloud"], cls.truth["extrinsics"])])
        cls.normal_camera = np.array([normal @ e[:, :3].T for normal, e in zip(cls.truth["normals"], cls.truth["extrinsics"])])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def prediction(self, name, **overrides):
        k = self.truth["intrinsics"].copy()
        k[:, :2, 2] += .5
        k[:, 0] /= 640
        k[:, 1] /= 480
        data = dict(camera_points=self.camera_points, depths=self.truth["depths"], masks=np.ones((4, 480, 640), dtype=bool),
                    normals=self.normal_camera, intrinsics=self.truth["intrinsics"], intrinsics_normalized=k, conditioning_mode=np.asarray("none"))
        data.update(overrides)
        path = self.root / f"{name}.npz"
        np.savez_compressed(path, **data)
        return path

    def receipt(self, prediction, mode="none"):
        path = self.root / f"{prediction.stem}-worker.json"
        path.write_text(json.dumps(dict(artifactSha256=hashlib.sha256(prediction.read_bytes()).hexdigest(),
                                       inputs=[dict(sourceIndex=index, sha256=item["sha256"]) for index, item in enumerate(self.truth["metadata"]["input_hashes"])],
                                       conditioning=dict(mode=mode), parameters=dict(forceProjection=True, applyMask=True),
                                       coordinates=dict(units="meters", cameraConvention="opencv", axes="x-right,y-down,z-forward", depth="camera-z",
                                                        normal="camera-facing signed unit vectors", worldFrame="independent-camera-frames"))))
        return path

    def test_oracle_evaluation_and_direct_point_map_contract(self):
        prediction = self.prediction("oracle")
        report = evaluate_moge(self.fixture, prediction, self.root / "oracle-report", self.receipt(prediction))
        self.assertLess(report["depth"]["NATIVE"]["rmse_m"], 1e-12)
        self.assertLess(report["cross_view"]["NATIVE"]["reprojection_rmse_pixels"], 1e-10)
        self.assertEqual(report["focal"]["absolute_error_pixels"], [[0., 0.]] * 4)
        self.assertTrue(report["focal"]["independent_accuracy_claim"])
        self.assertAlmostEqual(report["scale_anchor"]["fixed_scale"], 1, places=4)
        self.assertEqual(len(report["per_view"]["NATIVE"]), 4)
        self.assertTrue((self.root / "oracle-report/NATIVE-normal-01.png").is_file())
        self.assertTrue((self.root / "oracle-report/NATIVE-plane-residual-01.png").is_file())
        self.assertEqual(report["alignment"], "none; no camera fit, ICP, deformation or per-view/per-object scale")

    def test_camera_and_normal_world_conventions(self):
        recovered = world_points(self.camera_points, self.truth["extrinsics"])
        self.assertLess(np.max(np.abs(recovered - self.truth["cloud"])), 1e-12)
        recovered_normals = np.array([normal @ e[:, :3] for normal, e in zip(self.normal_camera, self.truth["extrinsics"])])
        self.assertLess(np.max(np.abs(recovered_normals - self.truth["normals"])), 1e-12)
        for camera_points, camera_normals in zip(self.camera_points, self.normal_camera):
            self.assertTrue(((camera_points * camera_normals).sum(-1) < 1e-9).all())

    def test_normalized_intrinsic_mapping_and_malformed_arrays(self):
        data = load_moge(self.prediction("mapped"))
        self.assertEqual(data["intrinsics"][0, 0, 0], 350)
        self.assertEqual(data["intrinsics"][0, 1, 1], 350)
        self.assertEqual(data["intrinsics"][0, 0, 2], 319.5)
        self.assertEqual(data["intrinsics"][0, 1, 2], 239.5)
        with self.assertRaisesRegex(ValueError, "boolean"):
            load_moge(self.prediction("bad-mask", masks=np.ones((4, 480, 640), dtype=np.uint8)))
        with self.assertRaisesRegex(ValueError, "four original"):
            load_moge(self.prediction("bad-size", depths=np.ones((4, 240, 320))))
        k = self.truth["intrinsics"].copy()
        k[:, 0, 2] += .5
        with self.assertRaisesRegex(ValueError, "integer-centre"):
            load_moge(self.prediction("bad-principal-point", intrinsics=k))

    def test_known_plane_oracle_residual_and_signed_normals(self):
        valid = np.ones(self.truth["depths"].shape, dtype=bool)
        report, _, _ = surface_metrics(self.truth["cloud"], valid, self.membership, self.truth["normals"])
        for name in ["front-wall", "floor", "opening-back", "hearth", "mantel"]:
            group = report["groups"][name]
            self.assertGreater(group["plane"]["valid_samples"], 100)
            self.assertLess(group["plane"]["rms_distance_cm"], 1e-10)
            self.assertLess(group["normals"]["mean_angular_error_degrees"], 1e-6)
        inverted, _, _ = surface_metrics(self.truth["cloud"], valid, self.membership, -self.truth["normals"])
        self.assertAlmostEqual(inverted["groups"]["front-wall"]["normals"]["mean_angular_error_degrees"], 180)
        self.assertEqual(inverted["groups"]["front-wall"]["normals"]["fraction_within_15_degrees"], 0)

    def test_plane_bias_and_scatter_are_distinct_without_refitting(self):
        valid = np.ones(self.truth["depths"].shape, dtype=bool)
        cloud = self.truth["cloud"] + self.truth["normals"] * .1
        report, _, _ = surface_metrics(cloud, valid, self.membership)
        for name in ["front-wall", "floor", "opening-back", "hearth", "mantel"]:
            plane = report["groups"][name]["plane"]
            self.assertAlmostEqual(plane["signed_mean_cm"], 10)
            self.assertAlmostEqual(plane["rms_distance_cm"], 10)
            self.assertLess(plane["scatter_std_cm"], 1e-10)

    def test_gt_only_face_neighbourhood_and_no_predicted_tail_trim(self):
        face_ids, eligible, _ = self.membership
        self.assertFalse(eligible[:, 0].any())
        self.assertFalse(eligible[:, -1].any())
        self.assertFalse(eligible[:, :, 0].any())
        self.assertFalse(eligible[:, :, -1].any())
        for dy, dx in [(0, 1), (1, 0), (1, 1), (-1, -1)]:
            adjacent = np.roll(face_ids, (dy, dx), axis=(1, 2))
            self.assertTrue((adjacent[eligible] == face_ids[eligible]).all())
        cloud = self.truth["cloud"].copy()
        selected = eligible & (self.truth["labels"] == 7)
        cloud[selected] += self.truth["normals"][selected] * 100
        report, _, _ = surface_metrics(cloud, np.ones(eligible.shape, dtype=bool), self.membership)
        self.assertAlmostEqual(report["groups"]["opening-back"]["plane"]["rms_distance_cm"], 10000)
        self.assertEqual(report["groups"]["opening-back"]["plane"]["prediction_coverage"], 1)

    def test_masked_geometry_and_missing_normals_remain_missing(self):
        valid = np.ones(self.truth["depths"].shape, dtype=bool)
        valid[self.truth["labels"] == 9] = False
        report, _, _ = surface_metrics(self.truth["cloud"], valid, self.membership, np.zeros_like(self.truth["normals"]))
        self.assertEqual(report["groups"]["hearth"]["plane"]["prediction_coverage"], 0)
        self.assertIsNone(report["groups"]["hearth"]["plane"]["rms_distance_cm"])
        self.assertEqual(report["groups"]["front-wall"]["normals"]["prediction_coverage"], 0)
        self.assertIsNone(report["groups"]["front-wall"]["normals"]["mean_angular_error_degrees"])
        dimensions = held_out_dimensions(self.truth["cloud"], self.truth["cloud"], self.truth["labels"], valid, self.truth["metadata"])
        hearth = next(item for item in dimensions if item["id"] == "hearth")
        self.assertEqual(hearth["samples"], 0)
        self.assertTrue(all(axis["held_out_absolute_error_cm"] is None for axis in hearth["axes"]))

    def test_direct_camera_point_changes_affect_bounds_without_depth_change(self):
        camera_points = self.camera_points.copy()
        camera_points[..., 0][self.truth["labels"] == 13] += .5
        cloud = world_points(camera_points, self.truth["extrinsics"])
        original = held_out_dimensions(self.truth["cloud"], self.truth["cloud"], self.truth["labels"], np.ones(self.truth["labels"].shape, dtype=bool), self.truth["metadata"])
        changed = held_out_dimensions(cloud, self.truth["cloud"], self.truth["labels"], np.ones(self.truth["labels"].shape, dtype=bool), self.truth["metadata"])
        before = next(item for item in original if item["id"] == "right-seat-box")
        after = next(item for item in changed if item["id"] == "right-seat-box")
        self.assertGreater(abs(before["axes"][0]["observed_extent_m"] - after["axes"][0]["observed_extent_m"]), .01)
        self.assertTrue(np.array_equal(camera_points[..., 2], self.camera_points[..., 2]))

    def test_conditioning_mode_and_oracle_fov_receipts_fail_closed(self):
        prediction = self.prediction("known-fov", conditioning_mode=np.asarray("known-fov"))
        receipt = self.receipt(prediction)
        with self.assertRaisesRegex(ValueError, "mode missing or differs"):
            evaluate_moge(self.fixture, prediction, self.root / "mode-mismatch", receipt)
        receipt = self.receipt(prediction, "known-fov")
        with self.assertRaisesRegex(ValueError, "four ordered finite oracle"):
            evaluate_moge(self.fixture, prediction, self.root / "missing-oracle-fov", receipt)
        payload = json.loads(receipt.read_text())
        payload["conditioning"].update(provenance="experiment-oracle", horizontalFovDegrees=[90.] * 4)
        receipt.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "differ from output"):
            evaluate_moge(self.fixture, prediction, self.root / "wrong-oracle-fov", receipt)
        prediction = self.prediction("missing-mode", conditioning_mode=np.asarray(["none"]))
        with self.assertRaisesRegex(ValueError, "NPZ conditioning mode"):
            load_moge(prediction)

    def test_receipt_coordinate_convention_cannot_be_guessed(self):
        prediction = self.prediction("coordinate-convention")
        receipt = self.receipt(prediction)
        payload = json.loads(receipt.read_text())
        payload["coordinates"]["worldFrame"] = "scene-y-up"
        receipt.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "independent camera points"):
            evaluate_moge(self.fixture, prediction, self.root / "wrong-coordinates", receipt)
        payload["coordinates"]["worldFrame"] = "independent-camera-frames"
        payload["coordinates"]["normal"] = "unoriented"
        receipt.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "camera-facing convention"):
            evaluate_moge(self.fixture, prediction, self.root / "wrong-normal-sign", receipt)

    def test_receipt_hash_fails_closed(self):
        prediction = self.prediction("receipt-hash")
        receipt = self.receipt(prediction)
        payload = json.loads(receipt.read_text())
        payload["artifactSha256"] = "0" * 64
        receipt.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            evaluate_moge(self.fixture, prediction, self.root / "bad-receipt", receipt)


if __name__ == "__main__":
    unittest.main()
