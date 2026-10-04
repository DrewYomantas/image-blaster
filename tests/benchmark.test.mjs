import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { attachBenchmarkEvidence, demonstrateMeasurementOverride } from "../benchmarks/geometry/run.mjs";
import { emptyScene } from "../engine/scene.mjs";

test("scale alignment retains original relative evidence and synthetic measurement precedence without technical approval", async (t) => {
  const root = await mkdtemp(path.join(os.tmpdir(), "image-blaster-alignment-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const truthPath = path.join(root, "truth.json"), reportPath = path.join(root, "report.json"), alignedPath = path.join(root, "aligned.npz");
  for (const file of [truthPath, reportPath, alignedPath]) await writeFile(file, "SYNTHETIC UNIT FIXTURE");
  const scene = emptyScene("synthetic-test", [{ id: "model", kind: "model", uri: "fixture://mock-model", license: { id: "MIT", commercialUse: "allowed", attribution: "Synthetic unit test" } }]);
  const axes = ["width", "height", "depth"].map((axis, index) => ({ axis, aligned_observed_extent_m: [1.03, .82, 0][index], original_oriented_extent_relative_units: [.515, .41, 0][index], complete_axis_observed: index !== 2, truth_visibility_fraction: .9 }));
  const report = { scale_anchor: { id: "opening-width", known_m: 1.2, inferred_original_units: .6, fixed_scale: 2 }, alignment: { rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], translation_m: [0, 0, 0] }, dimensions: [{ id: "opening-back", axes }] };
  const aligned = await attachBenchmarkEvidence(scene, report, { truthPath, reportPath, alignedPath });
  assert.equal(aligned.surfaces[0].properties.originalRelativeExtents[0].value.width.value, .515);
  assert.equal(aligned.surfaces[0].dimensions.width[0].value, 1.03);
  assert.equal(aligned.sources.find((source) => source.id === "synthetic-fixture-anchor").kind, "user-input");
  const override = demonstrateMeasurementOverride(aligned, 1.2);
  assert.equal(override.scene.surfaces[0].dimensions.width[0].value, 1.03);
  assert.equal(override.evidence.resolvedMetres, 1.2);
  assert.equal(override.evidence.technicalBlocked, true);
  assert.equal(override.evidence.installationApproved, false);
});
