import test from "node:test";
import assert from "node:assert/strict";
import { assertEvidencePreserved, assertScene, resolveFact, emptyScene } from "../engine/scene.mjs";
import { exportBenson, exportTPS } from "../engine/adapters.mjs";
import { authoritativeScene, fact, sceneFixture } from "./fixtures.mjs";

test("SceneSpec v1 validates empty and source-backed scenes", () => {
  assertScene(emptyScene("empty"));
  assertScene(sceneFixture());
});

test("measured dimensions override inference without deleting original evidence", () => {
  const scene = sceneFixture();
  const before = structuredClone(scene);
  assert.equal(resolveFact(scene.objects[0].dimensions.width).value, 0.9);
  const exported = exportBenson(scene);
  assert.equal(exported.entities[0].resolvedDimensions.width.state, "measured");
  assert.equal(exported.entities[0].dimensionEvidence.width.length, 2);
  assert.deepEqual(scene, before);
  assert.equal(exported.installationApproved, false);
});

test("conflicting measured/manufacturer values fail closed irrespective of confidence", () => {
  const scene = sceneFixture();
  scene.objects[0].dimensions.width.push(fact("width-manual", 1.0, "manufacturer-specified", ["manual"], 1));
  assert.throws(() => assertScene(scene), /Conflicting authoritative/);
});

test("conflicting authoritative properties fail closed", () => {
  const scene = sceneFixture();
  scene.objects[0].properties = { finish: [fact("finish-1", "a", "measured", ["measure"]), fact("finish-2", "b", "measured", ["measure"])] };
  assert.throws(() => assertScene(scene), /Conflicting authoritative/);
});

test("model evidence cannot masquerade as measurement or geometry authority", () => {
  for (const sourceIds of [["generator"], ["generator", "measure"]]) {
    const scene = sceneFixture();
    scene.objects[0].dimensions.width[1].sourceIds = sourceIds;
    assert.throws(() => assertScene(scene), /Model evidence|field-measurement/);
  }
  const scene = authoritativeScene();
  scene.objects[0].geometry.sourceIds.push("generator");
  assert.throws(() => assertScene(scene), /visual-only/);
});

test("authoritative artifact must match registered source bytes", () => {
  const scene = authoritativeScene();
  scene.artifacts[0].sha256 = "b".repeat(64);
  delete scene.objects[0].geometry;
  assert.throws(() => assertScene(scene), /registered geometry hash/);
});

test("invalid versions, references, IDs, confidence, dimensions and transforms are rejected", () => {
  const mutations = [
    (scene) => { scene.schemaVersion = 2; },
    (scene) => { scene.sources[0].id = "room"; },
    (scene) => { scene.objects[0].label.confidence = 1.1; },
    (scene) => { scene.objects[0].label.sourceIds = ["missing"]; },
    (scene) => { scene.objects[0].dimensions.width[0].value = -1; },
    (scene) => { scene.objects[0].transform[0].value.translation = [0, NaN, 0]; },
    (scene) => { scene.objects[0].geometry = { role: "visual-only", artifactId: "missing", sourceIds: ["photo"] }; },
    (scene) => { scene.relationships.push({ from: "opening", to: "missing", relation: fact("rel", "near") }); },
    (scene) => { scene.validations.push({ target: "missing", check: "qa", status: "pass", details: "" }); }
  ];
  for (const mutate of mutations) { const scene = sceneFixture(); mutate(scene); assert.throws(() => assertScene(scene)); }
});

test("Benson technical export rejects inferred or unvalidated geometry", () => {
  assert.throws(() => exportBenson(sceneFixture(), { purpose: "technical" }), /visual or missing geometry/);
  const scene = authoritativeScene();
  assert.equal(exportBenson(scene, { purpose: "technical" }).installationApproved, false);
  scene.objects[0].dimensions.height[0].state = "inferred";
  assert.throws(() => exportBenson(scene, { purpose: "technical" }), /Authoritative evidence/);
  const unvalidated = authoritativeScene();
  unvalidated.validations = [];
  assert.throws(() => exportBenson(unvalidated, { purpose: "technical" }), /registered geometry validation/);
});

test("adapter manifests preserve transform/property evidence and TPS authority", () => {
  const scene = sceneFixture();
  scene.objects[0].transform.push(fact("transform-inferred", { translation: [1, 0, 0], rotation: [0, 0, 0], scale: [1, 1, 1] }));
  scene.objects[0].properties = { condition: [fact("condition", "weathered")] };
  const tps = exportTPS(scene);
  assert.equal(tps.terrainAuthority, "TPS");
  assert.equal(tps.entities[0].collision.gameplayApproved, false);
  assert.deepEqual(tps.entities[0].transformEvidence, scene.objects[0].transform);
  assert.deepEqual(tps.entities[0].properties, scene.objects[0].properties);
});

test("authoritative transforms compare by semantic JSON, not key insertion order", () => {
  const scene = sceneFixture();
  scene.objects[0].transform.push(fact("reordered-transform", { scale: [1, 1, 1], rotation: [0, 0, 0], translation: [0, 0, 0] }, "manufacturer-specified", ["manual"]));
  assertScene(scene);
});

test("geometry cannot bind authority to a different registered artifact", () => {
  const scene = authoritativeScene();
  scene.sources.push({ ...scene.sources.find((source) => source.id === "registered"), id: "other-registered", sha256: "b".repeat(64) });
  scene.objects[0].geometry.sourceIds = ["other-registered"];
  assert.throws(() => assertScene(scene), /bind to the referenced/);
});

test("generation preserves source bytes and fact ownership, rather than only IDs", () => {
  const original = sceneFixture();
  const changedSource = structuredClone(original);
  changedSource.sources[0].sha256 = "b".repeat(64);
  assert.throws(() => assertEvidencePreserved(original, changedSource), /provenance/);
  const movedAxis = structuredClone(original);
  [movedAxis.objects[0].dimensions.width, movedAxis.objects[0].dimensions.height] = [movedAxis.objects[0].dimensions.height, movedAxis.objects[0].dimensions.width];
  assert.throws(() => assertEvidencePreserved(original, movedAxis), /ownership and dimension axes/);
});
