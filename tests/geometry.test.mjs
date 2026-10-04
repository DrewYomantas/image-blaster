import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile, mkdir } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { createDA3Provider } from "../engine/da3.mjs";
import { ProviderRegistry } from "../engine/providers.mjs";
import { runGeneration } from "../engine/run.mjs";
import { assertScene, assertEvidencePreserved, emptyScene, resolveFact } from "../engine/scene.mjs";
import { exportBenson } from "../engine/adapters.mjs";

const identity = { modelRevision: "e08cab65ca0ec38e7826075418411ab90cab4da3", checkpointSha256: "a".repeat(64), configSha256: "b".repeat(64), upstreamRevision: "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4", upstreamSourceSha256: "c".repeat(64), device: "cpu", dtype: "float32" };
const k = [[220, 0, 126], [0, 220, 98], [0, 0, 1]];
const e = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]];

async function fixture(t) {
  const root = await mkdtemp(path.join(os.tmpdir(), "image-blaster-geometry-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const inputs = [];
  for (let i = 0; i < 4; i++) {
    const file = path.join(root, `image-${i}.png`);
    await writeFile(file, `SYNTHETIC TEST IMAGE ${i}`);
    inputs.push({ path: file, mediaType: "image/png", license: { id: "CC0-1.0", commercialUse: "allowed", attribution: "Image Blaster synthetic test" } });
  }
  let calls = 0;
  const invoke = async (args) => {
    if (args[0] === "--identity") return structuredClone(identity);
    calls++;
    const request = JSON.parse(await readFile(args[1], "utf8"));
    assert.deepEqual(Object.keys(request).sort(), ["expectedIdentity", "inputs", "outputDir", "parameters"]);
    assert.deepEqual(Object.keys(request.parameters).sort(), ["processResolution", "profileWarmInference", "threads"]);
    assert.deepEqual(request.expectedIdentity, identity);
    assert.equal(request.parameters.processResolution, 256);
    assert.equal(request.inputs.length, 4);
    assert.deepEqual(Object.keys(request.inputs[0]), ["path"]);
    await mkdir(request.outputDir);
    const file = path.join(request.outputDir, "predictions.npz");
    await writeFile(file, "MOCK TENSORS, NOT REAL MODEL INFERENCE");
    return { identity, files: [file], cameras: request.inputs.map((_, sourceIndex) => ({ sourceIndex, intrinsics: k, extrinsics: e, imageWidth: 252, imageHeight: 196 })) };
  };
  const registry = new ProviderRegistry().register(createDA3Provider({ invoke }));
  return { root, inputs, registry, cacheDir: path.join(root, "cache"), calls: () => calls };
}

test("geometry worker boundary receives images only and emits relative inferred cameras/artifacts with cache reuse", async (t) => {
  const f = await fixture(t);
  const before = emptyScene("geometry-study");
  const request = { capability: "scene-geometry", inputs: f.inputs, scene: before };
  const first = await runGeneration(request, f);
  const scene = assertScene(first.result);
  assert.equal(scene.sources.filter((source) => source.kind === "image").length, 4);
  const model = scene.sources.find((source) => source.kind === "model");
  assert.ok(model.uri.endsWith(identity.modelRevision));
  assert.equal(model.sha256, identity.checkpointSha256);
  assert.equal(scene.cameras.length, 4);
  assert.equal(scene.cameras[0].pose[0].value.units, "relative");
  assert.equal(scene.cameras[0].pose[0].state, "inferred");
  assert.equal(scene.cameras[0].transform, undefined);
  assert.equal(scene.artifacts[0].role, "visual-only");
  assert.deepEqual(scene.artifacts[0].provider.checkpoint, identity);
  assert.equal(scene.artifacts[0].parameters.scale, "ambiguous");
  const repeat = await runGeneration(request, f);
  assert.equal(repeat.key, first.key);
  assert.equal(repeat.cached, true);
  assert.equal(f.calls(), 1);
  assert.equal(exportBenson(scene).installationApproved, false);
  assert.throws(() => exportBenson(scene, { purpose: "technical" }), /authoritative entities/);
});

test("structured camera evidence rejects nonphysical rotations, invalid projections and silent metric relabeling", async (t) => {
  const f = await fixture(t);
  const { result } = await runGeneration({ capability: "scene-geometry", inputs: f.inputs }, f);
  for (const mutate of [
    (s) => { s.cameras[0].projection[0].value.intrinsics[0][0] = -1; },
    (s) => { s.cameras[0].pose[0].value.worldToCamera[0][0] = -1; },
    (s) => { s.cameras[0].pose[0].value.units = "meters"; },
    (s) => { s.cameras[0].pose[0].state = "measured"; }
  ]) { const invalid = structuredClone(result); mutate(invalid); assert.throws(() => assertScene(invalid)); }
  const moved = structuredClone(result);
  const camera = moved.cameras.shift();
  moved.objects.push(camera);
  assert.throws(() => assertScene(moved), /cameras only/);
  const rewritten = structuredClone(result);
  rewritten.cameras[0].pose[0].value.worldToCamera[0][3] = 1;
  assert.throws(() => assertEvidencePreserved(result, rewritten), /camera evidence/);
});

test("mixed visual room preserves imperfect inference while the measured scoped opening width governs resolution", async (t) => {
  const f = await fixture(t);
  const { result: scene } = await runGeneration({ capability: "scene-geometry", inputs: f.inputs }, f);
  const modelId = scene.sources.find((source) => source.kind === "model").id;
  scene.sources.push({ id: "synthetic-measurement", kind: "field-measurement", uri: "fixture://benchmark/measurement-role-test", license: { id: "CC0-1.0", commercialUse: "allowed", attribution: "Synthetic benchmark, not a real field visit" } });
  const inferred = { id: "opening-inferred-width", value: 1.02, state: "inferred", sourceIds: [modelId], confidence: 0, note: "Synthetic imperfect visual estimate" };
  const measured = { id: "opening-measured-width", value: 1.2, state: "measured", sourceIds: ["synthetic-measurement"], confidence: 1, note: "Synthetic supplied measurement to test authority precedence" };
  scene.objects.push({ id: "visual-opening", label: { ...inferred, id: "opening-label", value: "Synthetic visual opening" }, dimensions: { width: [inferred, measured] }, geometry: { role: "visual-only", artifactId: scene.artifacts[0].id, sourceIds: [modelId] } });
  assertScene(scene);
  assert.equal(scene.objects[0].dimensions.width[0].value, 1.02);
  assert.equal(resolveFact(scene.objects[0].dimensions.width, { requireAuthoritative: true }).value, 1.2);
  const visualization = exportBenson(scene);
  assert.equal(visualization.entities[0].dimensionEvidence.width.length, 2);
  assert.equal(visualization.entities[0].resolvedDimensions.width.value, 1.2);
  assert.equal(visualization.installationApproved, false);
  assert.throws(() => exportBenson(scene, { purpose: "technical" }), /visual or missing geometry/);
});
