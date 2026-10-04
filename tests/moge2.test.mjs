import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile, mkdir, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { createMoGe2Provider, normalizeFovConditioning } from "../engine/moge2.mjs";
import { ProviderRegistry } from "../engine/providers.mjs";
import { runGeneration } from "../engine/run.mjs";
import { assertScene, resolveFact } from "../engine/scene.mjs";
import { exportBenson } from "../engine/adapters.mjs";

const fov = { mode: "known-fov", provenance: "experiment-oracle", horizontalFovDegrees: [84.87] };
async function fixture(t) {
  const root = await mkdtemp(path.join(os.tmpdir(), "image-blaster-moge2-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const image = path.join(root, "source.png");
  await writeFile(image, "MODEL-FREE TEST IMAGE");
  const identity = { modelRevision: "a".repeat(40), checkpointSha256: "b".repeat(64), upstreamRevision: "c".repeat(40), device: "cpu", dtype: "float32" };
  let calls = 0;
  const invoke = async (args) => {
    if (args[0] === "--identity") return structuredClone(identity);
    calls++;
    const req = JSON.parse(await readFile(args[1], "utf8"));
    assert.deepEqual(Object.keys(req).sort(), ["expectedIdentity", "inputs", "outputDir", "parameters", ...(req.conditioning ? ["conditioning"] : [])].sort());
    assert.deepEqual(Object.keys(req.inputs[0]), ["path"]);
    await mkdir(req.outputDir);
    const file = path.join(req.outputDir, "predictions.npz");
    await writeFile(file, "MOCK POINT MAP, NOT MODEL OUTPUT");
    return { identity: structuredClone(identity), conditioning: req.conditioning, files: [file], cameras: [{ sourceIndex: 0, intrinsics: [[350, 0, 319.5], [0, 350, 239.5], [0, 0, 1]], imageWidth: 640, imageHeight: 480 }] };
  };
  const workerSource = path.join(root, "worker.py");
  await writeFile(workerSource, "worker-version-1");
  return { root, identity, workerSource, registry: new ProviderRegistry().register(createMoGe2Provider({ invoke, implementationFiles: [workerSource] })), cacheDir: path.join(root, "cache"), request: { capability: "scene-geometry", providerId: "moge2-vits-normal", inputs: [{ path: image }] }, calls: () => calls };
}

test("MoGe FOV validation rejects implicit truth, nonfinite angles and image mismatch", () => {
  assert.equal(normalizeFovConditioning({ mode: "none" }, 1), undefined);
  assert.deepEqual(normalizeFovConditioning(fov, 1), fov);
  for (const value of [null, [], { ...fov, extrinsics: [] }, { ...fov, provenance: "measured" }, { ...fov, horizontalFovDegrees: [] }, { ...fov, horizontalFovDegrees: [NaN] }, { ...fov, horizontalFovDegrees: [0] }, { ...fov, horizontalFovDegrees: [179] }, { mode: "none", depth: [] }]) assert.throws(() => normalizeFovConditioning(value, 1));
});

test("MoGe cache separates FOV, parameters, checkpoint and worker while ignoring evaluation metadata", async (t) => {
  const f = await fixture(t);
  const first = await runGeneration(f.request, f);
  assert.equal((await runGeneration({ ...f.request, evaluation: { labels: "changed" } }, f)).cached, true);
  const conditioned = await runGeneration({ ...f.request, conditioning: fov }, f);
  assert.notEqual(first.key, conditioned.key);
  assert.equal((await runGeneration({ ...f.request, conditioning: fov }, f)).cached, true);
  const changedFov = await runGeneration({ ...f.request, conditioning: { ...fov, horizontalFovDegrees: [85] } }, f);
  const changedParams = await runGeneration({ ...f.request, parameters: { resolutionLevel: 8 } }, f);
  f.identity.modelRevision = "d".repeat(40);
  const changedModel = await runGeneration(f.request, f);
  await writeFile(f.workerSource, "worker-version-2");
  const changedWorker = await runGeneration(f.request, f);
  assert.equal(new Set([first.key, conditioned.key, changedFov.key, changedParams.key, changedModel.key, changedWorker.key]).size, 6);
  assert.equal(f.calls(), 6);
});

test("MoGe native metric point evidence remains inferred, pose-free and visual-only", async (t) => {
  const f = await fixture(t);
  const { result: scene } = await runGeneration(f.request, f);
  assertScene(scene);
  assert.equal(scene.cameras[0].pose, undefined);
  assert.equal(scene.cameras[0].projection[0].state, "inferred");
  assert.equal(scene.artifacts[0].parameters.units, "meters");
  assert.equal(scene.artifacts[0].parameters.evidence, "inferred");
  assert.equal(scene.artifacts[0].role, "visual-only");
  assert.equal(exportBenson(scene).installationApproved, false);
  assert.throws(() => exportBenson(scene, { purpose: "technical" }));
});

test("MoGe oracle FOV is supplied calibration and cannot promote metric geometry authority", async (t) => {
  const f = await fixture(t);
  const { result: scene } = await runGeneration({ ...f.request, conditioning: fov }, f);
  assert.equal(scene.cameras[0].projection[0].state, "user-specified");
  assert.ok(scene.sources.some((s) => s.kind === "user-input"));
  assert.ok(scene.sources.every((s) => !["field-measurement", "manufacturer-document", "registered-geometry"].includes(s.kind)));
  assert.equal(scene.artifacts[0].parameters.evidence, "inferred");
  assert.throws(() => resolveFact(scene.cameras[0].projection, { requireAuthoritative: true }));
});

test("MoGe preserves inferred dimension alongside authoritative synthetic measured override", async (t) => {
  const f = await fixture(t);
  const { result: scene } = await runGeneration(f.request, f);
  const modelId = scene.sources.find((s) => s.kind === "model").id;
  scene.sources.push({ id: "measurement", kind: "field-measurement", uri: "fixture://authority-regression", license: { id: "CC0-1.0", commercialUse: "allowed", attribution: "Synthetic authority test" } });
  const inferred = { id: "inferred-width", value: 1.02, state: "inferred", sourceIds: [modelId], confidence: 0 };
  const measured = { id: "measured-width", value: 1.2, state: "measured", sourceIds: ["measurement"], confidence: 1 };
  scene.objects.push({ id: "opening", label: { ...inferred, id: "label", value: "Synthetic opening" }, dimensions: { width: [inferred, measured] }, geometry: { role: "visual-only", artifactId: scene.artifacts[0].id, sourceIds: [modelId] } });
  assertScene(scene);
  assert.equal(resolveFact(scene.objects[0].dimensions.width, { requireAuthoritative: true }).value, 1.2);
  assert.equal(scene.objects[0].dimensions.width[0].value, 1.02);
  assert.throws(() => exportBenson(scene, { purpose: "technical" }), /visual or missing geometry/);
});

test("MoGe bounds numeric parameters before calling inference", async (t) => {
  const f = await fixture(t);
  for (const parameters of [{ resolutionLevel: -1 }, { resolutionLevel: 10 }, { resolutionLevel: 1.5 }, { threads: 0 }, { threads: 65 }]) await assert.rejects(runGeneration({ ...f.request, parameters }, f), /integer resolutionLevel/);
  assert.equal(f.calls(), 0);
});
