import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile, mkdir, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { normalizeConditioning } from "../engine/conditioning.mjs";
import { createDA3Provider } from "../engine/da3.mjs";
import { runGeneration } from "../engine/run.mjs";
import { ProviderRegistry } from "../engine/providers.mjs";
import { assertScene } from "../engine/scene.mjs";
import { exportBenson } from "../engine/adapters.mjs";

const k = [[350, 0, 319.5], [0, 350, 239.5], [0, 0, 1]];
const camera = (x, y) => [[1, 0, 0, -x || 0], [0, 1, 0, -y || 0], [0, 0, 1, 0], [0, 0, 0, 1]];
const oracle = () => ({ mode: "pose", provenance: "experiment-oracle", cameraConvention: "opencv-world-to-camera-scene-y-up-meters", intrinsics: [k, k, k].map((value) => structuredClone(value)), extrinsics: [camera(0, 0), camera(1, 0), camera(0, 1)] });

test("conditioning rejects malformed, reflected, nonfinite, mismatched and mislabeled cameras", () => {
  assert.equal(normalizeConditioning(undefined, 3), undefined);
  assert.equal(normalizeConditioning({ mode: "none" }, 3), undefined);
  for (const mutate of [
    (c) => { c.intrinsics.pop(); }, (c) => { c.extrinsics[0][3][3] = 0; },
    (c) => { c.intrinsics[0][0][0] = NaN; }, (c) => { c.intrinsics[0][0][0] = -1; },
    (c) => { c.extrinsics[0][0][0] = -1; }, (c) => { c.extrinsics[0][0][0] = 2; },
    (c) => { c.cameraConvention = "camera-to-world"; }, (c) => { c.provenance = "measured"; },
    (c) => { c.depth = [1]; }, (c) => { c.extrinsics = [camera(0, 0), camera(1, 0), camera(2, 0)]; },
    (c) => { c.mode = "intrinsics-only"; }, (c) => { c.mode = "none"; }
  ]) { const c = oracle(); mutate(c); assert.throws(() => normalizeConditioning(c, 3)); }
  assert.deepEqual(normalizeConditioning(oracle(), 3), oracle());
});

test("oracle cache keys separate camera bytes and modes while evaluation metadata hits; cameras never gain authority", async (t) => {
  const root = await mkdtemp(path.join(os.tmpdir(), "da3-conditioning-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const inputs = [];
  for (let i = 0; i < 3; i++) {
    const file = path.join(root, `${i}.png`);
    await writeFile(file, `mock image ${i}`);
    inputs.push({ path: file });
  }
  const identity = { modelRevision: "fixed", checkpointSha256: "a".repeat(64) };
  let calls = 0;
  const invoke = async (args) => {
    if (args[0] === "--identity") return identity;
    calls++;
    const request = JSON.parse(await readFile(args[1], "utf8"));
    assert.equal(request.depth, undefined);
    assert.equal(request.labels, undefined);
    assert.equal(request.evaluationMetadata, undefined);
    const c = request.conditioning;
    await mkdir(request.outputDir);
    const file = path.join(request.outputDir, "predictions.npz");
    await writeFile(file, "mock only");
    return { identity, files: [file], conditioning: c ? { ...c, depthScaleSource: c.mode === "pose" ? "DA3-supplied-camera-path" : "none" } : { mode: "none" },
      rawCameraPredictions: { independent: false, intrinsics: inputs.map(() => k), extrinsics: inputs.map(() => camera(0, 0).slice(0, 3)) },
      cameras: inputs.map((_, sourceIndex) => ({ sourceIndex, intrinsics: c?.mode === "pose" ? c.intrinsics[sourceIndex] : k, extrinsics: (c?.mode === "pose" ? c.extrinsics[sourceIndex] : camera(0, 0)).slice(0, 3), imageWidth: 640, imageHeight: 480 })) };
  };
  const options = { registry: new ProviderRegistry().register(createDA3Provider({ invoke })), cacheDir: path.join(root, "cache") };
  const request = { capability: "scene-geometry", providerId: "da3-small", inputs, conditioning: oracle() };
  const c = await runGeneration(request, options);
  const repeat = await runGeneration({ ...request, evaluationMetadata: { note: "does not affect inference", openingWidth: 1.2 } }, options);
  assert.equal(repeat.cached, true);
  assert.equal(repeat.key, c.key);
  assert.equal(calls, 1);
  const changedK = structuredClone(request);
  changedK.conditioning.intrinsics[0][0][0] += 1;
  const changedE = structuredClone(request);
  changedE.conditioning.extrinsics[0][0][3] += .01;
  const b = structuredClone(request);
  b.conditioning.mode = "intrinsics-only";
  delete b.conditioning.extrinsics;
  const results = [c, await runGeneration(changedK, options), await runGeneration(changedE, options), await runGeneration(b, options), await runGeneration({ ...request, conditioning: undefined }, options)];
  assert.equal(new Set(results.map((r) => r.key)).size, 5);
  const manifest = JSON.parse(await readFile(c.manifestPath, "utf8"));
  assert.deepEqual(manifest.request.conditioning, oracle());
  const scene = assertScene(c.result);
  assert.equal(scene.cameras[0].pose[0].state, "user-specified");
  assert.equal(scene.cameras[0].pose[0].value.scale, "supplied-camera");
  assert.equal(scene.cameras[0].properties.conditionedCameraDecoder[0].value.independentCameraAccuracyEvidence, false);
  assert.ok(scene.sources.some((s) => s.kind === "user-input" && s.uri.startsWith("experiment://")));
  assert.ok(scene.sources.every((s) => !["field-measurement", "manufacturer-document", "registered-geometry"].includes(s.kind)));
  assert.ok(scene.artifacts.every((a) => a.role === "visual-only"));
  assert.equal(exportBenson(scene).installationApproved, false);
  assert.throws(() => exportBenson(scene, { purpose: "technical" }));
  const invalid = structuredClone(scene);
  invalid.cameras[0].pose[0].state = "measured";
  assert.throws(() => assertScene(invalid));
});
