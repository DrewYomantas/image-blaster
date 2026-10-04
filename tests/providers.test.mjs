import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile, mkdir } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { createRegistry, normalizeParameters, ProviderRegistry } from "../engine/providers.mjs";
import { runGeneration } from "../engine/run.mjs";
import { assertPaidSubmission, withSpendPolicy } from "../engine/spend.mjs";
import { submitFalQueue } from "../.claude/scripts/asset-pipeline/fal-queue.mjs";
import { runFalWildcard } from "../.claude/scripts/fal/run-fal.mjs";
import { generateWorld } from "../.claude/scripts/world/generate-world.mjs";
import { sceneFixture } from "./fixtures.mjs";

const approval = (endpoint) => ({ allowPaid: true, maxCostUSD: 1, estimatedCostUSD: 0.5, approvalReference: "TEST ONLY - fake fetch", endpoints: [endpoint] });

async function setup(t) {
  const root = await mkdtemp(path.join(os.tmpdir(), "image-blaster-provider-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const image = path.join(root, "source.png");
  await writeFile(image, Buffer.from("89504e470d0a1a0a00000000", "hex"));
  return { root, image, cacheDir: path.join(root, "cache") };
}

test("registry includes separate semantic and geometric boundaries, retained hosted providers and local procedures", () => {
  const registry = createRegistry();
  assert.equal(registry.list().length, 10);
  assert.equal(registry.resolve({ capability: "scene-geometry" }).id, "da3-small");
  assert.equal(registry.resolve({ capability: "scene-analysis", mode: "auto" }).id, "local-evidence");
  assert.equal(registry.resolve({ capability: "object-3d", mode: "auto" }).id, "procedural-box");
  assert.throws(() => registry.resolve({ capability: "world-reconstruction", mode: "auto" }), /never falls back/);
  assert.throws(() => registry.resolve({ capability: "material" }), /No provider/);
  assert.throws(() => registry.resolve({ capability: "unknown" }), /Invalid capability/);
});

test("typed defaults eliminate duplicate defaults and reject bool/number string ambiguity", () => {
  const registry = createRegistry();
  const hunyuan = registry.resolve({ capability: "object-3d", providerId: "fal-hunyuan", mode: "paid-api" });
  assert.deepEqual(normalizeParameters(hunyuan, {}), normalizeParameters(hunyuan, { faceCount: 50000, enablePbr: true, generateType: "Normal", polygonType: "quadrilateral" }));
  assert.throws(() => normalizeParameters(hunyuan, { enablePbr: "false" }), /typed boolean/);
  assert.throws(() => normalizeParameters(hunyuan, { faceCount: "50000" }), /typed number/);
  assert.throws(() => normalizeParameters(hunyuan, { faceCount: 4 }), /between/);
  assert.throws(() => normalizeParameters(hunyuan, { model: "other" }), /Unsupported parameter/);
  const nano = registry.resolve({ capability: "image-edit", providerId: "fal-nano-banana", mode: "paid-api" });
  assert.throws(() => normalizeParameters(nano, { limitGenerations: "false" }), /typed boolean/);
  assert.throws(() => normalizeParameters(nano, { numImages: 0 }), /between/);
});

test("world engine adapter rejects empty prompt and implicit workspace caption fallback", async (t) => {
  const f = await setup(t);
  let called = false;
  const registry = createRegistry({ "world-labs": async () => { called = true; } });
  const provider = registry.resolve({ capability: "world-reconstruction", providerId: "world-labs", mode: "paid-api" });
  await assert.rejects(provider.generate({ key: "fixture", inputs: [{ path: f.image }], parameters: {}, prompt: "", outputDir: f.root }), /explicit prompt/);
  assert.equal(called, false);
});

test("paid engine execution rejects no approval, wrong endpoint and inadequate budget before run", async (t) => {
  const f = await setup(t);
  let calls = 0;
  const registry = createRegistry({ "fal-hunyuan": async () => { calls += 1; throw new Error("should never run"); } });
  const request = { capability: "object-3d", providerId: "fal-hunyuan", mode: "paid-api", inputs: [{ path: f.image }] };
  await assert.rejects(runGeneration(request, { ...f, registry }), /Paid generation blocked/);
  await assert.rejects(runGeneration(request, { ...f, registry, spend: approval("other-endpoint") }), /outside the approved/);
  await assert.rejects(runGeneration(request, { ...f, registry, spend: { ...approval("fal-ai/hunyuan3d-v3/image-to-3d"), maxCostUSD: 0.1 } }), /Paid generation blocked/);
  assert.equal(calls, 0);
});

test("low-level FAL guard precedes credentials and any network request", async (t) => {
  const f = await setup(t);
  const oldFetch = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async () => { calls += 1; throw new Error("network forbidden"); };
  t.after(() => { globalThis.fetch = oldFetch; });
  await assert.rejects(submitFalQueue("fixture-endpoint", {}), /Paid generation blocked/);
  await assert.rejects(runFalWildcard({ endpoint: "fixture-endpoint", input: {}, outputDir: path.join(f.root, "direct"), mode: "run" }), /Paid generation blocked/);
  await assert.rejects(generateWorld({ world: "fixture", image: f.image, prompt: "fixture", outputDir: path.join(f.root, "world") }), /Paid generation blocked/);
  assert.equal(calls, 0);
});

test("per-submission budget is depleted and endpoint mismatch fails", async () => {
  await withSpendPolicy(approval("fake"), async () => {
    assert.throws(() => assertPaidSubmission("other"), /endpoint/);
    assertPaidSubmission("fake");
    assertPaidSubmission("fake");
    assert.throws(() => assertPaidSubmission("fake"), /remaining budget/);
  });
});

test("all six legacy adapters dispatch correct input/output through injectable runtime", async (t) => {
  const f = await setup(t);
  const ids = ["fal-hunyuan", "fal-meshy", "fal-nano-banana", "fal-gpt-image", "world-labs", "fal-elevenlabs"];
  const runtimes = Object.fromEntries(ids.map((id) => [id, async (options) => {
    if (["fal-hunyuan", "fal-meshy", "world-labs"].includes(id)) assert.equal(options.image.endsWith("source.png"), true);
    if (["fal-nano-banana", "fal-gpt-image"].includes(id)) assert.equal(options.images.length, 1);
    if (id === "world-labs") assert.equal(options.world, "fixture");
    const file = path.join(options.outputDir, `${id}.artifact`);
    await writeFile(file, id);
    return id === "world-labs" ? { world_json: file } : id === "fal-elevenlabs" ? { files: [{ path: file }] } : { output_files: [file] };
  }]));
  const registry = createRegistry(runtimes);
  const outputDir = path.join(f.root, "out");
  await mkdir(outputDir);
  for (const id of ids) {
    const descriptor = registry.list().find((provider) => provider.id === id);
    const provider = registry.resolve({ capability: descriptor.capability, providerId: id, mode: "paid-api" });
    const result = await provider.generate({ key: "fixture", inputs: [{ path: f.image }], parameters: normalizeParameters(provider), prompt: "fixture", outputDir });
    assert.equal(result.files.length, 1);
    assert.equal(await readFile(result.files[0], "utf8"), id);
  }
});

test("real retained Hunyuan, Meshy, image edit and World Labs modules run through adapters with entirely mocked fetch", async (t) => {
  const f = await setup(t);
  const oldFetch = globalThis.fetch;
  const oldFalKey = process.env.FAL_KEY;
  const oldWorldKey = process.env.WORLD_LABS_API_KEY;
  process.env.FAL_KEY = "test-fixture-never-sent";
  process.env.WORLD_LABS_API_KEY = "test-fixture-never-sent";
  const submits = [];
  globalThis.fetch = async (url, options = {}) => {
    const uri = String(url);
    assert.ok(uri.startsWith("https://queue.fal.run/") || uri.startsWith("https://api.worldlabs.ai/"));
    if (options.method === "POST") {
      submits.push(JSON.parse(options.body));
      if (uri.includes("worldlabs")) return new Response(JSON.stringify({ operation_id: "fake-operation", done: true, response: { assets: { mesh: { collider_mesh_url: "data:model/gltf-binary;base64,Zml4dHVyZQ==" }, splats: { spz_urls: { full_res: "data:application/octet-stream;base64,Zml4dHVyZQ==" } } } } }));
      return new Response(JSON.stringify({ request_id: "fake-request" }));
    }
    if (uri.includes("/status")) return new Response(JSON.stringify({ status: "COMPLETED" }));
    return new Response(JSON.stringify({ model_mesh: { url: "data:model/gltf-binary;base64,Zml4dHVyZQ==", file_name: "fixture.glb", content_type: "model/gltf-binary" } }));
  };
  t.after(() => {
    globalThis.fetch = oldFetch;
    if (oldFalKey === undefined) delete process.env.FAL_KEY; else process.env.FAL_KEY = oldFalKey;
    if (oldWorldKey === undefined) delete process.env.WORLD_LABS_API_KEY; else process.env.WORLD_LABS_API_KEY = oldWorldKey;
  });
  const registry = createRegistry();
  for (const id of ["fal-hunyuan", "fal-meshy", "fal-nano-banana", "fal-gpt-image", "world-labs"]) {
    const descriptor = registry.list().find((provider) => provider.id === id);
    const result = await runGeneration({ capability: descriptor.capability, providerId: id, mode: "paid-api", inputs: [{ path: f.image }], prompt: "fixture prompt" }, { ...f, registry, spend: approval(descriptor.model) });
    assert.equal(result.cached, false);
    const manifest = JSON.parse(await readFile(result.manifestPath));
    assert.equal(manifest.status, "complete");
    assert.ok(manifest.artifacts.length);
    assert.ok(manifest.artifacts.every((file) => !file.path.startsWith("..")));
    assert.equal((await runGeneration({ capability: descriptor.capability, providerId: id, mode: "paid-api", inputs: [{ path: f.image }], prompt: "fixture prompt" }, f)).cached, true);
  }
  assert.equal(submits.length, 5);
  assert.equal(submits[0].face_count, 50000);
  assert.equal(submits[0].enable_pbr, true);
  assert.equal(submits[4].model, "marble-1.1");
});

test("free self-hosted registry provider can execute without cloud spend approval", async (t) => {
  const f = await setup(t);
  const registry = new ProviderRegistry();
  registry.register({ id: "owned-worker", implementationFiles: [new URL(import.meta.url)], model: "fixture", version: "1", capability: "object-3d", mode: "self-hosted", billing: "free", defaults: {}, extra: [], license: {}, async generate(request) { const file = path.join(request.outputDir, "fixture.obj"); await writeFile(file, "fixture"); return { result: { fixture: true }, files: [file] }; } });
  assert.equal((await runGeneration({ capability: "object-3d", mode: "self-hosted" }, { ...f, registry })).cached, false);
});

test("generated analysis cannot invent authoritative sources or discard inferred evidence", async (t) => {
  const f = await setup(t);
  const registry = createRegistry();
  registry.resolve({ capability: "scene-analysis" }).generate = async (request) => {
    const file = path.join(request.outputDir, "scene.json");
    const scene = sceneFixture();
    await writeFile(file, JSON.stringify(scene));
    return { result: scene, files: [file] };
  };
  await assert.rejects(runGeneration({ capability: "scene-analysis" }, { ...f, registry }), /introduce authoritative/);
});

test("provider cannot rewrite caller evidence or canonical request snapshot in place", async (t) => {
  const f = await setup(t);
  const registry = createRegistry();
  const original = sceneFixture();
  const before = structuredClone(original);
  registry.resolve({ capability: "scene-analysis" }).generate = async (request) => {
    request.scene.objects[0].dimensions.width[1].value = 4;
    request.parameters.sceneId = "rewritten";
    const file = path.join(request.outputDir, "scene.json");
    await writeFile(file, JSON.stringify(request.scene));
    return { result: request.scene, files: [file] };
  };
  await assert.rejects(runGeneration({ capability: "scene-analysis", scene: original }, { ...f, registry }), /preserve fact ownership/);
  assert.deepEqual(original, before);
  const keys = await (await import("node:fs/promises")).readdir(f.cacheDir);
  const manifest = JSON.parse(await readFile(path.join(f.cacheDir, keys[0], "manifest.json")));
  assert.equal(manifest.request.parameters.sceneId, "scene");
  assert.equal(manifest.request.scene.objects[0].dimensions.width[1].value, 0.9);
});

test("deterministic local box produces real OBJ vertices/triangles and reuses cache", async (t) => {
  const f = await setup(t);
  const request = { capability: "object-3d", parameters: { width: 2, height: 3, depth: 4 } };
  const result = await runGeneration(request, f);
  const obj = await readFile(result.result.path, "utf8");
  assert.equal(obj.split("\n").filter((line) => line.startsWith("v ")).length, 8);
  assert.equal(obj.split("\n").filter((line) => line.startsWith("f ")).length, 12);
  assert.ok(obj.includes("v 2 3 4"));
  assert.equal(result.result.authority, "visual-only");
  assert.equal((await runGeneration(request, f)).cached, true);
});
