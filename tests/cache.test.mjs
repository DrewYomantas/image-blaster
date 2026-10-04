import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile, mkdir, symlink } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { requestKey } from "../engine/cache.mjs";
import { runGeneration } from "../engine/run.mjs";
import { createRegistry } from "../engine/providers.mjs";
import { sceneFixture } from "./fixtures.mjs";
import { normalizeParameters } from "../engine/providers.mjs";

async function fixture(t) {
  const root = await mkdtemp(path.join(os.tmpdir(), "image-blaster-cache-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const image = path.join(root, "source.png");
  await writeFile(image, Buffer.from("89504e470d0a1a0a00000000", "hex"));
  return { root, image, cacheDir: path.join(root, "cache") };
}

test("request hash ignores JSON key order and rejects non-JSON values", () => {
  assert.equal(requestKey({ b: 2, a: 1 }), requestKey({ a: 1, b: 2 }));
  for (const value of [NaN, undefined, { x: Infinity }]) assert.throws(() => requestKey(value));
});

test("local source evidence is credential-free, content cached, and manifests are complete", async (t) => {
  const f = await fixture(t);
  const request = { capability: "scene-analysis", inputs: [{ path: f.image }], scene: sceneFixture() };
  const first = await runGeneration(request, f);
  const second = await runGeneration(request, f);
  assert.equal(first.cached, false);
  assert.equal(second.cached, true);
  assert.equal(first.key, second.key);
  assert.equal(first.result.objects.length, 1);
  const manifest = JSON.parse(await readFile(first.manifestPath));
  assert.equal(manifest.status, "complete");
  assert.equal(manifest.inputs[0].sha256.length, 64);
  assert.equal(manifest.artifacts[0].sha256.length, 64);
  assert.equal(manifest.spend.estimatedCostUSD, 0);
  const copied = path.join(f.root, "renamed.png");
  await writeFile(copied, await readFile(f.image));
  assert.equal((await runGeneration({ ...request, inputs: [{ path: copied }] }, f)).key, first.key);
});

test("source bytes, prompt, scene fields, defaults and model revision invalidate cache", async (t) => {
  const f = await fixture(t);
  const request = { capability: "scene-analysis", inputs: [{ path: f.image }] };
  const first = await runGeneration(request, f);
  assert.equal((await runGeneration({ ...request, parameters: { sceneId: "scene" } }, f)).key, first.key);
  const prompt = await runGeneration({ ...request, prompt: "review" }, f);
  assert.notEqual(prompt.key, first.key);
  const withScene = await runGeneration({ ...request, scene: sceneFixture() }, f);
  assert.notEqual(withScene.key, first.key);
  const registry = createRegistry();
  registry.resolve({ capability: "scene-analysis" }).version = "2";
  assert.notEqual((await runGeneration(request, { ...f, registry })).key, first.key);
  await writeFile(f.image, "changed input bytes");
  assert.notEqual((await runGeneration(request, f)).key, first.key);
});

test("corrupted cached artifacts fail without calling generation again", async (t) => {
  const f = await fixture(t);
  const request = { capability: "scene-analysis", inputs: [{ path: f.image }] };
  const first = await runGeneration(request, f);
  await writeFile(path.join(path.dirname(first.manifestPath), "artifacts", "scene.json"), "bad");
  await assert.rejects(runGeneration(request, f), /integrity failure/);
});

test("failure and busy locks prevent duplicate submission", async (t) => {
  const f = await fixture(t);
  const registry = createRegistry();
  const provider = registry.resolve({ capability: "scene-analysis" });
  let calls = 0;
  provider.generate = async () => { calls += 1; throw new Error("fixture failure"); };
  const request = { capability: "scene-analysis", inputs: [{ path: f.image }] };
  await assert.rejects(runGeneration(request, { ...f, registry }), /fixture failure/);
  await assert.rejects(runGeneration(request, { ...f, registry }), /inspection\/resume/);
  assert.equal(calls, 1);
  const manifestName = (await import("node:fs/promises")).readdir;
  const [key] = await manifestName(f.cacheDir);
  const manifest = JSON.parse(await readFile(path.join(f.cacheDir, key, "manifest.json")));
  assert.equal(manifest.status, "failed");
});

test("concurrent identical requests submit once", async (t) => {
  const f = await fixture(t);
  const registry = createRegistry();
  const provider = registry.resolve({ capability: "scene-analysis" });
  const generate = provider.generate;
  let finish;
  let entered;
  const started = new Promise((resolve) => { entered = resolve; });
  const hold = new Promise((resolve) => { finish = resolve; });
  provider.generate = async (request) => { entered(); await hold; return generate(request); };
  const request = { capability: "scene-analysis", inputs: [{ path: f.image }] };
  const first = runGeneration(request, { ...f, registry });
  await started;
  await assert.rejects(runGeneration(request, { ...f, registry }), /inspection\/resume|locked/);
  finish();
  await first;
  assert.equal((await runGeneration(request, f)).cached, true);
});

test("credentials and mutable remote URLs are rejected before provider execution", async (t) => {
  const f = await fixture(t);
  await assert.rejects(runGeneration({ capability: "scene-analysis", parameters: { apiKey: "not-a-real-key" } }, f), /Credentials/);
  await assert.rejects(runGeneration({ capability: "scene-analysis", inputs: [{ path: "https://example.invalid/source.png" }] }, f), /Stage remote/);
});

test("symlink outputs escaping cache are rejected", async (t) => {
  const f = await fixture(t);
  const outside = path.join(f.root, "outside");
  await mkdir(outside);
  await writeFile(path.join(outside, "scene.json"), "{}");
  const registry = createRegistry();
  registry.resolve({ capability: "scene-analysis" }).generate = async (request) => {
    const link = path.join(request.outputDir, "linked");
    await symlink(outside, link, "junction");
    return { result: {}, files: [path.join(link, "scene.json")] };
  };
  await assert.rejects(runGeneration({ capability: "scene-analysis" }, { ...f, registry }), /inside its cache/);
});

test("preplanted junction cannot redirect provider writes outside cache", async (t) => {
  const f = await fixture(t);
  const registry = createRegistry();
  const provider = registry.resolve({ capability: "scene-analysis" });
  const effective = { schemaVersion: 1, capability: "scene-analysis", provider: { id: provider.id, model: provider.model, version: provider.version }, inputs: [], parameters: normalizeParameters(provider), prompt: "" };
  const entry = path.join(f.cacheDir, requestKey(effective));
  const outside = path.join(f.root, "outside");
  await mkdir(entry, { recursive: true });
  await mkdir(outside);
  await symlink(outside, path.join(entry, "artifacts"), "junction");
  await assert.rejects(runGeneration({ capability: "scene-analysis" }, { ...f, registry }), /Nonempty cache entry/);
  assert.deepEqual(await (await import("node:fs/promises")).readdir(outside), []);
});
