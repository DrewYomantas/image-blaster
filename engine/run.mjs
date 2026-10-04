import { copyFile, mkdir, readdir, rename, rmdir, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { canonicalJSON, fileDigest, inside, readCached, requestKey, verifiedInside } from "./cache.mjs";
import { createRegistry, normalizeParameters } from "./providers.mjs";
import { assertEvidencePreserved, assertScene } from "./scene.mjs";
import { assertSpendPolicy, withSpendPolicy } from "./spend.mjs";
import { providerIdentity } from "./identity.mjs";

function rejectSecrets(value) {
  if (!value || typeof value !== "object") return;
  for (const [key, child] of Object.entries(value)) {
    if (/api.?key|authorization|password|secret|token/i.test(key)) throw new Error("Credentials must never appear in generation requests or manifests.");
    rejectSecrets(child);
  }
}

async function persist(file, data) {
  const temporary = `${file}.tmp`;
  await writeFile(temporary, `${JSON.stringify(data, null, 2)}\n`);
  await rename(temporary, file);
}

export async function runGeneration(request, { registry = createRegistry(), cacheDir = ".image-blaster/cache", spend = {} } = {}) {
  rejectSecrets(request);
  const { capability, providerId, mode = "local", inputs = [], parameters = {}, prompt = "", scene } = structuredClone(request);
  if (typeof prompt !== "string" || !Array.isArray(inputs)) throw new Error("Prompt must be a string and inputs must be an array.");
  if (scene) assertScene(scene);
  const provider = registry.resolve({ capability, providerId, mode });
  const normalized = normalizeParameters(provider, parameters);
  const sources = await Promise.all(inputs.map(async (input) => {
    if (!input.path || /^https?:|^data:/i.test(input.path)) throw new Error("Stage remote inputs locally before generation so cache keys hash actual content.");
    return { sha256: await fileDigest(input.path), extension: path.extname(input.path).toLowerCase(), mediaType: input.mediaType || "application/octet-stream", role: input.role || "source", license: input.license || { id: "UNKNOWN", commercialUse: "unknown", attribution: "" } };
  }));
  const effective = { schemaVersion: 1, capability, provider: await providerIdentity(provider), inputs: sources, parameters: normalized, prompt, ...(scene ? { scene } : {}) };
  const key = requestKey(effective);
  const entry = path.resolve(cacheDir, key);
  const manifestPath = path.join(entry, "manifest.json");
  await mkdir(path.resolve(cacheDir), { recursive: true });
  try { await verifiedInside(cacheDir, entry); }
  catch (error) { if (error.code !== "ENOENT") throw error; }
  const cached = await readCached(entry, key);
  if (cached) return { key, cached: true, manifestPath, result: cached.result };
  const policy = { ...spend };
  if (provider.billing === "metered") {
    assertSpendPolicy(policy);
    if (!policy.endpoints.includes(provider.model)) throw new Error("Paid generation blocked: selected provider/model is outside the approved endpoints.");
  }
  const lock = `${entry}.lock`;
  try { await mkdir(lock); }
  catch (error) { if (error.code === "EEXIST") throw new Error("Generation is already locked. Inspect its state before removing a stale lock."); throw error; }
  try {
    const existing = await readCached(entry, key);
    if (existing) return { key, cached: true, manifestPath, result: existing.result };
    await mkdir(entry, { recursive: true });
    await verifiedInside(cacheDir, entry);
    if ((await readdir(entry)).length) throw new Error("Nonempty cache entry without a manifest requires inspection; refusing to overwrite existing files.");
    await mkdir(path.join(entry, "inputs"), { recursive: true });
    const outputDir = path.join(entry, "artifacts");
    await mkdir(outputDir, { recursive: true });
    await verifiedInside(entry, path.join(entry, "inputs"));
    await verifiedInside(entry, outputDir);
    const staged = [];
    for (const [index, input] of inputs.entries()) {
      const extension = path.extname(input.path).toLowerCase();
      const target = path.join(entry, "inputs", `${index}${extension}`);
      await copyFile(input.path, target);
      if (await fileDigest(target) !== sources[index].sha256) throw new Error("Source changed while staging; retry from a stable input.");
      staged.push({ ...sources[index], path: target, bytes: (await stat(target)).size });
    }
    const manifest = {
      schemaVersion: 1, key, request: effective, provider: effective.provider, status: "running", startedAt: new Date().toISOString(),
      license: provider.license, inputs: staged.map((input) => ({ ...input, path: inside(entry, input.path) })), artifacts: [],
      spend: provider.billing === "free" ? { estimatedCostUSD: 0 } : { estimatedCostUSD: policy.estimatedCostUSD, maxCostUSD: policy.maxCostUSD, approvalReference: policy.approvalReference, endpoints: policy.endpoints }
    };
    await persist(manifestPath, manifest);
    try {
      const execute = () => provider.generate({ key, identity: structuredClone(effective.provider), inputs: structuredClone(staged), parameters: structuredClone(normalized), prompt, scene: scene ? structuredClone(scene) : undefined, outputDir });
      const generated = provider.billing === "free" ? await execute() : await withSpendPolicy(policy, execute);
      if (canonicalJSON(await providerIdentity(provider)) !== canonicalJSON(effective.provider)) throw new Error("Provider implementation/checkpoint changed during generation; result cannot be cached.");
      if (!generated || !Array.isArray(generated.files) || !generated.files.length) throw new Error("Provider returned no local artifacts.");
      rejectSecrets(generated.result);
      for (const input of staged) if (await fileDigest(input.path) !== input.sha256) throw new Error("Provider changed staged source evidence.");
      for (const file of generated.files) {
        manifest.artifacts.push({ path: await verifiedInside(entry, file), sha256: await fileDigest(file), bytes: (await stat(file)).size });
      }
      if (["scene-analysis", "scene-geometry"].includes(capability)) { assertScene(generated.result); assertEvidencePreserved(scene, generated.result); }
      manifest.result = generated.result;
      manifest.resultHash = requestKey(generated.result);
      manifest.status = "complete";
      manifest.completedAt = new Date().toISOString();
      await persist(manifestPath, manifest);
      return { key, cached: false, manifestPath, result: generated.result };
    } catch (error) {
      manifest.status = "failed";
      manifest.failedAt = new Date().toISOString();
      await persist(manifestPath, manifest);
      throw error;
    }
  } finally {
    await rmdir(lock);
  }
}
