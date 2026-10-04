import { writeFile } from "node:fs/promises";
import path from "node:path";
import { assertScene, emptyScene } from "./scene.mjs";
import { fileDigest } from "./cache.mjs";
import { createDA3Provider } from "./da3.mjs";
import { createMoGe2Provider } from "./moge2.mjs";

export const capabilities = ["scene-analysis", "scene-geometry", "image-edit", "object-3d", "world-reconstruction", "material", "audio", "decision"];
export const unknownLicense = { id: "UNKNOWN", commercialUse: "unknown", attribution: "" };

export class ProviderRegistry {
  #providers = new Map();

  register(provider) {
    if (!provider.id || !capabilities.includes(provider.capability) || !["local", "self-hosted", "paid-api"].includes(provider.mode) ||
        !["free", "metered"].includes(provider.billing) || !provider.model || !provider.version || !provider.implementationFiles?.length || typeof provider.generate !== "function" || this.#providers.has(provider.id)) {
      throw new Error("Invalid or duplicate provider registration.");
    }
    this.#providers.set(provider.id, provider);
    return this;
  }

  list() {
    return [...this.#providers.values()].map(({ generate, runLegacy, identity, implementationFiles, ...description }) => description);
  }

  resolve({ capability, providerId, mode = "local" }) {
    if (!capabilities.includes(capability) || !["local", "self-hosted", "paid-api", "auto"].includes(mode)) throw new Error("Invalid capability or routing mode.");
    const candidates = [...this.#providers.values()].filter((provider) => provider.capability === capability && (!providerId || provider.id === providerId) && (mode === "auto" || provider.mode === mode));
    const selected = mode === "auto" && !providerId ? candidates.find((provider) => provider.mode === "local") || candidates.find((provider) => provider.mode === "self-hosted") : candidates[0];
    if (!selected) throw new Error(`No provider for ${capability} in ${mode} mode. Configure one explicitly; auto never falls back to a paid API.`);
    return selected;
  }
}

const definitions = [
  { id: "fal-hunyuan", capability: "object-3d", model: "fal-ai/hunyuan3d-v3/image-to-3d", module: "../.claude/scripts/asset-pipeline/hunyuan-3d.mjs", exportName: "runHunyuan3D", defaults: { faceCount: 50000, enablePbr: true, generateType: "Normal", polygonType: "triangle" }, input: "image", extra: ["assetName"] },
  { id: "fal-meshy", capability: "object-3d", model: "fal-ai/meshy/v6/image-to-3d", module: "../.claude/scripts/asset-pipeline/meshy-3d.mjs", exportName: "runMeshy3D", defaults: { topology: "triangle", targetPolycount: 30000, symmetryMode: "auto", shouldRemesh: true, shouldTexture: true, riggingHeightMeters: 1.7, animationActionId: 12, enableSafetyChecker: true, enableAnimation: false, enableRigging: false, enablePbr: true }, input: "image", extra: ["assetName"] },
  { id: "fal-nano-banana", capability: "image-edit", model: "fal-ai/nano-banana-2/edit", module: "../.claude/scripts/asset-pipeline/nano-banana-edit.mjs", exportName: "runNanoBananaEdit", defaults: { numImages: 1, resolution: "1K", aspectRatio: "auto", outputFormat: "png", safetyTolerance: "4", limitGenerations: true }, input: "images", extra: ["seed"] },
  { id: "fal-gpt-image", capability: "image-edit", model: "openai/gpt-image-2/edit", module: "../.claude/scripts/asset-pipeline/gpt-image-2-edit.mjs", exportName: "runGptImage2Edit", defaults: { numImages: 1, quality: "medium", imageSize: "auto", outputFormat: "png" }, input: "images", extra: [] },
  { id: "world-labs", capability: "world-reconstruction", model: "marble-1.1", module: "../.claude/scripts/world/generate-world.mjs", exportName: "generateWorld", defaults: {}, input: "image", extra: [] },
  { id: "fal-elevenlabs", capability: "audio", model: "fal-ai/elevenlabs/sound-effects/v2", module: "../.claude/scripts/sfx/fal-elevenlabs-sfx.mjs", exportName: "generateSfx", defaults: { prefix: "sfx", count: 1, loop: false, promptInfluence: 0.3, outputFormat: "mp3_44100_128", kind: "sfx", postprocess: true }, extra: ["durationSeconds"] }
];

export function normalizeParameters(provider, parameters = {}) {
  if (!parameters || Array.isArray(parameters) || typeof parameters !== "object") throw new Error("Parameters must be an object.");
  const allowed = new Set([...Object.keys(provider.defaults || {}), ...(provider.extra || [])]);
  for (const key of Object.keys(parameters)) if (!allowed.has(key)) throw new Error(`Unsupported parameter ${key} for ${provider.id}.`);
  const normalized = { ...provider.defaults, ...parameters };
  for (const [key, value] of Object.entries(normalized)) {
    const expected = Object.hasOwn(provider.defaults || {}, key) ? typeof provider.defaults[key] : ["seed", "durationSeconds"].includes(key) ? "number" : "string";
    if (typeof value !== expected || (expected === "number" && !Number.isFinite(value))) throw new Error(`${key} must be a finite typed ${expected} value.`);
  }
  const enums = { generateType: ["Normal", "LowPoly", "Geometry"], polygonType: ["triangle", "quadrilateral"], topology: ["triangle", "quad"], symmetryMode: ["auto", "on", "off"], quality: ["low", "medium", "high"], outputFormat: ["png", "jpeg", "webp", "mp3_44100_128", "mp3_44100_192", "pcm_44100", "opus_48000_128"] };
  for (const [key, choices] of Object.entries(enums)) if (key in normalized && !choices.includes(normalized[key])) throw new Error(`Unsupported ${key}.`);
  for (const key of ["faceCount", "targetPolycount", "numImages", "count", "animationActionId", "seed"]) if (key in normalized && !Number.isInteger(normalized[key])) throw new Error(`${key} must be an integer.`);
  if ("faceCount" in normalized && (normalized.faceCount < 40000 || normalized.faceCount > 1500000)) throw new Error("faceCount must be between 40000 and 1500000.");
  for (const key of ["targetPolycount", "riggingHeightMeters", "width", "height", "depth"]) if (key in normalized && normalized[key] <= 0) throw new Error(`${key} must be positive.`);
  for (const key of ["numImages", "count"]) if (key in normalized && (normalized[key] < 1 || normalized[key] > 4)) throw new Error(`${key} must be between 1 and 4.`);
  if ("durationSeconds" in normalized && (normalized.durationSeconds < 0.5 || normalized.durationSeconds > 22)) throw new Error("durationSeconds must be between 0.5 and 22.");
  if ("promptInfluence" in normalized && (normalized.promptInfluence < 0 || normalized.promptInfluence > 1)) throw new Error("promptInfluence must be between 0 and 1.");
  if (provider.id === "fal-hunyuan" && normalized.generateType !== "LowPoly") delete normalized.polygonType;
  return normalized;
}

export function createRegistry(runtimes = {}) {
  const registry = new ProviderRegistry();
  registry.register(createDA3Provider());
  registry.register(createMoGe2Provider());
  registry.register({
    id: "local-evidence", capability: "scene-analysis", mode: "local", billing: "free", model: "evidence-envelope", version: "1", implementationFiles: [new URL(import.meta.url)], defaults: { sceneId: "scene" }, extra: [], license: { id: "MIT", commercialUse: "allowed", attribution: "Image Blaster contributors" },
    async generate(request) {
      const scene = request.scene ? structuredClone(assertScene(request.scene)) : emptyScene(request.parameters.sceneId || "scene");
      for (const [index, input] of request.inputs.entries()) {
        const existing = scene.sources.find((source) => source.kind === "image" && source.sha256 === input.sha256);
        if (!existing) scene.sources.push({ id: `input-${index}-${input.sha256.slice(0, 12)}`, kind: "image", uri: input.path, sha256: input.sha256, license: input.license || unknownLicense });
      }
      assertScene(scene);
      const file = path.join(request.outputDir, "scene.json");
      await writeFile(file, `${JSON.stringify(scene, null, 2)}\n`);
      return { result: scene, files: [file] };
    }
  });
  registry.register({
    id: "procedural-box", capability: "object-3d", mode: "local", billing: "free", model: "deterministic-box", version: "1", implementationFiles: [new URL(import.meta.url)], defaults: { width: 1, height: 1, depth: 1 }, extra: [], license: { id: "MIT", commercialUse: "allowed", attribution: "Image Blaster contributors" },
    async generate(request) {
      if (request.inputs.length || request.prompt || request.scene) throw new Error("Procedural box accepts numeric dimensions only; it does not reconstruct a source image.");
      const { width: w, height: h, depth: d } = request.parameters;
      const vertices = [[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0], [0, 0, d], [w, 0, d], [w, h, d], [0, h, d]];
      const triangles = [[1, 3, 2], [1, 4, 3], [5, 6, 7], [5, 7, 8], [1, 2, 6], [1, 6, 5], [4, 8, 7], [4, 7, 3], [1, 5, 8], [1, 8, 4], [2, 3, 7], [2, 7, 6]];
      const file = path.join(request.outputDir, "box.obj");
      await writeFile(file, `o visual_box\n${vertices.map((vertex) => `v ${vertex.join(" ")}`).join("\n")}\n${triangles.map((triangle) => `f ${triangle.join(" ")}`).join("\n")}\n`);
      return { result: { path: file, sha256: await fileDigest(file), format: "obj", units: "meters", dimensions: request.parameters, authority: "visual-only", triangles: 12, materialSupport: "none" }, files: [file] };
    }
  });
  for (const definition of definitions) {
    const runLegacy = async (options) => {
      const run = runtimes[definition.id] || (await import(definition.module))[definition.exportName];
      return run(options);
    };
    registry.register({
      ...definition, mode: "paid-api", billing: "metered", version: "upstream-4acb43b-adapter-1", checkpointRevision: "vendor-immutable-revision-unknown",
      implementationFiles: [new URL(import.meta.url), new URL(definition.module, import.meta.url)],
      ...(runtimes[definition.id] ? { implementationConfig: { injectedRuntime: runtimes[definition.id].toString() } } : {}), license: unknownLicense, runLegacy,
      async generate(request) {
        const paths = request.inputs.map((input) => input.path);
        if (definition.input === "image" && paths.length !== 1) throw new Error("This provider requires exactly one image.");
        if (definition.input === "images" && !paths.length) throw new Error("This provider requires input images.");
        if (["image-edit", "audio", "world-reconstruction"].includes(definition.capability) && !request.prompt.trim()) throw new Error("This provider requires an explicit prompt.");
        const options = { ...request.parameters, outputDir: request.outputDir, ...(request.prompt ? { prompt: request.prompt } : {}) };
        if (definition.input === "image") options.image = paths[0];
        if (definition.input === "images") options.images = paths;
        if (definition.id === "world-labs") options.world = request.key;
        const result = await runLegacy(options);
        const files = definition.id === "world-labs"
          ? [result.world_json, result.plate, result.glb, result.pano, result.thumbnail, ...Object.values(result.spz || {}), result.request_metadata].filter(Boolean)
          : [...(result.output_files || []), ...(result.files || []).map((file) => file.path)].filter(Boolean);
        return { result: JSON.parse(JSON.stringify(result)), files: [...new Set(files)] };
      }
    });
  }
  return registry;
}

export async function runLegacyProvider(providerId, capability, options) {
  return createRegistry().resolve({ providerId, capability, mode: "paid-api" }).runLegacy(options);
}
