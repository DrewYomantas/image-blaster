import { spawn } from "node:child_process";
import { writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { canonicalJSON, fileDigest, verifiedInside } from "./cache.mjs";
import { assertScene, emptyScene } from "./scene.mjs";

const worker = new URL("../workers/da3/worker.py", import.meta.url);
export const da3License = { id: "Apache-2.0", commercialUse: "allowed", attribution: "Depth Anything 3 contributors / ByteDance", termsUrl: "https://huggingface.co/depth-anything/DA3-SMALL" };

export function invokeDA3(args) {
  const python = process.env.IMAGE_BLASTER_DA3_PYTHON || fileURLToPath(new URL(process.platform === "win32" ? "../workers/da3/.venv/Scripts/python.exe" : "../workers/da3/.venv/bin/python", import.meta.url));
  return new Promise((resolve, reject) => {
    const child = spawn(python, [fileURLToPath(worker), ...args], { windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "", stderr = "";
    child.stdout.on("data", (data) => { stdout += data; if (stdout.length > 1024 * 1024) { child.kill(); reject(new Error("DA3 worker response exceeds protocol limit.")); } });
    child.stderr.on("data", (data) => { stderr = (stderr + data).slice(-8000); });
    child.on("error", (error) => reject(new Error(`DA3 local worker unavailable: ${error.message}. See docs/foundation/DA3-WORKER.md.`)));
    child.on("close", (code) => {
      if (code !== 0) { reject(new Error(`DA3 worker failed (${code}): ${stderr}`)); return; }
      try { resolve(JSON.parse(stdout)); } catch { reject(new Error("DA3 worker returned invalid JSON.")); }
    });
  });
}

export function createDA3Provider({ invoke = invokeDA3, implementationFiles = [new URL(import.meta.url), worker], implementationConfig = {} } = {}) {
  return {
    id: "da3-small", capability: "scene-geometry", mode: "local", billing: "free", model: "depth-anything/DA3-SMALL", version: "1",
    implementationFiles, implementationConfig: { ...implementationConfig, ...(invoke !== invokeDA3 ? { injectedWorker: invoke.toString() } : {}) }, license: da3License, defaults: { sceneId: "geometry-scene", processResolution: 256, threads: 6, profileWarmInference: false }, extra: [],
    identity: () => invoke(["--identity"]),
    async generate(request) {
      if (request.inputs.length < 2 || request.prompt) throw new Error("DA3 geometry requires overlapping local images and accepts no semantic prompt or ground-truth hints.");
      if (![256, 384].includes(request.parameters.processResolution) || !Number.isInteger(request.parameters.threads) || request.parameters.threads < 1 || request.parameters.threads > 64) throw new Error("DA3 requires resolution 256|384 and integer threads 1..64.");
      const requestPath = path.join(request.outputDir, "worker-request.json");
      await writeFile(requestPath, `${JSON.stringify({ inputs: request.inputs.map(({ path }) => ({ path })), outputDir: path.join(request.outputDir, "inference"), parameters: { processResolution: request.parameters.processResolution, threads: request.parameters.threads, profileWarmInference: request.parameters.profileWarmInference }, expectedIdentity: request.identity.checkpoint }, null, 2)}\n`);
      const response = await invoke(["--request", requestPath]);
      if (canonicalJSON(response.identity) !== canonicalJSON(request.identity.checkpoint)) throw new Error("DA3 checkpoint/implementation changed between identity check and inference.");
      const scene = request.scene ? structuredClone(request.scene) : emptyScene(request.parameters.sceneId);
      const prefix = `da3-${request.key.slice(0, 12)}`;
      const sourceIds = request.inputs.map((input, index) => {
        const source = scene.sources.find((item) => item.kind === "image" && item.sha256 === input.sha256);
        if (source) return source.id;
        const id = `${prefix}-input-${index}`;
        scene.sources.push({ id, kind: "image", uri: input.path, sha256: input.sha256, license: input.license });
        return id;
      });
      const modelId = `${prefix}-model`;
      scene.sources.push({ id: modelId, kind: "model", uri: `https://huggingface.co/depth-anything/DA3-SMALL/tree/${response.identity.modelRevision}`, sha256: response.identity.checkpointSha256, license: da3License });
      const fact = (id, value, sources = [...sourceIds, modelId]) => ({ id: `${prefix}-${id}`, value, state: "inferred", sourceIds: sources, confidence: 0, note: "Uncalibrated model evidence; confidence not calibrated and geometry is not metric or installation truth." });
      for (const [index, camera] of response.cameras.entries()) {
        if (camera.sourceIndex !== index || index >= sourceIds.length) throw new Error("DA3 camera/image correspondence is invalid.");
        scene.cameras.push({ id: `${prefix}-camera-${index}`, label: fact(`camera-${index}-label`, `Inferred camera ${index}`), dimensions: {},
          projection: [fact(`camera-${index}-projection`, { intrinsics: camera.intrinsics, imageWidth: camera.imageWidth, imageHeight: camera.imageHeight, convention: "opencv" })],
          pose: [fact(`camera-${index}-pose`, { worldToCamera: camera.extrinsics, coordinateFrame: "opencv-model-world", units: "relative", scale: "ambiguous" })],
          properties: { sourceImage: [fact(`camera-${index}-source`, sourceIds[index], [sourceIds[index], modelId])],
            ...(camera.pixelTransform ? { inputPixelTransform: [fact(`camera-${index}-pixel-transform`, { ...camera.pixelTransform, inputWidth: camera.inputWidth, inputHeight: camera.inputHeight }, [sourceIds[index], modelId])] } : {}) }
        });
      }
      if (response.cameras.length !== request.inputs.length || !response.files.length) throw new Error("DA3 returned incomplete geometric evidence.");
      const files = [];
      for (const [index, file] of response.files.entries()) {
        await verifiedInside(request.outputDir, file);
        files.push(file);
        scene.artifacts.push({ id: `${prefix}-geometry-${index}`, uri: file, format: path.extname(file).slice(1), sha256: await fileDigest(file), role: "visual-only", sourceIds: [...sourceIds, modelId], provider: request.identity,
          parameters: { ...request.parameters, coordinateFrame: "opencv-model-world", units: "relative", scale: "ambiguous", evidence: "inferred" }, license: da3License });
      }
      scene.validations.push({ target: modelId, check: "relative-geometry-scale", status: "needs-review", details: "DA3 Small geometry has ambiguous scale. Independent anchor calibration is required; model confidence is not installation approval." });
      assertScene(scene);
      const scenePath = path.join(request.outputDir, "scene-spec.json");
      await writeFile(scenePath, `${JSON.stringify(scene, null, 2)}\n`);
      return { result: scene, files: [...files, requestPath, scenePath] };
    }
  };
}
