import { spawn } from "node:child_process";
import { writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { canonicalJSON, fileDigest, requestKey, verifiedInside } from "./cache.mjs";
import { assertScene, emptyScene } from "./scene.mjs";

const worker = new URL("../workers/moge2/worker.py", import.meta.url);
export const moge2License = { id: "MIT", commercialUse: "allowed", attribution: "Microsoft MoGe / Ruicheng Wang; DINOv2 Apache-2.0 notices required", termsUrl: "https://huggingface.co/Ruicheng/moge-2-vits-normal" };

export function normalizeFovConditioning(value, count) {
  if (value === undefined) return undefined;
  if (!value || Array.isArray(value) || typeof value !== "object") throw new Error("Conditioning must be an explicit object.");
  if (value.mode === "none" && Object.keys(value).length === 1) return undefined;
  if (value.mode !== "known-fov" || value.provenance !== "experiment-oracle" || Object.keys(value).sort().join() !== "horizontalFovDegrees,mode,provenance") throw new Error("Only explicit experiment-oracle known-fov conditioning is supported.");
  if (!Array.isArray(value.horizontalFovDegrees) || value.horizontalFovDegrees.length !== count || value.horizontalFovDegrees.some((n) => typeof n !== "number" || !Number.isFinite(n) || n <= 1 || n >= 179)) throw new Error("FOV must contain one finite horizontal angle in (1,179) degrees per image.");
  return structuredClone(value);
}

export function invokeMoGe2(args) {
  const python = process.env.IMAGE_BLASTER_MOGE2_PYTHON || fileURLToPath(new URL(process.platform === "win32" ? "../workers/moge2/.venv/Scripts/python.exe" : "../workers/moge2/.venv/bin/python", import.meta.url));
  return new Promise((resolve, reject) => {
    const child = spawn(python, [fileURLToPath(worker), ...args], { windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "", stderr = "";
    child.stdout.on("data", (data) => { stdout += data; if (stdout.length > 1024 * 1024) { child.kill(); reject(new Error("MoGe worker response exceeds protocol limit.")); } });
    child.stderr.on("data", (data) => { stderr = (stderr + data).slice(-8000); });
    child.on("error", (error) => reject(new Error(`MoGe local worker unavailable: ${error.message}. See docs/foundation/MOGE2-WORKER.md.`)));
    child.on("close", (code) => {
      if (code !== 0) { reject(new Error(`MoGe worker failed (${code}): ${stderr}`)); return; }
      try { resolve(JSON.parse(stdout)); } catch { reject(new Error("MoGe worker returned invalid JSON.")); }
    });
  });
}

export function createMoGe2Provider({ invoke = invokeMoGe2, implementationFiles = [new URL(import.meta.url), worker, new URL("../workers/moge2/requirements.txt", import.meta.url)], implementationConfig = {} } = {}) {
  return {
    id: "moge2-vits-normal", capability: "scene-geometry", mode: "local", billing: "free", model: "Ruicheng/moge-2-vits-normal", version: "1",
    implementationFiles, implementationConfig: { ...implementationConfig, ...(invoke !== invokeMoGe2 ? { injectedWorker: invoke.toString() } : {}) }, license: moge2License,
    defaults: { sceneId: "geometry-scene", resolutionLevel: 9, threads: 6 }, extra: [], normalizeConditioning: normalizeFovConditioning,
    identity: () => invoke(["--identity"]),
    async generate(request) {
      if (!request.inputs.length || request.inputs.length > 8 || request.prompt) throw new Error("MoGe requires 1..8 independent RGB images and no semantic prompt.");
      if (!Number.isInteger(request.parameters.resolutionLevel) || request.parameters.resolutionLevel < 0 || request.parameters.resolutionLevel > 9 || !Number.isInteger(request.parameters.threads) || request.parameters.threads < 1 || request.parameters.threads > 64) throw new Error("MoGe requires integer resolutionLevel 0..9 and threads 1..64.");
      const requestPath = path.join(request.outputDir, "worker-request.json");
      await writeFile(requestPath, `${JSON.stringify({ inputs: request.inputs.map(({ path }) => ({ path })), outputDir: path.join(request.outputDir, "inference"), parameters: { resolutionLevel: request.parameters.resolutionLevel, threads: request.parameters.threads }, ...(request.conditioning ? { conditioning: request.conditioning } : {}), expectedIdentity: request.identity.checkpoint }, null, 2)}\n`);
      const response = await invoke(["--request", requestPath]);
      if (canonicalJSON(response.identity) !== canonicalJSON(request.identity.checkpoint)) throw new Error("MoGe checkpoint/implementation changed between identity check and inference.");
      if (response.cameras?.length !== request.inputs.length || !response.files?.length) throw new Error("MoGe returned incomplete geometric evidence.");
      const scene = request.scene ? structuredClone(request.scene) : emptyScene(request.parameters.sceneId);
      const prefix = `moge2-${request.key.slice(0, 12)}`;
      const sourceIds = request.inputs.map((input, index) => {
        const existing = scene.sources.find((s) => s.kind === "image" && s.sha256 === input.sha256);
        if (existing) return existing.id;
        const id = `${prefix}-input-${index}`;
        scene.sources.push({ id, kind: "image", uri: input.path, sha256: input.sha256, license: input.license });
        return id;
      });
      const modelId = `${prefix}-model`;
      scene.sources.push({ id: modelId, kind: "model", uri: `https://huggingface.co/Ruicheng/moge-2-vits-normal/tree/${response.identity.modelRevision}`, sha256: response.identity.checkpointSha256, license: moge2License });
      const oracleId = `${prefix}-experiment-oracle-fov`;
      if (request.conditioning) {
        if (canonicalJSON(response.conditioning) !== canonicalJSON(request.conditioning)) throw new Error("Worker omitted explicit oracle FOV provenance.");
        scene.sources.push({ id: oracleId, kind: "user-input", uri: "experiment://explicit-oracle-horizontal-fov", sha256: requestKey(request.conditioning), license: { id: "MIT", commercialUse: "allowed", attribution: "Synthetic experiment input, not field measurement" } });
      }
      const evidenceIds = [...sourceIds, modelId, ...(request.conditioning ? [oracleId] : [])];
      const fact = (id, value, supplied = false) => ({ id: `${prefix}-${id}`, value, state: supplied ? "user-specified" : "inferred", sourceIds: supplied ? [oracleId] : evidenceIds, confidence: 0, note: supplied ? "Experiment oracle FOV; not independent camera reconstruction or measured evidence." : "Independent monocular model inference in metric camera coordinates. No extrinsic reconstruction or installation authority." });
      for (const [index, camera] of response.cameras.entries()) {
        if (camera.sourceIndex !== index || camera.extrinsics !== undefined) throw new Error("MoGe must preserve image order and must not claim camera poses.");
        scene.cameras.push({ id: `${prefix}-camera-${index}`, label: fact(`camera-${index}-label`, `Monocular camera ${index}`), dimensions: {},
          projection: [fact(`camera-${index}-projection`, { intrinsics: camera.intrinsics, imageWidth: camera.imageWidth, imageHeight: camera.imageHeight, convention: "opencv" }, Boolean(request.conditioning))],
          properties: { sourceImage: [fact(`camera-${index}-source`, sourceIds[index])], ...(request.conditioning ? { fovConditioning: [fact(`camera-${index}-fov`, request.conditioning, true)] } : {}) }
        });
      }
      const files = [];
      for (const [index, file] of response.files.entries()) {
        await verifiedInside(request.outputDir, file);
        files.push(file);
        scene.artifacts.push({ id: `${prefix}-geometry-${index}`, uri: file, format: path.extname(file).slice(1), sha256: await fileDigest(file), role: "visual-only", sourceIds: evidenceIds, provider: request.identity,
          parameters: { ...request.parameters, units: "meters", coordinateFrame: "opencv-camera-per-image", scaleSource: "learned-metric-scale-head", evidence: "inferred", independentCameraReconstruction: false, sourceImageHashes: request.inputs.map((i) => i.sha256), forceProjection: true, applyMask: true, useFp16: false, ...(request.conditioning ? { conditioning: request.conditioning } : {}) }, license: moge2License });
      }
      scene.validations.push({ target: modelId, check: "geometry-scale-provenance", status: "needs-review", details: "Learned metric point maps remain inferred and visual-only. Every image is monocular; no world poses. Oracle extrinsics may be used in separate evaluation only." });
      assertScene(scene);
      const scenePath = path.join(request.outputDir, "scene-spec.json");
      await writeFile(scenePath, `${JSON.stringify(scene, null, 2)}\n`);
      return { result: scene, files: [...files, requestPath, scenePath] };
    }
  };
}
