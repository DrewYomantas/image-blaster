import { spawn } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { runGeneration } from "../../engine/run.mjs";
import { canonicalJSON, fileDigest } from "../../engine/cache.mjs";

const root = fileURLToPath(new URL("../../", import.meta.url));
const fixture = path.join(root, ".image-blaster/benchmark/synthetic-room-v1");
const output = path.join(root, ".image-blaster/benchmark/moge2-vits-normal-v1");
const python = path.join(root, "workers/moge2/.venv/Scripts/python.exe");
const json = async (file) => JSON.parse(await readFile(file, "utf8"));
const save = async (file, value) => writeFile(file, `${JSON.stringify(value, null, 2)}\n`);

async function evaluate(prediction, worker, target) {
  await new Promise((resolve, reject) => {
    const child = spawn(python, [path.join(root, "benchmarks/geometry/evaluate_moge.py"), "--fixture", fixture, "--prediction", prediction, "--worker-receipt", worker, "--out", target], { windowsHide: true, stdio: ["ignore", "ignore", "pipe"] });
    let stderr = "";
    child.stderr.on("data", (chunk) => { stderr = (stderr + chunk).slice(-12000); });
    child.on("error", reject);
    child.on("close", (code) => code === 0 ? resolve() : reject(new Error(stderr)));
  });
}

export async function main() {
  await mkdir(output, { recursive: true });
  const truth = await json(path.join(fixture, "ground-truth/geometry.json"));
  const regenerated = await json(path.join(output, "frozen-fixture/ground-truth/geometry.json"));
  const baseline = await json(path.join(root, "docs/foundation/geometry-results/da3-small-cpu-v1.json"));
  if (canonicalJSON(truth.input_hashes) !== canonicalJSON(regenerated.input_hashes) || canonicalJSON(truth.input_hashes) !== canonicalJSON(baseline.metrics.input_hashes)) throw new Error("Frozen source hash mismatch.");
  const preserved = [];
  for (const receipt of ["da3-small-cpu-v1.json", "camera-conditioning-v1/lane-C-result-v2.json"]) {
    const file = path.join(root, "docs/foundation/geometry-results", receipt);
    const data = await json(file);
    if (await fileDigest(data.metrics.prediction) !== data.metrics.prediction_sha256) throw new Error("Preserved DA3 tensors changed.");
    preserved.push({ file, sha256: await fileDigest(file), predictionSha256: data.metrics.prediction_sha256 });
  }
  if (await fileDigest(path.join(fixture, "ground-truth/geometry.json")) !== baseline.metrics.fixture_geometry_sha256) throw new Error("Canonical truth changed.");
  for (const image of truth.input_hashes) for (const folder of [fixture, path.join(output, "frozen-fixture")]) if (await fileDigest(path.join(folder, "inputs", image.file)) !== image.sha256) throw new Error("Frozen RGB bytes changed.");
  const declarationFile = path.join(root, "docs/foundation/geometry-results/moge2-vits-normal-v1/evaluation-declaration.json");
  const declaration = { file: declarationFile, sha256: await fileDigest(declarationFile), content: await json(declarationFile) };
  const common = { capability: "scene-geometry", providerId: "moge2-vits-normal", mode: "local", parameters: { sceneId: "synthetic-room-moge2", resolutionLevel: 9, threads: 6 }, inputs: truth.input_hashes.map((image) => ({ path: path.join(fixture, "inputs", image.file), mediaType: "image/png", license: { id: "MIT", commercialUse: "allowed", attribution: "Image Blaster synthetic fixture" } })) };
  const lanes = {};
  for (const lane of ["M1", "M2"]) {
    const request = { ...common };
    if (lane === "M2") {
      const fovs = truth.intrinsics.map((k) => 2 * Math.atan(truth.resolution[0] / (2 * k[0][0])) * 180 / Math.PI);
      request.conditioning = { mode: "known-fov", provenance: "experiment-oracle", horizontalFovDegrees: fovs };
      await save(path.join(output, "M2-conditioning.json"), request.conditioning);
    }
    const requestFile = path.join(output, `${lane}-request.json`);
    await save(requestFile, { request, note: "No depth, extrinsics, labels, dimensions or scale anchor enter inference.", evaluationDeclaration: declaration });
    const start = performance.now();
    const result = await runGeneration(request);
    const wall = (performance.now() - start) / 1000;
    const replayStart = performance.now();
    const replay = await runGeneration({ ...request, evaluationMetadata: { note: "changed report metadata has no inference effect" } });
    const replayWall = (performance.now() - replayStart) / 1000;
    if (!replay.cached || result.key !== replay.key) throw new Error("MoGe cache replay failed.");
    const npz = result.result.artifacts.find((a) => a.format === "npz").uri;
    const workerFile = result.result.artifacts.find((a) => path.basename(a.uri) === "worker-result.json").uri;
    const worker = await json(workerFile);
    if (canonicalJSON(worker.inputs.map((i) => i.sha256)) !== canonicalJSON(truth.input_hashes.map((i) => i.sha256))) throw new Error("MoGe worker source hashes mismatch.");
    const laneOutput = path.join(output, lane);
    await evaluate(npz, workerFile, laneOutput);
    const report = await json(path.join(laneOutput, "report.json"));
    const receiptFile = path.join(output, `${lane}-receipt.json`);
    let receipt = { lane, key: result.key, manifestPath: result.manifestPath, requestFile, prediction: npz, predictionSha256: await fileDigest(npz), firstCached: result.cached, engineWallSeconds: wall, replayCached: replay.cached, replayWallSeconds: replayWall, additionalForwardCallsOnReplay: 0, worker, report };
    if (result.cached) {
      const prior = await json(receiptFile);
      if (prior.key !== result.key) throw new Error("Unexpected cached MoGe result lacking matching first-run receipt.");
      receipt = { ...receipt, firstCached: prior.firstCached, engineWallSeconds: prior.engineWallSeconds, verificationReplay: { cached: true, wallSeconds: wall } };
    }
    await save(receiptFile, receipt);
    lanes[lane] = receipt;
    console.log(JSON.stringify({ lane, key: result.key, cached: result.cached, engineWallSeconds: wall, telemetry: worker.telemetry, depth: report.depth }));
  }
  if (lanes.M1.key === lanes.M2.key) throw new Error("MoGe lane cache collision.");
  for (const prior of preserved) if (await fileDigest(prior.file) !== prior.sha256) throw new Error("Original DA3 receipt changed.");
  if (await fileDigest(declarationFile) !== declaration.sha256) throw new Error("Predeclared evaluation changed during experiment.");
  await save(path.join(output, "matrix.json"), { startingHead: "55ce11da49bf1c8e7a8f2d12e14d9657de697ed3", preserved, declaration, lanes });
}

if (import.meta.main) main().catch((error) => { console.error(error); process.exitCode = 1; });
