import { spawn } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { runGeneration } from "../../engine/run.mjs";
import { fileDigest, canonicalJSON } from "../../engine/cache.mjs";

const root = fileURLToPath(new URL("../../", import.meta.url));
const python = path.join(root, "workers/da3/.venv/Scripts/python.exe");
const fixture = path.join(root, ".image-blaster/benchmark/synthetic-room-v1");
const evidence = path.join(root, "docs/foundation/geometry-results/camera-conditioning-v1");
const output = path.join(root, ".image-blaster/benchmark/camera-conditioning-v1");
const baselinePath = path.join(root, "docs/foundation/geometry-results/da3-small-cpu-v1.json");
const json = async (p) => JSON.parse(await readFile(p, "utf8"));
const save = async (p, value) => writeFile(p, `${JSON.stringify(value, null, 2)}\n`);

function evaluate(prediction, target, worker) {
  return new Promise((resolve, reject) => {
    const child = spawn(python, [path.join(root, "benchmarks/geometry/evaluate.py"), "--fixture", fixture, "--prediction", prediction, "--out", target, "--worker-receipt", worker], { windowsHide: true, stdio: ["ignore", "ignore", "pipe"] });
    let stderr = "";
    child.stderr.on("data", (d) => { stderr += d; });
    child.on("error", reject);
    child.on("close", (code) => code === 0 ? resolve() : reject(new Error(stderr)));
  });
}

export async function main() {
  await mkdir(output, { recursive: true });
  const baselineHash = await fileDigest(baselinePath);
  const baseline = await json(baselinePath);
  const truth = await json(path.join(fixture, "ground-truth/geometry.json"));
  const regenerated = await json(path.join(output, "frozen-fixture/ground-truth/geometry.json"));
  if (canonicalJSON(truth.input_hashes) !== canonicalJSON(baseline.metrics.input_hashes) || canonicalJSON(truth.input_hashes) !== canonicalJSON(regenerated.input_hashes)) throw new Error("Frozen fixture source hashes changed.");
  if (await fileDigest(path.join(fixture, "ground-truth/geometry.json")) !== baseline.metrics.fixture_geometry_sha256) throw new Error("Canonical ground truth changed.");
  for (const image of truth.input_hashes) for (const dir of [fixture, path.join(output, "frozen-fixture")]) if (await fileDigest(path.join(dir, "inputs", image.file)) !== image.sha256) throw new Error("Source bytes changed.");
  const declaration = await json(path.join(evidence, "conditioning-declaration.json"));
  if (declaration.truth_sha256 !== baseline.metrics.fixture_geometry_sha256) throw new Error("Conditioning derived from different truth.");
  const lanes = {};
  const aPrediction = baseline.metrics.prediction;
  if (await fileDigest(aPrediction) !== baseline.metrics.prediction_sha256) throw new Error("Original Lane A NPZ changed.");
  const aWorker = path.join(path.dirname(aPrediction), "worker-result.json");
  await evaluate(aPrediction, path.join(output, "lane-A-supplemental-v2"), aWorker);
  const aReport = await json(path.join(output, "lane-A-supplemental-v2/report.json"));
  for (const key of ["depth", "boundaries", "dimensions", "cameras", "focal", "scale_anchor", "alignment"]) if (canonicalJSON(aReport[key]) !== canonicalJSON(baseline.metrics[key])) throw new Error(`Lane A evaluation drift: ${key}`);
  lanes.A = { kind: "immutable original baseline; supplementary evaluation only, no inference rerun", originalReceiptSha256: baselineHash, originalKey: baseline.summary.key, summary: baseline.summary, metrics: aReport };
  await save(path.join(evidence, "lane-A-supplemental-v2.json"), lanes.A);
  const common = { capability: "scene-geometry", providerId: "da3-small", mode: "local", parameters: { sceneId: "synthetic-room-da3", processResolution: 256, threads: 6, profileWarmInference: true }, inputs: truth.input_hashes.map((image) => ({ path: path.join(fixture, "inputs", image.file), mediaType: "image/png", license: { id: "MIT", commercialUse: "allowed", attribution: "Image Blaster synthetic room fixture" } })) };
  for (const lane of ["B", "C", "D"]) {
    const conditioningPath = path.join(evidence, declaration.lanes[lane].file);
    if (await fileDigest(conditioningPath) !== declaration.lanes[lane].sha256) throw new Error("Predeclared camera inputs changed.");
    const request = { ...common, conditioning: await json(conditioningPath) };
    const requestFile = path.join(output, `lane-${lane}-request.json`);
    await save(requestFile, { request, conditioningFile: conditioningPath, conditioningFileSha256: declaration.lanes[lane].sha256, experimentOnly: true });
    const started = performance.now();
    const first = await runGeneration(request);
    const firstWallSeconds = (performance.now() - started) / 1000;
    const again = performance.now();
    const repeat = await runGeneration({ ...request, evaluationMetadata: { note: "evaluation-only metadata does not affect inference cache", lane } });
    const repeatWallSeconds = (performance.now() - again) / 1000;
    if (!repeat.cached || first.key !== repeat.key) throw new Error("Conditioned cache failed.");
    const prediction = first.result.artifacts.find((a) => a.format === "npz").uri;
    const workerPath = first.result.artifacts.find((a) => path.basename(a.uri) === "worker-result.json").uri;
    const worker = await json(workerPath);
    if (canonicalJSON(worker.inputs.map((i) => i.sha256)) !== canonicalJSON(truth.input_hashes.map((i) => i.sha256))) throw new Error("Worker consumed different source images.");
    const target = path.join(output, `lane-${lane}-evaluation-v2`);
    await evaluate(prediction, target, workerPath);
    const metrics = await json(path.join(target, "report.json"));
    let receipt = { lane, kind: "experiment-only oracle camera control; visual-only model depth", conditioningFile: declaration.lanes[lane], requestFile, key: first.key, manifestPath: first.manifestPath, firstCached: first.cached, repeatCached: repeat.cached, firstWallSeconds, repeatWallSeconds, additionalInferenceOnRepeat: 0, worker, metrics };
    const receiptPath = path.join(evidence, `lane-${lane}-result-v2.json`);
    if (first.cached) {
      const prior = await json(path.join(evidence, `lane-${lane}-result-v1.json`));
      if (prior.key !== first.key) throw new Error("Unexpected cached request without matching receipt.");
      receipt = { ...receipt, firstCached: prior.firstCached, firstWallSeconds: prior.firstWallSeconds, repeatWallSeconds: prior.repeatWallSeconds, originalInferenceReceipt: `lane-${lane}-result-v1.json`, evaluationImplementationSha256: await fileDigest(path.join(root, "benchmarks/geometry/evaluate.py")), verificationReplay: { firstCached: true, repeatCached: true, firstWallSeconds, repeatWallSeconds } };
    }
    await save(receiptPath, receipt);
    lanes[lane] = receipt;
    console.log(JSON.stringify({ lane, key: first.key, cached: first.cached, depth: metrics.depth, workerTelemetry: worker.telemetry, firstWallSeconds }));
  }
  if (new Set([baseline.summary.key, ...["B", "C", "D"].map((l) => lanes[l].key)]).size !== 4 || await fileDigest(baselinePath) !== baselineHash) throw new Error("Lane identity collision or original baseline mutation.");
  await save(path.join(output, "matrix-v2.json"), { baselineHash, declaration, lanes });
}

if (import.meta.main) main().catch((error) => { console.error(error); process.exitCode = 1; });
