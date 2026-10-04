import { spawn } from "node:child_process";
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { runGeneration } from "../../engine/run.mjs";
import { assertScene, resolveFact } from "../../engine/scene.mjs";
import { fileDigest } from "../../engine/cache.mjs";
import { implementationFingerprint } from "../../engine/identity.mjs";
import { exportBenson } from "../../engine/adapters.mjs";

export async function attachBenchmarkEvidence(scene, report, { truthPath, reportPath, alignedPath }) {
  const result = structuredClone(scene);
  const license = { id: "MIT", commercialUse: "allowed", attribution: "Image Blaster contributors; entirely synthetic benchmark" };
  const anchorId = "synthetic-fixture-anchor", annotationId = "synthetic-evaluation-annotations";
  const truthHash = await fileDigest(truthPath);
  result.sources.push({ id: anchorId, kind: "user-input", uri: "fixture://synthetic-room-v1/opening-width-anchor", sha256: truthHash, license });
  result.sources.push({ id: annotationId, kind: "user-input", uri: "fixture://synthetic-room-v1/evaluation-only-labels-and-pixel-annotations", sha256: truthHash, license });
  const modelIds = result.sources.filter((source) => ["model", "image"].includes(source.kind)).map((source) => source.id);
  const sourceIds = [...modelIds, anchorId, annotationId];
  const fact = (id, value) => ({ id, value, state: "inferred", sourceIds, confidence: 0, note: "Synthetic evaluation-only correspondences; calibrated visual evidence, no model semantic recognition or field measurement." });
  const artifactId = "synthetic-scale-aligned-points";
  const provider = { id: "benchmark-scale-alignment", model: "fixed-width-similarity", version: "1", implementation: await implementationFingerprint([new URL("evaluate.py", import.meta.url), new URL("fixture.py", import.meta.url), new URL(import.meta.url)]) };
  for (const [id, file, format] of [[artifactId, alignedPath, "npz"], ["synthetic-numeric-report", reportPath, "json"]]) result.artifacts.push({ id, uri: file, format, sha256: await fileDigest(file), role: "visual-only", sourceIds, provider, parameters: { coordinateFrame: "scene-y-up", units: "meters", scale: "anchored", anchor: report.scale_anchor, alignment: report.alignment }, license });
  for (const item of report.dimensions) {
    const id = `synthetic-${item.id}`;
    const dimensions = {};
    for (const axis of item.axes) if (axis.aligned_observed_extent_m > 0) dimensions[axis.axis] = [fact(`${id}-${axis.axis}-aligned`, axis.aligned_observed_extent_m)];
    const entity = { id, label: { id: `${id}-label`, value: item.id, state: "user-specified", sourceIds: [annotationId], confidence: 1, note: "Fixture annotation, not DA3 semantic object identification." }, dimensions,
      properties: { originalRelativeExtents: [fact(`${id}-original`, Object.fromEntries(item.axes.map((axis) => [axis.axis, { value: axis.original_oriented_extent_relative_units, units: "relative" }])))],
        scaleAlignment: [fact(`${id}-scale-alignment`, report.scale_anchor)], visibility: [fact(`${id}-visibility`, item.axes.map(({ axis, complete_axis_observed, truth_visibility_fraction }) => ({ axis, complete_axis_observed, truth_visibility_fraction })))] },
      geometry: { role: "visual-only", artifactId, sourceIds }
    };
    (item.id.includes("box") || item.id.includes("stool") ? result.objects : result.surfaces).push(entity);
  }
  return assertScene(result);
}

export function demonstrateMeasurementOverride(scene, knownWidth) {
  const result = structuredClone(scene);
  const opening = result.surfaces.find((entity) => entity.id === "synthetic-opening-back");
  if (!opening?.dimensions.width?.length) throw new Error("Benchmark needs inferred opening-width evidence.");
  const original = opening.dimensions.width[0].value;
  const measurementId = "synthetic-field-measurement-role-test";
  result.sources.push({ id: measurementId, kind: "field-measurement", uri: "fixture://synthetic-room-v1/simulated-field-measurement", license: { id: "MIT", commercialUse: "allowed", attribution: "Synthetic authority test; no actual customer field visit" } });
  opening.dimensions.width.push({ id: "synthetic-opening-measured-width", value: knownWidth, state: "measured", sourceIds: [measurementId], confidence: 1, note: "Synthetic supplied field-measurement role test only; not manufacturer or actual field truth." });
  assertScene(result);
  const resolved = resolveFact(opening.dimensions.width, { requireAuthoritative: true });
  const visualization = exportBenson(result);
  let technicalBlocked = false;
  try { exportBenson(result, { purpose: "technical" }); } catch { technicalBlocked = true; }
  if (!technicalBlocked || resolved.value !== knownWidth || opening.dimensions.width[0].value !== original || visualization.installationApproved) throw new Error("Authority override invariant failed.");
  return { scene: result, evidence: { kind: "synthetic field-measurement authority test", inferredOriginalMetres: original, measuredMetres: knownWidth, resolvedState: resolved.state, resolvedMetres: resolved.value, inferredPreserved: true, visualizationAllowed: true, technicalBlocked, installationApproved: false } };
}

function evaluate(fixture, prediction, output) {
  const python = process.env.IMAGE_BLASTER_DA3_PYTHON || fileURLToPath(new URL(process.platform === "win32" ? "../../workers/da3/.venv/Scripts/python.exe" : "../../workers/da3/.venv/bin/python", import.meta.url));
  return new Promise((resolve, reject) => {
    const child = spawn(python, [fileURLToPath(new URL("evaluate.py", import.meta.url)), "--fixture", fixture, "--prediction", prediction, "--out", output], { windowsHide: true, stdio: ["ignore", "ignore", "pipe"] });
    let error = "";
    child.stderr.on("data", (data) => { error = (error + data).slice(-8000); });
    child.on("error", reject);
    child.on("close", (code) => code === 0 ? resolve() : reject(new Error(`Benchmark evaluator failed (${code}): ${error}`)));
  });
}

export async function writeEvidencePackage(fixture, output, truth, report, summary) {
  const relative = (file) => path.relative(output, file).split(path.sep).map(encodeURIComponent).join("/");
  const escape = (value) => String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll('"', "&quot;");
  const cameras = truth.extrinsics.map((e, i) => {
    const centre = [0, 1, 2].map((axis) => -e.reduce((sum, row) => sum + row[axis] * row[3], 0));
    const x = 40 + (centre[0] + 2.4) * 100, y = 60 + centre[2] * 100;
    return `<circle cx="${x}" cy="${y}" r="7" fill="#135d99"/><text x="${x + 10}" y="${y}">C${i + 1}</text><line x1="${x}" y1="${y}" x2="280" y2="95" stroke="#135d99" opacity=".5"/>`;
  }).join("");
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="560" height="460" viewBox="0 0 560 460"><rect width="560" height="460" fill="white"/><g font-family="Arial" font-size="14" fill="#222"><text x="20" y="25">GROUND TRUTH: SYNTHETIC ROOM / CAMERA PLAN</text><rect x="40" y="60" width="480" height="360" fill="#f2f0e9" stroke="#333"/><line x1="220" y1="60" x2="340" y2="60" stroke="#b43d24" stroke-width="6"/><text x="190" y="50">Opening 1.20 m</text><text x="190" y="442">Room width 4.80 m; depth 3.60 m</text>${cameras}</g></svg>`;
  await writeFile(path.join(output, "GROUND-TRUTH-camera-plan.svg"), svg);
  const rows = [0, 1, 2, 3].map((i) => {
    const n = String(i + 1).padStart(2, "0"), source = String((i + 1) % 4 + 1).padStart(2, "0");
    return `<section><h2>View ${i + 1}</h2><div class="grid"><figure><figcaption>GROUND TRUTH: synthetic input</figcaption><img src="${relative(path.join(fixture, "inputs", `view-${n}.png`))}"></figure><figure><figcaption>GROUND TRUTH: depth in metres</figcaption><img src="GT-depth-${n}.png"></figure><figure><figcaption>MODEL INFERENCE: relative depth</figcaption><img src="INFERENCE-depth-${n}.png"></figure><figure><figcaption>SCALE-ALIGNED INFERENCE: depth in metres</figcaption><img src="ALIGNED-depth-${n}.png"></figure><figure><figcaption>SCALE-ALIGNED INFERENCE: view ${source} projected into fixed GT camera ${n}</figcaption><img src="ALIGNED-cross-view-${source}-to-${n}.png"></figure></div></section>`;
  }).join("");
  const html = `<!doctype html><html lang="en"><meta charset="utf-8"><title>Synthetic DA3 geometry evidence</title><style>body{font:16px system-ui;background:#f7f7f4;color:#222;max-width:1500px;margin:32px auto;padding:0 20px}.grid{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;width:270px}img{width:100%;height:auto}figcaption{min-height:58px;font-size:14px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:18px}a{color:#135d99}</style><h1>Synthetic DA3 Small geometry experiment</h1><p>Entirely project-authored fixture. No customer project, product geometry, measurement visit, installation approval or client import. One opening-width anchor calibrates scale; it is not a held-out accuracy result. GT labels are evaluation-only, not model semantic recognition.</p><p><a href="report.json">Numeric metrics</a> / <a href="summary.json">System and cache receipt</a> / <a href="scene-spec.json">SceneSpec and synthetic measurement override</a> / <a href="${relative(path.join(fixture, "ground-truth", "geometry.json"))}">Exact fixture/camera truth</a></p><img style="max-width:560px" src="GROUND-TRUTH-camera-plan.svg">${rows}<h2>Measured system behavior</h2><pre>${escape(JSON.stringify(summary, null, 2))}</pre><h2>Depth and camera metrics</h2><pre>${escape(JSON.stringify({ depth: report.depth, rotationErrorDegrees: report.cameras.aligned_rotation_error_degrees, positionErrorCm: report.cameras.aligned_translation_error_cm, boundaries: report.boundaries }, null, 2))}</pre></html>`;
  await writeFile(path.join(output, "evidence.html"), html);
}

export async function main() {
  const fixture = path.resolve(process.argv[2] || ".image-blaster/benchmark/synthetic-room-v1");
  const output = path.resolve(process.argv[3] || ".image-blaster/benchmark/da3-small-cpu-v1-profile");
  await mkdir(output, { recursive: true });
  const manifest = JSON.parse(await readFile(path.join(fixture, "inputs", "manifest.json"), "utf8"));
  for (const image of manifest.ordered_images) if (await fileDigest(path.join(fixture, "inputs", image.file)) !== image.sha256) throw new Error("Frozen fixture input hash mismatch; refusing inference.");
  const request = { capability: "scene-geometry", providerId: "da3-small", mode: "local", parameters: { sceneId: "synthetic-room-da3", profileWarmInference: true }, inputs: manifest.ordered_images.map((image) => ({ path: path.join(fixture, "inputs", image.file), mediaType: "image/png", license: { id: "MIT", commercialUse: "allowed", attribution: "Image Blaster synthetic room fixture" } })) };
  const started = performance.now();
  const first = await runGeneration(request);
  const firstWallSeconds = (performance.now() - started) / 1000;
  const repeatStarted = performance.now();
  const repeat = await runGeneration(request);
  const repeatWallSeconds = (performance.now() - repeatStarted) / 1000;
  if (!repeat.cached || repeat.key !== first.key) throw new Error("Identical geometry request must reuse cache.");
  const prediction = first.result.artifacts.find((artifact) => artifact.format === "npz").uri;
  const workerArtifact = first.result.artifacts.find((artifact) => path.basename(artifact.uri) === "worker-result.json");
  const workerResult = JSON.parse(await readFile(workerArtifact.uri, "utf8"));
  const truthPath = path.join(fixture, "ground-truth", "geometry.json");
  const truth = JSON.parse(await readFile(truthPath, "utf8"));
  const expectedHashes = manifest.ordered_images.map((image) => image.sha256);
  if (JSON.stringify(expectedHashes) !== JSON.stringify(truth.input_hashes.map((image) => image.sha256)) || JSON.stringify(expectedHashes) !== JSON.stringify(workerResult.inputs.map((image) => image.sha256)) || await fileDigest(prediction) !== workerResult.artifactSha256) throw new Error("Inference evidence does not match the frozen fixture truth and checkpoint receipt.");
  await evaluate(fixture, prediction, output);
  const reportPath = path.join(output, "report.json");
  const report = JSON.parse(await readFile(reportPath, "utf8"));
  const aligned = await attachBenchmarkEvidence(first.result, report, { truthPath, reportPath, alignedPath: path.join(output, "aligned.npz") });
  const override = demonstrateMeasurementOverride(aligned, report.scale_anchor.known_m);
  await writeFile(path.join(output, "scene-spec.json"), `${JSON.stringify(override.scene, null, 2)}\n`);
  const generationManifest = JSON.parse(await readFile(first.manifestPath, "utf8"));
  let summary = { kind: "entirely synthetic local CPU benchmark; no client or paid generation", key: first.key, manifestPath: first.manifestPath, firstCached: first.cached, repeatCached: repeat.cached, additionalInferenceOnRepeat: 0, firstWallSeconds, repeatWallSeconds, checkpoint: workerResult.identity,
    workerTelemetry: workerResult.telemetry, geometryArtifactBytes: (await stat(prediction)).size, generationArtifactBytes: generationManifest.artifacts.reduce((sum, artifact) => sum + artifact.bytes, 0), authorityOverride: override.evidence, metricsPath: reportPath };
  if (first.cached) {
    let prior;
    try { prior = JSON.parse(await readFile(path.join(output, "summary.json"), "utf8")); } catch (error) { if (error.code !== "ENOENT") throw error; }
    if (prior?.key === first.key && prior.firstCached === false) summary = { ...summary, firstCached: false, firstWallSeconds: prior.firstWallSeconds, repeatWallSeconds: prior.repeatWallSeconds, verificationReplay: { firstCached: true, repeatedCached: repeat.cached, firstWallSeconds, repeatWallSeconds, additionalInference: 0 } };
  }
  await writeFile(path.join(output, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
  await writeEvidencePackage(fixture, output, truth, report, summary);
  console.log(JSON.stringify(summary, null, 2));
}

if (import.meta.main) main().catch((error) => { console.error(error.message); process.exitCode = 1; });
