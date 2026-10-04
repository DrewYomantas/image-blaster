#!/usr/bin/env node
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { parseArgs, one, many } from "../.claude/scripts/asset-pipeline/fal-queue.mjs";
import { createRegistry } from "./providers.mjs";
import { runGeneration } from "./run.mjs";
import { assertScene } from "./scene.mjs";
import { exportBenson, exportTPS } from "./adapters.mjs";

const usage = `Usage: npm run engine -- <command>
  providers
  analyze --image <local-path> [--image <other-path>] --scene-id <slug> --out <scene.json> [--evidence <scene.json>]
  generate --request <request.json> [--cache-dir <directory>]
  validate --scene <scene.json>
  export --scene <scene.json> --target tps|benson --out <manifest.json> [--purpose visualization|technical]
Generation defaults to local. Paid jobs require an explicit approval policy in the request file; see docs/foundation/ENGINE.md.`;

async function json(file) {
  return JSON.parse(await readFile(file, "utf8"));
}

async function output(file, value) {
  if (!file || typeof file !== "string") throw new Error("--out <file> is required.");
  await writeFile(file, `${JSON.stringify(value, null, 2)}\n`);
}

export async function main(argv = process.argv.slice(2)) {
  const { flags, positionals } = parseArgs(argv);
  const command = positionals[0];
  if (!command || command === "help" || flags.help) { console.log(usage); return; }
  if (command === "providers") { console.log(JSON.stringify(createRegistry().list(), null, 2)); return; }
  if (command === "analyze") {
    const images = many(flags, "image");
    const out = one(flags, "out");
    if (!images.length || !out) throw new Error("analyze requires --image and --out.");
    const evidence = one(flags, "evidence");
    const result = await runGeneration({ capability: "scene-analysis", parameters: { sceneId: one(flags, "scene-id", "scene") }, inputs: images.map((file) => ({ path: file, mediaType: `image/${path.extname(file).slice(1)}` })), ...(evidence ? { scene: await json(evidence) } : {}) }, { cacheDir: one(flags, "cache-dir", ".image-blaster/cache") });
    await output(out, result.result);
    console.log(JSON.stringify({ key: result.key, cached: result.cached, manifestPath: result.manifestPath, scenePath: path.resolve(out), note: "Local evidence envelope only; no visual inference or 3D reconstruction was performed." }, null, 2));
    return;
  }
  if (command === "generate") {
    const file = one(flags, "request");
    if (!file) throw new Error("generate requires --request.");
    const document = await json(file);
    const result = await runGeneration(document.request, { cacheDir: one(flags, "cache-dir", ".image-blaster/cache"), spend: document.spend });
    console.log(JSON.stringify(result, null, 2));
    return;
  }
  if (command === "validate" || command === "export") {
    const file = one(flags, "scene");
    if (!file) throw new Error(`${command} requires --scene.`);
    const scene = assertScene(await json(file));
    if (command === "validate") { console.log(JSON.stringify({ valid: true, schemaVersion: scene.schemaVersion, note: "Contract and provenance checks only; geometry, source accuracy and installation rules remain unverified." })); return; }
    const target = one(flags, "target");
    const manifest = target === "tps" ? exportTPS(scene) : target === "benson" ? exportBenson(scene, { purpose: one(flags, "purpose", "visualization") }) : null;
    if (!manifest) throw new Error("export requires --target tps|benson.");
    await output(one(flags, "out"), manifest);
    console.log(JSON.stringify({ target, outputPath: path.resolve(one(flags, "out")) }));
    return;
  }
  throw new Error(`Unknown command ${command}.\n${usage}`);
}

if (import.meta.main) {
  main().catch((error) => { console.error(error.message); process.exitCode = 1; });
}
