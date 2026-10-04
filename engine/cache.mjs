import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, realpath, stat } from "node:fs/promises";
import path from "node:path";

export function canonicalJSON(value) {
  if (value === null || typeof value === "string" || typeof value === "boolean") return JSON.stringify(value);
  if (typeof value === "number" && Number.isFinite(value)) return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJSON).join(",")}]`;
  if (typeof value === "object" && Object.getPrototypeOf(value) === Object.prototype) {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalJSON(value[key])}`).join(",")}}`;
  }
  throw new Error("Requests must contain finite JSON values only.");
}

export function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}

export function requestKey(request) {
  return digest(canonicalJSON(request));
}

export async function fileDigest(file) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest("hex");
}

export function inside(directory, file) {
  const relative = path.relative(path.resolve(directory), path.resolve(file));
  if (!relative || relative.startsWith(`..${path.sep}`) || relative === ".." || path.isAbsolute(relative)) throw new Error("Artifact must be a file inside its cache entry.");
  return relative;
}

export async function verifiedInside(directory, file) {
  const relative = inside(directory, file);
  inside(await realpath(directory), await realpath(file));
  return relative;
}

export async function readCached(entry, key) {
  let manifest;
  try { manifest = JSON.parse(await readFile(path.join(entry, "manifest.json"), "utf8")); }
  catch (error) { if (error.code === "ENOENT") return undefined; throw error; }
  if (manifest.schemaVersion !== 1 || manifest.key !== key || requestKey(manifest.request) !== key) throw new Error("Cache manifest integrity failure.");
  if (manifest.status !== "complete") throw new Error("Existing incomplete generation requires inspection/resume; automatic resubmission is blocked.");
  for (const file of [...manifest.inputs, ...manifest.artifacts]) {
    const target = path.resolve(entry, file.path);
    await verifiedInside(entry, target);
    if (await fileDigest(target) !== file.sha256 || (await stat(target)).size !== file.bytes) throw new Error("Cache artifact integrity failure; automatic regeneration is blocked.");
  }
  if (requestKey(manifest.result) !== manifest.resultHash) throw new Error("Cache result integrity failure.");
  return manifest;
}
