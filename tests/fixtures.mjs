import { emptyScene } from "../engine/scene.mjs";

export const hash = "a".repeat(64);
export const unknownLicense = { id: "UNKNOWN", commercialUse: "unknown", attribution: "" };
export function source(id, kind, sha256) {
  return { id, kind, uri: `fixture://${id}`, license: unknownLicense, ...(sha256 ? { sha256 } : {}) };
}
export function fact(id, value, state = "inferred", sourceIds = ["photo"], confidence = 0.6) {
  return { id, value, state, sourceIds, confidence };
}
export function sceneFixture() {
  const scene = emptyScene("room", [source("photo", "image", hash), source("measure", "field-measurement"), source("manual", "manufacturer-document"), source("generator", "model"), source("registered", "registered-geometry", hash)]);
  scene.objects.push({ id: "opening", label: fact("label", "Opening", "observed"), dimensions: { width: [fact("width-inferred", 1.1), fact("width-measured", 0.9, "measured", ["measure"], 1)], height: [fact("height", 0.7, "measured", ["measure"], 1)], depth: [fact("depth", 0.5, "manufacturer-specified", ["manual"], 1)] }, transform: [fact("transform", { translation: [0, 0, 0], rotation: [0, 0, 0], scale: [1, 1, 1] }, "measured", ["measure"], 1)] });
  return scene;
}
export function authoritativeScene() {
  const scene = sceneFixture();
  scene.artifacts.push({ id: "mesh", uri: "fixture://registered.obj", format: "obj", sha256: hash, role: "authoritative", sourceIds: ["registered"], provider: { id: "registered-import", model: "fixture", version: "1" }, parameters: {}, license: unknownLicense });
  scene.objects[0].geometry = { role: "authoritative", artifactId: "mesh", sourceIds: ["registered"] };
  scene.validations.push({ target: "opening", check: "registered-geometry-dimension-match", status: "pass", details: "Synthetic contract fixture only; no installation validation." });
  return scene;
}
