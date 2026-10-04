import Ajv from "ajv";
import schema from "./scene.schema.json" with { type: "json" };
import { canonicalJSON } from "./cache.mjs";

const validate = new Ajv({ allErrors: true, strict: false }).compile(schema);
const authoritativeStates = new Set(["measured", "manufacturer-specified"]);

export function emptyScene(id, sources = []) {
  return {
    schemaVersion: 1,
    id,
    coordinates: { units: "meters", handedness: "right", up: "Y", rotation: "XYZ-radians" },
    sources,
    cameras: [], surfaces: [], objects: [], relationships: [], materials: [], artifacts: [], validations: []
  };
}

export function isAuthoritative(fact) {
  return authoritativeStates.has(fact.state);
}

export function resolveFact(evidence, { requireAuthoritative = false } = {}) {
  const authoritative = evidence.filter(isAuthoritative);
  if (authoritative.length) {
    if (authoritative.some((fact) => canonicalJSON(fact.value) !== canonicalJSON(authoritative[0].value))) {
      throw new Error("Conflicting authoritative evidence requires review.");
    }
    return authoritative[0];
  }
  if (requireAuthoritative) throw new Error("Authoritative evidence is required.");
  const rank = { "user-specified": 3, observed: 2, inferred: 1, generated: 0 };
  return [...evidence].sort((a, b) => rank[b.state] - rank[a.state] || b.confidence - a.confidence)[0];
}

export function assertScene(scene) {
  if (!validate(scene)) {
    throw new Error(`Invalid SceneSpec: ${validate.errors.map((error) => `${error.instancePath} ${error.message}`).join("; ")}`);
  }
  const sources = new Map(scene.sources.map((source) => [source.id, source]));
  const entities = [...scene.cameras, ...scene.surfaces, ...scene.objects];
  const artifacts = new Map(scene.artifacts.map((artifact) => [artifact.id, artifact]));
  const materials = new Set(scene.materials.map((material) => material.id));
  const entityIds = new Set(entities.map((entity) => entity.id));
  const ids = new Set([scene.id]);
  const takeId = (id) => {
    if (ids.has(id)) throw new Error(`Duplicate scene identifier: ${id}`);
    ids.add(id);
  };
  const refs = (sourceIds) => {
    for (const id of sourceIds) if (!sources.has(id)) throw new Error(`Unknown source: ${id}`);
  };
  const checkFact = (fact) => {
    takeId(fact.id);
    refs(fact.sourceIds);
    const kinds = fact.sourceIds.map((id) => sources.get(id).kind);
    if (isAuthoritative(fact) && kinds.includes("model")) throw new Error("Model evidence cannot support authoritative facts.");
    if (fact.state === "measured" && !kinds.includes("field-measurement")) throw new Error("Measured facts require field-measurement sources.");
    if (fact.state === "manufacturer-specified" && !kinds.includes("manufacturer-document")) throw new Error("Manufacturer facts require manufacturer-document sources.");
    if (fact.state === "generated" && !kinds.includes("model")) throw new Error("Generated facts require model sources.");
  };
  for (const item of [...scene.sources, ...entities, ...scene.materials, ...scene.artifacts]) takeId(item.id);
  for (const entity of entities) {
    checkFact(entity.label);
    for (const evidence of Object.values(entity.dimensions)) {
      evidence.forEach(checkFact);
      resolveFact(evidence);
    }
    for (const evidence of Object.values(entity.properties || {})) { evidence.forEach(checkFact); resolveFact(evidence); }
    if (entity.transform) { entity.transform.forEach(checkFact); resolveFact(entity.transform); }
    for (const key of ["projection", "pose"]) if (entity[key]) {
      if (!scene.cameras.includes(entity)) throw new Error("Camera evidence belongs on cameras only.");
      entity[key].forEach(checkFact); resolveFact(entity[key]);
    }
    for (const projection of entity.projection || []) {
      const k = projection.value.intrinsics;
      if (k[0][0] <= 0 || k[1][1] <= 0 || canonicalJSON(k[2]) !== "[0,0,1]") throw new Error("Camera intrinsics require positive focal lengths and a homogeneous last row.");
    }
    for (const pose of entity.pose || []) {
      const r = pose.value.worldToCamera.map((row) => row.slice(0, 3));
      const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
      const determinant = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0]) + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
      if (r.some((row, i) => r.some((other, j) => Math.abs(dot(row, other) - Number(i === j)) > 0.002)) || Math.abs(determinant - 1) > 0.002) throw new Error("Camera extrinsics require a proper orthonormal rotation.");
    }
    for (const id of entity.materialIds || []) if (!materials.has(id)) throw new Error(`Unknown material: ${id}`);
    if (entity.geometry) {
      refs(entity.geometry.sourceIds);
      const artifact = artifacts.get(entity.geometry.artifactId);
      if (!artifact) throw new Error(`Unknown geometry artifact: ${entity.geometry.artifactId}`);
      if (entity.geometry.role === "authoritative" && artifact.role !== "authoritative") throw new Error("Visual artifact cannot become authoritative geometry.");
      if (entity.geometry.role === "authoritative" && !entity.geometry.sourceIds.some((id) => sources.get(id).kind === "registered-geometry")) throw new Error("Authoritative geometry requires a registered-geometry source.");
      if (entity.geometry.role === "authoritative" && entity.geometry.sourceIds.some((id) => sources.get(id).kind === "model")) throw new Error("Model geometry is visual-only.");
      if (entity.geometry.role === "authoritative" && !entity.geometry.sourceIds.some((id) => artifact.sourceIds.includes(id) && sources.get(id).kind === "registered-geometry" && sources.get(id).sha256 === artifact.sha256)) throw new Error("Geometry authority must bind to the referenced registered artifact.");
    }
  }
  for (const material of scene.materials) {
    checkFact(material.description);
    for (const evidence of Object.values(material.properties || {})) { evidence.forEach(checkFact); resolveFact(evidence); }
    for (const id of material.artifactIds) if (!artifacts.has(id)) throw new Error(`Unknown material artifact: ${id}`);
  }
  for (const artifact of scene.artifacts) {
    refs(artifact.sourceIds);
    if (artifact.role === "authoritative" && !artifact.sourceIds.some((id) => sources.get(id).kind === "registered-geometry")) throw new Error("Authoritative artifact requires registered geometry provenance.");
    if (artifact.role === "authoritative" && artifact.sourceIds.some((id) => sources.get(id).kind === "model")) throw new Error("Generated artifacts are visual-only.");
    if (artifact.role === "authoritative" && !artifact.sourceIds.some((id) => sources.get(id).kind === "registered-geometry" && sources.get(id).sha256 === artifact.sha256)) throw new Error("Authoritative artifact must match the registered geometry hash.");
  }
  for (const relation of scene.relationships) {
    if (!entityIds.has(relation.from) || !entityIds.has(relation.to)) throw new Error("Relationship refers to an unknown entity.");
    checkFact(relation.relation);
  }
  for (const result of scene.validations) if (!ids.has(result.target)) throw new Error(`Unknown validation target: ${result.target}`);
  return scene;
}

export function assertEvidencePreserved(original, generated) {
  if (original) {
    if (original.id !== generated.id) throw new Error("Provider must preserve scene identity.");
    const retained = (before, after) => before.every((item) => after.some((candidate) => canonicalJSON(candidate) === canonicalJSON(item)));
    for (const collection of ["sources", "artifacts", "relationships", "validations"]) if (!retained(original[collection], generated[collection])) throw new Error("Provider must preserve input source, artifact, relationship and validation provenance.");
    for (const collection of ["objects", "surfaces", "cameras", "materials"]) {
      for (const entity of original[collection]) {
        const after = generated[collection].find((candidate) => candidate.id === entity.id);
        if (!after) throw new Error("Provider must preserve input entities.");
        for (const key of ["label", "description"]) if (entity[key] && canonicalJSON(entity[key]) !== canonicalJSON(after[key])) throw new Error("Provider must preserve entity fact ownership.");
        for (const key of ["dimensions", "properties"]) {
          for (const [slot, evidence] of Object.entries(entity[key] || {})) if (!retained(evidence, after[key]?.[slot] || [])) throw new Error("Provider must preserve fact ownership and dimension axes.");
        }
        for (const key of ["transform", "projection", "pose"]) if (!retained(entity[key] || [], after[key] || [])) throw new Error("Provider must preserve transform and camera evidence.");
        for (const key of ["materialIds", "artifactIds"]) if (!(entity[key] || []).every((id) => (after[key] || []).includes(id))) throw new Error("Provider must preserve material and artifact bindings.");
        if (entity.geometry?.role === "authoritative" && canonicalJSON(entity.geometry) !== canonicalJSON(after.geometry)) throw new Error("Provider cannot replace authoritative geometry.");
      }
    }
  }
  const facts = (scene) => {
    const found = new Map();
    const walk = (value) => {
      if (!value || typeof value !== "object") return;
      if (value.id && value.state && Object.hasOwn(value, "value")) found.set(value.id, value);
      for (const child of Object.values(value)) walk(child);
    };
    walk(scene);
    return found;
  };
  const originalFacts = facts(original);
  const generatedFacts = facts(generated);
  for (const [id, fact] of originalFacts) if (!generatedFacts.has(id) || canonicalJSON(generatedFacts.get(id)) !== canonicalJSON(fact)) throw new Error("Provider must preserve input fact evidence; append new facts instead of rewriting history.");
  for (const [id, fact] of generatedFacts) if (isAuthoritative(fact) && !originalFacts.has(id)) throw new Error("Provider cannot introduce authoritative facts.");
  for (const source of generated.sources.filter((item) => ["field-measurement", "manufacturer-document", "registered-geometry"].includes(item.kind))) {
    const known = original?.sources.find((item) => item.id === source.id);
    if (!known || canonicalJSON(known) !== canonicalJSON(source)) throw new Error("Provider cannot introduce authoritative sources.");
  }
  const knownArtifacts = original?.artifacts || [];
  for (const artifact of generated.artifacts.filter((item) => item.role === "authoritative")) if (!knownArtifacts.some((known) => canonicalJSON(known) === canonicalJSON(artifact))) throw new Error("Provider cannot introduce authoritative geometry.");
  for (const result of generated.validations.filter((item) => item.check === "registered-geometry-dimension-match")) if (!original?.validations.some((known) => canonicalJSON(known) === canonicalJSON(result))) throw new Error("Provider cannot introduce registered geometry approvals.");
}
